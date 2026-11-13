"""
AirTouch Kesintisiz Canlı Masaüstü Kamera, Sesli Komut ve Çift Ekran Kontrolcüsü (live_desktop_cam.py)
Normal laptop varsayılan kamerası (index 0) + mikrofon üzerinden çalışır:
- 21 Eklemli El İskeleti AI + Ekranda Canlı "Hangi Parmak Ne Yapar?" Rehberi
- Çevrimdışı Türkçe Sesli Komutlar (Vosk) + YouTube/Medya/Çift Ekran Geçişi
- Çift Monitör Desteği ([W] Ekran Taşı), Mini PiP Modu ([C]) ve Sistem Tepsisi (Tray)
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ACTION_PROFILES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_GESTURE_GUARD_MODE,
    DEFAULT_MAX_HAND_MOTION,
    DEFAULT_MIN_MARGIN,
    DEFAULT_PALM_ARM_SECONDS,
    DEFAULT_STABLE_FRAMES,
)
from core.camera import CameraStream, apply_low_light_boost, parse_camera_source
from core.engine import AirTouchEngine
from core.settings import SettingsManager
from core.system_controller import system_controller
from core.voice_commander import VoiceCommander, resolve_best_input_device, sound_feedback
from core.window_manager import window_manager

WINDOW_TITLE = "AirTouch - Canli Masaustu Kontrol (G: Rehber | K: El Kapi | V: Ses | C: Mini | W: Ekran | Q: Cikis)"


def _sync_engine_with_settings(engine: AirTouchEngine, settings_mgr: SettingsManager):
    cfg = settings_mgr.data
    mode = cfg.get("engine_mode", "landmark")
    if mode != engine.engine_mode:
        engine.set_mode(mode)

    engine.confidence_threshold = float(cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    engine.cooldown_seconds = float(cfg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))
    engine.required_stable_frames = int(cfg.get("required_stable_frames", DEFAULT_STABLE_FRAMES))
    engine.min_margin = float(cfg.get("min_margin", DEFAULT_MIN_MARGIN))
    engine.guard_mode = str(cfg.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE))
    engine.strict_neutral_reset = bool(cfg.get("strict_neutral_reset", True))
    engine.max_hand_motion = float(cfg.get("max_hand_motion", DEFAULT_MAX_HAND_MOTION))
    engine.palm_arm_seconds = float(cfg.get("palm_arm_seconds", DEFAULT_PALM_ARM_SECONDS))
    engine.show_skeleton = bool(cfg.get("show_skeleton", True))
    engine.gesture_actions = settings_mgr.get_actions_map(engine.engine_mode)

    prof_key = cfg.get("active_profile", "media")
    prof_info = ACTION_PROFILES.get(prof_key, {})
    engine.profile_label = prof_info.get("short_name", "Ozel")


def _create_tray_icon_image() -> Image.Image:
    img = Image.new("RGBA", (64, 64), (16, 22, 34, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, 60, 60], radius=14, fill=(18, 28, 44, 255), outline=(0, 215, 255, 255), width=3)
    d.ellipse([20, 22, 44, 48], fill=(0, 215, 255, 230))
    for fx in (17, 25, 33, 41):
        d.rounded_rectangle([fx, 11, fx + 5, 26], radius=2, fill=(0, 235, 140, 255))
    return img


def _start_system_tray(cmd_queue: "queue.Queue[str]"):
    try:
        import pystray
    except Exception:
        return None

    def _emit(cmd: str):
        def _handler(icon, item):
            try:
                cmd_queue.put_nowait(cmd)
            except queue.Full:
                pass

        return _handler

    menu = pystray.Menu(
        pystray.MenuItem("Pencereyi Göster / Gizle", _emit("toggle_hidden")),
        pystray.MenuItem("Mini PiP Modu Aç / Kapat", _emit("toggle_compact")),
        pystray.MenuItem("Diğer Ekrana Taşı", _emit("switch_monitor")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("El Takibini Duraklat / Devam", _emit("toggle_pause")),
        pystray.MenuItem("El Güvenlik Kapısı (Avuç / Direkt / Sadece Ses)", _emit("cycle_guard")),
        pystray.MenuItem("Sesli Komut Aç / Kapat", _emit("toggle_voice")),
        pystray.MenuItem("Profil Değiştir", _emit("cycle_profile")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Çıkış", _emit("quit")),
    )
    icon = pystray.Icon("AirTouch", _create_tray_icon_image(), "AirTouch Kontrol", menu)
    t = threading.Thread(target=icon.run, daemon=True)
    t.start()
    return icon


def _set_window_visibility(window_title: str, visible: bool):
    if sys.platform != "win32":
        return
    import ctypes

    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 5 if visible else 0)


def _apply_window_layout(
    window_title: str,
    compact_mode: bool,
    always_on_top: bool,
    monitor_index: int,
):
    w, h = (420, 255) if compact_mode else (960, 540)
    corner = "bottom_right" if compact_mode else "top_right"
    placed_mon = window_manager.position_window_on_monitor(
        window_title, monitor_index=monitor_index, width=w, height=h, corner=corner
    )
    top_flag = 1.0 if (always_on_top or compact_mode) else 0.0
    try:
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, top_flag)
    except Exception:
        pass
    return placed_mon


def run_live_desktop_camera(
    camera_source: Optional[str] = None,
    monitor_index: Optional[int] = None,
    compact_override: Optional[bool] = None,
    voice_override: Optional[bool] = None,
):
    settings_mgr = SettingsManager()
    cfg = settings_mgr.data

    resolved_source = (
        parse_camera_source(camera_source)
        if camera_source is not None
        else parse_camera_source(cfg.get("camera_source", 0))
    )
    if monitor_index is not None:
        cfg["hud_monitor_index"] = int(monitor_index)
    if compact_override is not None:
        cfg["hud_compact_mode"] = bool(compact_override)
    if voice_override is not None:
        cfg["voice_enabled"] = bool(voice_override)

    monitors = window_manager.get_monitors()
    best_mic = resolve_best_input_device(cfg.get("mic_device_index"))
    mic_desc = f"[{best_mic['index']}] {best_mic['name']}" if best_mic else "Bulunamadı"
    print("=" * 74)
    print("AirTouch v2.1 Pro — Masaüstü Canlı Kamera, Çift Ekran & Sesli Komut")
    print(f"Kamera: Laptop ({resolved_source}) | Ekran Sayısı: {len(monitors)} | Mikrofon: {mic_desc}")
    print("Kısayollar:")
    print("  [K] El Güvenlik Kapısı (Avuç / Direkt / Sadece Ses)  [G] Parmak Rehberi")
    print("  [V] Sesli Komut  [C] Mini PiP  [W] Ekran Taşı  [M] Profil  [SPACE] Bekle  [Q] Çık")
    print("=" * 74)

    engine = AirTouchEngine(
        engine_mode=cfg.get("engine_mode", "landmark"),
        confidence_threshold=cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
        cooldown_seconds=cfg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS),
        required_stable_frames=cfg.get("required_stable_frames", DEFAULT_STABLE_FRAMES),
        min_margin=cfg.get("min_margin", DEFAULT_MIN_MARGIN),
        guard_mode=cfg.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE),
        strict_neutral_reset=bool(cfg.get("strict_neutral_reset", True)),
        max_hand_motion=float(cfg.get("max_hand_motion", DEFAULT_MAX_HAND_MOTION)),
        palm_arm_seconds=float(cfg.get("palm_arm_seconds", DEFAULT_PALM_ARM_SECONDS)),
        running_mode="VIDEO",
    )
    _sync_engine_with_settings(engine, settings_mgr)

    cam = CameraStream(source=resolved_source, width=960, height=540)
    if not cam.start():
        print(f"\n[HATA] Kamera açılamadı (Kaynak: {resolved_source})!")
        print("İpucu: Başka bir uygulama (Zoom, tarayıcı sekmesi vb.) kamerayı kullanıyor olabilir.")
        engine.close()
        return

    # Sesli Komut Başlat
    voice_enabled = bool(cfg.get("voice_enabled", True))
    voice_cmd: Optional[VoiceCommander] = None
    if voice_enabled:
        voice_cmd = VoiceCommander(
            engine=cfg.get("voice_engine", "vosk"),
            wake_required=bool(cfg.get("voice_wake_required", False)),
            mic_device_index=cfg.get("mic_device_index"),
        )
        voice_cmd.start()

    tray_q: "queue.Queue[str]" = queue.Queue(maxsize=16)
    tray_icon = _start_system_tray(tray_q)

    compact_mode = bool(cfg.get("hud_compact_mode", False))
    show_guide = bool(cfg.get("show_gesture_guide", True))
    always_on_top = bool(cfg.get("always_on_top", False))
    hud_monitor = int(cfg.get("hud_monitor_index", 1))
    auto_context = bool(cfg.get("auto_context_profile", False))
    sound_on = bool(cfg.get("sound_feedback", True))
    window_hidden = False

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    hud_monitor = _apply_window_layout(WINDOW_TITLE, compact_mode, always_on_top, hud_monitor)

    profile_keys = list(ACTION_PROFILES.keys())
    last_settings_check = time.time()
    last_context_check = 0.0
    last_seen_ctx_hwnd: Optional[int] = None
    last_seen_ctx_type: Optional[str] = None
    target_ctx: Dict[str, Any] = window_manager.update_last_user_window()
    fps_smooth = 30.0
    prev_t = time.perf_counter()

    def _switch_profile(new_prof: str, announce: str = ""):
        nonlocal cfg
        if new_prof not in ACTION_PROFILES:
            return
        cfg = settings_mgr.apply_profile(new_prof)
        _sync_engine_with_settings(engine, settings_mgr)
        sound_feedback.play("profile", enabled=sound_on)
        print(f"Aktif Profil: {ACTION_PROFILES[new_prof]['name']} {announce}".strip())

    def _handle_meta(kind: str):
        nonlocal compact_mode, show_guide, hud_monitor, window_hidden, cfg, voice_enabled, voice_cmd
        if kind == "pause":
            engine.paused = True
            engine.reset_state()
            sound_feedback.play("pause", enabled=sound_on)
        elif kind == "resume":
            engine.paused = False
            engine.reset_state()
            sound_feedback.play("voice", enabled=sound_on)
        elif kind == "toggle_pause":
            engine.paused = not engine.paused
            engine.reset_state()
            sound_feedback.play("pause" if engine.paused else "voice", enabled=sound_on)
        elif kind in ("compact_on", "compact_off", "toggle_compact"):
            compact_mode = True if kind == "compact_on" else (False if kind == "compact_off" else not compact_mode)
            cfg["hud_compact_mode"] = compact_mode
            settings_mgr.save(cfg)
            hud_monitor = _apply_window_layout(WINDOW_TITLE, compact_mode, always_on_top, hud_monitor)
        elif kind == "guide_toggle":
            show_guide = not show_guide
            cfg["show_gesture_guide"] = show_guide
            settings_mgr.save(cfg)
        elif kind in ("switch_hud_monitor", "switch_monitor"):
            mons = window_manager.get_monitors()
            hud_monitor = (hud_monitor % max(1, len(mons))) + 1
            cfg["hud_monitor_index"] = hud_monitor
            settings_mgr.save(cfg)
            hud_monitor = _apply_window_layout(WINDOW_TITLE, compact_mode, always_on_top, hud_monitor)
        elif kind == "toggle_hidden":
            window_hidden = not window_hidden
            _set_window_visibility(WINDOW_TITLE, not window_hidden)
        elif kind == "toggle_voice":
            voice_enabled = not voice_enabled
            cfg["voice_enabled"] = voice_enabled
            settings_mgr.save(cfg)
            if voice_enabled:
                if voice_cmd is None:
                    voice_cmd = VoiceCommander(
                        engine=cfg.get("voice_engine", "vosk"),
                        wake_required=bool(cfg.get("voice_wake_required", False)),
                        mic_device_index=cfg.get("mic_device_index"),
                    )
                    voice_cmd.start()
            elif voice_cmd is not None:
                voice_cmd.stop()
                voice_cmd = None
        elif kind == "cycle_profile":
            curr_p = cfg.get("active_profile", "media")
            idx = profile_keys.index(curr_p) if curr_p in profile_keys else 0
            _switch_profile(profile_keys[(idx + 1) % len(profile_keys)])
        elif kind == "cycle_guard":
            guards = ["palm_arm", "strict_reset", "voice_only"]
            cur_g = cfg.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE)
            nxt_g = guards[(guards.index(cur_g) + 1) % len(guards)] if cur_g in guards else "palm_arm"
            cfg["gesture_guard_mode"] = nxt_g
            settings_mgr.save(cfg)
            engine.guard_mode = nxt_g
            engine.reset_state()
            labels_g = {"palm_arm": "Avuç Kapısı", "strict_reset": "Doğrudan + Kilit", "voice_only": "Sadece Ses"}
            sound_feedback.play("profile", enabled=sound_on)
            print(f"El Güvenlik Kapısı: {labels_g.get(nxt_g, nxt_g)}")

    try:
        while True:
            ret, raw_bgr = cam.read()
            if not ret or raw_bgr is None:
                time.sleep(0.015)
                if (cv2.waitKey(10) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                continue

            now_wall = time.time()

            # 1. Aktif pencere / ekran bağlamı & otomatik profil önerisi
            if now_wall - last_context_check >= 0.25:
                last_context_check = now_wall
                target_ctx = window_manager.update_last_user_window()
                if auto_context and target_ctx.get("hwnd"):
                    ctx_hwnd = target_ctx["hwnd"]
                    ctx_type = target_ctx.get("context", "workspace")
                    if ctx_hwnd != last_seen_ctx_hwnd or ctx_type != last_seen_ctx_type:
                        last_seen_ctx_hwnd = ctx_hwnd
                        last_seen_ctx_type = ctx_type
                        if ctx_type in ACTION_PROFILES and ctx_type != cfg.get("active_profile"):
                            _switch_profile(ctx_type, announce="(Oto-Bağlam)")

            # 2. settings.json canlı senkronizasyon
            if now_wall - last_settings_check >= 0.5:
                last_settings_check = now_wall
                if settings_mgr.reload_if_changed():
                    _sync_engine_with_settings(engine, settings_mgr)
                    cfg = settings_mgr.data
                    show_guide = bool(cfg.get("show_gesture_guide", True))
                    sound_on = bool(cfg.get("sound_feedback", True))
                    auto_context = bool(cfg.get("auto_context_profile", False))
                    if voice_cmd is not None:
                        voice_cmd.wake_required = bool(cfg.get("voice_wake_required", False))

            # 3. Sistem tepsisi menü komutları
            quit_requested = False
            while True:
                try:
                    t_cmd = tray_q.get_nowait()
                except queue.Empty:
                    break
                if t_cmd == "quit":
                    quit_requested = True
                    break
                _handle_meta(t_cmd)
            if quit_requested:
                break

            # 4. Sesli komutları tüket ve çalıştır
            if voice_cmd is not None:
                v_evt = voice_cmd.pop_command()
                if v_evt is not None:
                    v_type = v_evt["type"]
                    v_tgt = v_evt["target"]
                    if v_type == "youtube_play" and bool(cfg.get("trigger_enabled", True)):
                        res_msg = system_controller.open_youtube_video(v_evt.get("query", ""))
                        engine.last_action_message = f"[Ses] {res_msg}"
                        engine.total_triggers += 1
                        sound_feedback.play("voice", enabled=sound_on)
                        print(f"[SES KOMUTU] '{v_evt['heard_text']}' -> {res_msg}")
                    elif v_type == "action" and bool(cfg.get("trigger_enabled", True)):
                        res_msg = system_controller.execute_action(v_tgt)
                        if res_msg:
                            engine.last_action_message = f"[Ses] {res_msg}"
                            engine.total_triggers += 1
                            sound_feedback.play("voice", enabled=sound_on)
                            print(f"[SES KOMUTU] '{v_evt['heard_text']}' -> {res_msg}")
                    elif v_type == "profile":
                        _switch_profile(v_tgt, announce="(Ses Komutu)")
                    elif v_type == "meta":
                        _handle_meta(v_tgt)

            # 5. Kamera karesi işle
            target_w, target_h = (420, 255) if compact_mode else (960, 540)
            frame_bgr = cv2.resize(raw_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if cfg.get("mirror_camera", True):
                frame_bgr = cv2.flip(frame_bgr, 1)

            if cfg.get("low_light_boost", True):
                frame_bgr, _ = apply_low_light_boost(frame_bgr)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            cur_t = time.perf_counter()
            dt = max(1e-4, cur_t - prev_t)
            prev_t = cur_t
            fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / dt)

            v_status = voice_cmd.get_status() if voice_cmd is not None else {"running": False, "available": False, "mic_level": 0.0}
            guard_tags = {"palm_arm": "AVUC", "strict_reset": "DIREKT", "voice_only": "SALT-SES"}
            g_tag = guard_tags.get(engine.guard_mode, "AVUC")
            if compact_mode:
                footer_hint = f"[K] {g_tag} | [C] Genis | [W] Ekran | [V] Ses | [Q] Cikis"
            else:
                pause_str = "BEKLE" if engine.paused else "AKTIF"
                footer_hint = (
                    f"[K] Kapi:{g_tag} | [G] Rehber | [V] Ses:{'ACIK' if voice_enabled else 'KAPALI'} | "
                    f"[C] Mini | [W] Ekran:{hud_monitor} | [M] Profil | [SPACE] {pause_str} | [Q] Cikis"
                )

            report = engine.process_frame(
                frame_rgb,
                trigger_enabled=bool(cfg.get("trigger_enabled", True)),
                fps=fps_smooth,
                footer_hint=footer_hint,
                voice_status=v_status,
                target_context=target_ctx,
                show_guide=show_guide,
                compact_mode=compact_mode,
            )

            if report.get("action_executed"):
                sound_feedback.play("action", enabled=sound_on)
            elif report.get("arm_just_activated"):
                sound_feedback.play("arm", enabled=sound_on)

            if not window_hidden:
                hud_bgr = cv2.cvtColor(np.asarray(report["annotated_frame"]), cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW_TITLE, hud_bgr)
                try:
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == 32:  # SPACE
                _handle_meta("toggle_pause")
            elif key in (ord("k"), ord("K")):
                _handle_meta("cycle_guard")
            elif key in (ord("m"), ord("M")):
                _handle_meta("cycle_profile")
            elif key in (ord("g"), ord("G")):
                _handle_meta("guide_toggle")
            elif key in (ord("c"), ord("C")):
                _handle_meta("toggle_compact")
            elif key in (ord("w"), ord("W")):
                _handle_meta("switch_hud_monitor")
            elif key in (ord("v"), ord("V")):
                _handle_meta("toggle_voice")
            elif key in (ord("z"), ord("Z")):
                _handle_meta("toggle_hidden")
            elif key in (ord("x"), ord("X")):
                auto_context = not auto_context
                cfg["auto_context_profile"] = auto_context
                settings_mgr.save(cfg)
                print(f"Otomatik Bağlam Profili: {'AÇIK' if auto_context else 'KAPALI'}")
            elif key in (ord("h"), ord("H")):
                cfg["show_skeleton"] = not bool(cfg.get("show_skeleton", True))
                settings_mgr.save(cfg)
                engine.show_skeleton = cfg["show_skeleton"]
            elif key in (ord("l"), ord("L")):
                cfg["low_light_boost"] = not bool(cfg.get("low_light_boost", True))
                settings_mgr.save(cfg)
            elif key in (ord("a"), ord("A")):
                always_on_top = not always_on_top
                cfg["always_on_top"] = always_on_top
                settings_mgr.save(cfg)
                _apply_window_layout(WINDOW_TITLE, compact_mode, always_on_top, hud_monitor)
            elif key in (ord("t"), ord("T")):
                thresholds = [0.65, 0.72, 0.82]
                cur = round(float(engine.confidence_threshold), 2)
                nxt = thresholds[(thresholds.index(cur) + 1) % len(thresholds)] if cur in thresholds else 0.72
                cfg["confidence_threshold"] = nxt
                settings_mgr.save(cfg)
                engine.confidence_threshold = nxt
    finally:
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        if voice_cmd is not None:
            voice_cmd.stop()
        cam.stop()
        engine.close()
        cv2.destroyAllWindows()
        print("AirTouch canlı masaüstü penceresi kapatıldı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirTouch Canlı Masaüstü Kamerası")
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Kamera indeksi (varsayılan: 0 - laptop kamerası) veya video akış adresi",
    )
    parser.add_argument("--camera", dest="camera_opt", default=None, help="Kamera indeksi/URL")
    parser.add_argument("--monitor", type=int, default=None, help="Başlangıç ekran numarası (1, 2...)")
    parser.add_argument("--compact", action="store_true", help="Mini PiP köşe modunda başlat")
    parser.add_argument("--no-voice", action="store_true", help="Sesli komut dinleyicisini kapalı başlat")
    args = parser.parse_args()
    run_live_desktop_camera(
        camera_source=args.camera_opt if args.camera_opt is not None else args.source,
        monitor_index=args.monitor,
        compact_override=True if args.compact else None,
        voice_override=False if args.no_voice else None,
    )
