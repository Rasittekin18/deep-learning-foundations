"""
AirTouch - Temassız Masaüstü, YouTube, Çift Ekran ve Sesli Komut Merkezi (app.py)
Normal laptop varsayılan kamerası ve mikrofonu üzerinden çalışan,
MediaPipe İskelet AI + PyTorch + Çevrimdışı Vosk Türkçe Ses Tanıma destekli kontrol stüdyosu.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ACTION_PROFILES,
    AVAILABLE_ACTIONS,
    CNN_HISTORY_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_GESTURE_GUARD_MODE,
    DEFAULT_MAX_HAND_MOTION,
    DEFAULT_MIN_MARGIN,
    DEFAULT_PALM_ARM_SECONDS,
    DEFAULT_STABLE_FRAMES,
    LANDMARK_FINGER_PATTERNS,
    LANDMARK_GESTURE_ICONS,
    LANDMARK_GESTURE_NAMES,
    LANDMARK_HISTORY_PATH,
    LANDMARK_NUM_CLASSES,
    SIGN_LANGUAGE_CLASS_NAMES,
    VOSK_TR_MODEL_DIR,
)
from core.camera import CameraStream, apply_low_light_boost, list_available_cameras, parse_camera_source
from core.engine import AirTouchEngine
from core.landmark_trainer import (
    get_user_sample_counts,
    save_user_landmark_samples,
    train_landmark_model,
)
from core.settings import SettingsManager
from core.system_controller import system_controller
from core.voice_commander import (
    VOICE_COMMAND_SPECS,
    VoiceCommander,
    list_input_microphones,
    match_voice_command,
    sound_feedback,
)
from core.window_manager import window_manager

# ==================== SAYFA YAPILANDIRMASI ====================
st.set_page_config(
    page_title="AirTouch Pro — Temassız & Sesli Kontrol Merkezi",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 55%, #00f5a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        color: #9aa7b8;
        font-size: 0.98rem;
        margin-bottom: 1.0rem;
    }
    .status-card {
        background: rgba(20, 26, 38, 0.85);
        border: 1px solid rgba(58, 123, 213, 0.35);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .badge-action {
        background: linear-gradient(135deg, #0ba360, #3cba92);
        color: #ffffff;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: 600;
        display: block;
        text-align: center;
        margin-top: 8px;
    }
    .badge-warn {
        background: rgba(247, 151, 30, 0.18);
        border: 1px solid #f7971e;
        color: #ffd27f;
        padding: 8px 12px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        margin-top: 8px;
    }
    .gesture-chip {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 7px;
    }
    .finger-row {
        display: inline-flex;
        gap: 4px;
        align-items: flex-end;
        margin-right: 8px;
        vertical-align: middle;
    }
    .f-bar {
        display: inline-block;
        width: 7px;
        border-radius: 3px;
    }
    .f-on { background: #00d8ff; height: 15px; }
    .f-spec { background: #ffbe28; height: 15px; }
    .f-off { background: #3b4658; height: 7px; border-radius: 50%; }
</style>
""",
    unsafe_allow_html=True,
)


def render_finger_dots_html(gid: int) -> str:
    """5 parmağın açık/kapalı görsel şemasını HTML olarak döndürür."""
    dots = LANDMARK_FINGER_PATTERNS.get(gid, {}).get("dots", [0, 0, 0, 0, 0])
    labels = ("Baş", "İşaret", "Orta", "Yüzük", "Serçe")
    spans = []
    for d, lbl in zip(dots, labels):
        cls = "f-on" if d == 1 else ("f-spec" if d == 2 else "f-off")
        spans.append(f'<span class="f-bar {cls}" title="{lbl}"></span>')
    return '<span class="finger-row">' + "".join(spans) + "</span>"


# ==================== OTURUM DURUMU & YARDIMCILAR ====================
if "settings_mgr" not in st.session_state:
    st.session_state.settings_mgr = SettingsManager()

settings_mgr: SettingsManager = st.session_state.settings_mgr
cfg = settings_mgr.data

if "engine" not in st.session_state:
    st.session_state.engine = AirTouchEngine(
        engine_mode=cfg.get("engine_mode", "landmark"),
        confidence_threshold=cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
        cooldown_seconds=cfg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS),
        required_stable_frames=cfg.get("required_stable_frames", DEFAULT_STABLE_FRAMES),
        min_margin=cfg.get("min_margin", DEFAULT_MIN_MARGIN),
        guard_mode=cfg.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE),
        strict_neutral_reset=bool(cfg.get("strict_neutral_reset", True)),
        max_hand_motion=float(cfg.get("max_hand_motion", DEFAULT_MAX_HAND_MOTION)),
        palm_arm_seconds=float(cfg.get("palm_arm_seconds", DEFAULT_PALM_ARM_SECONDS)),
        running_mode="IMAGE",
    )

if "action_history" not in st.session_state:
    st.session_state.action_history = []

if "live_cam_active" not in st.session_state:
    st.session_state.live_cam_active = False

if "voice_only_active" not in st.session_state:
    st.session_state.voice_only_active = False

if "cam_stream" not in st.session_state:
    st.session_state.cam_stream = None

if "voice_commander" not in st.session_state:
    st.session_state.voice_commander = None

if "available_cameras" not in st.session_state:
    st.session_state.available_cameras = [0]

if "available_mics" not in st.session_state:
    st.session_state.available_mics = list_input_microphones()

if "monitors" not in st.session_state:
    st.session_state.monitors = window_manager.get_monitors()

sync_widget_keys = bool(st.session_state.pop("_sync_widget_keys", False))
cur_data = settings_mgr.data
if sync_widget_keys or "sidebar_profile_select" not in st.session_state:
    st.session_state["sidebar_profile_select"] = cur_data.get("active_profile", "media")
for gid in range(LANDMARK_NUM_CLASSES):
    lm_key = f"bind_lm_{gid}"
    if sync_widget_keys or lm_key not in st.session_state:
        st.session_state[lm_key] = cur_data["landmark_actions"].get(str(gid), "none")

engine: AirTouchEngine = st.session_state.engine


def stop_browser_media():
    """Tarayıcıdaki canlı kamera ve ses dinleyicisini güvenli durdurup donanımı serbest bırakır."""
    st.session_state.live_cam_active = False
    st.session_state.voice_only_active = False
    cam: CameraStream = st.session_state.get("cam_stream")
    if cam is not None:
        cam.stop()
        st.session_state.cam_stream = None
    vc: VoiceCommander = st.session_state.get("voice_commander")
    if vc is not None:
        vc.stop()
        st.session_state.voice_commander = None


def sync_engine_from_settings():
    cur = settings_mgr.data
    mode = cur.get("engine_mode", "landmark")
    if mode != engine.engine_mode:
        engine.set_mode(mode)
    engine.confidence_threshold = float(cur.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    engine.cooldown_seconds = float(cur.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))
    engine.required_stable_frames = int(cur.get("required_stable_frames", DEFAULT_STABLE_FRAMES))
    engine.min_margin = float(cur.get("min_margin", DEFAULT_MIN_MARGIN))
    engine.guard_mode = str(cur.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE))
    engine.strict_neutral_reset = bool(cur.get("strict_neutral_reset", True))
    engine.max_hand_motion = float(cur.get("max_hand_motion", DEFAULT_MAX_HAND_MOTION))
    engine.palm_arm_seconds = float(cur.get("palm_arm_seconds", DEFAULT_PALM_ARM_SECONDS))
    engine.show_skeleton = bool(cur.get("show_skeleton", True))
    engine.gesture_actions = settings_mgr.get_actions_map(engine.engine_mode)
    prof_key = cur.get("active_profile", "media")
    engine.profile_label = ACTION_PROFILES.get(prof_key, {}).get("short_name", "Özel")


def _execute_voice_event(v_evt: dict, allow_trigger: bool, play_sound: bool):
    """Sesli komutu tek merkezden çalıştırır (canlı kamera ve kamerasız sesli asistan)."""
    v_type = v_evt.get("type")
    v_tgt = v_evt.get("target")
    ts = time.strftime("%H:%M:%S")
    if v_type == "youtube_play" and allow_trigger:
        res_msg = system_controller.open_youtube_video(v_evt.get("query", ""))
        st.session_state.action_history.insert(0, f"[{ts}] 🎤 {res_msg} ('{v_evt['heard_text']}')")
        st.session_state.action_history = st.session_state.action_history[:8]
        sound_feedback.play("voice", enabled=play_sound)
    elif v_type == "action" and allow_trigger:
        res_msg = system_controller.execute_action(v_tgt)
        if res_msg:
            st.session_state.action_history.insert(0, f"[{ts}] 🎤 {res_msg} ('{v_evt['heard_text']}')")
            st.session_state.action_history = st.session_state.action_history[:8]
            sound_feedback.play("voice", enabled=play_sound)
    elif v_type == "profile":
        apply_profile_and_sync(v_tgt)
        sound_feedback.play("profile", enabled=play_sound)
    elif v_type == "meta":
        if v_tgt == "pause":
            engine.paused = True
        elif v_tgt == "resume":
            engine.paused = False


def apply_profile_and_sync(profile_key: str):
    """Profili uygular ve bir sonraki rerun başında widget durumlarının senkronize edilmesini ister."""
    updated = settings_mgr.apply_profile(profile_key)
    st.session_state["_sync_widget_keys"] = True
    sync_engine_from_settings()
    return updated


def build_prob_chart(report: dict) -> go.Figure:
    if report.get("engine_mode") == "landmark":
        names_map = LANDMARK_GESTURE_NAMES
        n_cls = LANDMARK_NUM_CLASSES
    else:
        names_map = SIGN_LANGUAGE_CLASS_NAMES
        n_cls = 10

    labels = [names_map.get(i, f"Sınıf {i}") for i in range(n_cls)]
    probs = [p * 100.0 for p in report["all_probabilities"][:n_cls]]
    colors = [
        "#00e676" if (report.get("has_hand") and i == report["class_id"]) else "#2c3e50"
        for i in range(n_cls)
    ]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"%{p:.0f}" if p >= 6 else "" for p in probs],
            textposition="inside",
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=310,
        xaxis=dict(range=[0, 100], title="Olasılık (%)"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==================== YAN PANEL (SIDEBAR) ====================
with st.sidebar:
    st.markdown("## 🖐️ AirTouch v2.1 Pro")
    st.caption("El İskeleti + Sesli Komut + Çift Ekran")
    st.divider()

    # Kamera Kaynağı
    cam_col1, cam_col2 = st.columns([2.2, 1.0])
    with cam_col1:
        cam_options = list(dict.fromkeys([0] + st.session_state.available_cameras))
        cur_cam = parse_camera_source(cfg.get("camera_source", 0))
        if cur_cam not in cam_options:
            cam_options.append(cur_cam)
        selected_cam = st.selectbox(
            "📷 Laptop Kamerası",
            options=cam_options,
            index=cam_options.index(cur_cam),
            format_func=lambda x: f"Laptop Varsayılan ({x})" if x == 0 else f"Kamera {x}",
        )
    with cam_col2:
        st.write("")
        st.write("")
        if st.button("🔍 Tara", help="Bağlı kameraları ve ekranları tara", width="stretch"):
            stop_browser_media()
            st.session_state.available_cameras = list_available_cameras(max_index=4) or [0]
            st.session_state.available_mics = list_input_microphones()
            st.session_state.monitors = window_manager.get_monitors()
            st.toast(f"Kameralar: {st.session_state.available_cameras} | Ekranlar: {len(st.session_state.monitors)}")

    # AI Motor Seçimi
    mode_labels = {
        "landmark": "🤖 İskelet AI (MediaPipe + PyTorch)",
        "cnn": "🎓 Kurs CNN (64×64 Sign Language)",
    }
    selected_mode = st.radio(
        "🧠 Yapay Zeka Motoru",
        options=["landmark", "cnn"],
        index=0 if cfg.get("engine_mode", "landmark") == "landmark" else 1,
        format_func=lambda m: mode_labels[m],
    )

    # Profil Seçimi
    if selected_mode == "landmark":
        prof_keys = list(ACTION_PROFILES.keys()) + ["custom"]
        selected_prof = st.selectbox(
            "🎭 Aktif Kullanım Profili",
            options=prof_keys,
            format_func=lambda k: ACTION_PROFILES[k]["name"] if k in ACTION_PROFILES else "⚙️ Özel (Özelleştirilmiş)",
            key="sidebar_profile_select",
        )
        if selected_prof != cfg.get("active_profile") and selected_prof in ACTION_PROFILES:
            cfg = apply_profile_and_sync(selected_prof)
            st.toast(f"Profil değiştirildi: {ACTION_PROFILES[selected_prof]['short_name']}")
            st.rerun()

    st.divider()
    trigger_enabled = st.toggle(
        "⚡ Windows Tuş Tetikleme",
        value=bool(cfg.get("trigger_enabled", True)),
        help="Açıkken medya, YouTube, sunum ve pencere geçiş tuşlarını Windows üzerinde çalıştırır.",
    )
    paused_mode = st.toggle(
        "⏸️ Bekleme Modu (El Kilidi)",
        value=engine.paused,
        help="Kamera çalışmaya devam eder ancak el jestleri tuş tetiklemez.",
    )
    engine.paused = paused_mode

    st.subheader("🎤 Sesli Komut & Çoklu Ekran")
    voice_enabled_val = st.toggle(
        "🎙️ Türkçe Sesli Komutlar",
        value=bool(cfg.get("voice_enabled", True)),
        help="Masaüstü HUD veya canlı odada 'oynat', 'sesi artır', 'diğer ekran' gibi komutları dinler.",
    )
    wake_req_val = st.checkbox(
        "🔑 Uyandırma Kelimesi ('Asistan / Air')",
        value=bool(cfg.get("voice_wake_required", False)),
        help="Hoparlörden yüksek sesli video izlerken yanlış algılamayı sıfıra indirir.",
    )
    voice_engine_val = st.selectbox(
        "Ses Motoru",
        options=["vosk", "google"],
        index=0 if cfg.get("voice_engine", "vosk") == "vosk" else 1,
        format_func=lambda v: "🔒 Vosk Çevrimdışı Türkçe (Hızlı)" if v == "vosk" else "🌐 Google Speech (tr-TR)",
    )

    monitors = st.session_state.monitors or [{"index": 1, "is_primary": True, "bounds": (0, 0, 1920, 1080)}]
    mon_indices = [int(m["index"]) for m in monitors]
    cur_mon = int(cfg.get("hud_monitor_index", 1))
    if cur_mon not in mon_indices:
        cur_mon = mon_indices[0]
    hud_monitor_val = st.selectbox(
        "🖥️ Masaüstü HUD Ekranı",
        options=mon_indices,
        index=mon_indices.index(cur_mon),
        format_func=lambda idx: next(
            (
                f"Ekran {m['index']} ({m['bounds'][2]-m['bounds'][0]}×{m['bounds'][3]-m['bounds'][1]}"
                f"{' • Birincil' if m['is_primary'] else ''})"
                for m in monitors
                if m["index"] == idx
            ),
            f"Ekran {idx}",
        ),
    )

    compact_val = st.checkbox("🪟 Mini PiP Köşe Modu", value=bool(cfg.get("hud_compact_mode", False)))
    guide_val = st.checkbox("🖐️ Ekranda Parmak Rehberi Göster", value=bool(cfg.get("show_gesture_guide", True)))
    auto_ctx_val = st.checkbox(
        "🔄 Aktif Uygulamaya Göre Oto-Profil",
        value=bool(cfg.get("auto_context_profile", False)),
        help="YouTube veya PowerPoint odağına geçildiğinde ilgili profili otomatik seçer.",
    )
    sound_val = st.checkbox("🔔 Onay Sesi Bildirimi", value=bool(cfg.get("sound_feedback", True)))

    st.subheader("🛡️ Kazara Tetikleme Koruması")
    guard_labels = {
        "palm_arm": "🖐️ Avuç İle Uyandır (Önerilen)",
        "strict_reset": "🔒 Doğrudan + Nötr Kilidi",
        "voice_only": "🎤 Yalnızca Sesli Komut (El Pasif)",
    }
    guard_keys = list(guard_labels.keys())
    cur_guard = cfg.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE)
    if cur_guard not in guard_keys:
        cur_guard = DEFAULT_GESTURE_GUARD_MODE
    guard_val = st.selectbox(
        "El Tetikleme Kapısı",
        options=guard_keys,
        index=guard_keys.index(cur_guard),
        format_func=lambda g: guard_labels[g],
        help="Avuç Kapısı modu: sıradan el hareketlerini yoksayar; önce Açık El gösterip ardından 1 komut vermenizi bekler.",
    )
    strict_reset_val = st.checkbox(
        "🔐 Komut Sonrası Nötr Zorunluluğu",
        value=bool(cfg.get("strict_neutral_reset", True)),
        help="Bir işlem yaptıktan sonra elinizi indirirken parmaklar değişse bile arka arkaya komut atmaz; Açık El ile sıfırlanır.",
    )

    st.subheader("⚙️ Hassasiyet Ayarları")
    conf_val = st.slider(
        "Güven Eşiği",
        min_value=0.45,
        max_value=0.95,
        value=float(cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)),
        step=0.03,
    )
    cool_val = st.slider(
        "Bekleme Süresi (sn)",
        min_value=0.25,
        max_value=3.00,
        value=float(cfg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)),
        step=0.05,
    )
    stable_val = st.slider(
        "Kararlılık Kare Sayısı",
        min_value=1,
        max_value=12,
        value=int(cfg.get("required_stable_frames", DEFAULT_STABLE_FRAMES)),
        step=1,
    )

    mirror_val = st.checkbox("🪞 Ayna Görüntüsü", value=bool(cfg.get("mirror_camera", True)))
    skel_val = st.checkbox("🦴 21 Nokta El İskeleti", value=bool(cfg.get("show_skeleton", True)))
    low_light_val = st.checkbox("🌙 Loş Işık İyileştirme (CLAHE)", value=bool(cfg.get("low_light_boost", True)))

    dirty = (
        selected_cam != cfg.get("camera_source")
        or selected_mode != cfg.get("engine_mode")
        or trigger_enabled != cfg.get("trigger_enabled")
        or guard_val != cfg.get("gesture_guard_mode", DEFAULT_GESTURE_GUARD_MODE)
        or strict_reset_val != bool(cfg.get("strict_neutral_reset", True))
        or voice_enabled_val != bool(cfg.get("voice_enabled", True))
        or wake_req_val != bool(cfg.get("voice_wake_required", False))
        or voice_engine_val != cfg.get("voice_engine", "vosk")
        or hud_monitor_val != int(cfg.get("hud_monitor_index", 1))
        or compact_val != bool(cfg.get("hud_compact_mode", False))
        or guide_val != bool(cfg.get("show_gesture_guide", True))
        or auto_ctx_val != bool(cfg.get("auto_context_profile", False))
        or sound_val != bool(cfg.get("sound_feedback", True))
        or abs(conf_val - float(cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))) > 1e-4
        or abs(cool_val - float(cfg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))) > 1e-4
        or stable_val != int(cfg.get("required_stable_frames", DEFAULT_STABLE_FRAMES))
        or mirror_val != bool(cfg.get("mirror_camera", True))
        or skel_val != bool(cfg.get("show_skeleton", True))
        or low_light_val != bool(cfg.get("low_light_boost", True))
    )
    if dirty:
        cfg["camera_source"] = selected_cam
        cfg["engine_mode"] = selected_mode
        cfg["trigger_enabled"] = trigger_enabled
        cfg["gesture_guard_mode"] = guard_val
        cfg["strict_neutral_reset"] = strict_reset_val
        cfg["voice_enabled"] = voice_enabled_val
        cfg["voice_wake_required"] = wake_req_val
        cfg["voice_engine"] = voice_engine_val
        cfg["hud_monitor_index"] = hud_monitor_val
        cfg["hud_compact_mode"] = compact_val
        cfg["show_gesture_guide"] = guide_val
        cfg["auto_context_profile"] = auto_ctx_val
        cfg["sound_feedback"] = sound_val
        cfg["confidence_threshold"] = conf_val
        cfg["cooldown_seconds"] = cool_val
        cfg["required_stable_frames"] = stable_val
        cfg["mirror_camera"] = mirror_val
        cfg["show_skeleton"] = skel_val
        cfg["low_light_boost"] = low_light_val
        cfg = settings_mgr.save(cfg)

    sync_engine_from_settings()

    st.divider()
    st.caption(f"Bağlı Ekran: **{len(monitors)}** | Vosk TR: **{'Hazır' if VOSK_TR_MODEL_DIR.exists() else 'Eksik'}**")
    st.caption(f"Toplam Tetikleme: **{engine.total_triggers}**")


# ==================== ANA BAŞLIK & MASAÜSTÜ BAŞLATICI ====================
st.markdown('<p class="hero-title">AirTouch Pro: Temassız, Sesli & Çift Ekran Kontrol Merkezi</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">YouTube, çoklu monitör ve uygulamalar arası geçişleri 21 eklemli el iskeleti + çevrimdışı Türkçe sesli komutlarla yönetin.</p>',
    unsafe_allow_html=True,
)

top_c1, top_c2 = st.columns([1.45, 1.0])
with top_c1:
    if st.button(
        f"🖥️ MASAÜSTÜ HUD'U BAŞLAT (Ekran {hud_monitor_val} • {'Mini PiP' if compact_val else 'Parmak Rehberli'})",
        type="primary",
        width="stretch",
    ):
        stop_browser_media()
        settings_mgr.save(cfg)
        cmd_args = [
            sys.executable,
            str(ROOT_DIR / "live_desktop_cam.py"),
            "--camera",
            str(cfg.get("camera_source", 0)),
            "--monitor",
            str(hud_monitor_val),
        ]
        if compact_val:
            cmd_args.append("--compact")
        if not voice_enabled_val:
            cmd_args.append("--no-voice")
        subprocess.Popen(cmd_args)
        st.success(
            f"✅ Masaüstü HUD penceresi **Ekran {hud_monitor_val}** üzerinde başlatıldı! "
            "`[W]` ile ekran değiştirir, `[C]` ile Mini köşe moduna geçer, `[V]` ile sesli komutu açıp kapatabilirsiniz."
        )
with top_c2:
    mon_summary = " · ".join(
        f"Ekran {m['index']} ({m['bounds'][2]-m['bounds'][0]}×{m['bounds'][3]-m['bounds'][1]})"
        for m in monitors
    )
    st.info(
        f"🖥️ **Ekran Düzeni:** {mon_summary}\n\n"
        f"🎤 **Sesli Komut:** {'Aktif (' + ('Vosk TR' if voice_engine_val == 'vosk' else 'Google TR') + ')' if voice_enabled_val else 'Kapalı'}"
    )

st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎮 Canlı Kontrol Odası",
        "⚙️ Profiller & Sesli Komutlar",
        "🎯 Özel Jest Stüdyosu",
        "📈 Model Metrikleri",
        "📖 Çift Ekran & Sistem Testi",
    ]
)

# ==================== SEKME 1: CANLI KONTROL ODASI ====================
with tab1:
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.05, 1.05, 0.9, 1.35])
    with ctrl_c1:
        if not st.session_state.live_cam_active:
            if st.button("▶️ Canlı Kamerayı (& Sesi) Başlat", type="primary", width="stretch"):
                st.session_state.voice_only_active = False
                st.session_state.live_cam_active = True
                if voice_enabled_val and st.session_state.voice_commander is None:
                    vc = VoiceCommander(
                        engine=voice_engine_val,
                        wake_required=wake_req_val,
                        mic_device_index=cfg.get("mic_device_index"),
                    )
                    vc.start()
                    st.session_state.voice_commander = vc
                st.rerun()
        else:
            if st.button("⏹️ Canlı Akışı Durdur", type="secondary", width="stretch"):
                stop_browser_media()
                st.rerun()
    with ctrl_c2:
        if not st.session_state.voice_only_active:
            if st.button("🎙️ Sadece Sesli Asistanı Aç", width="stretch"):
                stop_browser_media()
                st.session_state.voice_only_active = True
                vc = VoiceCommander(
                    engine=voice_engine_val,
                    wake_required=wake_req_val,
                    mic_device_index=cfg.get("mic_device_index"),
                )
                vc.start()
                st.session_state.voice_commander = vc
                st.rerun()
        else:
            if st.button("🔇 Sesli Asistanı Kapat", type="secondary", width="stretch"):
                stop_browser_media()
                st.rerun()
    with ctrl_c3:
        if st.button("🔄 Kilidi Sıfırla", width="stretch"):
            engine.reset_state()
            st.session_state.action_history.clear()
            st.toast("Motor durumu sıfırlandı.")
    with ctrl_c4:
        prof_name = ACTION_PROFILES.get(cfg.get("active_profile", "media"), {}).get("name", "Özel")
        st.caption(f"🎭 **Profil:** {prof_name} | 📷 **Kamera:** {cfg.get('camera_source', 0)}")

    col_cam, col_info, col_chart = st.columns([1.4, 0.95, 1.15])
    frame_slot = col_cam.empty()
    info_card_slot = col_info.empty()
    progress_slot = col_info.empty()
    badge_slot = col_info.empty()
    voice_slot = col_info.empty()
    tensor_slot = col_info.empty()
    chart_slot = col_chart.empty()
    history_slot = st.empty()

    if st.session_state.live_cam_active:
        cam: CameraStream = st.session_state.cam_stream
        target_src = parse_camera_source(cfg.get("camera_source", 0))
        if cam is None or not cam.is_running or cam.source != target_src:
            if cam is not None:
                cam.stop()
            cam = CameraStream(source=target_src, width=960, height=540)
            if not cam.start():
                st.session_state.live_cam_active = False
                frame_slot.error(
                    f"❌ Laptop kamerası açılamadı (Kaynak: {target_src}). "
                    "Başka bir sekme/uygulama kamerayı kullanıyor olabilir."
                )
                cam = None
            st.session_state.cam_stream = cam

        if cam is not None and cam.is_running:
            fps_smooth = 25.0
            prev_t = time.perf_counter()
            frame_idx = 0
            vc: VoiceCommander = st.session_state.get("voice_commander")

            while st.session_state.live_cam_active:
                ret, raw_bgr = cam.read()
                if not ret or raw_bgr is None:
                    time.sleep(0.02)
                    continue

                if vc is not None:
                    v_evt = vc.pop_command()
                    if v_evt is not None:
                        _execute_voice_event(v_evt, trigger_enabled, sound_val)
                        cfg = settings_mgr.data

                frame_bgr = cv2.resize(raw_bgr, (880, 495), interpolation=cv2.INTER_LINEAR)
                if cfg.get("mirror_camera", True):
                    frame_bgr = cv2.flip(frame_bgr, 1)
                if cfg.get("low_light_boost", True):
                    frame_bgr, _ = apply_low_light_boost(frame_bgr)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                cur_t = time.perf_counter()
                fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / max(1e-4, cur_t - prev_t))
                prev_t = cur_t

                v_stat = vc.get_status() if vc is not None else {"running": False, "available": False, "mic_level": 0.0}
                tgt_ctx = window_manager.update_last_user_window()

                report = engine.process_frame(
                    frame_rgb,
                    trigger_enabled=bool(cfg.get("trigger_enabled", True)),
                    fps=fps_smooth,
                    voice_status=v_stat,
                    target_context=tgt_ctx,
                    show_guide=bool(cfg.get("show_gesture_guide", True)),
                    compact_mode=False,
                )

                if report["action_executed"]:
                    ts = time.strftime("%H:%M:%S")
                    st.session_state.action_history.insert(
                        0, f"[{ts}] 🖐️ {report['action_executed']} ({report['class_name']})"
                    )
                    st.session_state.action_history = st.session_state.action_history[:8]
                    sound_feedback.play("action", enabled=sound_val)
                elif report.get("arm_just_activated"):
                    sound_feedback.play("arm", enabled=sound_val)

                frame_slot.image(report["annotated_frame"], width="stretch")

                has_hand = report["has_hand"]
                conf_pct = report["confidence"] * 100.0 if has_hand else 0.0
                info_card_slot.markdown(
                    f"""
                    <div class="status-card">
                        <span style="font-size: 0.85rem; color: #8c9bb0;">Algılanan El Hareketi</span>
                        <h3 style="margin: 6px 0; color: #4db8ff;">{report['class_name']}</h3>
                        <span style="font-size: 0.9rem;">Güven: <b>%{conf_pct:.1f}</b> | Durum: <b>{report['status_state']}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress_slot.progress(min(1.0, float(report["confidence"]) if has_hand else 0.0))

                if report["paused"]:
                    badge_slot.markdown('<div class="badge-warn">⏸️ Bekleme Modu — Tetikleme Duraklatıldı</div>', unsafe_allow_html=True)
                elif report["status_state"] == "VOICE_ONLY":
                    badge_slot.info("🎤 Sadece Ses Modu — El hareketleri pasif.")
                elif not has_hand:
                    badge_slot.info("✋ Elinizi laptop kameranızın önüne kaldırın.")
                elif report["action_executed"]:
                    badge_slot.markdown(f'<div class="badge-action">⚡ {report["action_executed"]}</div>', unsafe_allow_html=True)
                elif report["status_state"] == "ARMED":
                    badge_slot.success(f"🟢 El Hazır ({report.get('arm_remaining', 0.0):.0f}s) — Şimdi komut jestini tutun!")
                elif report["status_state"] == "LOCKED_WAIT_PALM":
                    badge_slot.markdown('<div class="badge-warn">🛡️ Kilitli — Komut için önce Açık El (5 parmak) gösterin</div>', unsafe_allow_html=True)
                elif report["status_state"] == "MOVING":
                    badge_slot.info("✋ El Hareketli — Komut vermek için sabit tutun.")
                elif report["status_state"] == "LATCHED":
                    badge_slot.markdown('<div class="badge-warn">🔒 Tek-Atış Kilitli — Yeni komut için Açık El gösterin</div>', unsafe_allow_html=True)
                elif report["status_state"] == "CONFIRMING":
                    badge_slot.info(f"⏳ Doğrulanıyor: %{int(report['confirm_progress'] * 100)}")
                elif report["is_cooldown_active"]:
                    badge_slot.markdown(f'<div class="badge-warn">⏳ Mola: {report["cooldown_remaining"]} sn</div>', unsafe_allow_html=True)
                else:
                    badge_slot.success("🎯 Hazır — Komut vermek için jesti tutun.")

                if vc is not None:
                    heard = v_stat.get("last_partial") or v_stat.get("last_heard") or "Dinleniyor..."
                    last_c = v_stat.get("last_command") or "—"
                    mic_nm = (v_stat.get("mic_name") or "")[:24]
                    voice_slot.caption(f"🎤 `{mic_nm}` | **Duyulan:** `{heard}` | **Son:** `{last_c}`")

                if engine.engine_mode == "cnn":
                    tensor_slot.image(report["processed_tensor"], caption="CNN Girdisi (64×64)", width=110, clamp=True)

                if frame_idx % 4 == 0:
                    fig = build_prob_chart(report)
                    chart_slot.plotly_chart(fig, width="stretch", key=f"live_prob_chart_{frame_idx}")

                if st.session_state.action_history:
                    history_slot.markdown(
                        "**📋 Son Tetiklenen Eylemler (El + Ses):**\n"
                        + "\n".join(f"- `{item}`" for item in st.session_state.action_history[:6])
                    )

                frame_idx += 1
                time.sleep(0.03)
    elif st.session_state.voice_only_active:
        @st.fragment(run_every=0.45)
        def _voice_assistant_panel():
            vc_live: VoiceCommander = st.session_state.get("voice_commander")
            if vc_live is None:
                frame_slot.warning("Ses motoru başlatılıyor...")
                return
            while True:
                evt = vc_live.pop_command()
                if evt is None:
                    break
                _execute_voice_event(evt, trigger_enabled, sound_val)
            st_v = vc_live.get_status()
            lvl = min(1.0, float(st_v.get("mic_level", 0.0)))
            heard_now = st_v.get("last_partial") or st_v.get("last_heard") or "Dinleniyor..."
            last_cmd = st_v.get("last_command") or "—"
            mic_title = st_v.get("mic_name") or "Otomatik Mikrofon"
            frame_slot.success(
                f"🎙️ **Sesli Asistan Aktif** — Mikrofon: `{mic_title}` ({st_v.get('engine', 'vosk-tr')})\n\n"
                f"- **Duyulan:** `{heard_now}`\n"
                f"- **Son Komut:** `{last_cmd}`\n\n"
                "Tarayıcıda veya başka ekranda çalışırken doğrudan konuşabilirsiniz: "
                "*\"video dursun\"*, *\"youtube'da barış manço aç\"*, *\"diğer ekran\"*"
            )
            progress_slot.progress(lvl, text=f"Mikrofon Seviyesi: %{int(lvl * 100)}")
            if st_v.get("error"):
                badge_slot.error(f"Mikrofon uyarısı: {st_v['error']}")
            else:
                badge_slot.success("🔊 Dinleme aktif (Vosk Çevrimdışı Türkçe)")
            if st.session_state.action_history:
                history_slot.markdown(
                    "**📋 Son Sesli Komut İşlemleri:**\n"
                    + "\n".join(f"- `{item}`" for item in st.session_state.action_history[:6])
                )

        _voice_assistant_panel()
    else:
        frame_slot.info(
            "📷 **Laptop Kamerası ve Türkçe Sesli Komut Hazır.**\n\n"
            "- Kamera + Sesli Komut için **'▶️ Canlı Kamerayı (& Sesi) Başlat'**'a basın.\n"
            "- Kamerasız sadece konuşarak yönetmek için **'🎙️ Sadece Sesli Asistanı Aç'**'a basın.\n"
            "- YouTube veya çift monitörde en yüksek performans için üstteki **'🖥️ MASAÜSTÜ HUD'U BAŞLAT'** butonunu kullanın."
        )
        with col_info:
            st.markdown("#### 🖐️ Hangi Parmak Ne Yapar?")
            actions_map = settings_mgr.get_actions_map("landmark")
            for gid in range(LANDMARK_NUM_CLASSES):
                act_key = actions_map.get(gid) or "none"
                act_lbl = AVAILABLE_ACTIONS.get(act_key, {}).get("label", "İşlem Yok")
                icon = LANDMARK_GESTURE_ICONS.get(gid, "🖐️")
                gname = LANDMARK_GESTURE_NAMES.get(gid, f"Jest {gid}")
                pose_hint = LANDMARK_FINGER_PATTERNS.get(gid, {}).get("pose", "")
                dots_html = render_finger_dots_html(gid)
                st.markdown(
                    f'<div class="gesture-chip">{dots_html}<b>{icon} {gname}</b><br>'
                    f'<span style="color:#8da0b8;font-size:0.78rem;">{pose_hint}</span><br>'
                    f'<span style="color:#00e5a0;font-size:0.86rem;">→ {act_lbl}</span></div>',
                    unsafe_allow_html=True,
                )
        with col_chart:
            st.markdown("#### 🎤 Hızlı Sesli Komutlar")
            st.markdown(
                """
                - **Doğrudan Video Açma:** *"youtube'da tarkan kuzum aç"*, *"youtube barış manço açılsın"*, *"youtube python dersleri aç"*
                - **YouTube / Medya:** *"video dursun"*, *"video başlasın"*, *"oynat"*, *"durdur"*, *"sesi artır"*, *"sesi kıs"*, *"tam ekran"*, *"ileri sar"*, *"sonraki video"*
                - **Çift Ekran & Geçiş:** *"diğer ekran"*, *"ekrana taşı"*, *"uygulama değiştir"*, *"sonraki sekme"*, *"youtube geç"*
                - **Mod & Kontrol:** *"odak modu"*, *"youtube modu"*, *"çift ekran modu"*, *"mini mod"*, *"bekleme modu"*
                """
            )


# ==================== SEKME 2: PROFİLLER & SESLİ KOMUTLAR ====================
with tab2:
    st.subheader("⚙️ Kullanım Profilleri, Parmak Eşleştirmeleri & Sesli Komutlar")
    st.markdown(
        "Burada yaptığınız değişiklikler anında **`settings.json`** dosyasına kaydedilir ve "
        "açık olan Masaüstü HUD penceresinde de otomatik güncellenir."
    )

    p_cols = st.columns(len(ACTION_PROFILES))
    for col, (pkey, pinfo) in zip(p_cols, ACTION_PROFILES.items()):
        with col:
            if st.button(pinfo["name"], key=f"prof_btn_{pkey}", width="stretch"):
                cfg = apply_profile_and_sync(pkey)
                st.success(f"✅ '{pinfo['short_name']}' profili uygulandı!")
                st.rerun()

    st.divider()
    action_keys = list(AVAILABLE_ACTIONS.keys())

    if engine.engine_mode == "landmark":
        st.markdown("#### 🤖 İskelet AI Parmak Hareketi Eşleştirmeleri")
        c_left, c_right = st.columns(2)
        updated_map = dict(cfg.get("landmark_actions", {}))
        for gid in range(LANDMARK_NUM_CLASSES):
            target_col = c_left if gid < 4 else c_right
            with target_col:
                cur_act = updated_map.get(str(gid), "none")
                if cur_act not in action_keys:
                    cur_act = "none"
                pose_hint = LANDMARK_FINGER_PATTERNS.get(gid, {}).get("pose", "")
                sel = st.selectbox(
                    f"{LANDMARK_GESTURE_ICONS.get(gid, '')} {LANDMARK_GESTURE_NAMES[gid]} ({pose_hint})",
                    options=action_keys,
                    format_func=lambda k: AVAILABLE_ACTIONS[k]["label"],
                    key=f"bind_lm_{gid}",
                )
                updated_map[str(gid)] = sel

        if updated_map != cfg.get("landmark_actions"):
            cfg["landmark_actions"] = updated_map
            cfg["active_profile"] = "custom"
            cfg = settings_mgr.save(cfg)
            st.session_state["_sync_widget_keys"] = True
            sync_engine_from_settings()
            st.toast("Özel tuş atamaları kaydedildi.")
            st.rerun()
    else:
        st.markdown("#### 🎓 Kurs CNN (Sign Language 0-9) Jest Eşleştirmeleri")
        c_left, c_right = st.columns(2)
        updated_cnn = dict(cfg.get("cnn_actions", {}))
        for gid in range(10):
            target_col = c_left if gid < 5 else c_right
            with target_col:
                cur_act = updated_cnn.get(str(gid), "none")
                if cur_act not in action_keys:
                    cur_act = "none"
                sel = st.selectbox(
                    f"Sınıf {gid}: {SIGN_LANGUAGE_CLASS_NAMES[gid]}",
                    options=action_keys,
                    index=action_keys.index(cur_act),
                    format_func=lambda k: AVAILABLE_ACTIONS[k]["label"],
                    key=f"bind_cnn_{gid}",
                )
                updated_cnn[str(gid)] = sel

        if updated_cnn != cfg.get("cnn_actions"):
            cfg["cnn_actions"] = updated_cnn
            cfg = settings_mgr.save(cfg)
            sync_engine_from_settings()
            st.toast("CNN tuş atamaları kaydedildi.")

    st.divider()
    st.markdown("#### 🎤 Türkçe Sesli Komut Sözlüğü & Canlı İfade Testi")
    mic_c1, mic_c2 = st.columns([1.2, 1.3])
    with mic_c1:
        mics = st.session_state.available_mics or []
        mic_options = [None] + [m["index"] for m in mics]
        cur_mic = cfg.get("mic_device_index")
        if cur_mic not in mic_options:
            cur_mic = None
        sel_mic = st.selectbox(
            "Giriş Mikrofonu:",
            options=mic_options,
            index=mic_options.index(cur_mic),
            format_func=lambda idx: "Otomatik (En İyi Gerçek Mikrofon)" if idx is None else next(
                (f"[{m['index']}] {m['name']}" for m in mics if m["index"] == idx), f"Mikrofon {idx}"
            ),
        )
        if sel_mic != cfg.get("mic_device_index"):
            cfg["mic_device_index"] = sel_mic
            cfg = settings_mgr.save(cfg)
            st.toast("Mikrofon seçimi kaydedildi.")
    with mic_c2:
        sample_phrase = st.text_input(
            "Bir sesli komut cümlesi yazıp çözümlemeyi test edin:",
            placeholder="Örn: video dursun  |  youtube'da barış manço aç  |  diğer ekran",
        )
        if sample_phrase:
            parsed = match_voice_command(sample_phrase, wake_required=wake_req_val)
            if parsed:
                st.success(f"Eşleşti → **{parsed['label']}** (`{parsed['type']}:{parsed['target']}`)")
            else:
                st.warning("Eşleşme bulunamadı (uyandırma kelimesi açık ise başına 'asistan' ekleyin).")

    voice_rows = [
        {
            "Kategori": "YouTube Dinamik Açma",
            "Örnek Türkçe İfade": "youtube'da [video / şarkı adı] aç / açılsın",
            "Çalıştırdığı Eylem": "Aranan YouTube videosunu doğrudan tarayıcıda açar",
        }
    ] + [
        {
            "Kategori": sp["category"],
            "Örnek Türkçe İfade": " / ".join(sp["examples"]),
            "Çalıştırdığı Eylem": sp["label"],
        }
        for sp in VOICE_COMMAND_SPECS
    ]
    st.dataframe(voice_rows, width="stretch", hide_index=True, height=280)


# ==================== SEKME 3: ÖZEL JEST STÜDYOSU ====================
with tab3:
    st.subheader("🎯 Özel Jest Veri Toplama & PyTorch Yeniden Eğitim Stüdyosu")
    st.markdown(
        "Kendi el anatominize veya kamera açınıza özel ek örnekler toplayarak "
        "**`AirTouchLandmarkNet`** PyTorch sinir ağını birkaç saniyede ince ayar (fine-tune) yapabilirsiniz."
    )

    counts = get_user_sample_counts()
    cols_cnt = st.columns(4)
    for idx in range(LANDMARK_NUM_CLASSES):
        with cols_cnt[idx % 4]:
            st.metric(
                f"{LANDMARK_GESTURE_ICONS[idx]} {LANDMARK_GESTURE_NAMES[idx].split('(')[0].strip()}",
                f"{counts.get(idx, 0)} örnek",
            )

    st.divider()
    rec_c1, rec_c2 = st.columns([1.2, 1.0])
    with rec_c1:
        target_gesture = st.selectbox(
            "Kayıt Edilecek Jest Sınıfı:",
            options=list(range(LANDMARK_NUM_CLASSES)),
            format_func=lambda i: f"{LANDMARK_GESTURE_ICONS[i]} {LANDMARK_GESTURE_NAMES[i]}",
        )
        sample_target_n = st.slider("Toplanacak Kare Sayısı", min_value=10, max_value=60, value=25, step=5)

        if st.button("📸 Kameradan Örnek Kaydet (2-3 sn)", type="primary", width="stretch"):
            stop_browser_media()
            from core.hand_tracker import HandTracker

            cam_src = parse_camera_source(cfg.get("camera_source", 0))
            rec_cam = CameraStream(source=cam_src, width=640, height=480)
            if not rec_cam.start():
                st.error(f"Kamera açılamadı (Kaynak: {cam_src}).")
            else:
                tracker = HandTracker(running_mode="IMAGE")
                collected = []
                p_bar = st.progress(0.0, text="El bekleniyor... Hareketi kameraya tutun.")
                preview_box = st.empty()
                t_deadline = time.time() + 10.0
                try:
                    while len(collected) < sample_target_n and time.time() < t_deadline:
                        ok, bgr = rec_cam.read()
                        if not ok or bgr is None:
                            time.sleep(0.02)
                            continue
                        if cfg.get("mirror_camera", True):
                            bgr = cv2.flip(bgr, 1)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        res = tracker.detect(rgb)
                        if res["detected"] and res["features"] is not None:
                            collected.append(res["features"])
                            p_bar.progress(
                                len(collected) / sample_target_n,
                                text=f"Kaydediliyor: {len(collected)}/{sample_target_n}",
                            )
                        preview_box.image(rgb, width=320)
                        time.sleep(0.04)
                finally:
                    tracker.close()
                    rec_cam.stop()

                if collected:
                    feats_arr = np.vstack(collected).astype(np.float32)
                    labels_arr = np.full((len(collected),), int(target_gesture), dtype=np.int64)
                    total_u = save_user_landmark_samples(feats_arr, labels_arr)
                    st.success(f"✅ {len(collected)} adet gerçek el örneği kaydedildi! (Toplam kullanıcı örneği: {total_u})")
                    st.rerun()
                else:
                    st.warning("⚠️ Süre içinde net el algılanamadı. Işığı kontrol edip tekrar deneyin.")

    with rec_c2:
        st.markdown("#### ⚡ PyTorch Model Güncelleme")
        st.info(
            "Örnekleri topladıktan sonra aşağıdaki butona basın. Sentetik biyomekanik veri ile "
            "sizin gerçek el örnekleriniz harmanlanarak `AirTouchLandmarkNet` ~4 saniyede yeniden eğitilir."
        )
        if st.button("🚀 AirTouchLandmarkNet Modelini Şimdi Eğit", type="primary", width="stretch"):
            with st.spinner("PyTorch sinir ağı eğitiliyor..."):
                _, hist = train_landmark_model(epochs=28)
                engine.reload_landmark_model()
            st.success(
                f"✅ Eğitim tamamlandı! Doğrulama Başarısı: **%{hist['best_val_acc']:.2f}** "
                f"(Süre: {hist['duration_sec']}s)"
            )


# ==================== SEKME 4: MODEL METRİKLERİ ====================
with tab4:
    st.subheader("📈 PyTorch Derin Öğrenme Eğitim Metrikleri")

    model_tab_view = st.radio(
        "İncelenecek Model Geçmişi:",
        ["🤖 AirTouchLandmarkNet (İskelet AI - Aktif)", "🎓 AirTouchCNN (Kurs Sign Language 64×64)"],
        horizontal=True,
    )

    hist_path = LANDMARK_HISTORY_PATH if "Landmark" in model_tab_view else CNN_HISTORY_PATH
    if hist_path.exists():
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Epoch", hist.get("epochs", len(hist["train_loss"])))
        m2.metric("Son Train Loss", f"{hist['train_loss'][-1]:.4f}")
        m3.metric("Son Val Loss", f"{hist['val_loss'][-1]:.4f}")
        m4.metric("En Yüksek Val Acc", f"%{max(hist['val_acc']):.2f}")

        ep_x = list(range(1, len(hist["train_loss"]) + 1))
        cl, cr = st.columns(2)
        with cl:
            fl = go.Figure()
            fl.add_trace(go.Scatter(x=ep_x, y=hist["train_loss"], mode="lines+markers", name="Train Loss"))
            fl.add_trace(go.Scatter(x=ep_x, y=hist["val_loss"], mode="lines+markers", name="Val Loss"))
            fl.update_layout(title="Kayıp (Loss) Eğrisi", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fl, width="stretch")
        with cr:
            fa = go.Figure()
            fa.add_trace(go.Scatter(x=ep_x, y=hist["train_acc"], mode="lines+markers", name="Train Acc %"))
            fa.add_trace(go.Scatter(x=ep_x, y=hist["val_acc"], mode="lines+markers", name="Val Acc %"))
            fa.update_layout(title="Doğruluk (Accuracy %) Eğrisi", xaxis_title="Epoch", yaxis_title="%")
            st.plotly_chart(fa, width="stretch")
    else:
        st.warning("Eğitim geçmişi dosyası henüz oluşturulmamış.")


# ==================== SEKME 5: ÇİFT EKRAN & SİSTEM TESTİ ====================
with tab5:
    st.subheader("📖 YouTube / Çift Ekran Rehberi & Donanım Testi")
    g_col, t_col = st.columns([1.25, 1.0])

    with g_col:
        st.markdown(
            """
            ### 🚀 Profesyonel Çift Ekran & YouTube Akışı
            1. **İki Ekranda Sorunsuz Çalışma:** Sağ üstteki menüden HUD'u **Ekran 2**'ye sabitleyin veya HUD açıkken `[W]` tuşuna basarak pencereyi ekranlar arasında taşıyın; `[C]` ile **Mini PiP köşe moduna** küçültün.
            2. **Akıllı Odak Koruması:** YouTube/PowerPoint tuşları (`F`, `J`, `L`, `Ctrl+Tab`) gönderilirken odak yanlışlıkla AirTouch penceresindeyse otomatik olarak son kullanıcı uygulamanıza geri döner.
            3. **Hangi Parmak Ne Yapar?** HUD üzerindeki `[G]` rehber paneli seçtiğiniz profildeki gerçek tuş atamalarını ve o an yaptığınız hareketi canlı olarak vurgular.
            4. **Sesli + Jest Hibrit Kontrol:** Örneğin 1 parmakla sesi ayarlarken sadece *"diğer ekran"* veya *"sonraki video"* diyebilirsiniz; sistem tepsisi (sağ alt saat yanı) üzerinden HUD penceresini gizleyip arka planda çalıştırabilirsiniz.
            """
        )

    with t_col:
        st.markdown("#### 🧪 Windows Medya & Ekran Geçiş Testi")
        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button("🔊 Sesi Artır (+4%)", width="stretch"):
                st.toast(system_controller.volume_up())
            if st.button("⏯️ Oynat / Durdur", width="stretch"):
                st.toast(system_controller.play_pause())
            if st.button("🖥️ Diğer Ekrana Odaklan", width="stretch"):
                st.toast(system_controller.focus_other_screen())
        with tc2:
            if st.button("🔉 Sesi Azalt (-4%)", width="stretch"):
                st.toast(system_controller.volume_down())
            if st.button("🔇 Sessize Al / Aç", width="stretch"):
                st.toast(system_controller.mute())
            if st.button("🪟 Sonraki Uygulamaya Geç", width="stretch"):
                st.toast(system_controller.next_app())

        st.divider()
        st.markdown("#### 🪟 Algılanan Açık Uygulama Pencereleri")
        win_list = window_manager.list_taskbar_windows()[:8]
        if win_list:
            st.dataframe(
                [
                    {
                        "Ekran": f"Ekran {w['monitor_index']}",
                        "Bağlam": w["context"].upper(),
                        "Pencere Başlığı": w["title"][:48],
                    }
                    for w in win_list
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("Listelenecek açık pencere bulunamadı.")
