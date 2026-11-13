"""
AirTouch Profesyonel HUD, Parmak Rehberi & İskelet Görselleştirme Motoru (hud_renderer.py)
Windows TrueType (Segoe UI) yazı tipleriyle kusursuz Türkçe karakter desteği,
21 eklemli el iskeleti çizimi, canlı "Hangi Parmak Ne Yapar?" diyagram paneli,
aktif pencere/monitör rozeti, canlı ses seviyesi göstergesi ve Mini PiP modu üretir.
"""
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    AVAILABLE_ACTIONS,
    LANDMARK_FINGER_PATTERNS,
    LANDMARK_GESTURE_SHORT,
    LANDMARK_NUM_CLASSES,
)
from core.hand_tracker import HAND_CONNECTIONS

FONT_CANDIDATES_BOLD = [
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_CANDIDATES_REG = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# PIL TrueType'ın kare kutu basabileceği renkli emojileri temizle (Türkçe harfler korunur!)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002300-\U000023FF"
    "\uFE0F\u200D"
    "]+",
    flags=re.UNICODE,
)


def clean_hud_text(text: str) -> str:
    """HUD üzerinde kutucuk çıkmasını önlemek için desteklenmeyen renkli emojileri temizler."""
    if not text:
        return ""
    cleaned = _EMOJI_RE.sub("", str(text))
    return " ".join(cleaned.split())


def _truncate_text(text: str, max_len: int) -> str:
    s = clean_hud_text(text)
    if len(s) <= max_len:
        return s
    return s[: max(1, max_len - 2)].rstrip() + ".."


@lru_cache(maxsize=24)
def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REG
    for path_str in candidates:
        p = Path(path_str)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_hand_skeleton_rgb(
    rgb_img: np.ndarray,
    pixel_landmarks: Optional[np.ndarray],
    accent_rgb: Tuple[int, int, int],
) -> np.ndarray:
    """RGB NumPy görüntüsü üzerine anti-aliased 21 eklemli el iskeleti çizer."""
    if pixel_landmarks is None:
        return rgb_img
    out = rgb_img.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in pixel_landmarks]

    for i, j in HAND_CONNECTIONS:
        cv2.line(out, pts[i], pts[j], accent_rgb, 2, cv2.LINE_AA)

    for idx, (px, py) in enumerate(pts):
        is_tip = idx in (4, 8, 12, 16, 20)
        radius = 5 if is_tip else (6 if idx == 0 else 3)
        fill = (255, 255, 255) if is_tip else accent_rgb
        cv2.circle(out, (px, py), radius, fill, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), radius + 1, (18, 22, 28), 1, cv2.LINE_AA)
    return out


def _draw_finger_dots(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    dots: Sequence[int],
    active: bool,
    accent: Tuple[int, int, int],
):
    """
    5 parmak durumunu ([Baş, İşaret, Orta, Yüzük, Serçe]) mini el/nokta şeması olarak çizer.
    1 = açık (dolu parmak çubuğu), 0 = kapalı (kısık nokta), 2 = özel/çimdik/başparmak-yukarı (altın).
    """
    # Başparmak biraz daha aşağıda/solda, orta parmak en yüksek olacak şekilde doğal el silüeti
    y_offsets = [5, 1, 0, 1, 3]
    heights = [8, 11, 12, 11, 9]
    dx = 8

    for idx, (val, yo, h) in enumerate(zip(dots, y_offsets, heights)):
        cx = x + idx * dx
        cy = y + yo
        if val == 1:
            fill = (*accent, 255) if active else (0, 210, 245, 235)
            draw.rounded_rectangle([cx, cy, cx + 5, cy + h], radius=2, fill=fill)
        elif val == 2:
            fill = (255, 198, 40, 255)
            draw.rounded_rectangle([cx, cy, cx + 5, cy + h], radius=2, fill=fill)
        else:
            # Kapalı parmak: tabanda küçük sönük nokta
            by = y + 10
            draw.ellipse([cx, by - 2, cx + 5, by + 3], fill=(75, 88, 108, 190))


def _draw_mic_bars(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    level: float,
    listening: bool,
):
    """5 çubuklu canlı mikrofon seviye göstergesi çizer."""
    bar_heights = [5, 8, 11, 8, 6]
    threshold_steps = [0.05, 0.16, 0.30, 0.50, 0.70]
    for idx, (bh, thr) in enumerate(zip(bar_heights, threshold_steps)):
        bx = x + idx * 5
        by2 = y + 12
        by1 = by2 - bh
        if not listening:
            col = (95, 105, 120, 180)
        elif level >= thr:
            col = (0, 235, 130, 255)
        else:
            col = (65, 92, 115, 200)
        draw.rectangle([bx, by1, bx + 3, by2], fill=col)


def _format_target_pill(target_context: Optional[Dict[str, Any]]) -> str:
    if not target_context:
        return "Hedef: Masaustu"
    title = _truncate_text(target_context.get("title") or target_context.get("process_name") or "Masaustu", 18)
    mon = int(target_context.get("monitor_index", 1))
    return f"E{mon}: {title}"


def render_hud_frame(
    frame_rgb: np.ndarray,
    report: Dict[str, Any],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    pixel_landmarks: Optional[np.ndarray] = None,
    show_skeleton: bool = True,
    profile_label: str = "Medya",
    fps: Optional[float] = None,
    footer_hint: Optional[str] = None,
    action_map: Optional[Dict[int, Optional[str]]] = None,
    voice_status: Optional[Dict[str, Any]] = None,
    target_context: Optional[Dict[str, Any]] = None,
    show_guide: bool = False,
    compact_mode: bool = False,
) -> Image.Image:
    """
    Kamera karesi üzerine tam Türkçe destekli modern HUD katmanı çizer ve PIL Image döndürür.
    Girdi çözünürlüğünü (width, height) koruyarak yan panel, ses durumu ve hedef ekran rozetlerini yerleştirir.
    """
    if isinstance(frame_rgb, Image.Image):
        rgb_arr = np.array(frame_rgb.convert("RGB"))
    else:
        rgb_arr = np.asarray(frame_rgb)
        if rgb_arr.ndim == 2:
            rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2RGB)

    state = report.get("status_state", "READY")
    is_confident = bool(report.get("is_confident", False))
    is_cooldown = bool(report.get("is_cooldown_active", False))
    action_executed = report.get("action_executed")
    last_action = report.get("last_action_message")
    paused = bool(report.get("paused", False))
    has_hand = bool(report.get("has_hand", True))
    stable_class = int(report.get("class_id", 0))
    class_name = clean_hud_text(report.get("class_name", "Hazır"))
    confidence = float(report.get("confidence", 0.0))
    progress = float(report.get("confirm_progress", 0.0))
    engine_mode = report.get("engine_mode", "landmark")

    # Renk paleti (RGB)
    COLOR_GREEN = (0, 235, 130)
    COLOR_AMBER = (255, 190, 20)
    COLOR_BLUE = (70, 175, 255)
    COLOR_CYAN = (0, 215, 255)
    COLOR_GRAY = (145, 155, 170)

    if paused or state == "VOICE_ONLY":
        accent = COLOR_GRAY
    elif action_executed or state == "TRIGGERED":
        accent = COLOR_GREEN
    elif state == "ARMED":
        accent = COLOR_GREEN
    elif state == "CONFIRMING":
        accent = COLOR_AMBER
    elif state == "LATCHED":
        accent = COLOR_BLUE
    elif state in ("LOCKED_WAIT_PALM", "MOVING"):
        accent = COLOR_GRAY
    elif is_cooldown:
        accent = COLOR_AMBER
    elif has_hand and is_confident:
        accent = COLOR_CYAN
    else:
        accent = COLOR_GRAY

    if show_skeleton and pixel_landmarks is not None:
        rgb_arr = draw_hand_skeleton_rgb(rgb_arr, pixel_landmarks, accent)

    img = Image.fromarray(rgb_arr).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size

    font_title = _get_font(15 if compact_mode else 16, bold=True)
    font_small = _get_font(12 if compact_mode else 13, bold=False)
    font_badge = _get_font(13 if compact_mode else 14, bold=True)
    font_guide_title = _get_font(13, bold=True)
    font_guide = _get_font(12, bold=False)

    # --- 1. ÜST DURUM ŞERİDİ ---
    hud_h = 38 if compact_mode else 46
    draw.rectangle([0, 0, width, hud_h], fill=(12, 16, 24, 222))
    draw.line([(0, hud_h), (width, hud_h)], fill=(*accent, 255), width=2)

    arm_rem = float(report.get("arm_remaining", 0.0))
    if paused:
        top_text = "BEKLEME MODU" if compact_mode else "BEKLEME MODU — Devam etmek için SPACE veya 'devam et' deyin"
        top_color = COLOR_GRAY
    elif state == "VOICE_ONLY":
        top_text = "Sadece Ses Modu" if compact_mode else "Sadece Ses Modu — El tetikleme kapalı ([K] ile el kilidini aç)"
        top_color = (185, 200, 220)
    elif action_executed:
        top_text = f"TETİKLENDİ: {_truncate_text(action_executed, 28 if compact_mode else 44)}"
        top_color = COLOR_GREEN
    elif state == "ARMED":
        top_text = f"EL HAZIR ({arm_rem:.0f}s)" if compact_mode else f"EL HAZIR ({arm_rem:.0f}s) — Şimdi komut jestini tutun"
        top_color = COLOR_GREEN
    elif state == "LOCKED_WAIT_PALM":
        top_text = "Açık El Göster" if compact_mode else "Kilitli — Komut için önce Açık El (5 Parmak) gösterin"
        top_color = (210, 220, 232)
    elif state == "MOVING":
        top_text = "Sabit Tutun" if compact_mode else "El Hareketli — Onay için elinizi kısa süre sabit tutun"
        top_color = COLOR_GRAY
    elif state == "LATCHED":
        top_text = f"KİLİTLİ ({class_name}) — Açık El ile sıfırla" if compact_mode else f"KİLİTLENDİ ({class_name}) — Yeni komut için Açık El gösterin"
        top_color = COLOR_BLUE
    elif state == "CONFIRMING":
        top_text = f"Onay %{int(progress * 100)}: {class_name}"
        top_color = COLOR_AMBER
    elif is_cooldown and last_action:
        rem = report.get("cooldown_remaining", 0.0)
        top_text = f"Mola ({rem:.1f}s) | Son: {_truncate_text(last_action, 26)}"
        top_color = COLOR_AMBER
    elif has_hand and is_confident:
        top_text = f"{class_name} (%{confidence * 100:.0f})"
        top_color = COLOR_CYAN
    elif has_hand:
        top_text = f"El Takip Ediliyor (%{confidence * 100:.0f})"
        top_color = (210, 220, 232)
    else:
        top_text = "AirTouch Hazır" if compact_mode else "AirTouch Hazır — Elinizi kaldırın veya sesli komut verin"
        top_color = (205, 215, 228)

    # Sağ üst rozetler (Profil/FPS + Hedef Ekran/Uygulama + Ses)
    cursor_x = width - 10
    if width >= 380:
        right_info = f"{clean_hud_text(profile_label)}"
        if compact_mode and target_context:
            right_info += f" · E{int(target_context.get('monitor_index', 1))}"
        elif fps is not None and fps > 0 and not compact_mode:
            right_info += f" · {fps:.0f} FPS"
        rb = draw.textbbox((0, 0), right_info, font=font_small)
        pill_w = (rb[2] - rb[0]) + 16
        px1 = cursor_x - pill_w
        if px1 > width // 2 - (40 if compact_mode else 0):
            draw.rounded_rectangle(
                [px1, 8, cursor_x, hud_h - 8],
                radius=7,
                fill=(26, 34, 48, 225),
                outline=(*accent, 180),
                width=1,
            )
            draw.text((px1 + 8, (hud_h - 14) // 2), right_info, font=font_small, fill=(224, 234, 245, 255))
            cursor_x = px1 - 8

    # Hedef pencere & monitör rozeti
    if target_context and width >= 660 and not compact_mode:
        tgt_text = _format_target_pill(target_context)
        tb = draw.textbbox((0, 0), tgt_text, font=font_small)
        tw = (tb[2] - tb[0]) + 16
        tx1 = cursor_x - tw
        if tx1 > width // 2 - 40:
            draw.rounded_rectangle(
                [tx1, 8, cursor_x, hud_h - 8],
                radius=7,
                fill=(22, 32, 46, 220),
                outline=(80, 150, 235, 190),
                width=1,
            )
            draw.text((tx1 + 8, (hud_h - 14) // 2), tgt_text, font=font_small, fill=(165, 212, 255, 255))
            cursor_x = tx1 - 8

    # Sesli komut durum rozeti
    if voice_status is not None and width >= 420:
        v_running = bool(voice_status.get("running", False))
        v_avail = bool(voice_status.get("available", False))
        v_lvl = float(voice_status.get("mic_level", 0.0))
        v_lbl = "SES" if (v_running and v_avail) else ("SES HAZIR" if v_running else "SES YOK")
        vb = draw.textbbox((0, 0), v_lbl, font=font_small)
        vw = (vb[2] - vb[0]) + 42
        vx1 = cursor_x - vw
        if vx1 > 180:
            v_outline = (0, 225, 135, 190) if (v_running and v_avail) else (120, 130, 145, 170)
            draw.rounded_rectangle(
                [vx1, 8, cursor_x, hud_h - 8],
                radius=7,
                fill=(18, 28, 38, 225),
                outline=v_outline,
                width=1,
            )
            draw.text((vx1 + 8, (hud_h - 14) // 2), v_lbl, font=font_small, fill=(210, 245, 228, 255) if v_avail else (175, 185, 195, 255))
            _draw_mic_bars(draw, cursor_x - 30, (hud_h - 12) // 2, v_lvl, v_running and v_avail)
            cursor_x = vx1 - 8

    max_title_chars = max(16, (cursor_x - 24) // 8)
    draw.text((12, (hud_h - 18) // 2), _truncate_text(top_text, max_title_chars), font=font_title, fill=(*top_color, 255))

    # --- 2. SAĞ KENAR CANLI "HANGİ PARMAK NE YAPAR?" REHBER PANELİ ---
    guide_x1 = width
    if show_guide and not compact_mode and engine_mode == "landmark" and width >= 740 and height >= 360:
        panel_w = 278
        guide_x1 = width - panel_w - 8
        bottom_margin = 34 if footer_hint else 10
        guide_y1 = hud_h + 8
        guide_y2 = height - bottom_margin
        draw.rounded_rectangle(
            [guide_x1, guide_y1, width - 8, guide_y2],
            radius=12,
            fill=(12, 16, 24, 214),
            outline=(68, 86, 112, 220),
            width=1,
        )
        hdr = f"HANGİ PARMAK NE YAPAR? ({clean_hud_text(profile_label)})"
        draw.text((guide_x1 + 12, guide_y1 + 9), _truncate_text(hdr, 34), font=font_guide_title, fill=(0, 218, 255, 255))
        draw.line([(guide_x1 + 10, guide_y1 + 30), (width - 18, guide_y1 + 30)], fill=(58, 72, 92, 210), width=1)

        eff_map = action_map or {}
        list_top = guide_y1 + 36
        avail_list_h = max(200, (guide_y2 - 36) - list_top)
        row_h = max(28, min(42, avail_list_h // LANDMARK_NUM_CLASSES))

        for gid in range(LANDMARK_NUM_CLASSES):
            ry1 = list_top + gid * row_h
            ry2 = ry1 + row_h - 4
            if ry2 > guide_y2 - 28:
                break

            is_cur = has_hand and is_confident and (gid == stable_class)
            if is_cur:
                draw.rounded_rectangle(
                    [guide_x1 + 8, ry1, width - 16, ry2],
                    radius=7,
                    fill=(*accent, 55),
                    outline=(*accent, 235),
                    width=2,
                )

            pat = LANDMARK_FINGER_PATTERNS.get(gid, {"dots": [0, 0, 0, 0, 0]})
            _draw_finger_dots(draw, guide_x1 + 14, ry1 + (row_h - 18) // 2, pat["dots"], active=is_cur, accent=accent)

            g_short = LANDMARK_GESTURE_SHORT.get(gid, f"Jest {gid}")
            act_key = eff_map.get(gid) or "none"
            act_meta = AVAILABLE_ACTIONS.get(act_key, {})
            act_lbl = clean_hud_text(act_meta.get("short_label") or act_meta.get("label") or "Hazır / Sıfırla")
            if act_key in ("none", None, ""):
                act_lbl = "Hazır / Kilit Aç"

            txt_x = guide_x1 + 60
            g_col = (*accent, 255) if is_cur else (228, 236, 245, 255)
            a_col = (248, 252, 255, 255) if is_cur else (155, 174, 198, 255)
            draw.text((txt_x, ry1 + 2), _truncate_text(g_short, 18), font=font_guide_title if is_cur else font_guide, fill=g_col)
            draw.text((txt_x, ry1 + 16), f"-> {_truncate_text(act_lbl, 27)}", font=font_guide, fill=a_col)

        # Panel alt kısmı: aktif sesli komut ipucu / son duyulan ifade
        v_box_y = guide_y2 - 26
        draw.line([(guide_x1 + 10, v_box_y - 4), (width - 18, v_box_y - 4)], fill=(58, 72, 92, 180), width=1)
        if voice_status and voice_status.get("last_partial"):
            v_msg = f"Dinliyor: \"{_truncate_text(voice_status['last_partial'], 22)}\""
            v_col = (0, 215, 255, 255)
        elif voice_status and voice_status.get("last_command"):
            v_msg = f"Ses: {_truncate_text(voice_status['last_command'], 28)}"
            v_col = (0, 235, 130, 255)
        elif voice_status and voice_status.get("last_heard"):
            v_msg = f"Duyulan: \"{_truncate_text(voice_status['last_heard'], 22)}\""
            v_col = (180, 200, 225, 255)
        else:
            v_msg = "Ses: 'video dursun', 'youtube .. ac'"
            v_col = (150, 170, 195, 255)
        draw.text((guide_x1 + 12, v_box_y), v_msg, font=font_guide, fill=v_col)

    # --- 3. EL HEDEF KUTUSU (BOUNDING BOX) ---
    if bbox is not None:
        max_x_bound = (guide_x1 - 8) if guide_x1 < width else (width - 2)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(2, x1), max(hud_h + 4, y1)
        x2, y2 = min(max_x_bound, x2), min(height - 2, y2)

        if x2 > x1 + 12 and y2 > y1 + 12:
            draw.rectangle([x1, y1, x2, y2], outline=(*accent, 190), width=2)

            # Köşe kancaları
            cl = min(24, max(10, (x2 - x1) // 4))
            cw = 4
            ac = (*accent, 255)
            draw.line([(x1, y1), (x1 + cl, y1)], fill=ac, width=cw)
            draw.line([(x1, y1), (x1, y1 + cl)], fill=ac, width=cw)
            draw.line([(x2, y1), (x2 - cl, y1)], fill=ac, width=cw)
            draw.line([(x2, y1), (x2, y1 + cl)], fill=ac, width=cw)
            draw.line([(x1, y2), (x1 + cl, y2)], fill=ac, width=cw)
            draw.line([(x1, y2), (x1, y2 - cl)], fill=ac, width=cw)
            draw.line([(x2, y2), (x2 - cl, y2)], fill=ac, width=cw)
            draw.line([(x2, y2), (x2, y2 - cl)], fill=ac, width=cw)

            # Kutu üstü Türkçe etiket + eylem adı
            act_hint = ""
            if has_hand and action_map is not None:
                ak = action_map.get(stable_class)
                if ak and ak in AVAILABLE_ACTIONS:
                    act_hint = f" -> {clean_hud_text(AVAILABLE_ACTIONS[ak].get('short_label') or AVAILABLE_ACTIONS[ak]['label'])}"

            if has_hand:
                box_label = f"{class_name} | %{confidence * 100:.0f}{act_hint if not compact_mode else ''}"
            else:
                box_label = "Elinizi Kameraya Gösterin"
            box_label = _truncate_text(box_label, 38)

            lb = draw.textbbox((0, 0), box_label, font=font_badge)
            lw = (lb[2] - lb[0]) + 16
            lh = 24
            ly1 = max(hud_h + 4, y1 - lh - 4)
            draw.rounded_rectangle(
                [x1, ly1, min(max_x_bound, x1 + lw), ly1 + lh],
                radius=6,
                fill=(14, 18, 26, 225),
                outline=ac,
                width=1,
            )
            draw.text((x1 + 8, ly1 + 3), box_label, font=font_badge, fill=ac)

            # Doğrulama ilerleme çubuğu
            if state == "CONFIRMING" and progress > 0.0:
                bar_y1 = min(height - 36, y2 + 6)
                bar_y2 = bar_y1 + 8
                draw.rounded_rectangle([x1, bar_y1, x2, bar_y2], radius=4, fill=(20, 26, 36, 220))
                fill_x2 = x1 + int((x2 - x1) * min(1.0, progress))
                if fill_x2 > x1 + 4:
                    draw.rounded_rectangle([x1, bar_y1, fill_x2, bar_y2], radius=4, fill=(*COLOR_AMBER, 255))

    # --- 4. ALT BİLGİ ŞERİDİ ---
    if footer_hint:
        fh = 26 if compact_mode else 28
        draw.rectangle([0, height - fh, width, height], fill=(12, 16, 24, 220))
        draw.line([(0, height - fh), (width, height - fh)], fill=(70, 82, 98, 255), width=1)
        draw.text((10, height - fh + 5), clean_hud_text(footer_hint), font=font_small, fill=(195, 208, 222, 255))

    composited = Image.alpha_composite(img, overlay).convert("RGB")

    # CNN modunda sol alt köşede küçük "Model Gözü (64x64)" önizlemesi
    proc = report.get("processed_tensor")
    if proc is not None and engine_mode == "cnn":
        arr_64 = np.clip(np.asarray(proc, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        thumb = Image.fromarray(arr_64).resize((68, 68)).convert("RGB")
        px_x, px_y = 12, max(hud_h + 8, height - (36 if footer_hint else 12) - 68)
        composited.paste(thumb, (px_x, px_y))
        d2 = ImageDraw.Draw(composited)
        d2.rectangle([px_x, px_y, px_x + 68, px_y + 68], outline=accent, width=2)

    return composited
