"""
AirTouch Çevrimdışı Türkçe Sesli Komut Motoru (voice_commander.py)
Vosk (Kaldi tabanlı çevrimdışı Türkçe model) + akıllı mikrofon seçimi + VAD hızlı çözümleme +
opsiyonel Google Speech yedek motoru ile medya, dinamik YouTube video açma, çift ekran ve
AirTouch kontrollerini sesle yönetir.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import VOSK_TR_MODEL_DIR

WAKE_WORDS = (
    "airtouch",
    "air touch",
    "asistan",
    "sistem",
    "hey air",
    "air",
    "bilgisayar",
)

# Öncelik sırasına göre statik sesli komut kalıpları (en spesifik çok kelimeli ifadeler önce gelir)
VOICE_COMMAND_SPECS: List[Dict[str, Any]] = [
    # --- AirTouch Mod & HUD Kontrolleri ---
    {
        "type": "profile",
        "target": "youtube",
        "category": "Profil & Mod",
        "label": "YouTube Moduna Geç",
        "examples": ["youtube modu", "video modu"],
        "patterns": ["youtube modu", "yutup modu", "video modu", "youtube profili"],
    },
    {
        "type": "profile",
        "target": "workspace",
        "category": "Profil & Mod",
        "label": "Çift Ekran & Pencere Moduna Geç",
        "examples": ["çift ekran modu", "çalışma modu", "pencere modu"],
        "patterns": ["cift ekran modu", "cift ekran", "calisma modu", "pencere modu", "ofis modu"],
    },
    {
        "type": "profile",
        "target": "media",
        "category": "Profil & Mod",
        "label": "Medya Moduna Geç",
        "examples": ["medya modu", "müzik modu"],
        "patterns": ["medya modu", "muzik modu", "medya profili"],
    },
    {
        "type": "profile",
        "target": "presentation",
        "category": "Profil & Mod",
        "label": "Sunum Moduna Geç",
        "examples": ["sunum modu", "slayt modu"],
        "patterns": ["sunum modu", "sunum profili", "slayt modu", "sunun modu"],
    },
    {
        "type": "profile",
        "target": "browser",
        "category": "Profil & Mod",
        "label": "Tarayıcı Moduna Geç",
        "examples": ["tarayıcı modu", "okuma modu"],
        "patterns": ["tarayici modu", "okuma modu", "tarayici profili"],
    },
    {
        "type": "profile",
        "target": "focus",
        "category": "Profil & Mod",
        "label": "Odak Modu (3 Temel Jest)",
        "examples": ["odak modu", "basit mod", "sade mod"],
        "patterns": ["odak modu", "basit mod", "sade mod", "tek el modu"],
    },
    {
        "type": "meta",
        "target": "pause",
        "category": "AirTouch Kontrol",
        "label": "El Takibini Beklemeye Al",
        "examples": ["bekleme modu", "takibi duraklat"],
        "patterns": ["bekleme modu", "takibi duraklat", "el takibini durdur", "el takibini kapat", "beklemeye al"],
    },
    {
        "type": "meta",
        "target": "resume",
        "category": "AirTouch Kontrol",
        "label": "El Takibini Devam Ettir",
        "examples": ["devam et", "takibi aç", "aktif et"],
        "patterns": ["devam et", "takibi ac", "takibi baslat", "el takibini ac", "aktif et"],
    },
    {
        "type": "meta",
        "target": "compact_on",
        "category": "AirTouch Kontrol",
        "label": "Mini Köşe Moduna Geç",
        "examples": ["mini mod", "küçük mod"],
        "patterns": ["mini mod", "kucuk mod", "kucuk pencere", "minimum"],
    },
    {
        "type": "meta",
        "target": "compact_off",
        "category": "AirTouch Kontrol",
        "label": "Tam HUD Görünümüne Dön",
        "examples": ["büyük mod", "normal mod"],
        "patterns": ["buyuk mod", "normal mod", "tam gorunum", "tam pencere", "genis mod"],
    },
    {
        "type": "meta",
        "target": "guide_toggle",
        "category": "AirTouch Kontrol",
        "label": "Parmak Rehberini Aç / Kapat",
        "examples": ["rehber", "parmak rehberi"],
        "patterns": ["parmak rehberi", "rehberi ac", "rehberi kapat", "kilavuzu ac", "kilavuzu kapat", "rehber", "yardim"],
    },
    {
        "type": "meta",
        "target": "switch_hud_monitor",
        "category": "AirTouch Kontrol",
        "label": "AirTouch Penceresini Diğer Ekrana Al",
        "examples": ["kontrolü diğer ekrana al", "kamerayı taşı"],
        "patterns": ["kontrolu diger ekrana al", "kamerayi tasi", "hud tasi", "kamerayi diger ekrana"],
    },
    # --- Çift Ekran & Uygulamalar Arası Geçiş ---
    {
        "type": "action",
        "target": "move_window_other_screen",
        "category": "Çift Ekran & Uygulama",
        "label": "Aktif Pencereyi Diğer Ekrana Taşı",
        "examples": ["ekrana taşı", "pencereyi taşı"],
        "patterns": ["pencereyi diger ekrana tasi", "pencereyi tasi", "ekrana tasi", "diger ekrana tasi"],
    },
    {
        "type": "action",
        "target": "focus_other_screen",
        "category": "Çift Ekran & Uygulama",
        "label": "Diğer Ekrandaki Uygulamaya Geç",
        "examples": ["diğer ekran", "ekran değiştir", "öteki ekran"],
        "patterns": ["diger ekrana gec", "diger ekran", "oteki ekran", "ekran degistir", "ikinci ekran", "birinci ekran", "ekrana gec"],
    },
    {
        "type": "action",
        "target": "prev_app",
        "category": "Çift Ekran & Uygulama",
        "label": "Önceki Uygulamaya Dön",
        "examples": ["önceki uygulama", "önceki pencere"],
        "patterns": ["onceki uygulama", "onceki pencere", "geri don"],
    },
    {
        "type": "action",
        "target": "next_app",
        "category": "Çift Ekran & Uygulama",
        "label": "Sonraki Uygulamaya Geç (Alt+Tab)",
        "examples": ["uygulama değiştir", "pencere değiştir", "sonraki uygulama"],
        "patterns": ["uygulama degistir", "pencere degistir", "sonraki uygulama", "sonraki pencere", "diger uygulama", "alt tab"],
    },
    {
        "type": "action",
        "target": "browser_next_tab",
        "category": "Çift Ekran & Uygulama",
        "label": "Tarayıcı Sonraki Sekme",
        "examples": ["sonraki sekme", "sekme değiştir"],
        "patterns": ["sonraki sekme", "sekme degistir", "sonraki set mi", "sonraki cekme", "diger sekme"],
    },
    {
        "type": "action",
        "target": "browser_prev_tab",
        "category": "Çift Ekran & Uygulama",
        "label": "Tarayıcı Önceki Sekme",
        "examples": ["önceki sekme", "önceki sayfa"],
        "patterns": ["onceki sekme", "onceki set mi", "onceki cekme"],
    },
    {
        "type": "action",
        "target": "focus_youtube",
        "category": "Çift Ekran & Uygulama",
        "label": "YouTube Penceresine Odaklan / Aç",
        "examples": ["youtube geç", "youtube aç", "youtube'da [video adı] aç"],
        "patterns": ["youtube gec", "youtube a gec", "youtube ac", "yutup gec", "tarayiciya gec", "chrome a gec"],
    },
    {
        "type": "action",
        "target": "focus_spotify",
        "category": "Çift Ekran & Uygulama",
        "label": "Spotify Penceresine Odaklan",
        "examples": ["spotify geç", "spotify aç"],
        "patterns": ["spotify gec", "spotify ac", "muzige gec"],
    },
    {
        "type": "action",
        "target": "snap_left",
        "category": "Çift Ekran & Uygulama",
        "label": "Pencereyi Sola Yasla",
        "examples": ["sola yasla", "pencereyi sola"],
        "patterns": ["sola yasla", "pencereyi sola"],
    },
    {
        "type": "action",
        "target": "snap_right",
        "category": "Çift Ekran & Uygulama",
        "label": "Pencereyi Sağa Yasla",
        "examples": ["sağa yasla", "pencereyi sağa"],
        "patterns": ["saga yasla", "pencereyi saga"],
    },
    # --- YouTube & Medya Kontrolleri ---
    {
        "type": "action",
        "target": "yt_forward_10",
        "category": "YouTube & Medya",
        "label": "YouTube +10 sn İleri Sar",
        "examples": ["ileri sar", "on saniye ileri"],
        "patterns": ["ileri sar", "on saniye ileri", "on saniye gec"],
    },
    {
        "type": "action",
        "target": "yt_rewind_10",
        "category": "YouTube & Medya",
        "label": "YouTube -10 sn Geri Sar",
        "examples": ["geri sar", "on saniye geri"],
        "patterns": ["geri sar", "on saniye geri"],
    },
    {
        "type": "action",
        "target": "yt_next_video",
        "category": "YouTube & Medya",
        "label": "YouTube Sonraki Video",
        "examples": ["sonraki video", "sıradaki video"],
        "patterns": ["sonraki video", "siradaki video", "yeni video", "videoyu gec"],
    },
    {
        "type": "action",
        "target": "yt_captions",
        "category": "YouTube & Medya",
        "label": "YouTube Altyazı Aç / Kapat",
        "examples": ["altyazı aç", "altyazı kapat"],
        "patterns": ["altyazi ac", "altyazi kapat", "altyazi"],
    },
    {
        "type": "action",
        "target": "next_track",
        "category": "YouTube & Medya",
        "label": "Sonraki Parça",
        "examples": ["sonraki parça", "sonraki şarkı"],
        "patterns": ["sonraki parca", "sonraki sarki", "siradaki sarki", "sarkiyi gec"],
    },
    {
        "type": "action",
        "target": "prev_track",
        "category": "YouTube & Medya",
        "label": "Önceki Video / Parça",
        "examples": ["önceki video", "önceki parça"],
        "patterns": ["onceki video", "onceki parca", "onceki sarki"],
    },
    {
        "type": "action",
        "target": "volume_up",
        "category": "Ses Kontrolü",
        "label": "Sesi Artır (+4%)",
        "examples": ["sesi artır", "ses aç", "sesi yükselt"],
        "patterns": ["sesi artir", "sesi yukselt", "ses ac", "sesi ac", "ses yukari", "sesi biraz ac"],
    },
    {
        "type": "action",
        "target": "volume_down",
        "category": "Ses Kontrolü",
        "label": "Sesi Azalt (-4%)",
        "examples": ["sesi azalt", "ses kıs", "sesi kıs"],
        "patterns": ["sesi azalt", "sesini azalt", "ses kis", "sesi kis", "sesi dusur", "sesi alcalt", "ses asagi"],
    },
    {
        "type": "action",
        "target": "mute",
        "category": "Ses Kontrolü",
        "label": "Sessize Al / Sesi Aç",
        "examples": ["sesi kapat", "sessize al", "sessiz"],
        "patterns": ["sesi kapat", "sessize al", "sesi kes", "sessiz mod", "sessiz"],
    },
    {
        "type": "action",
        "target": "fullscreen",
        "category": "YouTube & Sunum",
        "label": "Tam Ekran Aç / Kapat (F)",
        "examples": ["tam ekran", "ekranı büyüt"],
        "patterns": ["tam ekran", "ekrani buyut", "tam ekrana gec", "buyuk ekran", "cam ekran"],
    },
    {
        "type": "action",
        "target": "esc",
        "category": "YouTube & Sunum",
        "label": "Tam Ekrandan Çık (ESC)",
        "examples": ["ekrandan çık", "küçült", "çıkış"],
        "patterns": ["tam ekrandan cik", "ekrandan cik", "ekrani kucult"],
    },
    {
        "type": "action",
        "target": "next_slide",
        "category": "YouTube & Sunum",
        "label": "Sonraki Slayt (Sağ Ok)",
        "examples": ["sonraki slayt", "slayt ileri"],
        "patterns": ["sonraki slayt", "slayt ileri"],
    },
    {
        "type": "action",
        "target": "prev_slide",
        "category": "YouTube & Sunum",
        "label": "Önceki Slayt (Sol Ok)",
        "examples": ["önceki slayt", "slayt geri"],
        "patterns": ["onceki slayt", "slayt geri"],
    },
    {
        "type": "action",
        "target": "scroll_down",
        "category": "Tarayıcı & Okuma",
        "label": "Sayfayı Aşağı Kaydır",
        "examples": ["aşağı kaydır", "sayfa aşağı"],
        "patterns": ["asagi kaydir", "sayfayi asagi", "sayfa asagi", "in asagi"],
    },
    {
        "type": "action",
        "target": "scroll_up",
        "category": "Tarayıcı & Okuma",
        "label": "Sayfayı Yukarı Kaydır",
        "examples": ["yukarı kaydır", "sayfa yukarı"],
        "patterns": ["yukari kaydir", "sayfayi yukari", "sayfa yukari", "cik yukari"],
    },
    {
        "type": "action",
        "target": "play_pause",
        "category": "YouTube & Medya",
        "label": "Oynat / Durdur",
        "examples": ["video dursun", "videoyu durdur", "video başlasın", "oynat"],
        "patterns": [
            "video dursun",
            "videoyu dursun",
            "videoyu durdur",
            "videoyu duraklat",
            "muzik dursun",
            "sarki dursun",
            "yayin dursun",
            "video dur",
            "video baslasin",
            "video oynasin",
            "video devam etsin",
            "videoyu oynat",
            "videoyu baslat",
            "videoyu ac",
            "sarki baslasin",
            "muzik baslasin",
            "muzik calsin",
            "oynat durdur",
            "duraklat",
            "durdur",
            "oynat",
            "baslat",
            "dur",
        ],
    },
]

_TR_ASCII_MAP = str.maketrans("çğıöşüÇĞIÖŞÜ", "cgiosuCGIOSU")

# Vosk Türkçe modelinin sık duyduğu İngilizce/teknik terim düzeltmeleri (sorgu çıkarımında)
_QUERY_PHONETIC_FIXES = {
    "fayton": "python",
    "payton": "python",
    "paytin": "python",
    "yutup": "youtube",
    "yutub": "youtube",
}

_YT_TRAILING_VERBS = {
    "acilsin",
    "ac",
    "agac",
    "calsin",
    "cal",
    "oynasin",
    "oynat",
    "baslasin",
    "baslat",
    "izle",
    "ara",
    "bul",
    "getir",
    "goster",
    "dinlet",
    "dinle",
}

_YT_FILLER_TOKENS = {
    "youtube",
    "yutup",
    "yutub",
    "utube",
    "da",
    "de",
    "ta",
    "te",
    "dan",
    "den",
    "a",
    "e",
    "su",
    "bu",
    "o",
    "bir",
    "lutfen",
    "hemen",
    "simdi",
    "video",
    "videoyu",
    "videolar",
    "videosu",
    "videosunu",
    "sarki",
    "sarkiyi",
    "sarkisi",
    "sarkisini",
}

_LOOPBACK_NON_MIC_KEYWORDS = (
    "stereo",
    "karışım",
    "karisim",
    "what u hear",
    "loopback",
    "wave out",
    "hoparlör",
    "hoparlor",
    "speaker",
    "pc hoparl",
    "output with hap",
    "mapper",
    "sound mapper",
    "birincil ses",
    "primary sound capture",
    "hat ()",
    "line in",
)


def normalize_voice_text(text: str) -> str:
    """Türkçe metni küçük harf, aksansız-ASCII dostu ve noktalama işaretlerinden arınmış forma çevirir."""
    if not text:
        return ""
    s = str(text).replace("İ", "i").replace("I", "i").replace("\u0307", "").lower()
    s = s.translate(_TR_ASCII_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _match_dynamic_youtube_query(norm: str, raw_text: str) -> Optional[Dict[str, str]]:
    """
    'youtube'da barış manço aç', 'youtube tarkan kuzum açılsın', 'python dersleri videosunu aç'
    gibi doğal Türkçe video açma cümlelerinden arama sorgusunu çıkarır.
    """
    tokens = norm.split()
    if len(tokens) < 2:
        return None

    has_yt_brand = tokens[0] in ("youtube", "yutup", "yutub", "utube")
    if has_yt_brand:
        # 'youtube modu', 'youtube profili', 'youtube gec', 'youtube a gec' gibi statik komutları ayır
        if any(tok in ("modu", "mod", "profili", "profil", "gec") for tok in tokens[1:]):
            return None
        rest = tokens[1:]
        # Baştaki ekleri atla ('da', 'ta', 'de', 'te', 'dan', 'a')
        while rest and rest[0] in ("da", "de", "ta", "te", "dan", "den", "a", "e"):
            rest = rest[1:]
    else:
        # 'tarkan kuzum videosu ac' / 'baris manco sarkisini cal' gibi YT kelimesiz cümleler
        if len(tokens) < 3:
            return None
        if not (tokens[-1] in _YT_TRAILING_VERBS and tokens[-2] in ("videosu", "videosunu", "sarkisi", "sarkisini")):
            return None
        rest = tokens

    # Sondaki eylem ve dolgu kelimelerini temizle
    while rest and (rest[-1] in _YT_TRAILING_VERBS or rest[-1] in ("videosu", "videosunu", "video", "videoyu", "sarkisi", "sarkisini")):
        rest = rest[:-1]

    # Kalan kelimelerden salt dolgu zamirlerini ('su', 'bu') ele
    cleaned = [_QUERY_PHONETIC_FIXES.get(tok, tok) for tok in rest if tok not in _YT_FILLER_TOKENS]
    query = " ".join(cleaned).strip()

    if len(query) >= 2:
        return {
            "type": "youtube_play",
            "target": "youtube_search_play",
            "query": query,
            "label": f"YouTube'da Aç: {query}",
            "matched_phrase": "youtube <video> aç",
            "heard_text": raw_text.strip(),
        }

    # Örn. sadece "youtube şu video açılsın" veya "youtube aç" denmişse YouTube'a geç/aç
    if has_yt_brand and any(tok in _YT_TRAILING_VERBS or tok in ("video", "videoyu", "videosu") for tok in tokens[1:]):
        return {
            "type": "action",
            "target": "focus_youtube",
            "label": "YouTube Penceresini Aç / Odaklan",
            "matched_phrase": "youtube aç",
            "heard_text": raw_text.strip(),
        }
    return None


def match_voice_command(text: str, wake_required: bool = False) -> Optional[Dict[str, str]]:
    """
    Duyulan Türkçe metni doğrulanmış AirTouch komut niyetine (action/profile/meta/youtube_play) eşler.
    `wake_required=True` ise uyandırma kelimesi ('asistan', 'air', 'sistem' vb.) yoksa None döndürür.
    """
    norm = normalize_voice_text(text)
    if not norm:
        return None

    has_wake = False
    for ww in WAKE_WORDS:
        ww_norm = normalize_voice_text(ww)
        if re.search(rf"(^|\s){re.escape(ww_norm)}(\s|$)", norm):
            has_wake = True
            norm = re.sub(rf"(^|\s){re.escape(ww_norm)}(\s|$)", " ", norm, count=1)
            norm = " ".join(norm.split())
            break

    if wake_required and not has_wake:
        return None
    if not norm:
        return None

    # 1. Önce dinamik YouTube video açma / arama cümlelerini kontrol et
    yt_dyn = _match_dynamic_youtube_query(norm, str(text))
    if yt_dyn is not None:
        return yt_dyn

    # 2. Sabit komut kalıplarında en uzun eşleşmeyi bul
    padded = f" {norm} "
    best_match = None
    best_len = 0
    for spec in VOICE_COMMAND_SPECS:
        for pat in spec["patterns"]:
            p_norm = normalize_voice_text(pat)
            if f" {p_norm} " in padded and len(p_norm) > best_len:
                best_len = len(p_norm)
                best_match = {
                    "type": spec["type"],
                    "target": spec["target"],
                    "label": spec["label"],
                    "matched_phrase": pat,
                    "heard_text": str(text).strip(),
                }
    return best_match


def get_cpp_safe_model_path(model_dir: Path = VOSK_TR_MODEL_DIR) -> Optional[str]:
    """
    Kaldi / Vosk C++ katmanının Türkçe klasör adlarında (`Masaüstü`, `Yazılım` vb.)
    başarısız olmaması için Windows 8.3 kısa yolunu veya ASCII önbellek yolunu döndürür.
    """
    target = Path(model_dir).resolve()
    if not target.exists():
        return None

    raw = str(target)
    if raw.isascii():
        return raw

    if sys.platform == "win32":
        buf = ctypes.create_unicode_buffer(1024)
        res = ctypes.windll.kernel32.GetShortPathNameW(raw, buf, 1024)
        short_path = buf.value if res > 0 else raw
        if short_path and short_path.isascii() and Path(short_path).exists():
            return short_path

    local_base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AirTouch" / "models" / target.name
    if str(local_base).isascii():
        marker = local_base / "final.mdl"
        if not marker.exists():
            local_base.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target, local_base, dirs_exist_ok=True)
        return str(local_base)
    return raw


def _is_non_mic_device(name: str) -> bool:
    low = (name or "").lower()
    return any(kw in low for kw in _LOOPBACK_NON_MIC_KEYWORDS)


def _score_input_device(idx: int, dev: Dict[str, Any], hostapis: List[Dict[str, Any]], default_in: Optional[int]) -> int:
    name = str(dev.get("name", ""))
    low = name.lower()
    if _is_non_mic_device(name):
        return -1000

    score = 0
    if "mikrofon dizisi" in low or "microphone array" in low:
        score += 70
    elif "mikrofon" in low or "microphone" in low or "mic" in low:
        score += 55
    if "usb" in low or "webcam" in low:
        score += 20
    if "kulaklık" in low or "kulaklik" in low or "headset" in low or "hands-free" in low:
        score += 15

    ha_idx = int(dev.get("hostapi", 0))
    ha_name = str(hostapis[ha_idx].get("name", "")).lower() if 0 <= ha_idx < len(hostapis) else ""
    if "mme" in ha_name:
        score += 25
    elif "directsound" in ha_name:
        score += 18
    elif "wasapi" in ha_name:
        score += 8
    elif "wdm-ks" in ha_name:
        score -= 30  # WDM-KS Bluetooth aygıtları kilitliyse -9999 hatası verebilir

    if default_in is not None and idx == default_in:
        score += 10
    return score


def get_ranked_input_devices(preferred_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Windows'ta 'Stereo Karışımı' varsayılan yapılsa bile gerçek mikrofonları
    en güvenilir olandan başlayarak puan sırasıyla döndürür.
    """
    try:
        import sounddevice as sd
    except Exception:
        return []

    try:
        devs = list(sd.query_devices())
        hostapis = list(sd.query_hostapis())
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    except Exception:
        return []

    candidates: List[Dict[str, Any]] = []
    for idx, dev in enumerate(devs):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        name = str(dev.get("name", f"Mikrofon {idx}")).strip()
        sc = _score_input_device(idx, dev, hostapis, default_in)
        if preferred_index is not None and idx == preferred_index:
            sc += 500
        candidates.append(
            {
                "index": idx,
                "name": name,
                "default_samplerate": float(dev.get("default_samplerate", 16000.0)),
                "score": sc,
                "is_real_mic": sc > 0,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def resolve_best_input_device(preferred_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Kullanılacak en iyi gerçek mikrofon kaydını döndürür."""
    ranked = get_ranked_input_devices(preferred_index=preferred_index)
    return ranked[0] if ranked else None


def list_input_microphones() -> List[Dict[str, Any]]:
    """Kullanılabilir gerçek giriş mikrofonlarını (Stereo Karışımı/hoparlörleri süzerek) listeler."""
    ranked = get_ranked_input_devices(preferred_index=None)
    seen = set()
    mics: List[Dict[str, Any]] = []
    best_idx = ranked[0]["index"] if ranked else None

    for item in ranked:
        if item["score"] <= -500:
            continue
        name = item["name"]
        key = name[:26].lower()
        if key in seen and item["index"] != best_idx:
            continue
        seen.add(key)
        mics.append(
            {
                "index": item["index"],
                "name": name,
                "is_default": item["index"] == best_idx,
            }
        )
    return mics


def _resample_to_16k_int16(chunk_bytes: bytes, src_rate: int, apply_agc: bool = True, noise_floor: float = 0.003) -> Tuple[bytes, float]:
    """Gelen PCM-16 veriyi 16 kHz'e yeniden örnekler, RMS hesaplar ve kısık mikrofona güvenli AGC uygular."""
    samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b"", 0.0

    # DC offset çıkar
    samples = samples - float(np.mean(samples))

    if src_rate != 16000 and src_rate > 0:
        n_out = max(1, int(round(samples.size * 16000.0 / float(src_rate))))
        xp = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        samples = np.interp(x_new, xp, samples)

    rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0)

    if apply_agc:
        peak = float(np.max(np.abs(samples)))
        # Gürültü tabanının belirgin üzerindeki konuşma parçalarını güçlendir
        if rms > max(0.0022, noise_floor * 1.55) and 150.0 < peak < 11000.0:
            gain = min(8.0, 12000.0 / peak)
            samples = samples * gain

    clipped = np.clip(samples, -32767.0, 32767.0).astype(np.int16)
    return clipped.tobytes(), rms


class SoundFeedbackWorker:
    """Video karesini dondurmadan asenkron kısa onay tonu çalar."""

    def __init__(self):
        self._q: "queue.Queue[Tuple[int, int]]" = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            freq, dur = self._q.get()
            if sys.platform == "win32":
                try:
                    import winsound

                    winsound.Beep(freq, dur)
                except Exception:
                    pass

    def play(self, kind: str = "action", enabled: bool = True):
        if not enabled or sys.platform != "win32":
            return
        tones = {
            "action": (920, 55),
            "voice": (1180, 65),
            "arm": (1040, 45),
            "pause": (480, 80),
            "profile": (760, 60),
        }
        freq, dur = tones.get(kind, (880, 50))
        try:
            self._q.put_nowait((freq, dur))
        except queue.Full:
            pass


sound_feedback = SoundFeedbackWorker()


class VoiceCommander:
    """
    Sürekli arka plan mikrofon dinleyici & Türkçe sesli komut çözücü.
    Windows'ta 'Stereo Karışımı' tuzağını atlayıp gerçek mikrofonu otomatik seçer,
    VAD ile konuşma bittiği anda ~0.55s içinde komutu çözümler.
    """

    def __init__(
        self,
        engine: str = "vosk",
        wake_required: bool = False,
        mic_device_index: Optional[int] = None,
        cooldown_seconds: float = 0.70,
        model_dir: Path = VOSK_TR_MODEL_DIR,
    ):
        self.engine_preference = engine
        self.wake_required = bool(wake_required)
        self.mic_device_index = mic_device_index
        self.cooldown_seconds = float(cooldown_seconds)
        self.model_dir = Path(model_dir)

        self._lock = threading.Lock()
        self._cmd_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._running = False
        self._available = False
        self._active_engine = "none"
        self._active_mic_name = ""
        self._active_mic_index: Optional[int] = None
        self._mic_level = 0.0
        self._last_partial = ""
        self._last_heard = ""
        self._last_command = ""
        self._last_command_time = 0.0
        self._last_error: Optional[str] = None
        self._last_trigger_ts = 0.0

    def start(self) -> bool:
        if self._running:
            return True
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        with self._lock:
            self._mic_level = 0.0
            self._last_partial = ""

    def pop_command(self) -> Optional[Dict[str, Any]]:
        """Ana döngüden tüketilmek üzere sıradaki sesli komutu non-blocking döndürür."""
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "available": self._available,
                "engine": self._active_engine,
                "mic_name": self._active_mic_name,
                "mic_index": self._active_mic_index,
                "wake_required": self.wake_required,
                "mic_level": float(self._mic_level),
                "last_partial": self._last_partial,
                "last_heard": self._last_heard,
                "last_command": self._last_command,
                "last_command_time": self._last_command_time,
                "error": self._last_error,
            }

    def _enqueue_if_matched(self, text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        with self._lock:
            self._last_heard = cleaned[:64]
            self._last_partial = ""

        matched = match_voice_command(cleaned, wake_required=self.wake_required)
        if not matched:
            return

        now = time.time()
        if now - self._last_trigger_ts < self.cooldown_seconds:
            return
        self._last_trigger_ts = now

        with self._lock:
            self._last_command = matched["label"]
            self._last_command_time = now

        try:
            self._cmd_queue.put_nowait(matched)
        except queue.Full:
            pass

    def _open_working_input_stream(self, sd, callback, chunk_ms: float = 125.0):
        """
        En yüksek puanlı gerçek mikrofonları dener; 16 kHz desteklenmezse
        aygıtın doğal örnekleme hızını açıp yazılım içinde 16 kHz'e çevirir.
        """
        candidates = get_ranked_input_devices(preferred_index=self.mic_device_index)
        if not candidates:
            candidates = [{"index": self.mic_device_index, "name": "Varsayılan", "default_samplerate": 16000.0}]

        last_exc: Optional[Exception] = None
        for cand in candidates:
            dev_idx = cand["index"]
            dev_name = cand["name"]
            nat_sr = int(round(cand.get("default_samplerate") or 16000))
            rates = [16000] if nat_sr == 16000 else [16000, nat_sr]

            for sr in rates:
                blocksize = max(512, int(round(sr * (chunk_ms / 1000.0))))
                try:
                    stream = sd.RawInputStream(
                        samplerate=sr,
                        blocksize=blocksize,
                        device=dev_idx,
                        dtype="int16",
                        channels=1,
                        callback=callback,
                    )
                    with self._lock:
                        self._active_mic_index = dev_idx
                        self._active_mic_name = dev_name
                    return stream, sr
                except Exception as exc:
                    last_exc = exc
        raise RuntimeError(f"Hiçbir giriş mikrofonu açılamadı: {last_exc}")

    def _run_loop(self):
        try:
            import sounddevice as sd
        except Exception as exc:
            with self._lock:
                self._available = False
                self._last_error = f"sounddevice yok: {exc}"
            self._running = False
            return

        vosk_rec = None
        if self.engine_preference == "vosk":
            safe_dir = get_cpp_safe_model_path(self.model_dir)
            if safe_dir:
                try:
                    import vosk

                    vosk.SetLogLevel(-1)
                    model = vosk.Model(safe_dir)
                    vosk_rec = vosk.KaldiRecognizer(model, 16000)
                    with self._lock:
                        self._active_engine = "vosk-tr"
                        self._available = True
                        self._last_error = None
                except Exception as exc:
                    vosk_rec = None
                    with self._lock:
                        self._last_error = f"Vosk yüklenemedi: {exc}"

        if vosk_rec is not None:
            self._run_vosk_stream(sd, vosk_rec)
        else:
            self._run_google_fallback_stream(sd)

    def _run_vosk_stream(self, sd, rec):
        audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

        def _audio_cb(indata, frames, time_info, status):
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        try:
            stream, active_sr = self._open_working_input_stream(sd, _audio_cb, chunk_ms=125.0)
            noise_floor = 0.003
            last_speech_ts = 0.0
            has_unflushed = False
            prev_partial = ""

            with stream:
                while not self._stop_event.is_set():
                    try:
                        raw_chunk = audio_q.get(timeout=0.20)
                    except queue.Empty:
                        with self._lock:
                            self._mic_level *= 0.65
                        if has_unflushed and (time.time() - last_speech_ts) >= 0.55:
                            fin = json.loads(rec.FinalResult()).get("text", "")
                            has_unflushed = False
                            prev_partial = ""
                            self._enqueue_if_matched(fin)
                        continue

                    pcm_16k, rms = _resample_to_16k_int16(
                        raw_chunk, src_rate=active_sr, apply_agc=True, noise_floor=noise_floor
                    )
                    if not pcm_16k:
                        continue

                    now = time.time()
                    is_loud = rms > max(0.004, noise_floor * 2.1)
                    if not is_loud:
                        noise_floor = 0.96 * noise_floor + 0.04 * min(rms, 0.025)
                    else:
                        last_speech_ts = now

                    norm_lvl = min(1.0, rms * 14.0)
                    with self._lock:
                        self._mic_level = 0.55 * self._mic_level + 0.45 * norm_lvl

                    if rec.AcceptWaveform(pcm_16k):
                        res_txt = json.loads(rec.Result()).get("text", "")
                        has_unflushed = False
                        prev_partial = ""
                        self._enqueue_if_matched(res_txt)
                    else:
                        partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                        if partial:
                            with self._lock:
                                self._last_partial = partial[:56]
                            if partial != prev_partial:
                                prev_partial = partial
                                last_speech_ts = now
                            has_unflushed = True
                        elif has_unflushed and (now - last_speech_ts) >= 0.55:
                            fin = json.loads(rec.FinalResult()).get("text", "")
                            has_unflushed = False
                            prev_partial = ""
                            self._enqueue_if_matched(fin)
        except Exception as exc:
            with self._lock:
                self._available = False
                self._last_error = f"Mikrofon akış hatası: {exc}"
        finally:
            self._running = False

    def _run_google_fallback_stream(self, sd):
        try:
            import speech_recognition as sr
        except Exception as exc:
            with self._lock:
                self._available = False
                self._last_error = f"Ses tanıma modülü bulunamadı: {exc}"
            self._running = False
            return

        with self._lock:
            self._active_engine = "google-tr"
            self._available = True
            self._last_error = None

        recognizer = sr.Recognizer()
        speech_buf: List[bytes] = []
        silent_chunks = 0
        recording = False
        audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

        def _audio_cb(indata, frames, time_info, status):
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        try:
            stream, active_sr = self._open_working_input_stream(sd, _audio_cb, chunk_ms=100.0)
            noise_floor = 0.004
            with stream:
                while not self._stop_event.is_set():
                    try:
                        chunk = audio_q.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    pcm_16k, rms = _resample_to_16k_int16(
                        chunk, src_rate=active_sr, apply_agc=True, noise_floor=noise_floor
                    )
                    if not pcm_16k:
                        continue

                    with self._lock:
                        self._mic_level = 0.55 * self._mic_level + 0.45 * min(1.0, rms * 14.0)

                    if rms >= max(0.010, noise_floor * 2.4):
                        recording = True
                        silent_chunks = 0
                        speech_buf.append(pcm_16k)
                        if len(speech_buf) > 45:
                            self._recognize_google_chunk(sr, recognizer, b"".join(speech_buf))
                            speech_buf.clear()
                            recording = False
                    elif recording:
                        speech_buf.append(pcm_16k)
                        silent_chunks += 1
                        if silent_chunks >= 6:
                            if len(speech_buf) >= 4:
                                self._recognize_google_chunk(sr, recognizer, b"".join(speech_buf))
                            speech_buf.clear()
                            recording = False
                            silent_chunks = 0
                    else:
                        noise_floor = 0.96 * noise_floor + 0.04 * min(rms, 0.025)
        except Exception as exc:
            with self._lock:
                self._available = False
                self._last_error = f"Mikrofon hatası: {exc}"
        finally:
            self._running = False

    def _recognize_google_chunk(self, sr_mod, recognizer, pcm_bytes: bytes):
        try:
            audio_data = sr_mod.AudioData(pcm_bytes, 16000, 2)
            text = recognizer.recognize_google(audio_data, language="tr-TR")
            self._enqueue_if_matched(text)
        except Exception:
            pass
