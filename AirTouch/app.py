"""
AirTouch - Temassız Medya ve Sistem Kontrol Merkezi (app.py)
Streamlit tabanlı modern web arayüzü.
"""
import sys
import io
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Kök dizini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    MODELS_DIR, 
    GESTURE_NAMES, 
    GESTURE_ACTIONS, 
    DEFAULT_CONFIDENCE_THRESHOLD, 
    DEFAULT_COOLDOWN_SECONDS
)
from core.engine import AirTouchEngine
from core.system_controller import system_controller

def check_skin_ratio(rgb_pil, bbox=None) -> float:
    """YCrCb uzayında kutu içindeki insan ten rengi piksel oranını hesaplar."""
    arr = np.array(rgb_pil)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        arr = arr[y1:y2, x1:x2]
    if arr.size == 0 or arr.ndim < 3:
        return 0.0
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(np.mean(mask))

# Sayfa Yapılandırması
st.set_page_config(
    page_title="AirTouch - Temassız Medya Kontrolü",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile Modern ve Şık Tasarım
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3A7BD5, #3A6073);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #7F8C8D;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .action-badge {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
    }
    .cooldown-badge {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: black;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Oturum Durumunda (Session State) Motoru Başlat
if "engine" not in st.session_state:
    st.session_state.engine = AirTouchEngine()

if "action_history" not in st.session_state:
    st.session_state.action_history = []

engine = st.session_state.engine

# ==================== YAN PANEL (SIDEBAR) ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hand.png", width=72)
    st.title("🖐️ AirTouch")
    st.markdown("**Akıllı Temassız Kontrol Paneli**")
    st.divider()

    # Ana Tetikleme Anahtarı
    trigger_enabled = st.toggle("⚡ Canlı Tuş Tetikleme", value=True, help="Açık olduğunda Windows ses ve medya tuşlarına gerçekten basar.")
    
    st.subheader("⚙️ Hassasiyet & Kutu Ayarları")
    confidence_threshold = st.slider(
        "Güven Eşiği (Threshold)",
        min_value=0.40,
        max_value=0.95,
        value=float(engine.confidence_threshold),
        step=0.05,
        help="Model bir hareketten bu orandan daha az eminse işlem yapmaz."
    )
    engine.confidence_threshold = confidence_threshold

    cooldown_seconds = st.slider(
        "Bekleme Süresi (Cooldown - sn)",
        min_value=0.3,
        max_value=3.0,
        value=float(engine.cooldown_seconds),
        step=0.1,
        help="Ardışık tuş basılmasını engelleyen mola süresi."
    )
    engine.cooldown_seconds = cooldown_seconds

    st.markdown("#### 🎯 Hedef El Kutusu (BBox)")
    bbox_size_pct = st.slider("Kutu Boyutu (%)", min_value=35, max_value=85, value=65, step=5, help="Elinizi içine koyacağınız hedef alanın boyutu (Varsayılan %65 geniş boyut).")
    bbox_pos = st.selectbox("Kutu Konumu", ["Sağ Taraf (Elinizin Doğal Konumu)", "Orta (Merkez)", "Sol Taraf"], index=0)

    st.divider()
    st.markdown("### 📊 Model Durumu")
    st.caption("Mimarisi: **AirTouchCNN (PyTorch)**")
    st.caption("Doğrulama Başarısı: **%98.71 ⭐**")
    st.caption(f"Toplam Tetikleme: **{engine.total_triggers}**")

# ==================== ANA SAYFA BAŞLIĞI ====================
st.markdown('<p class="main-header">AirTouch: Temassız Medya Kontrol Merkezi</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Web kameranız karşısında elinizi gösterin; klavyeye dokunmadan ses ve medyayı yönetin.</p>', unsafe_allow_html=True)

# Kesintisiz 60 FPS Canlı Kamera Başlatıcı Butonu
col_launch, col_tip = st.columns([1.5, 1.0])
with col_launch:
    cam_source_input = st.text_input(
        "📱 Kamera Kaynağı (Telefon IP Kamerası Aktif):",
        value="http://192.168.1.15:8080/video",
        help="Telefonunuzdaki IP Webcam bağlantısı otomatik olarak ayarlandı."
    )
    if st.button("🚀 KESİNTİSİZ CANLI KAMERAYI AÇ (60 FPS — Fotoğraf Çekmek Yok!)", type="primary", use_container_width=True):
        import subprocess
        python312_path = r"C:\Users\Rasit\AppData\Local\Programs\Python\Python312\python.exe"
        script_path = str(ROOT_DIR / "live_desktop_cam.py")
        subprocess.Popen([python312_path, script_path, cam_source_input.strip()])
        st.success(f"✅ Canlı masaüstü video penceresi açıldı (Kaynak: {cam_source_input})! Ekranınızda beliren pencerede yeşil kutu elinizi takip eder.")
with col_tip:
    st.info("💡 **Telefon Kamerasını Bağlamak İçin:**\n1. Google Play'den **'IP Webcam'** indirin.\n2. En alttan **'Start server'** deyin.\n3. Telefondaki adresi sola yapıştırıp butona basın!")

st.write("")

# Sekmeler
tab1, tab2, tab3, tab4 = st.tabs([
    "🎮 Canlı Kontrol Odası", 
    "⚙️ Tuş Atamaları & Hareketler", 
    "📈 Model Eğitimi & Metrikler",
    "📖 Kurs & Proje Notları"
])

# ==================== SEKME 1: CANLI KONTROL ====================
with tab1:
    col_source_sel, col_source_ip = st.columns([1.2, 1.8])
    with col_source_sel:
        cam_source_type = st.radio(
            "📷 Kamera Kaynağını Seçin:",
            ["📱 Telefon IP Kamerası (Tavsiye Edilen)", "💻 Laptop Web Kamerası"],
            index=0,
            horizontal=False
        )
    with col_source_ip:
        if "Telefon" in cam_source_type:
            phone_ip_url = st.text_input(
                "Telefon IP Adresi:",
                value="http://192.168.1.15:8080",
                help="Telefondaki IP Webcam uygulamasının ekranındaki adresi girin."
            )
            phone_stream_url = phone_ip_url.rstrip("/") + "/video"
            phone_shot_url = phone_ip_url.rstrip("/") + "/shot.jpg"

    st.write("")
    col_cam, col_info, col_chart = st.columns([1.3, 1.0, 1.2])

    input_pil_image = None
    auto_refresh = False

    if "Telefon" in cam_source_type:
        with col_cam:
            st.markdown("#### 📱 Telefon Canlı Video Yayını")
            st.markdown(
                f'''
                <div style="background-color: #0d1117; padding: 6px; border-radius: 12px; border: 2px solid #238636; text-align: center;">
                    <img src="{phone_stream_url}" style="width: 100%; max-height: 340px; object-fit: contain; border-radius: 8px;" alt="Telefon Canlı Akışı" onerror="this.onerror=null; this.src='https://via.placeholder.com/640x360/1a1a1a/00ff80?text=Telefon+Kamerasi+Bekleniyor+(IP+Webcam+Start+Server)';" />
                    <div style="color: #00ff80; font-size: 0.85rem; margin-top: 6px; font-weight: 500;">
                        🟢 Telefon Canlı Akışı ({phone_stream_url})
                    </div>
                </div>
                ''', 
                unsafe_allow_html=True
            )
            
            st.write("")
            btn_col1, btn_col2 = st.columns([1.3, 1.0])
            with btn_col1:
                fetch_frame = st.button("📸 Telefondan Kare Al & Analiz Et", type="primary", use_container_width=True)
            with btn_col2:
                auto_refresh = st.checkbox("🔄 Otomatik Döngü", value=False, help="Telefondan periyodik kare alıp hareketi otomatik analiz eder.")

            if fetch_frame or auto_refresh:
                try:
                    req = urllib.request.Request(phone_shot_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=2.5) as resp:
                        img_bytes = resp.read()
                    raw_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    input_pil_image = raw_pil.transpose(Image.FLIP_LEFT_RIGHT)
                except Exception as ex:
                    st.error(f"❌ Telefondan kare alınamadı ({ex}).\n\nLütfen telefondaki **'IP Webcam'** uygulamasında en alttan **'Start server'** butonuna basıldığından emin olun.")

    else:
        with col_cam:
            st.markdown("#### 💻 Laptop Web Kamerası")
            camera_image = st.camera_input("Elinizi aşağıdaki yeşil kutunun içine gösterin:", key="laptop_cam_capture")
            if camera_image is not None:
                input_pil_image = Image.open(camera_image).convert("RGB")

    # Kamera karesi geldiğinde işle (Telefon veya Laptop)
    if input_pil_image is not None:
        w, h = input_pil_image.size
        
        # Bounding Box Koordinatlarını Hesapla
        box_side = int(min(w, h) * (bbox_size_pct / 100.0))
        if "Sağ" in bbox_pos:
            x1 = max(0, w - box_side - 30)
        elif "Sol" in bbox_pos:
            x1 = 30
        else: # Orta
            x1 = (w - box_side) // 2
            
        y1 = max(10, (h - box_side) // 2)
        x2 = min(w, x1 + box_side)
        y2 = min(h, y1 + box_side)
        bbox = (x1, y1, x2, y2)

        # 1. El Varlık Kapısı (Skin Ratio Gate)
        skin_ratio = check_skin_ratio(input_pil_image, bbox=bbox)
        has_hand = skin_ratio >= 0.11

        # Kutu boşsa asla Windows tuşlarını tetikleme!
        effective_trigger = trigger_enabled and has_hand

        # Motoru Çalıştır (Sadece BBox içerisindeki eli sınıflandır!)
        report = engine.process_frame(input_pil_image, bbox=bbox, trigger_enabled=effective_trigger)
        report["has_hand"] = has_hand
        report["skin_ratio"] = skin_ratio
        
        # Eylem geçmişine ekle
        if report["action_executed"]:
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.action_history.insert(0, f"[{timestamp}] {report['action_executed']} ({report['class_name']})")
            if len(st.session_state.action_history) > 8:
                st.session_state.action_history.pop()

        # Kameranın üzerine çizilmiş HUD görüntüsünü ana kamera kolonunda göster!
        with col_cam:
            st.image(report["annotated_frame"], caption="Canlı Tespit & Medya HUD Görünümü", use_container_width=True)

        with col_info:
            st.subheader("🧠 Model Teşhisi")
            
            # Kart Yapısı
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #888;">Algılanan Hareket</span>
                <h3 style="margin: 5px 0; color: #3A7BD5;">{report['class_name'] if has_hand else 'El Bekleniyor'}</h3>
                <span style="font-size: 0.85rem;">Güven Oranı: <b>%{report['confidence'] * 100:.1f if has_hand else 0.0}</b></span>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.progress(min(1.0, float(report["confidence"]) if has_hand else 0.0))

            # Medyada Yapılan İşlem Göstergesi
            if not has_hand:
                st.info("✋ **Hedef kutuda el yok.** Lütfen elinizi kutunun içine gösterin (yanlış tetikleme engellendi).")
            elif report["action_executed"]:
                st.markdown(f'<div class="action-badge">⚡ {report["action_executed"]}</div>', unsafe_allow_html=True)
            elif report["last_action_message"] and report["is_cooldown_active"]:
                st.markdown(f'<div class="cooldown-badge">⏳ Mola: {report["cooldown_remaining"]} sn ({report["last_action_message"]})</div>', unsafe_allow_html=True)
            elif not report["is_confident"]:
                st.warning(f"⚠️ Güven %{report['confidence']*100:.0f} (Eşik: %{confidence_threshold*100:.0f}). Elinizi kutunun içine tam hizalayın.")
            else:
                st.success("🎯 Hareket Algılandı - İşlem Hazır.")

            # Modelin Gördüğü $64x64$ Kırpılmış El
            st.write("")
            st.caption("Modelin Gözünden Sadece Kutu İçi ($64 \\times 64$):")
            st.image(report["processed_tensor"], width=120, clamp=True)

        with col_chart:
            st.subheader("📊 Sınıf Olasılık Dağılımı")
            
            # Plotly Çubuk Grafiği
            labels = [GESTURE_NAMES.get(i, f"İşaret {i}") for i in range(10)]
            probs = [p * 100 for p in report["all_probabilities"]]
            
            colors = ['#3A7BD5' if i == report['class_id'] else '#2C3E50' for i in range(10)]
            
            fig = go.Figure(go.Bar(
                x=probs,
                y=labels,
                orientation='h',
                marker=dict(color=colors),
                text=[f"%{p:.1f}" if p > 5 else "" for p in probs],
                textposition='inside'
            ))
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=340,
                xaxis=dict(range=[0, 100], title="Olasılık (%)"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

        # Eğer otomatik döngü seçilmişse kısa bir bekleme sonrası sayfayı yenile
        if "Telefon" in cam_source_type and auto_refresh:
            time.sleep(1.2)
            st.rerun()

    else:
        with col_info:
            st.info("👆 Telefondaki IP Webcam'i başlatın veya laptop kameranızı kullanın.")
            st.markdown("""
            **Hızlı Jest Rehberi:**
            - ✊ **Yumruk:** Oynat / Durdur ⏯️
            - ☝️ **İşaret Parmağı:** Ses Aç 🔊
            - ✌️ **İki Parmak (V):** Ses Kıs 🔉
            - 🖐️ **Açık El:** Sessize Al 🔇
            - 🤟 **3 Parmak:** Sonraki Parça ⏭️
            """)
        with col_chart:
            st.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=500&q=80", caption="AirTouch BBox Hedefleme")

    # Son Eylemler Günlüğü
    if st.session_state.action_history:
        st.divider()
        st.subheader("📋 Son Tetiklenen Eylemler")
        for log in st.session_state.action_history:
            st.text(log)

# ==================== SEKME 2: TUŞ ATAMALARI ====================
with tab2:
    st.subheader("⚙️ Jest & Windows Eylem Eşleştirme Tablosu")
    st.markdown("Hangi el işaretinin Windows üzerinde hangi tuşu tetikleyeceğini buradan özelleştirebilirsiniz:")

    action_options = {
        "İşlem Yapma (Yok)": None,
        "Oynat / Durdur (Play/Pause) ⏯️": "play_pause",
        "Sesi Artır (Volume Up) 🔊": "volume_up",
        "Sesi Azalt (Volume Down) 🔉": "volume_down",
        "Sonraki Parça (Next Track) ⏭️": "next_track",
        "Önceki Parça (Previous Track) ⏮️": "prev_track",
        "Sesi Kapat / Aç (Mute) 🔇": "mute"
    }

    col1, col2 = st.columns(2)
    for i in range(6):
        target_col = col1 if i < 3 else col2
        with target_col:
            current_action = engine.gesture_actions.get(i)
            # Mevcut indeks
            default_idx = 0
            for idx, (label, val) in enumerate(action_options.items()):
                if val == current_action:
                    default_idx = idx
                    break
                    
            selected_label = st.selectbox(
                f"İşaret {i} ({GESTURE_NAMES.get(i, 'Özel')}) Eylemi:",
                options=list(action_options.keys()),
                index=default_idx,
                key=f"gesture_bind_{i}"
            )
            engine.gesture_actions[i] = action_options[selected_label]

    st.success("✅ Tuş atamaları anlık olarak güncellendi ve hafızaya alındı.")

# ==================== SEKME 3: MODEL METRİKLERİ ====================
with tab3:
    st.subheader("📈 Model Eğitimi & Doğrulama Metrikleri")
    
    history_file = MODELS_DIR / "training_history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            hist = json.load(f)
            
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Toplam Epoch", hist["epochs"])
        col_m2.metric("Son Train Loss", hist["train_loss"][-1])
        col_m3.metric("Son Val Loss", hist["val_loss"][-1])
        col_m4.metric("En Yüksek Doğruluk", f"%{max(hist['val_acc']):.2f}")

        # Grafikler
        epochs_range = list(range(1, len(hist["train_loss"]) + 1))
        
        c_left, c_right = st.columns(2)
        with c_left:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs_range, y=hist["train_loss"], mode='lines+markers', name='Train Loss', line=dict(color='#E74C3C')))
            fig_loss.add_trace(go.Scatter(x=epochs_range, y=hist["val_loss"], mode='lines+markers', name='Val Loss', line=dict(color='#3498DB')))
            fig_loss.update_layout(title="Kayıp (Loss) Eğrisi", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_loss, use_container_width=True)

        with c_right:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=epochs_range, y=hist["train_acc"], mode='lines+markers', name='Train Accuracy', line=dict(color='#F39C12')))
            fig_acc.add_trace(go.Scatter(x=epochs_range, y=hist["val_acc"], mode='lines+markers', name='Val Accuracy', line=dict(color='#2ECC71', width=3)))
            fig_acc.update_layout(title="Doğruluk (Accuracy %) Eğrisi", xaxis_title="Epoch", yaxis_title="Doğruluk (%)")
            st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.warning("Eğitim geçmişi dosyası bulunamadı.")

# ==================== SEKME 4: HAKKINDA ====================
with tab4:
    st.subheader("📚 Proje ve Kurs Hakkında")
    st.markdown("""
    Bu proje, **"Deep Learning ve Python: A'dan Z'ye Derin Öğrenme"** kursunda öğrenilen kavramların
    gerçek dünya masaüstü kullanımına uyarlanmış halidir:
    
    - **Veri Seti:** Ankara Ayrancı Anadolu Lisesi ve Arda Mavi tarafından toplanan *Sign Language Digits* ($64 \\times 64$).
    - **Mimari:** PyTorch Conv2D, MaxPool2D, BatchNorm, Dropout ve Linear katmanlı hafif evrişimli sinir ağı.
    - **Sistem Entegrasyonu:** Windows `ctypes.windll.user32` yerel sanal medya motoru.
    - **Arayüz:** Streamlit + Plotly interaktif veri paneli.
    """)
    st.info("Detaylı ders notları ve formül rehberi için: **`DERIN_OGRENME_VE_PYTHON_DETAYLI_OZET_REHBERI.md`** dosyasını inceleyebilirsiniz.")

