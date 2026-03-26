import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIGURATION
st.set_page_config(
    page_title="📱 Mobile Price Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-box {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
    }
    
    .result-price {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .result-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #9b59b6;
        margin-bottom: 15px;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(155, 89, 182, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Model'i yükle"""
    try:
        with open('mobile_price.pkl', 'rb') as f:
            bundle = pickle.load(f)
        
        model = bundle['model']
        return model
        
    except FileNotFoundError:
        try:
            model = joblib.load('mobile_price.joblib')
            return model
            
        except FileNotFoundError:
            st.error("❌ mobile_price.pkl veya mobile_price.joblib bulunamadı!")
            return None
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_price_category_info(category):
    """Fiyat kategorisi bilgisi"""
    categories = {
        0: {
            'name': '💰 Low Cost',
            'description': 'Bütçe Dostu',
            'emoji': '💚',
            'range': '$100 - $300',
            'target': 'Giriş seviyesi kullanıcılar',
        },
        1: {
            'name': '💵 Medium Cost',
            'description': 'Orta Fiyat',
            'emoji': '🟡',
            'range': '$300 - $600',
            'target': 'Standart kullanıcılar',
        },
        2: {
            'name': '💳 High Cost',
            'description': 'Yüksek Fiyat',
            'emoji': '🔴',
            'range': '$600 - $1200',
            'target': 'Premium kullanıcılar',
        },
        3: {
            'name': '👑 Very High Cost',
            'description': 'Çok Yüksek Fiyat',
            'emoji': '⬛',
            'range': '$1200+',
            'target': 'Lüks segment',
        }
    }
    return categories.get(category, categories[0])

def predict_price_category(model, features_dict):
    """Fiyat kategorisi tahmini yap"""
    try:
        # Model'in beklediği feature sırası
        feature_order = [
            'ram', 'performance_score', 'ram_x_cores', 'ram_to_memory',
            'battery_power', 'battery_per_weight', 'resolution',
            'px_width', 'px_height', 'pixel_density'
        ]
        
        # Feature vector oluştur
        feature_vector = []
        for feature in feature_order:
            if feature in features_dict:
                feature_vector.append(features_dict[feature])
            else:
                feature_vector.append(0)
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Tahmin yap
        prediction = model.predict(X)
        
        if isinstance(prediction, np.ndarray):
            prediction = int(prediction.item())
        else:
            prediction = int(prediction)
        
        prediction = max(0, min(3, prediction))
        
        # Olasılıkları al
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)[0]
        else:
            probas = None
        
        return prediction, probas
        
    except Exception as e:
        st.error(f"❌ Tahmin hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>📱 Mobile Price Classifier</h1>
        <p>🔮 Telefon özelliklerine göre fiyat kategorisini belirleyin</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model yükle
    model = load_model()
    if model is None:
        st.stop()
    
    st.success("✓ Model başarıyla yüklendi!")
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">📊 Telefon Özelliklerini Girin</div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        features_dict = {}
        extra_features = {}
        
        # ============================================================
        # TEMEL ÖZELLİKLER (MODEL İÇİN KULLANILANLAR)
        # ============================================================
        st.markdown("**⚙️ Temel Donanım Özellikleri**")
        st.caption("🤖 Bu özellikler fiyat tahminine etki eder")
        
        col1, col2 = st.columns(2)
        with col1:
            ram = st.slider("🧠 RAM (MB)", 512, 16000, 1024, 256,
                          help="Düşük: 512-2048 | Orta: 2048-4096 | Yüksek: 4096+")
            features_dict['ram'] = ram
        
        with col2:
            cores = st.slider("🔧 CPU Cores", 2, 12, 4)
        
        col1, col2 = st.columns(2)
        with col1:
            performance_score = st.slider("⚡ Performans Skoru (0-100)", 0, 100, 30,
                                        help="Düşük: 0-40 | Orta: 40-70 | Yüksek: 70-100")
            features_dict['performance_score'] = performance_score
        
        with col2:
            memory_capacity = st.slider("💾 Depolama (GB)", 16, 512, 32, 16,
                                      help="Düşük: 16-64 | Orta: 64-256 | Yüksek: 256+")
        
        col1, col2 = st.columns(2)
        with col1:
            battery_power = st.slider("🔋 Pil (mAh)", 1000, 6000, 2500, 100,
                                    help="Düşük: 1000-3000 | Orta: 3000-5000 | Yüksek: 5000+")
            features_dict['battery_power'] = battery_power
        
        with col2:
            weight = st.slider("⚖️ Ağırlık (g)", 100, 250, 180, 5)
        
        col1, col2 = st.columns(2)
        with col1:
            screen_size = st.slider("📏 Ekran Boyutu (inç)", 3.5, 7.0, 5.5, 0.1)
        
        with col2:
            pixel_density = st.slider("📊 Pixel Density (ppi)", 70, 500, 200, 10,
                                    help="Düşük: 70-200 | Orta: 200-400 | Yüksek: 400+")
            features_dict['pixel_density'] = pixel_density
        
        col1, col2 = st.columns(2)
        with col1:
            px_width = st.slider("📐 Piksel Genişlik", 720, 2400, 1080, 90)
            features_dict['px_width'] = px_width
        
        with col2:
            px_height = st.slider("📐 Piksel Yükseklik", 1280, 3200, 2160, 160)
            features_dict['px_height'] = px_height
        
        # Hesaplanan Model Features
        ram_x_cores = (ram / 1000) * cores
        features_dict['ram_x_cores'] = ram_x_cores
        
        ram_to_memory = ram / (memory_capacity * 1024)
        features_dict['ram_to_memory'] = ram_to_memory
        
        battery_per_weight = battery_power / weight
        features_dict['battery_per_weight'] = battery_per_weight
        
        resolution = pixel_density * screen_size
        features_dict['resolution'] = resolution
        
        st.divider()
        
        # ============================================================
        # İLAVE ÖZELLİKLER (SADECE GÖRÜNÜM İÇİN)
        # ============================================================
        st.markdown("**✨ İlave Özellikler**")
        st.caption("ℹ️ Bu özellikler fiyat tahminine etki etmez (bilgi amaçlı)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            wifi = st.checkbox("📡 WiFi", value=True)
            extra_features['wifi'] = wifi
        
        with col2:
            bluetooth = st.checkbox("🔵 Bluetooth", value=True)
            extra_features['bluetooth'] = bluetooth
        
        with col3:
            nfc = st.checkbox("💳 NFC", value=False)
            extra_features['nfc'] = nfc
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fast_charging = st.checkbox("⚡ Hızlı Şarj", value=True)
            extra_features['fast_charging'] = fast_charging
        
        with col2:
            fingerprint = st.checkbox("👆 Parmak İzi", value=True)
            extra_features['fingerprint'] = fingerprint
        
        with col3:
            face_recognition = st.checkbox("😊 Yüz Tanıma", value=False)
            extra_features['face_recognition'] = face_recognition
        
        col1, col2, col3 = st.columns(3)
        with col1:
            water_resistance = st.checkbox("💧 Su Geçirmez", value=False)
            extra_features['water_resistance'] = water_resistance
        
        with col2:
            wireless_charging = st.checkbox("🔌 Wireless Şarj", value=False)
            extra_features['wireless_charging'] = wireless_charging
        
        with col3:
            stereo_speakers = st.checkbox("🔊 Stereo Hoparlör", value=False)
            extra_features['stereo_speakers'] = stereo_speakers
        
        col1, col2 = st.columns(2)
        with col1:
            usb_type = st.selectbox("🔗 USB Tipi", ['Micro USB', 'USB-C', 'Lightning'])
            extra_features['usb_type'] = usb_type
        
        with col2:
            sim_slots = st.slider("📞 SIM Slot", 1, 3, 2)
            extra_features['sim_slots'] = sim_slots
        
        st.divider()
        predict_button = st.button("🔮 Fiyat Kategorisini Belirle", use_container_width=True)
    
    # ============================================================
    # MAIN AREA - RESULTS
    # ============================================================
    if predict_button:
        with st.spinner('⏳ Fiyat kategorisi belirleniyor...'):
            # Tahmin yap
            predicted_category, probas = predict_price_category(model, features_dict)
            
            if predicted_category is not None:
                category_info = get_price_category_info(predicted_category)
                
                # Sonuç kutusu
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">{category_info['emoji']} Fiyat Kategorisi</div>
                    <div class="result-price">{category_info['name']}</div>
                    <div class="result-label">{category_info['description']} - {category_info['range']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✓ Tahmin başarıyla tamamlandı!")
                
                st.divider()
                
                # Ana Metrikleri göster
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🧠 RAM", f"{ram} MB")
                with col2:
                    st.metric("⚡ Performans", f"{performance_score}/100")
                with col3:
                    st.metric("💾 Depolama", f"{memory_capacity} GB")
                with col4:
                    st.metric("🔋 Pil", f"{battery_power} mAh")
                
                st.divider()
                
                # Detaylı Bilgiler
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⚙️ Donanım Özellikleri")
                    hardware_df = pd.DataFrame({
                        'Özellik': ['RAM', 'Performans', 'Çekirdek', 'Depolama', 'Ağırlık'],
                        'Değer': [f"{ram} MB", f"{performance_score}/100", f"{cores} cores",
                                 f"{memory_capacity} GB", f"{weight} g"]
                    })
                    st.table(hardware_df)
                
                with col2:
                    st.subheader("🖥️ Ekran Özellikleri")
                    screen_df = pd.DataFrame({
                        'Özellik': ['Boyut', 'Pixel Density', 'Genişlik', 'Yükseklik', 'Çözünürlük'],
                        'Değer': [f"{screen_size:.1f}\"", f"{pixel_density} ppi",
                                 f"{px_width} px", f"{px_height} px", f"{resolution:.0f}"]
                    })
                    st.table(screen_df)
                
                st.divider()
                
                # Hesaplanan Oranlar
                st.subheader("📊 Hesaplanan Model Parametreleri")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🧠✖️ RAM × Cores", f"{ram_x_cores:.2f}")
                
                with col2:
                    st.metric("💾 RAM/Memory", f"{ram_to_memory:.4f}")
                
                with col3:
                    st.metric("🔋⚖️ Pil/Ağırlık", f"{battery_per_weight:.2f}")
                
                st.divider()
                
                # Olasılık Dağılımı
                if probas is not None:
                    st.subheader("📊 Kategori Olasılıkları")
                    
                    prob_data = {
                        'Kategori': ['💰 Low Cost', '💵 Medium Cost', '💳 High Cost', '👑 Very High Cost'],
                        'Olasılık (%)': [round(p * 100, 2) for p in probas]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    
                    fig_prob = px.bar(
                        prob_df, 
                        x='Kategori', 
                        y='Olasılık (%)',
                        title='Fiyat Kategorisi Olasılıkları',
                        color='Kategori',
                        color_discrete_map={
                            '💰 Low Cost': '#2ecc71',
                            '💵 Medium Cost': '#f39c12',
                            '💳 High Cost': '#e74c3c',
                            '👑 Very High Cost': '#34495e'
                        }
                    )
                    fig_prob.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    st.divider()
                
                # Kategori Bilgileri
                st.subheader(f"{category_info['emoji']} Hedef Kitle")
                st.info(f"""
                **Kategori:** {category_info['name']}
                
                **Fiyat Aralığı:** {category_info['range']}
                
                **Hedef:** {category_info['target']}
                """)
                
                st.divider()
                
                # İlave Özellikler Özeti
                st.subheader("✨ İlave Özellikler Özeti")
                
                extra_features_list = []
                if extra_features['wifi']:
                    extra_features_list.append("📡 WiFi")
                if extra_features['bluetooth']:
                    extra_features_list.append("🔵 Bluetooth")
                if extra_features['nfc']:
                    extra_features_list.append("💳 NFC")
                if extra_features['fast_charging']:
                    extra_features_list.append("⚡ Hızlı Şarj")
                if extra_features['fingerprint']:
                    extra_features_list.append("👆 Parmak İzi Sensörü")
                if extra_features['face_recognition']:
                    extra_features_list.append("😊 Yüz Tanıma")
                if extra_features['water_resistance']:
                    extra_features_list.append("💧 Su Geçirmez (IP Rating)")
                if extra_features['wireless_charging']:
                    extra_features_list.append("🔌 Wireless Şarj")
                if extra_features['stereo_speakers']:
                    extra_features_list.append("🔊 Stereo Hoparlör")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Mevcut Özellikler:**")
                    if extra_features_list:
                        for feature in extra_features_list:
                            st.success(f"✅ {feature}")
                    else:
                        st.info("ℹ️ Ek özellik seçilmedi")
                
                with col2:
                    st.write("**Bağlantı Bilgileri:**")
                    st.markdown(f"""
                    - **USB Tipi:** {extra_features['usb_type']}
                    - **SIM Slot:** {extra_features['sim_slots']} slot
                    """)
                
                st.divider()
                
                # Öneriler
                st.subheader("💡 Öneriler")
                
                recommendations = []
                
                if ram < 2048:
                    recommendations.append("🧠 RAM kapasitesi düşük - Temel görevler için uygun")
                elif ram < 4096:
                    recommendations.append("🧠 RAM kapasitesi orta - Çoğu uygulama için yeterli")
                else:
                    recommendations.append("🧠 RAM kapasitesi yüksek - Gaming ve multitasking için ideal")
                
                if battery_power < 3500:
                    recommendations.append("🔋 Pil kapasitesi kısıtlı - Sık şarja ihtiyaç duyabilir")
                elif battery_power > 5000:
                    recommendations.append("🔋 Pil kapasitesi bol - 2-3 gün yeterli olacak")
                else:
                    recommendations.append("🔋 Pil kapasitesi iyi - Tüm gün yeterli")
                
                if pixel_density < 200:
                    recommendations.append("📊 Ekran piksel yoğunluğu düşük - Standart görünüm")
                elif pixel_density > 400:
                    recommendations.append("📊 Ekran piksel yoğunluğu yüksek - Çok keskin görünüm")
                
                if performance_score < 40:
                    recommendations.append("⚡ Performans temel görevler için uygun")
                elif performance_score < 70:
                    recommendations.append("⚡ Performans iyi - Oyun ve uygulamalar rahat çalışır")
                else:
                    recommendations.append("⚡ Performans üstün - Yoğun uygulamalar için ideal")
                
                if extra_features['water_resistance']:
                    recommendations.append("💧 Su geçirmez - Zorlu koşullarda güvenli kullanım")
                
                if extra_features['wireless_charging']:
                    recommendations.append("🔌 Kablosuz şarj - Konforlu şarj deneyimi")
                
                if extra_features['face_recognition'] and extra_features['fingerprint']:
                    recommendations.append("🔒 Çift biyometrik - Maksimum güvenlik")
                
                for rec in recommendations:
                    st.info(rec)
            
            else:
                st.error("❌ Tahmin başarısız")
                st.warning("⚠️ Lütfen tüm özellikleri kontrol et")

if __name__ == "__main__":
    main()