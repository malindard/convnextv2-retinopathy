import logging
import streamlit as st
import torch
import timm
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from streamlit_option_menu import option_menu

# ------------------ CSS Kustom ------------------
st.set_page_config(page_title="Klasifikasi Retinopati Diabetik", page_icon="👁️", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .stMarkdown {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Sidebar Info ------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Prediksi", "Tentang", "Kontak"],
        icons=["activity", "info-circle", "envelope"],
        default_index=0,
    )
    st.markdown("### ℹ️ Info Website")
    st.markdown("""
    **Author:** Malinda Ratnaduhita  
    **Model:** ConvNeXt V2  
    **Judul Skripsi:** Optimasi Klasifikasi Retinopati Diabetik Menggunakan ConvNeXt dan Metode Ben Graham pada Citra Fundus Retina
    """)

# ------------------ Logging ------------------
logging.basicConfig(level=logging.WARNING)

# ------------------ Cek Retina ------------------
def is_likely_retina(image: Image.Image) -> bool:
    img_np = np.array(image)
    h, w, _ = img_np.shape
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=h/4,
        param1=50, param2=25, minRadius=int(min(h, w)*0.2), maxRadius=int(min(h, w)*0.6)
    )
    if circles is not None:
        mean_color = img_np.mean(axis=(0, 1))
        r, g, b = mean_color
        if (r + g) / 2 > b and r > 60 and g > 60:
            return True
        else:
            logging.warning(f"Lingkaran terdeteksi, tetapi dominasi warna tidak sesuai: R={r}, G={g}, B={b}")
            return False
    logging.warning("Tidak ada lingkaran yang terdeteksi. Kemungkinan bukan citra retina.")
    return False

# ------------------ Load Model ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = timm.create_model(
        'convnextv2_base.fcmae_ft_in22k_in1k_384',
        pretrained=False,
        num_classes=5
    )
    state_dict = torch.load("model\convnextv2_base_weights.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ------------------ Halaman Prediksi ------------------
if selected == "Prediksi":
    st.title("👁️ Klasifikasi Retinopati Diabetik menggunakan ConvNeXt")
    st.markdown("""
        Website ini menggunakan model Deep Learning (ConvNeXt) untuk mengklasifikasikan tingkat keparahan Retinopati Diabetik dari citra fundus retina.
        \n**Cara Penggunaan:**
        \n1. Unggah gambar fundus retina pada kolom di sebelah kiri.
        \n2. Website akan memvalidasi apakah gambar yang diunggah adalah citra retina.
        \n3. Jika valid, klik tombol "Prediksi" untuk melihat hasilnya di sebelah kanan.
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Unggah Gambar")
        uploaded_file = st.file_uploader("Pilih gambar retina", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang Diunggah", width=400)

    with col2:
        st.header("Hasil Prediksi")
        if uploaded_file:
            if not is_likely_retina(image):
                st.warning("⚠️ Gambar bukan citra retina. Silakan unggah ulang.")
            else:
                st.info("✅ Citra valid. Klik 'Prediksi' untuk memproses.")
                if st.button("Prediksi", use_container_width=True):
                    img_array = np.array(image)
                    img_blur = cv2.GaussianBlur(img_array, (0, 0), 5)
                    img_preproc = cv2.addWeighted(img_array, 4, img_blur, -4, 128)
                    img_preproc = Image.fromarray(img_preproc.astype('uint8'), 'RGB')

                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(img_preproc).unsqueeze(0).to(device)

                    with st.spinner("⏳ Memproses..."):
                        model = load_model()
                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.softmax(output[0], dim=0).cpu().numpy()

                    kelas = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
                    idx = np.argmax(probs)
                    hasil_kelas = kelas[idx]

                    st.success(f"🎯 **Prediksi:** {hasil_kelas} (Probabilitas: {probs[idx]*100:.2f}%)")
                    st.markdown("---")
                    st.subheader("📊 Analisis Probabilitas Diagnosis")
                    st.write("""
                    Tabel di bawah ini menunjukkan probabilitas (keyakinan model) untuk setiap tingkat keparahan. Kelas dengan probabilitas tertinggi adalah hasil prediksi utama.
                    """)
                    df_probs = pd.DataFrame({'Probabilitas': probs * 100}, index=kelas)
                    df_table = df_probs.reset_index()
                    df_table.columns = ['Diagnosis', 'Probabilitas']
                    st.dataframe(df_table.style.format({'Probabilitas': "{:.2f}%"}), use_container_width=True, hide_index=True)
        else:
            st.info("📥 Silakan unggah gambar retina.")

# ------------------ Halaman Tentang ------------------
elif selected == "Tentang":
    st.title("Tentang")
    st.markdown("""
    ## 💡 Tentang Website Klasifikasi Retinopati Diabetik

    Website ini dikembangkan sebagai alat bantu diagnosis awal untuk **Retinopati Diabetik (DR)**, suatu komplikasi serius akibat penyakit diabetes yang menyerang retina dan dapat menyebabkan kebutaan. Dengan memanfaatkan **Deep Learning**, website ini mampu mengklasifikasikan tingkat keparahan DR dari **gambar fundus retina**.

    ---

    ### 🧠 Tujuan Pengembangan
    - 💻 **Membantu dokter dan tenaga medis** dalam proses skrining awal DR secara cepat dan akurat.
    - 📊 **Meningkatkan efisiensi dan objektivitas** dalam evaluasi gambar fundus retina.

    ---

    ### 🛠️ Teknologi yang Digunakan
    - **Model Deep Learning:** `ConvNeXt V2`, pretrained untuk klasifikasi 5 kelas DR
    - **Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
    - **Framework:** PyTorch, TorchVision
    - **Frontend UI:** Streamlit
    - **Preprocessing:** Metode Ben Graham

    ---

    ### 🧪 Tingkat Keparahan Retinopati Diabetik yang Dideteksi
    1. **No DR** — Retina normal
    2. **Mild** — Mikroaneurisma ringan
    3. **Moderate** — Perdarahan dan eksudat lebih banyak
    4. **Severe** — Banyak pembuluh darah abnormal, risiko tinggi
    5. **PDR (Proliferative DR)** — Pertumbuhan pembuluh darah baru, risiko kebutaan tinggi

    ---

    ### 📈 Akurasi Model
    Test Accuracy mencapai **86.44%**, berkat teknik *Ben Graham preprocessing* dan arsitektur ConvNeXt V2.

    ---

    ### 🚫 Catatan Penting
    > Website ini **bukan pengganti diagnosis dokter**, melainkan sebagai **alat bantu skrining awal**. Validasi klinis tetap diperlukan sebelum keputusan medis diambil.

    """)

# ------------------ Halaman Kontak ------------------
elif selected == "Kontak":
    st.title("Kontak")
    st.markdown("""
    **Malinda Ratnaduhita**  
    Mahasiswa Teknik Informatika dan Pengembang ML  
    📧 Email: malindard@students.unnes.ac.id  
    """)
