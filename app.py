import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import os
import base64
import tempfile
import cv2
import time

from utils1 import load_model, detect_image_streamlit, detect_video_streamlit, class_names, class_colors

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Kematangan TBS Kelapa Sawit",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function untuk base64
@st.cache_data
def get_image_base64(filepath):
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: File '{filepath}' tidak ditemukan.")
        return ""

# CSS Styles
def load_css():
    return """
    <style>
    /* Basic Styling */
    .stSidebar > div:first-child {
        /* Default Streamlit colors */
    }

    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stText, .stAlert {
        color: var(--text-color);
        text-shadow: none;
    }

    .stButton > button, .stDownloadButton > button {
        width: 100% !important;
        padding: 12px 20px;
        font-size: 1rem;
        text-align: center;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    /* Button Styles */
    .stButton button.primary-btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton button.primary-btn:hover {
        background-color: #45a049;
    }

    .stDownloadButton button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stDownloadButton button:hover {
        background-color: #0056b3;
    }

    /* Layout responsive */
    .main .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stImage img, .stVideo video {
            max-width: 100% !important;
            height: auto !important;
        }
    }

    /* Card Styles */
    .card {
        background-color: rgba(128, 128, 128, 0.15);
        backdrop-filter: blur(5px);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.2);
        text-align: center;
    }

    .info-box {
        background-color: rgba(128, 128, 128, 0.15);
        backdrop-filter: blur(5px);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.2);
    }

    /* Detection containers */
    .detection-container, .result-container {
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        background-color: var(--secondary-background-color);
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        display: flex; /* Use flexbox for consistent height */
        flex-direction: column;
        justify-content: flex-start; /* Align content to the top */
        min-height: 100px; /* Reduced to be very compact */
    }

    .result-container {
        border-color: #28a745;
    }

    .detection-container .stImage img, .result-container .stImage img {
        max-height: 380px !important; /* Adjust height to fit container padding */
        min-height: 50px !important; /* Reduced min-height */
        object-fit: contain !important;
        border-radius: 8px !important;
        width: 100% !important;
        flex-grow: 1; /* Allow image to grow within container */
    }

    .detection-container .stVideo video, .result-container .stVideo video {
        max-height: 380px !important; /* Adjust height for video */
        min-height: 50px !important; /* Reduced min-height */
        object-fit: contain !important;
        border-radius: 8px !important;
        width: 100% !important;
        flex-grow: 1; /* Allow video to grow within container */
    }

    /* Developer card */
    .developer-card {
        background-color: rgba(128, 128, 128, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
    }

    .developer-photo {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #007bff;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.6);
    }

    .developer-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 0.5rem;
    }

    .developer-info {
        text-align: center;
    }

    /* Detection results */
    .simple-detection-box {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .simple-detection-box h4 {
        color: #28a745;
        margin-bottom: 0.6rem;
        font-size: 1.2rem;
    }

    /* How to use items */
    .how-to-use-item {
        background-color: rgba(128, 128, 128, 0.08);
        backdrop-filter: blur(6px);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }

    .how-to-use-item:hover {
        transform: translateY(-5px);
    }

    .how-to-use-item .icon {
        font-size: 1.8rem;
        color: #4CAF50;
    }

    /* Footer */
    .footer-container {
        margin-top: 2rem;
        text-align: center;
        padding: 1rem;
        background-color: rgba(128, 128, 128, 0.15);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .section-heading {
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        color: var(--text-color);
    }

    /* Added for Detection Page - make elements fill width where appropriate */
    .stRadio > label, .stSlider > label {
        width: 100%;
    }
    .stRadio div[role="radiogroup"] {
        display: flex;
        justify-content: space-around;
        width: 100%;
    }
    .stRadio div[role="radiogroup"] label {
        flex-grow: 1;
        text-align: center;
        margin: 0 5px;
        padding: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        background-color: rgba(128, 128, 128, 0.05);
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .stRadio div[role="radiogroup"] label.st-dg { /* selected radio button */
        background-color: #4CAF50 !important;
        color: white !important;
        border-color: #4CAF50 !important;
    }
    .stRadio div[role="radiogroup"] label.st-dg span {
        color: white !important;
    }

    /* Style for file uploader area */
    .stFileUploader > div > div {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background-color: var(--secondary-background-color);
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
    }

    /* Instruction message styling */
    .instruction-message {
        display: flex;
        flex-direction: column;
        align-items: center; /* Keep centered horizontally */
        justify-content: flex-start; /* Align content to top */
        text-align: center;
        background-color: var(--secondary-background-color);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem; /* Padding will define height now */
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        width: 100%; /* Ensure it takes full width of its parent */
    }
    .instruction-message p, .instruction-message h4 {
        margin: 0.5rem 0; /* Adjust internal margins */
    }

    </style>
    """

st.markdown(load_css(), unsafe_allow_html=True)

# Load images
def load_images():
    images = {
        'top_banner': "static/banner_top25.png",
        'main_banner': "static/main_banner.png",
        'bottom_banner': "static/banner_bottom12.png",
        'developer_photo': "static/profile.jpg",
        'mentah': "static/mentah.png",
        'kurang_matang': "static/kurang matang.png",
        'matang': "static/matang.png",
        'terlalu_matang': "static/terlalu matang.png"
    }
    return images

# Sidebar
def create_sidebar():
    images = load_images()

    if os.path.exists(images['top_banner']):
        st.sidebar.image(images['top_banner'], use_column_width=True)

    menu = st.sidebar.selectbox("📂 Menu Navigasi", ["🏠 Home", "🔍 Detection", "ℹ️ About"])

    if os.path.exists(images['bottom_banner']):
        st.sidebar.image(images['bottom_banner'], use_column_width=True)

    return menu

# Main banner
def show_main_banner():
    images = load_images()
    if os.path.exists(images['main_banner']):
        st.markdown(f'<img src="data:image/png;base64,{get_image_base64(images["main_banner"])}" style="width:100%; height:auto; display:block; margin-bottom: 1rem;">', unsafe_allow_html=True)

# Load model
def initialize_model():
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None

    if not st.session_state.model_loaded:
        with st.spinner("🔄 Memuat model deteksi..."):
            model = load_model()
            if model:
                st.session_state.model = model
                st.session_state.model_loaded = True
            else:
                st.error("❌ Gagal memuat model.")
                st.stop()

# Footer function
def show_footer():
    st.markdown("""
        <div class="footer-container">
            <p style="font-size:0.9em; color:gray;">
                © 2024 Deteksi Kematangan TBS Kelapa Sawit. All rights reserved.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Home page
def show_home_page():
    st.markdown("---")
    st.markdown("""
        <div class='card'>
            <h2>Selamat Datang di Aplikasi Deteksi TBS Kelapa Sawit!</h2>
            <p>
            Solusi cerdas ini dirancang untuk mendeteksi tingkat kematangan <strong>Tandan Buah Segar (TBS)</strong> kelapa sawit secara otomatis melalui <em>gambar</em> dan <em>video</em>.
            Aplikasi ini dibuat untuk mempermudah proses pemanenan dengan teknologi deteksi berbasis kecerdasan buatan (Artificial Intelligence)</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # How to use section
    st.markdown("<h3 class='section-heading'>💡 How to Use the App</h3>", unsafe_allow_html=True)

    how_to_use_steps = [
        "Navigasikan ke menu <strong>\"🔍 Deteksi\"</strong> untuk memulai proses identifikasi.",
        "Pilih jenis media yang ingin Anda analisis: <strong>Gambar</strong> atau <strong>Video</strong>.",
        "Unggah file gambar atau video Anda melalui fitur <em>drag & drop</em> atau penelusuran file.",
        "Sesuaikan <strong>Confidence Threshold</strong> untuk mengatur sensitivitas deteksi.",
        "Tekan tombol <strong>\"Deteksi Gambar\"</strong> atau <strong>\"Deteksi Video\"</strong> dan biarkan AI bekerja!",
        "Lihat hasil deteksi dan unduh file yang sudah dianotasi jika diperlukan."
    ]

    for i, step in enumerate(how_to_use_steps, 1):
        st.markdown(f"""
        <div class="how-to-use-item">
            <span class="icon">{i}️⃣</span>
            <p>{step}</p>
        </div>
        """, unsafe_allow_html=True)

    # Classification section
    st.markdown("<h3 class='section-heading'>📈 Classification of Ripeness</h3>", unsafe_allow_html=True)
    

    images = load_images()
    classifications = [
        ("Mentah", "🟢 ""Buah sawit yang masih mentah umumnya memiliki warna hitam keunguan. Pada tahap ini, buah belum mengalami proses pematangan dan kandungan minyaknya masih sangat rendah. Jika dipanen dalam kondisi ini, hasil minyak yang diperoleh akan jauh dari optimal, sehingga tidak layak untuk dipanen karena akan menurunkan efisiensi dan nilai ekonomis.", images['mentah']),
        ("Kurang matang", "🟡 Buah yang belum matang sempurna menunjukkan warna hitam kemerahan. Meskipun sudah mulai mengalami perubahan warna, buah ini masih berada dalam proses pematangan dan belum mencapai titik maksimal dalam kandungan minyaknya. Memanen buah pada tahap ini juga belum direkomendasikan karena hasil minyaknya belum cukup tinggi, dan dapat mengurangi kualitas serta kuantitas produksi.", images['kurang_matang']),
        ("matang", "🟠 Buah sawit yang matang sempurna memiliki warna merah terang yang mencolok. Ini adalah indikator bahwa buah telah mencapai tingkat kematangan optimal, di mana kandungan minyak berada pada tingkat maksimum. Pada tahap inilah waktu terbaik untuk melakukan panen, karena hasil minyak yang diperoleh akan berkualitas tinggi dan efisien secara produksi.", images['matang']),
        ("Terlalu matang", "🔴 Buah yang terlalu matang ditandai dengan warna oranye kemerahan. Meskipun masih dapat dipanen, buah pada tahap ini mulai mengalami penurunan kualitas akibat proses pembusukan atau degradasi alami. Kandungan minyaknya juga mulai menurun, sehingga jika terus dibiarkan, buah ini berisiko menurunkan mutu minyak secara keseluruhan dan berdampak pada kualitas produksi akhir. Oleh karena itu, penting untuk memastikan waktu panen dilakukan saat buah berada pada kondisi matang sempurna.", images['terlalu_matang'])
    ]

    st.markdown("<p>Tingkat kematangan TBS kelapa sawit dapat diklasifikasikan sebagai berikut:</p>", unsafe_allow_html=True)
    for title, description, image_path in classifications:
        image_base64 = get_image_base64(image_path) if os.path.exists(image_path) else ""
        st.markdown(f"""
        <div class="info-box">
            <details>
                <summary><h4>{title}</h4></summary>
                <p>{description}</p>
                <p style="text-align: center;"><img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto; border-radius: 8px;"></p>
            </details>
        </div>
        """, unsafe_allow_html=True)
    show_footer()

# Detection functions
def handle_image_detection(uploaded_file, confidence_threshold):
    # Reset hasil deteksi ketika file baru diupload
    if uploaded_file and ('current_uploaded_file' not in st.session_state or
                            st.session_state.current_uploaded_file != uploaded_file.name):
        st.session_state.detection_result = None
        st.session_state.detection_status = None
        st.session_state.current_uploaded_file = uploaded_file.name
        st.session_state.download_image_ready = False # NEW: Flag to control download button visibility

    # Initialize session state for detection results
    for key in ['detection_result', 'detection_status', 'download_image_ready']: # Include new flag
        if key not in st.session_state:
            st.session_state[key] = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📥 Gambar Asli")
        # Display the image directly within the detection-container
        st.markdown('<div class="detection-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Use Streamlit columns directly for button alignment
        button_col1, button_col2, button_col3 = st.columns(3) # Max 3 buttons: Detect, Download, Reset

        with button_col1:
            detect_button = st.button("🔍 Deteksi", use_container_width=True, type="primary")

        download_image_data = None
        if st.session_state.get('download_image_ready', False) and st.session_state.detection_result:
            try:
                img_buf = BytesIO()
                st.session_state.detection_result.save(img_buf, format='PNG')
                download_image_data = img_buf.getvalue()
            except Exception as e:
                st.error(f"Error preparing image for download: {e}")
                download_image_data = None 

        with button_col2:
            if download_image_data is not None: 
                st.download_button(
                    "💾 Download", 
                    data=download_image_data,
                    file_name="hasil_deteksi.png",
                    mime="image/png",
                    use_container_width=True,
                    key="image_download_button_col"
                )
            else:
                
                st.markdown("<div style='height:46px;'></div>", unsafe_allow_html=True) # Approx button height
        

        with button_col3:
            # The reset button logic
            reset_disabled = not (st.session_state.detection_result is not None or st.session_state.detection_status is not None)
            reset_button = st.button(
                "🗑️ Reset", 
                use_container_width=True,
                type="secondary",
                key="image_reset_button_col",
                disabled=reset_disabled
            )
            if reset_button and not reset_disabled: 
                for key in ['detection_result', 'detection_status', 'current_uploaded_file', 'download_image_ready']: # Add download_image_ready to reset
                    st.session_state[key] = None
                st.rerun()

    with col2:
        st.markdown("#### ✅ Hasil Deteksi")
        # Always wrap content in result-container
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        image_result_placeholder = st.empty() 

        if detect_button:
            with st.spinner("Memproses deteksi gambar..."):
                result_image = detect_image_streamlit(uploaded_file, st.session_state.model, conf=confidence_threshold)

                if result_image and isinstance(result_image, Image.Image):
                    st.session_state.detection_result = result_image
                    st.session_state.detection_status = "success"
                    st.session_state.download_image_ready = True 
                    st.rerun() 
                else:
                    st.session_state.detection_result = "not_found"
                    st.session_state.detection_status = "no_object"
                    st.session_state.download_image_ready = False 

        if st.session_state.detection_status == "success" and st.session_state.detection_result:
            image_result_placeholder.image(st.session_state.detection_result, caption="Hasil Deteksi", use_column_width=True)
        elif st.session_state.detection_status == "no_object":
            # Using instruction-message directly here for consistency
            image_result_placeholder.markdown('<div class="instruction-message">'
                                                '<h4>⚠️ Objek TBS tidak ditemukan dalam gambar!</h4>'
                                                '<p>💡 Pastikan gambar mengandung Tandan Buah Segar (TBS) kelapa sawit yang jelas terlihat.</p>'
                                                '</div>', unsafe_allow_html=True)
        else: # Initial state or after reset
            image_result_placeholder.markdown('<div class="instruction-message">'
                                                '<p>Unggah gambar dan klik tombol deteksi untuk memproses.</p>'
                                                '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) 


        # Summary box outside the main display area, but still in col2
        if st.session_state.detection_status == "success" and st.session_state.detection_result:
            st.markdown("---")
            st.markdown("### ✅ Ringkasan Deteksi")
            st.markdown(f"""
                <div class="simple-detection-box">
                    <h4>Deteksi Berhasil!</h4>
                    <p>Gambar Anda telah berhasil diproses.</p>
                    <p>Confidence Threshold yang digunakan: <strong>{confidence_threshold:.2f}</strong></p>
                    <p style="font-style: italic;">Hasil visual ditampilkan di atas. Anda dapat mengunduhnya.</p>
                </div>
            """, unsafe_allow_html=True)

def handle_video_detection(uploaded_video, confidence_threshold):
    # Reset hasil deteksi video ketika file baru diupload
    if uploaded_video and ('current_uploaded_video' not in st.session_state or
                            st.session_state.current_uploaded_video != uploaded_video.name):
        st.session_state.video_detection_status = None
        st.session_state.current_uploaded_video = uploaded_video.name
        st.session_state.temp_video_path = None
        st.session_state.processed_frames_data = None 
        st.session_state.video_fps = None 
        st.session_state.video_width = None
        st.session_state.video_height = None
        st.session_state.download_video_ready = False 

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎞️ Video Asli")
        st.markdown('<div class="detection-container">', unsafe_allow_html=True)
        st.video(uploaded_video, format='video/mp4', start_time=0)
        st.markdown('</div>', unsafe_allow_html=True)

        # Use Streamlit columns directly for button alignment
        button_col1, button_col2, button_col3 = st.columns(3) 

        with button_col1:
            detect_video_button = st.button("🎬 Deteksi", use_container_width=True, type="primary")

        
        download_video_data = None
        output_video_filename = "hasil_deteksi_video.mp4"
        # Using tempfile.NamedTemporaryFile for safer temporary file handling
        temp_output_path_dl = os.path.join(tempfile.gettempdir(), output_video_filename) # Define path here

        if st.session_state.get('download_video_ready', False) and st.session_state.get('processed_frames_data') is not None:
            if st.session_state.get('video_fps') and st.session_state.get('video_width') and st.session_state.get('video_height'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                    out_dl = cv2.VideoWriter(temp_output_path_dl, fourcc, st.session_state.video_fps,
                                            (st.session_state.video_width, st.session_state.video_height))

                    if out_dl.isOpened():
                        for frame_bgr in st.session_state.processed_frames_data:
                            out_dl.write(frame_bgr)
                        out_dl.release()

                        with open(temp_output_path_dl, "rb") as file:
                            download_video_data = file.read()
                        # Clean up the temporary file immediately after reading
                        os.remove(temp_output_path_dl)
                    else:
                        st.error("Gagal membuat file video untuk diunduh. Pastikan codec 'mp4v' tersedia.")
                except Exception as e:
                    st.error(f"Error creating video for download: {e}")
            else:
                st.warning("Video info (FPS, width, height) tidak tersedia untuk mengunduh. Coba deteksi ulang.")

        with button_col2:
            if download_video_data is not None: # ONLY render if data is ready
                st.download_button(
                    "💾 Download",
                    data=download_video_data,
                    file_name=output_video_filename,
                    mime="video/mp4",
                    use_container_width=True,
                    key="video_download_button_col"
                )
            else:
                # Placeholder to maintain layout
                st.markdown("<div style='height:46px;'></div>", unsafe_allow_html=True) # Approx button height
        # --- MODIFICATION END ---

        with button_col3:
            # Reset button logic
            reset_disabled = not (st.session_state.video_detection_status is not None)
            reset_video_button = st.button(
                "🗑️ Reset",
                use_container_width=True,
                type="secondary",
                key="video_reset_button_col",
                disabled=reset_disabled
            )
            if reset_video_button and not reset_disabled:
                if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                    os.remove(st.session_state.temp_video_path)
                    st.session_state.temp_video_path = None
                for key in ['video_detection_status', 'current_uploaded_video', 'temp_video_path', 'processed_frames_data', 'video_fps', 'video_width', 'video_height', 'download_video_ready']: # Add download_video_ready to reset
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    with col2:
        st.markdown("#### ✅ Hasil Deteksi")
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        dynamic_video_content_placeholder = st.empty()

        if detect_video_button:
            if uploaded_video:
                dynamic_video_content_placeholder.markdown('<div class="instruction-message">'
                                                            '<h4>🔄 Memproses deteksi video...</h4>'
                                                            '</div>', unsafe_allow_html=True)

                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_video.read())
                    temp_video_path = temp_file.name
                    st.session_state.temp_video_path = temp_video_path

                try:
                    # Initialize for collecting frames for download
                    st.session_state.processed_frames_data = []

                    # Get video properties for later saving
                    cap = cv2.VideoCapture(temp_video_path)
                    st.session_state.video_fps = cap.get(cv2.CAP_PROP_FPS)
                    st.session_state.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    st.session_state.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    frame_count = 0
                    for processed_frame_rgb in detect_video_streamlit(temp_video_path, st.session_state.model, conf=confidence_threshold):
                        dynamic_video_content_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True, caption="Hasil Deteksi Video")
                        # Convert RGB to BGR for OpenCV VideoWriter
                        st.session_state.processed_frames_data.append(cv2.cvtColor(np.array(processed_frame_rgb), cv2.COLOR_RGB2BGR))
                        frame_count += 1

                    if frame_count > 0:
                        st.session_state.video_detection_result = "displayed_live"
                        st.session_state.video_detection_status = "success"
                        st.session_state.download_video_ready = True # Set flag to True after successful processing
                        st.rerun() # Rerun to display download button immediately
                    else:
                        dynamic_video_content_placeholder.empty()
                        st.session_state.video_detection_result = "not_found"
                        st.session_state.video_detection_status = "no_object"
                        st.session_state.download_video_ready = False # Ensure false if no objects found

                except Exception as e:
                    dynamic_video_content_placeholder.empty()
                    st.error(f"Terjadi kesalahan saat memproses video: {e}")
                    st.exception(e)
                    st.session_state.download_video_ready = False # Ensure false on error
                finally:
                    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                        os.remove(st.session_state.temp_video_path)
                        st.session_state.temp_video_path = None
            else:
                dynamic_video_content_placeholder.empty()
                dynamic_video_content_placeholder.markdown('<div class="instruction-message">'
                                                            '<h4>Silakan unggah video terlebih dahulu.</h4>'
                                                            '</div>', unsafe_allow_html=True)

        # Display initial/no-object messages based on session state
        if st.session_state.video_detection_status == "success" and st.session_state.video_detection_result == "displayed_live":
            # Video is already in dynamic_video_content_placeholder
            pass
        elif st.session_state.video_detection_status == "no_object":
            dynamic_video_content_placeholder.empty()
            dynamic_video_content_placeholder.markdown('<div class="instruction-message">'
                                                '<h4>⚠️ Objek TBS tidak ditemukan dalam video!</h4>'
                                                '<p>💡 Pastikan video mengandung Tandan Buah Segar (TBS) kelapa sawit yang jelas terlihat.</p>'
                                                '</div>', unsafe_allow_html=True)
        elif st.session_state.video_detection_status is None: # Initial state (no detection attempted yet or after reset)
            dynamic_video_content_placeholder.empty()
            # MODIFICATION START: Removed specific text strings for initial state
            dynamic_video_content_placeholder.markdown('<div class="instruction-message">'
                                                '<p></p>'
                                                '</div>', unsafe_allow_html=True)
            # MODIFICATION END:
        st.markdown('</div>', unsafe_allow_html=True) # Close the result-container

        # Summary box after the main detection display area, still in col2
        if st.session_state.video_detection_status == "success" and st.session_state.processed_frames_data:
            st.markdown("---")
            st.markdown("### ✅ Ringkasan Deteksi Video")
            st.markdown(f"""
                <div class="simple-detection-box">
                    <h4>Deteksi Video Berhasil!</h4>
                    <p>Video Anda telah berhasil diproses.</p>
                    <p>Confidence Threshold yang digunakan: <strong>{confidence_threshold:.2f}</strong></p>
                    <p style="font-style: italic;">Hasil visual ditampilkan di atas. Anda dapat mengunduhnya.</p>
                </div>
            """, unsafe_allow_html=True)


# Detection page
def show_detection_page():
    st.markdown("---")
    st.header("Deteksi Kematangan TBS")

    # Kontrol input di bagian atas, lebih terstruktur
    with st.container():

        col_type, col_conf = st.columns([0.7, 0.3]) # Lebihkan ruang untuk radio button

        with col_type:
            # Added custom CSS for radio buttons for elegance
            st.markdown("""
                <div class="stRadioWrapper">
                    <label class="stRadioLabel">Pilih Tipe File:</label>
                </div>
            """, unsafe_allow_html=True)
            file_type = st.radio("", ["Gambar", "Video"], key="file_type_radio", horizontal=True)

        with col_conf:
            st.markdown("<p style='margin-bottom: 0.5rem;'><strong>Confidence Threshold:</strong></p>", unsafe_allow_html=True)
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05, label_visibility="collapsed")
            st.markdown(f"<p style='text-align: center; font-size:0.9em; color:gray;'>Nilai saat ini: <strong>{confidence_threshold:.2f}</strong></p>", unsafe_allow_html=True)

    st.markdown("---") # Garis pemisah setelah pengaturan

    # Area unggah dan tampilan hasil deteksi
    if file_type == "Gambar":
        st.markdown("<h4 class='section-heading'>🖼️ Unggah Gambar Anda</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar untuk dianalisis (JPG/PNG):", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            handle_image_detection(uploaded_file, confidence_threshold)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="detection-container">'
                                        '<h4></h4>' # MODIFICATION: Removed "Unggah gambar untuk memulai deteksi."
                                        '<p></p>' # MODIFICATION: Removed "Format yang didukung: JPG, JPEG, PNG"
                                        '</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="result-container">'
                                        '<p></p>' # MODIFICATION: Removed "Hasil deteksi akan muncul di sini."
                                        '</div>', unsafe_allow_html=True)

    else: # Video
        st.markdown("<h4 class='section-heading'>🎞️ Unggah Video Anda</h4>", unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Pilih video untuk dianalisis (MP4/MOV):", type=["mp4", "mov"], label_visibility="collapsed")
        if uploaded_video:
            handle_video_detection(uploaded_video, confidence_threshold)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="detection-container">'
                                        '<h4></h4>' # MODIFICATION: Removed "Unggah video untuk memulai deteksi."
                                        '<p></p>' # MODIFICATION: Removed "Format yang didukung: MP4, MOV"
                                        '</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="result-container">'
                                        '<p></p>' # MODIFICATION: Removed "Hasil deteksi akan muncul di sini."
                                        '</div>', unsafe_allow_html=True)
    show_footer()

# About page
def show_about_page():
    st.markdown("---")
    st.markdown("<h2 class='section-heading'>ℹ️ Our App</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class='card'>
            <p style='font-size:1rem; line-height:1.6; text-align:justify;'>
                Aplikasi <strong>Deteksi Kematangan TBS Kelapa Sawit</strong> ini adalah proyek yang dikembangkan
                untuk membantu petani dan industri kelapa sawit dalam menentukan tingkat kematangan
                tandan buah segar secara otomatis menggunakan teknologi <em>Computer Vision</em>.
                Model deteksi dibangun dengan arsitektur <strong>YOLOv8</strong> yang dilatih khusus
                untuk mengidentifikasi various tingkat kematangan TBS.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 class='section-heading'>👨‍💻 Developer</h3>", unsafe_allow_html=True)

    images = load_images()
    developer_photo_base64 = get_image_base64(images['developer_photo']) if os.path.exists(images['developer_photo']) else ""

    if not developer_photo_base64:
        st.warning(f"File foto pengembang '{images['developer_photo']}' tidak ditemukan.")

    st.markdown(f"""
        <div class='developer-card'>
            <img src="data:image/png;base64,{developer_photo_base64}" class="developer-photo" alt="Foto Pengembang">
            <h3 class="developer-name">Zahwa Genoveva</h3>
            <div class="developer-info">
                <p><strong>Mahasiswa:</strong> Gunadarma University</p>
                <p><strong>Github:</strong> <a href="https://github.com/Awaviviana09" style="color: #007bff;">ZahGenoveva.com</a></p>
                <p><strong>Linkedin:</strong> <a href="https://www.linkedin.com/in/awagenovieve/" style="color: #007bff;">ZahwaGenoveva.com</a></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    show_footer()

# Main application logic
def main():
    initialize_model()
    menu_selection = create_sidebar()

    show_main_banner()

    if menu_selection == "🏠 Home":
        show_home_page()
    elif menu_selection == "🔍 Detection":
        show_detection_page()
    elif menu_selection == "ℹ️ About":
        show_about_page()

if __name__ == "__main__":
    main()