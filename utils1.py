import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import os

# Tambahan untuk PyTorch 2.6+ agar bisa load YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.modules.conv import Conv
add_safe_globals([Conv])

# Nama kelas dan warna bounding box (BGR)
class_names = ['kurang matang', 'matang', 'mentah', 'terlalu matang']
class_colors = {
    0: (102, 214, 255),      # kurang matang - kuning pastel (BGR)
    1: (111, 118, 239),      # matang - hijau mint (BGR)
    2: (128, 0, 128),        # mentah - merah rose (BGR)
    3: (178, 138, 17),       # terlalu matang - biru elegan (BGR)
}

@st.cache_resource
def load_model():
    """
    Memuat model YOLOv8 dan menyimpannya dalam cache Streamlit.
    """
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}. Pastikan file 'best.pt' ada di direktori yang sama.")
        return None

def draw_boxes_on_frame(frame, results, class_names, class_colors):
    """
    Fungsi pembantu untuk menggambar bounding box dan label pada sebuah frame gambar.
    """
    frame_with_boxes = frame.copy()

    if results.boxes is not None and len(results.boxes.data) > 0:
        for result in results.boxes.data:
            x1, y1, x2, y2, score, cls = result.tolist()
            class_id = int(cls)
            label = f"{class_names[class_id]} ({score * 100:.1f}%)"
            color = class_colors.get(class_id, (255, 255, 255))
            
            # Gambar bounding box
            cv2.rectangle(frame_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Gambar latar belakang label
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame_with_boxes, (int(x1), int(y1) - text_height - 10), 
                          (int(x1) + text_width + 6, int(y1)), color, -1)
            
            # Gambar teks label
            cv2.putText(frame_with_boxes, label, (int(x1) + 3, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_with_boxes

def detect_image_streamlit(image_file, model, conf=0.3):
    """
    Melakukan deteksi objek pada gambar yang diunggah dan mengembalikan gambar beranotasi.
    """
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = model.predict(image_bgr, conf=conf, verbose=False)[0]

    if results.boxes is None or len(results.boxes.data) == 0:
        return None 

    processed_frame_bgr = draw_boxes_on_frame(image_bgr, results, class_names, class_colors)
    
    image_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(image_rgb)
    return result_image

def detect_video_streamlit(video_path, model, conf=0.3):
    """
    Melakukan deteksi objek pada video frame per frame.
    Mengembalikan generator yang menghasilkan setiap frame yang sudah dianotasi (BGR)
    dan juga mengembalikan list dari semua frame yang sudah diproses (BGR) untuk diunduh nanti.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Gagal membuka file video. Pastikan format video didukung dan file tidak rusak.")
        return # Keluar dari generator jika video tidak dapat dibuka

    # Mengambil properti video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Simpan properti ini ke session state agar bisa diakses nanti
    # Ini mungkin perlu dilakukan di app.py jika Anda ingin lebih ketat dalam mengelola session state
    # atau passed as part of a tuple/dict with the frames
    st.session_state['video_fps'] = fps
    st.session_state['video_width'] = width
    st.session_state['video_height'] = height

    processed_frames_for_download = [] # List untuk menyimpan semua frame yang diproses

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)[0]
        processed_frame = draw_boxes_on_frame(frame, results, class_names, class_colors)
        
        processed_frames_for_download.append(processed_frame) # Simpan frame untuk diunduh
        
        # Mengembalikan frame dalam format RGB untuk tampilan Streamlit
        yield cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    cap.release()
    
    # Setelah loop selesai, simpan processed_frames_for_download ke session_state
    # Ini akan diakses oleh app.py untuk membuat file yang dapat diunduh
    st.session_state['processed_frames_data'] = processed_frames_for_download