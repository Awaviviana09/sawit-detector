<h1 align="center">🌴 <b> Detecting Palm Fruit Ripeness Using Website-Based YOLOv8 </b></h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/YOLOv8-Detection-green?logo=ultralytics">
  <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

<p align="center"> 
  <i>Automatic Fresh Fruit Bunch (FFB) Ripeness Classification using YOLOv8 and Streamlit</i>
  <br>
  <i>Developed by Zahwa Genoveva</i>
</p>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/8f3c8a72-0b0f-4045-817e-2417d7c40907" 
       alt="Object Detection" 
       width="700px">
</p>

---
## 📝 Deskripsi Proyek

Sistem ini dirancang untuk mendeteksi **tingkat kematangan Tandan Buah Segar (TBS) kelapa sawit** secara otomatis menggunakan model **YOLOv8**. Aplikasi berbasis **web Streamlit** ini mampu mengklasifikasikan kematangan buah berdasarkan fitur visual warna kulit buah dengan akurasi tinggi, sehingga mendukung penerapan AI dalam otomasi proses penilaian kualitas buah kelapa sawit.

---

## ⚙️ Fitur Utama

-  **Deteksi Otomatis** — Menggunakan YOLOv8 dengan pendekatan anchor-free untuk hasil yang lebih akurat dan cepat.  
-  **Model Cerdas** — Dilatih dengan 6.592 citra TBS dari platform Roboflow, mencakup 4 tingkat kematangan.  
-  **Aplikasi Web Interaktif** — Dibangun menggunakan **Streamlit**, dapat melakukan deteksi langsung dari gambar maupun video.  
-  **Visualisasi Hasil** — Menampilkan bounding box, label tingkat kematangan, dan confidence score.   

---

## 🏗️ Arsitektur Sistem

<p align="center">
  <img src="https://github.com/user-attachments/assets/550c0bc2-34a8-41e5-8c80-facaf48aa537" 
       alt="Arsitektur YOLOv8"
       width="700px">
</p>

> *Gambar 1. Struktur arsitektur YOLOv8 terdiri dari tiga bagian utama: *Backbone, Neck,* dan *Head*,*


| Komponen | Deskripsi |
|-----------|------------|
| **Backbone** | Menggunakan *CSPNet* dengan modul **C2f** dan **SPPF (Spatial Pyramid Pooling – Fast)** untuk ekstraksi fitur utama. |
| **Neck** | Menggabungkan *Feature Pyramid Network (FPN)* dan *Path Aggregation Network (PANet)* untuk deteksi multi-skala. |
| **Head** | Desain **anchor-free split head** untuk menghasilkan bounding box dan klasifikasi secara efisien. |
---

## 🧩 Metode Pengembangan — CRISP-DM

Metodologi **Cross Industry Standard Process for Data Mining (CRISP-DM)** diterapkan dalam seluruh proses penelitian.  
CRISP-DM merupakan pendekatan standar yang banyak digunakan dalam proses pengembangan sistem berbasis data mining karena memiliki alur kerja yang terstruktur, fleksibel, dan telah terbukti efektif di berbagai sektor (Kuncoro, 2023). 

### 1️⃣ Business Understanding

Tahap awal difokuskan untuk memahami **tujuan bisnis dan permasalahan utama**, yaitu bagaimana meningkatkan efektivitas dan akurasi dalam menentukan kematangan buah kelapa sawit.  
Hasil dari tahap ini adalah definisi kebutuhan sistem yang mampu mengotomatisasi proses identifikasi tingkat kematangan TBS dengan presisi tinggi.

| **Klasifikasi Buah Matang** | **Klasifikasi Buah Terlalu Matang** | **Klasifikasi Buah Kurang Matang** | **Klasifikasi Buah Mentah** |
|----------------|----------------|----------------|----------------|
|<p align="center"><img src="https://github.com/user-attachments/assets/8a94e615-aa55-4d17-ad12-784156325de7" width="300px"></p> |<p align="center"><img src="https://github.com/user-attachments/assets/068d5538-c286-4cec-bc9f-d00cffd5b5e2" width="300px"></p>|<p align="center"><img src="https://github.com/user-attachments/assets/855b0302-e279-4e88-b6cc-9da7ae4bf910" width="300px"></p>|<p align="center"><img src="https://github.com/user-attachments/assets/c5f57194-61fc-4caa-ad6d-75601a792e58" width="300px"></p>|

### 2️⃣ Data Understanding

<p align="center">
  <img src="https://github.com/user-attachments/assets/e7d1fe77-8a1e-4704-a32b-9f6eff3558e0" width="650px">
</p>
Tahapan ini mencakup **pengumpulan dan eksplorasi dataset** dari platform **Roboflow**, berjumlah **6.592 citra** TBS dengan variasi tingkat kematangan dan sudut pengambilan gambar.  
Analisis dilakukan untuk memahami distribusi data, kondisi pencahayaan, dan potensi noise, agar model mampu mengenali pola visual secara menyeluruh.

| Kategori | Deskripsi |
|-----------|------------|
| 🍈 Mentah | Buah berwarna hitam keunguan tanpa indikasi kemerahan. |
| 🍊 Setengah Matang | Warna kulit mulai berubah ke hitam kemerahan. |
| 🍎 Matang | Warna merah-oranye dominan dan minyak mulai keluar. |
| 🍂 Lewat Matang | Buah gelap dan banyak yang lepas dari tandan. |

Dataset dikumpulkan dari berbagai sudut dan kondisi pencahayaan untuk meningkatkan generalisasi model.

### 3️⃣ Data Preparation

Dataset kemudian diproses melalui **anotasi manual** untuk menandai objek buah, diikuti **augmentasi data** seperti rotasi, perubahan kecerahan, dan blur.  
Langkah ini bertujuan memperluas variasi data, meningkatkan generalisasi model, serta mengurangi potensi overfitting selama pelatihan.

| **Annotation** | **Split** |
|----------------|----------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/f2eaccd5-898a-45df-a0e0-27290e623d96" width="350px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/f53cc484-d6d8-4d61-b9a1-5927e0e57880" width="350px"></p> |


| **Auto - Orient** | **Auto - Adjust Contrast** | **Resize** |
|----------------|----------------|----------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/cc9ba19c-39d2-4d50-a6c7-f2dcbb38fa88" width="300px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/648523ff-9043-4b19-8e0f-5ee398c4e4d0" width="300px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/8be0fbd4-d374-43ad-848e-da015b017a33" width="300px"></p> |


### 4️⃣ Modeling
<p align="center">
  <img src="https://github.com/user-attachments/assets/fc04c234-c882-4088-9354-bfed4b57200f" width="650px">
</p>

Model dikembangkan menggunakan **YOLOv8** dengan konfigurasi anchor-free, learning rate terkontrol, serta pelatihan selama **3.107 jam**.  
Hasil pelatihan menunjukkan performa tinggi dengan **Precision 94.5%**, **Recall 94.7%**, dan **mAP@0.50 mencapai 98%**.  
Proses *hyperparameter tuning* dilakukan untuk mengoptimalkan batch size, epoch, dan augmentasi data.

Evaluasi Kinerja Model : 

| **Metrik** | **Nilai** |
|-------------|------------|
| Precision | 94.5% |
| Recall | 94.7% |
| F1-Confidence | 95% |
| mAP@0.50 | 98% |
| mAP@0.5–0.95 | 88% |

> Model menunjukkan performa yang stabil dan akurat untuk deteksi visual TBS kelapa sawit.
---

### 5️⃣ Evaluation

Tahap evaluasi digunakan untuk menilai performa model menggunakan berbagai metrik seperti **Confusion Matrix**, **Precision–Recall Curve**, dan **mAP**.  
Berikut visualisasi hasil evaluasi pada model YOLOv8:

---

| **Gambar 1 — Confusion Matrix** |
|:-------------------------------:|
| <p align="center"><img src="https://github.com/user-attachments/assets/e878e9b9-4f3b-41b9-b006-30d7a7885ec2" width="550px"></p> |
| *Menunjukkan distribusi antara prediksi benar (True Positive) dan salah (False Positive/False Negative) untuk tiap kelas kematangan.* |

---

| **Gambar 2 — Precision–Recall Curve** |
|:--------------------------------------:|
| <p align="center"><img src="https://github.com/user-attachments/assets/1ab0ba5a-82d7-45a4-b0e7-544945af7317" width="550px"></p> |
| *Kurva menggambarkan keseimbangan antara presisi dan sensitivitas model; area mendekati 1 menandakan kinerja deteksi sangat baik.* |

---


| **Gambar 3 — F1-Confidence Curve** |
|:-----------------------------------:|
| <p align="center"><img src="https://github.com/user-attachments/assets/ba43fedc-a1b0-4e33-b7af-607e041e1127" width="550px"></p> |
| *Menunjukkan hubungan antara confidence score dan nilai F1; semakin mendekati puncak, semakin stabil performa model terhadap variasi data.* |

---


| **Gambar 4 — Precision Curve per Class** |
|:------------------------------------------:|
| <p align="center"><img src="https://github.com/user-attachments/assets/b42399c0-0607-485c-8d06-bbe21fd2c244" width="550px"></p> |
| *Visualisasi presisi tiap kelas kematangan buah — menunjukkan seberapa akurat model dalam mengenali tiap kategori.* |

---

| **Gambar 5 — mAP Curve (Mean Average Precision)** |
|:--------------------------------------------------:|
| <p align="center"><img src="https://github.com/user-attachments/assets/b5806a5c-0c77-4eca-8ff2-1e52e79bfe74" width="550px"></p> |
| *mAP menilai akurasi keseluruhan deteksi model terhadap semua kelas. Nilai mAP@0.50 mencapai 98%, menandakan performa sangat tinggi.* |

---

### 6️⃣ Deployment
Model kemudian diintegrasikan ke dalam aplikasi **Streamlit berbasis web**, yang memungkinkan pengguna melakukan deteksi langsung terhadap gambar dan video.  
Aplikasi dapat dijalankan tanpa instalasi tambahan, menjadikannya solusi **praktis, efisien, dan mudah digunakan** di lingkungan perkebunan.

---


## 🏠 Tampilan Antarmuka Aplikasi

### 🏡 **Halaman Home**
<p align="center">Menampilkan antarmuka awal sistem dengan form input untuk pengunggahan gambar.</p>

| **Input Gambar 1** | **Input Gambar 2** |
|--------------------|--------------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/a03cd4dc-4218-4795-bc85-ba34f24d3f05" width="360px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/73925b6c-c6c8-4aa6-a905-50e413a06365" width="360px"></p> |

> *Menampilkan halaman Home pada aplikasi web Streamlit.*
---

### 🧾 **Halaman Deteksi**
<p align="center">Menampilkan proses dan hasil deteksi tingkat kematangan TBS kelapa sawit.</p>

| **Deteksi 1** | **Deteksi 2** | **Deteksi 3** |
|----------------|----------------|----------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/8d481304-872a-4400-aee3-d38ba754c6b4" width="300px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/bc48e246-a215-4a5a-a1f1-e316e88af071" width="300px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/a810477b-9278-46f4-a958-cd3b161b864b" width="300px"></p> |

| **Deteksi 4** | **Deteksi 5** |
|----------------|----------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/cfdd4346-a268-4ef0-b2e4-8a95b9fceb96" width="350px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/b8316a94-581b-4d85-b96c-353fd4f14c9e" width="350px"></p> |

> *Menampilka halaman deteksi pada sistem YOLOv8 berbasis Streamlit.*

---
### ℹ️ **Halaman Tentang (About)**
| **Tampilan Halaman About** |
|-----------------------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/09048e8d-ca23-4c20-8c55-dfbfc9c82827" width="700px"></p> |

> *Menampilka halaman informasi (About) aplikasi.*

---
## 🖼️ Visualisasi Hasil Deteksi

| **Citra Asli** | **Setelah Deteksi (YOLOv8)** |
|-----------------|-------------------------------|
| <p align="center"><img src="https://github.com/user-attachments/assets/58f8ced7-85bf-4344-b1fb-31b3cea7cdc2" width="350px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/e3e183f9-e6b3-4951-9115-1c70882689cd" width="350px"></p> |
| <p align="center"><img src="https://github.com/user-attachments/assets/9de1204f-3119-405b-b7f1-4ff8f592e8da" width="350px"></p> | <p align="center"><img src="https://github.com/user-attachments/assets/db233630-d732-4c4e-a1d7-976e556be92c" width="350px"></p> |

> *Menampilka Perbandingan hasil citra asli dan hasil deteksi tingkat kematangan TBS.*

---

## ⚙️ Teknologi yang Digunakan  

| Komponen | Teknologi |
|-----------|------------|
| Bahasa Pemrograman | Python |
| Framework ML | PyTorch, TensorFlow |
| Model Deteksi | YOLOv8 (Ultralytics) |
| Web Framework | Streamlit |
| Dataset Platform | Roboflow |
| Library Pendukung | OpenCV, NumPy, Matplotlib, Pandas |

---

## 💻 Cara Menjalankan Aplikasi  

1. **Clone Repositori**
   ```bash
   git clone https://github.com/username/palm-ripeness-detection.git
   cd palm-ripeness-detection

2. **Instal dependensi**
   ```bash
    pip install -r requirements.txt
3. **Jalankan Aplikasi**
   ```bash
   streamlit run app.py
4. **Unggah gambar / video**, lalu amati hasil deteksi kematangan secara langsung.
   ```

## 🧠 Kesimpulan

The automatic detection system for oil palm Fresh Fruit Bunch (FFB) ripeness based on YOLOv8 was successfully developed and implemented through a Streamlit web platform. The model was trained using 6,592 annotated images from Roboflow and achieved strong performance with a mAP@0.50 of 98%, precision of 94.5%, and recall of 94.7%.
The system performs an automatic grading process of FFB ripeness based on visual color features, providing fast, consistent, and accurate classification results. Its implementation enhances harvest efficiency, minimizes human subjectivity, and contributes to improved productivity in the palm oil industry.
Although the model performs well, further optimization of hyperparameter tuning and data augmentation techniques is recommended to enhance adaptability under real field conditions and various environmental factors.

---

## 👩‍💻 Pengembang

**Zahwa Genoveva**  
📚 Program Studi Informatika — Universitas Gunadarma  
🌐 [LinkedIn](https://www.linkedin.com/in/awagenovieve/) • [GitHub](https://github.com/Awaviviana09)

---

<p align="center">
  <i>“Empowering Smart Agriculture through Artificial Intelligence.”</i> 🌱  
</p>











