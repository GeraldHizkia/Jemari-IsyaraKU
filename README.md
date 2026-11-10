# Jemari-IsyaraKU — Jemari BISINDO Translator 2.0 (Kosakata & Alfabet)

> **Untuk membantu teman Tuli berkomunikasi lebih cepat, alami, dan inklusif.**

Sistem pengenal Bahasa Isyarat Indonesia (BISINDO) real-time berbasis *Computer Vision* yang menggabungkan:
1. **Mode Kosakata (Word-Level Recognition)**: Mengenali 32 kosakata BISINDO dari dataset penelitian **WL-BISINDO** menggunakan sekuens landmark temporal dan model *Bi-LSTM Attention* / *SPOTER*.
2. **Mode Alfabet (Alphabet Recognition)**: Pengenal 26 abjad isyarat (A–Z) berbasis model *Multi-Layer Perceptron (MLP)* langsung di browser.
3. **Penyusun Kalimat & Suara**: Mengakumulasi gestur menjadi teks bahasa Indonesia dan mengubahnya menjadi suara via Text-to-Speech (TTS).

---

## 📁 Struktur Direktori

```text
jemari-bisindo/
├── backend/
│   ├── app.py                # Server FastAPI & endpoint inferensi
│   ├── word_model.py         # Arsitektur Bi-LSTM Attention & Transformer
│   └── inference.py          # Mesin inferensi temporal real-time
├── pipeline/
│   ├── labels.json           # Definisi 32 kelas kosakata BISINDO
│   ├── organize_wlbisindo.py # Script pengorganisir dataset (fixed 32 kelas)
│   ├── extract_landmarks.py  # Ekstraksi koordinat MediaPipe dari video
│   └── train_word_model.py   # Skrip pelatihan model PyTorch
├── models/
│   ├── alphabet/
│   │   └── model.json        # Bobot model alfabet Jemari
│   └── word/
│       ├── best_model.pt     # Checkpoint bobot kosakata terbaik
│       ├── labels.json       # Metadata 32 label kata
│       └── model_meta.json   # Konfigurasi model
├── web/
│   ├── index.html            # Antarmuka web utama
│   ├── style.css             # Desain antarmuka modern
│   ├── app.js                # Orkestrator aplikasi
│   ├── hands.js              # MediaPipe Multi-Hands (2 tangan)
│   ├── alphabet.js           # Pengenal alfabet client-side
│   ├── wordRecognition.js    # Pengenal kata temporal & client API
│   └── tts.js                # Modul Text-to-Speech Bahasa Indonesia
└── README.md
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Jalankan Backend Server
Di terminal PowerShell:
```powershell
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Buka di Browser
Akses:
👉 **http://127.0.0.1:8000**

- Klik **▶ Mulai Kamera** dan berikan izin webcam.
- Pilih mode yang diinginkan:
  - **Mode Kosakata**: Lakukan gerakan isyarat (misal "Terima kasih", "Makan", "Saya", "Air"). Anda dapat mengklik tombol **⚡ Tangkap Gestur Kata** atau membiarkan sistem mendeteksi saat tangan selesai bergerak.
  - **Mode Alfabet**: Tunjukkan pose satu tangan untuk mengeja huruf A–Z.
- Klik **🔊 Suarakan (TTS)** untuk mendengarkan hasil kalimat yang sudah dirangkai.

---

## 📊 32 Kelas Kosakata WL-BISINDO

| Label | Kata | Kategori | Label | Kata | Kategori |
|---:|---|---|---:|---|---|
| 0 | Air | Kebutuhan/Benda | 16 | Mengapa | Kata Tanya |
| 1 | Belajar | Aktivitas | 17 | Bagaimana | Kata Tanya |
| 2 | Cari | Aktivitas | 18 | Merah | Warna |
| 3 | Hari | Waktu | 19 | Kuning | Warna |
| 4 | Ingat | Kognisi | 20 | Hijau | Warna |
| 5 | Lagi | Keterangan | 21 | Hitam | Warna |
| 6 | Maaf | Etika/Sosial | 22 | Dengar | Sensori |
| 7 | Makan | Aktivitas | 23 | Berangkat | Aktivitas |
| 8 | Motor | Transportasi | 24 | Datang | Aktivitas |
| 9 | Saya | Kata Ganti | 25 | Teman | Sosial/Relasi |
| 10 | Terima kasih | Etika/Sosial | 26 | Keluarga | Sosial/Relasi |
| 11 | Tuli | Identitas/Komunitas | 27 | Rumah | Tempat |
| 12 | Apa | Kata Tanya | 28 | Pagi | Waktu |
| 13 | Siapa | Kata Tanya | 29 | Siang | Waktu |
| 14 | Kapan | Kata Tanya | 30 | Sore | Waktu |
| 15 | Di mana | Kata Tanya | 31 | Malam | Waktu |

---

## 🛠️ Melatih Model dengan Dataset Lengkap WL-BISINDO

1. **Unduh Video dari Kaggle:**
   ```bash
   kaggle datasets download -d glennleonali/wl-bisindo -p dataset_raw --unzip
   ```
2. **Organisasi Video (32 Kelas Dinamis):**
   ```bash
   python pipeline/organize_wlbisindo.py --metadata SD_split_metadata.json --source dataset_raw --output dataset/SD
   ```
3. **Ekstraksi Landmark Spatio-Temporal:**
   ```bash
   python pipeline/extract_landmarks.py --dataset_dir dataset/SD --output wl_bisindo_features.npz
   ```
4. **Training Model Bi-LSTM / Transformer:**
   ```bash
   python pipeline/train_word_model.py --data wl_bisindo_features.npz --model_type bilstm --epochs 40 --output models/word
   ```

---

## 📝 Lisensi & Atribusi
- **Jemari**: MIT License (Alvnvnc - [bisindo-translator](https://github.com/Alvnvnc/bisindo-translator))
- **WL-BISINDO**: CC BY-NC 4.0 (AceKinnn / Glenn Leonali dkk. - [WL-BISINDO](https://github.com/AceKinnn/WL-BISINDO))
- **SPOTER**: Apache 2.0 / MIT (Matyas Bohacek - [spoter](https://github.com/matyasbohacek/spoter))
