# Pelatihan dan Deployment Model Machine Learning Skala Besar

*Notebook* ini membahas tantangan dan solusi terkait **pelatihan dan *deployment* model *Machine Learning* pada skala besar**, yang meliputi penggunaan TensorFlow untuk distribusi pelatihan dan berbagai metode untuk *serving* model.

---

## 1. Persiapan Lingkungan dan Dataset Umum

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `pandas`, serta `matplotlib.pyplot` untuk operasi numerik dan visualisasi.

Dataset yang digunakan untuk demonstrasi adalah **Fashion MNIST**, yang terdiri dari 70.000 gambar item pakaian (10 kelas).
* Data pelatihan (`X_train_full`, `y_train_full`) dan pengujian (`X_test`, `y_test`) dimuat dari `tf.keras.datasets.fashion_mnist`.
* **Normalisasi Fitur**: Nilai piksel gambar (0-255) dinormalisasi ke rentang 0-1 untuk membantu pelatihan Neural Network.
    $$
    X_{\text{normalized}} = \frac{X_{\text{original}}}{255.0}
    $$
* **Pemisahan Data Validasi**: Sebagian dari set pelatihan dipisahkan sebagai set validasi (`X_valid`, `y_valid`) untuk memantau kinerja model selama pelatihan.
* **Transformasi Label**: Label di-*one-hot encode* untuk beberapa konfigurasi model.

---

## 2. Pelatihan Terdistribusi (Distributed Training)

Melatih model *Deep Learning* yang besar pada dataset yang sangat besar bisa sangat memakan waktu. **Pelatihan terdistribusi** memungkinkan Anda untuk mendistribusikan beban kerja pelatihan ke beberapa perangkat (GPU) atau beberapa mesin (CPU/GPU) untuk mempercepat prosesnya. TensorFlow menyediakan Strategy API untuk tujuan ini.

### A. Strategy API (tf.distribute.Strategy)

`tf.distribute.Strategy` adalah API yang fleksibel untuk distribusi pelatihan. Ini memungkinkan Anda untuk mengubah model dan kode pelatihan yang ada untuk berjalan pada beberapa *device* atau beberapa mesin dengan perubahan kode minimal.

* **`MirroredStrategy`**:
    * Paling umum digunakan untuk pelatihan terdistribusi **pada satu mesin dengan beberapa GPU**.
    * Setiap GPU memiliki salinan lengkap dari model (mirror).
    * Setiap GPU memproses subset berbeda dari *mini-batch* data.
    * Gradien dihitung secara independen di setiap GPU, lalu dijumlahkan (*all-reduce*) dan diterapkan ke bobot model di setiap GPU, menjaga bobot tetap sinkron.
    * Ideal untuk memanfaatkan beberapa GPU di satu server.

* **`CentralStorageStrategy`**:
    * Juga digunakan untuk pelatihan terdistribusi pada satu mesin, tetapi **bobot model disimpan di CPU**, dan operasi dikirim ke GPU.
    * Kurang *scalable* dibandingkan `MirroredStrategy` karena bottleneck CPU, tetapi mungkin berguna dalam skenario tertentu.

* **`MultiWorkerMirroredStrategy`**:
    * Digunakan untuk pelatihan terdistribusi **pada beberapa mesin (multiple workers)**.
    * Setiap mesin (worker) memiliki beberapa GPU, dan `MirroredStrategy` diterapkan di dalam setiap worker.
    * Gradien dijumlahkan di antara semua GPU di semua worker.
    * Membutuhkan konfigurasi lingkungan `TF_CONFIG` untuk mengidentifikasi peran setiap worker (master, worker, chief, dll.).

* **`TPUStrategy`**:
    * Digunakan untuk pelatihan pada Google Cloud **TPUs** (Tensor Processing Units). TPUs adalah akselerator khusus yang dioptimalkan untuk *Deep Learning*.

### B. Konsep Kunci dalam Pelatihan Terdistribusi

* **Sinkronisasi (Synchronous Training)**: Semua *device* atau *worker* melatih pada saat yang sama, dan gradien dijumlahkan serta bobot diperbarui secara sinkron. Ini memastikan konsistensi model.
* **Batch Size Global**: Total *batch size* dibagi di antara semua *device*. Misalnya, jika *global batch size* adalah 64 dan ada 4 GPU, setiap GPU akan memproses *mini-batch* 16.
* **Fungsi `create_model()`**: Disarankan untuk membungkus pembuatan model dalam sebuah fungsi, yang kemudian diteruskan ke `strategy.scope()`.

---

## 3. TensorFlow Serving: Deployment Model

Setelah model dilatih, langkah selanjutnya adalah menyebarkannya untuk inferensi pada skala produksi.

### A. Menyimpan Model dalam Format SavedModel

* TensorFlow menggunakan format **SavedModel** untuk menyimpan model yang sudah dilatih. Format ini mencakup arsitektur model, bobot, dan logika *inference* (grafik komputasi) dalam satu paket.
* `model.save(filepath, save_format='tf')`: Menyimpan model ke direktori yang ditentukan. Setiap versi model harus disimpan dalam sub-direktori bernomor (misalnya, `1/`, `2/`) untuk memungkinkan *versioning* dan *model rollback*.

### B. Menggunakan TensorFlow Serving

TensorFlow Serving adalah sistem fleksibel dan berkinerja tinggi untuk *serving* model ML dalam produksi.

* **Keunggulan**:
    * ***Hot Swapping***: Dapat memuat versi model baru secara otomatis saat tersedia, tanpa perlu menghentikan server.
    * ***Model Rollback***: Mudah untuk kembali ke versi model sebelumnya jika ada masalah.
    * ***Batching Otomatis***: Mampu mengumpulkan permintaan inferensi yang masuk ke dalam *batch* untuk efisiensi GPU/TPU yang lebih baik.
    * **API Standar**: Menawarkan API gRPC dan RESTful untuk permintaan inferensi.

* **Langkah-langkah Khas untuk *Deployment* (Konseptual)**:
    1.  **Ekspor Model**: Simpan model dalam format SavedModel di direktori yang sesuai (misalnya, `model_name/version_number/`).
    2.  **Jalankan TensorFlow Serving**: Jalankan kontainer Docker TensorFlow Serving, menunjuk ke direktori model Anda.
    3.  **Kirim Permintaan Inferensi**: Kirim permintaan HTTP POST (untuk RESTful API) atau gRPC ke server TensorFlow Serving dengan data input.

### C. Metode Deployment Lainnya (Konseptual)

* **TensorFlow Lite**: Untuk *deployment* di perangkat *mobile* dan *edge devices* (misalnya, Android, iOS, Raspberry Pi). Mengkompres model dan mengoptimalkannya untuk inferensi berlatensi rendah.
* **TensorFlow.js**: Untuk *deployment* di *browser* atau di Node.js.
* **Integrasi Platform Cloud**: Layanan seperti Google Cloud AI Platform, AWS SageMaker, Azure Machine Learning menyediakan layanan terkelola untuk pelatihan dan *deployment* skala besar, seringkali memanfaatkan TensorFlow Serving di baliknya.

---
