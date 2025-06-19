# Memuat dan Pra-pemrosesan Data dengan TensorFlow: TFRecords dan TFTransform

*Notebook* ini membahas berbagai teknik lanjutan untuk memuat dan pra-pemrosesan data dalam *pipeline* *machine learning* menggunakan TensorFlow, dengan fokus pada dataset besar dan efisiensi. Topik utama mencakup penggunaan **TFRecords** untuk format data yang efisien, **Dataset API** untuk membangun *pipeline* input, serta konsep **TFTransform** untuk transformasi data yang konsisten antara pelatihan dan inferensi.

---

## 1. Persiapan Lingkungan dan Dataset Umum

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `pandas`, `sklearn.preprocessing`, `matplotlib.pyplot`, dan `os`. Konfigurasi *plotting* juga diatur untuk konsistensi visual.

Dataset yang digunakan untuk demonstrasi adalah **Fashion MNIST** (untuk klasifikasi gambar) dan **California Housing** (untuk regresi).
* **Fashion MNIST**: Dimuat dari `tf.keras.datasets.fashion_mnist`. Gambar dinormalisasi ke rentang 0-1 dan dataset dipisahkan menjadi set pelatihan, validasi, dan pengujian.
* **California Housing**: Diperoleh dari UCI ML Repository. Data numerik (harga rumah) diskalakan, dan fitur kategorikal (`ocean_proximity`) di-*one-hot encode*. Dataset ini dibagi menjadi pelatihan, validasi, dan pengujian.

---

## 2. Dataset API: Membangun Pipeline Input Efisien

**Dataset API (`tf.data`)** adalah cara yang direkomendasikan di TensorFlow untuk membangun *pipeline* input yang *scalable* dan efisien.

* **Membuat Dataset**:
    * `tf.data.Dataset.from_tensor_slices(tensors)`: Membuat dataset dari tensor NumPy atau TensorFlow yang ada. Ini cocok untuk dataset yang pas di memori.
* **Transformasi Dataset**: Dataset API menyediakan berbagai metode untuk transformasi data yang sangat efisien:
    * `repeat()`: Mengulang dataset beberapa *epoch*.
    * `shuffle(buffer_size)`: Mengacak data dalam *buffer*.
    * `batch(batch_size)`: Mengelompokkan instance menjadi *mini-batch*.
    * `map(function)`: Menerapkan fungsi transformasi ke setiap elemen dataset.
        * `num_parallel_calls=tf.data.AUTOTUNE`: Mengatur jumlah pemanggilan paralel secara otomatis untuk optimasi kinerja.
    * `prefetch(buffer_size)`: Memuat *batch* berikutnya ke memori saat *batch* saat ini sedang diproses. Ini sangat meningkatkan efisiensi I/O dan mencegah *bottleneck*.
        * `tf.data.AUTOTUNE`: TensorFlow secara otomatis akan menyesuaikan ukuran *buffer* *prefetch*.
* **Melatih Model dengan Dataset API**: Model Keras dapat dilatih langsung menggunakan objek `tf.data.Dataset` yang sudah dibuat.

---

## 3. TFRecords: Format Data yang Efisien

**TFRecords** adalah format biner bervolume rendah yang direkomendasikan oleh TensorFlow untuk menyimpan data. Ini sangat efisien untuk memuat data besar karena:
* **Ukuran File Kecil**: Format biner lebih ringkas daripada teks (misalnya CSV).
* **Akses Cepat**: Dirancang untuk I/O paralel dan *streamed*.
* **Portabilitas**: Dapat digunakan di berbagai lingkungan TensorFlow.

* **Menyimpan Data ke TFRecords**:
    * Data di-*encode* sebagai **`tf.train.Example`** protocol buffers. Setiap `Example` berisi kamus `features`, di mana setiap *feature* adalah `tf.train.Feature` yang dapat menyimpan `bytes_list`, `float_list`, atau `int64_list`.
    * `tf.io.TFRecordWriter`: Digunakan untuk menulis `Example` yang telah diserialisasi ke file `.tfrecord`.
* **Membaca Data dari TFRecords**:
    * `tf.data.TFRecordDataset(filenames)`: Membuat dataset dari satu atau lebih file TFRecord.
    * **Parsing `tf.train.Example`**: Karena `Example` disimpan dalam format biner, Anda perlu mendefinisikan *schema* (struktur) untuk mendekode fitur-fitur tersebut. `tf.io.FixedLenFeature`, `tf.io.VarLenFeature`, dan `tf.io.RaggedFeature` digunakan untuk mendefinisikan tipe dan bentuk fitur yang diharapkan.
    * `tf.io.parse_single_example(serialized_example, feature_description)`: Digunakan dalam operasi `map()` pada dataset untuk mendekode setiap `Example` menjadi tensor yang dapat digunakan.

---

## 4. Pra-pemrosesan Data Skala Besar dengan TFTransform (Konseptual)

**TFTransform (`tf.Transform`)** adalah pustaka yang dirancang untuk transformasi data *end-to-end* yang konsisten antara fase pelatihan dan inferensi, terutama untuk dataset skala besar.

* **Masalah Inkonsistensi**: Ketika Anda melakukan transformasi data (misalnya, penskalaan standar atau normalisasi) pada set pelatihan, Anda perlu memastikan bahwa transformasi yang *sama* persis diterapkan pada data inferensi (data baru yang belum terlihat). Jika tidak, model akan membuat prediksi yang salah.
* **Bagaimana TFTransform Membantu**:
    1.  **Analisis Data**: `tf.Transform` akan melakukan "analisis" pada set pelatihan untuk menghitung statistik yang diperlukan untuk transformasi (misalnya, *mean* dan *standard deviation* untuk standardisasi, kosa kata untuk *one-hot encoding*).
    2.  **Transformasi Data**: Statistik ini kemudian digunakan untuk menerapkan transformasi yang *sama* pada set pelatihan dan secara otomatis menghasilkan kode transformasi yang dapat dieksekusi selama inferensi.
    3.  **Konsistensi Terjamin**: Ini menjamin bahwa data yang masuk ke model selama pelatihan memiliki distribusi yang sama dengan data yang masuk ke model selama inferensi, mencegah *training-serving skew*.
* **Use Case**: Sangat berguna dalam *production pipeline* di mana konsistensi transformasi sangat penting dan dataset bisa sangat besar (memerlukan Apache Beam untuk pemrosesan paralel
