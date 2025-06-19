# Neural Networks dengan Keras: Membangun dan Melatih Jaringan Saraf Tiruan

*Notebook* ini membahas konsep dasar **Jaringan Saraf Tiruan (Neural Networks)** dan implementasinya menggunakan pustaka **Keras** di atas TensorFlow. Fokusnya adalah pada klasifikasi gambar dengan dataset MNIST, yang mencakup langkah-langkah mulai dari pra-pemrosesan data hingga pelatihan dan evaluasi model.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal melibatkan impor pustaka yang diperlukan seperti `tensorflow` dan `keras` (sebagai bagian dari TensorFlow 2.x), `numpy` untuk operasi numerik, dan `matplotlib.pyplot` untuk visualisasi.

Dataset yang digunakan untuk demonstrasi adalah **MNIST**, yang terdiri dari 70.000 gambar tulisan tangan angka (0-9).
* Data pelatihan (`X_train`, `y_train`) dan pengujian (`X_test`, `y_test`) dimuat langsung dari `keras.datasets.mnist`.
* **Normalisasi Fitur**: Nilai piksel gambar (yang awalnya berkisar dari 0 hingga 255) dinormalisasi ke rentang 0-1. Ini adalah langkah pra-pemrosesan standar untuk *Neural Networks* karena membantu algoritma optimasi bekerja lebih efisien.
    $$
    X_{\text{normalized}} = \frac{X_{\text{original}}}{255.0}
    $$
* **Pemisahan Data Validasi**: Sebagian kecil dari data pelatihan (misalnya, 5.000 sampel) dipisahkan sebagai set validasi. Data ini akan digunakan selama pelatihan untuk memantau kinerja model pada data yang belum terlihat dan mendeteksi *overfitting*.

---

## 2. Membangun Model Klasifikasi Gambar dengan Sequential API

Keras menyediakan **Sequential API** untuk membangun model *Neural Network* lapis demi lapis, yang sangat cocok untuk arsitektur *feedforward*.

* **Arsitektur Model**:
    1.  **Input Layer (`Flatten`)**: Lapisan pertama ini mengubah gambar 2D (28x28 piksel) menjadi vektor 1D (784 piksel). Ini tidak memiliki parameter yang dapat dilatih; ini hanya operasi *reshaping*.
        \[
        \text{Input Shape: } (28, 28) \rightarrow \text{Output Shape: } (784,)
        \]
    2.  **Hidden Layers (`Dense` dengan `relu`)**: Beberapa lapisan `Dense` (atau *fully connected*) ditambahkan. Setiap neuron di lapisan ini terhubung ke setiap neuron di lapisan sebelumnya.
        * `units`: Jumlah neuron di lapisan.
        * `activation="relu"`: **Rectified Linear Unit** (ReLU) adalah fungsi aktivasi populer yang membantu mengatasi masalah *vanishing gradients* dan memperkenalkan non-linearitas ke dalam model.
            \[
            \text{ReLU}(x) = \max(0, x)
            \]
    3.  **Output Layer (`Dense` dengan `softmax`)**: Lapisan terakhir memiliki satu neuron untuk setiap kelas (10 neuron untuk MNIST, karena ada 10 angka).
        * `activation="softmax"`: Fungsi aktivasi *softmax* digunakan untuk tugas klasifikasi *multiclass*. Ini mengonversi skor mentah (logits) dari setiap neuron output menjadi probabilitas yang berjumlah 1, menunjukkan keyakinan model untuk setiap kelas.
            \[
            \text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
            \]
            Di mana $z_j$ adalah skor (logit) untuk kelas $j$, dan $K$ adalah jumlah total kelas.

* **Melihat Ringkasan Model (`model.summary()`)**: Ini menampilkan informasi penting tentang model, termasuk jumlah lapisan, tipe setiap lapisan, bentuk output setiap lapisan, dan jumlah parameter yang dapat dilatih.

---

## 3. Kompilasi Model (Compile the Model)

Setelah arsitektur model didefinisikan, model perlu dikompilasi sebelum pelatihan.

* **`optimizer`**: Algoritma yang akan digunakan untuk menyesuaikan bobot model selama pelatihan. `Adam` adalah pilihan populer yang bekerja dengan baik dalam banyak kasus.
* **`loss`**: Fungsi kerugian (loss function) yang akan diukur selama pelatihan. Ini mengukur seberapa jauh prediksi model dari label sebenarnya.
    * `sparse_categorical_crossentropy`: Digunakan untuk klasifikasi *multiclass* ketika label target adalah integer (bukan *one-hot encoded*).
* **`metrics`**: Metrik yang akan dievaluasi selama pelatihan dan pengujian untuk memantau kinerja model. `accuracy` adalah metrik umum untuk klasifikasi.

---

## 4. Melatih Model (Train the Model)

Model dilatih menggunakan metode `fit()`.

* **`X_train`, `y_train`**: Data pelatihan.
* **`epochs`**: Jumlah iterasi pelatihan penuh pada seluruh set pelatihan.
* **`validation_data`**: Set validasi yang digunakan untuk memantau kinerja model pada data yang belum dilihat selama setiap *epoch*, membantu mendeteksi *overfitting*.
* **`history`**: Objek yang dikembalikan oleh `fit()` yang berisi riwayat nilai *loss* dan metrik lainnya selama pelatihan untuk setiap *epoch*. Ini dapat diplot untuk menganalisis kurva pembelajaran.

---

## 5. Mengevaluasi Model (Evaluate the Model)

Setelah pelatihan selesai, model dievaluasi pada set pengujian (`X_test`, `y_test`) untuk mendapatkan estimasi yang tidak bias tentang kinerja generalisasinya.

* **`model.evaluate()`**: Mengembalikan nilai *loss* dan metrik yang ditentukan selama kompilasi.

---

## 6. Menggunakan Model untuk Membuat Prediksi

Model yang sudah terlatih dapat digunakan untuk membuat prediksi pada instance baru.

* **`model.predict()`**: Mengembalikan probabilitas untuk setiap kelas.
* **`np.argmax()`**: Digunakan pada hasil `predict()` untuk mendapatkan indeks kelas dengan probabilitas tertinggi, yang merupakan prediksi kelas model.
* Visualisasi prediksi pada beberapa gambar pengujian menunjukkan kemampuan model dalam mengidentifikasi angka.

---

## 7. Membangun Model dengan Functional API (Alternatif)

Keras juga menawarkan **Functional API** untuk membangun model dengan arsitektur yang lebih kompleks, seperti:
* Model dengan input majemuk (multiple inputs).
* Model dengan output majemuk (multiple outputs).
* Model dengan cabang yang dapat digabungkan kembali (graph-like models).

Meskipun tidak digunakan secara mendalam dalam *notebook* ini untuk contoh MNIST, konsep dasarnya adalah mendefinisikan lapisan-lapisan dan kemudian menghubungkannya secara eksplisit, bukan hanya dalam urutan sequential.

---

## 8. Membangun dan Melatih Model Lebih Lanjut (Contoh Regresi)

*Notebook* ini juga menyertakan contoh sederhana untuk melatih *Neural Network* untuk tugas regresi, seperti memprediksi harga rumah dari dataset California.

* **Arsitektur Regresi**:
    * Lapisan output memiliki `units=1` (karena memprediksi satu nilai numerik) dan tidak ada fungsi aktivasi (`activation=None` atau dihilangkan) untuk prediksi regresi linier, atau fungsi aktivasi yang sesuai jika prediksi harus dalam rentang tertentu.
* **Fungsi Kerugian (Loss Function) untuk Regresi**:
    * `mse` (Mean Squared Error) adalah fungsi kerugian umum untuk regresi.
        $$
        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        $$
        Di mana $y_i$ adalah nilai sebenarnya dan $\hat{y}_i$ adalah prediksi.

---

**Kesimpulan:**

*Notebook* ini berfungsi sebagai panduan praktis untuk memulai dengan *Neural Networks* menggunakan Keras. Ini mencakup alur kerja penting dari persiapan data, pembangunan model (dengan Sequential API), kompilasi, pelatihan, evaluasi, hingga membuat prediksi. Konsep kunci seperti normalisasi, fungsi aktivasi (ReLU, Softmax), fungsi kerugian (Cross-entropy, MSE), dan optimisasi (Adam) dijelaskan dalam konteks implementasi praktis.
