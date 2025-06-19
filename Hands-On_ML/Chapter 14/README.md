# Deep Computer Vision dengan CNNs: Klasifikasi Gambar Lanjutan

*Notebook* ini membahas **Convolutional Neural Networks (CNNs)**, arsitektur *Neural Network* yang sangat dominan dalam bidang *Computer Vision*. Fokusnya adalah pada penerapan CNN untuk tugas klasifikasi gambar, dengan membahas berbagai jenis lapisan dan teknik penting yang membuat CNN begitu efektif.

---

## 1. Persiapan Lingkungan dan Dataset Umum

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `pandas`, `sklearn.preprocessing`, `matplotlib.pyplot`, dan `os`.

Dataset yang digunakan untuk demonstrasi adalah **CIFAR10**, yang terdiri dari 60.000 gambar berwarna (32x32 piksel) dari 10 kelas berbeda (misalnya, pesawat, mobil, burung, kucing, anjing, dll.).
* Data pelatihan (`X_train_full`, `y_train_full`) dan pengujian (`X_test`, `y_test`) dimuat dari `tf.keras.datasets.cifar10`.
* **Normalisasi Fitur**: Nilai piksel gambar (0-255) dinormalisasi ke rentang 0-1 untuk membantu pelatihan *Neural Network*.
  $$
  X_{\text{normalized}} = \frac{X_{\text{original}}}{255.0}
  $$
* **Pemisahan Data Validasi**: Sebagian dari set pelatihan dipisahkan sebagai set validasi (`X_valid`, `y_valid`) untuk memantau kinerja model selama pelatihan.

---

## 2. Arsitektur CNN Dasar

CNN umumnya terdiri dari beberapa jenis lapisan yang disusun secara berurutan:

### A. Lapisan Konvolusional (Convolutional Layers - `Conv2D`)

* **Konsep**: Lapisan konvolusional adalah blok bangunan utama CNN. Mereka bekerja dengan meluncurkan (convolving) sebuah *filter* (atau *kernel*) kecil di atas gambar input untuk mendeteksi pola lokal. Setiap *filter* mendeteksi jenis fitur yang berbeda (misalnya, tepi, tekstur).
* **Operasi Konvolusi**: Filter bergeser melintasi gambar, dan pada setiap posisi, ia melakukan produk titik (dot product) antara piksel di bawah filter dan nilai-nilai filter.
    $$
    (I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) K(m, n)
    $$
    Di mana $I$ adalah gambar input, $K$ adalah filter (kernel), dan $(i, j)$ adalah posisi piksel output.
* **Parameter Kunci**:
    * `filters`: Jumlah filter yang akan digunakan. Setiap filter menghasilkan sebuah *feature map*.
    * `kernel_size`: Dimensi filter (misalnya, `(3, 3)` untuk filter 3x3).
    * `activation`: Fungsi aktivasi (misalnya, `relu`) yang diterapkan setelah operasi konvolusi.
    * `padding`:
        * `"valid"`: Tidak ada padding. Ukuran output akan lebih kecil dari input.
        * `"same"`: Menambahkan padding nol di sekitar input sehingga ukuran output *feature map* sama dengan input (jika `strides=1`).
    * `strides`: Jumlah langkah filter bergeser. `strides=(2, 2)` akan mengurangi dimensi spasial *feature map*.

### B. Lapisan Pooling (`MaxPooling2D`)

* **Konsep**: Lapisan *pooling* mengurangi dimensi spasial dari *feature map* (misalnya, lebar dan tinggi) tanpa mengurangi jumlah *feature map* (kedalaman). Ini membantu mengurangi jumlah parameter dan komputasi, serta membuat model lebih *robust* terhadap pergeseran kecil dalam input.
* **Max Pooling**: Mengambil nilai maksimum dari setiap jendela kecil di *feature map*.
* **Parameter Kunci**:
    * `pool_size`: Dimensi jendela pooling (misalnya, `(2, 2)`).
    * `strides`: Seberapa jauh jendela pooling bergeser. Biasanya sama dengan `pool_size`.

### C. Lapisan Flatten (`Flatten`)

* **Konsep**: Setelah beberapa lapisan konvolusional dan *pooling*, *feature map* 2D perlu diubah menjadi vektor 1D untuk dapat diumpankan ke lapisan *dense* tradisional.
* **Implementasi**: Lapisan `Flatten()` melakukan *reshaping* ini secara otomatis.

### D. Lapisan Dense (`Dense`)

* **Konsep**: Lapisan *fully connected* yang biasa digunakan di *Neural Network* non-konvolusional.
* **Parameter Kunci**:
    * `units`: Jumlah neuron di lapisan.
    * `activation`: Fungsi aktivasi (misalnya, `relu` untuk *hidden layers*, `softmax` untuk *output layer* klasifikasi *multiclass*).

### E. Lapisan Dropout (`Dropout`)

* **Konsep**: Teknik regularisasi yang menonaktifkan sebagian neuron secara acak selama pelatihan untuk mencegah *overfitting*.
* **Parameter `rate`**: Probabilitas neuron untuk di-*drop* (dinonaktifkan).

---

## 3. Membangun dan Melatih CNN untuk CIFAR10

*Notebook* ini membangun sebuah model CNN sederhana untuk klasifikasi CIFAR10:
1.  **Input Layer**: Menerima gambar dengan bentuk `(32, 32, 3)` (lebar, tinggi, saluran warna).
2.  **Blok Konvolusional-Pooling**: Beberapa pasang lapisan `Conv2D` diikuti oleh `MaxPooling2D`.
    * Lapisan `Conv2D` pertama seringkali memiliki jumlah filter yang lebih kecil dan kemudian meningkat di lapisan-lapisan berikutnya, sementara dimensi spasial (`MaxPooling2D`) menurun.
    * `activation="relu"` umumnya digunakan untuk lapisan konvolusional dan *hidden dense*.
3.  **Lapisan Flatten**: Mengubah output dari blok konvolusional-pooling terakhir menjadi vektor 1D.
4.  **Lapisan Dense (Hidden)**: Satu atau lebih lapisan *fully connected* (misalnya, `Dense(100, activation="relu")`).
5.  **Lapisan Output**: `Dense(10, activation="softmax")` untuk 10 kelas klasifikasi.

* **Kompilasi Model**: Model dikompilasi dengan *optimizer* (misalnya, `Adam`), fungsi *loss* (`sparse_categorical_crossentropy` untuk label integer), dan *metrics* (`accuracy`).
* **Pelatihan Model**: Model dilatih menggunakan `model.fit()` pada data pelatihan, dengan memantau kinerja pada set validasi.
* **Evaluasi Model**: Kinerja akhir model dievaluasi pada set pengujian (`model.evaluate()`).

---

## 4. Transfer Learning dengan CNNs

*Transfer Learning* adalah strategi yang sangat efektif untuk tugas *Computer Vision*. Daripada melatih CNN dari awal, kita dapat menggunakan model *pre-trained* (yang sudah dilatih pada dataset sangat besar seperti ImageNet) sebagai *feature extractor*.

* **Konsep**: Lapisan konvolusional awal dari model *pre-trained* (yang mempelajari fitur-fitur umum seperti tepi, tekstur) dibekukan (bobotnya tidak diperbarui). Lapisan *output* baru ditambahkan dan dilatih pada data tugas yang baru.
* **Keunggulan**: Mempercepat pelatihan secara drastis, membutuhkan data pelatihan yang jauh lebih sedikit, dan seringkali menghasilkan kinerja yang lebih baik daripada melatih model dari awal.
* **Implementasi**:
    * Menggunakan model dari `tf.keras.applications` (misalnya, `ResNet50V2`, `InceptionV3`, `Xception`).
    * Memuat model *pre-trained* tanpa lapisan *top* (lapisan *output* akhir).
    * Membekukan lapisan dasar model *pre-trained* (`base_model.trainable = False`).
    * Menambahkan lapisan `GlobalAveragePooling2D` (merata-ratakan *feature map*) dan lapisan `Dense` baru untuk klasifikasi tugas spesifik Anda.
    * Melatih model.
* **Fine-tuning**: Setelah melatih lapisan baru, Anda bisa secara opsional "mencairkan" beberapa lapisan teratas dari model *pre-trained* dan melatihnya dengan *learning rate* yang sangat kecil. Ini memungkinkan model untuk menyesuaikan fitur yang lebih spesifik untuk tugas baru Anda.
