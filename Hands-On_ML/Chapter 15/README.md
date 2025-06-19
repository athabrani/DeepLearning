# Memproses Urutan (Sequences) menggunakan RNNs dan CNNs

*Notebook* ini membahas berbagai pendekatan untuk memproses data urutan (seperti deret waktu atau teks) menggunakan **Recurrent Neural Networks (RNNs)** dan juga mengadaptasi **Convolutional Neural Networks (CNNs)** untuk tugas-tugas ini. Fokus utamanya adalah pada memprediksi nilai berikutnya dalam deret waktu dan menangani berbagai jenis lapisan RNN.

---

## 1. Persiapan Lingkungan dan Dataset Umum

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `pandas`, `sklearn.preprocessing`, `matplotlib.pyplot`, dan `os`.

Dataset yang digunakan untuk demonstrasi adalah **deret waktu sintetis** yang dihasilkan menggunakan fungsi `generate_time_series()`.
* Deret waktu ini memiliki pola musiman, tren, dan *noise*.
* Data dibagi menjadi set pelatihan, validasi, dan pengujian.
* Tujuan: Memprediksi nilai berikutnya dalam deret waktu berdasarkan beberapa nilai sebelumnya. Formatnya adalah `(batch_size, n_steps, n_inputs)`, di mana `n_steps` adalah jumlah *timesteps* input, dan `n_inputs` adalah jumlah fitur per *timestep*.

---

## 2. Memprediksi Urutan (Predicting Sequences)

### 2.1. Model Baseline (Dasar)

* **Baseline Naif (Naive Baseline)**: Memprediksi nilai berikutnya sebagai nilai saat ini. Ini adalah titik referensi yang sangat sederhana untuk mengevaluasi seberapa baik model lain bekerja. Metrik `MSE` (Mean Squared Error) digunakan untuk evaluasi.

### 2.2. Regresi Linier untuk Deret Waktu

* **Model**: `Dense` layer tunggal tanpa fungsi aktivasi dapat digunakan sebagai model regresi linier sederhana.
* **Input Shape**: Model ini membutuhkan input 2D (`(batch_size, n_steps * n_inputs)`). Jika inputnya 3D (`(batch_size, n_steps, n_inputs)`), `Flatten` layer diperlukan sebelumnya.

---

## 3. Recurrent Neural Networks (RNNs)

RNNs dirancang khusus untuk memproses data urutan dengan memiliki koneksi siklik yang memungkinkan informasi dipertahankan dari satu *timestep* ke *timestep* berikutnya.

### 3.1. Simple RNN

* `tf.keras.layers.SimpleRNN`: Implementasi dasar dari lapisan RNN.
* **Input Shape**: Menerima input 3D (`(batch_size, n_steps, n_inputs)`).
* **Output**:
    * Jika `return_sequences=False` (default): Output adalah vektor 2D `(batch_size, units)` yang merepresentasikan *hidden state* terakhir. Ini cocok untuk prediksi *many-to-one* (misalnya, klasifikasi sentimen dari urutan teks).
    * Jika `return_sequences=True`: Output adalah urutan 3D `(batch_size, n_steps, units)` yang merepresentasikan *hidden state* di setiap *timestep*. Ini cocok untuk prediksi *many-to-many* atau *sequence-to-sequence*.

### 3.2. Memprediksi Satu Nilai Berikutnya (`SimpleRNN` Many-to-One)

* Model menggunakan satu lapisan `SimpleRNN` diikuti oleh satu lapisan `Dense` sebagai lapisan output.
* Lapisan `SimpleRNN` akan mengembalikan *hidden state* terakhir, yang kemudian diumpankan ke lapisan `Dense` untuk prediksi nilai tunggal.

### 3.3. Memprediksi Urutan (Many-to-Many Output)

Jika Anda perlu memprediksi urutan nilai, bukan hanya nilai tunggal, arsitektur model perlu diadaptasi:

* **Untuk Setiap Timestep**: Jika lapisan RNN mengembalikan urutan (`return_sequences=True`), Anda dapat melampirkan lapisan `Dense` ke setiap *timestep* output untuk menghasilkan urutan prediksi.
    * `TimeDistributed(Dense(...))`: Ini adalah *wrapper* yang menerapkan lapisan `Dense` secara independen ke setiap *timestep* input. Lapisan ini dibutuhkan jika Anda ingin lapisan `Dense` Anda memiliki bobot yang berbeda untuk setiap *timestep*.
    * Contoh: `model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))`
* **Pada Praktiknya**: Seringkali, jika lapisan `Dense` mengikuti lapisan RNN yang memiliki `return_sequences=True`, Keras secara otomatis akan menerapkan lapisan `Dense` ke setiap *timestep* output jika lapisan `Dense` tersebut tidak memiliki `TimeDistributed` wrapper. Namun, `TimeDistributed` diperlukan jika Anda ingin memastikan bahwa lapisan `Dense` memperlakukan setiap *timestep* secara terpisah.

---

## 4. Lapisan RNN Populer Lainnya

RNN sederhana memiliki masalah *vanishing gradients* pada urutan panjang. Layer-layer ini mengatasi masalah tersebut:

### 4.1. Long Short-Term Memory (LSTM)

* **Konsep**: LSTM adalah jenis RNN yang dirancang khusus untuk mempelajari dependensi jangka panjang. Ia memiliki "sel memori" dan "gerbang" (input, forget, output gate) yang mengontrol aliran informasi ke dalam dan keluar sel, memungkinkan informasi penting dipertahankan lebih lama.
* **Gerbang-gerbang LSTM**:
    * **Forget Gate**: Menentukan informasi apa dari *cell state* sebelumnya yang harus "dilupakan".
    * **Input Gate**: Menentukan informasi baru apa dari input saat ini yang harus "diingat".
    * **Output Gate**: Mengontrol informasi apa dari *cell state* yang akan diekspos sebagai *hidden state* (output) pada *timestep* saat ini.
* **Implementasi**: `tf.keras.layers.LSTM`.

### 4.2. Gated Recurrent Unit (GRU)

* **Konsep**: GRU adalah versi yang lebih sederhana dari LSTM, dengan jumlah gerbang yang lebih sedikit (reset gate dan update gate). Ini seringkali memiliki kinerja yang mirip dengan LSTM tetapi dengan komputasi yang lebih ringan.
* **Implementasi**: `tf.keras.layers.GRU`.

---

## 5. Time-Series Forecasting dengan CNNs (WaveNet Architecture)

Meskipun RNN secara tradisional digunakan untuk urutan, CNN juga dapat beradaptasi. Untuk deret waktu, CNN biasanya menggunakan **Convolutional Layers 1D (`Conv1D`)**.

* **Konsep**: `Conv1D` meluncurkan filter di sepanjang satu dimensi (dimensi waktu) dari urutan. Ini efektif dalam menangkap pola lokal dalam urutan (mirip dengan bagaimana `Conv2D` menangkap pola lokal dalam gambar).
* **Dilated Convolutions (`dilation_rate`)**:
    * `Conv1D` dengan `dilation_rate > 1` (juga dikenal sebagai *atrous convolution*) memungkinkan filter memiliki "lubang" di dalamnya, sehingga dapat mencakup area input yang lebih luas tanpa meningkatkan jumlah parameter.
    * Ini sangat berguna untuk urutan panjang karena memungkinkan bidang reseptif yang besar dengan sedikit lapisan, mengatasi masalah *vanishing gradients* yang mungkin terjadi pada RNN yang sangat dalam.
* **WaveNet**: Arsitektur yang populer menggunakan *dilated convolutions* untuk pemodelan deret waktu dan audio generatif.

---

## 6. Menggabungkan RNN dan CNN

Dimungkinkan juga untuk menggabungkan lapisan `Conv1D` (untuk mengekstrak fitur lokal dari urutan) dengan lapisan RNN (untuk memproses dependensi jangka panjang dari fitur yang diekstrak) dalam satu model. Ini seringkali menghasilkan model yang lebih kuat.

---
