# Model Kustom dan Pelatihan dengan TensorFlow: Fleksibilitas Tingkat Lanjut

*Notebook* ini membahas cara membangun model kustom dan mengelola alur pelatihan menggunakan TensorFlow secara lebih fleksibel, melampaui `Sequential` dan `Functional API` yang standar. Fokus utamanya adalah pada pembuatan lapisan kustom, model kustom, fungsi *loss* kustom, metrik kustom, serta pelatihan dengan `tf.GradientTape` dan `tf.function`.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` dan `keras`, `numpy`, `pandas`, serta `matplotlib.pyplot` untuk operasi numerik dan visualisasi.

Dataset yang digunakan untuk demonstrasi adalah **Fashion MNIST**, yang terdiri dari 70.000 gambar item pakaian (10 kelas).
* Data pelatihan (`X_train_full`, `y_train_full`) dan pengujian (`X_test`, `y_test`) dimuat dari `tf.keras.datasets.fashion_mnist`.
* **Normalisasi Fitur**: Nilai piksel gambar (0-255) dinormalisasi ke rentang 0-1 untuk membantu pelatihan Neural Network.
  $$
  X_{\text{normalized}} = \frac{X_{\text{original}}}{255.0}
  $$
* **Pemisahan Data Validasi**: Sebagian dari set pelatihan (misalnya, 5.000 sampel) dipisahkan sebagai set validasi (`X_valid`, `y_valid`) untuk memantau kinerja model selama pelatihan dan mendeteksi *overfitting*.

---

## 2. Lapisan Kustom (Custom Layers)

Keras memungkinkan Anda untuk membuat lapisan kustom dengan mewarisi kelas `tf.keras.layers.Layer`. Ini memberikan fleksibilitas penuh untuk mendefinisikan perilaku lapisan, termasuk bobot yang dapat dilatih dan operasi *forward pass*.

* **Metode Kunci**:
    * `__init__(self, ...)`: Metode konstruktor untuk menginisialisasi atribut lapisan, termasuk *hyperparameter*.
    * `build(self, input_shape)`: Metode ini dipanggil pertama kali lapisan digunakan. Di sinilah Anda dapat membuat variabel (bobot, *bias*) yang bergantung pada bentuk *input*.
        * Gunakan `self.add_weight()` untuk membuat variabel yang dapat dilatih.
    * `call(self, inputs)`: Metode ini mendefinisikan operasi *forward pass* lapisan. Di sinilah Anda menulis logika perhitungan output lapisan berdasarkan input.
    * `compute_output_shape(self, input_shape)`: (Opsional, tapi baik untuk didefinisikan) Digunakan untuk menentukan bentuk output lapisan, berguna untuk debugging dan validasi model.
* **Contoh Lapisan Kustom**: *Notebook* ini mendemonstrasikan `Dense` Layer kustom dan `Softmax` Layer kustom. Anda juga dapat membuat lapisan yang memiliki koneksi yang berbeda, lapisan yang menggunakan *sparse tensors*, atau lapisan yang melakukan operasi kompleks.

---

## 3. Model Kustom (Custom Models)

Untuk membuat arsitektur model yang lebih kompleks daripada yang dapat dibangun dengan `Sequential` atau `Functional API` (misalnya, model dengan *loop* internal, *skip connections* non-standar, atau *multiple inputs/outputs* yang kompleks), Anda dapat mewarisi kelas `tf.keras.Model`.

* **Metode Kunci**:
    * `__init__(self, ...)`: Menginisialisasi sub-lapisan (lapisan `Dense`, lapisan kustom lainnya) yang akan digunakan dalam model.
    * `call(self, inputs)`: Mendefinisikan *forward pass* seluruh model, mengalirkan input melalui sub-lapisan yang telah didefinisikan.
* **Contoh Model Kustom**: *Notebook* ini kemungkinan menunjukkan model klasifikasi sederhana untuk Fashion MNIST menggunakan `tf.keras.Model`. Ini memungkinkan Anda untuk mengelola aliran data antar lapisan secara eksplisit.

---

## 4. Fungsi Loss Kustom (Custom Loss Functions)

Ketika fungsi *loss* bawaan Keras tidak sesuai dengan kebutuhan Anda, Anda dapat membuat fungsi *loss* kustom. Fungsi *loss* kustom harus mengambil `y_true` (label sebenarnya) dan `y_pred` (prediksi model) sebagai argumen, dan mengembalikan sebuah skalar yang merepresentasikan nilai kerugian.

* **Contoh Fungsi Loss Kustom**: *Notebook* ini mungkin menunjukkan contoh di mana Anda ingin memberikan bobot yang berbeda pada kesalahan klasifikasi kelas tertentu, atau menerapkan penalti yang unik.
* **Implementasi**: Fungsi *loss* kustom diteruskan ke argumen `loss` saat mengkompilasi model.

---

## 5. Metrik Kustom (Custom Metrics)

Mirip dengan fungsi *loss* kustom, Anda dapat membuat metrik kustom untuk memantau kinerja model selama pelatihan. Metrik kustom juga mengambil `y_true` dan `y_pred` sebagai argumen dan mengembalikan sebuah skalar.

* **Contoh Metrik Kustom**: *Notebook* ini mungkin menunjukkan metrik yang mengukur presisi, rekal, atau f1-score untuk kasus spesifik yang tidak dicakup oleh metrik bawaan.
* **Implementasi**: Metrik kustom diteruskan ke argumen `metrics` saat mengkompilasi model.

---

## 6. Pelatihan dengan `tf.GradientTape` dan `tf.function`

Untuk kontrol penuh atas alur pelatihan (misalnya, untuk menerapkan algoritma pelatihan kustom, *custom loss* yang kompleks, atau interaksi *layer* yang unik), Anda dapat menggunakan `tf.GradientTape` dan `tf.function`.

### A. `tf.GradientTape`

`tf.GradientTape` adalah API kontekstual di TensorFlow yang memungkinkan Anda untuk merekam operasi selama *forward pass* dan kemudian menghitung gradien operasi tersebut selama *backward pass*.

* **Cara Kerja**:
    1.  Tandai variabel yang dapat dilatih (`model.trainable_variables`).
    2.  Hitung prediksi model (`y_pred = model(X_batch)`).
    3.  Hitung *loss* (`loss = custom_loss(y_true, y_pred)`).
    4.  Hitung gradien *loss* terhadap variabel yang dapat dilatih (`gradients = tape.gradient(loss, model.trainable_variables)`).
    5.  Terapkan gradien ke optimizer (`optimizer.apply_gradients(zip(gradients, model.trainable_variables))`).
* Ini adalah inti dari *custom training loop* dan memungkinkan implementasi algoritma optimasi yang sepenuhnya kustom.

### B. `tf.function`

`tf.function` adalah dekorator yang mengkompilasi fungsi Python menjadi grafik TensorFlow yang dapat dieksekusi. Ini memberikan peningkatan kinerja yang signifikan karena grafik TensorFlow dapat dioptimalkan dan dieksekusi lebih efisien di *backend* (misalnya, GPU/TPU).

* **Cara Kerja**: Mengkompilasi kode Python ke dalam grafik komputasi statis, yang kemudian dapat di-*trace* dan dioptimalkan oleh TensorFlow.
* **Penggunaan**: Menerapkan `@tf.function` ke fungsi pelatihan langkah tunggal (atau seluruh *training loop*) dapat mempercepat eksekusi.
* **Manfaat**: Peningkatan kecepatan, kemampuan untuk menyebarkan model ke berbagai platform (misalnya, TensorFlow Lite, TensorFlow.js).

---

## 7. Custom Training Loop Penuh

*Notebook* ini membangun *custom training loop* lengkap, menggabungkan semua konsep di atas:
1.  Inisialisasi *optimizer* dan metrik.
2.  Iterasi melalui *epoch*.
3.  Untuk setiap *epoch*, iterasi melalui *mini-batch* data.
4.  Di setiap *mini-batch*:
    * Lakukan *forward pass* menggunakan `tf.GradientTape`.
    * Hitung *loss*.
    * Hitung gradien.
    * Perbarui bobot model menggunakan *optimizer*.
    * Perbarui metrik pelatihan.
5.  Evaluasi metrik validasi di akhir setiap *epoch*.
6.  Gunakan `tf.function` untuk mengkompilasi *step* pelatihan per *mini-batch* untuk kinerja optimal.

---

**Kesimpulan:**

*Notebook* ini menunjukkan fleksibilitas luar biasa yang ditawarkan TensorFlow untuk membangun dan melatih *Neural Networks*. Dengan lapisan kustom, model kustom, fungsi *loss* dan metrik kustom, serta *custom training loops* yang dioptimalkan dengan `tf.GradientTape` dan `tf.function`, pengembang dapat mengimplementasikan arsitektur dan algoritma pembelajaran yang sangat spesifik dan canggih, mendorong batas-batas *Deep Learning*.
