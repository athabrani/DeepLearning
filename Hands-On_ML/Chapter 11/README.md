# Melatih Deep Neural Networks (DNNs): Tantangan dan Solusi

*Notebook* ini membahas tantangan-tantangan yang muncul saat melatih Deep Neural Networks (DNNs) dan memperkenalkan berbagai teknik serta strategi untuk mengatasinya. Fokus utamanya adalah pada masalah *vanishing/exploding Gradients*, *non-saturating activation functions*, *Batch Normalization*, *optimizer* yang lebih cepat, *transfer learning*, dan teknik regularisasi seperti *Dropout*.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` dan `keras`, `numpy`, `pandas`, serta `matplotlib.pyplot` untuk operasi numerik dan visualisasi.

Dataset yang digunakan untuk demonstrasi adalah **Fashion MNIST**, yang terdiri dari 70.000 gambar item pakaian (10 kelas).
* Data pelatihan (`X_train`, `y_train`) dan pengujian (`X_test`, `y_test`) dimuat dari `keras.datasets.fashion_mnist`.
* **Normalisasi Fitur**: Nilai piksel gambar (0-255) dinormalisasi ke rentang 0-1 untuk membantu pelatihan *Neural Network*.
    $$
    X_{\text{normalized}} = \frac{X_{\text{original}}}{255.0}
    $$
* **Pemisahan Data Validasi**: Sebagian dari set pelatihan dipisahkan sebagai set validasi untuk memantau kinerja model selama pelatihan.

---

## 2. Masalah Vanishing/Exploding Gradients

* **Vanishing Gradients**: Saat *backpropagation*, gradien seringkali menjadi sangat kecil seiring dengan perjalanannya mundur melalui lapisan-lapisan. Ini menyebabkan pembaruan bobot di lapisan awal menjadi sangat kecil, sehingga pelatihan menjadi sangat lambat atau bahkan terhenti. Ini sering terjadi dengan fungsi aktivasi Sigmoid dan Tanh pada DNN yang dalam.
* **Exploding Gradients**: Sebaliknya, gradien bisa menjadi sangat besar, menyebabkan pembaruan bobot yang sangat besar dan membuat model tidak stabil.

### Solusi: Non-saturating Activation Functions

Fungsi aktivasi Sigmoid dan Tanh mengalami masalah "saturasi" (outputnya menjadi sangat datar di ujung, menyebabkan gradien mendekati nol). Solusi modern adalah menggunakan fungsi aktivasi non-saturating:

* **ReLU (Rectified Linear Unit)**:
    $$
    \text{ReLU}(x) = \max(0, x)
    $$
    * **Kelebihan**: Cepat dihitung, tidak mengalami *vanishing gradients* untuk $x > 0$.
    * **Kekurangan**: Masalah "dying ReLUs" (neuron bisa mati total jika outputnya selalu negatif).
* **Leaky ReLU**: Mengatasi masalah "dying ReLUs" dengan memungkinkan sedikit gradien untuk $x < 0$.
    $$
    \text{Leaky ReLU}(x, \alpha) = \max(\alpha x, x) \quad \text{where } \alpha \text{ is a small positive constant}
    $$
* **PReLU (Parametric ReLU)**: Parameter $\alpha$ dipelajari selama pelatihan.
* **ELU (Exponential Linear Unit)**:
    $$
    \text{ELU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha (e^x - 1) & \text{if } x < 0 \end{cases}
    $$
    * **Kelebihan**: Mengungguli ReLU, terutama untuk DNNs yang dalam. Dapat mengambil nilai negatif, membantu mendorong rata-rata output mendekati nol, yang membantu mengatasi masalah *vanishing gradients*.
* **SELU (Scaled ELU)**: Versi penskalaan dari ELU yang dirancang untuk memastikan output tetap ternormalisasi (mean 0, varians 1) jika lapisan diinisialisasi dengan benar, yang secara teoritis dapat membuat jaringan "self-normalizing".

### Solusi: He Initialization (Inisialisasi Bobot)

Inisialisasi bobot yang tepat sangat penting. Untuk fungsi aktivasi ReLU (dan variasinya), inisialisasi **He Initialization** direkomendasikan. Ini menginisialisasi bobot secara acak dengan skala yang memperhitungkan jumlah *input* ke neuron.

* **He Initialization**: Bobot diambil dari distribusi Gaussian dengan mean 0 dan standar deviasi $\sqrt{2 / n_{\text{inputs}}}$, atau dari distribusi seragam dengan $r = \sqrt{6 / n_{\text{inputs}}}$.

---

## 3. Batch Normalization (BN)

Batch Normalization adalah teknik yang sangat efektif untuk mengatasi masalah *vanishing/exploding gradients* dan memungkinkan pembangunan DNNs yang lebih dalam.

* **Cara Kerja**: BN menambahkan operasi normalisasi setelah lapisan *dense* atau *convolutional* (dan biasanya sebelum fungsi aktivasi). Ia menormalisasi input setiap *mini-batch* ke lapisan tersebut, sehingga memiliki mean 0 dan varians 1.
    \[
    \hat{x}^{(k)} = \frac{x^{(k)} - \mu_{\mathcal{B}}^{(k)}}{\sqrt{(\sigma_{\mathcal{B}}^{(k)})^2 + \epsilon}}
    \]
    Kemudian, hasil normalisasi ini diskalakan dan digeser menggunakan dua parameter yang dipelajari per fitur (gamma $\gamma$ dan beta $\beta$).
    \[
    y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
    \]
    Di mana $\mu_{\mathcal{B}}^{(k)}$ adalah *mean* mini-batch, dan $\sigma_{\mathcal{B}}^{(k)}$ adalah *standard deviation* mini-batch untuk fitur $k$.
* **Manfaat**:
    * Mengurangi *internal covariate shift* (perubahan distribusi *input* ke lapisan tengah).
    * Memungkinkan penggunaan *learning rate* yang lebih tinggi.
    * Bertindak sebagai bentuk regularisasi, mengurangi kebutuhan *Dropout*.
    * Membuat model lebih *robust* terhadap inisialisasi bobot.
* **Implementasi**: Lapisan `BatchNormalization` ditambahkan setelah lapisan `Dense` (atau `Flatten` di awal).

---

## 4. Optimizer yang Lebih Cepat

Penggunaan *optimizers* yang lebih canggih daripada *Stochastic Gradient Descent* (SGD) dasar dapat mempercepat pelatihan DNNs secara signifikan.

* **Momentum Optimization**: Mempercepat SGD dengan menambahkan "momentum" ke pembaruan gradien, membantu optimizer melewati lembah-lembah datar dan *local optima*.
* **Nesterov Accelerated Gradient (NAG)**: Versi yang sedikit lebih baik dari Momentum Optimization.
* **AdaGrad (Adaptive Gradient)**: Menyesuaikan *learning rate* untuk setiap parameter, menurunkan *learning rate* untuk parameter dengan gradien sering atau besar.
* **RMSProp (Root Mean Square Propagation)**: Mirip dengan AdaGrad tetapi hanya mempertimbangkan gradien dari iterasi terbaru, mengatasi masalah *learning rate* yang terlalu cepat menurun di AdaGrad.
* **Adam (Adaptive Moment Estimation)**: Menggabungkan ide-ide dari Momentum Optimization dan RMSProp. Ini adalah salah satu *optimizer* yang paling populer dan seringkali pilihan *default* yang baik.
* **Nadam (Nesterov Adam)**: Adam dengan Nesterov momentum.

---

## 5. Mencegah Overfitting Melalui Regularisasi

DNNs dengan banyak parameter sangat rentan terhadap *overfitting*. Selain Batch Normalization, ada beberapa teknik regularisasi lain:

### A. Regularisasi $L_1$ dan $L_2$

* Menambahkan penalti ke fungsi biaya yang bergantung pada ukuran bobot.
    * **$L_1$ Regularization (Lasso)**: Menambahkan jumlah nilai absolut bobot. Mendorong beberapa bobot menjadi nol, melakukan seleksi fitur.
    * **$L_2$ Regularization (Ridge/Weight Decay)**: Menambahkan jumlah kuadrat bobot. Mendorong bobot menjadi lebih kecil tetapi tidak nol.
* Di Keras, ini diimplementasikan menggunakan argumen `kernel_regularizer` dalam lapisan `Dense`.

### B. Dropout

*Dropout* adalah salah satu teknik regularisasi paling populer untuk *Neural Networks*.

* **Cara Kerja**: Selama pelatihan, *Dropout* secara acak "menonaktifkan" (mengatur output menjadi nol) sebagian kecil neuron di setiap *epoch*. Ini memaksa jaringan untuk belajar representasi *robust* yang tidak terlalu bergantung pada neuron tertentu.
* **Parameter `rate`**: Probabilitas neuron untuk di-*drop* (dinonaktifkan).
* **Pada Waktu Inferensi**: Saat membuat prediksi (mode inferensi), *Dropout* tidak diterapkan. Sebagai gantinya, bobot *output* dari setiap neuron dikalikan dengan `(1 - rate)` untuk mengkompensasi fakta bahwa lebih sedikit neuron yang aktif dibandingkan saat pelatihan.
* **Implementasi**: Lapisan `Dropout` ditambahkan setelah lapisan `Dense` di Keras.

---

## 6. Transfer Learning (Pembelajaran Transfer)

*Transfer Learning* adalah teknik di mana Anda mengambil model yang sudah dilatih pada satu tugas (seringkali pada dataset yang sangat besar) dan menggunakannya sebagai titik awal untuk tugas lain yang serupa.

* **Cara Kerja**: Bagian awal dari model yang sudah dilatih (lapisan-lapisan konvolusional yang mengekstrak fitur dasar) dibekukan (bobotnya tidak diperbarui selama pelatihan). Lapisan *output* baru ditambahkan dan dilatih pada data tugas baru.
* **Manfaat**:
    * Mempercepat pelatihan secara signifikan.
    * Membutuhkan data pelatihan yang jauh lebih sedikit untuk tugas baru.
    * Sangat efektif untuk tugas klasifikasi gambar.
* **Implementasi**:
    * Menggunakan model *pre-trained* dari `tf.keras.applications` (misalnya, `ResNet50V2`).
    * Membekukan lapisan dasar model *pre-trained* (`model_pre_trained.trainable = False`).
    * Menambahkan lapisan *output* baru di atas model *pre-trained*.
    * Melatih hanya lapisan baru ini pada awalnya.
    * Opsional: Melakukan *fine-tuning* dengan "mencairkan" beberapa lapisan atas dari model *pre-trained* dan melatihnya dengan *learning rate* yang sangat kecil.

---

## 7. Model Pra-Pelatihan Lebih Cepat

*Notebook* ini juga menunjukkan cara melatih model secara efisien dengan:
* **`callbacks`**: Fungsi yang dapat dipanggil selama pelatihan pada berbagai titik (misalnya, akhir setiap *epoch*).
    * `ModelCheckpoint`: Untuk menyimpan model terbaik selama pelatihan (berdasarkan kinerja validasi).
    * `EarlyStopping`: Untuk menghentikan pelatihan secara otomatis ketika kinerja validasi tidak lagi membaik selama beberapa *epoch* (`patience`). Ini mencegah *overfitting* dan menghemat waktu.
* **TensorBoard**: Untuk visualisasi *loss* dan metrik lainnya selama pelatihan secara interaktif.

---

**Kesimpulan:**

Melatih *Deep Neural Networks* melibatkan mengatasi tantangan seperti *vanishing/exploding gradients* dan *overfitting*. *Notebook* ini menunjukkan bahwa solusi seperti fungsi aktivasi non-saturating (ReLU, ELU), inisialisasi bobot yang tepat (He Initialization), Batch Normalization, *optimizer* canggih (Adam), regularisasi (Dropout), dan *transfer learning* sangat penting untuk membangun dan melatih DNNs yang stabil, efisien, dan berkinerja tinggi.
