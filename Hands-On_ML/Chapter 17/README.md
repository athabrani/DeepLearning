# Autoencoders dan Generative Adversarial Networks (GANs): Pembelajaran Tanpa Pengawasan Generatif

*Notebook* ini membahas dua arsitektur *Neural Network* yang kuat untuk **pembelajaran tanpa pengawasan generatif**: **Autoencoders** dan **Generative Adversarial Networks (GANs)**. Keduanya memiliki kemampuan untuk mempelajari representasi data yang efisien dan menghasilkan data baru yang mirip dengan data pelatihan.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `matplotlib.pyplot` untuk operasi numerik dan visualisasi, serta `os`.

Dataset yang digunakan untuk demonstrasi adalah **Fashion MNIST**, yang terdiri dari 70.000 gambar item pakaian (10 kelas).
* Data pelatihan (`X_train_full`, `y_train_full`) dan pengujian (`X_test`, `y_test`) dimuat dari `tf.keras.datasets.fashion_mnist`.
* **Normalisasi Fitur**: Nilai piksel gambar (0-255) dinormalisasi ke rentang 0-1 untuk membantu pelatihan *Neural Network*.
    $$
    X_{\text{normalized}} = \frac{X_{\text{original}}}{255.0}
    $$
* **Pemisahan Data Validasi**: Sebagian dari set pelatihan dipisahkan sebagai set validasi (`X_valid`, `y_valid`) untuk memantau kinerja model selama pelatihan.

---

## 2. Autoencoders

Autoencoder adalah jenis *Neural Network* yang dilatih untuk mempelajari representasi data input dengan mencoba merekonstruksi inputnya sendiri. Mereka terdiri dari dua bagian utama:

* **Encoder**: Bagian ini mengambil input dan mengompresnya menjadi representasi berdimensi lebih rendah (disebut *encoding* atau *latent representation*).
* **Decoder**: Bagian ini mengambil *encoding* dan mencoba merekonstruksi input asli dari representasi terkompresi tersebut.

Tujuan autoencoder adalah meminimalkan *reconstruction loss*, yaitu perbedaan antara input asli dan output yang direkonstruksi.

### A. Autoencoder Stacked (Fully Connected)

Ini adalah autoencoder dasar yang dibangun dengan lapisan `Dense` (fully connected).

* **Arsitektur**:
    * Input: Gambar yang di-*flatten* (misalnya, 784 piksel untuk Fashion MNIST).
    * Encoder: Serangkaian lapisan `Dense` yang mengurangi dimensi input secara bertahap (misalnya, 784 -> 300 -> 150 -> *encoding_dim*).
    * *Encoding Layer* (Hidden Layer): Lapisan tengah dengan dimensi terendah (misalnya, 30 neuron), merepresentasikan *encoding* data.
    * Decoder: Serangkaian lapisan `Dense` yang memperluas dimensi kembali ke ukuran input asli (misalnya, *encoding_dim* -> 150 -> 300 -> 784).
    * Output: Merekonstruksi gambar asli (misalnya, 784 piksel).
* **Fungsi Aktivasi**: Umumnya menggunakan `relu` untuk *hidden layers* dan `sigmoid` untuk *output layer* jika piksel dinormalisasi ke 0-1, karena `sigmoid` menghasilkan output antara 0 dan 1.
* **Fungsi Kerugian (Loss Function)**: `binary_crossentropy` atau `mean_squared_error` (MSE). Untuk gambar biner atau nilai piksel dinormalisasi ke 0-1, `binary_crossentropy` sering digunakan:
    $$
    L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
    $$
    Di mana $y_i$ adalah nilai piksel asli dan $\hat{y}_i$ adalah nilai piksel yang direkonstruksi.
* **Melatih Autoencoder**: Model dilatih menggunakan `model.fit()` pada data pelatihan, dengan *input* dan *target* yang sama (misalnya, `X_train` sebagai input dan `X_train` sebagai target).

### B. Autoencoder Konvolusional (Convolutional Autoencoder)

Untuk data gambar, Autoencoder Konvolusional (CAE) biasanya memberikan kinerja yang lebih baik daripada Autoencoder *Fully Connected*.

* **Arsitektur**:
    * Encoder: Menggunakan lapisan `Conv2D` dan `MaxPooling2D` untuk mengurangi dimensi spasial dan mengekstrak fitur, menghasilkan *encoding* berdimensi lebih rendah.
    * Decoder: Menggunakan lapisan `Conv2DTranspose` (juga dikenal sebagai *deconvolutional* atau *upsampling* konvolusional) untuk memperbesar kembali *feature maps* ke ukuran gambar asli.
* **Manfaat**: Mampu memanfaatkan struktur spasial dalam gambar, menghasilkan rekonstruksi yang lebih baik.

### C. Autoencoder Denoising

Autoencoder dapat dilatih untuk membersihkan *noise* dari gambar.

* **Cara Kerja**: Input yang diberikan kepada autoencoder adalah gambar yang berisik, tetapi *target* yang ingin direkonstruksi adalah gambar asli yang bersih.
* Ini memaksa autoencoder untuk mempelajari representasi *robust* dari gambar asli, mengabaikan *noise*.

### D. Autoencoder Sparse

Autoencoder *sparse* memperkenalkan penalti ke fungsi biaya untuk mendorong *encoding* agar memiliki banyak nilai nol. Ini dapat menghasilkan representasi yang lebih efisien dan terinterpretasi.

### E. Autoencoder Variasional (Variational Autoencoder - VAE)

VAE adalah jenis autoencoder generatif yang berbeda. Daripada menghasilkan *encoding* tunggal, *encoder* VAE menghasilkan distribusi probabilitas (mean dan standar deviasi) untuk setiap dimensi dalam *latent space*.

* **Konsep**: VAE mencoba untuk memastikan bahwa *latent space* terdistribusi secara normal (Gaussian) dan bahwa *cluster* yang serupa terletak berdekatan di *latent space*.
* **Sampling**: Sebuah sampel diambil dari distribusi ini untuk menghasilkan *encoding* yang kemudian diumpankan ke *decoder*.
* **Fungsi Kerugian**: VAE memiliki dua komponen *loss*:
    1.  *Reconstruction loss* (seberapa baik *decoder* merekonstruksi input).
    2.  *KL divergence loss* (seberapa dekat distribusi *latent space* yang dihasilkan dengan distribusi Gaussian standar). Ini adalah penalti regularisasi.
* **Kemampuan Generatif**: VAE adalah model generatif yang lebih baik karena *latent space*-nya yang terstruktur dengan baik memungkinkan interpolasi yang halus dan pengambilan sampel yang bermakna untuk menghasilkan instance baru.

---

## 3. Generative Adversarial Networks (GANs)

GANs adalah arsitektur generatif yang sangat inovatif yang terdiri dari dua jaringan saraf yang saling berkompetisi:

* **Generator**: Jaringan ini menerima *noise* acak sebagai input dan menghasilkan data baru yang realistis (misalnya, gambar). Tujuannya adalah untuk menghasilkan data yang cukup realistis sehingga *Discriminator* tidak dapat membedakannya dari data asli.
* **Discriminator**: Jaringan ini menerima data dari dua sumber: data asli (dari set pelatihan) dan data yang dihasilkan oleh *Generator*. Tujuannya adalah untuk membedakan antara data "asli" dan data "palsu" yang dihasilkan oleh *Generator*.

### A. Proses Pelatihan (Adversarial Training)

Pelatihan GAN bersifat adversarial (bertentangan), di mana kedua jaringan dilatih secara bergantian, seolah-olah bermain permainan *minimax*:

1.  **Latih *Discriminator***: *Discriminator* dilatih untuk mengklasifikasikan data asli sebagai "asli" (target 1) dan data yang dihasilkan *Generator* sebagai "palsu" (target 0). Bobot *Generator* dibekukan selama tahap ini.
2.  **Latih *Generator***: *Generator* dilatih untuk menghasilkan data yang dapat "menipu" *Discriminator* (membuat *Discriminator* mengklasifikasikan output *Generator* sebagai "asli"). Bobot *Discriminator* dibekukan selama tahap ini.
Kedua langkah ini diulang berkali-kali. Seiring waktu, *Generator* menjadi lebih baik dalam menghasilkan data realistis, dan *Discriminator* menjadi lebih baik dalam membedakan antara data asli dan palsu, hingga *Generator* dapat menghasilkan data yang tidak dapat dibedakan.

* **Fungsi Kerugian**:
    * **Discriminator Loss**: Biasanya *binary cross-entropy* untuk klasifikasi biner (asli vs. palsu).
    * **Generator Loss**: *Generator* juga menggunakan *binary cross-entropy*, tetapi dengan tujuan membuat *Discriminator* memprediksi "asli" untuk output *Generator*.
    \[
    \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]
    \]
    Di mana $D(x)$ adalah probabilitas *Discriminator* bahwa $x$ adalah data asli, $G(z)$ adalah output *Generator* untuk *noise* $z$, $p_{\text{data}}(x)$ adalah distribusi data asli, dan $p_z(z)$ adalah distribusi *noise* input.

### B. Deep Convolutional GANs (DCGANs)

DCGANs adalah kelas GAN yang menggunakan lapisan konvolusional (terutama `Conv2DTranspose` di *Generator* dan `Conv2D` di *Discriminator*) untuk menghasilkan gambar. Mereka telah terbukti sangat efektif dalam menghasilkan gambar yang sangat realistis.

---
