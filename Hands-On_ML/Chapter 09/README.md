# Pembelajaran Tanpa Pengawasan (Unsupervised Learning): Clustering, GMMs, dan Anomaly Detection

*Notebook* ini membahas berbagai algoritma **Pembelajaran Tanpa Pengawasan (Unsupervised Learning)**, yang bertujuan untuk menemukan pola dan struktur dalam data tanpa adanya label target. Fokus utamanya adalah pada teknik *clustering* (pengelompokan), di mana algoritma mengidentifikasi kelompok-kelompok (cluster) alami dalam data.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal adalah mengimpor pustaka yang diperlukan seperti `numpy` dan `matplotlib.pyplot` untuk operasi numerik dan visualisasi, serta berbagai modul `scikit-learn` yang relevan.

Dataset yang umum digunakan untuk demonstrasi adalah **MNIST**, yang terdiri dari 70.000 gambar tulisan tangan angka (0-9).
* `X_digits` (fitur) memiliki bentuk `(70000, 784)`, artinya 70.000 gambar, masing-masing dengan 784 piksel (28x28).
* `y_digits` (label) memiliki bentuk `(70000,)`, yang merupakan label angka 0-9 untuk setiap gambar. Meskipun ini adalah tugas tanpa pengawasan, label digunakan untuk evaluasi kinerja *clustering* (misalnya, seberapa baik *cluster* yang ditemukan sesuai dengan kategori angka sebenarnya).

---

## 2. Clustering (Pengelompokan)

*Clustering* adalah tugas untuk mengelompokkan set instance sedemikian rupa sehingga instance dalam *cluster* yang sama lebih mirip satu sama lain daripada instance di *cluster* lain.

### 2.1. K-Means

K-Means adalah salah satu algoritma *clustering* yang paling populer dan efisien.

* **Cara Kerja**:
    1.  Inisialisasi $k$ *centroid* (titik pusat *cluster*) secara acak.
    2.  Setiap instance ditetapkan ke *centroid* terdekat.
    3.  *Centroid* diperbarui dengan mengambil rata-rata instance yang ditugaskan ke *cluster* tersebut.
    4.  Langkah 2 dan 3 diulang hingga *centroid* tidak banyak bergerak atau mencapai jumlah iterasi maksimum.
* **Fungsi Biaya (Inertia)**: Tujuan K-Means adalah meminimalkan *inertia*, yang merupakan jumlah kuadrat jarak antara setiap instance dan *centroid*-nya.
    $$
    J = \sum_{i=0}^{n-1} \min_{k} (||x_i - \mu_k||^2)
    $$
    Di mana $x_i$ adalah instance, dan $\mu_k$ adalah *centroid* dari *cluster* $k$.
* **Inisialisasi Centroid**: Karena K-Means dapat macet di *local optimum* (hasil *clustering* dapat bervariasi tergantung inisialisasi awal), `scikit-learn` menggunakan strategi **k-means++** secara *default* untuk inisialisasi *centroid* yang lebih cerdas, yang cenderung menghasilkan *cluster* yang lebih baik. Anda juga dapat menentukan `n_init` untuk menjalankan K-Means beberapa kali dengan inisialisasi berbeda dan memilih yang terbaik.
* **Menemukan Jumlah Cluster Optimal (Metode Elbow)**:
    * Plot *inertia* terhadap jumlah *cluster* ($k$).
    * Cari "siku" (elbow point) pada plot, di mana penurunan *inertia* mulai melambat secara signifikan. Ini sering menunjukkan jumlah *cluster* yang optimal.
* **Limitasi K-Means**:
    * Membutuhkan jumlah *cluster* ($k$) yang ditentukan sebelumnya.
    * Tidak bekerja dengan baik untuk *cluster* yang tidak berbentuk bulat atau *cluster* dengan kepadatan yang sangat berbeda.
    * Sensitif terhadap *outlier*.
* **Pre-processing**: Menskalakan fitur (misalnya, menggunakan `StandardScaler`) sangat penting untuk K-Means karena algoritma ini berbasis jarak.

### 2.2. Batch K-Means vs Mini-Batch K-Means

* **Batch K-Means**: Seperti dijelaskan di atas, menghitung *centroid* berdasarkan semua instance di *cluster*.
* **Mini-Batch K-Means**: Menghitung *centroid* hanya pada subset acak (mini-batch) dari instance di setiap iterasi. Ini jauh lebih cepat untuk dataset besar, tetapi mungkin menghasilkan *cluster* yang sedikit kurang optimal.

### 2.3. Propagasi Afinitas (Affinity Propagation)

*Afﬁnity Propagation* adalah algoritma *clustering* yang tidak memerlukan jumlah *cluster* yang ditentukan sebelumnya. Algoritma ini mengidentifikasi "exemplars" (instance yang paling representatif) dalam data dan membangun *cluster* di sekitar mereka.

### 2.4. *Mean-Shift*

*Mean-Shift* adalah algoritma *clustering* berbasis kepadatan. Ini bekerja dengan mencari mode (puncak kepadatan) dalam distribusi titik data.

### 2.5. Spectral Clustering

*Spectral Clustering* menggunakan spektrum Laplacian dari *graph* kedekatan data untuk melakukan reduksi dimensi dan *clustering*. Ini efektif untuk *cluster* yang tidak berbentuk konveks.

---

## 3. Gaussian Mixture Models (GMMs)

GMM adalah model *clustering* probabilistik yang mengasumsikan bahwa data dihasilkan dari campuran beberapa distribusi Gaussian (normal).

* **Cara Kerja**: GMM mencoba memperkirakan parameter (mean, kovarians, bobot) dari distribusi Gaussian yang membentuk data. Instance ditugaskan ke Gaussian yang paling mungkin menghasilkannya.
* **Implementasi**: `GaussianMixture` dari `sklearn.mixture` digunakan.
* **Menemukan Jumlah Cluster Optimal**: GMM dapat dievaluasi menggunakan metrik seperti **Bayesian Information Criterion (BIC)** atau **Akaike Information Criterion (AIC)**. Semakin rendah BIC/AIC, semakin baik modelnya.
    $$
    BIC = -2 \log(L) + k \log(n)
    $$
    $$
    AIC = -2 \log(L) + 2k
    $$
    Di mana $L$ adalah *likelihood* dari model, $k$ adalah jumlah parameter, dan $n$ adalah jumlah sampel.
* **GMMs dapat Digunakan untuk**:
    * **Clustering**: Jika Anda mengetahui jumlah *cluster* atau dapat menemukannya melalui BIC/AIC.
    * **Deteksi Anomali (Anomaly Detection)**: Instance yang memiliki *likelihood* rendah untuk dihasilkan oleh GMM (berada di daerah kepadatan rendah) dapat dianggap sebagai anomali.
    * **Denormalisasi Data**: GMM dapat digunakan untuk menghasilkan sampel baru yang mirip dengan data pelatihan.

---

## 4. Deteksi Anomali dan *Novelty*

Deteksi anomali (atau *outlier detection*) adalah tugas untuk mengidentifikasi instance yang menyimpang secara signifikan dari sebagian besar data.

* **One-Class SVM**:
    * `OneClassSVM` dari `sklearn.svm` dilatih pada data "normal" dan mencoba menemukan batas yang mengelilingi data ini. Instance di luar batas dianggap sebagai *outlier* atau anomali.
    * Parameter `nu` mengontrol fraksi *outlier* yang diharapkan.
* **Isolation Forest**:
    * `IsolationForest` dari `sklearn.ensemble` membangun pohon isolasi secara acak. Anomali adalah instance yang dapat diisolasi dengan cepat (memiliki jalur yang pendek di pohon).
* **Local Outlier Factor (LOF)**:
    * `LocalOutlierFactor` dari `sklearn.neighbors` mengukur kepadatan lokal suatu instance relatif terhadap tetangganya. Instance di daerah berdensitas rendah dianggap *outlier*.

---

## 5. Algoritma Clustering Lainnya

* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
    * Algoritma berbasis kepadatan yang dapat menemukan *cluster* berbentuk arbitrer dan mengidentifikasi *noise*. Tidak memerlukan jumlah *cluster* yang ditentukan sebelumnya.
* **Agglomerative Clustering**:
    * Membangun hierarki *cluster* secara bertahap, menggabungkan *cluster* terkecil pada setiap langkah.
* **Birch (Balanced Iterative Reducing and Clustering using Hierarchies)**:
    * Algoritma *clustering* hierarkis yang mengoptimalkan memori untuk dataset besar dengan membangun *Clustering Feature (CF) Tree*.

---
