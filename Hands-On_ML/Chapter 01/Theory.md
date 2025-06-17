# Analisis Hubungan Antara PDB per Kapita dan Kepuasan Hidup: Pendekatan Regresi Linier dan K-Nearest Neighbors

Notebook ini mengeksplorasi hubungan antara Produk Domestik Bruto (PDB) per kapita suatu negara dengan tingkat kepuasan hidup penduduknya, menggunakan data historis dan mengaplikasikan dua model machine learning dasar: Regresi Linier dan K-Nearest Neighbors. Tujuannya adalah untuk memahami apakah kekayaan suatu negara (yang diukur dengan PDB per kapita) dapat memprediksi tingkat kepuasan hidup, serta untuk mendemonstrasikan konsep underfitting dan overfitting dalam pemodelan.

## 1. Persiapan Lingkungan dan Data

Sebelum memulai analisis, notebook ini memastikan bahwa lingkungan Python dan pustaka scikit-learn yang relevan sudah terinstal dan memiliki versi yang memadai. Pustaka penting seperti pandas untuk manipulasi data, matplotlib untuk visualisasi, dan numpy untuk operasi numerik juga diimpor.

Data yang digunakan berasal dari dua sumber utama:

- **OECD Better Life Index (BLI)**: Menyediakan berbagai indikator kualitas hidup, termasuk "Life satisfaction" (kepuasan hidup).
- **IMF's World Economic Outlook (WEO) database**: Menyediakan data PDB per kapita untuk berbagai negara.

Fungsi `prepare_country_stats` digunakan untuk membersihkan dan menggabungkan kedua dataset ini. Proses ini melibatkan:

- **Penyaringan Data OECD BLI**: Hanya memilih baris di mana "INEQUALITY" (ketidaksetaraan) adalah "TOT" (total), yang berarti data untuk keseluruhan populasi.
- **Pivot Table**: Mengubah format data OECD BLI dari long format menjadi wide format, di mana setiap indikator menjadi kolom dan setiap negara menjadi indeks, sehingga memudahkan pengaksesan nilai "Life satisfaction".
- **Penamaan Ulang dan Pengaturan Indeks GDP**: Mengganti nama kolom PDB per kapita menjadi lebih deskriptif dan mengatur "Country" sebagai indeks.
- **Penggabungan Data**: Menggabungkan kedua dataframe `oecd_bli` dan `gdp_per_capita` berdasarkan indeks "Country" (nama negara), menghasilkan dataframe lengkap `full_country_stats`.
- **Pengurutan dan Pemilihan Kolom**: Mengurutkan data berdasarkan "GDP per capita" dan memilih hanya kolom "GDP per capita" dan "Life satisfaction" untuk analisis lebih lanjut.
- **Penghapusan Outlier/Data yang Tidak Representatif**: Beberapa indeks negara dihapus (`remove_indices`) untuk memastikan data yang digunakan untuk pelatihan awal lebih representatif dan tidak terlalu dipengaruhi oleh outlier yang bisa mengaburkan hubungan linier. Data yang dihapus ini kemudian disimpan dalam `missing_data` untuk tujuan demonstrasi selanjutnya.

Data yang telah disiapkan (`sample_data`) kemudian diplot dalam scatter plot untuk memberikan gambaran visual awal tentang hubungan antara PDB per kapita dan kepuasan hidup.

## 2. Pemodelan dengan Regresi Linier

### A. Regresi Linier Sederhana

Regresi Linier adalah model machine learning dasar yang bertujuan untuk menemukan hubungan linier terbaik antara satu atau lebih fitur independen (PDB per kapita, X) dan satu variabel dependen (Kepuasan Hidup, y). Model ini mencoba menemukan garis lurus yang paling cocok dengan data, yang dapat direpresentasikan dengan persamaan:

\[
y = \theta_0 + \theta_1 x
\]

Di mana:
- \(y\) adalah variabel dependen (Kepuasan Hidup).
- \(x\) adalah fitur independen (PDB per kapita).
- \(\theta_0\) adalah intercept (titik potong Y), yaitu nilai \(y\) ketika \(x\) adalah nol.
- \(\theta_1\) adalah koefisien (kemiringan garis), yang menunjukkan seberapa besar perubahan \(y\) untuk setiap unit perubahan \(x\).

Notebook ini mengaplikasikan `LinearRegression` dari `sklearn.linear_model` pada `sample_data`. Setelah melatih model (`model.fit(X, y)`), koefisien \(\theta_0\) dan \(\theta_1\) yang optimal diperoleh. Koefisien-koefisien ini kemudian digunakan untuk memprediksi kepuasan hidup sebuah negara baru (Siprus) berdasarkan PDB per kapitanya. Visualisasi menunjukkan garis regresi yang dihasilkan bersama dengan titik-titik data, termasuk prediksi untuk Siprus.

### B. Demonstrasi Underfitting dan Overfitting

Bagian ini menunjukkan bagaimana pemilihan model dan kompleksitasnya dapat memengaruhi kinerja.

- **Kasus Underfitting**: Plot scatter awal dengan beberapa garis linier yang digambar secara manual (dengan koefisien \(\theta_0\) dan \(\theta_1\) yang berbeda) mendemonstrasikan underfitting. Ini terjadi ketika model terlalu sederhana untuk menangkap pola yang sebenarnya dalam data, menghasilkan akurasi yang rendah pada data pelatihan maupun data baru. Garis-garis ini jelas tidak mewakili tren data dengan baik.
- **Kasus Overfitting**: Untuk mendemonstrasikan overfitting, sebuah model regresi polinomial dengan derajat tinggi (derajat 30) digunakan, dikombinasikan dengan `StandardScaler` dalam sebuah pipeline. Model ini dilatih pada seluruh dataset (`full_country_stats`).

  - **PolynomialFeatures**: Mengubah fitur input menjadi fitur polinomial (misalnya, jika inputnya \(x\), ia akan menghasilkan \(x, x^2, x^3, \dots, x^{30}\)). Ini secara signifikan meningkatkan kompleksitas model, memungkinkannya untuk menyesuaikan diri dengan fluktuasi kecil dalam data.
  - **StandardScaler**: Melakukan scaling (penyesuaian skala) pada fitur, yang penting saat menggunakan fitur polinomial untuk mencegah masalah numerik dan membantu algoritma optimasi bekerja lebih baik.
  - **Pipeline**: Menggabungkan langkah-langkah preprocessing dan model menjadi satu objek yang koheren.

Model berderajat tinggi ini mampu melewati setiap titik data pelatihan dengan sangat presisi, menghasilkan kurva yang sangat "bergelombang". Namun, ini adalah contoh klasik overfitting: model mempelajari "kebisingan" dalam data pelatihan, bukan hanya pola yang mendasarinya. Akibatnya, kinerja model ini akan sangat buruk pada data baru yang tidak terlihat, karena ia terlalu spesifik untuk data pelatihan yang ada.

Plot ini secara visual membandingkan:
- Garis regresi linier yang dilatih pada `sample_data` (data parsial yang sudah "dibersihkan" dari outlier).
- Garis regresi linier yang dilatih pada `full_country_stats` (seluruh data, termasuk outlier).
- Kurva regresi polinomial derajat 30 yang dilatih pada `full_country_stats`.

Perbandingan ini menyoroti bahwa model linier pada `sample_data` mungkin menunjukkan kinerja yang lebih baik secara keseluruhan pada data baru dibandingkan model polinomial yang overfit, meskipun model polinomial terlihat "lebih cocok" dengan data pelatihan.

## 3. Pemodelan dengan K-Nearest Neighbors (KNN)

Sebagai alternatif dari regresi linier, model KNeighborsRegressor juga diimplementasikan. Model ini adalah algoritma non-parametrik yang memprediksi nilai target berdasarkan rata-rata nilai target dari k tetangga terdekat di ruang fitur. Dalam kasus ini, \(k\) diatur menjadi 3 (`n_neighbors=3`).

Model dilatih pada data yang sama (\(X, y\)) yang digunakan untuk regresi linier. Prediksi untuk Siprus juga dilakukan dengan model KNN ini. Hasilnya sedikit berbeda dari regresi linier, menunjukkan bahwa model yang berbeda dapat menghasilkan prediksi yang berbeda tergantung pada algoritma dan asumsinya.

Model KNN dapat menangkap hubungan non-linier, tetapi pemilihannya (\(k\)) juga krusial untuk menghindari overfitting atau underfitting. Nilai \(k\) yang terlalu kecil dapat menyebabkan overfitting (sensitif terhadap noise), sementara \(k\) yang terlalu besar dapat menyebabkan underfitting (rata-rata terlalu banyak data yang jauh).

## 4. Kesimpulan Teoritis

Notebook ini secara efektif mendemonstrasikan bahwa dalam machine learning:
- **Kualitas Data Itu Penting**: Proses persiapan dan pembersihan data (`prepare_country_stats`) sangat krusial. Data yang tidak representatif atau mengandung outlier dapat secara signifikan memengaruhi model dan menyebabkan underfitting (jika data yang dibuang mengandung informasi penting) atau overfitting (jika model terlalu fokus pada noise).
- **Pemilihan Model**: Model yang berbeda (seperti regresi linier vs. KNN) memiliki asumsi dan kemampuan yang berbeda dalam menangkap pola dalam data.
- **Kompleksitas Model**: Penting untuk menemukan keseimbangan yang tepat dalam kompleksitas model.
  - Model Sederhana (Regresi Linier): Rentan terhadap underfitting jika hubungan data non-linier atau jika data yang digunakan tidak lengkap/representatif.
  - Model Kompleks (Regresi Polinomial Derajat Tinggi): Rentan terhadap overfitting, di mana model mempelajari noise dan detail spesifik dari data pelatihan, menyebabkan kinerja buruk pada data baru.
- **Regulasi (Regularization)**: Meskipun tidak dijelaskan secara detail dalam narasi Anda, kode menunjukkan penggunaan `Ridge regression`. Ini adalah teknik regularization yang membantu mencegah overfitting pada model linier (terutama dengan fitur polinomial) dengan menambahkan penalti pada ukuran koefisien model, sehingga membuat model lebih sederhana dan generalizable.
