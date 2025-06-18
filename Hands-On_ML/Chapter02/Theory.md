# Proyek Machine Learning End-to-End: Membangun Model Prediksi Harga Rumah di California

*Notebook* ini memandu melalui proses lengkap membangun proyek *Machine Learning*, mulai dari perolehan data hingga penyebaran model. Fokusnya adalah pada dataset harga rumah di California, yang melibatkan serangkaian langkah standar dalam alur kerja *machine learning* untuk menghasilkan model prediktif.

---

## 1. Persiapan Lingkungan dan Perolehan Data

Langkah pertama adalah memastikan lingkungan Python siap dengan pustaka yang diperlukan dan mengonfigurasi pengaturan umum untuk visualisasi.

* **Impor Pustaka**: Pustaka standar seperti `numpy` dan `pandas` untuk manipulasi data, `matplotlib` dan `seaborn` untuk visualisasi, serta berbagai modul dari `sklearn` untuk *machine learning* diimpor.
* **Pengaturan Plot**: Konfigurasi `matplotlib` disesuaikan untuk memastikan plot memiliki ukuran label yang konsisten dan grid yang rapi, yang penting untuk visualisasi data yang jelas.
* **Fungsi `save_fig`**: Fungsi *helper* ini dibuat untuk menyimpan plot yang dihasilkan ke dalam format gambar (`.png`) dengan resolusi tinggi, yang berguna untuk dokumentasi atau presentasi.
* **Perolehan Data (Fetch the Data)**: Data harga rumah California diunduh dari repositori GitHub yang ditentukan. Proses ini melibatkan:
    * Membuat direktori khusus untuk dataset jika belum ada.
    * Mendefinisikan URL unduhan untuk `housing.tgz` (file terkompresi yang berisi dataset).
    * Mengunduh file terkompresi.
    * Mengekstrak file `housing.csv` dari arsip `.tgz`.
* **Memuat Data (Load the Data)**: File `housing.csv` yang sudah diekstrak kemudian dimuat ke dalam *DataFrame* `pandas`. Ini adalah titik awal untuk eksplorasi dan pra-pemrosesan data.

---

## 2. Eksplorasi Data (Exploratory Data Analysis - EDA)

Setelah data dimuat, langkah selanjutnya adalah memahami struktur dan karakteristiknya.

* **Inspeksi Cepat Data**:
    * `housing.head()`: Menampilkan beberapa baris pertama *DataFrame* untuk mendapatkan gambaran sekilas tentang data dan jenis kolom yang ada.
    * `housing.info()`: Memberikan ringkasan informasi *DataFrame*, termasuk jumlah baris, tipe data setiap kolom, dan jumlah nilai non-null. Ini sangat berguna untuk mengidentifikasi kolom dengan nilai yang hilang.
    * `housing["ocean_proximity"].value_counts()`: Menghitung frekuensi kemunculan setiap kategori unik dalam kolom kategorikal `ocean_proximity`. Ini membantu memahami distribusi nilai-nilai kategorikal.
    * `housing.describe()`: Menghasilkan statistik deskriptif (count, mean, std, min, max, kuartil) untuk kolom numerik, memberikan gambaran tentang rentang dan distribusi data.
* **Histogram**: `housing.hist(bins=50, figsize=(20, 15))` membuat histogram untuk setiap kolom numerik. Histogram sangat penting untuk:
    * Mengidentifikasi distribusi data (normal, skew, dll.).
    * Mendeteksi *outlier*.
    * Memahami rentang nilai.
    * Mengidentifikasi kolom yang perlu diskalakan atau ditransformasi.

---

## 3. Membuat Set Pengujian (Create a Test Set)

Sangat penting untuk memisahkan data pengujian *sebelum* melakukan analisis atau pra-pemrosesan yang mendalam pada data pelatihan. Ini mencegah *data snooping bias*, di mana informasi dari set pengujian "bocor" ke set pelatihan, menyebabkan model terlihat lebih baik dari yang sebenarnya.

* **Pembagian Acak**: Metode sederhana adalah membagi data secara acak. Namun, untuk set data yang lebih kecil, pembagian acak murni dapat menyebabkan *sampling bias* yang signifikan.
* **Pembagian Bertingkat (Stratified Sampling)**: Untuk memastikan set pelatihan dan pengujian merepresentasikan populasi asli secara proporsional berdasarkan atribut penting, pembagian bertingkat dilakukan. Dalam kasus ini, atribut `income_cat` (kategori pendapatan) dibuat untuk memastikan setiap strata pendapatan terwakili dengan baik di kedua set.
    * `housing["median_income"].hist()`: Memvisualisasikan distribusi `median_income`.
    * `pd.cut()`: Digunakan untuk mengkategorikan `median_income` menjadi beberapa *bin* (kategori) dengan ukuran yang kurang lebih sama, karena *median income* adalah atribut yang sangat penting untuk memprediksi harga rumah.
    * `StratifiedShuffleSplit`: Melakukan pembagian bertingkat yang memastikan proporsi `income_cat` yang sama di set pelatihan (`strat_train_set`) dan set pengujian (`strat_test_set`).
* **Menghapus Atribut Tambahan**: Kolom `income_cat` dihapus setelah pembagian set selesai.

---

## 4. Visualisasi Data untuk Memperoleh Wawasan (Discover and Visualize the Data to Gain Insights)

Langkah ini melibatkan visualisasi data pelatihan untuk menemukan pola, korelasi, dan *outlier*.

* **Membuat Salinan Data**: Sebuah salinan dari set pelatihan (`housing_copy`) dibuat untuk eksperimen visualisasi tanpa merusak data asli.
* **Scatter Plot Geografis**:
    * `housing_copy.plot(kind="scatter", x="longitude", y="latitude")`: Membuat plot sebar dasar dari posisi geografis.
    * `alpha=0.1`: Mengatur transparansi titik untuk menunjukkan kepadatan data di area tertentu.
    * `s=housing_copy["population"]/100`: Mengatur ukuran titik berdasarkan populasi (menampilkan daerah berpenduduk padat sebagai lingkaran yang lebih besar).
    * `c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True`: Mengatur warna titik berdasarkan nilai harga rumah, menggunakan *colormap* 'jet' dan menampilkan *colorbar*. Ini membantu memvisualisasikan daerah dengan harga rumah tinggi/rendah.
    * Visualisasi ini mengonfirmasi bahwa harga rumah sangat berkorelasi dengan lokasi dan kepadatan penduduk (misalnya, area di dekat laut dan daerah padat penduduk cenderung lebih mahal).

* **Mencari Korelasi**: `housing_copy.corr()` menghitung koefisien korelasi Pearson antara semua pasangan kolom numerik.
    * Koefisien korelasi berkisar dari -1 (korelasi negatif kuat) hingga +1 (korelasi positif kuat). Nilai mendekati 0 menunjukkan tidak ada korelasi linier.
    * Korelasi `median_house_value` dengan `median_income` sangat tinggi (sekitar 0.68).
    * Korelasi dengan `total_rooms`, `housing_median_age`, dan `households` juga dieksplorasi.
* **Scatter Matrix (Pairplot)**: `pd.plotting.scatter_matrix()` membuat plot sebar untuk setiap pasangan atribut numerik. Ini berguna untuk memvisualisasikan korelasi dan distribusi secara bersamaan. Hanya atribut dengan korelasi tinggi terhadap `median_house_value` yang dipilih untuk plot ini agar tidak terlalu padat.

* **Mengeksplorasi Atribut Kombinasi**: Fitur baru (kombinasi atribut yang sudah ada) dibuat untuk melihat apakah mereka memiliki kekuatan prediktif yang lebih baik. Contohnya:
    * `rooms_per_household`: Jumlah rata-rata kamar per rumah tangga.
    * `bedrooms_per_room`: Rasio kamar tidur terhadap total kamar.
    * `population_per_household`: Jumlah rata-rata orang per rumah tangga.
    * Korelasi dihitung kembali untuk fitur-fitur baru ini, dan ditemukan bahwa `bedrooms_per_room` memiliki korelasi negatif yang lebih kuat dengan harga rumah daripada `total_bedrooms` atau `total_rooms`.

---

## 5. Pra-pemrosesan Data (Prepare the Data for Machine Learning Algorithms)

Langkah ini mengubah data mentah menjadi format yang sesuai untuk model *machine learning*.

* **Memisahkan Fitur dan Label**: `X_train` dan `y_train` dipisahkan dari `strat_train_set`.
* **Penanganan Nilai yang Hilang (Missing Values)**: Kolom `total_bedrooms` memiliki beberapa nilai yang hilang. Beberapa strategi untuk menanganinya:
    1.  Menghapus baris yang mengandung nilai yang hilang (`housing.dropna()`).
    2.  Menghapus seluruh kolom (`housing.drop()`).
    3.  Mengisi nilai yang hilang dengan suatu nilai (misalnya, median, mean, atau nol) (`housing.fillna()`).
    * Strategi ke-3 (mengisi dengan median) dipilih dan `SimpleImputer` dari `sklearn.impute` digunakan. *Imputer* di-*fit* pada data pelatihan (untuk menghitung median) dan kemudian digunakan untuk men-*transform* baik data pelatihan maupun data pengujian.
* **Penanganan Teks dan Atribut Kategorikal**: Kolom `ocean_proximity` bersifat kategorikal (teks).
    * `OrdinalEncoder`: Mengonversi kategori teks menjadi angka ordinal (0, 1, 2, ...). Masalahnya, model mungkin berasumsi bahwa angka yang lebih tinggi berarti kategori yang "lebih baik" (misalnya, 2 lebih baik dari 1), padahal belum tentu demikian.
    * `OneHotEncoder`: Mengonversi setiap kategori unik menjadi kolom biner baru (1 jika kategori tersebut, 0 jika bukan). Ini adalah pendekatan yang lebih baik untuk mencegah masalah urutan ordinal. `OneHotEncoder` yang digunakan di sini mengembalikan *sparse matrix* untuk efisiensi.
* **Transformasi Kustom**: Pembuat *transformer* kustom (`CombinedAttributesAdder`) dapat dibuat untuk menggabungkan atribut dan menghasilkan fitur baru, seperti `rooms_per_household`. Ini memastikan bahwa transformasi ini dapat diaplikasikan secara konsisten ke data pelatihan dan pengujian.

* **Skala Fitur (Feature Scaling)**: Kolom numerik memiliki rentang nilai yang sangat berbeda, yang dapat memengaruhi kinerja beberapa algoritma *machine learning*.
    * **Min-Max Scaling (Normalization)**: Menggeser dan menskalakan nilai sehingga berada dalam rentang 0-1. Rentan terhadap *outlier*.
    * **Standardization (Z-score normalization)**: Menskalakan nilai sehingga memiliki mean 0 dan varians 1. Kurang terpengaruh oleh *outlier*.
    * `StandardScaler` dari `sklearn.preprocessing` dipilih dan diaplikasikan pada semua atribut numerik.

* **Pipeline Transformasi Numerik dan Kategorikal**: `ColumnTransformer` dari `sklearn.compose` digunakan untuk menerapkan transformasi yang berbeda pada kolom numerik dan kategorikal secara bersamaan dalam satu *pipeline* pra-pemrosesan. Ini memastikan alur kerja yang terorganisir dan dapat direproduksi.

---

## 6. Pemilihan dan Pelatihan Model (Select and Train a Model)

Setelah data siap, model-model *machine learning* dapat dilatih.

* **Model Regresi Linier (Linear Regression)**:
    * Model dasar yang diasumsikan sebagai titik awal.
    * `lin_reg = LinearRegression()` dilatih pada data pelatihan yang sudah diproses.
    * `mean_squared_error` dan `rmse` (Root Mean Squared Error) dihitung untuk mengevaluasi kinerja model.
    * `cross_val_score` juga digunakan untuk evaluasi *cross-validation* yang lebih robust.
* **Model Pohon Keputusan (Decision Tree Regressor)**:
    * Model non-parametrik yang dapat menangkap hubungan non-linier.
    * `tree_reg = DecisionTreeRegressor()` dilatih.
    * RMSE dan *cross-validation score* dihitung. Seringkali, model pohon keputusan cenderung *overfit* data pelatihan, yang bisa terlihat dari RMSE pelatihan yang rendah tetapi RMSE *cross-validation* yang lebih tinggi.
* **Model Random Forest (RandomForestRegressor)**:
    * Model ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi *overfitting*.
    * `forest_reg = RandomForestRegressor()` dilatih.
    * RMSE dan *cross-validation score* dihitung. *Random Forest* umumnya menunjukkan kinerja yang lebih baik daripada pohon keputusan tunggal.
    * Disimpan menggunakan `joblib` untuk penggunaan di masa mendatang.

---

## 7. Penyesuaian Halus Model (Fine-Tune Your Model)

Untuk mendapatkan kinerja terbaik dari model yang sudah dipilih, *fine-tuning* parameter model (hyperparameter) adalah langkah krusial.

* **Grid Search**: `GridSearchCV` dari `sklearn.model_selection` digunakan untuk mencari kombinasi *hyperparameter* terbaik secara sistematis.
    * Didefinisikan sebuah *dictionary* `param_grid` yang berisi *hyperparameter* yang ingin diuji beserta rentang nilainya.
    * `GridSearchCV` akan melatih model dengan setiap kombinasi *hyperparameter* yang mungkin dan mengevaluasinya menggunakan *cross-validation*.
    * `fit()` dijalankan pada data pelatihan untuk menemukan *best_params_` (kombinasi parameter terbaik) dan `best_estimator_` (model terbaik).
* **Analisis Pentingnya Fitur (Feature Importance)**: Untuk model seperti *Random Forest*, `feature_importances_` dapat digunakan untuk melihat atribut mana yang paling berkontribusi pada prediksi model. Ini dapat memberikan wawasan tentang data dan membantu dalam *feature selection* di masa depan.
* **Evaluasi pada Set Pengujian**: Setelah *fine-tuning* selesai dan model terbaik (`final_model`) ditemukan, model tersebut dievaluasi satu kali pada set pengujian (`X_test_prepared`, `y_test`) yang belum pernah dilihat model sebelumnya. Ini memberikan estimasi yang tidak bias tentang kinerja model pada data baru.
    * `mean_squared_error` dan `rmse` dihitung pada set pengujian.

---

## 8. Menggunakan Model (Use Your Model)

Model yang sudah dilatih dan disesuaikan dapat digunakan untuk membuat prediksi pada data baru.

* Contoh prediksi dilakukan pada beberapa instance baru, menunjukkan bagaimana model menghasilkan estimasi harga rumah.

---

## 9. Menyebarkan Model (Deploy Your Model)

Langkah terakhir dalam alur kerja *machine learning* adalah menyebarkan model sehingga dapat digunakan dalam aplikasi nyata.

* **Menyimpan Model**: Model yang sudah dilatih (`final_model`) disimpan ke disk menggunakan pustaka `joblib`. Ini memungkinkan model untuk dimuat kembali nanti tanpa perlu melatih ulang.
* **Memuat Model**: Model dapat dimuat kembali ke memori kapan saja untuk membuat prediksi baru.

---

**Kesimpulan:**

*Notebook* ini secara komprehensif mengilustrasikan siklus hidup proyek *machine learning*: mulai dari pemahaman awal data, pra-pemrosesan yang cermat (penanganan nilai hilang, kategorikal, *scaling*), pembagian data yang tepat, visualisasi untuk wawasan, pemilihan dan pelatihan berbagai model, *fine-tuning* *hyperparameter*, hingga evaluasi final dan penyebaran. Proyek ini menekankan pentingnya setiap langkah untuk membangun model prediktif yang robust dan akurat.
