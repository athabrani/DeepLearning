# Reduksi Dimensi (Dimensionality Reduction): PCA, LLE, dan t-SNE

*Notebook* ini membahas konsep **Reduksi Dimensi**, sebuah teknik yang sangat penting dalam *machine learning* untuk mengurangi jumlah fitur (dimensi) dalam dataset sambil tetap mempertahankan informasi penting semaksimal mungkin. Reduksi dimensi membantu mengatasi **kutukan dimensi (curse of dimensionality)**, di mana kinerja model menurun dan kompleksitas komputasi meningkat seiring bertambahnya dimensi fitur.

Tujuan utama reduksi dimensi adalah:
1.  **Mengurangi *Noise***: Fitur-fitur yang tidak relevan atau *redundant* dapat dianggap sebagai *noise* yang mempersulit model untuk menemukan pola yang mendasarinya.
2.  **Visualisasi Data**: Data berdimensi tinggi sulit untuk divisualisasikan. Reduksi ke 2D atau 3D memungkinkan plot dan pemahaman visual.
3.  **Efisiensi Komputasi**: Model melatih lebih cepat dengan lebih sedikit fitur.
4.  **Mengurangi Kebutuhan Memori**: Menyimpan data berdimensi lebih rendah memerlukan lebih sedikit ruang.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal adalah mengimpor pustaka yang diperlukan seperti `numpy` dan `matplotlib.pyplot` untuk operasi numerik dan visualisasi, serta modul-modul `scikit-learn` yang relevan.

Dataset yang umum digunakan untuk demonstrasi reduksi dimensi adalah **MNIST**, yang terdiri dari 70.000 gambar tulisan tangan angka (0-9).
* `X` (fitur) memiliki bentuk `(70000, 784)`, artinya 70.000 gambar, masing-masing dengan 784 piksel (28x28).
* `y` (label) memiliki bentuk `(70000,)`, yang merupakan label angka 0-9 untuk setiap gambar.
* Dataset ini dibagi menjadi set pelatihan (60.000 sampel) dan set pengujian (10.000 sampel).

---

## 2. Proyeksi (Projection) vs. Manifold Learning

Reduksi dimensi dapat dilakukan melalui dua pendekatan utama:

* **Proyeksi (Projection)**: Jika data berdimensi tinggi sebenarnya terletak dekat dengan *subspace* berdimensi lebih rendah (misalnya, data 3D yang hampir datar seperti *swiss roll*), maka data dapat diproyeksikan ke *subspace* tersebut. Pendekatan ini adalah yang paling langsung. **PCA** adalah contoh algoritma proyeksi.
* **Manifold Learning**: Banyak algoritma reduksi dimensi non-linier menggunakan pendekatan ini. Mereka mengasumsikan bahwa data berdimensi tinggi terletak pada *manifold* berdimensi rendah yang "tertekuk" atau "terpelintir" dalam ruang dimensi yang lebih tinggi. Algoritma ini mencoba "membuka gulungan" *manifold* tersebut. **LLE** dan **t-SNE** adalah contoh algoritma *manifold learning*.

---

## 3. Analisis Komponen Utama (Principal Component Analysis - PCA)

PCA adalah algoritma reduksi dimensi paling populer. Ini bekerja dengan mengidentifikasi *hyperplane* (atau *principal components*) yang paling mendekati data. PCA memproyeksikan data ke *hyperplane* ini untuk mengurangi dimensi.

### A. Mempertahankan Varian (Preserving Variance)

* PCA memilih sumbu yang mempertahankan varians data sebanyak mungkin. Varians menunjukkan seberapa "tersebar" data di sepanjang sumbu tertentu; sumbu dengan varians tinggi mengandung lebih banyak informasi.
* *Principal Components* (PC) adalah sumbu-sumbu ortogonal yang ditemukan oleh PCA. PC pertama adalah sumbu yang mempertahankan varians terbesar, PC kedua mempertahankan varians terbesar yang ortogonal terhadap PC pertama, dan seterusnya.

### B. Implementasi PCA di Scikit-learn

* `PCA` dari `sklearn.decomposition`:
    * **Menentukan Dimensi Target (`n_components`)**: Anda dapat menentukan jumlah dimensi yang ingin Anda kurangi.
        * Misalnya, `pca = PCA(n_components=2)` akan mengurangi data ke 2 dimensi.
        * `pca.fit_transform(X_train)`: Melatih PCA pada data pelatihan dan mengubahnya.
    * **Menentukan Varian yang Dijelaskan**: Alternatifnya, Anda dapat menentukan proporsi varians yang ingin dipertahankan (misalnya, `pca = PCA(n_components=0.95)` akan memilih jumlah dimensi minimum yang mempertahankan 95% varians).
        * `pca.explained_variance_ratio_`: Atribut ini memberikan rasio varians yang dijelaskan oleh setiap *Principal Component* individual.
        * Anda dapat memplot varians kumulatif untuk memutuskan berapa banyak dimensi yang harus dipertahankan.
    * **Kompresi Data**: Setelah PCA melatih, Anda dapat menggunakan `pca.inverse_transform()` untuk mengembalikan data ke dimensi aslinya, dengan hilangnya informasi minimal. Ini berguna untuk *denoising* atau kompresi.

### C. PCA Inkremental (Incremental PCA)

* `IncrementalPCA` dari `sklearn.decomposition`:
    * Digunakan untuk dataset yang sangat besar yang tidak dapat ditampung seluruhnya dalam memori.
    * Ini membagi dataset menjadi mini-batch dan melatih PCA secara bertahap pada setiap mini-batch.

### D. PCA Acak (Randomized PCA)

* `RandomizedPCA` (sekarang bagian dari `PCA` dengan `svd_solver='randomized'`) adalah metode PCA yang lebih cepat untuk mengurangi dimensi ke sejumlah kecil dimensi target, karena menggunakan algoritma stokastik untuk menemukan *Principal Components* yang baik secara aproksimasi.

---

## 4. Pemetaan Proyeksi Linier Lokal (Locally Linear Embedding - LLE)

LLE adalah algoritma *manifold learning* non-linier. Ia bekerja dengan mengidentifikasi bagaimana setiap instance pelatihan secara linier bergantung pada tetangga terdekatnya. Kemudian, ia mencari representasi berdimensi rendah dari data di mana hubungan linier lokal ini dipertahankan.

* `LocallyLinearEmbedding` dari `sklearn.manifold`:
    * `n_components`: Dimensi target.
    * `n_neighbors`: Jumlah tetangga terdekat yang dipertimbangkan untuk setiap instance.
* **Keunggulan**: LLE dapat "membuka gulungan" *manifold* yang rumit, seperti contoh *Swiss Roll*.
* **Keterbatasan**: LLE mungkin tidak bekerja dengan baik jika hubungan linier lokal tidak berlaku atau jika dataset sangat berisik.

---

## 5. t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE adalah algoritma reduksi dimensi non-linier, yang sangat cocok untuk **visualisasi data berdimensi tinggi** dengan memetakan titik data ke ruang 2D atau 3D. Ia berfokus pada pelestarian struktur tetangga lokal dan global data.

* `TSNE` dari `sklearn.manifold`:
    * `n_components`: Biasanya 2 atau 3 untuk tujuan visualisasi.
    * `perplexity`: Parameter kunci yang mengontrol bagaimana t-SNE menyeimbangkan perhatian terhadap tetangga lokal dan global. Ini dapat dianggap sebagai perkiraan jumlah tetangga terdekat yang dipertimbangkan untuk setiap titik.
    * `n_iter`: Jumlah iterasi untuk optimasi.
* **Keunggulan**: t-SNE menghasilkan plot yang sering kali mengungkapkan *cluster* atau struktur data yang tersembunyi. Ini sangat baik untuk memvisualisasikan data yang kompleks.
* **Keterbatasan**: t-SNE relatif lambat untuk dataset yang sangat besar dan sangat sensitif terhadap *hyperparameter* `perplexity`. Hasilnya juga bersifat stokastik, artinya menjalankan t-SNE beberapa kali dapat menghasilkan plot yang sedikit berbeda. Ini **bukan** algoritma yang baik untuk reduksi dimensi yang digunakan sebagai langkah *preprocessing* untuk model *machine learning* selanjutnya, karena sifatnya yang non-deterministik dan lambat.

---
