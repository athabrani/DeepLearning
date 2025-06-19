# Pohon Keputusan (Decision Trees): Klasifikasi dan Regresi

*Notebook* ini membahas konsep dasar, implementasi, dan berbagai aspek dari algoritma Pohon Keputusan (Decision Trees) untuk tugas klasifikasi dan regresi. Pohon Keputusan adalah model yang intuitif dan mudah diinterpretasikan, yang bekerja dengan membagi ruang fitur secara rekursif menjadi wilayah-wilayah yang semakin homogen.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal adalah mengimpor pustaka yang diperlukan seperti `numpy`, `matplotlib.pyplot` untuk visualisasi, serta modul-modul `scikit-learn` yang relevan. Konfigurasi *plotting* juga diatur untuk memastikan konsistensi visual.

### Data Iris untuk Klasifikasi

* Dataset Iris digunakan sebagai contoh utama untuk tugas klasifikasi. Dataset ini berisi pengukuran bunga Iris (panjang/lebar kelopak dan mahkota) dari tiga spesies berbeda.
* Hanya dua fitur (panjang kelopak `petal length` dan lebar kelopak `petal width`) dan dua kelas (`Iris-Versicolor` dan `Iris-Virginica`) yang dipilih untuk penyederhanaan visualisasi klasifikasi biner.

---

## 2. Melatih dan Memvisualisasikan Pohon Keputusan untuk Klasifikasi

### A. Melatih `DecisionTreeClassifier`

* `DecisionTreeClassifier` dari `sklearn.tree` digunakan untuk melatih model klasifikasi pohon keputusan.
* `max_depth`: Parameter ini mengontrol kedalaman maksimum pohon. Membatasi kedalaman membantu mencegah *overfitting*.
* `fit(X, y)`: Model dilatih pada data fitur (`X`) dan label target (`y`).

### B. Memvisualisasikan Pohon Keputusan

* `export_graphviz`: Fungsi ini digunakan untuk menghasilkan file `.dot` yang merepresentasikan struktur pohon keputusan.
* File `.dot` ini kemudian dapat dikonversi menjadi gambar (misalnya `.png`) menggunakan Graphviz *software*. Visualisasi ini memungkinkan kita untuk melihat:
    * **Node Internal**: Merepresentasikan sebuah keputusan (kondisi fitur).
        * `gini`: Mengukur *impurity* Gini (ketidakmurnian) node. Node "murni" (semua instance di node tersebut termasuk dalam satu kelas) memiliki Gini = 0.
        * `samples`: Jumlah instance pelatihan yang mencapai node ini.
        * `value`: Distribusi jumlah instance per kelas di node ini.
        * `class`: Kelas mayoritas di node ini.
    * **Node Daun (Leaf Nodes)**: Merepresentasikan prediksi akhir.
* Pohon Keputusan bekerja dengan membagi data berdasarkan kondisi fitur yang paling informatif, bertujuan untuk menciptakan node-node yang isinya semakin seragam dalam hal kelas.

### C. Membuat Prediksi dan Memvisualisasikan Batas Keputusan

* Pohon keputusan dapat membuat prediksi untuk instance baru.
* Batas keputusan pohon divisualisasikan pada plot 2D. Pohon keputusan menciptakan batas keputusan berbentuk persegi panjang (atau *axis-aligned hyperplanes* di dimensi yang lebih tinggi). Wilayah-wilayah ini disebut **wilayah keputusan** (decision regions).

---

## 3. Sensitivitas Terhadap Rotasi dan Data Outlier

Pohon Keputusan memiliki beberapa kelemahan:

* **Sensitivitas Terhadap Rotasi**: Pohon Keputusan peka terhadap orientasi data. Jika data diputar sedikit, pohon dapat menghasilkan batas keputusan yang sangat berbeda dan kurang optimal, karena pembagiannya selalu sejajar dengan sumbu fitur. Ini membuat pohon keputusan cenderung kurang baik dalam menangani data yang tidak sejajar dengan sumbu utama.
* **Sensitivitas Terhadap Perubahan Kecil (Outlier)**: Model pohon keputusan bisa sangat sensitif terhadap perubahan kecil pada data pelatihan, terutama terhadap *outlier*. Menghilangkan satu *outlier* atau membuat perubahan kecil pada beberapa instance pelatihan dapat menghasilkan struktur pohon yang sangat berbeda. Ini karena perubahan tersebut dapat memengaruhi pemilihan *split* terbaik di node atas.

---

## 4. Analisis Impurity Gini vs. Entropy

`DecisionTreeClassifier` mendukung dua kriteria *impurity* (ketidakmurnian) untuk memecah node:

* **Gini Impurity (default)**: Mengukur seberapa sering instance yang dipilih secara acak dari subset akan salah diklasifikasikan jika instance tersebut secara acak diberi label sesuai dengan distribusi label dalam subset. Node "murni" memiliki Gini = 0.
    \[
    G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2
    \]
    Di mana $p_{i,k}$ adalah rasio instance kelas $k$ dalam node $i$.
* **Entropy**: Mengukur ketidakpastian atau "kekacauan" informasi. Pohon cenderung meminimalkan entropi (mencari *split* yang paling mengurangi ketidakpastian). Node "murni" memiliki Entropy = 0.
    \[
    H_i = - \sum_{k=1}^{n} p_{i,k} \log_2(p_{i,k})
    \]
* **Perbedaan Praktis**: Gini lebih cepat dihitung karena tidak melibatkan logaritma. Keduanya seringkali menghasilkan pohon yang sangat mirip, meskipun Gini cenderung mengisolasi kelas mayoritas di cabangnya, sementara Entropy cenderung menghasilkan pohon yang lebih seimbang.

---

## 5. Regularisasi (Regularization)

Pohon Keputusan cenderung mudah *overfit* data pelatihan karena fleksibilitasnya. Regularisasi digunakan untuk membatasi kompleksitas pohon dan meningkatkan generalisasinya.

* **`max_depth`**: Kedalaman maksimum pohon. Parameter regularisasi yang paling umum.
* **`min_samples_split`**: Jumlah minimum sampel yang diperlukan untuk memecah sebuah node.
* **`min_samples_leaf`**: Jumlah minimum sampel yang harus dimiliki sebuah node daun.
* **`max_leaf_nodes`**: Jumlah maksimum node daun.
* **`max_features`**: Jumlah fitur yang dipertimbangkan untuk setiap *split* di setiap node.

Melatih pohon dengan berbagai *hyperparameter* regularisasi (misalnya, `min_samples_leaf=4`) dan membandingkan *cross-validation score* (akurasi) menunjukkan bagaimana regularisasi dapat meningkatkan kinerja pada data yang tidak terlihat dengan mengurangi *overfitting*.

---

## 6. Regresi dengan Pohon Keputusan (`DecisionTreeRegressor`)

Pohon Keputusan juga dapat digunakan untuk tugas regresi. Alih-alih memprediksi kelas, mereka memprediksi nilai numerik.

* **Konsep**: Untuk regresi, pohon membagi data berdasarkan fitur hingga mencapai node daun. Prediksi untuk instance baru di node daun tersebut adalah nilai rata-rata dari semua instance pelatihan di node daun itu.
* **Melatih `DecisionTreeRegressor`**:
    * `DecisionTreeRegressor` dari `sklearn.tree` digunakan.
    * Parameter regularisasi seperti `max_depth` juga berlaku.
* **Visualisasi Regresi**: Plot regresi menunjukkan bagaimana pohon keputusan menciptakan prediksi bertahap (*step-wise*) atau konstan per wilayah keputusan.
    * *Overfitting* dalam regresi dapat terjadi ketika pohon terlalu dalam, menghasilkan prediksi yang sangat "bergelombang" dan mengikuti *noise* dalam data pelatihan.

---
