# Ensemble Learning dan Random Forests: Meningkatkan Akurasi Model

*Notebook* ini membahas konsep **Ensemble Learning**, sebuah teknik dalam *machine learning* yang menggabungkan prediksi dari beberapa model dasar (disebut *estimators* atau *weak learners*) untuk mencapai kinerja prediktif yang lebih baik dan lebih *robust* daripada model tunggal. Fokus utama adalah pada algoritma **Random Forest**, yang merupakan salah satu metode *ensemble* paling populer dan efektif.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal melibatkan impor pustaka standar seperti `numpy` dan `matplotlib` untuk operasi numerik dan visualisasi, serta modul-modul `scikit-learn` yang relevan. Konfigurasi *plotting* juga diatur untuk memastikan konsistensi visual.

Dataset MNIST, yang terdiri dari gambar tulisan tangan angka (0-9), digunakan sebagai data contoh untuk klasifikasi. Data ini dibagi menjadi set pelatihan (60.000 gambar) dan set pengujian (10.000 gambar) untuk evaluasi model yang tidak bias.

---

## 2. Voting Classifiers (Hard Voting)

*Voting Classifier* adalah salah satu metode *ensemble* paling sederhana. Ini bekerja dengan melatih beberapa *classifier* yang berbeda pada set pelatihan yang sama dan kemudian memprediksi kelas yang paling banyak "dipilih" oleh *classifier* individual.

* **Implementasi**:
    * Beberapa *classifier* yang berbeda dipilih: `LogisticRegression`, `RandomForestClassifier`, dan `SVC`. Penting untuk menggunakan *classifier* yang beragam (bekerja dengan prinsip yang berbeda) untuk mendapatkan manfaat *ensemble* yang maksimal.
    * `VotingClassifier` dari `sklearn.ensemble` digunakan untuk menggabungkan mereka.
    * `voting="hard"`: Ini berarti *voting* dilakukan berdasarkan prediksi kelas yang paling sering muncul (mayoritas voting). Misalnya, jika tiga *classifier* memprediksi [1, 1, 0], maka hasil akhirnya adalah 1.
* **Evaluasi**: Akurasi dari `VotingClassifier` dan setiap *classifier* individual (yang dilatih secara terpisah) dibandingkan. Umumnya, *Voting Classifier* akan mengungguli *classifier* individual terbaiknya, terutama jika *classifier* individualnya cukup baik dan beragam.

---

## 3. Bagging dan Pasting Ensembles

Bagging (Bootstrap Aggregating) dan Pasting adalah metode *ensemble* di mana *classifier* individual dilatih pada subset yang berbeda dari set pelatihan.

* **Bagging**: Subset pelatihan diambil dengan *replacement* (bootstrap samples). Ini berarti satu instance pelatihan dapat diambil beberapa kali untuk subset yang sama.
* **Pasting**: Subset pelatihan diambil tanpa *replacement*.
* **Implementasi**:
    * `BaggingClassifier` dari `sklearn.ensemble` digunakan.
    * `base_estimator`: Model dasar yang akan dilatih (misalnya, `DecisionTreeClassifier`).
    * `n_estimators`: Jumlah *estimators* dalam *ensemble*.
    * `max_samples`: Jumlah sampel yang diambil untuk setiap *estimator*.
    * `bootstrap=True`: Untuk Bagging (default).
    * `bootstrap=False`: Untuk Pasting.
* **Manfaat**: Kedua metode ini membantu mengurangi *variance* (overfitting) dari model dasar dengan melatihnya pada subset data yang sedikit berbeda. Prediksi akhir adalah rata-rata (untuk regresi) atau voting (untuk klasifikasi) dari semua *estimators*.

---

## 4. Random Forests

**Random Forest** adalah jenis *ensemble* **Bagging** yang sangat populer, di mana *base estimators*-nya adalah **Pohon Keputusan** (`DecisionTreeClassifier`). Random Forest menambahkan lapisan *randomness* ekstra saat melatih setiap pohon:

* **Random Subspace (Feature Randomness)**: Saat mencari *split* terbaik pada setiap node pohon, algoritma Random Forest hanya mempertimbangkan subset acak dari fitur yang tersedia, bukan semua fitur. Ini membuat pohon-pohon menjadi lebih beragam.
* **Bootstrap Aggregating**: Setiap pohon dilatih pada subset pelatihan yang diambil dengan *replacement*.

* **Implementasi**:
    * `RandomForestClassifier` dari `sklearn.ensemble` digunakan.
    * `n_estimators`: Jumlah pohon dalam hutan.
    * `max_leaf_nodes` atau `max_depth`: Digunakan untuk mengatur regularisasi pohon individu.
    * `n_jobs`: Untuk pemrosesan paralel (jika tersedia).
* **Manfaat**: Random Forest sangat efektif karena:
    * Mengurangi *variance* secara signifikan dibandingkan pohon keputusan tunggal.
    * Kurang rentan terhadap *overfitting* daripada BaggingClassifier dengan DecisionTree, karena adanya *feature randomness*.
    * Umumnya sangat akurat dan robust.

### Out-of-Bag (OOB) Evaluation

Karena setiap *estimator* dalam BaggingClassifier (termasuk Random Forest) dilatih hanya pada subset pelatihan yang diambil secara acak, ada sebagian instance pelatihan yang tidak pernah terlihat oleh *estimator* tertentu. Instance ini disebut "out-of-bag" (OOB) instance.

* `oob_score=True`: Mengaktifkan evaluasi OOB.
* Model dapat dievaluasi pada OOB instance tanpa perlu set validasi terpisah, memberikan estimasi yang cukup akurat tentang kinerja generalisasi model.

### Feature Importance (Pentingnya Fitur)

Random Forest dapat mengukur pentingnya setiap fitur dengan melihat seberapa besar penurunan *impurity* (atau *error*) yang dibawa oleh fitur tersebut dalam semua pohon di hutan.

* `feature_importances_`: Atribut yang menyimpan skor pentingnya fitur.
* Ini berguna untuk *feature selection* dan memahami data.

---

## 5. Boosting

Boosting adalah teknik *ensemble* lain yang populer, di mana *estimators* dilatih secara berurutan. Setiap *estimator* mencoba memperbaiki kesalahan yang dibuat oleh *estimator* sebelumnya.

### A. AdaBoost (Adaptive Boosting)

* `AdaBoostClassifier` dari `sklearn.ensemble` digunakan.
* Setiap *weak learner* dilatih pada versi *weighted* dari dataset asli. Instance yang salah diklasifikasikan oleh *estimator* sebelumnya diberikan bobot yang lebih tinggi agar *estimator* berikutnya lebih fokus pada mereka.
* Prediksi akhir adalah *weighted majority voting*.

### B. Gradient Boosting

* `GradientBoostingClassifier` (untuk klasifikasi) dan `GradientBoostingRegressor` (untuk regresi) dari `sklearn.ensemble` digunakan.
* Boosting ini bekerja dengan melatih *weak learners* (biasanya pohon keputusan) secara berurutan. Setiap pohon dilatih untuk memprediksi "residual" (sisa kesalahan) dari prediksi *ensemble* sebelumnya, bukan bobot ulang instance.
* **`learning_rate`**: Mengontrol seberapa besar kontribusi setiap pohon baru. Nilai kecil berarti Anda memerlukan lebih banyak pohon tetapi bisa mendapatkan *ensemble* yang lebih *robust*.
* **`n_estimators`**: Jumlah tahapan boosting (jumlah pohon).
* **`subsample`**: Mengontrol proporsi sampel yang diambil untuk setiap pohon (membangun pohon pada subset acak dari data).

---

## 6. Stacking (Stacked Generalization)

Stacking adalah metode *ensemble* di mana *final predictor* (disebut *blender* atau *meta-learner*) dilatih untuk menggabungkan prediksi dari beberapa *estimators* dasar.

* *Blender* belajar kapan harus mempercayai satu *estimator* lebih dari yang lain, atau bagaimana menggabungkan prediksi mereka secara optimal.
* Proses ini biasanya melibatkan *cross-validation* untuk menghasilkan prediksi "out-of-fold" dari *estimators* dasar, yang kemudian menjadi input untuk *blender*.

---
