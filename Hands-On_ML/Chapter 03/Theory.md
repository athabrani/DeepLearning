# Klasifikasi Angka MNIST: Mengukur Performa Model Binary dan Multiclass

*Notebook* ini mengeksplorasi konsep dasar klasifikasi menggunakan dataset MNIST yang terkenal, yang terdiri dari 70.000 gambar tulisan tangan angka (0-9). Fokus utamanya adalah melatih *classifier* untuk mengidentifikasi angka '5' (klasifikasi biner) dan kemudian memperluasnya ke klasifikasi *multiclass* (membedakan semua 10 angka). Selain itu, *notebook* ini membahas berbagai metrik evaluasi kinerja model dan teknik analisis kesalahan.

---

## 1. Persiapan Data dan Lingkungan

Dataset MNIST diunduh menggunakan `fetch_openml` dari `scikit-learn`. Dataset ini terdiri dari dua bagian utama:
- `X`: Matriks fitur dengan 70.000 gambar, di mana setiap gambar direpresentasikan sebagai *array* 784 piksel (28x28 piksel).
- `y`: Vektor target yang berisi label angka (0-9) untuk setiap gambar.

Sebagian kecil dari dataset ditampilkan untuk memvisualisasikan beberapa contoh angka tulisan tangan. Dataset kemudian dibagi menjadi set pelatihan (60.000 gambar) dan set pengujian (10.000 gambar) untuk evaluasi model yang tidak bias. Label target (`y`) dikonversi ke tipe data `uint8` untuk konsistensi.

---

## 2. Klasifikasi Biner: Mendeteksi Angka '5'

Langkah pertama adalah membuat *classifier* biner yang hanya akan membedakan antara angka '5' dan bukan angka '5'.
- Label baru `y_train_5` dan `y_test_5` dibuat, di mana `True` menunjukkan angka '5' dan `False` menunjukkan bukan angka '5'.

### `SGDClassifier`

Model `SGDClassifier` (Stochastic Gradient Descent classifier) digunakan untuk tugas klasifikasi biner ini. `SGDClassifier` adalah *linear classifier* yang sangat efisien untuk dataset besar karena melatih model secara *stochastically* (secara bertahap pada satu atau sekelompok kecil sampel).

Model dilatih pada `X_train` dan `y_train_5`. Setelah pelatihan, model dapat memprediksi apakah suatu digit adalah '5' atau bukan.

### Pengukuran Performa

Meskipun akurasi adalah metrik yang umum, untuk *binary classification*, terutama pada dataset yang tidak seimbang (seperti MNIST di mana angka '5' hanya sekitar 10% dari total data), akurasi bisa menjadi metrik yang menyesatkan. Oleh karena itu, *notebook* ini menekankan metrik performa lainnya:

#### A. Validasi Silang (Cross-Validation)

Validasi silang digunakan untuk mengevaluasi kinerja model secara lebih andal daripada hanya menggunakan satu pembagian *train-test*.
- `cross_val_score`: Menghitung akurasi model di beberapa *fold* (bagian) data. Hasil akurasi untuk `SGDClassifier` menunjukkan kinerja yang cukup tinggi (sekitar 95-96%).
- `StratifiedKFold`: Digunakan untuk memastikan bahwa setiap *fold* memiliki proporsi kelas yang sama dengan seluruh dataset, yang sangat penting untuk dataset tidak seimbang. Implementasi manual validasi silang ini juga menunjukkan akurasi yang serupa.

**Catatan tentang Akurasi yang Menyesatkan:** Sebuah *dummy classifier* (`Never5Classifier`) yang selalu memprediksi 'bukan 5' juga dievaluasi. Meskipun *classifier* ini tidak pernah mendeteksi angka '5', akurasinya masih tinggi (sekitar 90%) karena hanya sekitar 10% dari data adalah angka '5'. Ini menunjukkan mengapa akurasi saja tidak cukup untuk evaluasi *binary classifier* pada data tidak seimbang.

#### B. Matriks Konfusi (Confusion Matrix)

Matriks konfusi memberikan gambaran yang lebih detail tentang kesalahan yang dibuat *classifier*.
- `cross_val_predict`: Digunakan untuk mendapatkan prediksi dari *cross-validation* pada set pelatihan.
- Matriks konfusi dihitung menggunakan `confusion_matrix(y_train_5, y_train_pred)`.

Interpretasi Matriks Konfusi:
\[
\begin{pmatrix}
TN & FP \\
FN & TP
\end{pmatrix}
\]
- **True Negative (TN)**: Diprediksi `False` (bukan 5) dan memang `False`.
- **False Positive (FP)**: Diprediksi `True` (angka 5) tetapi sebenarnya `False`. (Error Tipe I)
- **False Negative (FN)**: Diprediksi `False` (bukan 5) tetapi sebenarnya `True`. (Error Tipe II)
- **True Positive (TP)**: Diprediksi `True` (angka 5) dan memang `True`.

Membandingkan matriks konfusi `SGDClassifier` dengan *perfect classifier* (yang tidak memiliki FP dan FN) memberikan pemahaman yang lebih baik tentang area kesalahan.

#### C. Presisi (Precision), Rekal (Recall), dan F1-Score

Metrik ini lebih informatif untuk klasifikasi biner, terutama pada kelas minoritas.
- **Presisi**: Akurasi prediksi positif.
  \[
  Precision = \frac{TP}{TP + FP}
  \]
  Presisi `SGDClassifier` sekitar 83.7%, yang berarti ketika model memprediksi '5', ia benar sekitar 83.7% dari waktu.
- **Rekal (Recall / Sensitivity)**: Proporsi positif aktual yang teridentifikasi dengan benar.
  \[
  Recall = \frac{TP}{TP + FN}
  \]
  Rekal `SGDClassifier` sekitar 65.1%, yang berarti model hanya mendeteksi sekitar 65.1% dari semua angka '5' yang sebenarnya ada.
- **F1-Score**: Rata-rata harmonis dari presisi dan rekal. Ini adalah metrik yang baik untuk membandingkan *classifier* jika presisi dan rekal sama pentingnya.
  \[
  F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
  \]
  F1-Score `SGDClassifier` adalah 0.732.

#### D. Kurva Presisi/Rekal (Precision/Recall Curve)

* `SGDClassifier` dapat memberikan skor kepercayaan untuk setiap instance, bukan hanya prediksi biner. `decision_function` mengembalikan skor ini.
* Kurva Presisi/Rekal menunjukkan *trade-off* antara presisi dan rekal untuk berbagai *threshold* (ambang batas).
  - Menaikkan *threshold* akan meningkatkan presisi (lebih sedikit FP) tetapi menurunkan rekal (lebih banyak FN).
  - Menurunkan *threshold* akan meningkatkan rekal (lebih sedikit FN) tetapi menurunkan presisi (lebih banyak FP).
* Plot ini membantu memilih *threshold* yang sesuai dengan kebutuhan bisnis (misalnya, untuk mencapai presisi 90%, rekalnya akan menjadi sekitar 48%).

#### E. Kurva ROC (Receiver Operating Characteristic)

Kurva ROC memplot *True Positive Rate* (Rekal) terhadap *False Positive Rate* (FPR) untuk berbagai *threshold*.
- **FPR** adalah rasio instance negatif yang salah diklasifikasikan sebagai positif.
  \[
  FPR = \frac{FP}{FP + TN}
  \]
- **AUC (Area Under the Curve)**: Mengukur area di bawah kurva ROC. AUC yang sempurna adalah 1, sedangkan *random classifier* memiliki AUC 0.5.
  - `SGDClassifier` mencapai AUC 0.960.
  - `RandomForestClassifier` (model ensemble) juga dievaluasi dan menunjukkan kinerja yang jauh lebih baik (AUC 0.998), dengan kurva ROC yang mendekati pojok kiri atas, menunjukkan bahwa ia membuat lebih sedikit kesalahan.

---

## 3. Klasifikasi *Multiclass*

Untuk klasifikasi *multiclass* (membedakan antara 0-9), `scikit-learn` secara otomatis menggunakan strategi *One-vs-Rest* (OvR) atau *One-vs-One* (OvO) ketika melatih *binary classifier* (seperti `SVC` atau `SGDClassifier`) pada dataset *multiclass*.

- **OvR (One-vs-Rest)**: Melatih 10 *binary classifier*, satu untuk setiap angka (misalnya, '0' vs. 'bukan 0', '1' vs. 'bukan 1', dst.). Kelas dengan skor tertinggi yang akan dipilih. `OneVsRestClassifier` adalah kelas *wrapper* yang secara eksplisit menerapkan strategi ini.
- **OvO (One-vs-One)**: Melatih *binary classifier* untuk setiap pasangan kelas (misalnya, '0' vs. '1', '0' vs. '2', dst.). Untuk 10 kelas, ini berarti $10 \times 9 / 2 = 45$ *classifier*. Kelas yang paling sering "menang" dalam pertarungan *pair-wise* yang akan dipilih. OvO lebih disukai untuk algoritma yang *scale poorly* dengan ukuran dataset (misalnya, `SVC`), karena melatih banyak *classifier* pada subset data yang lebih kecil.

### Evaluasi Model *Multiclass*

- `SVC` (Support Vector Machine classifier) dilatih pada sebagian kecil data dan dapat memprediksi angka.
- `sgd_clf` (setelah dilatih ulang pada `y_train` yang *multiclass*) juga dievaluasi akurasinya menggunakan *cross-validation*.
- **Scaling Fitur**: Ketika `StandardScaler` digunakan untuk menskalakan fitur sebelum melatih `SGDClassifier` *multiclass*, akurasi model meningkat secara signifikan (dari sekitar 87% menjadi 89-90%). *Scaling* adalah langkah *preprocessing* yang penting untuk banyak algoritma *machine learning*, terutama yang sensitif terhadap skala fitur.

---

## 4. Analisis Kesalahan (Error Analysis)

Matriks konfusi dapat digunakan untuk menganalisis jenis kesalahan yang dibuat oleh *multiclass classifier*.
- Normalisasi matriks konfusi (membagi setiap nilai dengan jumlah total instance di baris yang sesuai) membantu fokus pada *error rates* (tingkat kesalahan) daripada jumlah absolut.
- Plot matriks konfusi yang dinormalisasi dengan diagonalnya diset ke nol (untuk mengabaikan prediksi yang benar) akan menyoroti kelas-kelas di mana model cenderung salah memprediksi atau salah memprediksikan kelas lain.
  - Misalnya, angka '8' dan '9' sering bingung satu sama lain, atau '3' dan '5' sering bingung. Analisis visual ini dapat menginspirasi ide-ide untuk *preprocessing* data tambahan atau *feature engineering* (misalnya, menambah fitur untuk mendeteksi lingkaran).

---

## 5. Klasifikasi *Multilabel*

Dalam tugas *multilabel classification*, sebuah instance dapat memiliki beberapa label secara bersamaan.
- Contoh: Klasifikasi angka yang memprediksi apakah angka tersebut 'besar' (>=7) DAN 'ganjil'.
- `KNeighborsClassifier` adalah salah satu algoritma yang dapat menangani *multilabel classification* secara langsung. Ia dilatih pada `y_multilabel` (yang berisi dua label biner untuk setiap instance).
- `f1_score` dengan `average="macro"` digunakan untuk mengevaluasi *multilabel classifier*. *Macro average* menghitung F1-score untuk setiap label secara terpisah, lalu mengambil rata-rata, memberikan bobot yang sama untuk semua label.

---

## 6. Klasifikasi *Multioutput* (Denoising)

Klasifikasi *multioutput* adalah generalisasi *multilabel classification* di mana setiap label adalah *multiclass* (bukan hanya biner). Ini juga dapat disebut sebagai regresi *multioutput*.
- Contoh: Membersihkan gambar *digit* yang bising (*denoising*).
  - Gambar input adalah *noisy digit* (digit dengan *noise* acak).
  - Output yang diinginkan adalah *clean digit* (digit asli tanpa *noise*).
- Setiap piksel pada *clean digit* adalah label *multiclass* (nilai piksel dari 0 hingga 255). Jadi, *classifier* harus menghasilkan 784 label, di mana setiap label dapat memiliki 256 kemungkinan nilai.
- `KNeighborsClassifier` juga digunakan untuk tugas ini, dilatih pada *noisy images* sebagai input dan *clean images* sebagai output.
- Visualisasi menunjukkan bahwa model berhasil membersihkan *noisy digit* menjadi *clean digit*.
