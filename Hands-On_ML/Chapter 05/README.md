# Melatih Model Linier: Regresi Linier, Polinomial, Ridge, Lasso, dan Elastic Net

*Notebook* ini membahas berbagai metode untuk melatih model regresi linier, mulai dari konsep dasar hingga teknik regularisasi lanjutan. Fokusnya adalah pada pemahaman matematis dan implementasi praktis model-model ini menggunakan pustaka `scikit-learn` dan `numpy`.

---

## 1. Persiapan Lingkungan dan Data

Langkah awal dalam *notebook* ini adalah mengimpor pustaka yang diperlukan seperti `numpy` dan `matplotlib.pyplot` untuk operasi numerik dan visualisasi, serta modul-modul `sklearn` yang relevan. Konfigurasi *plotting* juga diatur untuk memastikan konsistensi visual.

Dataset contoh dibuat menggunakan fungsi `np.random.rand()` untuk menghasilkan data yang bising (`X` dan `y`) dengan hubungan linier dasar yang ditambahkan *noise* acak.

\[
y = a \cdot x + b + \text{noise}
\]

---

## 2. Regresi Linier Standar

Regresi Linier adalah salah satu model prediktif paling fundamental, yang bertujuan untuk menemukan hubungan linier antara satu atau lebih fitur independen ($X$) dan variabel target ($y$).

### A. Persamaan Normal

Solusi *least squares* untuk Regresi Linier dapat ditemukan secara analitis menggunakan **Persamaan Normal** (Normal Equation):

\[
\hat{\boldsymbol{\theta}} = (\mathbf{X}^{\text{T}} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} \mathbf{y}
\]

Di mana:
- $\hat{\boldsymbol{\theta}}$ adalah vektor parameter model (termasuk *bias* $\theta_0$ dan *feature weights* $\theta_1, \theta_2, \ldots, \theta_n$).
- $\mathbf{X}$ adalah matriks fitur, dengan satu kolom bias yang berisi semua angka 1 di depannya.
- $\mathbf{X}^{\text{T}}$ adalah transpose dari $\mathbf{X}$.
- $\mathbf{y}$ adalah vektor target.

Implementasi `numpy.linalg.inv()` dan `numpy.dot()` digunakan untuk menghitung solusi ini secara langsung.

### B. `LinearRegression` dari Scikit-Learn

`scikit-learn` menyediakan kelas `LinearRegression` yang mengimplementasikan Persamaan Normal.

* `lin_reg = LinearRegression()`: Menginisialisasi model.
* `lin_reg.fit(X, y)`: Melatih model pada data.
* `lin_reg.intercept_`: Mengakses nilai bias ($\theta_0$).
* `lin_reg.coef_`: Mengakses koefisien fitur ($\theta_1, \ldots, \theta_n$).

Model ini kemudian digunakan untuk membuat prediksi pada data baru.

---

## 3. Regresi Polinomial

Regresi Polinomial memungkinkan model linier untuk menyesuaikan diri dengan data non-linier dengan menambahkan pangkat fitur sebagai fitur baru.

* `PolynomialFeatures(degree=d)`: Mengubah matriks fitur menjadi matriks yang diperluas dengan menambahkan fitur polinomial hingga derajat $d$. Misalnya, jika `degree=2` dan fitur input adalah $a$ dan $b$, maka fitur yang dihasilkan adalah $1, a, b, a^2, b^2, ab$.
* *Pipeline*: Sebuah *pipeline* sering digunakan untuk menggabungkan `PolynomialFeatures` dengan `LinearRegression` secara berurutan.

Analisis dilakukan untuk mengamati bagaimana model polinomial dengan derajat berbeda (*underfitting* vs. *overfitting*) menyesuaikan diri dengan data.

---

## 4. Kurva Pembelajaran (Learning Curves)

Kurva pembelajaran memplot kinerja model (misalnya, RMSE) terhadap ukuran set pelatihan. Ini membantu mendiagnosis *underfitting* dan *overfitting*.

* **Untuk Model *Underfitting***: Baik *error* pada set pelatihan maupun *error* pada set validasi akan tinggi dan relatif datar. Menambah data pelatihan tidak banyak membantu, karena model terlalu sederhana.
* **Untuk Model *Overfitting***: *Error* pada set pelatihan akan rendah, tetapi *error* pada set validasi akan jauh lebih tinggi. Ada celah signifikan antara kedua kurva. Menambah data pelatihan dapat membantu mengurangi *overfitting*.
* **Untuk Model yang Baik**: *Error* pada kedua set akan rendah, dan kurva akan menyatu.

---

## 5. Regresi Regularisasi

Teknik regularisasi digunakan untuk mengurangi *overfitting* dengan menambahkan penalti pada ukuran koefisien model, mendorong model untuk menjadi lebih sederhana.

### A. Ridge Regression

*Ridge Regression* menambahkan penalti $L_2$ (norma Euclidean kuadrat) ke fungsi biaya:

\[
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} \theta_i^2
\]

Di mana:
- $\alpha$ adalah *hyperparameter* yang mengontrol kekuatan regularisasi. $\alpha=0$ adalah Regresi Linier biasa.
- Penalti diterapkan pada koefisien fitur, tidak termasuk bias ($\theta_0$).

Implementasi `Ridge` dari `sklearn.linear_model` digunakan. Penggunaan *scaling* sangat penting sebelum menerapkan *Ridge regression*, karena *Ridge* sensitif terhadap skala fitur.

### B. Lasso Regression (Least Absolute Shrinkage and Selection Operator Regression)

*Lasso Regression* menambahkan penalti $L_1$ (norma Manhattan) ke fungsi biaya:

\[
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} |\theta_i|
\]

Ciri khas Lasso adalah kemampuannya untuk melakukan **seleksi fitur** (feature selection) secara otomatis, karena cenderung mendorong *feature weights* yang tidak penting menjadi nol.

Implementasi `Lasso` dari `sklearn.linear_model` digunakan.

### C. Elastic Net

*Elastic Net* adalah gabungan dari *Ridge* dan *Lasso*, menambahkan penalti $L_1$ dan $L_2$ ke fungsi biaya:

\[
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1 - r}{2} \alpha \sum_{i=1}^{n} \theta_i^2
\]

Di mana:
- $r$ adalah *mix ratio* antara penalti $L_1$ dan $L_2$. $r=1$ adalah Lasso, $r=0$ adalah Ridge.

*Elastic Net* umumnya dipilih ketika ada keraguan antara *Ridge* dan *Lasso*, karena menggabungkan kekuatan seleksi fitur Lasso dengan stabilitas Ridge. Implementasi `ElasticNet` dari `sklearn.linear_model` digunakan.

---

## 6. Regresi Linier Stokastik (Stochastic Gradient Descent - SGD)

Gradient Descent adalah algoritma optimasi umum yang dapat digunakan untuk melatih berbagai model *machine learning*, termasuk Regresi Linier, dengan secara iteratif menyesuaikan parameter model untuk meminimalkan fungsi biaya.

### A. Konsep Umum Gradient Descent

Gradient Descent bekerja dengan menghitung gradien (turunan parsial terhadap setiap parameter) dari fungsi biaya relatif terhadap parameter model, dan kemudian bergerak ke arah yang berlawanan dengan gradien untuk menemukan minimum.

\[
\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_{\text{old}})
\]

Di mana:
- $\eta$ adalah *learning rate* (tingkat pembelajaran), yang mengontrol ukuran langkah.
- $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ adalah vektor gradien fungsi biaya.

### B. Batch Gradient Descent

Setiap langkah dalam *Batch Gradient Descent* melibatkan perhitungan gradien pada seluruh set pelatihan. Ini bisa lambat untuk dataset besar.

### C. Stochastic Gradient Descent (SGD)

*Stochastic Gradient Descent* (SGD) menghitung gradien hanya pada satu instance pelatihan yang dipilih secara acak di setiap langkah. Ini jauh lebih cepat untuk dataset besar, tetapi jalannya lebih "berisik" dan mungkin tidak langsung menuju minimum.

* `SGDRegressor` dari `sklearn.linear_model` adalah implementasi Regresi Linier menggunakan SGD.
* *Hyperparameter* seperti `max_iter` (jumlah epoch), `tol` (toleransi konvergensi), `penalty` (jenis regularisasi), dan `eta0` (learning rate awal) dapat disesuaikan.

### D. Mini-batch Gradient Descent

Mini-batch GD menghitung gradien pada subset kecil (mini-batch) dari set pelatihan. Ini adalah kompromi antara *Batch GD* (stabil tetapi lambat) dan *SGD* (cepat tetapi bising).

---

**Kesimpulan:**

*Notebook* ini menyediakan fondasi yang kuat dalam pemodelan regresi linier, mulai dari solusi analitis hingga metode optimasi iteratif. Pemahaman tentang berbagai bentuk regularisasi (Ridge, Lasso, Elastic Net) dan algoritma Gradient Descent (Batch, Stochastic, Mini-batch) adalah kunci untuk membangun model yang *robust* dan *generalizable* dalam menghadapi tantangan *underfitting* dan *overfitting*.
