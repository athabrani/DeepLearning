# Natural Language Processing (NLP) dengan RNNs dan Mekanisme Atensi (Attention)

*Notebook* ini membahas penerapan **Recurrent Neural Networks (RNNs)** untuk tugas-tugas **Natural Language Processing (NLP)**, khususnya terjemahan mesin, dan memperkenalkan konsep penting **Mekanisme Atensi (Attention)**. Fokusnya adalah pada cara model memproses urutan teks dan meningkatkan kinerjanya dalam memahami konteks jangka panjang.

---

## 1. Persiapan Lingkungan dan Dataset Umum

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `matplotlib.pyplot` untuk operasi numerik dan visualisasi, serta `os`.

Dataset yang digunakan untuk demonstrasi adalah **terjemahan mesin** (kemungkinan besar pasangan bahasa, seperti Inggris-Spanyol atau Inggris-Jerman).
* Data biasanya terdiri dari kalimat-kalimat sumber dan target.
* **Tokenisasi**: Teks perlu dikonversi menjadi urutan angka (token ID). `tf.keras.preprocessing.text.Tokenizer` digunakan untuk membangun kamus (vocabulary) kata-ke-ID dan mengonversi teks ke urutan integer.
* **Padding**: Urutan teks seringkali memiliki panjang yang berbeda. `tf.keras.preprocessing.sequence.pad_sequences` digunakan untuk membuat semua urutan memiliki panjang yang sama dengan menambahkan padding (biasanya nol). Ini penting karena *Neural Network* membutuhkan input dengan bentuk (shape) yang konsisten.
* **Output Encoding**: Untuk terjemahan mesin, label target (kalimat target) juga sering kali di-*tokenize* dan di-*padded*.

---

## 2. Encoder-Decoder Model untuk Terjemahan Mesin

Arsitektur umum untuk tugas *sequence-to-sequence* (seperti terjemahan mesin) adalah model **Encoder-Decoder**.

* **Encoder**:
    * Menerima urutan input (kalimat sumber).
    * Biasanya merupakan RNN (misalnya, LSTM atau GRU) yang memproses urutan *timestep* demi *timestep*.
    * Mengonversi urutan input menjadi representasi *context vector* berdimensi tetap (juga disebut *thought vector* atau *encoder state*). Vektor ini seharusnya menangkap "makna" dari seluruh kalimat input.
* **Decoder**:
    * Menerima *context vector* dari *encoder* sebagai input awal.
    * Juga merupakan RNN yang menghasilkan urutan output (kalimat target) *timestep* demi *timestep*.
    * Pada setiap *timestep*, *decoder* juga menerima input dari *token* yang diprediksi sebelumnya (atau *ground truth* *token* dalam *teacher forcing*).
* **`tf.keras.layers.Embedding`**: Digunakan sebagai lapisan pertama dalam *encoder* dan *decoder*. Lapisan ini mengonversi token ID integer menjadi vektor kepadatan berdimensi tetap (word embeddings).
    \[
    \text{Word Embeddings} = \text{Embedding Layer}(\text{Token IDs})
    \]
* **`tf.keras.layers.LSTM` atau `tf.keras.layers.GRU`**: Lapisan RNN yang digunakan dalam *encoder* dan *decoder* untuk memproses urutan.
    * `return_state=True`: Dalam *encoder*, ini memungkinkan Anda untuk mendapatkan *final hidden state* dan *cell state* (untuk LSTM) yang akan diteruskan ke *decoder* sebagai *initial state*.
    * `return_sequences=True`: Dalam *encoder*, ini diperlukan jika Anda ingin menggunakan *output* dari setiap *timestep* *encoder* oleh mekanisme atensi.

### Tantangan Model Encoder-Decoder Tanpa Atensi

* **Bottleneck Context Vector**: Seluruh informasi dari kalimat input harus dikompresi ke dalam satu *context vector* berdimensi tetap. Untuk kalimat yang panjang, ini menjadi *bottleneck* yang signifikan, menyebabkan model kesulitan mengingat bagian awal kalimat input.
* **Long-Range Dependencies**: Sulit bagi RNN dasar untuk menangkap dependensi yang sangat jauh dalam urutan.

---

## 3. Mekanisme Atensi (Attention Mechanism)

Mekanisme atensi dikembangkan untuk mengatasi *bottleneck* *context vector* dalam model Encoder-Decoder. Alih-alih hanya mengandalkan *context vector* tunggal, *decoder* dapat **"memperhatikan"** bagian-bagian relevan dari urutan input pada setiap langkah prediksi output.

* **Cara Kerja**:
    1.  *Encoder* menghasilkan **urutan *hidden states*** (output) untuk setiap *timestep* input (bukan hanya *final state*).
    2.  Pada setiap *timestep* *decoder*, *decoder* menghitung **skor atensi** antara *hidden state* *decoder* saat ini dan *hidden state* *encoder* di setiap *timestep*.
    3.  Skor atensi ini kemudian di-*softmax*-kan untuk mendapatkan **bobot atensi**, yang menunjukkan seberapa relevan setiap *hidden state* *encoder* dengan *hidden state* *decoder* saat ini.
    4.  Bobot atensi digunakan untuk menghitung ***context vector* yang berbobot** (weighted context vector) dengan mengambil rata-rata berbobot dari *hidden states* *encoder*.
        \[
        \text{Context Vector}_t = \sum_{j=1}^{\text{N_steps_encoder}} \text{Attention Weight}_{t,j} \times \text{Encoder Hidden State}_j
        \]
    5.  *Context vector* ini kemudian digabungkan dengan *hidden state* *decoder* saat ini dan diumpankan ke lapisan *feedforward* untuk memprediksi *token* output berikutnya.

* **Manfaat Atensi**:
    * Memungkinkan *decoder* untuk fokus pada bagian relevan dari input pada setiap langkah, mirip dengan bagaimana manusia membaca dan menerjemahkan kalimat.
    * Mengatasi *bottleneck* *context vector* pada kalimat panjang.
    * Meningkatkan kinerja terjemahan yang signifikan.
    * Membuat model lebih *interpretable* (kita bisa melihat bagian mana dari input yang diperhatikan oleh model).

---

## 4. Multi-Head Attention (Konseptual)

*Multi-Head Attention* adalah ekstensi dari mekanisme atensi yang memungkinkan model untuk "memperhatikan" informasi dari berbagai "sudut pandang" yang berbeda secara bersamaan.

* **Cara Kerja**: Alih-alih satu mekanisme atensi, *Multi-Head Attention* menjalankan beberapa mekanisme atensi secara paralel (disebut "kepala" atau "heads"). Setiap kepala mempelajari representasi atensi yang berbeda. Hasil dari setiap kepala kemudian digabungkan (concatenated) dan ditransformasi secara linier untuk menghasilkan representasi atensi akhir.
* **Manfaat**: Memungkinkan model untuk menangkap dependensi yang lebih kaya dan kompleks dalam data. Ini adalah komponen kunci dari arsitektur Transformer.

---
