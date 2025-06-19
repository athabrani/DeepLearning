# Pembelajaran Penguatan (Reinforcement Learning - RL): Melatih Agen untuk Bertindak di Lingkungan

*Notebook* ini membahas konsep dasar **Pembelajaran Penguatan (Reinforcement Learning - RL)**, sebuah paradigma *machine learning* di mana agen belajar cara bertindak di lingkungan untuk memaksimalkan *reward* (hadiah) kumulatifnya. Fokus utamanya adalah pada masalah **CartPole** (menyeimbangkan tiang di kereta yang bergerak) sebagai contoh, dan memperkenalkan algoritma dasar seperti **Policy Gradients** dan **Deep Q-Networks (DQN)**.

---

## 1. Persiapan Lingkungan dan Konsep Dasar RL

Langkah awal melibatkan impor pustaka standar seperti `tensorflow` (dengan `keras`), `numpy`, `matplotlib.pyplot`, dan `gym`.

### A. Lingkungan (Environment)

* **Konsep**: Dalam RL, **lingkungan** adalah sistem di mana agen berinteraksi. Lingkungan memiliki *state* (keadaan) yang menggambarkan situasi saat ini.
* **Gymnasium (sebelumnya OpenAI Gym)**: Pustaka yang menyediakan berbagai lingkungan RL yang sudah jadi, seperti CartPole.
    * `gym.make("CartPole-v1")`: Menginisialisasi lingkungan CartPole.
    * `env.observation_space.shape`: Mengembalikan bentuk *state* (pengamatan) dari lingkungan. Untuk CartPole, ini adalah 4 nilai kontinu (posisi kereta, kecepatan kereta, sudut tiang, kecepatan sudut tiang).
    * `env.action_space.n`: Mengembalikan jumlah tindakan diskrit yang tersedia. Untuk CartPole, ini adalah 2 (mendorong kereta ke kiri atau ke kanan).
    * `env.reset()`: Mengatur ulang lingkungan ke *state* awal, mengembalikan *initial observation*.
    * `env.step(action)`: Melakukan tindakan dalam lingkungan. Ini mengembalikan *next observation*, *reward*, *done* (boolean yang menunjukkan apakah episode selesai), dan *info* tambahan.
* **Episode**: Sebuah urutan tindakan, *state*, dan *reward* dari *state* awal hingga *state* terminal.

### B. Agen (Agent)

* **Konsep**: **Agen** adalah entitas yang belajar dan mengambil tindakan di lingkungan. Tujuannya adalah untuk belajar *policy* (strategi) yang akan memaksimumkan *reward* kumulatif jangka panjang.
* **Policy ($\pi$)**: Strategi yang digunakan agen untuk memilih tindakan berdasarkan *state* saat ini. Ini bisa berupa fungsi deterministik (`action = \pi(state)`) atau probabilistik (`P(action | state)`).
* **Reward**: Sinyal umpan balik dari lingkungan yang menunjukkan seberapa baik tindakan agen.

---

## 2. Strategi Kebijakan (Policy Strategy)

Pendekatan ini secara langsung melatih *Neural Network* untuk mewakili *policy* agen.

### A. Neural Network sebagai Policy Estimator

* **Arsitektur**: Sebuah *Neural Network* (misalnya, dengan lapisan `Dense` dan aktivasi `softmax` di *output layer* untuk tindakan diskrit) dapat mengambil *state* sebagai input dan menghasilkan probabilitas untuk setiap tindakan yang mungkin.
    * Input: *Observation* dari lingkungan.
    * Output: Probabilitas untuk setiap tindakan (misalnya, 2 probabilitas untuk CartPole: dorong kiri, dorong kanan).
* **Memilih Tindakan**: Agen mengambil tindakan dengan memilih tindakan secara acak berdasarkan probabilitas yang dihasilkan oleh *policy network*.

### B. Policy Gradients

**Policy Gradients** adalah kelas algoritma RL yang secara langsung mengoptimalkan parameter *policy network* dengan menggunakan *gradient ascent* pada fungsi *reward* yang diharapkan.

* **Intuisi**: Jika agen mengambil tindakan yang menghasilkan *reward* positif, gradien *policy* disesuaikan untuk membuat tindakan tersebut lebih mungkin terjadi di masa depan. Jika *reward* negatif, tindakan tersebut menjadi kurang mungkin.
* **Fungsi Biaya (Loss Function)**: Dalam Policy Gradients, kita tidak secara langsung meminimalkan *loss* seperti dalam *supervised learning*. Sebaliknya, kita memaksimalkan *reward* yang diharapkan. Ini seringkali diterjemahkan ke dalam meminimalkan *negative expected reward*.
    * Gradien dihitung dari *log-probability* tindakan yang diambil, dikalikan dengan *reward* yang diperoleh (atau *advantage*).
    \[
    \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \approx \frac{1}{M} \sum_{i=1}^{M} \sum_{t=1}^{T_i} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_{i,t} | s_{i,t}) R(\tau_i)
    \]
    Di mana $J(\boldsymbol{\theta})$ adalah fungsi *reward* yang diharapkan, $\pi_{\boldsymbol{\theta}}$ adalah *policy network*, $a_{i,t}$ adalah tindakan, $s_{i,t}$ adalah *state*, $R(\tau_i)$ adalah *reward* kumulatif untuk episode $\tau_i$, dan $M$ adalah jumlah episode.
* **Perhitungan Gradien Kustom**: `tf.GradientTape` digunakan untuk menghitung gradien secara manual selama *custom training loop*.
* **Discount Factor ($\gamma$)**: *Reward* di masa depan didiskon (diberi bobot lebih rendah) daripada *reward* instan. Ini mendorong agen untuk mencari *reward* yang lebih cepat.
    \[
    \text{Discounted Return} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
    \]
* **Keunggulan**: Policy Gradients dapat belajar *policy* yang kompleks dan probabilistik.
* **Kekurangan**: Memiliki *variance* yang tinggi (perlu banyak episode untuk konvergensi).

---

## 3. Deep Q-Networks (DQN)

DQN adalah salah satu algoritma RL yang paling sukses, menggabungkan *Q-Learning* dengan *Deep Neural Networks*.

### A. Q-Learning

* **Q-Value ($Q(s, a)$)**: Estimasi nilai (hadiah kumulatif yang diharapkan) untuk mengambil tindakan $a$ di *state* $s$, dan kemudian mengikuti *policy* optimal selanjutnya.
* **Persamaan Bellman (Bellman Equation)**: Dasar untuk memperbarui Q-value.
    \[
    Q(s, a) = r + \gamma \max_{a'} Q(s', a')
    \]
    * $r$: *Reward* instan.
    * $s'$: *Next state*.
    * $\gamma$: *Discount factor*.
* **Optimal Policy**: Agen mengikuti *policy* optimal dengan memilih tindakan $a$ yang memaksimalkan $Q(s, a)$ untuk *state* $s$.

### B. Deep Q-Networks (DQN)

* **Arsitektur**: Menggunakan *Neural Network* (Q-Network) untuk mengaproksimasi fungsi Q-value. Inputnya adalah *state*, dan *output*-nya adalah Q-value untuk setiap tindakan yang mungkin di *state* tersebut.
* **Pengalaman Replay (Experience Replay)**: Untuk mengatasi masalah korelasi dalam data pelatihan (karena urutan pengalaman sangat berkorelasi), DQN menyimpan pengalaman agen (tuple `(s, a, r, s', done)`) dalam *replay buffer*. Selama pelatihan, *mini-batch* pengalaman diambil secara acak dari *buffer* ini. Ini mendemokrasikan dan menstabilkan pelatihan.
* **Jaringan Target (Target Network)**: Untuk mengatasi ketidakstabilan, DQN menggunakan dua Q-Network:
    * **Online Q-Network**: Jaringan yang bobotnya diperbarui pada setiap langkah pelatihan.
    * **Target Q-Network**: Salinan dari Online Q-Network yang bobotnya dibekukan dan hanya diperbarui secara berkala (misalnya, setiap beberapa ratus iterasi). Ini digunakan untuk menghitung target Q-value, menyediakan target yang lebih stabil untuk pelatihan.
* **Epsilon-Greedy Policy**: Untuk mendorong eksplorasi (mencoba tindakan baru) dan eksploitasi (mengambil tindakan terbaik yang diketahui), agen mengikuti *epsilon-greedy policy*.
    * Dengan probabilitas $\epsilon$, agen memilih tindakan acak.
    * Dengan probabilitas $1 - \epsilon$, agen memilih tindakan yang memaksimalkan Q-value (tindakan "greedy").
    * $\epsilon$ biasanya dimulai tinggi dan berkurang seiring waktu.

---

**Kesimpulan:**

*Notebook* ini memperkenalkan dasar-dasar Pembelajaran Penguatan, dari interaksi agen-lingkungan hingga algoritma inti. Policy Gradients secara langsung mengoptimalkan *policy*, sementara DQN menggabungkan *Q-Learning* dengan *Deep Neural Networks* dan teknik stabilisasi seperti *experience replay* dan *target networks*. Memahami konsep-konsep ini adalah kunci untuk melatih agen agar dapat membuat keputusan optimal dalam lingkungan yang kompleks.
