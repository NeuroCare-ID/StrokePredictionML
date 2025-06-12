# NeuroCare 🧠💙

## Website Prediksi, Pencegahan, dan Penanganan Risiko Stroke

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-brightgreen.svg)](https://scikit-learn.org/)

## 📋 Deskripsi Proyek

**NeuroCare** adalah sebuah platform web inovatif yang dirancang untuk menjadi solusi komprehensif dalam pencegahan, deteksi dini, dan manajemen stroke. Dengan memanfaatkan kekuatan _Machine Learning_, NeuroCare bertujuan untuk memberdayakan masyarakat Indonesia dengan alat prediksi risiko yang akurat dan informasi kesehatan yang mudah diakses. Proyek ini dikembangkan dengan semangat inovasi di bidang kesehatan sebagai bagian dari **Coding Camp 2025**.

## 🎯 Latar Belakang

Stroke merupakan salah satu tantangan kesehatan terbesar di Indonesia, menempati **posisi kedua sebagai penyebab kematian** dan menjadi penyebab utama kecacatan dengan prevalensi 10,9 per 1.000 penduduk. Data dari Kemenkes (2023) menyoroti urgensi masalah ini, di mana deteksi dini menjadi kunci untuk mencegah komplikasi fatal dan meningkatkan kualitas hidup pasien. NeuroCare hadir sebagai jawaban proaktif untuk mengatasi masalah ini dengan menyediakan platform yang tidak hanya prediktif, tetapi juga edukatif dan suportif.

## ✨ Fitur Utama

Platform NeuroCare dirancang dengan berbagai fitur utama yang saling terintegrasi untuk memberikan layanan kesehatan stroke yang holistik:

🤖 **1. Sistem Prediksi Risiko Stroke:**

- Menggunakan **Deep Neural Network (DNN)** yang dibangun dengan _TensorFlow/Keras_ untuk akurasi prediksi yang tinggi.
- Implementasi algoritma skoring klinis **CHA₂DS₂-VASc** sebagai fitur tambahan untuk meningkatkan relevansi medis.
- Model dilatih dengan teknik _stratified upsampling_ untuk menangani ketidakseimbangan data, menghasilkan performa yang andal bahkan untuk kasus minoritas (pasien berisiko stroke).

📊 **2. Analisis Data dan Pra-pemrosesan Komprehensif:**

- **Exploratory Data Analysis (EDA)** mendalam untuk mengungkap pola dan korelasi tersembunyi antar variabel kesehatan.
- **Visualisasi data interaktif** menggunakan _Matplotlib_ dan _Seaborn_ untuk menyajikan temuan secara intuitif.
- **Pra-pemrosesan data otomatis** yang mencakup penanganan nilai hilang dengan imputasi median dan penskalaan fitur menggunakan _MinMaxScaler_.

🎯 **3. Model Machine Learning yang Dioptimalkan:**

- **Arsitektur Jaringan Saraf Tiruan Sekuensial** yang terdiri dari:
  - _Input Layer:_ 128 neuron dengan aktivasi ReLU, diikuti oleh _BatchNormalization_ dan _Dropout_ (0.3) untuk mencegah _overfitting_.
  - _Hidden Layer 1:_ 64 neuron dengan aktivasi ReLU, juga dilengkapi dengan _BatchNormalization_ dan _Dropout_ (0.2).
  - _Hidden Layer 2:_ 32 neuron dengan aktivasi ReLU.
  - _Output Layer:_ 1 neuron dengan fungsi aktivasi _Sigmoid_, ideal untuk masalah klasifikasi biner.
- **Proses Training yang Efisien:**
  - _Optimizer:_ Adam dengan _learning rate_ 0.003 yang telah dioptimalkan.
  - _Loss Function:_ _Binary Crossentropy_, standar untuk klasifikasi biner.
  - _Callbacks:_ Dilengkapi dengan _EarlyStopping_ dan _callback_ kustom untuk menghentikan pelatihan saat akurasi mencapai >92%, memastikan efisiensi tanpa mengorbankan performa.

📱 **4. Siap untuk Deployment:**

- Model disimpan dalam format `.h5` untuk kemudahan integrasi dengan _backend_ Python.
- Fungsi inferensi yang terstruktur rapi, siap untuk diintegrasikan ke dalam API.
- Arsitektur proyek yang modular, memisahkan antara data, model, dan kode aplikasi.

## 🛠️ Tech Stack

Proyek ini dibangun menggunakan teknologi modern di bidang _Data Science_ dan pengembangan web:

- **Machine Learning & Data Science:**
  - `Python 3.8+`
  - `TensorFlow 2.x / Keras`
  - `Scikit-learn`
  - `Pandas` & `NumPy`
  - `Matplotlib` & `Seaborn`
  - `Imbalanced-learn`
- **IDE & Tools:**
  - `Jupyter Notebook` & `Google Colab`
  - `Visual Studio Code`
  - `Git` & `GitHub`
- **Dataset:**
  - [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

## 📁 Struktur Proyek

```
NeuroCare/
├── 📄 healthcare-dataset-stroke-data.csv
├── 📓 Capstone_Project_NeuroCare.ipynb
├── 📓 Capstone_Project_NeuroCare.py
└── 📄 README.md
```

## 👥 Tim Pengembang

**ID Tim: CC25-CF073**

- **Machine Learning:**
  - Muh. Rofif Rahman Fuadi (Universitas Brawijaya)
  - Fatmah Fianka Syafrudin (Universitas Brawijaya)
  - Muhammad Husain Fadhlillah (Universitas Brawijaya)
- **Frontend/Backend:**
  - Fauzan Al Hafizh (Universitas Indraprasta PGRI)
  - Dwi Danty Aisyah (UIN Maulana Malik Ibrahim Malang)

---

## 🚀 Cara Mereplikasi Proyek (Replication Guide)

Berikut adalah panduan langkah demi langkah untuk mereplikasi seluruh proses _machine learning_ yang telah kami lakukan, dari persiapan data hingga evaluasi model.

### **Prasyarat**

Pastikan Anda memiliki Python 3.8+ dan `pip` terinstal di sistem Anda. Sebaiknya gunakan lingkungan virtual (_virtual environment_) untuk menjaga dependensi proyek tetap terisolasi.

```bash
# Buat lingkungan virtual
python -m venv neurocare_env

# Aktifkan di Windows
neurocare_env\Scripts\activate

# Aktifkan di macOS/Linux
source neurocare_env/bin/activate
```

### **Langkah 1: Instalasi Pustaka (Libraries)**

Instal semua pustaka yang dibutuhkan menggunakan `pip`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### **Langkah 2: Akuisisi dan Pemuatan Data**

Dataset yang kami gunakan bersumber dari Kaggle. Anda dapat mengunduhnya secara manual atau menggunakan Kaggle API.

```python
# Import library dasar
import pandas as pd

# Muat dataset
# Pastikan file 'healthcare-dataset-stroke-data.csv' berada di direktori yang sama
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Hapus kolom 'id' dan 'work_type' yang tidak relevan untuk pemodelan
df.drop(['id', 'work_type'], axis=1, inplace=True)

# Tampilkan 5 baris pertama untuk verifikasi
print(df.head())
```

### **Langkah 3: Pembersihan dan Pra-pemrosesan Data**

Tahap ini sangat krusial untuk memastikan kualitas data sebelum dimasukkan ke dalam model.

**3.1. Menghapus Outlier Kategorikal**
Terdapat satu entri 'Other' pada fitur `gender` yang kami hapus karena tidak representatif.

```python
# Hapus baris dengan gender 'Other'
df = df[df['gender'] != 'Other'].copy()
```

**3.2. Menangani Nilai yang Hilang (Missing Values)**
Kolom `bmi` memiliki 201 nilai yang hilang. Kami mengisinya menggunakan strategi **imputasi median**.

```python
from sklearn.impute import SimpleImputer

# Inisialisasi imputer dengan strategi median
imp_median = SimpleImputer(strategy='median')

# Terapkan imputer pada kolom 'bmi'
df[['bmi']] = imp_median.fit_transform(df[['bmi']])
```

**3.3. Menangani Ketidakseimbangan Data (Imbalance Dataset)**
Kelas target `stroke` sangat tidak seimbang. Kami menggunakan teknik **Stratified Upsampling** untuk menyeimbangkan distribusi kelas sambil menjaga proporsi faktor risiko penting (`heart_disease` dan `hypertension`).

```python
from sklearn.utils import resample

# Pisahkan antara kelas mayoritas dan minoritas
df_majority = df[df.stroke == 0]
df_minority = df[df.stroke == 1]

# Kelompokkan kelas minoritas berdasarkan faktor risiko
minority_groups = df_minority.groupby(['heart_disease', 'hypertension'])

# Lakukan upsampling terstratifikasi
upsampled_list = []
for _, group in minority_groups:
    n_samples = df_majority.shape[0] // minority_groups.ngroups
    upsampled_group = resample(group,
                               replace=True,
                               n_samples=n_samples,
                               random_state=42)
    upsampled_list.append(upsampled_group)

# Gabungkan kembali dataset
upsampled_minority = pd.concat(upsampled_list)
df_balanced = pd.concat([df_majority, upsampled_minority])

# Acak dataset untuk menghindari bias urutan
df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
```

### **Langkah 4: Rekayasa Fitur (Feature Engineering)**

Kami membuat fitur baru, yaitu skor **CHA₂DS₂-VASc**, untuk memberikan konteks klinis tambahan pada model.

```python
# Fungsi untuk menghitung skor CHA₂DS₂-VASc
def calculate_cha2ds2vasc(row):
    score = 0
    if row['age'] >= 75:
        score += 2
    elif row['age'] >= 65:
        score += 1
    if row['hypertension'] == 1:
        score += 1
    if row['heart_disease'] == 1:
        score += 1
    if row['gender'] == 'Female':
        score += 1
    # Fitur lain seperti Diabetes (glucose), Stroke/TIA (tidak ada di data) tidak dihitung
    return score

# Terapkan fungsi untuk membuat kolom baru
df['cha2ds2vasc_score'] = df.apply(calculate_cha2ds2vasc, axis=1)
```

### **Langkah 5: Encoding dan Penskalaan Fitur**

Model _machine learning_ memerlukan input numerik. Oleh karena itu, fitur kategorikal perlu diubah.

**5.1. One-Hot Encoding**
Kami mengubah fitur kategorikal menjadi format numerik menggunakan _One-Hot Encoding_.

```python
categorical_features = df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
```

**5.2. Penskalaan Fitur (Feature Scaling)**
Kami menggunakan `MinMaxScaler` untuk menskalakan semua fitur numerik ke dalam rentang [0, 1].

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Pisahkan fitur (X) dan target (y)
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Lakukan penskalaan
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### **Langkah 6: Pengembangan dan Pelatihan Model**

Kami membangun model _Deep Neural Network_ menggunakan TensorFlow/Keras.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Arsitektur Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilasi Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callback untuk berhenti lebih awal
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Latih Model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=2
)
```

### **Langkah 7: Evaluasi Model**

Setelah pelatihan, kami mengevaluasi performa model pada data pengujian.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Lakukan prediksi
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Tampilkan Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Hitung dan tampilkan AUC Score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc:.4f}")

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### **Langkah 8: Penyimpanan Model**

Simpan model yang telah dilatih untuk digunakan di tahap _deployment_.

```python
# Simpan model dalam format H5
model.save("model.h5")
print("Model berhasil disimpan sebagai model.h5")
```

Dengan mengikuti langkah-langkah di atas, Anda dapat mereplikasi proses pengembangan model prediksi risiko stroke NeuroCare secara lengkap.

## 📞 Kontak dan Support

### Project Links:

- **GitHub Repository**: [NeuroCare-ID](https://github.com/NeuroCare-ID)
- **Demo**: (coming soon)

### Team Contact:

- **Email**: neurocareid@gmail.com

---

### ⭐ Jika proyek ini bermanfaat, jangan lupa berikan **star** di GitHub!

**Made with ❤️ by NeuroCare Team @2025**
