<h1 align="center">Tugas Besar 2 IF3170 Inteligensi Artifisial </h1>

<h1 align="center">Implementasi Algoritma Pembelajaran Mesin</h1>

## Deskripsi

Proyek ini bertujuan untuk mengimplementasikan tiga algoritma pembelajaran mesin, yaitu K-Nearest Neighbors (KNN), Gaussian Naive-Bayes, dan ID3 menggunakan pendekatan from scratch serta membandingkan hasilnya dengan pustaka scikit-learn. Dataset yang digunakan adalah UNSW-NB15, yang berisi berbagai jenis serangan siber dan aktivitas jaringan normal.

## Fitur Utama

1.  **Implementasi Algoritma From Scratch**

    - KNN: Mendukung tiga metrik jarak (Euclidean, Manhattan, dan Minkowski).
    - Gaussian Naive-Bayes: Perhitungan probabilitas berdasarkan distribusi Gaussian.
    - ID3: Algoritma pohon keputusan menggunakan perhitungan entropy dan information gain.

2.  **Implementasi Menggunakan Scikit-Learn**
    - KNN: Menggunakan KNeighborsClassifier.
    - Naive-Bayes: Menggunakan GaussianNB.
    - ID3: Menggunakan DecisionTreeClassifier dengan criterion='entropy'.
3.  **Model Saving dan Loading**

    Model yang telah dilatih dapat disimpan dalam format .pkl menggunakan ModelLoader.

4.  **Data Cleaning dan Preprocessing**

    Penanganan missing values, duplikasi data, encoding fitur kategorikal, scaling, serta feature engineering untuk meningkatkan performa model.

5.  **Evaluasi Model**

    Menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.

## Cara Run Program

1. **Clone Repository**

   git clone https://github.com/AlbertChoe/Tubes2-AI.git
   cd Tubes2-AI/src

2. **Jalankan Notebook Utama**

   Buka file notebook notebooks/main.ipynb di Jupyter Notebook dan langsung run.

### **Kelompok 21: BebanKaggle**

|   NIM    |          Nama           | Pembagian Tugas                                    |
| :------: | :---------------------: | -------------------------------------------------- |
| 13522045 |     Elbert Chailes      | Pipeline, Data preprocessing, Model ,Documentation |
| 13522073 |   Juan Alfred Widjaya   | Pipeline, Data preprocessing, Model ,Documentation |
| 13522081 |         Albert          | Pipeline, Data preprocessing, Model ,Documentation |
| 13522113 | William Glory Henderson | Pipeline, Data preprocessing, Model ,Documentation |
