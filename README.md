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
