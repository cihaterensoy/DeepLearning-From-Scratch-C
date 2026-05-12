# DeepLearning-From-Scratch-C

This project is a neural network system developed entirely in the C programming language, without using any high-level libraries, to recognize handwritten digits from the MNIST dataset. The main focus of the project is to understand the mathematical foundations of artificial intelligence algorithms and low-level memory management.

## Project Purpose

The goal is to understand the internal workings of machine learning models by implementing fundamental algorithms such as forward propagation and backpropagation from scratch. The project is particularly based on manual control of memory management and data structures.

## Technical Features

* **Language:** C
* **Dataset:** MNIST Handwritten Digits
* **Compiler:** GCC
* **File Loader:** Custom functions developed to process MNIST `.idx` format files (e.g., `load_mnist_images`)

## Installation and Build

To compile and run the project, the following commands are used:

```bash
# Building the project
gcc -Iinclude -Wall main.c src/engine.c src/mnist.c -o yapay_zeka.exe

# Running the program
./yapay_zeka.exe
```

## Training and Test Performance

### Training Process (Screenshot1.jpg):

| Training Stage | Loss   | Accuracy |
| -------------- | ------ | -------- |
| Epoch 0        | 0.1465 | %92.97   |
| Epoch 1        | 0.0854 | %96.57   |
| Epoch 5        | 0.0567 | %98.15   |

### Final Test Results (Screenshot2.jpg):

* Total Test Images: 10,000
* Correct Predictions: 9,684
* Test Accuracy: %96.84

# DeepLearning-From-Scratch-C

Bu çalışma, MNIST veri setindeki el yazısı rakamları tanımak için herhangi bir yüksek seviyeli kütüphane kullanmadan tamamen C dili ile geliştirilmiş bir sinir ağı projesidir. Projenin temel odağı, yapay zeka algoritmalarının matematiksel arka planını ve düşük seviyeli bellek yönetimini anlamaktır.

## Projenin Amacı

Yapay zeka modellerinin iç işleyişini, ileri besleme ve geri yayılım gibi temel algoritmaları sıfırdan kodlayarak kavramak hedeflenmiştir. Proje, özellikle bellek yönetimi ve veri yapılarının manuel kontrolü üzerine kuruludur.

## Teknik Özellikler

*   **Dil:** C
*   **Veri Seti:** MNIST El Yazısı Rakamlar
*   **Derleyici:** GCC
*   **Dosya Okuyucu:** `.idx` formatındaki MNIST dosyalarını işlemek için geliştirilen özel fonksiyonlar (`load_mnist_images` gibi)

## Kurulum ve Derleme

Projeyi derlemek ve çalıştırmak için aşağıdaki komutlar kullanılmaktadır:

```bash
# Projenin derlenmesi
gcc -Iinclude -Wall main.c src/engine.c src/mnist.c -o yapay_zeka.exe

# Programın çalıştırılması
./yapay_zeka.exe
```

## Eğitim ve Test Performansı

**Eğitim Süreci (Screenshot1.jpg):**

| Eğitim Safhası | Hata Payı (Loss) | Doğruluk (Accuracy) |
| :--- | :--- | :--- |
| Epoch 0 | 0.1465 | %92.97 |
| Epoch 1 | 0.0854 | %96.57 |
| Epoch 5 | 0.0567 | %98.15 |

**Final Test Sonuçları (Screenshot2.jpg):**
*   **Toplam Test Resmi:** 10.000
*   **Doğru Tahmin Sayısı:** 9684
*   **Test Başarı Oranı:** %96.84

