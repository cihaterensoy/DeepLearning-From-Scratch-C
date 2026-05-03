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

