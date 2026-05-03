#include "include/engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(42); // Kıyaslama yapabilmek için sabit

    int num_test_img, num_test_labels;

    double** test_images = load_mnist_images("data/t10k-images.idx3-ubyte", &num_test_img);
    int* test_labels = load_mnist_labels("data/t10k-labels.idx1-ubyte", &num_test_labels);

    if (!test_images || !test_labels) {
        printf("Hata: MNIST test verileri yuklenemedi!\n");
        return 1;
    }

    // eğitim mimarisi
    Layer* hidden_layer = create_layer(784, 64);
    Layer* output_layer = create_layer(64, 10);

    //Ağırlıkları Yükle
    printf("Agirliklar dosyalardan yukleniyor...\n");
    load_weights(hidden_layer, "hidden_weights.bin");
    load_weights(output_layer, "output_weights.bin");

    // Performans İpucu: Girişleri bir kez oluştur ve tekrar kullan
    // is_param=1 veriyoruz ki clear_graph() bunlari hafizadan silmesin
    Value* inputs[784];
    for (int j = 0; j < 784; j++) {
        inputs[j] = create_value(0.0, 1); 
    }

    int correct = 0;
    printf("Test basliyor (10.000 resim)...\n");

    for (int i = 0; i < num_test_img; i++) {
        // Malloc YAPMADAN sadece veriyi güncelliyoruz
        for (int j = 0; j < 784; j++) {
            inputs[j]->data = test_images[i][j];
        }

        // İleri Besleme (Forward Pass)
        Value** hidden_out = forward_layer(hidden_layer, inputs);
        Value** preds = forward_layer(output_layer, hidden_out);
        // apply_backward çağırmadığımız için topo dizisini manuel dolduruyoruz ki clear_graph() neyi sileceğini bilsin
        for(int j = 0; j < 10; j++) {
            build_topo(preds[j]);
        }
        // Tahmin Bulma
        int prediction = 0;
        for (int j = 1; j < 10; j++) {
            if (preds[j]->data > preds[prediction]->data)
                prediction = j;
        }

        if (prediction == test_labels[i])
            correct++;

        // Sadece geçici işlem dizilerini serbest bırakıyoruz
        free(hidden_out);
        free(preds);
        
        // Ara işlemleri (relu, multiply sonuçları vb.) temizler
        clear_graph(); 

        // Her 1000 resimde bir ilerlemeyi göster
        if ((i + 1) % 1000 == 0) {
            printf("Islenen: %d / %d...\n", i + 1, num_test_img);
        }
    }

    printf("\n--- TEST SONUCU ---\n");
    printf("Toplam Resim: %d\n", num_test_img);
    printf("Dogru Tahmin: %d\n", correct);
    printf("Test Accuracy: %.2f%%\n",(double)correct * 100.0 / num_test_img);

    // Temizlik (Program kapanırken)
    for(int j=0; j<784; j++) free(inputs[j]);
    // Not: Katmanları ve resim dizilerini de free etmek iyi bir pratiktir.

    return 0;
}