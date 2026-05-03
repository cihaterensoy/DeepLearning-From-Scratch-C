/*
#include "include/engine.h"

int main() {

   int num_img,num_labels;
   double** images = load_mnist_images("data/train-images.idx3-ubyte", &num_img);
   int* labels = load_mnist_labels("data/train-labels.idx1-ubyte", &num_labels);

   Layer* output_layer =create_layer(784,10);
   double learning_rate = 0.01;

   for(int epoch=0;epoch<5;epoch++){

    double epoch_loss=0;
    for(int i=0;i<1000;i++){

        Value* input_v[784];
        for (int j = 0; j < 784; j++) {
            input_v[j] = create_value(images[i][j]);
        }
        Value** preds = forward_layer(output_layer, input_v);
        Value* loss = calc_loss(preds, labels[i]);
        epoch_loss += loss->data;
        apply_backward(loss);
        update_weights(output_layer, learning_rate);
    }
    printf("Epoch %d, Ortalama Loss: %f\n", epoch, epoch_loss / 1000);
   }
    Value* input_v[784];
    for(int i = 0; i < 784; i++) {
        input_v[i] = create_value(images[0][i]);
    }

    Value** predictions = forward_layer(output_layer, input_v);

    printf("\n--- Tahmin ---\n");
    for(int i = 0; i < 10; i++) {
        printf("Rakam %d skoru: %f\n", i, predictions[i]->data);
    }
   return 0;
}
   */
#include "include/engine.h"
#include <time.h>

int main() {
    //srand(time(NULL));
    srand(42);
    int num_train_img, num_labels;

    // 1. Verileri Yükle
    double** train_images = load_mnist_images("data/train-images.idx3-ubyte", &num_train_img);
    int* train_labels = load_mnist_labels("data/train-labels.idx1-ubyte", &num_labels);
    
    // 2. Ağı Kur (784 Giriş -> 10 Çıkış)
    Layer* hidden_layer = create_layer(784, 64);
    Layer* output_layer = create_layer(64,10);
    double learning_rate = 0.01;

    printf("Egitim basliyor...\n");

    for (int epoch = 0; epoch < 6; epoch++) {
        double total_loss = 0;
        int correct = 0;

        for (int i = 0; i < 50000; i++) { // Hız için ilk 5000 resim
            // Girdileri hazırla
            Value* inputs[784];
            for(int j=0; j<784; j++) inputs[j] = create_value(train_images[i][j],0);

            // Forward Pass
            //Value** preds = forward_layer(output_layer, inputs);
            Value** hidden_outputs = forward_layer(hidden_layer, inputs);
            Value** final_predictions = forward_layer(output_layer, hidden_outputs);
            // En yüksek tahmini bul (Doğruluk ölçümü için)
            int prediction = 0;
            for(int j=1; j<10; j++) if(final_predictions[j]->data > final_predictions[prediction]->data) prediction = j;
            if(prediction == train_labels[i]) correct++;

            // Loss ve Backprop
            Value* loss = calc_loss(final_predictions, train_labels[i]);
            total_loss += loss->data;
            
            apply_backward(loss);
            update_weights(output_layer, learning_rate);
            update_weights(hidden_layer, learning_rate);
            clear_graph(); 
            free(final_predictions); // prediction dizisini de serbest bırak
            free(hidden_outputs);
        }
        //printf("Epoch %d | Hata: %.4f | Dogruluk: %%%d\n", epoch, total_loss/5000, (correct*100)/5000);
        printf("Epoch %d | Hata: %.4f | Dogruluk: %%%.2f\n", epoch, total_loss/50000, (double)(correct*100)/50000);
    }
    printf("\nAgirliklar kaydediliyor...\n");
    
    // Her katmanı ayrı bir dosyaya kaydediyoruz
    save_weights(hidden_layer, "hidden_weights.bin");
    save_weights(output_layer, "output_weights.bin");

    printf("Basariyla kaydedildi\n");
    return 0;
}