#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Value {
    double data;//kendi değeri
    double grad;//türev değeri
    struct Value *left;//ata
    struct Value *right;//ata
    void (*backward)(struct Value*);//hangi işlemin yapılacağını tutuyoruz
    int visited;//ziyaret edildi mi bayrağı
    int is_param; //1 ise ağırlıktır silme, 0 ise geçicidir sil
}Value;
//nöron ve katman yapısı
typedef struct{
    Value** weight; //her girdi için bir ağırlık
    Value* bias; //bir tane oluyor zaten
    int nin;//girdi sayısı

}Neuron;
typedef struct {
    Neuron** neurons;//katmandaki nöronlar
    int nout;//cıktı sayısı, kaç nöronumuz var
}Layer;


//
//AĞIRLIKLARI KAYDETMEK İÇİN GERKELİ FONK TANIMI
void save_weights(Layer* l, char* filename);
void load_weights(Layer* l, char* filename);
//önceden fonksiyon tanımları
Value* create_value(double val, int is_param);
void clear_graph();
Value* add(Value* a, Value* b);
Value* sub(Value* a, Value* b);
Value* multiply(Value* a,Value* b);
Value* relu(Value* a);
void apply_backward(Value* out);
void free_graph(Value* v);//hafızayı yenilemek için yazılacak fonk
void build_topo(Value* v);

uint32_t swap_endian(uint32_t val);
double** load_mnist_images(char* filename, int* num_images);
int* load_mnist_labels(char* filename, int* num_labels);

//nöronlar için fonk tanıtım
Neuron* create_neuron(int nin);
Layer* create_layer(int nin, int nout);
Value** forward_layer(Layer* l, Value** inputs);
//loss fonk

Value* calc_loss(Value** predictions, int target_label);
void update_weights(Layer* l, double learning_rate);



#endif