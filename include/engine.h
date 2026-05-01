#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Value {
    double data;//kendi değeri
    double grad;//türev değeri
    struct Value *left;//ata
    struct Value *right;//ata
    void (*backward)(struct Value*);//hangi işlemin yapılacağını tutuyoruz
    int visited;//ziyaret edildi mi bayrağı
}Value;

//önceden fonksiyon tanımları
Value* create_value(double val);
Value* add(Value* a, Value* b);
Value* multiply(Value* a,Value* b);
Value* relu(Value* a);
void apply_backward(Value* out);
void free_graph(Value* v);//hafızayı yenilemek için yazılacak fonk

#endif