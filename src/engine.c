#include "../include/engine.h"

Value* create_value(double val){
    Value* v= (Value*)malloc(sizeof(Value));
    v->data=val;
    v->grad=0.0;//başlangıçta birikm,i grad yok
    v->left=v->right=NULL;
    v->backward=NULL;//yapılacak işlem
    v->visited=0;
    return v;
}

void add_backward(Value* v){
    if(v->right) v->right->grad += v->grad;
    if(v->left) v->left->grad += v->grad;
}

Value* add(Value* a, Value* b){
    Value* out=create_value(a->data+b->data);
    out->left=a;out->right=b;
    out->backward=add_backward;
    return out;
}

void mul_backward(Value* v){
    if(v->left) v->left->grad +=v->right->data*v->grad;
    if(v->right) v->right->grad += v->left->data*v->grad;
}
Value* multiply(Value* a, Value* b){
    Value* out = create_value(a->data*b->data);
    out->left=a;out->right=b;
    out->backward=mul_backward;
    return out;
}

void relu_backward(Value* v)
{
    if(v->left){
        double der = (v->left->data>0)?1.0:0.0;
        v->left->grad+=der*v->grad;
    }
}

Value* relu(Value* a){
    Value* out = create_value(a->data>0?a->data:0);
    out->left=a;
    out->backward=relu_backward;
    return out;
}

//topolojik sıralama işlemi için
Value* topo[1000000];
int topo_p=0;

void build_topo(Value* v){
    if(v==NULL||v->visited)return;
    v->visited=1;
    build_topo(v->left);
    build_topo(v->right);
    topo[topo_p++]=v;
}//bu bir nevi ağaç gezme fonksiyonu

void apply_backward(Value* out){
    topo_p=0;
    build_topo(out);
    out->grad=1.0;
    for(int i=topo_p-1;i>=0;i--){
        if(topo[i]->backward) topo[i]->backward(topo[i]);
    }
}