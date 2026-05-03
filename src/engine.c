#include <math.h>
#include <time.h>
#include "../include/engine.h"
Value* create_value(double val, int is_param){
    Value* v= (Value*)malloc(sizeof(Value));
    v->data=val;
    v->grad=0.0;//başlangıçta birikm,i grad yok
    v->left=v->right=NULL;
    v->backward=NULL;//yapılacak işlem
    v->visited=0;
    v->is_param = is_param; // Bayrağı ata
    return v;
}

void add_backward(Value* v){
    if(v->right) v->right->grad += v->grad;
    if(v->left) v->left->grad += v->grad;
}

Value* add(Value* a, Value* b){
    Value* out=create_value(a->data+b->data,0);
    out->left=a;out->right=b;
    out->backward=add_backward;
    return out;
}

void sub_backward(Value* v){
    //input_grad += (çıktının türevi wrt input) * (çıktının grad’i)
    //a-b => b->grad += (∂out/∂b) * v->grad => += (-1) * v->grad => -= v->grad
    //a artarsa → sonuç artar → pozitif etki 
    //b artarsa → sonuç azalır → negatif etki 
    if(v->left) v->left->grad += ((1)*v->grad);
    if(v->right) v->right->grad += ((-1)*v->grad);
    
}

Value* sub(Value* a, Value* b){
    Value* out=create_value(a->data-b->data,0);
    out->left=a;out->right=b;
    out->backward=sub_backward;
    return out;
}

void mul_backward(Value* v){
    if(v->left) v->left->grad +=v->right->data*v->grad;
    if(v->right) v->right->grad += v->left->data*v->grad;
}
Value* multiply(Value* a, Value* b){
    Value* out = create_value(a->data*b->data,0);
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
    Value* out = create_value(a->data>0?a->data:0,0);
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

double rand_gen(){
    // rand() → 0 ile RAND_MAX arası sayı üretir
    // bunu 0.0 - 1.0 aralığına çekiyoruz
    // sonra 2 ile çarpıp -1 kaydırarak -1 ile 1 arası yapıyoruz
    return ((double)rand()/(double)RAND_MAX)*2.0-1.0;//-1.0 ile 1.0 arasında rastgele sayı üretme
}

//nöron oluşturma
Neuron* create_neuron(int nin){
    Neuron* n = (Neuron*)malloc(sizeof(Neuron));
    n->nin=nin;//bu nörona kaç giriş geliyor
    n->weight = (Value**)malloc(nin*sizeof(Value*));  // weights = ağırlıklar dizisi (her input için 1 ağırlık)
    for(int i=0;i<nin;i++){
        n->weight[i]=create_value(rand_gen()*0.01,1);//her ağırlık için rastgele kücük bir değer oluşturuyoruz//ağrılıklar kalıcı bu yüzden 1
    }
    n->bias =create_value(0.0,1); //bias genelde 0'dan başlar
    return n;
}

Layer* create_layer(int nin,int nout){
    Layer* l=(Layer*)malloc(sizeof(Layer));//layer için bellek ayırıyoruz
    l->nout=nout;//bu aktmandaki nöron sayısını veriyoruz
    //neuron listesi
    l->neurons=(Neuron**)malloc(nout*sizeof(Neuron*));
    //her nöronu oluşstur
    for(int i=0;i<nout;i++){
        l->neurons[i]=create_neuron(nin);//her nöronun girişi bir önceki katmanın cıkısı kadar olacak
    }
    return l;
}

Value** forward_layer(Layer* l, Value** inputs){
    //burada her nörona input verip bir output almanın fonkunu yazıyoru
    Value** outputs=(Value**)malloc(l->nout*sizeof(Value*));//cıktırların tutlacağı dizi
    //her nöron için hesaplama yapılacak for
    for(int i=0;i<l->nout;i++){
        //linear kısmı oluşturuyoruz yani w1*x1+bias
        Value* act= l->neurons[i]->bias;// başlangıç değeri = bias
        for(int j=0;j<l->neurons[i]->nin;j++){
            //tüm inputları dolaşıyoruz
            Value* mul = multiply(inputs[j],l->neurons[i]->weight[j]);//input*weight
            act=add(act,mul);//biasla ağırlıkları topla
        }

        outputs[i]=relu(act);//relu fonkuna soktuk cıktıya
    }
    return outputs;
}

//modelin hatasını hesaplmaak için fonk 
Value* calc_loss(Value** predictions,int target_label){
    //toplam hatayı tutuyoruz
    Value* total_loss=create_value(0.0,0);
    for(int i=0;i<10;i++){
        //one-hot encoding yapıyoruz
        double target = (i==target_label) ?1.0:0.0;
        Value* target_v = create_value(target,0);
        Value* diff = sub(predictions[i], target_v);//tahmin ile hedef arasındaki farkı hesaplıyoruz

        //farkın karesini alıyorum
        //negatif farklar pozitifleri yok etmesin diye
        Value* sq_diff=multiply(diff,diff);
        total_loss=add(total_loss,sq_diff);
    }
    return total_loss;
}

void update_weights(Layer* l, double learning_rate) {
    for(int i=0;i<l->nout;i++){
        Neuron* n = l->neurons[i];
        for(int j=0;j<n->nin;j++){//her inputun ağırlığını geziyoruz
            n->weight[j]->data -=learning_rate*n->weight[j]->grad;//Ağırlığı, hatayı azaltacak yönde güncelle
            n->weight[j]->grad=0;//bir sonraki resim için gradyanı sıfırladık
        }
        n->bias->data -=learning_rate*n->bias->grad;
        n->bias->grad=0;
    }
}
void clear_graph() {
    for (int i = 0; i < topo_p; i++) {
        Value* v = topo[i];
        v->visited = 0; // Bir sonraki resim için sıfırla
        
        if (v->is_param == 0) {
            free(v); // Geçici işlemse hafızadan sil
        } else {
            v->grad = 0; // Ağırlıksa sadece gradyanı sıfırla
        }
    }
    topo_p = 0; // Listeyi boşalt
}

void save_weights(Layer* l, char* filename){
    FILE* f=fopen(filename,"wb");//write binary
    if(!f) return;
    for(int i=0;i<l->nout;i++){//her bir nöron için dönüyoz
        Neuron* n= l->neurons[i];
        for(int j=0;j<n->nin;j++){//her bir ağırlık için dönüyoruz
            //sadece datayı kaydedeceğiz
            fwrite(&(n->weight[j]->data),sizeof(double),1,f);
        }
        fwrite(&(n->bias->data),sizeof(double),1,f);//nöronun biasını yazdık
    }
    fclose(f);
}

void load_weights(Layer* l, char* filename){
    FILE* f=fopen(filename,"rb");//read binary
    if(!f) return;
    for(int i=0;i<l->nout;i++){//her bir nöron için dönüyoz
        Neuron* n= l->neurons[i];
        for(int j=0;j<n->nin;j++){//her bir ağırlık için dönüyoruz
            //sadece datayı kaydedeceğiz
            fread(&(n->weight[j]->data),sizeof(double),1,f);
        }
        fread(&(n->bias->data),sizeof(double),1,f);//nöronun biasını yazdık
    }
    fclose(f);

}