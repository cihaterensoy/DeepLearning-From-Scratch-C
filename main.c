#include "include/engine.h"

int main() {
    Value* a = create_value(-4.0);
    Value* b = create_value(2.0);
    
    // (a * b) -> ReLU
    Value* c = multiply(a, b); // -8.0
    Value* d = relu(c);        // 0.0
    
    apply_backward(d);
    
    printf("Sonuc: %f\n", d->data); // 0.0 bekliyoruz
    printf("a'nin etkisi (grad): %f\n", a->grad); // 0.0 bekliyoruz (ReLU kapalı)
    
    return 0;
}