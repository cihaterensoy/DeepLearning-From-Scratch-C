#include <stdint.h>
#include "../include/engine.h"
//bu dosyada disteki dosyayı acıp pikselleri tek tek okuyacağız

// MNIST dosyaları Big-Endian formatındadır, Intel işlemciler Little-Endian kullanır.
// Bu fonksiyon sayıları bizim anlayacağımız formata çevirir.
uint32_t swap_endian(uint32_t val){
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
    //BYTE SIRASINI DÜZELTİYORUM
}

//görüntüleri yükelyeceğiz DOSYA->BYTE->PIXEL-> NORMALIZE -> RAM
double** load_mnist_images(char* filename,int* num_images){
    FILE* f=fopen(filename,"rb");
    if(!f)
    {
        printf("Hata: Resim dosyasi acilamadi: %s\n", filename);
        return NULL;
    }
    uint32_t magic,n,rows,cols;//dosya türünü belirten sayıdır, kaç byte okunacak yani boyut,adet, hangi dosyadan okuyacağız(önceceden actığım fopen)
    fread(&magic,4,1,f); // dosyanın gerçekten MNIST image dosyası olup olmadığını belirtir
    fread(&n,4,1,f); // kaç tane resim var 
    fread(&rows,4,1,f);//yükseklik
    fread(&cols,4,1,f);//genişlik

    *num_images=swap_endian(n);//binary kaydırma yapıyoruz 
    int r=swap_endian(rows);//binary kaydırma
    int c = swap_endian(cols);//binary kaydırma
    int pixels_per_image=r*c;//mnist 28x28=784 özellik olacak sinir ağına girecek

    //bellekte resim boyutu kadar yer ayırıyoruz
    double** images =(double**)malloc((*num_images)*sizeof(double*));//pointerları tutacak bir pointer
    uint8_t* temp_pixels = (uint8_t*)malloc(pixels_per_image);// geçici olarak pixel verisini tutmak için

    for(int i=0;i<*num_images;i++){
        images[i]= (double*)malloc(pixels_per_image*sizeof(double));//resim boyutu kadar yer ayırdık hafızada// her resim için 784 double'lık yer ayır

        fread(temp_pixels,1,pixels_per_image,f);// dosyadan 784 byte oku (1 pixel = 1 byte)
        for(int j=0;j<pixels_per_image;j++){
            //pikselleri normalize ediyoruz
            images[i][j]=temp_pixels[j]/255.0;
        }
    }
    free(temp_pixels);// geçici buffer'ı temizle
    fclose(f);// dosyayı kapat
    return images;

}

//etiketleri yükleyen fonk
int* load_mnist_labels(char* filename,int* num_labels){
    FILE* f = fopen(filename,"rb");
    if(!f)return NULL;
    uint32_t magic,n;
    fread(&magic,4,1,f);
    fread(&n,4,1,f);
    *num_labels=swap_endian(n);// kaç tane label var

    int* labels=(int*)malloc((*num_labels)*sizeof(int));
    uint8_t label;
    for(int i=0;i<*num_labels;i++){
        fread(&label,1,1,f);
        labels[i]=(int)label;
    }
    fclose(f);
    return labels;
}