#ifdef __MINGW32__
    #include <sys/stat.h>
    #define stat64i32 _stat
#endif

#include "raylib.h"
#include "include/engine.h" 
#include <stdio.h>
#include <float.h>
#include <stdbool.h>

#define CANVAS_SIZE 280
#define MNIST_SIZE 28

// --- OTOMATİK MERKEZLEME FONKSİYONU ---
void CenterAndScaleDigit(Image *img) {
    int minX = img->width, minY = img->height, maxX = 0, maxY = 0;
    bool found = false;
    Color *pixels = LoadImageColors(*img);

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            if (pixels[y * img->width + x].r > 20) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                found = true;
            }
        }
    }
    UnloadImageColors(pixels);
    if (!found) return;

    Rectangle bounds = { (float)minX, (float)minY, (float)(maxX - minX + 1), (float)(maxY - minY + 1) };
    ImageCrop(img, bounds);

    int targetSize = 20; 
    float scale = (float)targetSize / (bounds.width > bounds.height ? bounds.width : bounds.height);
    ImageResize(img, (int)(bounds.width * scale), (int)(bounds.height * scale));

    Image canvas = GenImageColor(MNIST_SIZE, MNIST_SIZE, BLACK);
    int offsetX = (MNIST_SIZE - img->width) / 2;
    int offsetY = (MNIST_SIZE - img->height) / 2;
    ImageDraw(&canvas, *img, (Rectangle){ 0, 0, (float)img->width, (float)img->height }, 
              (Rectangle){ (float)offsetX, (float)offsetY, (float)img->width, (float)img->height }, WHITE);
    
    UnloadImage(*img);
    *img = canvas;
}

int main() {
    // Ekranı biraz genişletiyoruz ki olasılık tablosu sığsın
    const int screenWidth = 850;
    const int screenHeight = 500;
    InitWindow(screenWidth, screenHeight, "C-Neural Core: Analiz Modu");
    SetTargetFPS(60);

    Layer* hidden = create_layer(784, 64); 
    Layer* output = create_layer(64, 10);

    load_weights(hidden, "hidden_weights.bin");
    load_weights(output, "output_weights.bin");

    Value* inputs[784];
    for (int j = 0; j < 784; j++) {
        inputs[j] = create_value(0.0, 1); 
    }

    RenderTexture2D canvas = LoadRenderTexture(CANVAS_SIZE, CANVAS_SIZE);
    BeginTextureMode(canvas);
    ClearBackground(BLACK);
    EndTextureMode();

    // Analiz için değişkenler
    float confidences[10] = { 0 };
    Image modelViewImg = GenImageColor(MNIST_SIZE, MNIST_SIZE, BLACK);
    Texture2D modelViewTex = LoadTextureFromImage(modelViewImg);
    int predictedDigit = -1;

    while (!WindowShouldClose()) {
        // --- INPUT ---
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePos = GetMousePosition();
            if (CheckCollisionPointRec(mousePos, (Rectangle){10, 10, CANVAS_SIZE, CANVAS_SIZE})) {
                BeginTextureMode(canvas);
                DrawCircle(mousePos.x - 10, mousePos.y - 10, 12, WHITE);
                EndTextureMode();
            }
        }

        if (IsKeyPressed(KEY_R)) {
            BeginTextureMode(canvas);
            ClearBackground(BLACK);
            EndTextureMode();
            predictedDigit = -1;
            for(int i=0; i<10; i++) confidences[i] = 0;
        }

        // --- TAHMİN VE ANALİZ ---
        if (IsKeyPressed(KEY_SPACE)) {
            Image img = LoadImageFromTexture(canvas.texture);
            ImageFlipVertical(&img); 
            CenterAndScaleDigit(&img); 
            
            // Modelin gördüğü resmi önizleme için sakla
            UnloadTexture(modelViewTex);
            modelViewTex = LoadTextureFromImage(img);
            
            Color* pixels = LoadImageColors(img);
            for(int i = 0; i < 784; i++) {
                inputs[i]->data = (double)pixels[i].r / 255.0;
            }

            Value** h_out = forward_layer(hidden, inputs);
            Value** final_out = forward_layer(output, h_out);
            
            // Olasılıkları hesapla (Softmax benzeri normalizasyon)
            double sum = 0;
            for(int i = 0; i < 10; i++) sum += final_out[i]->data;
            
            double maxVal = -DBL_MAX;
            for(int i = 0; i < 10; i++) {
                if(sum > 0) confidences[i] = (float)(final_out[i]->data / sum);
                else confidences[i] = 0;

                if(final_out[i]->data > maxVal) {
                    maxVal = final_out[i]->data;
                    predictedDigit = i;
                }
            }

            for(int i = 0; i < 10; i++) build_topo(final_out[i]); 
            clear_graph(); 
            free(h_out); 
            free(final_out);
            UnloadImage(img);
            UnloadImageColors(pixels);
        }

        // --- RENDER ---
        BeginDrawing();
            ClearBackground(GetColor(0x111111FF));
            
            // 1. Çizim Alanı
            DrawText("ÇİZİM ALANI", 10, 300, 20, GOLD);
            DrawRectangleLines(9, 9, CANVAS_SIZE + 2, CANVAS_SIZE + 2, RAYWHITE);
            DrawTextureRec(canvas.texture, (Rectangle){0, 0, (float)canvas.texture.width, (float)-canvas.texture.height}, (Vector2){10, 10}, WHITE);
            
            // 2. Modelin Gördüğü (28x28 Önizleme)
            DrawText("MODELİN GÖRDÜĞÜ", 310, 10, 18, SKYBLUE);
            DrawRectangleLines(309, 39, 114, 114, GRAY);
            // 28x28'lik resmi 4 kat büyüterek gösteriyoruz
            DrawTextureEx(modelViewTex, (Vector2){310, 40}, 0, 4.0f, WHITE);
            
            // 3. Olasılık Tablosu
            DrawText("TAHMİN OLASILIKLARI", 460, 10, 18, SKYBLUE);
            for(int i = 0; i < 10; i++) {
                Color barColor = (i == predictedDigit) ? YELLOW : DARKGRAY;
                
                // Rakam etiketi
                DrawText(TextFormat("%d:", i), 460, 45 + (i * 35), 20, WHITE);
                
                // Olasılık barı
                DrawRectangle(490, 45 + (i * 35), 250, 20, BLACK); // Arka plan
                DrawRectangle(490, 45 + (i * 35), (int)(confidences[i] * 250), 20, barColor); // Doluluk
                
                // Yüzde metni
                DrawText(TextFormat("%%%0.1f", confidences[i] * 100), 750, 45 + (i * 35), 18, barColor);
            }

            // 4. Büyük Sonuç Paneli
            DrawRectangle(310, 200, 120, 120, DARKGRAY);
            DrawRectangleLines(310, 200, 120, 120, GOLD);
            if(predictedDigit != -1) {
                DrawText(TextFormat("%d", predictedDigit), 350, 225, 80, YELLOW);
            } else {
                DrawText("-", 355, 225, 80, GRAY);
            }
            DrawText("SONUÇ", 335, 330, 20, GOLD);

            // Alt Bilgi
            DrawText("SPACE: Analiz Et | R: Temizle", 10, 460, 16, LIGHTGRAY);
            DrawText("Mimari: 784-64-10 | Pure C", 640, 475, 14, DARKGRAY);
            
        EndDrawing();
    }

    for(int j = 0; j < 784; j++) free(inputs[j]);
    UnloadTexture(modelViewTex);
    UnloadRenderTexture(canvas);
    CloseWindow();
    return 0;
}