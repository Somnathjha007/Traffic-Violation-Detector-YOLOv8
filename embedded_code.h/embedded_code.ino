#include <Arduino.h>
#include <esp_heap_caps.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_allocator.h>

#include "model_data.h"  // Include the TensorFlow Lite model array

// Allocate model in PSRAM
void* AllocateModelInPSRAM(size_t size) {
    void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    if (!ptr) {
        Serial.printf("Failed to allocate %d bytes in PSRAM\n", size);
    } else {
        Serial.printf("Allocated %d bytes in PSRAM at address: %p\n", size, ptr);
    }
    return ptr;
}

// Allocate Tensor Memory in PSRAM
void* AllocateTensorInPSRAM(size_t size) {
    void* ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    if (!ptr) {
        Serial.printf("Failed to allocate %d bytes in PSRAM\n", size);
    } else {
        Serial.printf("Allocated %d bytes in PSRAM at address: %p\n", size, ptr);
    }
    return ptr;
}

// Setup TensorFlow Lite Model
void SetupModelInPSRAM(const tflite::Model* model) {
    Serial.println("Initializing TensorFlow Lite...");

    // Define the operation resolver
    tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddLogistic();  // Ensure LOGISTIC operation is added

    // Allocate memory for tensors in PSRAM
    constexpr size_t kTensorArenaSize = 1024 * 400;  // Adjust size as needed
    uint8_t* tensor_arena = (uint8_t*)AllocateTensorInPSRAM(kTensorArenaSize);
    
    
    if (!tensor_arena) {
        Serial.println("Failed to allocate tensor arena in PSRAM!");
        return;
    }

    // Create the TensorFlow Lite MicroInterpreter
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // Allocate tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        return;
    }

    Serial.println("Tensor allocation successful in PSRAM!");
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("Allocating model in PSRAM...");

    // Allocate memory for model in PSRAM
    void* model_psram = AllocateModelInPSRAM(sizeof(g_model));
    if (!model_psram) {
        Serial.println("Failed to allocate model in PSRAM!");
        return;
    }

    // Copy model data to PSRAM
    memcpy(model_psram, g_model, sizeof(g_model));

    // Load the model from PSRAM
    const tflite::Model* model = tflite::GetModel(model_psram);
    if (model == nullptr) {
        Serial.println("Failed to load model!");
        return;
    }

    Serial.println("Model successfully stored in PSRAM!");

    // Set up the TensorFlow Lite interpreter with the model
    SetupModelInPSRAM(model);

    Serial.println("Model setup complete!");
}

void loop() {
    // Placeholder for inference logic
}
