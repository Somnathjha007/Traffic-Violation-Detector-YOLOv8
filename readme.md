# Traffic Light & Zebra Crossing Detection with Violation Monitoring

**&**

# Quantized TFLite Model Pipeline for Embedded ML

---

##  Traffic Light & Zebra Crossing Detection with Violation Monitoring

This project provides a solution to detect traffic lights (red/green) and zebra crossings using YOLOv8 models. It also performs traffic violation detection based on red light presence, zebra crossing detection, and vehicle speed.

### Features

#### Standalone Detection (Image-based)

* Detects red traffic lights and zebra crossings in a local image.
* Checks for violations when red light is on, no zebra crossing is detected, and vehicle speed > 0.
* Visualizes detections and violation status with bounding boxes and labels on images.
* Ideal for quick testing and prototyping in a Jupyter notebook.

#### Flask API Server

* Hosts a REST API to upload images and process them remotely.
* Returns violation status based on the uploaded image.
* Supports motor control commands with endpoints to set and get commands, designed for ESP32 or similar hardware integration.
* Saves annotated images for review.

### Getting Started

#### Prerequisites

* Python 3.7 or higher
* Install required packages:

  
  pip install ultralytics opencv-python numpy matplotlib flask
  

#### Models

* Download and place the YOLOv8 model files inside the `models/` directory:

  * `traffic_light_model.pt` (for detecting traffic lights)
  * `zebra_crossing_yolov8.pt` (for detecting zebra crossings)

### Usage

#### Standalone Detection (In Notebook)

* Load an image.
* Run detection and violation check functions.
* View output image with bounding boxes and violation status inline.

#### Running Flask Server

* Run the Flask app to start the API server:

  
  python app.py
  
* API Endpoints:

  * `POST /upload` — Upload an image for violation detection.
  * `POST /set_command` — Set motor control command (`F`, `B`, `S`, `R`, `L`).
  * `GET /get_command` — Retrieve the current motor command.
* The server listens on `http://0.0.0.0:5000` by default, allowing ESP32 or other clients to connect.

### How Violation Detection Works

1. Detect if a **red traffic light** is present in the image.
2. If red light is detected, check if a **zebra crossing** is also present.
3. If red light is on and no zebra crossing detected, check if the vehicle is moving (`SPEED > 0`).
4. Violation is flagged if the vehicle is moving through a red light without a zebra crossing.
5. Annotated images and results are saved or returned depending on the mode.

### Notes

* The `SPEED` variable simulates vehicle speed and can be updated to reflect actual inputs.
* Annotated images are saved under the `uploads/` directory.
* Adjust confidence thresholds in the code as needed.
* Designed for integration with embedded systems for real-time traffic violation monitoring.

---

##  Quantized TFLite Model Pipeline for Embedded ML

This project covers a complete workflow to prepare a deep learning model for deployment on **microcontrollers or edge devices** using TensorFlow Lite for Microcontrollers (TFLM).

### Overview

The pipeline includes:

1.  **Convert a Keras model to a fully quantized INT8 TFLite model**
2.  **Convert that `.tflite` model into a C-style header (`model.h`)**
3.  **Estimate the tensor arena size needed for microcontroller deployment**

---

### 1. Keras to Quantized TFLite (INT8)

**Script: `quantize_model.py`**

```python
import tensorflow as tf
import numpy as np

# Load your original Keras model (.h5 or SavedModel)
model_path = r"D:\project\red_green_classifier.h5"
model = tf.keras.models.load_model(model_path)

# Create a converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for INT8 calibration
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert and save
quantized_model = converter.convert()
with open("model_int8.tflite", "wb") as f:
    f.write(quantized_model)

print("✅ Successfully converted to model_int8.tflite")
```

---

### 2. Convert TFLite Model to `model.h` (C Array)

```python
import numpy as np

# Load TFLite model
with open("model_int8.tflite", "rb") as f:
    tflite_model = f.read()

# Convert to hex array
hex_array = ", ".join(f"0x{b:02x}" for b in tflite_model)

# Write to header file
header_content = f"""\
#ifndef MODEL_H
#define MODEL_H

#include <cstdint>

alignas(8) const unsigned char g_model[] = {{
    {hex_array}
}};

const int g_model_len = {len(tflite_model)};

#endif  // MODEL_H
"""

with open("model.h", "w") as f:
    f.write(header_content)

print("✅ model.h file created successfully!")
```

---

### 3. Estimate Tensor Arena Size

```python
import tensorflow.lite as tflite
import numpy as np

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

# Estimate total tensor memory
tensor_details = interpreter.get_tensor_details()
arena_size = sum(np.prod(t['shape']) * np.dtype(t['dtype']).itemsize for t in tensor_details)

print(f"🔹 Estimated Tensor Arena Size: {arena_size} bytes")


