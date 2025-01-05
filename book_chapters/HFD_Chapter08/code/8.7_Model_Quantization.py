# Quantization is a powerful optimization technique that reduces the precision of model weights and activations, 
# enabling faster and more efficient inference. 
# This approach is particularly useful when deploying models on resource-constrained devices, such as mobile phones, 
# IoT devices, or embedded systems. 

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)
This example demonstrates TensorFlow Lite quantization, enabling faster inference on edge devices.

