# This example demonstrates TensorFlow’s MirroredStrategy,
# where data parallelism is achieved by splitting the dataset across multiple GPUs. 
# Each GPU trains the model on its subset of data, and the gradients are synchronized after each batch.

import tensorflow as tf 

strategy = tf.distribute.MirroredStrategy()
with strategy.scope(): 
model = tf.keras.Sequential([ 
      tf.keras.layers.Embedding(input_dim=10000, output_dim=256), 
      tf.keras.layers.GlobalAveragePooling1D(), 
      tf.keras.layers.Dense(1, activation='sigmoid') 
]) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   
model.fit(x_train, y_train, epochs=5, batch_size=512)
