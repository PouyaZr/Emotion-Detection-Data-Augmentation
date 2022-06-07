import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt 
import os

train_path = "C:\\Users\\pouya\\Desktop\\emotion detection\\train"
test_path = "C:\\Users\\pouya\\Desktop\\emotion detection\\test"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                          rotation_range = 10,
                                                          horizontal_flip=True,
                                                          width_shift_range=0.1,
                                                          height_shift_range=0.1,
                                                          fill_mode = 'nearest')

train = datagen.flow_from_directory(train_path, target_size=(48, 48), 
                                    class_mode="sparse", 
                                    color_mode="grayscale", 
                                    batch_size=64
                                    )
test = datagen.flow_from_directory(test_path, 
                                   target_size=(48, 48), 
                                   class_mode="sparse", 
                                   color_mode="grayscale", 
                                   batch_size=64
                                   )

net = models.Sequential([
                         layers.Conv2D(64, (3, 3), activation='relu', input_shape = (48, 48, 1)),
                         layers.BatchNormalization(),
                         layers.Conv2D(64, (3, 3), activation='relu'),
                         layers.BatchNormalization(),

                         layers.MaxPool2D(),

                         layers.Conv2D(128, (3, 3), activation='relu', input_shape = (48, 48, 1)),
                         layers.BatchNormalization(),
                         layers.Conv2D(128, (3, 3), activation='relu'),
                         layers.BatchNormalization(),

                         layers.MaxPool2D(),

                         layers.Flatten(),
                         layers.Dense(64, activation='relu'),
                         layers.BatchNormalization(),
                         layers.Dense(7, activation='softmax')
                         ])
net.summary()


net.compile(
            optimizer="sgd", 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
           )

H = net.fit_generator(generator=train,
                      steps_per_epoch=28709//64,
                      epochs=50,
                      validation_data=test,
                      validation_steps=7178//64)

net.save("emotion_model.h5")

plt.style.use('ggplot')
plt.plot(H.history["accuracy"], label = 'train accuracy')
plt.plot(H.history["val_accuracy"], label = 'test accuracy')
plt.plot(H.history["loss"], label = 'train loss')
plt.plot(H.history["val_loss"], label = 'test loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Accuracy/Loss')
plt.show()