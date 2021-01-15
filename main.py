#!/usr/bin/env python
# coding: utf-8

# In[51]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mlflow
import mlflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

import modules.images.utils as im


# In[52]:


# J'aimerais savoir la taille des images de notre jeu de données.
taille_image = cv2.imread("./data/test/apple/0001.png")
taille_image.shape


# In[53]:


image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

TRAIN_DATA_DIR = './data/train'
TRAIN_IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 32

train_generator = image_data_generator.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='training')
    
validation_generator = image_data_generator.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='validation')


# In[54]:


epochs = 1
patience = 8


# In[55]:


model = Sequential()

model.add(Conv2D(16, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same', input_shape=(taille_image.shape)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(32, activation='elu'))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=3, activation='elu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=5, activation='elu', kernel_initializer='he_uniform', padding='same'))
model.add(Dense(256, activation='elu'))

#Toujours à la fin
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

model.summary()


# In[56]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es_callback = EarlyStopping(monitor='val_loss', patience=patience)

training = model.fit_generator(train_generator, epochs=epochs, callbacks=[es_callback], validation_data=validation_generator, shuffle=False)


# In[57]:


with mlflow.start_run():

    # Log parameters
    mlflow.log_param("Epochs", epochs)
    mlflow.log_param("Patience", patience)
    
    # Log metrics
    mlflow.log_metric("Training Accuracy", training.history['accuracy'][-1])
    mlflow.log_metric("Validation Accuracy", training.history['val_accuracy'][-1])
    mlflow.log_metric("Training Loss", training.history['loss'][-1])
    mlflow.log_metric("Validation Loss", training.history['val_loss'][-1])
   
    # Log model
    mlflow.keras.log_model(model, "model")


# In[ ]:




