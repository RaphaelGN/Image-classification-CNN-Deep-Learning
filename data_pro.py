import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model



import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime


#import dataset
import pathlib
import os


from tensorflow.keras.utils import load_img, img_to_array
#from keras.preprocessing.image import img_to_array
import warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

def get_data(chemin):

    data_dir = pathlib.Path(chemin)
    return data_dir 
def datagen_dataset(train_path,test_path):

    datagen = ImageDataGenerator()
    train_dataset = datagen.flow_from_directory(train_path,class_mode = 'binary')
    test_dataset = datagen.flow_from_directory(test_path,class_mode = 'binary')
    return train_dataset, test_dataset
def parameters_fit(train_dataset,test_dataset,data_dir):
    batch_size = 3
    img_height = 200
    img_width = 200

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = val_data.class_names
    print(class_names)
    return train_data,val_data, class_names

def affichage_image(train_data, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):
        for i in range(3):
            ax = plt.subplot(1, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()        
    return 
def build_model():
    num_classes = 2
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(128,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],)

    return model
def parameters_fit_tf(train_data,val_data):

    model=build_model()

    logdir="logs"    
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
                                                   embeddings_data=train_data)

    history=model.fit( 
        train_data,
        validation_data=val_data,
        epochs=2,
        callbacks=[tensorboard_callback]
    )
    return history,logdir,tensorboard_callback, model

if __name__ == "__main__":
    chemin='./data_/data/'
  
    train_path="/Users/raphael/Documents/programming/programmation/CNN/data_/data/"

    test_path="/Users/raphael/Documents/programming/programmation/CNN/data_/data/pikachu/"

    data_dir=get_data(chemin)
    train_dataset,test_dataset=datagen_dataset(train_path,test_path)
    train_data,val_data,class_names=parameters_fit(train_dataset,test_dataset,data_dir)
    
    #affichage_image(train_data,class_names)
    history,logdir,tensorboard_callback,model=parameters_fit_tf(train_data,val_data)
   # history.summary()
    model.save('./model_save/dessert.h5') 
    model.save_weights('./model_save/weights.h5')
    print(history.history)
    plt.plot(history.history["val_accuracy"],color="r",label="val_accuracy")
    plt.title("Accuracy Graph")
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    plt.show()

