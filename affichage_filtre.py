import os
import matplotlib as plt
import matplotlib.pyplot as plt 

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np

def display_image_filtered(name_image,model,layer_name,image):
    inp= model.inputs 
    out1= model.get_layer(layer_name).output  
    feature_map_1= Model(inputs= inp, outputs= out1)  
    img=cv2.resize(image,(200,200))              
    input_img= np.expand_dims(img, axis=0)      
    f=feature_map_1.predict(input_img) 
    dim = f.shape[3]
    print(f'{layer_name} | Features Shape: {f.shape}')
    print(f'Dimension {dim}')
    fig= plt.figure(figsize=(30,30))
    if not os.path.exists(f'results_{name_image}'):
        os.makedirs(f'results_{name_image}')        
    for i in range(dim):
        ax = fig.add_subplot(int(dim/2),int(dim/2),i+1)
        ax.axis('off')
        ax.imshow(f[0,:,:,i])
        plt.imsave(f'results_{name_image}/{name_image}_{layer_name}_{i}.jpg',f[0,:,:,i])

def display_filter(model, layer_name):
    layer = model.get_layer(layer_name)
    filter, bias= layer.get_weights()
    dim = filter.shape[3]
    print(f'{layer_name} | Filter Shape: {filter.shape} Bias Shape: {bias.shape}')
    print(f'Dimension {dim}')
    f_min, f_max = filter.min(), filter.max()
    filter = (filter - f_min) / (f_max - f_min)
    print(filter.shape)
    fig= plt.figure(figsize=(30,30))
    for i in range(dim):
        ax = fig.add_subplot(int(dim/2),int(dim/2),i+1)
        ax.axis('off')
        try:
            ax.imshow(filter[:,:,:,i])
        except:
            ax.imshow(filter[:,:,:,i][0])
    plt.show()       

if __name__ == "__main__":
    model = tf.keras.models.load_model('./model_save/dessert.h5') #  chemin de fichier 
    photo_1 = cv2.imread("./data_/data/cloudy/train_12.jpg", cv2.IMREAD_COLOR)
    photo_2 = cv2.imread("/Users/raphael/Documents/programming/programmation/CNN/data_/data/desert/desert(1).jpg", cv2.IMREAD_COLOR)
   
    num = 4 # number of layer
    for name in ['photo_1','photo_2']:
        if name == 'photo_1':
            image = photo_1
        elif name == 'photo_2':
            image = photo_2
        plt.axis('off')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        for i in range(1,num):
            if num == 0 and i==0:
                print('////////////////////////////////////////////////////////')
                print(f'{i+1}st convolutionnal layer')
                display_image_filtered(name,model,f'conv2d',image)
                print('--------')
                print(f'{i-1}nd Pooling')
                display_image_filtered(name,model,f'max_pooling2d',image)
                print('////////////////////////////////////////////////////////')
            else:
                print('#---------------------------------------------------#')
                print(f'{i+1}st convolutionnal layer')
                display_image_filtered(name,model,f'conv2d_{num-i}',image)
                print('--------')
                print(f'{i+1}nd Pooling')
                display_image_filtered(name,model,f'max_pooling2d_{num-i}',image)
                print('#---------------------------------------------------#')

                
    display_filter(model,f'conv2d_{num-1}')