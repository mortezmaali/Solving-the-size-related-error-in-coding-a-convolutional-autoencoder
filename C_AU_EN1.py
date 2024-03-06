# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:20:44 2021

@author: mamiri
"""

import tensorflow as tf
from keras.layers import MaxPooling2D, Add, LeakyReLU, BatchNormalization,Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, UpSampling2D, InputLayer
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K
import keras as keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv2
import scipy.io
from scipy.io import savemat
import shutil
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam


cleaned_image = scipy.io.loadmat('C:/Users/Morteza/Desktop/PhD/Clean_n.mat')
cleaned_image = cleaned_image['image_c_n']

Unclean = scipy.io.loadmat('C:/Users/Morteza/Desktop/PhD/Unclean_n.mat')
Unclean = Unclean['image_y_n']  

#Y = Y.reshape(34, Y.shape[0], Y.shape[1], 2)
X = np.array(Unclean)

#pdb.set_trace()

x1 = keras.Input(shape=(100, 150, 3))

x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
x3 = Conv2D(8, (3, 3), activation='sigmoid', padding='same', strides=2)(x2)
x4 = Conv2D(8, (3, 3), activation='relu', padding='same')(x3)
x5 = Conv2D(8, (3, 3), activation='sigmoid', padding='same', strides=2)(x4)

x6 = Conv2D(8, (3, 3), activation='tanh', padding='same')(x5)

x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(8, (3, 3), activation='tanh', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)

x10 = Conv2D(3, (3, 3), activation='tanh', padding='same')(x9)

s1n = x10.get_shape().as_list()
resn = []
for val in s1n:
    if val != None :
        resn.append(val)
        
x10 = tf.reshape(x10,(resn[0], resn[1],resn[2]))
x10 = tf.image.resize(x10,[100,150])
x10=tf.reshape(x10,(1,100,150,3))

model = keras.Model(x1, x10)

Y = np.array(cleaned_image)

X=np.reshape(X,(1,100, 150,3))
Y=np.reshape(Y,(1,100, 150,3))
#output = model.predict(X)

model.compile(optimizer='Adam', loss='mse')
model.fit(X, Y, epochs=2)
output = model.predict(X)
    #Xp=tf.math.real(output)

output = np.reshape(output,(100, 150, 3))

plt.imshow(output)

savemat("output2n3.mat", {"foo":output})