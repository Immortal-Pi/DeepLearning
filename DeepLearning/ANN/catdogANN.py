# get the data
# view sample images and classes
# transforming the dataset to a 1d array after reshaping all the images 32*32*3
# normalize the data
# knowing the distribution of class in data
# encoding the categorical variable to numeric
# building deep neural network
# predicting
# visualize inspection of prediction

import os
import re

import keras.layers
import numpy as np
from keras import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Flatten, Softmax, InputLayer
import imageio.v2 as imageio
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras_lr_finder as lr_find
from clr_callback import *
from keras.models import load_model


train_path_dogs = 'datasets/catdogdatasets/training_set/dogs/'
train_path_cats = 'datasets/catdogdatasets/training_set/cats/'
train_dataframe=pd.DataFrame()
temp_dog=[]
ytemp_dog=[]
img_names_dog=[]
for img_name in os.listdir(train_path_dogs):
    img_dog=imageio.imread(os.path.join(train_path_dogs,img_name))
    img=np.array(Image.fromarray(img_dog).resize((32,32))).astype('float32')
    temp_dog.append(img)
    img_names_dog.append(img_name)
    ytemp_dog.append(img_name.split('.')[0])




temp_cats=[]
ytemp_cats=[]
img_names_cats=[]
for img_name in os.listdir(train_path_cats):
    img_cats=imageio.imread(os.path.join(train_path_cats,img_name))
    img=np.array(Image.fromarray(img_cats).resize((32,32))).astype('float32')
    temp_dog.append(img)
    img_names_cats.append(img_name)
    ytemp_cats.append(img_name.split('.')[0])



train_dataframe['ID']=img_names_dog
train_dataframe['Class']=ytemp_dog
train_dataframe.to_csv('traindata.csv')



ytemp_all=ytemp_dog+ytemp_cats

x_train=np.stack(temp_dog)
y_train=np.stack(ytemp_all)
print(x_train.shape,y_train)



#normalize the data
x_train=x_train/255
lb=LabelEncoder()
y_train=lb.fit_transform(y_train)
y_train=to_categorical(y_train)


input_layer_size=(32,32,3)
hidden_layer_size=500
output_layer_size=2
epochs=100
dropout=0.30
batchsize=128
model = Sequential([
    keras.layers.Flatten(input_shape=input_layer_size),
    keras.layers.Dense(hidden_layer_size, activation='relu'),
    keras.layers.Dropout(dropout),
    keras.layers.Dense(output_layer_size, activation='softmax')
])
cb_triangular_lr=CyclicLR(base_lr=0.0001,max_lr=0.001,step_size=2000.,mode='triangular2')
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
cb_save=keras.callbacks.TensorBoard(log_dir='catdogmodel', write_graph=False)
model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batchsize, callbacks=[cb_triangular_lr,cb_save])
#optimization function check

# def optimization_function(optimization_list):
#     for i in range(len(optimization_list)):
#         model.compile(loss='categorical_crossentropy', optimizer=optimization_list[i], metrics=['accuracy'],)
#         val=re.search('optimizers\..*\so',string=str(optimization_list[i])).group(0)[11:][:-2]
#         logdir=r'optimz\\'+val
#         cb=keras.callbacks.TensorBoard(log_dir=logdir,write_graph=False)
#         model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batchsize, verbose=1,callbacks=[cb])
#
# optimization_list=[tf.optimizers.SGD(),tf.optimizers.Adam(),tf.optimizers.Adadelta(),tf.optimizers.RMSprop(),tf.optimizers.Adagrad()]
# optimization_function(optimization_list)



