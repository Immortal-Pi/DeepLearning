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

train=pd.read_csv('datasets/train.csv')
test=pd.read_csv('datasets/test.csv')

temp=[]
for i in range(len(train['ID'])):
    img_path=os.path.join('datasets/Train', train['ID'][i])
    img=imageio.imread(img_path)
    img=np.array(Image.fromarray(img).resize((32,32))).astype('float32')
    temp.append(img)
temp=[]
for i in range(len(test['ID'])):
    img_path=os.path.join('datasets/Test', test['ID'][i])
    img=imageio.imread(img_path)
    img=np.array(Image.fromarray(img).resize((32,32))).astype('float32')
    temp.append(img)

x_train=np.stack(temp)
x_test=np.stack(temp)

x_train=x_train/255
x_test=x_test/255


lb=LabelEncoder()
y_train=lb.fit_transform(train['Class'])
y_train=to_categorical(y_train)
# print(y_train)

input_unit_size=(32,32,3)
hidden_unit_size=500
output_unit_size=3
batch_size=128
epochs=21
dropout=0.30

model=Sequential([
    keras.layers.Flatten(input_shape=input_unit_size),
    keras.layers.Dense(units=hidden_unit_size,activation='relu',kernel_initializer=tf.initializers.HeNormal(),
                       bias_initializer=tf.initializers.constant(value=0.01),kernel_regularizer=keras.regularizers.l2()),
    keras.layers.Dropout(dropout),
    keras.layers.Dense(units=output_unit_size,activation='softmax',kernel_initializer=tf.initializers.HeNormal())
])
cb_triangular_lr=CyclicLR(base_lr=0.0001,max_lr=0.001,step_size=2000.,mode='triangular2')
model.compile(loss='categorical_crossentropy',optimizer=tf.optimizers.Adam(),metrics=['accuracy'])
cb_save=keras.callbacks.TensorBoard(log_dir='optimal_model', write_graph=False)
# model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,verbose=1,callbacks=[cb_triangular_lr,cb_save])

# model.save('optimum_model.h5')
model=load_model('optimum_model.h5')
#visual inspection of preductions
idx=4014
img_name=test['ID'][idx]
img=imageio.imread(os.path.join('datasets/Test', test['ID'][idx]))
plt.imshow(np.array(Image.fromarray(img).resize((128,128))))
pred=model.predict(x_test)
pred_res=tf.argmax(pred,axis=1)
# print(pred_res)
print(f'Original:{train["Class"][idx]}{train["ID"][idx]} Predicted: {lb.inverse_transform(pred_res)[idx]}{test["ID"][idx]}')
plt.show()


