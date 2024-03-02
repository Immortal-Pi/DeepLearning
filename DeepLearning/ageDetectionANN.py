import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import utils
from keras.models import Sequential
from keras.layers import Dense,Flatten,InputLayer
import keras
from keras.utils import np_utils
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image
from keras import Sequential


#get the data
# print(tf.version.VERSION)
train=pd.read_csv('datasets/train.csv')
test=pd.read_csv('datasets/test.csv')


#display random movie character with age group
# np.random.seed(10) #subsequent calls for np.random will produce the same sequence of random numbers
# idx=np.random.choice(train.index) #get random index
# # print(idx)
# img_name=train['ID'][idx]
# img=imageio.imread(os.path.join('datasets/Train',img_name))
# print(f'age group: {train["Class"][idx]}')
# plt.imshow(img)
# plt.axis('off')
# plt.show()


#transforming the dataset to a 1d array after reshaping all the images 32*32*3
temp=[]
for img_name in train['ID']:
    img_path=os.path.join('datasets/Train',img_name)
    img=imageio.imread(img_path)
    img=np.array(Image.fromarray(img).resize((32,32))).astype('float32')
    temp.append(img)

train_x=np.stack(temp)
# print(train_x)

temp=[]
for img_name in test['ID']:
    img_path=os.path.join('datasets/Test',img_name)
    img=imageio.imread(img_path)
    img=np.array(Image.fromarray(img).resize((32,32))).astype('float32')
    temp.append(img)
test_x=np.stack(temp)
print(test_x)


#normalize the data
train_x=train_x/255.
test_x=test_x/255

#knowing the distribution of class in data
print(train['Class'].value_counts(normalize=True))


#encoding the categorical variable to numeric
lb=LabelEncoder() #encode to ineger values
train_y=lb.fit_transform(train['Class']) #integer encoded labels
# print(train_y)
train_y=np_utils.to_categorical(train_y) #converts the integer encoded label to binary matrix (suitable for deeplearning models)
# print(train_y)




#building deep neural network
input_num_units= (32,32,3)
hidden_num_units=500
output_num_units=3
epochs=5
batch_size=128

#try kersas.Sequential
model=Sequential([
    # InputLayer(input_shape=input_num_units),Flatten(),
    # Dense(units=hidden_num_units,activation='relu'),
    # Dense(units=output_num_units,activation='softmax')
    keras.layers.Flatten(input_shape=input_num_units),
    keras.layers.Dense(hidden_num_units,activation='relu'),
    keras.layers.Dense(output_num_units,activation='softmax')
])
model.summary()
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1)

#accuracy is 62.95% it is recommended that we use 20% to 30% as validation data for checking models work on unseen data
model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)

#accuracy is low but since validation accuracy is similar to traning accuracy its not overfitted



#predicting and importing the results in a csv file
pred=model.predict(test_x)
pred_classes=tf.argmax(pred,axis=1)
print(pred_classes)

#visualize inspection of prediction
idx=2483
img_name=test['ID'][idx]
img=imageio.imread(os.path.join('datasets/Test',img_name))
plt.imshow(np.array(Image.fromarray(img).resize((128,128))))
pred=model.predict(test_x)
pred_classes=tf.argmax(pred,axis=1)
pred_classes_original=lb.inverse_transform(pred_classes)
print(f'Original:{train["Class"][idx]} Prediction: {pred_classes_original[idx]}')

plt.show()
# print(pred_classes)
# idx=2481
# img_name=test['ID'][idx]
# i