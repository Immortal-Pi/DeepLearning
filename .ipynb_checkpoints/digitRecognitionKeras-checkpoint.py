import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from tensorflow.
# from tensorflow.keras.layers import Flatten, Dense
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  accuracy_score
import cv2

(xtrain,ytrain),(xtest,ytest)=keras.datasets.mnist.load_data()
#show images to know
# plt.imshow(xtrain[4])
# print(ytrain[4])
# plt.show()

#scale the dataset from 0-1
xtrain=xtrain/255
xtest=xtest/255

#build model
model=tf.keras.Sequential()

#convet the higher dimension layer to 1d array
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#dense layer
model.add(tf.keras.layers.Dense(128,activation='relu'))

#output layer
model.add(tf.keras.layers.Dense(10,activation='softmax'))
print(model.summary())

#compile the module
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#fit the model
history=model.fit(xtrain,ytrain,batch_size=64,epochs=10,verbose=1,validation_split=0.2)


#evaluate the model
print(model.evaluate(xtest,ytest))


#prediction
y_prob=model.predict(xtest)
print(y_prob.argmax(axis=1))
# print(ytest[0:3])
# print(accuracy_score(ytest,y_prob))
# for x in [0,2,3,4,6,8,9]:
#     img=cv2.imread(f'data/numbers/{x}.png')[:,:,0]
#     img=np.array([img])
#     prediction = model.predict(img)
#     print(f' result :{np.argmax(prediction)}')
#     plt.imshow(img[0])
#     plt.show()


#training loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('epochs')

plt.legend()

#accuracy plot
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('epochs')
plt.legend()
plt.tight_layout()
plt.show()