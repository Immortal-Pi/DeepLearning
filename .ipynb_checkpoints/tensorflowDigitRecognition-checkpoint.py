import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#neuralink method :- not predicting at all
mnist = tf.keras.datasets.mnist
print(tf.__version__)
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
# for i in range(1,20):
#     plt.imshow(xtrain[i])
#     plt.show()
#normalize the data to scale it down

xtrain = tf.keras.utils.normalize(xtrain,axis=1)
xtest = tf.keras.utils.normalize(xtest,axis=1)

#model the data
model=tf.keras.models.Sequential()

#add new layer (falt layer is 1D layer) each pixel of the input layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#2 hidden layers
#all the layers are connected between the previous layer
#units:- number of neurons
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))

#output layer
#unit:- number of neurons
#softmax:- sclaes the value down so that the average comes to 1
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


#compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=3)

loss,accuracy = model.evaluate(xtest,ytest)
print(f'accuracy: {accuracy}, loss: {loss}')
model.save('digits.model')


#predict images created in paint

for x in [0,2,3,4,6,8,9]:
    img=cv2.imread(f'data/numbers/{x}.png')[:,:,0]
    img=np.array([img])
    prediction = model.predict(img)
    print(f' result :{np.argmax(prediction)}')
    plt.imshow(img[0])
    plt.show()