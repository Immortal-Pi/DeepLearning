import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pandas as pd

#read
train_data=pd.read_csv('datasets/fashionmnisttrain.csv')
# print(train_data)


#display sample images
# img=train_data.iloc[5:6,1:].values.reshape(28,28)
# plt.imshow(img)
# plt.show()

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#creating validation data from test data
val_data=train_data.iloc[:5000,:]
test_data= train_data.iloc[5000:,:]

#fetching the labels
train_labels=train_data['label']
val_labels=val_data['label']
test_label=test_data['label']


#reshape the traning and validation data
train_images=train_data.iloc[:,1:].values.reshape(60000,28,28)

val_images=val_data.iloc[:,1:].values.reshape(5000,28,28)

#scaling the data in range of 0-1
train_images=train_images/255.0
val_images=val_images/255.0


# case 1 : neural network with one hidden layer without activation function
#case 1 validation accuracy is very low 10.22%

# model = keras.Sequential([
#     #conversion of higher dimensional data 2D - 1D
#     keras.layers.Flatten(input_shape=(28,28)),
#     #hidden layer with 1 neuron and linear activation function
#     keras.layers.Dense(1,activation=tf.keras.activations.linear),
#     #output layer
#     keras.layers.Dense(10,activation=keras.activations.linear)
# ])
#
# #defining parameters like optimizer, loss function and evaluating metric
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=keras.optimizers.Adam(),
#     metrics=['accuracy']
# )
#
# model.fit(train_images,train_labels,epochs=5,validation_data=(val_images,val_labels))




#case 2 : neural network with one hidden layer having 10 neurons
#case 2 validation accuracy is 42.03%
# model2=keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(10,activation=keras.activations.linear),
#     keras.layers.Dense(10,activation=keras.activations.linear)
# ])
# model2.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=keras.optimizers.Adam(),
#     metrics=['accuracy']
# )
# model2.fit(train_images,train_labels,epochs=5,validation_data=(val_images,val_labels))




#case 3 : introduce non-linearity to the above case to make it a clssification model
#case 3 validation accuracy is 85.20%
# model3=keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(10,activation=tf.nn.relu),
#     keras.layers.Dense(10,activation=tf.nn.softmax)
# ])
#
# model3.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=keras.optimizers.Adam(),
#     metrics=['accuracy']
# )
# model3.fit(train_images,train_labels,epochs=5,validation_data=(val_images,val_labels))



#case 4 : increase the hidden layers
#case validation accuracy is 86.78%

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(10,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(train_images,train_labels,epochs=5,validation_data=(val_images,val_labels))