import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import keras
import pandas as pd
import numpy as np
from PIL import Image
import os
import warnings
import keras
import imageio.v2 as imageio
import tensorflow as tf
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import Sequential
from mlxtend.evaluate import scoring,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix








labels=pd.read_csv('datasets/cifar10Labels.csv',index_col=0)
print(labels)

#view random image
# img_idx=17
# print(labels['label'][img_idx])
# image=imageio.imread(os.path.join('datasets/cifar10',str(img_idx)+'.png'))
# plt.imshow(image)
# plt.show()


#split the data into train and test followup with its transformation and normalization
ytrain,ytest=train_test_split(labels['label'],test_size=0.3,random_state=42)
train_index,test_index=ytrain.index,ytest.index #storing index for later use


temp=[]
for img_idx in ytrain.index:
    img_path=os.path.join('datasets/cifar10/',str(img_idx)+'.png')
    img=np.array(Image.open(img_path)).astype('float32')
    temp.append(img)

x_train=np.stack(temp)
temp=[]
for img_idx in ytest.index:
    img_path=os.path.join('datasets/cifar10/',str(img_idx)+'.png')
    img=np.array(Image.open(img_path)).astype('float32')
    temp.append(img)
x_test=np.stack(temp)

#normalize the data
x_train=x_train/255
x_test=x_test/255
print(x_train)

#label encode
lb=LabelEncoder()
ytrain=lb.fit_transform(ytrain)
ytest=lb.fit_transform(ytest)
ytrainlast=ytrain
ytrain=to_categorical(ytrain)
print(ytrain)

#define CNN model
num_classes=10
model=Sequential([

    #first convolutional layer
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',activation='relu',kernel_regularizer=keras.regularizers.l2(0.001),input_shape=(32,32,3),name='Conv_1'),
    #normalizing the parameters from last layer to speed up the performance (optinal)
    keras.layers.BatchNormalization(name='BN_1'),
    #adding first pooling layer
    keras.layers.MaxPool2D(pool_size=(2,2),name='MaxPool_1'),
    #adding second convolutional layer
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',activation='relu',kernel_regularizer=keras.regularizers.l2(0.001),name='Conv_2'),
    #normalizing
    keras.layers.BatchNormalization(name='BN_2'),
    #adding second pooling layer
    keras.layers.MaxPool2D(pool_size=(2,2),name='MaxPool_2'),
    #flatten input
    keras.layers.Flatten(name='Flat'),
    #Fully connected layer
    keras.layers.Dense(num_classes,activation='softmax',name='pred_layer')

])
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=tf.optimizers.Adam(),metrics=['accuracy'])
cpfile=r'CIFAR10_checkpoint.hdf5'
cb_checkpoint=keras.callbacks.ModelCheckpoint(cpfile,monitor='val_acc',verbos=1,save_best_only=True,mode='max')
epochs=5
model.fit(x_train,ytrain,epochs=epochs,validation_split=0.2,callbacks=[cb_checkpoint])



#predictions
pred_classes=tf.argmax(model.predict(x_test[30:100]),axis=1)
pred=lb.inverse_transform(pred_classes)
print(pred)
act_pred=ytest[30:100]

res=pd.DataFrame([pred,act_pred]).T
res.columns=['predicted','actual']
print(res)



#visualization using confusion matrix
train_x_classes=model.predict(x_train)
# print(ytrainlast)
# print(np.argmax(train_x_classes,axis=1))
train_acc=scoring(lb.inverse_transform(np.argmax(train_x_classes,axis=1)),lb.inverse_transform(ytrainlast))
# print(np.argmax(model.predict(x_test),axis=1).shape,ytest.shape)
# print()
tesr_acc=scoring(lb.inverse_transform(np.argmax(model.predict(x_test),axis=1)),lb.inverse_transform(ytest))
print(f'train accuracy: {train_acc} test accuracy: {tesr_acc}')


def plot_cm(cm,text):
    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(conf_mat=cm,colorbar=True,figsize=(8,8),cmap='Greens',show_absolute=False,show_normed=True)
    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names,rotation=45,fontsize=12)
    plt.yticks(tick_marks,class_names,fontsize=12)
    plt.xlabel('predicted label',fontsize=14)
    plt.ylabel('true label',fontsize=14)
    plt.title(text)
    plt.show()

train_cm=confusion_matrix(lb.inverse_transform(ytrainlast),lb.inverse_transform(np.argmax(train_x_classes,axis=1)))
test_cm=confusion_matrix(lb.inverse_transform(ytest),lb.inverse_transform(np.argmax(model.predict(x_test),axis=1)))
plot_cm(train_cm,'Confusion Matrix on train data')
plot_cm(test_cm,'Confusion Matrix on test data')
