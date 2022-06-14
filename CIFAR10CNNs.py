from __future__ import print_function
import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from keras import backend as K
from keras.optimizers import SGD

#set D as true to train and execute CNN D
#set D as false and E as true to train and execute CNN E
#set D as false and E as false to train and execute CNN F
D = True
E = False

train = 24000 
test = 33500
val = 2500

batch_size = 32
num_classes = 10
epochs = 15
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

x_train, Xtemp, y_train, ytemp = train_test_split(x, y, train_size=train, random_state=0)

x_test, X_prop_test, y_test, y_prop_test = train_test_split(Xtemp, ytemp, train_size=val, random_state=0)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

if D:
    print("model D")
    model.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(15, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dense(7, activation='relu'))
    model.add(Flatten())
    model.add(Dense(units = 10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

elif E:
    print("model E") 
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

else:
    print("model F")
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # example output part of the model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if E:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    model.fit(x_train, y_train_cat,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test_cat),
              shuffle=True)

# Score trained model.
X_prop_test = X_prop_test.astype('float32')
X_prop_test /= 255
y_prop_test_cat = keras.utils.to_categorical(y_prop_test, num_classes)

if E:
    scores = model.evaluate(X_prop_test, y_prop_test, verbose=1)
else:
    scores = model.evaluate(X_prop_test, y_prop_test_cat, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# print(X_prop_test.shape)

classification_v = model.predict_classes(x_test)
# print(classification_v)

score_final_v = model.predict(x_test)

classification = model.predict_classes(X_prop_test)
# print(classification)

score_final = model.predict(X_prop_test)

#training set printing
trainingset=pd.DataFrame()

for i in range(0, train):
    ma = np.matrix(x_train[i,:,:,0])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array0=pd.DataFrame(ar)

    ma = np.matrix(x_train[i,:,:,1])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array1=pd.DataFrame(ar)

    ma = np.matrix(x_train[i,:,:,2])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array2=pd.DataFrame(ar)

    array = pd.concat([array0, array1, array2], axis=1)

    array.insert(3072,'label',y_train[i])
    trainingset=trainingset.append(array)

trainingset.to_csv('training.csv', index = False, header = True)


#validation set printing
validationset=pd.DataFrame()

for i in range(0, val):
    ma = np.matrix(x_test[i,:,:,0])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array0=pd.DataFrame(ar)

    ma = np.matrix(x_test[i,:,:,1])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array1=pd.DataFrame(ar)

    ma = np.matrix(x_test[i,:,:,2])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array2=pd.DataFrame(ar)

    array = pd.concat([array0, array1, array2], axis=1)

    array.insert(3072,'label',y_test[i])
    array.insert(3073,'SUT',classification_v[i])
    array.insert(3074,'PredictedLabel0',score_final_v[i][0])
    array.insert(3075,'PredictedLabel1',score_final_v[i][1])
    array.insert(3076,'PredictedLabel2',score_final_v[i][2])
    array.insert(3077,'PredictedLabel3',score_final_v[i][3])
    array.insert(3078,'PredictedLabel4',score_final_v[i][4])
    array.insert(3079,'PredictedLabel5',score_final_v[i][5])
    array.insert(3080,'PredictedLabel6',score_final_v[i][6])
    array.insert(3081,'PredictedLabel7',score_final_v[i][7])
    array.insert(3082,'PredictedLabel8',score_final_v[i][8])
    array.insert(3083,'PredictedLabel9',score_final_v[i][9])
    validationset=validationset.append(array)

validationset.to_csv('validation.csv', index = False, header = True)


#test set printing
testset=pd.DataFrame()

for i in range(0, test):
    #print(Xtest_reshaped[i].shape)
    ma = np.matrix(X_prop_test[i,:,:,0])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array0=pd.DataFrame(ar)

    ma = np.matrix(X_prop_test[i,:,:,1])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array1=pd.DataFrame(ar)

    ma = np.matrix(X_prop_test[i,:,:,2])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array2=pd.DataFrame(ar)

    array = pd.concat([array0, array1, array2], axis=1)
   
    array.insert(3072,'label',y_prop_test[i])
    array.insert(3073,'SUT',classification[i])
    array.insert(3074,'PredictedLabel0',score_final[i][0])
    array.insert(3075,'PredictedLabel1',score_final[i][1])
    array.insert(3076,'PredictedLabel2',score_final[i][2])
    array.insert(3077,'PredictedLabel3',score_final[i][3])
    array.insert(3078,'PredictedLabel4',score_final[i][4])
    array.insert(3079,'PredictedLabel5',score_final[i][5])
    array.insert(3080,'PredictedLabel6',score_final[i][6])
    array.insert(3081,'PredictedLabel7',score_final[i][7])
    array.insert(3082,'PredictedLabel8',score_final[i][8])
    array.insert(3083,'PredictedLabel9',score_final[i][9])
    
    if(ytest[i] < 5):
        array.insert(3084,'EP', 0)
    else:
        array.insert(3084,'EP', 1)

    if(ytest[i] in (2, 3, 4, 5, 6, 7)):
        array.insert(3085,'FP', 0)
    else:
        array.insert(3085,'FP', 1)
    
    testset=testset.append(array)

testset.to_csv('test.csv', index = False, header = True)
