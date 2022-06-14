from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D, AveragePooling2D
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K

#set A as true to train and execute CNN A
#set A as false and B as true to train and execute CNN B
#set A as false and B as false to train and execute CNN C
A = True
B = False

train = 28000
val = 2500
test = 70000 - (train+val)

print('Downloading MNIST…')
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
print('Done!')

x = np.concatenate((Xtrain, Xtest))
y = np.concatenate((ytrain, ytest))

Xtrain, Xtemp, ytrain, ytemp = train_test_split(x, y, train_size=train, random_state=0)

Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, train_size=val, random_state=0)

tot_train_examples = train
tot_test_examples = val
width=28
height=28
channels = 1
f_size1 = 32 
f_size2= 16  

Xtrain_reshaped = Xtrain.reshape(tot_train_examples,width,height,channels)
Xval_reshaped = Xval.reshape(tot_test_examples,width,height,channels)

print('New shape ',Xtrain_reshaped[0].shape)

y_train_cat = to_categorical(ytrain)
y_val_cat = to_categorical(yval)
print(y_train_cat[0])

model = Sequential()

if A:
    print("model A")
    model.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=(width,height,channels)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(15, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dense(7, activation='relu'))
    model.add(Flatten())
    model.add(Dense(units = 10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

elif B:
    print("model B")
    model.add(Conv2D(f_size1, kernel_size=3, activation='relu', input_shape=(width,height,channels)))
    model.add(Dropout(0.3))
    model.add(Conv2D(f_size2, kernel_size=3, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

else:
    print("model C")
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])

history = model.fit(Xtrain_reshaped, y_train_cat, validation_data=(Xval_reshaped, y_val_cat), epochs=4,batch_size=128,shuffle=True)

Xtest_reshaped = Xtest.reshape(test,width,height,channels)
y_test_cat = to_categorical(ytest)

#print(Xtest_reshaped.shape)

classification_v = model.predict_classes(Xval_reshaped)
#print(classification_v)

score_final_v = model.predict(Xval_reshaped)

classification = model.predict_classes(Xtest_reshaped)
#print(classification)

score_final = model.predict(Xtest_reshaped)

#print(score_final[0][0])

get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
layer_output = get_3rd_layer_output(np.expand_dims(Xtest_reshaped[0], axis=0))[0]

#print(layer_output.shape)

#training set csv printing
trainingset=pd.DataFrame()

for i in range(0, train):
    ma = np.matrix(Xtrain_reshaped[i])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array=pd.DataFrame(ar)
    array.insert(784,'label',ytrain[i])
    trainingset=trainingset.append(array)

trainingset.to_csv('training.csv', index = False, header = True)
print("training.csv completed")

#validation set csv printing
validationset=pd.DataFrame()

for i in range(0, val):
    ma = np.matrix(Xval_reshaped[i])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array=pd.DataFrame(ar)
    array.insert(784,'label',yval[i])
    array.insert(785,'SUT',classification_v[i])
    array.insert(786,'PredictedLabel0',score_final_v[i][0])
    array.insert(787,'PredictedLabel1',score_final_v[i][1])
    array.insert(788,'PredictedLabel2',score_final_v[i][2])
    array.insert(789,'PredictedLabel3',score_final_v[i][3])
    array.insert(790,'PredictedLabel4',score_final_v[i][4])
    array.insert(791,'PredictedLabel5',score_final_v[i][5])
    array.insert(792,'PredictedLabel6',score_final_v[i][6])
    array.insert(793,'PredictedLabel7',score_final_v[i][7])
    array.insert(794,'PredictedLabel8',score_final_v[i][8])
    array.insert(795,'PredictedLabel9',score_final_v[i][9])
    validationset=validationset.append(array)

validationset.to_csv('validation.csv', index = False, header = True)

print("validation.csv completed")

#test set csv printing
testset=pd.DataFrame()

for i in range(0, test):
    #print(Xtest_reshaped[i].shape)
    ma = np.matrix(Xtest_reshaped[i])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    # print(res2)
    # print(ytest[0])
    array=pd.DataFrame(ar)
    array.insert(784,'label',ytest[i])
    array.insert(785,'SUT',classification[i])
    array.insert(786,'PredictedLabel0',score_final[i][0])
    array.insert(787,'PredictedLabel1',score_final[i][1])
    array.insert(788,'PredictedLabel2',score_final[i][2])
    array.insert(789,'PredictedLabel3',score_final[i][3])
    array.insert(790,'PredictedLabel4',score_final[i][4])
    array.insert(791,'PredictedLabel5',score_final[i][5])
    array.insert(792,'PredictedLabel6',score_final[i][6])
    array.insert(793,'PredictedLabel7',score_final[i][7])
    array.insert(794,'PredictedLabel8',score_final[i][8])
    array.insert(795,'PredictedLabel9',score_final[i][9])
    
    if(ytest[i] < 5):
        array.insert(796,'EP', 0)
    else:
        array.insert(796,'EP', 1)

    if(ytest[i] in (0, 3, 6, 8, 9)):
        array.insert(797,'FP', 0)
    elif(ytest[i] in (1, 4, 7)):
        array.insert(797,'FP', 1)
    else:
        array.insert(797,'FP', 2)
    
    testset=testset.append(array)

testset.to_csv('test.csv', index = False, header = True)
print("test.csv completed")
