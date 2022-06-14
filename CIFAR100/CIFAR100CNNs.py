from keras.datasets import cifar100
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D, AveragePooling2D
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K


train = 40000 
test = 15000
val = 5000

batch_size = 32
num_classes = 100
epochs = 15
num_predictions = 20

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

x_train, Xtemp, y_train, ytemp = train_test_split(x, y, train_size=train, random_state=0)

x_val, x_test, y_val, y_test = train_test_split(Xtemp, ytemp, train_size=val, random_state=0)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255

x_test = x_test.astype('float32')
x_test /= 255

model = load_model("models/modelL.h5")

scores = model.evaluate(x_test, to_categorical(y_test), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# print(X_prop_test.shape)

classification_v = model.predict_classes(x_val)
# print(classification_v)

score_final_v = model.predict(x_val)

classification = model.predict_classes(x_test)
# print(classification)

score_final = model.predict(x_test)

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
    ma = np.matrix(x_val[i,:,:,0])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array0=pd.DataFrame(ar)

    ma = np.matrix(x_val[i,:,:,1])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array1=pd.DataFrame(ar)

    ma = np.matrix(x_val[i,:,:,2])
    ar = ma.flatten()
    res = str(ar)[1:-1]
    res2 = str(res)[1:-1]
    array2=pd.DataFrame(ar)

    array = pd.concat([array0, array1, array2], axis=1)

    array.insert(3072,'label',y_val[i])
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
    array.insert(3084,'PredictedLabel10',score_final_v[i][10])
    array.insert(3085,'PredictedLabel11',score_final_v[i][11])
    array.insert(3086,'PredictedLabel12',score_final_v[i][12])
    array.insert(3087,'PredictedLabel13',score_final_v[i][13])
    array.insert(3088,'PredictedLabel14',score_final_v[i][14])
    array.insert(3089,'PredictedLabel15',score_final_v[i][15])
    array.insert(3090,'PredictedLabel16',score_final_v[i][16])
    array.insert(3091,'PredictedLabel17',score_final_v[i][17])
    array.insert(3092,'PredictedLabel18',score_final_v[i][18])
    array.insert(3093,'PredictedLabel19',score_final_v[i][19])
    array.insert(3094,'PredictedLabel20',score_final_v[i][20])
    array.insert(3095,'PredictedLabel21',score_final_v[i][21])
    array.insert(3096,'PredictedLabel22',score_final_v[i][22])
    array.insert(3097,'PredictedLabel23',score_final_v[i][23])
    array.insert(3098,'PredictedLabel24',score_final_v[i][24])
    array.insert(3099,'PredictedLabel25',score_final_v[i][25])
    array.insert(3100,'PredictedLabel26',score_final_v[i][26])
    array.insert(3101,'PredictedLabel27',score_final_v[i][27])
    array.insert(3102,'PredictedLabel28',score_final_v[i][28])
    array.insert(3103,'PredictedLabel29',score_final_v[i][29])
    array.insert(3104,'PredictedLabel30',score_final_v[i][30])
    array.insert(3105,'PredictedLabel31',score_final_v[i][31])
    array.insert(3106,'PredictedLabel32',score_final_v[i][32])
    array.insert(3107,'PredictedLabel33',score_final_v[i][33])
    array.insert(3108,'PredictedLabel34',score_final_v[i][34])
    array.insert(3109,'PredictedLabel35',score_final_v[i][35])
    array.insert(3110,'PredictedLabel36',score_final_v[i][36])
    array.insert(3111,'PredictedLabel37',score_final_v[i][37])
    array.insert(3112,'PredictedLabel38',score_final_v[i][38])
    array.insert(3113,'PredictedLabel39',score_final_v[i][39])
    array.insert(3114,'PredictedLabel40',score_final_v[i][40])
    array.insert(3115,'PredictedLabel41',score_final_v[i][41])
    array.insert(3116,'PredictedLabel42',score_final_v[i][42])
    array.insert(3117,'PredictedLabel43',score_final_v[i][43])
    array.insert(3118,'PredictedLabel44',score_final_v[i][44])
    array.insert(3119,'PredictedLabel45',score_final_v[i][45])
    array.insert(3120,'PredictedLabel46',score_final_v[i][46])
    array.insert(3121,'PredictedLabel47',score_final_v[i][47])
    array.insert(3122,'PredictedLabel48',score_final_v[i][48])
    array.insert(3123,'PredictedLabel49',score_final_v[i][49])
    array.insert(3124,'PredictedLabel50',score_final_v[i][50])
    array.insert(3125,'PredictedLabel51',score_final_v[i][51])
    array.insert(3126,'PredictedLabel52',score_final_v[i][52])
    array.insert(3127,'PredictedLabel53',score_final_v[i][53])
    array.insert(3128,'PredictedLabel54',score_final_v[i][54])
    array.insert(3129,'PredictedLabel55',score_final_v[i][55])
    array.insert(3130,'PredictedLabel56',score_final_v[i][56])
    array.insert(3131,'PredictedLabel57',score_final_v[i][57])
    array.insert(3132,'PredictedLabel58',score_final_v[i][58])
    array.insert(3133,'PredictedLabel59',score_final_v[i][59])
    array.insert(3134,'PredictedLabel60',score_final_v[i][60])
    array.insert(3135,'PredictedLabel61',score_final_v[i][61])
    array.insert(3136,'PredictedLabel62',score_final_v[i][62])
    array.insert(3137,'PredictedLabel63',score_final_v[i][63])
    array.insert(3138,'PredictedLabel64',score_final_v[i][64])
    array.insert(3139,'PredictedLabel65',score_final_v[i][65])
    array.insert(3140,'PredictedLabel66',score_final_v[i][66])
    array.insert(3141,'PredictedLabel67',score_final_v[i][67])
    array.insert(3142,'PredictedLabel68',score_final_v[i][68])
    array.insert(3143,'PredictedLabel69',score_final_v[i][69])
    array.insert(3144,'PredictedLabel70',score_final_v[i][70])
    array.insert(3145,'PredictedLabel71',score_final_v[i][71])
    array.insert(3146,'PredictedLabel72',score_final_v[i][72])
    array.insert(3147,'PredictedLabel73',score_final_v[i][73])
    array.insert(3148,'PredictedLabel74',score_final_v[i][74])
    array.insert(3149,'PredictedLabel75',score_final_v[i][75])
    array.insert(3150,'PredictedLabel76',score_final_v[i][76])
    array.insert(3151,'PredictedLabel77',score_final_v[i][77])
    array.insert(3152,'PredictedLabel78',score_final_v[i][78])
    array.insert(3153,'PredictedLabel79',score_final_v[i][79])
    array.insert(3154,'PredictedLabel80',score_final_v[i][80])
    array.insert(3155,'PredictedLabel81',score_final_v[i][81])
    array.insert(3156,'PredictedLabel82',score_final_v[i][82])
    array.insert(3157,'PredictedLabel83',score_final_v[i][83])
    array.insert(3158,'PredictedLabel84',score_final_v[i][84])
    array.insert(3159,'PredictedLabel85',score_final_v[i][85])
    array.insert(3160,'PredictedLabel86',score_final_v[i][86])
    array.insert(3161,'PredictedLabel87',score_final_v[i][87])
    array.insert(3162,'PredictedLabel88',score_final_v[i][88])
    array.insert(3163,'PredictedLabel89',score_final_v[i][89])
    array.insert(3164,'PredictedLabel90',score_final_v[i][90])
    array.insert(3165,'PredictedLabel91',score_final_v[i][91])
    array.insert(3166,'PredictedLabel92',score_final_v[i][92])
    array.insert(3167,'PredictedLabel93',score_final_v[i][93])
    array.insert(3168,'PredictedLabel94',score_final_v[i][94])
    array.insert(3169,'PredictedLabel95',score_final_v[i][95])
    array.insert(3170,'PredictedLabel96',score_final_v[i][96])
    array.insert(3171,'PredictedLabel97',score_final_v[i][97])
    array.insert(3172,'PredictedLabel98',score_final_v[i][98])
    array.insert(3173,'PredictedLabel99',score_final_v[i][99])

    validationset=validationset.append(array)

validationset.to_csv('validation.csv', index = False, header = True)


#test set printing
testset=pd.DataFrame()

for i in range(0, test):
    #print(Xtest_reshaped[i].shape)
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
    array.insert(3084,'PredictedLabel10',score_final[i][10])
    array.insert(3085,'PredictedLabel11',score_final[i][11])
    array.insert(3086,'PredictedLabel12',score_final[i][12])
    array.insert(3087,'PredictedLabel13',score_final[i][13])
    array.insert(3088,'PredictedLabel14',score_final[i][14])
    array.insert(3089,'PredictedLabel15',score_final[i][15])
    array.insert(3090,'PredictedLabel16',score_final[i][16])
    array.insert(3091,'PredictedLabel17',score_final[i][17])
    array.insert(3092,'PredictedLabel18',score_final[i][18])
    array.insert(3093,'PredictedLabel19',score_final[i][19])
    array.insert(3094,'PredictedLabel20',score_final[i][20])
    array.insert(3095,'PredictedLabel21',score_final[i][21])
    array.insert(3096,'PredictedLabel22',score_final[i][22])
    array.insert(3097,'PredictedLabel23',score_final[i][23])
    array.insert(3098,'PredictedLabel24',score_final[i][24])
    array.insert(3099,'PredictedLabel25',score_final[i][25])
    array.insert(3100,'PredictedLabel26',score_final[i][26])
    array.insert(3101,'PredictedLabel27',score_final[i][27])
    array.insert(3102,'PredictedLabel28',score_final[i][28])
    array.insert(3103,'PredictedLabel29',score_final[i][29])
    array.insert(3104,'PredictedLabel30',score_final[i][30])
    array.insert(3105,'PredictedLabel31',score_final[i][31])
    array.insert(3106,'PredictedLabel32',score_final[i][32])
    array.insert(3107,'PredictedLabel33',score_final[i][33])
    array.insert(3108,'PredictedLabel34',score_final[i][34])
    array.insert(3109,'PredictedLabel35',score_final[i][35])
    array.insert(3110,'PredictedLabel36',score_final[i][36])
    array.insert(3111,'PredictedLabel37',score_final[i][37])
    array.insert(3112,'PredictedLabel38',score_final[i][38])
    array.insert(3113,'PredictedLabel39',score_final[i][39])
    array.insert(3114,'PredictedLabel40',score_final[i][40])
    array.insert(3115,'PredictedLabel41',score_final[i][41])
    array.insert(3116,'PredictedLabel42',score_final[i][42])
    array.insert(3117,'PredictedLabel43',score_final[i][43])
    array.insert(3118,'PredictedLabel44',score_final[i][44])
    array.insert(3119,'PredictedLabel45',score_final[i][45])
    array.insert(3120,'PredictedLabel46',score_final[i][46])
    array.insert(3121,'PredictedLabel47',score_final[i][47])
    array.insert(3122,'PredictedLabel48',score_final[i][48])
    array.insert(3123,'PredictedLabel49',score_final[i][49])
    array.insert(3124,'PredictedLabel50',score_final[i][50])
    array.insert(3125,'PredictedLabel51',score_final[i][51])
    array.insert(3126,'PredictedLabel52',score_final[i][52])
    array.insert(3127,'PredictedLabel53',score_final[i][53])
    array.insert(3128,'PredictedLabel54',score_final[i][54])
    array.insert(3129,'PredictedLabel55',score_final[i][55])
    array.insert(3130,'PredictedLabel56',score_final[i][56])
    array.insert(3131,'PredictedLabel57',score_final[i][57])
    array.insert(3132,'PredictedLabel58',score_final[i][58])
    array.insert(3133,'PredictedLabel59',score_final[i][59])
    array.insert(3134,'PredictedLabel60',score_final[i][60])
    array.insert(3135,'PredictedLabel61',score_final[i][61])
    array.insert(3136,'PredictedLabel62',score_final[i][62])
    array.insert(3137,'PredictedLabel63',score_final[i][63])
    array.insert(3138,'PredictedLabel64',score_final[i][64])
    array.insert(3139,'PredictedLabel65',score_final[i][65])
    array.insert(3140,'PredictedLabel66',score_final[i][66])
    array.insert(3141,'PredictedLabel67',score_final[i][67])
    array.insert(3142,'PredictedLabel68',score_final[i][68])
    array.insert(3143,'PredictedLabel69',score_final[i][69])
    array.insert(3144,'PredictedLabel70',score_final[i][70])
    array.insert(3145,'PredictedLabel71',score_final[i][71])
    array.insert(3146,'PredictedLabel72',score_final[i][72])
    array.insert(3147,'PredictedLabel73',score_final[i][73])
    array.insert(3148,'PredictedLabel74',score_final[i][74])
    array.insert(3149,'PredictedLabel75',score_final[i][75])
    array.insert(3150,'PredictedLabel76',score_final[i][76])
    array.insert(3151,'PredictedLabel77',score_final[i][77])
    array.insert(3152,'PredictedLabel78',score_final[i][78])
    array.insert(3153,'PredictedLabel79',score_final[i][79])
    array.insert(3154,'PredictedLabel80',score_final[i][80])
    array.insert(3155,'PredictedLabel81',score_final[i][81])
    array.insert(3156,'PredictedLabel82',score_final[i][82])
    array.insert(3157,'PredictedLabel83',score_final[i][83])
    array.insert(3158,'PredictedLabel84',score_final[i][84])
    array.insert(3159,'PredictedLabel85',score_final[i][85])
    array.insert(3160,'PredictedLabel86',score_final[i][86])
    array.insert(3161,'PredictedLabel87',score_final[i][87])
    array.insert(3162,'PredictedLabel88',score_final[i][88])
    array.insert(3163,'PredictedLabel89',score_final[i][89])
    array.insert(3164,'PredictedLabel90',score_final[i][90])
    array.insert(3165,'PredictedLabel91',score_final[i][91])
    array.insert(3166,'PredictedLabel92',score_final[i][92])
    array.insert(3167,'PredictedLabel93',score_final[i][93])
    array.insert(3168,'PredictedLabel94',score_final[i][94])
    array.insert(3169,'PredictedLabel95',score_final[i][95])
    array.insert(3170,'PredictedLabel96',score_final[i][96])
    array.insert(3171,'PredictedLabel97',score_final[i][97])
    array.insert(3172,'PredictedLabel98',score_final[i][98])
    array.insert(3173,'PredictedLabel99',score_final[i][99])
    testset=testset.append(array)

testset.to_csv('test.csv', index = False, header = True)