import pickle
from saltybet.process import process
import sqlite3
import numpy

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.regularizers import l1, l2, l1_l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import plot_model

from sklearn.preprocessing import OneHotEncoder

from scipy import sparse

import random
import sys
import json

def buildDict(data):
    data = [x[:2] for x in data]
    names = [x[0] for x in data]
    names.extend([x[1] for x in data])
    names = list(set(names))
    names.sort()
    names = list(enumerate(names))
    names = [(x[1],x[0]) for x in names]
    namedict = {x[0]:x[1] for x in names}
    return namedict

def buildMatrix(data, nameDict, isSparse=1):

    if isSparse == 1:
        nameMat = sparse.lil_matrix((len(data),len(nameDict)))
        labelMat = sparse.lil_matrix((len(data),len(nameDict)))
    else:
        nameMat = numpy.zeros((len(data),len(nameDict)))
        labelMat = numpy.zeros((len(data),len(nameDict)))

    dataMat = numpy.zeros((len(data),len(data[0])-3))

    for row, value in enumerate(data):
        name1, name2, winner, *rest = value
        nameMat[row,nameDict[name1]] = 1
        nameMat[row,nameDict[name2]] = 1
        dataMat[row] = rest
        if winner == name1:
            labelMat[row,nameDict[name1]] = 1
        elif winner == name2:
            labelMat[row,nameDict[name2]] = 1
    
    return [[nameMat, dataMat], [labelMat]]

def dataGen(data, labels, batch_size=1):
    while True:
        for i in range(data[0].shape[0]):
            if i != 0 and i % batch_size == 0:
                yield [[data[0][i-batch_size:i].todense(), data[1][i-batch_size:i]], [labels[0][i-batch_size:i].todense()]]

def genModel(nameCount, dataCount, optimizer):

    units = int(nameCount/8)
    # units = nameCount

    nameInput = Input(shape=(nameCount,), name='nameInput')
    x = Dense(units, kernel_regularizer=l2(l=0.001))(nameInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    x = Dense(nameCount)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    nameOutput = Dropout(0.5, name='nameOutput')(x)

    dataInput = Input(shape=(dataCount,), name='dataInput')
    x = Dense(dataCount, kernel_regularizer=l2(l=0.001))(dataInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # x = Dense(dataCount, kernel_regularizer=l2(l=0.001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    x = Dense(dataCount)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    dataOutput = Dropout(0.5, name='dataOutput')(x)

    x = keras.layers.concatenate([nameOutput, dataOutput])

    x = Dense(units, kernel_regularizer=l2(l=0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    x = Dense(nameCount)(x)
    mainOutput = Activation('softmax', name='mainOutput')(x)

    model = Model(inputs=[nameInput, dataInput], outputs=[mainOutput])

    model.compile(loss='categorical_crossentropy',
        optimizer = optimizer,
        metrics=['acc'])

    plot_model(model, to_file='model.png')

    model.summary()

    return model

def optimizer(name, val):
    if name == "SGD":
        return SGD(lr=val)
    if name == "RMSprop":
        return RMSprop(lr=val)
    if name == "Adagrad":
        return Adagrad(lr=val)
    if name == "Adadelta":
        return Adadelta(lr=val)
    if name == "Adam":
        return Adam(lr=val)
    if name == "Adamax":
        return Adamax(lr=val)
    if name == "Nadam":
        return Nadam(lr=val)
            
if __name__ == "__main__":

    try:
        with open('data/data.p', 'rb') as fp:
            datadict = pickle.load(fp)
    except:
        print("FAILED TO OPEN DATADICT")
        datadict = {}

    conn = sqlite3.connect('data/data.db')
    cursor = conn.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('BEGIN TRANSACTION')

    data = cursor.execute('''

        SELECT redName, blueName, winner, 
            CAST(red.wins AS FLOAT)/(red.wins+red.losses) AS winrate, red.mu, red.sigma, 
            CAST(blue.wins AS FLOAT)/(blue.wins+blue.losses) AS winrate, blue.mu, blue.sigma 
            from fights 
            INNER JOIN characters AS red ON fights.redName = red.name and fights.mode = red.mode and (fights.mode = "Matchmaking" or fights.mode = "Tournament")
            INNER JOIN characters AS blue ON fights.blueName = blue.name and fights.mode = blue.mode and (fights.mode = "Matchmaking" or fights.mode = "Tournament")
            WHERE (fights.mode = "Matchmaking" or fights.mode = "Tournament") and red.wins+red.losses >= 10 and blue.wins+blue.losses >= 10 and redName != blueName;

    ''').fetchall()

    conn.close()

    # optDict = { 
    #     'Adagrad':[0.01, 0.001, 0.0001], 
    #     'Adam':[0.001, 0.0001, 0.00001], 
    #     'Adamax':[0.002, 0.0002, 0.00002], 
    #     'Nadam':[0.002, 0.0002, 0.00002]
    # }

    optDict = { 
        'Adagrad':[0.01, 0.001], 
        'Adam':[0.0001], 
        'Adamax':[0.002, 0.0002], 
        'Nadam':[0.0002]
    }

    resDict = {}

    for key, value in optDict.items():
        resDict[key] = {}
        
        for optimizers in optDict[key]:
            resDict[key][optimizers] = {}
            resDict[key][optimizers]['train'] = []
            resDict[key][optimizers]['evaluate'] = []

    # samples = len(data)
    samples = 10000
    batch_size = 1024

    # nameDict = buildDict(data)

    for i in range(5):        
        mid = random.randint(int(samples/2),len(data)-int(samples/2))

        dataTemp = data[mid-int(samples/2):mid+int(samples/2)]

        nameDict = buildDict(dataTemp)

        trainData, trainLabels = buildMatrix(dataTemp[:int(0.9*len(dataTemp))], nameDict)

        trainEvalData, trainEvalLabels = buildMatrix(dataTemp[int(0.9*0.9*len(dataTemp)):int(0.9*len(dataTemp))], nameDict, isSparse=0)

        evalData, evalLabels = buildMatrix(dataTemp[int(0.9*len(dataTemp)):], nameDict, isSparse=0)

        for name, value in optDict.items():

            for lr in value:

                model = genModel(trainData[0].shape[1], trainData[1].shape[1], optimizer(name, lr))

                earlystopping = EarlyStopping(monitor='loss',min_delta=0.001, patience=10)
                tensorboard = TensorBoard(log_dir='./logs/{}/{}/{}'.format(name, str(lr).replace('.', '_'), i))

                model.fit_generator(dataGen(trainData, trainLabels, batch_size), 
                    epochs=250, 
                    steps_per_epoch=trainData[0].shape[0]/batch_size, 
                    validation_data=(evalData, evalLabels), 
                    callbacks=[earlystopping, tensorboard])

                scoreTrain = model.evaluate(trainEvalData, trainEvalLabels, batch_size=batch_size)
                print(scoreTrain)
                scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
                print(scoreEval)
                resDict[name][lr]['train'].append(scoreTrain[1])
                resDict[name][lr]['evaluate'].append(scoreEval[1])

                K.clear_session()

                with open('results.txt', 'w') as fp:
                    json.dump(resDict, fp, indent=4)