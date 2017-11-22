import pickle
from saltybet.process import process
import sqlite3
import numpy

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm

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

def buildMatrix(data, nameDict):
    dataMat = sparse.lil_matrix((len(data),len(nameDict)+len(data[0])-3))
    labelMat = sparse.lil_matrix((len(data),len(nameDict)))
    rowlen = len(nameDict)

    for row, value in enumerate(data):
        name1, name2, winner, *rest = value
        dataMat[row,nameDict[name1]] = -1
        dataMat[row,nameDict[name2]] = 1
        dataMat[row,rowlen:] = rest
        if winner == name1:
            labelMat[row,nameDict[name1]] = 1
        elif winner == name2:
            labelMat[row,nameDict[name2]] = 1
    
    return dataMat, labelMat

def buildMatrixGen(data, nameDict, batch_size = 1):
    rowLen = len(nameDict)
    dataOut = []
    labelOut = []

    while True:
        count = 0
        for value in data:
            name1, name2, winner, *rest = value
            dataRow = numpy.zeros(len(nameDict)+len(data[0])-3)
            dataRow[nameDict[name1]] = -1
            dataRow[nameDict[name2]] = 1
            dataRow[rowLen:] = rest
            label = numpy.zeros(len(nameDict))
            if winner == name1:
                label[nameDict[name1]] = 1
            elif winner == name2:
                label[nameDict[name2]] = 1
            dataOut.append(dataRow)
            labelOut.append(label)
            count += 1
            if count % batch_size == 0:
                yield (numpy.array(dataOut),numpy.array(labelOut))
                dataOut = []
                labelOut = []

def genModel(input_dim, optimizer):
    model = Sequential()

    model.add(Dense(input_dim,input_dim=input_dim))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    # model.add(Dense(2048))
    # model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    # model.add(Dense(512))
    # model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    # model.add(Dense(128))
    # model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    model.add(Dense(nameCount))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer = optimizer,
        metrics=['acc'])

    model.summary()

    return model
            
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
            red.mu, red.sigma, 
            blue.mu, blue.sigma 
            from fights 
            INNER JOIN characters AS red ON fights.redName = red.name and fights.mode = red.mode and fights.mode = "Matchmaking" 
            INNER JOIN characters AS blue ON fights.blueName = blue.name and fights.mode = blue.mode and fights.mode = "Matchmaking" 
            WHERE fights.mode = "Matchmaking" and red.wins+red.losses >= 10 and blue.wins+blue.losses >= 10 and redName != blueName;

    ''').fetchall()

    conn.close()

    optDict = {

        'RMSprop':[('0.0001', RMSprop(lr=0.0001)), ('0.00001', RMSprop(lr=0.00001))], 
        'Adagrad':[('0.001', Adagrad(lr=0.001)), ('0.0001', Adagrad(lr=0.0001))], 
        'Adadelta':[('0.1', Adadelta(lr=0.1)), ('0.01', Adadelta(lr=0.01))], 
        'Adam':[('0.0001', Adam(lr=0.0001)), ('0.00001', Adam(lr=0.00001))], 
        'Nadam':[('0.0002', Nadam(lr=0.0002)), ('0.00002', Nadam(lr=0.00002))]

    }

    resDict = {}

    for key, value in optDict.items():
        resDict[key] = {}
        
        for optimizers in optDict[key]:
            resDict[key][optimizers[0]] = {}
            resDict[key][optimizers[0]]['train'] = []
            resDict[key][optimizers[0]]['evaluate'] = []

    samples = 1000
    batch_size = 32

    for i in range(5):        
        mid = random.randint(int(samples/2),len(data)-int(samples/2))

        dataTemp = data[mid-int(samples/2):mid+int(samples/2)]

        nameDict = buildDict(dataTemp)
        nameCount = len(nameDict)

        trainData = dataTemp[:int(0.9*len(dataTemp))]

        trainEvalData, trainEvalLabels = buildMatrix(dataTemp[int(0.9*0.9*len(dataTemp)):int(0.9*len(dataTemp))], nameDict)
        trainEvalData = trainEvalData.todense()
        trainEvalLabels = trainEvalLabels.todense()

        evalData, evalLabels = buildMatrix(dataTemp[int(0.9*len(dataTemp)):], nameDict)
        evalData = evalData.todense()
        evalLabels = evalLabels.todense()

        for name, value in optDict.items():
            for lr, optimizer in value:

                model = genModel(nameCount+len(data[0])-3, optimizer)

                model.fit_generator(buildMatrixGen(trainData, nameDict, batch_size), 
                    epochs=100, 
                    steps_per_epoch=len(trainData)/batch_size, 
                    validation_data=(evalData, evalLabels),
                    shuffle=True)

                scoreTrain = model.evaluate(trainEvalData, trainEvalLabels, batch_size=batch_size)
                print(scoreTrain)
                scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
                print(scoreEval)
                resDict[name][lr]['train'].append(scoreTrain[1])
                resDict[name][lr]['evaluate'].append(scoreEval[1])

                with open('results.txt', 'w') as fp:
                    json.dump(resDict, fp, indent=4)