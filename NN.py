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


    # data = cursor.execute('''

    #     SELECT redName, blueName, winner, 
    #         red.mu, red.sigma, 
    #         blue.mu, blue.sigma, 
    #         (CAST(red.wins AS float)/(red.wins+red.losses)) as redWinrate, 
    #         (CAST(blue.wins AS float)/(blue.wins+blue.losses)) as blueWinrate, 
    #         CAST((strftime('%s',red.avgWinTime)-strftime('%s','00:00:00'))-(strftime('%s',red.avgLossTime)-strftime('%s','00:00:00')) as float) as redTime, 
    #         CAST((strftime('%s',blue.avgWinTime)-strftime('%s','00:00:00'))-(strftime('%s',blue.avgLossTime)-strftime('%s','00:00:00'))as float) as blueTime
    #         from fights 
    #         INNER JOIN characters AS red ON fights.redName = red.name and fights.mode = red.mode and fights.mode = "Matchmaking" 
    #         INNER JOIN characters AS blue ON fights.blueName = blue.name and fights.mode = blue.mode and fights.mode = "Matchmaking" 
    #         WHERE fights.mode = "Matchmaking" and red.wins+red.losses >= 10 and blue.wins+blue.losses >= 10 and redName != blueName;

    # ''').fetchall()

    data = cursor.execute('''

        SELECT redName, blueName, winner, 
            red.mu, red.sigma, 
            blue.mu, blue.sigma 
            from fights 
            INNER JOIN characters AS red ON fights.redName = red.name and fights.mode = red.mode and fights.mode = "Matchmaking" 
            INNER JOIN characters AS blue ON fights.blueName = blue.name and fights.mode = blue.mode and fights.mode = "Matchmaking" 
            WHERE fights.mode = "Matchmaking" and red.wins+red.losses >= 10 and blue.wins+blue.losses >= 10 and redName != blueName;

    ''').fetchall()

    # data = [list(x) for x in data]

    # for count,value in enumerate(data):
    #     redName, blueName, *_ = value

    #     redStreak = cursor.execute('''

    #         SELECT matchid, redMu+redMuChange as temp from fights where redName = ? and mode = "Matchmaking"
    #         UNION ALL
    #         SELECT matchId, blueMu+blueMuChange as temp from fights where blueName = ? and mode = "Matchmaking"
    #         ORDER BY matchId DESC LIMIT 10;

    #     ''', (redName,redName)).fetchall()

    #     redStreak = [x[1] for x in redStreak[::-1]]

    #     blueStreak = cursor.execute('''

    #         SELECT matchid, redMu+redMuChange as temp from fights where redName = ? and mode = "Matchmaking"
    #         UNION ALL
    #         SELECT matchId, blueMu+blueMuChange as temp from fights where blueName = ? and mode = "Matchmaking"
    #         ORDER BY matchId DESC LIMIT 10;

    #     ''', (blueName,blueName)).fetchall()

    #     blueStreak = [x[1] for x in blueStreak[::-1]]

    #     value.extend(redStreak)
    #     value.extend(blueStreak)

    #     if count % 10000 == 0:
    #         print(count/len(data))

    conn.close()

    samples = 1000
    mid = random.randint(1000,len(data)-1000)

    data = data[mid-samples:mid+samples]

    nameDict = buildDict(data)
    nameCount = len(nameDict)

    # sys.exit()
    
    model = Sequential()
    # model.add(Dense(nameCount+len(data[0])-3,input_dim=nameCount+len(data[0])-3))
    model.add(Dense(nameCount+len(data[0])-3,input_dim=nameCount+len(data[0])-3))
    # model.add(Dense(len(data[0])-3,input_dim=len(data[0])-3))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    model.add(Dense(2048))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    model.add(Dense(nameCount))
    # model.add(Dense(2))
    model.add(Activation('softmax'))

    # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

    '''
    Works with 1000 samples - 

    Dense(2048)
    Dense(512)
    Dense(128)

    RMSprop(lr=0.0001)
)
    Adagrad(lr=0.001)
    Adagrad(lr=0.0001)

    Adadelta(lr=1.0)
    Adadelta(lr=0.1)

    Adam(lr=0.001)
    Adam(lr=0.0001)

    '''

    model.compile(loss='categorical_crossentropy',
        optimizer = Adam(lr=0.0001),
        metrics=['acc'])

    # random.shuffle(data)

    model.summary()

    trainData = data[:int(0.9*len(data))]
    testDataTrain = data[int(0.9*0.9*len(data)):int(0.9*len(data))]
    testDataEval = data[int(0.9*len(data)):]

    batch_size = 8

    testData, testLabels = buildMatrix(testDataEval, nameDict)
    testData = testData.todense()
    testLabels = testLabels.todense()

    model.fit_generator(buildMatrixGen(trainData, nameDict, batch_size), 
        epochs=100, 
        steps_per_epoch=len(trainData)/batch_size, 
        validation_data=(testData,testLabels),
        shuffle=True)
        # callbacks=[ModelCheckpoint('data/checkpoint.k', monitor='acc', save_best_only=True, mode='max')])

    testData, testLabels = buildMatrix(testDataTrain, nameDict)
    testData = testData.todense()
    testLabels = testLabels.todense()

    score = model.evaluate(testData, testLabels, batch_size=batch_size)
    print(score)

    testData, testLabels = buildMatrix(testDataEval, nameDict)
    testData = testData.todense()
    testLabels = testLabels.todense()

    score = model.evaluate(testData, testLabels, batch_size=batch_size)
    print(score)