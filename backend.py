import pickle
import sqlite3
import random

import numpy
from scipy import sparse
import pandas

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

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

def predict(model, data, labels, cutoff = 0.5):
    res = model.predict(data)

    data = data[0]
    labels = labels[0]

    count = 0
    total = 0

    cutoff = 0.5

    for i in range(data.shape[0]):
        dataVals = numpy.where(data[i])[0]
        labelVals = numpy.where(labels[i])[0]

        if res[i][dataVals[0]] - res[i][dataVals[1]] > cutoff and labels[i][dataVals[0]] == 1:
            count += 1

        elif res[i][dataVals[1]] - res[i][dataVals[0]] > cutoff and labels[i][dataVals[1]] == 1:
            count += 1

        if res[i][dataVals[0]] - res[i][dataVals[1]] > cutoff or res[i][dataVals[1]] - res[i][dataVals[0]] > cutoff:
            total += 1

        # total += 1

    print(total)
    print(data.shape[0])
    print(count/total)

def dataGenerator(data, labels, batch_size=1):
    while True:
        for i in range(data[0].shape[0]):
            if i != 0 and i % batch_size == 0:
                yield [[data[0][i-batch_size:i].todense(), data[1][i-batch_size:i]], [labels[0][i-batch_size:i].todense()]]

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

def genOptDict():
    optDict = { 
        'Adam':[0.0001]
    }

    return optDict

def loadData(dbPath='data/data.db'):

    conn = sqlite3.connect(dbPath)
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

    df = pandas.DataFrame(data)

    for value in [4,5,7,8]:
        df[value] = (df[value]-df[value].min()) / (df[value].max()-df[value].min())

    # return data
    return df.values.tolist()

def makeData(data, nameDict, samples=10000):
    # mid = random.randint(int(samples/2),len(data)-int(samples/2))

    # dataTemp = data[mid-int(samples/2):mid+int(samples/2)]

    dataTemp = random.sample(data, samples)

    trainData, trainLabels = buildMatrix(dataTemp[:int(0.9*len(dataTemp))], nameDict)

    trainEvalData, trainEvalLabels = buildMatrix(dataTemp[int(0.9*0.9*len(dataTemp)):int(0.9*len(dataTemp))], nameDict, isSparse=0)

    evalData, evalLabels = buildMatrix(dataTemp[int(0.9*len(dataTemp)):], nameDict, isSparse=0)

    return trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels

def genModel(nameCount, dataCount, optimizer):

    units = int(nameCount/8)
    # units = nameCount

    nameInput = Input(shape=(nameCount,), name='nameInput')
    x = Dense(units, kernel_regularizer=l2(l=0.0001))(nameInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    nameOutput = Dense(nameCount, name='nameOutput')(x)

    dataInput = Input(shape=(dataCount,), name='dataInput')
    x = Dense(dataCount, kernel_regularizer=l2(l=0.0001))(dataInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(dataCount, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # x = Dense(dataCount, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # x = Dense(dataCount, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    dataOutput = Dense(dataCount, name='dataOutput')(x)

    x = keras.layers.concatenate([nameOutput, dataOutput])

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
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