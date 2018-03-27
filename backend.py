import pickle
import sqlite3
import random

import numpy
from scipy import sparse
import pandas

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

        redStats = rest[:int(len(rest)/2)]
        blueStats = rest[int(len(rest)/2):]

        if(nameDict[name1] > nameDict[name2]):
            rest = redStats+blueStats
        elif(nameDict[name2] > nameDict[name1]):
            rest = blueStats+redStats

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

    cutoff = 0.0

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

        index = list(range(data[0].shape[0]))
        random.shuffle(index)
        data = [data[0][index], data[1][index]]
        labels = [labels[0][index]]

        for i in range(data[0].shape[0]):
            if i != 0 and i % batch_size == 0:
                yield [[data[0][i-batch_size:i].todense(), data[1][i-batch_size:i]], [labels[0][i-batch_size:i].todense()]]

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
            CAST(red.wins AS FLOAT)/(red.wins+red.losses) AS winrate, strftime('%s', red.avgMatchTime)-strftime('%s', '00:00:00') AS avgTimeRed, red.mu, red.sigma, 
            CAST(blue.wins AS FLOAT)/(blue.wins+blue.losses) AS winrate, strftime('%s', blue.avgMatchTime)-strftime('%s', '00:00:00') AS avgTimeBlue, blue.mu, blue.sigma 
            from fights 
            INNER JOIN characters AS red ON fights.redName = red.name and fights.mode = red.mode and (fights.mode = "Matchmaking" or fights.mode = "Tournament")
            INNER JOIN characters AS blue ON fights.blueName = blue.name and fights.mode = blue.mode and (fights.mode = "Matchmaking" or fights.mode = "Tournament")
            WHERE (fights.mode = "Matchmaking" or fights.mode = "Tournament") and red.wins+red.losses >= 10 and blue.wins+blue.losses >= 10 and redName != blueName;

    ''').fetchall()

    conn.close()

    df = pandas.DataFrame(data)

    for value in [4,5,6,8,9,10]:
        df[value] = (df[value]-df[value].min()) / (df[value].max()-df[value].min())

    return df.values.tolist()

    # return data

def makeData(data, nameDict, samples=10000):
    # mid = random.randint(int(samples/2),len(data)-int(samples/2))

    # dataTemp = data[mid-int(samples/2):mid+int(samples/2)]

    dataTemp = random.sample(data, samples)

    trainData, trainLabels = buildMatrix(dataTemp[:int(0.9*len(dataTemp))], nameDict)

    trainEvalData, trainEvalLabels = buildMatrix(dataTemp[int(0.9*0.9*len(dataTemp)):int(0.9*len(dataTemp))], nameDict, isSparse=0)

    evalData, evalLabels = buildMatrix(dataTemp[int(0.9*len(dataTemp)):], nameDict, isSparse=0)

    return trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels