import pickle
import sqlite3
import random

import numpy
from scipy import sparse
import pandas

import configparser

def readConfig(path='./config/config.cfg'):
    config = configparser.RawConfigParser()
    config.read(path)
    samples = int(config.get("Saltybet", "samples"))
    testsamples = int(config.get("Saltybet", "testsamples"))
    epochs = [int(x) for x in str.split(config.get("Saltybet", "epochs"), ', ')]
    optimizers = str.split(config.get("Saltybet", "optimizer"), ', ')
    lr = [float(x) for x in str.split(config.get("Saltybet", "lr"), ', ')]
    l2 = [float(x) for x in str.split(config.get("Saltybet", "l2"), ', ')]

    return samples, testsamples, epochs, optimizers, lr, l2

def getData(samples, testsamples, mode, path='./data/data.db'):
    data = loadData(path)

    if mode == "train":
        return makeDataTrain(data[:samples], buildDict(data[:samples+testsamples]))
    if mode == "eval":
        return makeDataEval(data[samples:samples+testsamples], buildDict(data[:samples+testsamples]))

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

        if nameDict[name1] < nameDict[name2]:
            rest = redStats+blueStats
        elif nameDict[name2] < nameDict[name1]:
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

    cutoff = 0.75

    for i in range(data.shape[0]):
        dataVals = numpy.where(data[i])[0]
        labelVals = numpy.where(labels[i])[0]

        if res[i][dataVals[0]] - res[i][dataVals[1]] > cutoff and labels[i][dataVals[0]] == 1:
            count += 1

        elif res[i][dataVals[1]] - res[i][dataVals[0]] > cutoff and labels[i][dataVals[1]] == 1:
            count += 1

        if res[i][dataVals[0]] - res[i][dataVals[1]] > cutoff or res[i][dataVals[1]] - res[i][dataVals[0]] > cutoff:
            total += 1

    print(total)
    print(data.shape[0])
    print(count/total)

def predictOne(model, name1, name2, nameDict, dbPath='data/data.db'):

    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('BEGIN TRANSACTION')

    data = cursor.execute('''

        SELECT 
        redWins, redLosses, redUpsetBets, redUpsetMu, redWinrate, redAvgMatchTime, redAvgWinTime, redAvgLossTime, redMu, redSigma,
		blueWins, blueLosses, blueUpsetBets, blueUpsetMu, blueWinrate, blueAvgMatchTime, blueAvgWinTime, blueAvgLossTime, blueMu, blueSigma 
        FROM 
        (SELECT 
            (SELECT CAST(red.wins-min(allData.wins) AS FLOAT)/(max(allData.wins)-min(allData.wins))) AS redWins, 
            (SELECT CAST(red.losses-min(allData.losses) AS FLOAT)/(max(allData.losses)-min(allData.losses))) AS redLosses, 
            red.upsetBets AS redUpsetBets, 
            red.upsetMu as redUpsetMu, 
            CAST(red.wins AS FLOAT)/(red.wins+red.losses) AS redWinrate, 
            (SELECT CAST(red.mu-min(allData.mu) AS FLOAT)/(max(allData.mu)-min(allData.mu))) AS redMu, 
            (SELECT CAST(red.sigma-min(allData.sigma) AS FLOAT)/(max(allData.sigma)-min(allData.sigma))) AS redSigma, 

            (SELECT CAST(blue.wins-min(allData.wins) AS FLOAT)/(max(allData.wins)-min(allData.wins))) AS blueWins, 
            (SELECT CAST(blue.losses-min(allData.losses) AS FLOAT)/(max(allData.losses)-min(allData.losses))) AS blueLosses, 
            blue.upsetBets AS blueUpsetBets, 
            blue.upsetMu AS blueUpsetMu, 
            CAST(blue.wins AS FLOAT)/(blue.wins+blue.losses) AS blueWinrate, 
            (SELECT CAST(blue.mu-min(allData.mu) AS FLOAT)/(max(allData.mu)-min(allData.mu))) AS blueMu, 
            (SELECT CAST(blue.sigma-min(allData.sigma) AS FLOAT)/(max(allData.sigma)-min(allData.sigma))) AS blueSigma 
            FROM 
            characters AS red 
            INNER JOIN
            characters AS blue 
            INNER JOIN
            characters AS allData 
            WHERE red.mode = "Matchmaking" AND red.name = ? AND blue.mode = "Matchmaking" AND blue.name = ?
        )
        
        INNER JOIN
        
        (SELECT
            (SELECT CAST((strftime('%s', red.avgMatchTime)-strftime('%s', '00:00:00'))-min(timeData.avgMatchTime) AS FLOAT)/(max(timeData.avgMatchTime)-min(timeData.avgMatchTime))) AS redAvgMatchTime, 
            (SELECT CAST((strftime('%s', blue.avgMatchTime)-strftime('%s', '00:00:00'))-min(timeData.avgMatchTime) AS FLOAT)/(max(timeData.avgMatchTime)-min(timeData.avgMatchTime))) AS blueAvgMatchTime, 
			(SELECT CAST((strftime('%s', red.avgWinTime)-strftime('%s', '00:00:00'))-min(timeData.avgWinTime) AS FLOAT)/(max(timeData.avgWinTime)-min(timeData.avgWinTime))) AS redAvgWinTime, 
            (SELECT CAST((strftime('%s', blue.avgWinTime)-strftime('%s', '00:00:00'))-min(timeData.avgWinTime) AS FLOAT)/(max(timeData.avgWinTime)-min(timeData.avgWinTime))) AS blueAvgWinTime, 
			(SELECT CAST((strftime('%s', red.avgLossTime)-strftime('%s', '00:00:00'))-min(timeData.avgLossTime) AS FLOAT)/(max(timeData.avgLossTime)-min(timeData.avgLossTime))) AS redAvgLossTime, 
            (SELECT CAST((strftime('%s', blue.avgLossTime)-strftime('%s', '00:00:00'))-min(timeData.avgLossTime) AS FLOAT)/(max(timeData.avgLossTime)-min(timeData.avgLossTime))) AS blueAvgLossTime 
            FROM 
            characters AS red 
            INNER JOIN
            characters AS blue 
            INNER JOIN	
            (SELECT strftime('%s', avgMatchTime)-strftime('%s', '00:00:00') AS avgMatchTime, 
			strftime('%s', avgWinTime)-strftime('%s', '00:00:00') AS avgWinTime, 
			strftime('%s', avgLossTime)-strftime('%s', '00:00:00') AS avgLossTime FROM characters
			) AS timeData 
            WHERE red.mode = "Matchmaking" AND red.name = ? AND blue.mode = "Matchmaking" AND blue.name = ?
        )

    ''', (name1, name2, name1, name2)).fetchone()

    conn.close()

    nameMat = numpy.zeros((1, len(nameDict)))
    dataMat = numpy.zeros((1, len(data)))

    nameMat[0,nameDict[name1]] = 1
    nameMat[0,nameDict[name2]] = 1

    redStats = data[:int(len(data)/2)]
    blueStats = data[int(len(data)/2):]

    if nameDict[name1] < nameDict[name2]:
        data = redStats+blueStats
    elif nameDict[name2] < nameDict[name1]:
        data = blueStats+redStats

    dataMat[0] = data

    res = model.predict([nameMat, dataMat])[0]

    cutoff = 0

    print(res[nameDict[name1]])
    print(res[nameDict[name2]])

    if res[nameDict[name1]] - res[nameDict[name2]] > cutoff:
        print(name1)

    elif res[nameDict[name2]] - res[nameDict[name1]] > cutoff:
        print(name2)

    print("")

def dataGenerator(data, labels, batch_size=1):
    while True:

        index = list(range(data[0].shape[0]))
        random.shuffle(index)
        data = [data[0][index], data[1][index]]
        labels = [labels[0][index]]

        for i in range(data[0].shape[0]):
            if i != 0 and i % batch_size == 0:
                yield [[data[0][i-batch_size:i].todense(), data[1][i-batch_size:i]], [labels[0][i-batch_size:i].todense()]]

def loadData(dbPath='data/data.db'):

    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('BEGIN TRANSACTION')

    cursor.execute('''

	    DROP INDEX IF EXISTS temp

	''')

    conn.commit()

    cursor.execute('''

	    CREATE INDEX temp ON fights(mode)

	''')

    conn.commit()

    data = cursor.execute('''

        SELECT redName, blueName, winner, 

            redAuthor.winrate AS redAuthorWinrate, red.wins AS redWins, red.losses AS redLosses, red.hist AS redHist, red.upsetBets AS redUpsetBets, 
            red.upsetMu AS redUpsetMu, red.expectedProfits AS redExpectedProfits, red.expectedProfitsAvg AS redExpectedProfitsAvg, 
            CAST(red.wins AS FLOAT)/(red.wins+red.losses) AS redWinrate, 
            strftime('%s', red.avgMatchTime)-strftime('%s', '00:00:00') AS redAvgMatchTime, 
            strftime('%s', red.avgWinTime)-strftime('%s', '00:00:00') AS redAvgWinTime, 
            strftime('%s', red.avgLossTime)-strftime('%s', '00:00:00') AS redAvgLossTime, 
            red.avgOdds AS redAvgOdds, red.avgWinOdds AS redAvgWinOdds, red.avgLossOdds AS redAvgLossOdds, 
            red.mu AS redMu, red.sigma AS redSigma, 

            blueAuthor.winrate AS blueAuthorWinrate, blue.wins AS blueWins, blue.losses AS blueLosses, blue.hist AS blueHist, blue.upsetBets AS blueUpsetBets, 
            blue.upsetMu AS blueUpsetMu, blue.expectedProfits AS blueExpectedProfits, blue.expectedProfitsAvg AS blueExpectedProfitsAvg, 
            CAST(blue.wins AS FLOAT)/(blue.wins+blue.losses) AS blueWinrate, 
            strftime('%s', blue.avgMatchTime)-strftime('%s', '00:00:00') AS blueAvgMatchTime, 
            strftime('%s', blue.avgWinTime)-strftime('%s', '00:00:00') AS blueAvgWinTime, 
            strftime('%s', blue.avgLossTime)-strftime('%s', '00:00:00') AS blueAvgLossTime, 
            blue.avgOdds AS blueAvgOdds, blue.avgWinOdds AS blueAvgWinOdds, blue.avgLossOdds AS blueAvgLossOdds, 
            blue.mu AS blueMu, blue.sigma AS blueSigma

            from fights 
            INNER JOIN characters AS red ON fights.redName = red.name AND fights.mode = red.mode 
            INNER JOIN characters AS blue ON fights.blueName = blue.name AND fights.mode = blue.mode 
			LEFT JOIN author AS redAuthor ON fights.redAuthor = redAuthor.name
			LEFT JOIN author AS blueAuthor ON fights.blueAuthor = blueAuthor.name
            WHERE (fights.mode = "Matchmaking" or fights.mode = "Tournament") AND red.wins+red.losses >= 10 AND blue.wins+blue.losses >= 10
            AND red.avgWinTime IS NOT NULL AND red.avgLossTime IS NOT NULL AND blue.avgWinTime IS NOT NULL AND blue.avgLossTime IS NOT NULL 
            AND red.avgWinOdds IS NOT NULL AND red.avgLossOdds IS NOT NULL AND blue.avgWinOdds IS NOT NULL AND blue.avgLossOdds IS NOT NULL

    ''').fetchall()

    headerList = [x[0] for x in cursor.description]

    cursor.execute('''

	    DROP INDEX temp

	''')

    conn.commit()

    conn.close()

    df = pandas.DataFrame(data, columns=headerList)

    df = df.dropna()

    for value in [
        "redAvgMatchTime", "redAvgWinTime", "redAvgLossTime", 
        "blueAvgMatchTime", "blueAvgWinTime", "blueAvgLossTime"
        ]:
        df[value] = (df[value]-df[value].min()) / (df[value].max()-df[value].min())

    for value in [
        "redWins", "redLosses", "redExpectedProfits", "redExpectedProfitsAvg", "redAvgOdds", "redAvgWinOdds", "redAvgLossOdds","redMu", "redSigma", 
        "blueWins", "blueLosses", "blueExpectedProfits", "blueExpectedProfitsAvg", "blueAvgOdds","blueAvgWinOdds", "blueAvgLossOdds", "blueMu", "blueSigma" 
        ]:
        df[value] = (df[value]-df[value].mean()) / df[value].std()

    # df = df.drop(["redWins","redLosses", "blueWins", "blueLosses", 
    #     "redMu", "blueMu", "redSigma", "blueSigma", 
    #     "redExpectedProfits", "redExpectedProfitsAvg", "blueExpectedProfits", "blueExpectedProfitsAvg"], axis=1)

    df = df.drop(["redWins","redLosses", "blueWins", "blueLosses", 
        "redMu", "blueMu", "redSigma", "blueSigma"], axis=1)

    print(df.columns.values.tolist())

    return df.values.tolist()

def makeDataTrain(data, nameDict):
    trainData, trainLabels = buildMatrix(data[:int(0.9*len(data))], nameDict)

    trainEvalData, trainEvalLabels = buildMatrix(data[int(0.9*0.9*len(data)):int(0.9*len(data))], nameDict, isSparse=0)

    evalData, evalLabels = buildMatrix(data[int(0.9*len(data)):], nameDict, isSparse=0)

    return trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels

def makeDataEval(data, nameDict):
    evalData, evalLabels = buildMatrix(data, nameDict, isSparse=0)

    return evalData, evalLabels