import backend
import time

if __name__ == "__main__":

    data = backend.loadData('data/data.db')

    optDict = backend.genOptDict()
    
    samples = 10000
    batch_size = 128

    trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels = backend.makeData(data, samples)

    for name, value in optDict.items():
        for lr in value:

            model = backend.genModel(trainData[0].shape[1], trainData[1].shape[1], backend.optimizer(name, lr))

            while True:

                try:
                    model.load_weights('./models/model.hdf5')
                except:
                    continue

                scoreTrain = model.evaluate(trainEvalData, trainEvalLabels, batch_size=batch_size)
                print(scoreTrain)
                scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
                print(scoreEval)

                print('')

                backend.predict(model, trainEvalData, trainEvalLabels)
                backend.predict(model, evalData, evalLabels)

                print('')

                time.sleep(5)