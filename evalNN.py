import backend
import time

if __name__ == "__main__":

    data = backend.loadData('data/data.db')

    optDict = backend.genOptDict()
    
    start = 10000
    samples = 10000
    batch_size = 128

    testData, testLabels, testEvalData, testEvalLabels, evalData, evalLabels = backend.makeData(data[start:start+samples], backend.buildDict(data), samples)

    for name, value in optDict.items():
        for lr in value:

            model = backend.genModel(testData[0].shape[1], testData[1].shape[1], backend.optimizer(name, lr))

            while True:

                try:
                    model.load_weights('./models/model.hdf5')
                except:
                    continue

                scoretest = model.evaluate(testEvalData, testEvalLabels, batch_size=batch_size)
                print(scoretest)
                scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
                print(scoreEval)

                print('')

                backend.predict(model, testEvalData, testEvalLabels)
                backend.predict(model, evalData, evalLabels)

                print('')

                time.sleep(5)