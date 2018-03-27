import backend
import model
import time

import configparser

if __name__ == "__main__":

    config = configparser.RawConfigParser()
    config.read("config/config.cfg")
    samples = int(config.get("Saltybet", "samples"))
    testsamples = int(config.get("Saltybet", "testsamples"))

    data = backend.loadData('data/data.db')

    optDict = backend.genOptDict()
    
    batch_size = 128

    testData, testLabels, testEvalData, testEvalLabels, evalData, evalLabels = backend.makeData(data[samples:samples+testsamples], backend.buildDict(data[:samples+testsamples]), testsamples)

    for name, value in optDict.items():
        for lr in value:

            model = model.genModel(testData[0].shape[1], testData[1].shape[1], model.optimizer(name, lr))

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