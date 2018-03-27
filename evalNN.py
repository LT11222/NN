import modelFuncs
import backend
import configparser
import time

if __name__ == "__main__":

    config = configparser.RawConfigParser()
    config.read("config/config.cfg")
    samples = int(config.get("Saltybet", "samples"))
    testsamples = int(config.get("Saltybet", "testsamples"))

    data = backend.loadData('data/data.db')

    batch_size = 128

    evalData, evalLabels = backend.makeDataEval(data[samples:samples+testsamples], backend.buildDict(data[:samples+testsamples]), testsamples)

    model = modelFuncs.load_model('./models/model.hdf5')

    while True:
        scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
        print(scoreEval)

        print('')

        backend.predict(model, evalData, evalLabels)

        print('')

        time.sleep(5)