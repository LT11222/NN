import modelFuncs
import backend
import time

if __name__ == "__main__":

    samples, testsamples = backend.readConfig()

    data = backend.loadData('data/data.db')

    batch_size = 128

    evalData, evalLabels = backend.makeDataEval(data[samples:samples+testsamples], backend.buildDict(data[:samples+testsamples]))

    model = modelFuncs.load_model('./models/model.hdf5')

    while True:

        try:
            model.load_weights('./models/model.hdf5')
        except:
            time.sleep(5)
            continue

        scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
        print(scoreEval)

        print('')

        backend.predict(model, evalData, evalLabels)

        print('')

        time.sleep(5)