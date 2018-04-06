import modelFuncs
import backend
import itertools
from keras import backend as K

def eval(data, samples, epochs, optimizer, lr, l2):

    batch_size = 128

    evalData, evalLabels = data

    try:
        model = modelFuncs.load_model('./models/{}_{}_{}_{}_{}.hdf5'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')))
    except:
        return

    print('{}_{}_{}_{}_{}.hdf5'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')))

    scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
    print(scoreEval)
    print('')

    backend.predict(model, evalData, evalLabels)
    print('')

    K.clear_session()

if __name__ == "__main__":

    samples, testsamples, epochs, optimizer, lr, l2 = backend.readConfig()
    data = backend.getData(samples, testsamples, "eval")

    parameters = itertools.product(epochs, optimizer, lr, l2)

    for value in parameters:
        eval(data, samples, *value)