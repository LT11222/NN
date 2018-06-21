import modelFuncs
import backend
import itertools
from keras import backend as K

def train(data, samples, epochs, optimizer, lr, l2):
    
    batch_size = 128

    trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels = data

    model = modelFuncs.genModel(trainData[0].shape[1], trainData[1].shape[1], modelFuncs.optimizer(optimizer, lr), l2)

    checkpoint = modelFuncs.ModelCheckpoint('./models/{}_{}_{}_{}_{}.hdf5'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')), monitor='val_acc', save_best_only=True, mode='max')
    tensorboard = modelFuncs.TensorBoard(log_dir='./logs/{}_{}_{}_{}_{}/'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')))
    earlystopping = modelFuncs.EarlyStopping(monitor='loss', min_delta=0.001, patience=10)
    reducelronplateau = modelFuncs.ReduceLROnPlateau(monitor='val_acc', factor=0.5, min_delta=0.01, patience=5)

    model.fit_generator(backend.dataGenerator(trainData, trainLabels, batch_size), 
        epochs=epochs,
        steps_per_epoch=trainData[0].shape[0]/batch_size, 
        validation_data=[evalData, evalLabels], 
        callbacks=[checkpoint, tensorboard, earlystopping, reducelronplateau])

    scoreTrain = model.evaluate(trainEvalData, trainEvalLabels, batch_size=batch_size)
    print(scoreTrain)
    scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
    print(scoreEval)

    print('')

    backend.predict(model, trainEvalData, trainEvalLabels)
    backend.predict(model, evalData, evalLabels)

    K.clear_session()

if __name__ == "__main__":

    samples, testsamples, epochs, optimizer, lr, l2 = backend.readConfig()
    data = backend.getData(samples, testsamples, "train")

    parameters = itertools.product(epochs, optimizer, lr, l2)

    for value in parameters:
        train(data, samples, *value)