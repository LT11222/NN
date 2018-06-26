import modelFuncs
import backend
import itertools
from keras import backend as K

def trainNames(data, samples, epochs, optimizer, lr, l2):
    
    batch_size = 128

    trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels = data

    print(trainData[0].shape)
    print(trainData[1].shape)

    model = modelFuncs.genModelNames(trainData[0].shape[1], modelFuncs.optimizer(optimizer, lr), l2)

    checkpoint = modelFuncs.ModelCheckpoint('./models/names_{}_{}_{}_{}_{}.hdf5'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')), monitor='val_acc', save_best_only=True, mode='max')
    tensorboard = modelFuncs.TensorBoard(log_dir='./logs/names_{}_{}_{}_{}_{}/'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')))
    earlystopping = modelFuncs.EarlyStopping(monitor='loss', min_delta=0.001, patience=10)
    reducelronplateau = modelFuncs.ReduceLROnPlateau(monitor='val_acc', factor=0.5, min_delta=0.01, patience=5)

    model.fit_generator(backend.dataGeneratorNames(trainData[0], trainLabels[0], batch_size), 
        epochs=epochs,
        steps_per_epoch=trainData[0].shape[0]/batch_size, 
        validation_data=[evalData[0], evalLabels[0]], 
        callbacks=[checkpoint, tensorboard, earlystopping, reducelronplateau])

    scoreTrain = model.evaluate(trainEvalData[0], trainEvalLabels[0], batch_size=batch_size)
    print(scoreTrain)
    scoreEval = model.evaluate(evalData[0], evalLabels[0], batch_size=batch_size)
    print(scoreEval)

    print('')

    backend.predict(model, trainEvalData, trainEvalLabels)
    backend.predict(model, evalData, evalLabels)

    K.clear_session()

def trainData(data, samples, epochs, optimizer, lr, l2):
    
    batch_size = 128

    trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels = data

    model = modelFuncs.genModelData(trainData[1].shape[1], modelFuncs.optimizer(optimizer, lr), l2)

    checkpoint = modelFuncs.ModelCheckpoint('./models/data_{}_{}_{}_{}_{}.hdf5'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')), monitor='val_acc', save_best_only=True, mode='max')
    tensorboard = modelFuncs.TensorBoard(log_dir='./logs/data_{}_{}_{}_{}_{}/'.format(samples, epochs, optimizer, str(lr).replace('.', '_'), str(l2).replace('.','_')))
    earlystopping = modelFuncs.EarlyStopping(monitor='loss', min_delta=0.001, patience=10)
    reducelronplateau = modelFuncs.ReduceLROnPlateau(monitor='val_acc', factor=0.5, min_delta=0.01, patience=5)

    model.fit_generator(backend.dataGeneratorData(trainData[1], trainLabels[1], batch_size), 
        epochs=epochs,
        steps_per_epoch=trainData[1].shape[0]/batch_size, 
        validation_data=[evalData[1], evalLabels[1]], 
        callbacks=[checkpoint, tensorboard, earlystopping, reducelronplateau])

    scoreTrain = model.evaluate(trainEvalData[1], trainEvalLabels[1], batch_size=batch_size)
    print(scoreTrain)
    scoreEval = model.evaluate(evalData[1], evalLabels[1], batch_size=batch_size)
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
        trainNames(data, samples, *value)
        # trainData(data, samples, *value)