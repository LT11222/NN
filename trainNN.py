import modelFuncs
import backend

if __name__ == "__main__":

    samples, testsamples = backend.readConfig()

    data = backend.loadData('data/data.db')

    batch_size = 512

    trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels = backend.makeDataTrain(data[:samples], backend.buildDict(data[:samples+testsamples]))

    model = modelFuncs.genModel(trainData[0].shape[1], trainData[1].shape[1], modelFuncs.optimizer("Adam", 0.0001))

    filepath='./models/model.hdf5'
    checkpoint = modelFuncs.ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
    earlystopping = modelFuncs.EarlyStopping(monitor='loss',min_delta=0.001, patience=10)
    tensorboard = modelFuncs.TensorBoard(log_dir='./logs/{}/{}/'.format("Adam", str(0.0001).replace('.', '_')))

    model.fit_generator(backend.dataGenerator(trainData, trainLabels, batch_size), 
        epochs=1000,
        steps_per_epoch=trainData[0].shape[0]/batch_size, 
        validation_data=[evalData, evalLabels], 
        callbacks=[checkpoint, earlystopping, tensorboard])

    scoreTrain = model.evaluate(trainEvalData, trainEvalLabels, batch_size=batch_size)
    print(scoreTrain)
    scoreEval = model.evaluate(evalData, evalLabels, batch_size=batch_size)
    print(scoreEval)

    print('')

    backend.predict(model, trainEvalData, trainEvalLabels)
    backend.predict(model, evalData, evalLabels)