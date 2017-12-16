import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K

import backend

if __name__ == "__main__":

    data = backend.loadData('data/data.db')

    optDict = backend.genOptDict()

    samples = 50000
    batch_size = 512

    trainData, trainLabels, trainEvalData, trainEvalLabels, evalData, evalLabels = backend.makeData(data, samples)

    for name, value in optDict.items():
        for lr in value:

            model = backend.genModel(trainData[0].shape[1], trainData[1].shape[1], backend.optimizer(name, lr))

            filepath='./models/model.hdf5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
            earlystopping = EarlyStopping(monitor='loss',min_delta=0.001, patience=10)
            tensorboard = TensorBoard(log_dir='./logs/{}/{}'.format(name, str(lr).replace('.', '_')))

            model.fit_generator(backend.dataGenerator(trainData, trainLabels, batch_size), 
                epochs=100, 
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

            K.clear_session()