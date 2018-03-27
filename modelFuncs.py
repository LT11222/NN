import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model

def optimizer(name, val):
    if name == "SGD":
        return SGD(lr=val)
    if name == "RMSprop":
        return RMSprop(lr=val)
    if name == "Adagrad":
        return Adagrad(lr=val)
    if name == "Adadelta":
        return Adadelta(lr=val)
    if name == "Adam":
        return Adam(lr=val)
    if name == "Adamax":
        return Adamax(lr=val)
    if name == "Nadam":
        return Nadam(lr=val)

def genModel(nameCount, dataCount, optimizer):

    units = int(nameCount/8)
    # units = nameCount

    nameInput = Input(shape=(nameCount,), name='nameInput')
    x = Dense(units, kernel_regularizer=l2(l=0.0001))(nameInput)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    nameOutput = Dense(nameCount, name='nameOutput')(x)

    dataInput = Input(shape=(dataCount,), name='dataInput')
    x = Dense(dataCount, kernel_regularizer=l2(l=0.0001))(dataInput)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # x = Dense(dataCount, kernel_regularizer=l2(l=0.0001))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # # x = Dropout(0.5)(x)

    dataOutput = Dense(dataCount, name='dataOutput')(x)

    x = keras.layers.concatenate([nameOutput, dataOutput])

    x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # x = Dense(units, kernel_regularizer=l2(l=0.0001))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # # x = Dropout(0.5)(x)

    x = Dense(nameCount)(x)
    x = BatchNormalization()(x)
    mainOutput = Activation('softmax', name='mainOutput')(x)

    model = Model(inputs=[nameInput, dataInput], outputs=[mainOutput])

    model.compile(loss='categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy'])

    plot_model(model, to_file='model.png')

    model.summary()

    return model