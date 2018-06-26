import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, GaussianNoise, concatenate
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
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

def genModelNames(nameCount, optimizer, l2Val):

    units = int(nameCount/8)

    nameInput = Input(shape=(nameCount,), name='nameInput')
    x = Dense(units, kernel_regularizer=l2(l=l2Val))(nameInput)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(units, kernel_regularizer=l2(l=l2Val))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(nameCount)(x)
    x = BatchNormalization()(x)

    nameOutput = Activation('softmax', name='nameOutput')(x)

    model = Model(inputs=[nameInput], outputs=[nameOutput])

    model.compile(loss='categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy'])

    plot_model(model, to_file='modelNames.png')

    model.summary()

    return model

def genModelData(dataCount, optimizer, l2Val):

    dataInput = Input(shape=(dataCount,), name='dataInput')
    x = Dense(dataCount, kernel_regularizer=l2(l=l2Val))(dataInput)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # x = Dense(int(dataCount*2), kernel_regularizer=l2(l=l2Val))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    x = Dense(dataCount, kernel_regularizer=l2(l=l2Val))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(2)(x)
    x = BatchNormalization()(x)

    dataOutput = Activation('softmax', name='dataOutput')(x)

    model = Model(inputs=[dataInput], outputs=[dataOutput])

    model.compile(loss='categorical_crossentropy',
        optimizer = optimizer,
        metrics=['accuracy'])

    plot_model(model, to_file='modelData.png')

    model.summary()

    return model