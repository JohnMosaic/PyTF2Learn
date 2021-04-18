# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def network_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def train_model(x_train, y_train, x_test, y_test, model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), validation_freq=1)
    return history


def make_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.axis('on')
    plt.savefig('vgg13_cifar10_exp2_accuracy.svg')
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.axis('on')
    plt.savefig('vgg13_cifar10_exp2_loss.svg')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    model = network_model()
    history = train_model(x_train, y_train, x_test, y_test, model)
    make_plot(history)
    exit()
