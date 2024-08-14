import keras
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout  # import needed layers
from keras.metrics import Recall, Precision
from keras.models import Sequential  # import sequential API. Sequential is good for 1 data input.
from keras.src.layers import BatchNormalization

import globals


def vgg19():
    vgg19_model = VGG19(
        input_shape=(globals.PNG_size[0], globals.PNG_size[1], 3),
        weights=None,  # bu None olacak
        include_top=True,  # bu true olacak
        classes=4
    )

    for layer in vgg19_model.layers:
        layer.trainable = True

    model = Sequential()
    model.add(vgg19_model)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', Precision(), Recall()], )

    return model


def vgg19_2():
    vgg19_model = VGG19(
        input_shape=(globals.PNG_size[0], globals.PNG_size[1], 3),
        weights=None,  # bu None olacak
        include_top=True,  # bu true olacak
        classes=4
    )

    for layer in vgg19_model.layers:
        layer.trainable = True

    model = Sequential()
    model.add(vgg19_model)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', Precision(), Recall()], )

    return model


def sequential():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def sequential2(X_train):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    return model


def flux(X_train):
    model = Sequential()
    model.add(
        Masking(mask_value=0., input_shape=(X_train.shape[1], 1)))  # Masking katmanı, padding değerlerinin öğrenilmesini önler
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
