import tensorflow as tf

import keras
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout  # import needed layers
from keras.metrics import Recall, Precision
from keras.models import Sequential  # import sequential API. Sequential is good for 1 data input.
from keras.src.layers import BatchNormalization, Activation
from keras.optimizers import Adam

from keras.regularizers import l2

import globals


def create_datagen():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def vgg19_Berk(classes):
    vgg19_model = VGG19(
        input_shape=(globals.PNG_size[0], globals.PNG_size[1], 3),
        weights='imagenet',  # Use pre-trained weights from ImageNet
        include_top=False,  # Do not include the top fully connected layers
        classes=classes
    )

    # Set only the last block of the VGG19 model to be trainable
    for layer in vgg19_model.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(vgg19_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    return model


def vgg19(classes):
    vgg19_model = VGG19(
        input_shape=(globals.PNG_size[0], globals.PNG_size[1], 3),
        weights=None,  # bu None olacak
        include_top=True,  # bu true olacak
        classes=classes
    )

    for layer in vgg19_model.layers:
        layer.trainable = True

    model = Sequential()
    model.add(vgg19_model)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', 'Precision', 'Recall'])

    return model


def vgg19_2(classes):
    vgg19_model = VGG19(
        input_shape=(globals.PNG_size[0], globals.PNG_size[1], 3),
        weights=None,  # bu None olacak
        include_top=True,  # bu true olacak
        classes=classes
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
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', 'Precision', 'Recall'])

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
        Masking(mask_value=0.,
                input_shape=(X_train.shape[1], 1)))  # Masking katmanı, padding değerlerinin öğrenilmesini önler
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
