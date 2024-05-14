import keras
from keras import Sequential
from keras.applications import VGG19
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.metrics import Recall, Precision
from keras.src.layers import BatchNormalization, Activation
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


def vgg19_3(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
