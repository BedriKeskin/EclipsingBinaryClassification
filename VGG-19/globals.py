from enum import Enum
import cv2
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
import os

PNG_size = (224, 224)


class Roche(Enum):
    Detached = 0
    SemiDetached = 1
    OverContact = 2
    Ellipsoidal = 3


def ImagesLabels(pngFiles):
    images = []
    labels = []

    for png in pngFiles:
        try:
            image = load_img(png, target_size=PNG_size)
            image = img_to_array(image)
            image = image[:, :, ::-1]
            image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
            image = image.astype('float32')
            image = preprocess_input(image)

            images.append(image)

            Morph = float(os.path.basename(png).split("_")[2])

            if Morph <= 0.4:
                labels.append(Roche.Detached.value)
            elif Morph <= 0.7:
                labels.append(Roche.SemiDetached.value)
            elif Morph <= 0.8:
                labels.append(Roche.OverContact.value)
            else:
                labels.append(Roche.Ellipsoidal.value)

        except Exception as e:
            print(f"{e} Error", png)

    return images, labels


def ImagesAndLabels(pngFiles, classType):
    images = []
    labels = []

    for png in pngFiles:
        try:
            image = load_img(png, target_size=PNG_size)
            image = img_to_array(image)
            image = image[:, :, ::-1]
            image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
            image = image.astype('float32')
            image = preprocess_input(image)

            images.append(image)
            labels.append(classType.value)

        except Exception as e:
            print(f"{e} Error")

    return images, labels


def ImagesAndLabels2(pngFiles, classType):  # eski
    images = []
    labels = []

    for png in pngFiles:
        try:
            img = cv2.imread(png)
            img_fromarray = Image.fromarray(img, "RGB")
            img_resized = img_fromarray.resize(PNG_size)
            img_np = np.array(img_resized)
            images.append(img_np)
            labels.append(classType.value)

        except Exception as e:
            print(f"{e} Error")

    return images, labels
