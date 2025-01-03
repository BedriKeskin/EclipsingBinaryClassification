import glob
import shutil
from enum import Enum
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
# from keras.utils import img_to_array
# from keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from astropy.io.votable import parse
from more_itertools.more import sample
from keras.src.utils import to_categorical

PNG_size = (224, 224)

class Roche(Enum):
    Detached = 0
    SemiDetached = 1
    OverContact = 2
    Ellipsoidal = 3

class LCMorph(Enum):
    EA = 0
    EB = 1
    EW = 2

def ProcessImage(file):
    image = load_img(file, target_size=PNG_size)
    image = img_to_array(image)
    image = image[:, :, ::-1]
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image = preprocess_input(image)
    return image

def PrepareXy(base_dir):
    dirs = [LCMorph.EA.name, LCMorph.EB.name, LCMorph.EW.name]
    images, labels = [], []
    for directory in dirs:
        for index, file in enumerate(sample(glob.glob(base_dir + directory + "/*.png"), 1000000)):
            image = ProcessImage(file)
            images.append(image)
            if directory == LCMorph.EA.name:
                labels.append(LCMorph.EA.value)
            elif directory == LCMorph.EB.name:
                labels.append(LCMorph.EB.value)
            elif directory == LCMorph.EW.name:
                labels.append(LCMorph.EW.value)
    X = np.array(images)
    labels = np.array(labels)
    y = to_categorical(labels, len(dirs))
    return X, y

def GetLabels(Enumeration):
    labels = []
    for item in Enumeration:
        labels.append(item.name)

    return labels

def GetLabel(morph):
    if morph <= 0.5:
        label = Roche.Detached.name
    elif morph <= 0.7:
        label = Roche.SemiDetached.name
    elif morph <= 0.8:
        label = Roche.OverContact.name
    else:
        label = Roche.Ellipsoidal.name

    return label


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

            if Morph <= 0.5:
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


def CreateFolder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)
    os.makedirs(os.path.join(folder, Roche.Detached.name))
    os.makedirs(os.path.join(folder, Roche.SemiDetached.name))
    os.makedirs(os.path.join(folder, Roche.OverContact.name))
    os.makedirs(os.path.join(folder, Roche.Ellipsoidal.name))


# CVSs
def FluxLabelsInFolder(folder, columnName):
    csvFiles = glob.glob(folder + "/*.csv")

    fluxes = []
    labels = []

    for csv in csvFiles:
        try:
            df = pd.read_csv(csv, delim_whitespace=False, index_col=False)
            df = df.iloc[:, columnName]
            list = df.values.tolist()

            morph = float(os.path.basename(csv).split("_")[2])
            label = GetLabel(morph)

            fluxes.append(list)
            labels.append(label)

        except Exception as e:
            print(f"Error: {e}", csv)

    return fluxes, labels


def votable_to_pandas(votable_file):
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()
