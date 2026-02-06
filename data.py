import glob
import os

import pandas as pd

import globals


def ImagesLabelsInFolder(folder):
    PNGs = glob.glob(folder+"/*.png")
    # PNGs = random.sample(PNGs, 10)
    images, labels = globals.ImagesLabels(PNGs)

    return images, labels


def ImagesAndLabelsInFolder(folder):
    Detached = glob.glob(folder+"/Detached/*.png")
    SemiDetached = glob.glob(folder+"/SemiDetached/*.png")
    OverContact = glob.glob(folder+"/OverContact/*.png")
    Ellipsoidal = glob.glob(folder+"/Ellipsoidal/*.png")

    images = []
    images += globals.ImagesAndLabels(Detached, globals.Roche.Detached)[0]
    images += globals.ImagesAndLabels(SemiDetached, globals.Roche.SemiDetached)[0]
    images += globals.ImagesAndLabels(OverContact, globals.Roche.OverContact)[0]
    images += globals.ImagesAndLabels(Ellipsoidal, globals.Roche.Ellipsoidal)[0]

    labels = []
    labels += globals.ImagesAndLabels(Detached, globals.Roche.Detached)[1]
    labels += globals.ImagesAndLabels(SemiDetached, globals.Roche.SemiDetached)[1]
    labels += globals.ImagesAndLabels(OverContact, globals.Roche.OverContact)[1]
    labels += globals.ImagesAndLabels(Ellipsoidal, globals.Roche.Ellipsoidal)[1]

    return images, labels


def ImagesAndLabelsPredictInFolder(folder):
    Detached = glob.glob(folder+"/Detached_prediction/*.png")
    SemiDetached = glob.glob(folder+"/SemiDetached_prediction/*.png")
    OverContact = glob.glob(folder+"/OverContact_prediction/*.png")
    Ellipsoidal = glob.glob(folder+"/Ellipsoidal_prediction/*.png")

    images = []
    images += globals.ImagesAndLabels(Detached, globals.Roche.Detached)[0]
    images += globals.ImagesAndLabels(SemiDetached, globals.Roche.SemiDetached)[0]
    images += globals.ImagesAndLabels(OverContact, globals.Roche.OverContact)[0]
    images += globals.ImagesAndLabels(Ellipsoidal, globals.Roche.Ellipsoidal)[0]

    labels = []
    labels += globals.ImagesAndLabels(Detached, globals.Roche.Detached)[1]
    labels += globals.ImagesAndLabels(SemiDetached, globals.Roche.SemiDetached)[1]
    labels += globals.ImagesAndLabels(OverContact, globals.Roche.OverContact)[1]
    labels += globals.ImagesAndLabels(Ellipsoidal, globals.Roche.Ellipsoidal)[1]

    return images, labels
