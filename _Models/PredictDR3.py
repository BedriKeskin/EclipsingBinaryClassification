# MergeKeplerTessGaiaUnique.csv dosyasını alır, Prediction DR3ClassPrediction ve maxValue
# sütunlarını ekler, Öğrenmiş ML modelini açar, Gaia/PNGMowlaviFit klasorundeki png dosyalarını
# alır, sınıflandırır, yeni isimli ilgili klasore kopyalar, 3 yeni sütunu doldurup
# MergeKeplerTessGaiaUniquePredict.csv olarak diske kopyalar.

import glob
import os
import shutil
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
import globals

KeplerTessGaia = pd.read_csv('../Gaia/KeplerTessGaia.csv', index_col=False,
                             dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

KeplerTessGaia.insert(7, "Prediction", "")
KeplerTessGaia.insert(8, "DR3ClassPrediction", "")
KeplerTessGaia.insert(9, "maxValue", np.NaN)

model = keras.saving.load_model('./Sequential_20240518-094535.keras')

DR3PNG = glob.glob("../Gaia/PNGMowlaviFit/*.png")

for index, png in enumerate(DR3PNG):
    print("\n", index, png)
    image = load_img(png, target_size=(224, 224))
    image = img_to_array(image)
    image = image[None, :, :, ::-1]
    image = image.reshape(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
    image = image.astype('float32')
    image = preprocess_input(image)

    prediction = model.predict(image, verbose=1, batch_size=224)
    maxValue, index = max([(value, index) for index, value in enumerate(prediction[0])])

    if index == 0:
        Type = globals.Roche.Detached.name
    elif index == 1:
        Type = globals.Roche.SemiDetached.name
    elif index == 2:
        Type = globals.Roche.OverContact.name
    elif index == 3:
        Type = globals.Roche.Ellipsoidal.name

    DR3ID = Path(png).stem
    KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'Prediction'] = ', '.join(
        [str(x) for x in prediction[0]])
    KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'DR3ClassPrediction'] = Type
    KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'maxValue'] = maxValue
    filename = DR3ID + "_" + str(maxValue) + ".png"
    shutil.copyfile(png, os.path.join("./PNGMowlaviFit_Sequential/" + Type, filename))

KeplerTessGaia.to_csv("KeplerTessGaia_Predict_Sequential.csv", encoding='utf-8', index=False)
