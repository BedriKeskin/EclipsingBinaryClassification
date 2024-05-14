# MergeKeplerTessGaiaUnique.csv dosyasını alır, Prediction DR3ClassPrediction ve maxValue
# sütunlarını ekler, Öğrenmiş ML modelini açar, PNGMowlaviFit klasorundeki png dosyalarını
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

MergeKeplerTessGaiaUnique = pd.read_csv(
    '/Users/tiga/My Drive/Doktora/EclipsingBinaryClassification/Gaia/MergeKeplerTessGaiaUnique.csv', index_col=False,
    dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

MergeKeplerTessGaiaUnique.insert(7, "Prediction", "")
MergeKeplerTessGaiaUnique.insert(8, "DR3ClassPrediction", "")
MergeKeplerTessGaiaUnique.insert(9, "maxValue", np.NaN)

model = keras.saving.load_model(
    '/Users/tiga/My Drive/Doktora/EclipsingBinaryClassification/VGG-19/MyVgg19_20240123-014058.keras')

DR3PNG = glob.glob("/Users/tiga/My Drive/Doktora/EclipsingBinaryClassification/Gaia/PNGMowlaviFit/*.png")

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
        Type = "Detached"
    elif index == 1:
        Type = "SemiDetached"
    elif index == 2:
        Type = "OverContact"
    elif index == 3:
        Type = "Ellipsoidal"

    DR3ID = Path(png).stem
    MergeKeplerTessGaiaUnique.loc[MergeKeplerTessGaiaUnique['DR3'] == DR3ID, 'Prediction'] = ', '.join([str(x) for x in prediction[0]])
    MergeKeplerTessGaiaUnique.loc[MergeKeplerTessGaiaUnique['DR3'] == DR3ID, 'DR3ClassPrediction'] = Type
    MergeKeplerTessGaiaUnique.loc[MergeKeplerTessGaiaUnique['DR3'] == DR3ID, 'maxValue'] = maxValue
    filename = DR3ID + "_" + str(maxValue) + ".png"
    shutil.copyfile(png, os.path.join("/Users/tiga/My Drive/Doktora/EclipsingBinaryClassification/Gaia/PNGMowlaviFit/"+Type, filename))

MergeKeplerTessGaiaUnique.to_csv("MergeKeplerTessGaiaUniquePredict.csv", encoding='utf-8', index=False)
