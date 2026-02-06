# MergeKeplerTessGaiaUnique.csv dosyasını alır, Prediction DR3ClassPrediction ve maxValue
# sütunlarını ekler, Öğrenmiş ML modelini açar, Gaia/PNGMowlaviFit klasorundeki png dosyalarını
# alır, sınıflandırır, yeni isimli ilgili klasore kopyalar, 3 yeni sütunu doldurup
# MergeKeplerTessGaiaUniquePredict.csv olarak diske kopyalar.

import glob
import os
import shutil
from pathlib import Path

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Garantiye almak icin
import tf_keras as keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import globals

KeplerTessGaia = pd.read_csv('../Gaia/KeplerTessGaia.csv', index_col=False,
                             dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

KeplerTessGaia.insert(7, "Prediction", "")
KeplerTessGaia.insert(8, "DR3ClassPrediction", "")
KeplerTessGaia.insert(9, "maxValue", np.nan)

#Berk Sequential
model = keras.saving.load_model('Vgg19_20240825-125419.keras')

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
    DR3ID = DR3ID.split("_")[0]
    KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'Prediction'] = ', '.join([str(x) for x in prediction[0]])
    KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'DR3ClassPrediction'] = Type
    KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'maxValue'] = maxValue
    # filename = DR3ID + "_" + str(maxValue) + ".png"
    filename = DR3ID + ".png"
    # shutil.copyfile(png, os.path.join("./PNGMowlaviFit_Berk/" + Type, filename))
    shutil.copyfile(png, os.path.join("./PNGMowlaviFit/" + Type, filename))

'''
# Flux
model = keras.saving.load_model('./Flux_20240814-010156.keras')

folderNameDR3Prediction = "PNG_flux_DR3Prediction"
globals.CreateFolder(folderNameDR3Prediction)

KeplerTessGaia = pd.read_csv('../Gaia/KeplerTessGaia.csv', index_col=False,
                             dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

KeplerTessGaia.insert(7, "Prediction", "")
KeplerTessGaia.insert(8, "DR3ClassPrediction", "")
KeplerTessGaia.insert(9, "maxValue", np.NaN)

LCdatas = glob.glob("../Gaia/LCdata/*.xml")

for index, file in enumerate(LCdatas):  # (sample(LCdatas, 100)):
    print("\n", index, file)

    try:
        LC = globals.votable_to_pandas(file)
        LC = LC[LC['band'] == 'G']  # G band only
        LC = LC[LC['rejected_by_photometry'] == False]
        LC = LC[LC['rejected_by_variability'] == False]

        LC = LC.drop('source_id', axis=1)
        LC = LC.drop('flux', axis=1)
        LC = LC.drop('flux_error', axis=1)
        LC = LC.drop('flux_over_error', axis=1)
        LC = LC.drop('transit_id', axis=1)
        LC = LC.drop('rejected_by_photometry', axis=1)
        LC = LC.drop('rejected_by_variability', axis=1)
        LC = LC.drop('other_flags', axis=1)
        LC = LC.drop('solution_id', axis=1)
        LC = LC.drop('band', axis=1)

        LC.sort_values(by=['time'], inplace=True)
        LC['time'] = LC['time'] + 2450000
        LC['mag'] = -1 * LC['mag']

        x = np.array(LC['mag'])
        LC['mag'] = (x - np.mean(x)) / np.max(np.abs(x - np.mean(x)))  # normalize

        LC.index = pd.to_datetime(LC['time'], origin='julian', unit='D')
        LC = LC.drop('time', axis=1)
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(file).split("_")[3])
        T0 = T0 + 2450000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = 1 / float(os.path.basename(file).split("_")[5][:-4])

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        print("len(ts_folded) ", len(ts_folded))
        #  ts_binned = aggregate_downsample(ts_folded, n_bins=100, aggregate_func=np.nanmedian)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.1 * u.min, aggregate_func=np.nanmedian)
        non_n_bad_indexes = ~np.isnan(ts_binned.to_pandas().to_numpy()[:, 1])
        ts_binned = ts_binned[non_n_bad_indexes]
        print("len(ts_binned) ", len(ts_binned))

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        # ydata = ts_binned['flux']
        ydata = ts_binned['mag']

        df1 = pd.DataFrame(ts_binned.columns,
                           columns=['time_bin_start', 'time_bin_size', 'mag'])  # Tekrar DataFrame'e donustur
        # df1.set_index('time_bin_start', inplace=True)  # gerek yok

        prediction = model.predict(df1['mag'], verbose=1, batch_size=224)
        maxValue, index = max([(value, index) for index, value in enumerate(prediction[0])])

        if index == 0:
            Type = globals.Roche.Detached.name
        elif index == 1:
            Type = globals.Roche.SemiDetached.name
        elif index == 2:
            Type = globals.Roche.OverContact.name
        elif index == 3:
            Type = globals.Roche.Ellipsoidal.name

        DR3ID = os.path.basename(file).split("_")[0]
        KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'Prediction'] = ', '.join(
            [str(x) for x in prediction[0]])
        KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'DR3ClassPrediction'] = Type
        KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'maxValue'] = maxValue

        KeplerAndTessId = ""

        KeplerId = KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'KIC'].values[0]
        if pd.isna(KeplerId):
            KeplerAndTessId += 'KIC'
        else:
            KeplerAndTessId += 'KIC' + KeplerId

        TessId = KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'TIC'].values[0]
        if pd.isna(TessId):
            KeplerAndTessId += '_TIC'
        else:
            KeplerAndTessId += '_TIC' + TessId

        # Plot the result
        plt.plot(xdata, ydata, color='black', marker='.', ls='')
        plt.savefig(folderNameDR3Prediction + '/' + Type + '/' + KeplerAndTessId + '_' + os.path.basename(file)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print("Error: ", e, file)
'''

KeplerTessGaia.to_csv("KeplerTessGaia_Predict.csv", encoding='utf-8', index=False)

true_labels = KeplerTessGaia['KeplerClass'].fillna('') + KeplerTessGaia['TessClass'].fillna('')
predicted_class_names = KeplerTessGaia['DR3ClassPrediction']

accuracy = accuracy_score(true_labels, predicted_class_names)
precision = precision_score(true_labels, predicted_class_names, pos_label='positive', average='micro')
recall = recall_score(true_labels, predicted_class_names, pos_label='positive', average='micro')
f1 = f1_score(true_labels, predicted_class_names, average='weighted')

print('accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('F1 score: ', f1)

labels = globals.GetLabels(globals.Roche)

cm = confusion_matrix(true_labels, predicted_class_names, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix_-_Gaia')
plt.savefig('ConfusionMatrix_Gaia')