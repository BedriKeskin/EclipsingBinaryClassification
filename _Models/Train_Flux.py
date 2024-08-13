import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import globals
import models

now1 = datetime.datetime.now()
print("Time start: ", now1)

folderName = "PNG_flux"
globals.CreateFolder(folderName)

folderNameDR3Prediction = folderName + "_DR3Prediction"
globals.CreateFolder(folderNameDR3Prediction)

FluxesKepler, LabelsKepler = [], []
CsvsKepler = glob.glob("../Kepler/LCdata/*.csv")

for index, file in enumerate(CsvsKepler[:10]):
    print("\n", index, file)

    try:
        LC = pd.read_csv(file, delim_whitespace=False, index_col=False)
        LC = LC[:].values
        LC = pd.DataFrame(LC,
                          columns=['index', 'BJD', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux',
                                   'dtr_err', 'blanc'])
        LC = LC.drop('index', axis=1)
        LC = LC.drop('phase', axis=1)
        LC = LC.drop('raw_err', axis=1)

        LC['BJD'] = LC['BJD'] + 2400000
        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(file).split("_")[4])
        T0 = T0 + 2400000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(file).split("_")[6][:-4])

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, n_bins=100, aggregate_func=np.nanmedian)

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        ydata = ts_binned['dtr_flux']

        x = np.array(ydata)
        normalized = (x - np.mean(x)) / np.max(np.abs(x - np.mean(x)))
        FluxesKepler.append(normalized)

        df1 = pd.DataFrame()
        df1.insert(0, 'P', P)
        df1.insert(0, 'T0', float(os.path.basename(file).split("_")[4]))
        morph = float(os.path.basename(file).split("_")[2])
        df1.insert(0, 'morph', morph)
        df1.insert(0, 'ID', os.path.basename(file).split("_")[0])

        label = globals.GetLabel(morph)
        df1['label'] = label
        LabelsKepler.append(label)

        df = pd.DataFrame()
        df = pd.concat([df, df1], ignore_index=True)

        plt.plot(xdata, ydata, color='black', marker='.', ls='')
        plt.savefig(folderName + '/' + label + '/' + os.path.basename(file)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print(f"Error: {e}", file)

print("len(FluxesKepler): ", len(FluxesKepler))

FluxesTess, LabelsTess = [], []
CsvsTess = glob.glob("../Tess/LCdata/*.csv")

for index, file in enumerate(CsvsTess[:10]):
    print("\n", index, file)

    try:
        LC = pd.read_csv(file, delim_whitespace=False, index_col=False)
        LC = LC[:].values
        LC = pd.DataFrame(LC, columns=['index', 'BJD', 'ColA', 'NormalizedFlux', 'ColC'])
        LC = LC.drop('index', axis=1)
        LC = LC.drop('ColA', axis=1)
        LC = LC.drop('ColC', axis=1)

        LC['BJD'] = LC['BJD'] + 2450000
        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')
        LC = LC.drop('BJD', axis=1)
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(file).split("_")[4])
        T0 = T0 + 2450000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(file).split("_")[6][:-4])

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, n_bins=100, aggregate_func=np.nanmedian)

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        ydata = ts_binned['NormalizedFlux']

        x = np.array(ydata)
        normalized = (x - np.mean(x)) / np.max(np.abs(x - np.mean(x)))
        FluxesTess.append(normalized)

        df1 = pd.DataFrame()
        df1.insert(0, 'P', P)
        df1.insert(0, 'T0', float(os.path.basename(file).split("_")[4]))
        morph = float(os.path.basename(file).split("_")[2])
        df1.insert(0, 'morph', morph)
        df1.insert(0, 'ID', os.path.basename(file).split("_")[0])

        label = globals.GetLabel(morph)
        df1['label'] = label
        LabelsTess.append(label)

        df = pd.DataFrame()
        df = pd.concat([df, df1], ignore_index=True)

        plt.plot(xdata, ydata, color='black', marker='.', ls='')
        plt.savefig(folderName + '/' + label + '/' + os.path.basename(file)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print(f"Error: {e}", file)

print("len(FluxesTess): ", len(FluxesTess))

Fluxes = FluxesKepler + FluxesTess
Labels = LabelsKepler + LabelsTess

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Labels)

# Dizileri pad edelim ki aynı uzunlukta olsunlar
padded_fluxes = pad_sequences(Fluxes, dtype='float32', padding='post')

# Veriyi train/test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(padded_fluxes, encoded_labels, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, -1)  # (batch_size, timesteps, input_dim)
X_test = np.expand_dims(X_test, -1)  # Aynı şekli test verisine uygulama

checkpoint = ModelCheckpoint(
    filepath='Flux_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True
)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

model = models.flux(X_train)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=1,
                    callbacks=[checkpoint, earlystop])  # , batch_size=256 ,

loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss: ', loss, ' Test Accuracy: ', accuracy)

# Prediction
KeplerTessGaia = pd.read_csv('../Gaia/KeplerTessGaia.csv', index_col=False,
                             dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

KeplerTessGaia.insert(7, "Prediction", "")
KeplerTessGaia.insert(8, "DR3ClassPrediction", "")
KeplerTessGaia.insert(9, "maxValue", np.NaN)

LCdatas = glob.glob("../Gaia/LCdata/*.xml")

for index, file in enumerate(LCdatas[:10]):  # (sample(LCdatas, 100)):
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
        ts_binned = aggregate_downsample(ts_folded, n_bins=100, aggregate_func=np.nanmedian)
        print("len(ts_binned) ", len(ts_binned))
        # non_n_bad_indexes = ~np.isnan(ts_binned.to_pandas().to_numpy()[:, 1])
        # ts_binned = ts_binned[non_n_bad_indexes]

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        # ydata = ts_binned['flux']
        ydata = ts_binned['mag']

        df1 = pd.DataFrame(ts_binned.columns,
                           columns=['time_bin_start', 'time_bin_size', 'mag'])  # Tekrar DataFrame'e dönüştür
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
        print("KeplerId ", KeplerId)

        if not pd.isna(KeplerId):
            KeplerAndTessId += 'KIC' + KeplerId

        TessId = KeplerTessGaia.loc[KeplerTessGaia['DR3'] == DR3ID, 'TIC'].values[0]
        print("TessId ", TessId)

        if not pd.isna(TessId):
            KeplerAndTessId += '_TIC ' + TessId

        print("KeplerAndTessId ", KeplerAndTessId)
        # Plot the result
        plt.plot(xdata, ydata, color='black', marker='.', ls='')
        plt.savefig(folderNameDR3Prediction + '/' + label + '/' + KeplerAndTessId + '_' + os.path.basename(file)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print(f"Error: {e}", file)

KeplerTessGaia.to_csv("KeplerTessGaia_Predict_Flux.csv", encoding='utf-8', index=False)

'''
padded_fluxesGaia = pad_sequences(FluxesGaia, dtype='float32', padding='post')
padded_fluxesGaia = np.expand_dims(padded_fluxesGaia, -1)  # (batch_size, timesteps, input_dim)

predictions = model.predict(padded_fluxesGaia)
predicted_labels = np.argmax(predictions, axis=1)
predicted_class_names = label_encoder.inverse_transform(predicted_labels)

# Örnek gerçek etiketler
true_labels = ["A", "B", "C", "D"]
'''

true_labels = KeplerTessGaia['KeplerClass'].fillna('') + KeplerTessGaia['TessClass'].fillna('')
predicted_class_names = KeplerTessGaia['DR3ClassPrediction']

accuracy = accuracy_score(true_labels, predicted_class_names)
precision = precision_score(true_labels, predicted_class_names, pos_label='positive', average='micro')
recall = recall_score(true_labels, predicted_class_names, pos_label='positive', average='micro')
f1 = f1_score(true_labels, predicted_class_names, average='weighted')

print(f'accuracy: {accuracy}')
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'F1 score: {f1}')

cm = confusion_matrix(true_labels, predicted_class_names, labels=label_encoder.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix_Flux_Gaia')
plt.savefig('ConfusionMatrix_Kepler_Flux_Gaia')

now2 = datetime.datetime.now()
print("Time end: ", now2)
print("Duration: ", now2 - now1)
