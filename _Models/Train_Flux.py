import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import globals
import models

now1 = datetime.datetime.now()
print("Time start: ", now1)

folderName = "PNG_flux"
globals.CreateFolder(folderName)

FluxesKepler, LabelsKepler = [], []
LCdataKepler = glob.glob("../Kepler/LCdata/*.csv")

for index, file in enumerate(LCdataKepler):
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
        print("Error: ", e, file)

print("len(FluxesKepler): ", len(FluxesKepler))

FluxesTess, LabelsTess = [], []
LCdataTess = glob.glob("../Tess/LCdata/*.csv")

for index, file in enumerate(LCdataTess):
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
        print("Error: ", e, file)

print("len(FluxesTess): ", len(FluxesTess))

Fluxes = FluxesKepler + FluxesTess
Labels = LabelsKepler + LabelsTess

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Labels)

# Dizileri pad edelim ki ayni uzunlukta olsunlar
padded_fluxes = pad_sequences(Fluxes, dtype='float32', padding='post')

# Veriyi train/test olarak ayirma
X_train, X_test, y_train, y_test = train_test_split(padded_fluxes, encoded_labels, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, -1)  # (batch_size, timesteps, input_dim)
X_test = np.expand_dims(X_test, -1)  # Ayni sekli test verisine uygulama

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

now2 = datetime.datetime.now()
print("Time end: ", now2)
print("Duration: ", now2 - now1)
