import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
import os

folderName = "LombScargle"

if not os.path.exists(folderName):
    os.makedirs(folderName)

if not os.path.exists(os.path.join(folderName, 'Detached')):
    os.makedirs(os.path.join(folderName, 'Detached'))

if not os.path.exists(os.path.join(folderName, 'SemiDetached')):
    os.makedirs(os.path.join(folderName, 'SemiDetached'))

if not os.path.exists(os.path.join(folderName, 'OverContact')):
    os.makedirs(os.path.join(folderName, 'OverContact'))

if not os.path.exists(os.path.join(folderName, 'Ellipsoidal')):
    os.makedirs(os.path.join(folderName, 'Ellipsoidal'))

LCdatas = glob.glob("./LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)
        LC = LC[:].values
        LC = pd.DataFrame(LC, columns=['index', 'BJD', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux',
                                           'dtr_err', 'blanc'])
        LC = LC.drop('index', axis=1)
        LC = LC.drop('phase', axis=1)
        LC = LC.drop('raw_err', axis=1)

        LC['BJD'] = LC['BJD'] + 2400000
        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')

        time = LC['BJD']
        flux = LC['dtr_flux']

        frequencies, power = LombScargle(time, flux).autopower()

        max_power_index = np.argmax(power)
        strongest_frequency = frequencies[max_power_index]

        P = float(os.path.basename(LCdata).split("_")[6][:-4])
        # En güçlü frekansı yazdır
        print(f"frequencies len: {len(frequencies)} , max power: {power[max_power_index]} , En güçlü frekans: {strongest_frequency} 1/gün, gün: {1/strongest_frequency} , P: {P} , P/P' : {P*strongest_frequency}")

        morph = float(os.path.basename(LCdata).split("_")[2])
        label = ""

        if morph <= 0.4:
            label = "Detached"
        elif morph <= 0.7:
            label = "SemiDetached"
        elif morph <= 0.8:
            label = "OverContact"
        else:
            label = "Ellipsoidal"

        # Power spectrum'u plotla
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, power)
        plt.xlabel('Frekans (1/gün)')
        plt.ylabel('Güç')
        plt.title('Lomb-Scargle Power Spectrum of the Light Curve')
        plt.grid(True)
        plt.savefig(folderName + '/' + label + '/' + os.path.basename(LCdata)[:-4] + '_Pls_'+ str(1/strongest_frequency) + '_PoPls_' + str(P*strongest_frequency) + '.png')
        plt.close()

    except Exception as e:
        print(f"Error: {e}")