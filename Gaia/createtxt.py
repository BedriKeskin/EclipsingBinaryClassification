import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.signal import medfilt
#from sklearn.ensemble import IsolationForest
import globals


folder = "StarShadow"
if not os.path.exists(folder):
    os.makedirs(folder)

LCdatas = glob.glob("./LCdata/*.xml")

for index, file in enumerate(LCdatas):  # (sample(LCdatas, 100)):
    print("\n", index, file)

    try:
        LC = globals.votable_to_pandas(file)
        LC = LC[LC['band'] == 'G']  # G band only
        LC = LC[LC['rejected_by_photometry'] == False]
        LC = LC[LC['rejected_by_variability'] == False]

        LC = LC.drop('source_id', axis=1)
        #LC = LC.drop('flux', axis=1)
        #LC = LC.drop('flux_error', axis=1)
        LC = LC.drop('flux_over_error', axis=1)
        LC = LC.drop('transit_id', axis=1)
        LC = LC.drop('rejected_by_photometry', axis=1)
        LC = LC.drop('rejected_by_variability', axis=1)
        LC = LC.drop('other_flags', axis=1)
        LC = LC.drop('solution_id', axis=1)
        LC = LC.drop('band', axis=1)
        print("\n", LC.head())

        LC.sort_values(by=['time'], inplace=True)
        LC['time'] = LC['time'] + 2450000
        LC['mag'] = -1 * LC['mag']

        x = np.array(LC['mag'])
        LC['mag'] = (x - np.mean(x)) / np.max(np.abs(x - np.mean(x)))  # normalize

        LC.index = pd.to_datetime(LC['time'], origin='julian', unit='D')

        fig, axs = plt.subplots(1, 1)
        axs.plot(LC['time'], LC['flux'], 'k.')# raw plot
        axs.set_title(len(LC))

        plt.savefig(folder+'/' + os.path.basename(file)[:-4] + '.png')
        plt.close(fig)

        df = pd.DataFrame({"bjd": LC['time'], "dtr_flux": LC['flux'], "dtr_err": LC['flux_error']})
        # df = df.dropna()
        df.to_csv(folder+'/' + os.path.basename(file)[:-4] + '.txt', sep=' ', index=False, header=False)

    except Exception as e:
        print(f"Error: {e}", file)
