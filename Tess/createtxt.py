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
import statsmodels.api as sm
from scipy.signal import medfilt
from sklearn.ensemble import IsolationForest

folder = "StarShadow"
if not os.path.exists(folder):
    os.makedirs(folder)

LCdatas = glob.glob("./LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)
        LC = LC[:].values
        LC = pd.DataFrame(LC, columns=['index', 'BJD', 'BJD_err', 'NormalizedFlux', 'NormalizedFlux_err'])

        LC = LC.drop('index', axis=1)
        LC = LC.drop('BJD_err', axis=1)

        LC['BJD'] = LC['BJD'] + 2450000
        #LC = LC.astype({'BJD': str, 'NormalizedFlux': str, 'NormalizedFlux_err': str})

        T0 = float(os.path.basename(LCdata).split("_")[4])
        T0 = T0 + 2400000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(LCdata).split("_")[6][:-4])

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(LC['BJD'], LC['NormalizedFlux'], 'k.')  # raw plot
        axs[0].set_title(len(LC))

        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')
        timeSeries = TimeSeries.from_pandas(LC)

        if len(LC) < 10000:
            ts_binned = aggregate_downsample(timeSeries, time_bin_size=40 * u.min, aggregate_func=np.nanmedian)
        else:
            ts_binned = aggregate_downsample(timeSeries, n_bins=10000, aggregate_func=np.nanmedian)

        ts_binned = ts_binned.to_pandas().dropna()
        axs[1].plot(ts_binned['BJD'], ts_binned['NormalizedFlux'], 'k.')  # binned plot
        axs[1].set_title(len(ts_binned))

        plt.savefig(folder + '/' + os.path.basename(LCdata)[:-4] + '.png')
        plt.close(fig)

        df = pd.DataFrame({"BJD": ts_binned['BJD'], "NormalizedFlux": ts_binned['NormalizedFlux'], "NormalizedFlux_err": ts_binned['NormalizedFlux_err']})
        df = df.dropna()
        df.to_csv(folder + '/' + os.path.basename(LCdata)[:-4] + '.txt', sep=' ', index=False, header=False)
    except Exception as e:
        print(f"Error: {e}")