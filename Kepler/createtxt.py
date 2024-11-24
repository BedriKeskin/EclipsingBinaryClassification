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
#from sklearn.ensemble import IsolationForest


folder = "StarShadow2"
if not os.path.exists(folder):
    os.makedirs(folder)

LCdatas = glob.glob("./LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)

        T0 = float(os.path.basename(LCdata).split("_")[4])
        T0 = T0 + 2400000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(LCdata).split("_")[6][:-4])

        # csv'lerin başında # var. Bunu yok etmek için sondaki dtr_err sütununu siliyorum, sadece değerleri alıyorum, tekrar sütun isimlerini ekleyip dataframe çeviriyorum.
        LC = LC.drop(['dtr_err'], axis=1)
        LC = LC[:].values

        if LC.shape[1] == 9:
            LC = pd.DataFrame(LC,
                              columns=['index', 'bjd', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err',
                                       'dtr_flux', 'dtr_err'])
        elif LC.shape[1] == 11:
            LC = pd.DataFrame(LC,
                              columns=['index', 'bjd', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err',
                                       'dtr_flux', 'dtr_err', 'rscl_flux', 'rscl_err'])
            LC = LC.drop(['rscl_flux'], axis=1)
            LC = LC.drop(['rscl_err'], axis=1)
        else:
            print('Dataframe\'e çevrilemedi!')

        LC = LC.drop(['index'], axis=1)
        LC = LC.drop(['phase'], axis=1)
        LC = LC.drop(['raw_flux'], axis=1)
        LC = LC.drop(['raw_err'], axis=1)
        LC = LC.drop(['corr_flux'], axis=1)
        LC = LC.drop(['corr_err'], axis=1)

        LC['bjd'] = LC['bjd'] + 2400000

        fig, axs = plt.subplots(1, 1)
        axs.plot(LC['bjd'], LC['dtr_flux'], 'k.')# raw plot
        axs.set_title(len(LC))

        # LC.index = pd.to_datetime(LC['bjd'], origin='julian', unit='D')
        # timeSeries = TimeSeries.from_pandas(LC)
        #
        # if len(LC) < 10000:
        #     ts_binned = aggregate_downsample(timeSeries, time_bin_size=40 * u.min, aggregate_func=np.nanmedian)
        # else:
        #     ts_binned = aggregate_downsample(timeSeries, n_bins=10000, aggregate_func=np.nanmedian)
        #
        # ts_binned = ts_binned.to_pandas().dropna()
        # axs[1].plot(ts_binned['bjd'], ts_binned['dtr_flux'], 'k.')# binned plot
        # axs[1].set_title(len(ts_binned))

        plt.savefig(folder+'/' + os.path.basename(LCdata)[:-4] + '.png')
        plt.close(fig)

        df = pd.DataFrame({"bjd": LC['bjd'], "dtr_flux": LC['dtr_flux'], "dtr_err": LC['dtr_err']})
        # df = df.dropna()
        df.to_csv(folder+'/' + os.path.basename(LCdata)[:-4] + '.txt', sep=' ', index=False, header=False)
    except Exception as e:
        print(f"Error: {e}")