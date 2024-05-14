import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample

if not os.path.exists("PNG"):
    os.makedirs("PNG")

LCdatas = glob.glob("./LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)

        # csv'lerin başında # var. Bunu yok etmek için sondaki dtr_err sütununu siliyorum, sadece değerleri alıyorum, tekrar sütun isimlerini ekleyip dataframe çeviriyorum.
        LC = LC.drop(['dtr_err'], axis=1)
        LC = LC[:].values

        if LC.shape[1] == 9:
            LC = pd.DataFrame(LC,
                              columns=['index', 'bjd', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux', 'dtr_err'])
        elif LC.shape[1] == 11:
            LC = pd.DataFrame(LC,
                              columns=['index', 'bjd', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux', 'dtr_err', 'rscl_flux', 'rscl_err'])
        else:
            print('Dataframe\'e çevrilemedi!')

        LC['bjd'] = LC['bjd'] + 2400000
        LC.index = pd.to_datetime(LC['bjd'], origin='julian', unit='D')
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(LCdata).split("_")[4])
        T0 = T0 + 2400000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(LCdata).split("_")[6][:-4])

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=10 * u.min, aggregate_func=np.nanmedian)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(ts_binned.time_bin_start.jd, ts_binned['dtr_flux'], 'k.')
        plt.axis('off')

        plt.savefig('PNG/' + os.path.basename(LCdata)[:-4] + '.png')
        plt.close(fig)

    except Exception as e:
        print(f"{e} Error")
