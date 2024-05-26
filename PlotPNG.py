import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io.votable import parse_single_table
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample

foldername = "PNG-1x"
if not os.path.exists(foldername):
    os.makedirs(foldername)

LCdatas = glob.glob("./LCdata2/*.xml")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    table = parse_single_table(LCdata).to_table()
    LC = pd.DataFrame.from_dict(table, orient='index').transpose()
    LC = LC.astype({'mag': float})
    LC = LC.astype({'flux': float})
    LC = LC.astype({'flux_error': float})
    LC = LC.astype({'flux_over_error': float})
    LC = LC.astype({'rejected_by_variability': bool})
    LC = LC.astype({'rejected_by_photometry': bool})
    LC = LC[LC['time'] != '--']  # time boş satırları gözardı et
    LC = LC[(LC['rejected_by_variability'] == False) | (
            LC['rejected_by_photometry'] == False)]  # rejected olanları gözardı et
    LC = LC[(LC['band'] == 'G')]  # sadece G bandı al

    LC['time'] = LC['time'] + 2455197.5

    LC = LC.drop(['source_id'], axis=1)
    LC = LC.drop(['transit_id'], axis=1)
    LC = LC.drop(['solution_id'], axis=1)
    LC = LC.drop(['other_flags'], axis=1)
    LC = LC.drop(['rejected_by_variability'], axis=1)
    LC = LC.drop(['rejected_by_photometry'], axis=1)
    LC = LC.drop(['band'], axis=1)

    LC.index = pd.to_datetime(LC['time'], origin='julian', unit='D')
    LC.rename(columns={'time': 'bjd'}, inplace=True)

    T0 = float(os.path.basename(LCdata).split("_")[3])
    T0 = T0 + 2455197.5
    T0 = pd.to_datetime(T0, origin='julian', unit='D')
    T0 = Time(T0, format='datetime')

    f = float(os.path.basename(LCdata).split("_")[5][:-4])
    P = 1 / f

    timeSeries = TimeSeries.from_pandas(LC)
    ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)

    ts_binned = aggregate_downsample(ts_folded, time_bin_size=1 * u.min, aggregate_func=np.min)

    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    ax.plot(ts_binned.time_bin_start.jd, ts_binned['flux'], 'k.')
    plt.savefig(foldername + '/' + os.path.basename(LCdata)[:-4] + '.png')
    plt.close()
