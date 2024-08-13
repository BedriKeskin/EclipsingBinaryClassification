import glob
import os
import shutil
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from symfit import parameters, variables, sin, cos, Fit
from symfit.core.minimizers import NelderMead, BFGS
# from sklearn import preprocessing
import data

# np.set_printoptions(threshold=np.inf)

df = pd.DataFrame(
    columns=['ID', 'T0', 'P', 'a0', 'a1', 'a10', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'w'])

order = 10
phaseFold = 3
folderName = "PNG_" + str(phaseFold) + "xP"

#shutil.rmtree(folderName)

if not os.path.exists(folderName):
    os.makedirs(folderName)


def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # withSine
    # series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x) for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    # withoutSine
    series = a0 + sum(ai * cos(i * f * x) for i, (ai) in enumerate(cos_a, start=1))
    return series


x, y = variables('x, y')
w, = parameters('w')

model_dict = {y: fourier_series(x, f=w, n=order)}
print(model_dict)

LCdatas = glob.glob("./LCdata/*.xml")

for index, LCdata in enumerate(LCdatas[:100]):  # (sample(LCdatas, 100)):
    print("\n", index, LCdata)

    try:
        LC = data.votable_to_pandas(LCdata)
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
        # LC['mag'] = preprocessing.normalize([LC['mag']])[0]

        LC.index = pd.to_datetime(LC['time'], origin='julian', unit='D')
        LC = LC.drop('time', axis=1)
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(LCdata).split("_")[3])
        T0 = T0 + 2450000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = 1 / float(os.path.basename(LCdata).split("_")[5][:-4])

        ts_folded = timeSeries.fold(period=3*P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=1 * u.min, aggregate_func=np.nanmedian)
        # non_n_bad_indexes = ~np.isnan(ts_binned.to_pandas().to_numpy()[:, 1])
        # ts_binned = ts_binned[non_n_bad_indexes]
        print(f"ts_binned : {len(ts_binned)}")

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        # ydata = ts_binned['flux']
        ydata = ts_binned['mag']

        # Define a Fit object for this model and data
        fit = Fit(model_dict, x=xdata, y=ydata, minimizer=[NelderMead, BFGS])
        fit_result = fit.execute()
        # print(fit_result)

        df1 = pd.DataFrame(fit_result.params, index=[0])
        df1.insert(0, 'P', P)
        df1.insert(0, 'T0', float(os.path.basename(LCdata).split("_")[3]))
        df1.insert(0, 'ID', os.path.basename(LCdata).split("_")[0])

        df = pd.concat([df, df1], ignore_index=True)

        # Plot the result
        plt.plot(xdata, ydata, color='black', marker='.', ls='')
        plt.plot(xdata, fit.model(x=xdata, **fit_result.params).y, color='red', ls='-')
        #plt.show(block=False)
        plt.savefig(folderName + '/' + os.path.basename(LCdata)[:-4] + '.png')
        #plt.pause(0.5)
        plt.close('all')

    except Exception as e:
        print(f"Error: {e}")

df.to_csv('FourierCoeffs_Gaia_'+str(phaseFold)+'xP.csv', index=False)

