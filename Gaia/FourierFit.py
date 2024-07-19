import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from symfit import parameters, variables, sin, cos, Fit
from astropy.io.votable import parse
from symfit.core.minimizers import NelderMead, BFGS
from sklearn import preprocessing

np.set_printoptions(threshold=np.inf)

df = pd.DataFrame(
    columns=['ID', 'T0', 'P', 'a0', 'a1', 'a10', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'w'])

order = 10
folderName = "PNG" + str(order) + "_withoutSine"

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


def votable_to_pandas(votable_file):
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()


LCdatas = glob.glob("./LCdata/*.xml")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = votable_to_pandas(LCdata)

        LC = LC.drop('source_id', axis=1)
        LC = LC.drop('band', axis=1)
        LC = LC.drop('mag', axis=1)
        LC = LC.drop('flux_error', axis=1)
        LC = LC.drop('flux_over_error', axis=1)
        LC = LC.drop('transit_id', axis=1)
        LC = LC.drop('rejected_by_photometry', axis=1)
        LC = LC.drop('rejected_by_variability', axis=1)
        LC = LC.drop('other_flags', axis=1)
        LC = LC.drop('solution_id', axis=1)

        LC['flux'] = preprocessing.normalize([LC['flux']])[0]

        LC['time'] = LC['time'] + 2450000
        LC.index = pd.to_datetime(LC['time'], origin='julian', unit='D')
        LC = LC.drop('time', axis=1)
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(LCdata).split("_")[3])
        T0 = T0 + 2450000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = 1 / float(os.path.basename(LCdata).split("_")[5][:-4])

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        print(ts_folded)
        #ts_binned = aggregate_downsample(ts_folded, time_bin_start=ts_folded['time'], n_bins=len(ts_folded['time']), aggregate_func=np.nanmedian)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.1 * u.min, aggregate_func=np.nanmedian)
        print(ts_binned)

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        ydata = ts_binned['flux']
        print(ydata.info)

        # Define a Fit object for this model and data
        fit = Fit(model_dict, x=xdata, y=ydata, minimizer=[NelderMead, BFGS])
        fit_result = fit.execute()
        # print(fit_result)

        df1 = pd.DataFrame(fit_result.params, index=[0])
        df1.insert(0, 'P', P)
        df1.insert(0, 'T0', float(os.path.basename(LCdata).split("_")[4]))
        df1.insert(0, 'ID', os.path.basename(LCdata).split("_")[0])

        df = pd.concat([df, df1], ignore_index=True)

        # Plot the result
        plt.plot(xdata, ydata, color='black', ls=':')
        plt.plot(xdata, fit.model(x=xdata, **fit_result.params).y, color='red', ls='-')
        plt.savefig(folderName + '/' + os.path.basename(LCdata)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print(f"{e} Error")

df.to_csv('FourierCoeffs_Gaia.csv', index=False)
