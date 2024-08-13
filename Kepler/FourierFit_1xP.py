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

df = pd.DataFrame(
    columns=['ID', 'morph', 'T0', 'P', 'a0', 'a1', 'a10', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'w', 'label'])

order = 10
folderName = "PNG" + str(order) + "_withoutSine_BJD"

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

LCdatas = glob.glob("./LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)
        LC = LC[:].values
        LC = pd.DataFrame(LC,
                          columns=['index', 'BJD', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux',
                                   'dtr_err', 'blanc'])
        LC = LC.drop('index', axis=1)
        LC = LC.drop('phase', axis=1)
        LC = LC.drop('raw_err', axis=1)

        LC['BJD'] = LC['BJD'] + 2400000
        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')
        LC = LC.drop('BJD', axis=1)
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(LCdata).split("_")[4])
        T0 = T0 + 2400000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(LCdata).split("_")[6][:-4])

        x = (max(timeSeries['time']) - min(timeSeries['time']))
        print(x.dt.days)
        P = P * x.astype(int)

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=10 * u.min, aggregate_func=np.nanmedian)
        # print(ts_binned)

        xdata = ts_binned.time_bin_start.jd
        xdata = xdata / (-xdata[0] * 2)
        print(xdata)
        ydata = ts_binned['dtr_flux']
        print(ydata)

        # Define a Fit object for this model and data
        fit = Fit(model_dict, x=xdata, y=ydata)
        fit_result = fit.execute()
        # print(fit_result)

        df1 = pd.DataFrame(fit_result.params, index=[0])
        df1.insert(0, 'P', P)
        df1.insert(0, 'T0', float(os.path.basename(LCdata).split("_")[4]))
        morph = float(os.path.basename(LCdata).split("_")[2])
        df1.insert(0, 'morph', morph)
        df1.insert(0, 'ID', os.path.basename(LCdata).split("_")[0])

        label = ""

        if morph <= 0.4:
            label = "Detached"
        elif morph <= 0.7:
            label = "SemiDetached"
        elif morph <= 0.8:
            label = "OverContact"
        else:
            label = "Ellipsoidal"

        df1['label'] = label

        df = pd.concat([df, df1], ignore_index=True)

        # Plot the result
        plt.plot(xdata, ydata, color='black', ls=':')
        plt.plot(xdata, fit.model(x=xdata, **fit_result.params).y, color='red', ls='-')
        plt.savefig(folderName + '/' + label + '/' + os.path.basename(LCdata)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print(f"Error: {e}")

df.to_csv('FourierCoeffs_Kepler.csv', index=False)
