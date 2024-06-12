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

order = 11
folderName = "PNG" + str(order)

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
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


x, y = variables('x, y')
w, = parameters('w')

model_dict = {y: fourier_series(x, f=w, n=order-1)}
print(model_dict)

# Make step function data
LCdatas = glob.glob("./LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)
        LC = LC[:].values
        LC = pd.DataFrame(LC, columns=['index', 'BJD', 'ColA', 'NormalizedFlux', 'ColC'])

        LC['BJD'] = LC['BJD'] + 2450000
        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')
        timeSeries = TimeSeries.from_pandas(LC)

        T0 = float(os.path.basename(LCdata).split("_")[4])
        T0 = T0 + 2450000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')
        P = float(os.path.basename(LCdata).split("_")[6][:-4])

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=10 * u.min, aggregate_func=np.nanmedian)

        xdata = ts_binned.time_bin_start.jd
        ydata = ts_binned['NormalizedFlux']

        # Define a Fit object for this model and data
        fit = Fit(model_dict, x=xdata, y=ydata)
        fit_result = fit.execute()
        print(fit_result)

        # Plot the result
        plt.plot(xdata, ydata, color='black', ls=':')
        plt.plot(xdata, fit.model(x=xdata, **fit_result.params).y, color='red', ls='-')
        plt.savefig(folderName + '/' + os.path.basename(LCdata)[:-4] + '.png')
        plt.close()

    except Exception as e:
        print(f"{e} Error")
