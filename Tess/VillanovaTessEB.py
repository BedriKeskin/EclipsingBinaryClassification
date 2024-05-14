# Villanova TESS'den LC verisini indirir ve plot eder, plotu diske kaydeder.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample

Villanova_tess_EB = pd.read_csv('Villanova_tess_EB.csv', index_col=False, dtype={"tic__tess_id": "string"})
# Villanova_tess_EB['bjd0'] = Villanova_tess_EB['bjd0'].astype(float)
Villanova_tess_EB = Villanova_tess_EB.replace(r'^\s*$', np.nan, regex=True)  # boşlukları NaN'a çevir
Villanova_tess_EB = Villanova_tess_EB.astype({'period': float, 'bjd0': float, 'morph_coeff': float})
# Villanova_tess_EB = Villanova_tess_EB.astype({'bjd0': float})
Villanova_tess_EB = Villanova_tess_EB[~np.isnan(Villanova_tess_EB['morph_coeff'])]  # Morfoloji boş satırları gözardı et
Villanova_tess_EB = Villanova_tess_EB[~np.isnan(Villanova_tess_EB['period'])]  # period boş satırları gözardı et
Villanova_tess_EB = Villanova_tess_EB[~np.isnan(Villanova_tess_EB['bjd0'])]  # T0 boş satırları gözardı et
# Villanova_tess_EB = Villanova_tess_EB.sample(5)
# Villanova_tess_EB = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == ' 0455206965']
TessEBlist = Villanova_tess_EB['tic__tess_id']

counter = 0

for EBTICid in TessEBlist:
    counter += 1
    print(counter, EBTICid)

    try:
        LC = pd.read_csv('http://tessebs.villanova.edu/static/catalog/lc_data/tic' + EBTICid.strip() + '.01.norm.lc',
                         delim_whitespace=True, index_col=False)
        LC.columns = ['BJD', 'ColA', 'NormalizedFlux', 'ColC']
        LC = LC.drop(['ColA', 'ColC'], axis=1)

        LC['BJD'] = LC['BJD'] + 2450000
        LC.index = pd.to_datetime(LC['BJD'], origin='julian', unit='D')
        timeSeries = TimeSeries.from_pandas(LC)

        # periodogram = BoxLeastSquares.from_timeseries(timeSeries, 'NormalizedFlux')
        # results = periodogram.autopower()
        # best = np.argmax(results.power)
        # period = results.period[best]
        # print(period)
        # exit()

        Morph = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == EBTICid, 'morph_coeff'].values[0]
        P = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == EBTICid, 'period'].values[0]
        T0 = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == EBTICid, 'bjd0'].values[0]
        T0 = T0 + 2450000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=10 * u.min, aggregate_func=np.nanmedian)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(ts_folded.time.jd, ts_folded['NormalizedFlux'], 'k.')
        ax.plot(ts_binned.time_bin_start.jd, ts_binned['NormalizedFlux'], 'r-')
        plt.axis('off')

        if Morph <= 0.4:
            # timeSeries.write('Detached/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('Detached/TIC' + EBTICid.strip() + "_morp" + str(Morph) + ".png", dpi=50, bbox_inches='tight')
        elif Morph <= 0.7:
            # timeSeries.write('SemiDetached/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('SemiDetached/TIC' + EBTICid.strip() + "_morp" + str(Morph) + ".png", dpi=50,
                        bbox_inches='tight')
        elif Morph <= 0.8:
            # timeSeries.write('OverContact/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('OverContact/TIC' + EBTICid.strip() + "_morp" + str(Morph) + ".png", dpi=50,
                        bbox_inches='tight')
        else:
            # timeSeries.write('Ellipsoidal/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('Ellipsoidal/TIC' + EBTICid.strip() + "_morp" + str(Morph) + ".png", dpi=50,
                        bbox_inches='tight')

        plt.close(fig)

    except:
        print("LC verisi sunucudan getirilemedi")
