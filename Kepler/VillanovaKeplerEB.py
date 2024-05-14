import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample

Villanova_kepler_EB = pd.read_csv('Villanova_kepler_EB.csv', index_col=False, dtype={"KIC": "string"})
# Villanova_kepler_EB['bjd0'] = Villanova_kepler_EB['bjd0'].astype(float)
Villanova_kepler_EB = Villanova_kepler_EB.replace(r'^\s*$', np.nan, regex=True)  # boşlukları NaN'a çevir
Villanova_kepler_EB = Villanova_kepler_EB.astype({'period': float, 'bjd0': float, 'morph': float})
# Villanova_kepler_EB = Villanova_kepler_EB.astype({'bjd0': float})
Villanova_kepler_EB = Villanova_kepler_EB[~np.isnan(Villanova_kepler_EB['morph'])]  # Morfoloji boş satırları gözardı et
Villanova_kepler_EB = Villanova_kepler_EB[~np.isnan(Villanova_kepler_EB['period'])]  # period boş satırları gözardı et
Villanova_kepler_EB = Villanova_kepler_EB[~np.isnan(Villanova_kepler_EB['bjd0'])]  # T0 boş satırları gözardı et
# Villanova_kepler_EB = Villanova_kepler_EB.sample(5)
# Villanova_kepler_EB = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == ' 0455206965']
KeplerEBlist = Villanova_kepler_EB['KIC']

counter = 0

for EBKICid in KeplerEBlist:
    counter += 1
    print(counter, EBKICid)

    try:
        LC = pd.read_csv('http://keplerebs.villanova.edu/data/?k=' + EBKICid.strip() + '&cadence=lc&data=data',
                         delim_whitespace=True, index_col=False)

        # csv'lerin başında # var. Bunu yok etmek için sondaki dtr_err sütununu siliyorum, sadece değerleri alıyorum, tekrar sütun isimlerini ekleyip dataframe çeviriyorum.
        LC = LC.drop(['dtr_err'], axis=1)
        LC = LC[:].values
        LC = pd.DataFrame(LC, columns=['bjd', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux',
                                       'dtr_err'])

        LC['bjd'] = LC['bjd'] + 2400000
        LC.index = pd.to_datetime(LC['bjd'], origin='julian', unit='D')
        timeSeries = TimeSeries.from_pandas(LC)

        # periodogram = BoxLeastSquares.from_timeseries(timeSeries, 'NormalizedFlux')
        # results = periodogram.autopower()
        # best = np.argmax(results.power)
        # period = results.period[best]
        # print(period)
        # exit()

        Morph = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == EBKICid, 'morph'].values[0]
        P = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == EBKICid, 'period'].values[0]
        T0 = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == EBKICid, 'bjd0'].values[0]

        T0 = T0 + 2400000
        T0 = pd.to_datetime(T0, origin='julian', unit='D')
        T0 = Time(T0, format='datetime')

        ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=10 * u.min, aggregate_func=np.nanmedian)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(ts_folded.time.jd, ts_folded['NormalizedFlux'], 'k.')
        ax.plot(ts_binned.time_bin_start.jd, ts_binned['dtr_flux'], 'r-')
        plt.axis('off')

        if len(EBKICid) == 7:  # idsi 7 hane olanların başına 0 ekelyerek bütün idleri 8 hane yapmak lazım
            EBKICid = "0" + EBKICid

        if Morph <= 0.4:
            # timeSeries.write('Detached/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('Detached2/KIC' + EBKICid.strip() + "_morp" + str(Morph) + ".png", dpi=50, bbox_inches='tight')
        elif Morph <= 0.7:
            # timeSeries.write('SemiDetached/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('SemiDetached2/KIC' + EBKICid.strip() + "_morp" + str(Morph) + ".png", dpi=50,
                        bbox_inches='tight')
        elif Morph <= 0.8:
            # timeSeries.write('OverContact/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('OverContact2/KIC' + EBKICid.strip() + "_morp" + str(Morph) + ".png", dpi=50,
                        bbox_inches='tight')
        else:
            # timeSeries.write('Ellipsoidal/TIC' + EBTICid.strip() + "_morp" + str(Morph) + '.csv', delimiter='\t', overwrite=True)
            fig.savefig('Ellipsoidal2/KIC' + EBKICid.strip() + "_morp" + str(Morph) + ".png", dpi=50,
                        bbox_inches='tight')

        plt.close(fig)

    except:
        print("LC verisi sunucudan getirilemedi")
