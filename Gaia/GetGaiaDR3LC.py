# MergeKeplerTessGaia.csv'deki DR3 IDsi olan yıldızların LCsini indirir
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.io.votable import parse_single_table
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from astroquery.gaia import Gaia

now1 = datetime.datetime.now()
print("Start time: ", now1)

if not os.path.exists("LCdata"):
    os.makedirs("LCdata")
if not os.path.exists("PNG"):
    os.makedirs("PNG")

MergeKeplerTessGaia = pd.read_csv('../EclipsingBinaryClassification/Gaia/MergeKeplerTessGaia.csv', index_col=False,
                                  dtype={"KIC": "string", "TIC": "string", "DR3": "string"})
DR3NotNull = MergeKeplerTessGaia[MergeKeplerTessGaia['DR3'] != ""]  # boş satırları gözardı et
sortedDR3 = DR3NotNull.sort_values('DR3').reset_index(drop=True, inplace=False)

sortedDR3.to_csv(r'MergeKeplerTessGaiaSortedDR3.csv', header=True, index=True, sep=',', mode='w')

for index, row in sortedDR3.iterrows():
    if index > 5444:
        print("\n", index, row['DR3'])

        job = Gaia.launch_job(
            "SELECT source_id, global_ranking, reference_time, frequency FROM gaiadr3.vari_eclipsing_binary where source_id=" +
            row['DR3']
        )
        results = job.get_results()

        if len(results) > 0:
            result = results[0]

            url = 'https://gea.esac.esa.int/data-server/data?ID=Gaia+DR3+' + str(
                result['source_id']) + '&RETRIEVAL_TYPE=EPOCH_PHOTOMETRY&VALID_DATA=true'
            print(url)
            byte = requests.get(url).content

            with open('LCdata/' + str(result['source_id']) + '_reference_time_' + str(result['reference_time']) + '_frequency_' + str(result['frequency']) + '.xml',
                      'wb') as file:
                file.write(byte)
                file.close()

            table = parse_single_table('LCdata/' + str(result['source_id']) + '.xml').to_table()
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

            timeSeries = TimeSeries.from_pandas(LC)

            T0 = result['reference_time'] + 2455197.5
            T0 = pd.to_datetime(T0, origin='julian', unit='D')
            T0 = Time(T0, format='datetime')
            P = 1 / result['frequency']

            ts_folded = timeSeries.fold(period=P * u.day, epoch_time=T0)
            ts_binned = aggregate_downsample(ts_folded, time_bin_size=10 * u.min, aggregate_func=np.nanmedian)

            fig, ax = plt.subplots(1, 1)
            plt.axis('off')
            ax.plot(ts_binned.time_bin_start.jd, ts_binned['flux'], 'r.')
            plt.savefig('PNG/' + str(result['source_id']) + '.png')
            plt.close()

        else:
            print(row['DR3'], "gaiadr3.vari_eclipsing_binary tablosunda bulunmamadı.")

now2 = datetime.datetime.now()
print("End time: ", now2)
print("Elapsed time: ", now2 - now1)
