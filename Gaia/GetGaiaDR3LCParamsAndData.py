# MergeKeplerTess writes the data of the stars in GaiaUnique.csv from the
# gaiadr3.vari_eclipsing_binary table prepared by Mowlavi (2023) to the
# vari_eclipsing_binary.csv file and downloads the LC data to the LCData folder.

import datetime
import os
import pandas as pd
import requests
from astroquery.gaia import Gaia

now1 = datetime.datetime.now()
print("Start time: ", now1)

columns = ['solution_id', 'source_id', 'global_ranking', 'reference_time', 'frequency', 'frequency_error',
           'geom_model_reference_level', 'geom_model_reference_level_error', 'geom_model_gaussian1_phase',
           'geom_model_gaussian1_phase_error', 'geom_model_gaussian1_sigma', 'geom_model_gaussian1_sigma_error',
           'geom_model_gaussian1_depth', 'geom_model_gaussian1_depth_error', 'geom_model_gaussian2_phase',
           'geom_model_gaussian2_phase_error', 'geom_model_gaussian2_sigma', 'geom_model_gaussian2_sigma_error',
           'geom_model_gaussian2_depth', 'geom_model_gaussian2_depth_error',
           'geom_model_cosine_half_period_amplitude', 'geom_model_cosine_half_period_amplitude_error',
           'geom_model_cosine_half_period_phase', 'geom_model_cosine_half_period_phase_error', 'model_type',
           'num_model_parameters', 'reduced_chi2', 'derived_primary_ecl_phase', 'derived_primary_ecl_phase_error',
           'derived_primary_ecl_duration', 'derived_primary_ecl_duration_error', 'derived_primary_ecl_depth',
           'derived_primary_ecl_depth_error', 'derived_secondary_ecl_phase', 'derived_secondary_ecl_phase_error',
           'derived_secondary_ecl_duration', 'derived_secondary_ecl_duration_error', 'derived_secondary_ecl_depth',
           'derived_secondary_ecl_depth_error'
           ]

if not os.path.exists("LCdata"):
    os.makedirs("LCdata")

MergeKeplerTessGaiaUnique = pd.read_csv('MergeKeplerTessGaiaUnique.csv', index_col=False,
                                        dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

lst = []

for index, row in MergeKeplerTessGaiaUnique.iterrows():
    if 1 == 1:  # index > 5444:
        print("\n", index, row['DR3'])

        job = Gaia.launch_job("SELECT * FROM gaiadr3.vari_eclipsing_binary where source_id=" + row['DR3'])
        results = job.get_results()

        if len(results) > 0:
            result = results[0]
            lst.append(result)

            url = 'https://gea.esac.esa.int/data-server/data?ID=Gaia+DR3+' + str(
                result['source_id']) + '&RETRIEVAL_TYPE=EPOCH_PHOTOMETRY&VALID_DATA=true'
            print(url)
            byte = requests.get(url).content

            with open('LCdata/' + str(result['source_id']) + '_reference_time_' + str(
                    result['reference_time']) + '_frequency_' + str(result['frequency']) + '.xml',
                      'wb') as file:
                file.write(byte)
                file.close()
        else:
            print(row['DR3'], "gaiadr3.vari_eclipsing_binary tablosunda bulunmamadı.")

vari_eclipsing_binary = pd.DataFrame(lst, columns=columns)
vari_eclipsing_binary.to_csv("vari_eclipsing_binary.csv", encoding='utf-8', index=False)

now2 = datetime.datetime.now()
print("End time: ", now2)
print("Elapsed time: ", now2 - now1)
