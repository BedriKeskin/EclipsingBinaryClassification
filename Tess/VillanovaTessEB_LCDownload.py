# Downloads Villanova TESS Eclipsing Binary Light Curves
# http://tessebs.villanova.edu/

import os
import pandas as pd
import datetime

now1 = datetime.datetime.now()
print("Start time: ", now1)

if not os.path.exists("LCdata"):
    os.makedirs("LCdata")

Villanova_tess_EB = pd.read_csv('Villanova_tess_EB.csv', index_col=False, dtype={"tic__tess_id": "string"})
TessEBlist = Villanova_tess_EB['tic__tess_id']

for index, EBTICid in enumerate(TessEBlist):
    print(index, EBTICid)
    LCdata = None

    try:
        LCdata = pd.read_csv(
            'http://tessebs.villanova.edu/static/catalog/lc_data/tic' + EBTICid.strip() + '.01.norm.lc',
            delim_whitespace=True, index_col=False)

    except:
        print(EBTICid, " LC data could not be fetched from server")

    if LCdata is not None:
        Morph = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == EBTICid, 'morph_coeff'].values[0]
        T0 = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == EBTICid, 'bjd0'].values[0]
        P = Villanova_tess_EB.loc[Villanova_tess_EB['tic__tess_id'] == EBTICid, 'period'].values[0]

        LCdata.to_csv(
            r'LCdata/TIC' + str(EBTICid) + "_morp_" + str(Morph) + "_T0_" + str(T0) + "_P_" + str(P) + ".csv",
            header=True, index=True, sep=',', mode='w')

now2 = datetime.datetime.now()
print("End time: ", now2)
print("Elapsed time: ", now2 - now1)
