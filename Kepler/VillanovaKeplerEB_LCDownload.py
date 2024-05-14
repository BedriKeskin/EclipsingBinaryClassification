# Downloads Villanova Kepler Eclipsing Binary Light Curves
# http://keplerebs.villanova.edu/

import datetime
import os
import pandas as pd

now1 = datetime.datetime.now()
print("Start time: ", now1)

if not os.path.exists("LCdata"):
    os.makedirs("LCdata")

Villanova_kepler_EB = pd.read_csv('Villanova_kepler_EB.csv', index_col=False, dtype={"KIC": "string"})
KeplerEBlist = Villanova_kepler_EB['KIC']

for index, EBKICid in enumerate(KeplerEBlist):
    print(index, EBKICid)
    LCdata = None

    try:
        LCdata = pd.read_csv('http://keplerebs.villanova.edu/data/?k=' + EBKICid.strip() + '&cadence=lc&data=data',
                             delim_whitespace=True, index_col=False)

    except Exception as error:
        print(EBKICid, "An exception occurred:", error)

    if LCdata is not None:
        Morph = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == EBKICid, 'morph'].values[0]
        T0 = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == EBKICid, 'bjd0'].values[0]
        P = Villanova_kepler_EB.loc[Villanova_kepler_EB['KIC'] == EBKICid, 'period'].values[0]

        filename = str(EBKICid)
        if len(filename) == 7:  # to make all filenames KICXXXXXXXX 8 digit format
            filename = '0' + filename

        LCdata.to_csv(
            r'LCdata/KIC' + filename + "_morp_" + str(Morph) + "_T0_" + str(T0) + "_P_" + str(P) + ".csv",
            header=True, index=True, sep=',', mode='w')

now2 = datetime.datetime.now()
print("End time: ", now2)
print("Elapsed time: ", now2 - now1)
