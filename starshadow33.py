import glob
import star_shadow as sts
from datetime import datetime
import random

file = "/Users/bedrikeskin/EclipsingBinaryClassification/Kepler/LCdata/KIC02697935_morp_-1.0_T0_55008.46944_P_21.5133595.csv.txt"

sts.analyse_lc_from_file(file, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='none',
                         save_dir=None, overwrite=True, verbose=True)

xxxx

LCdatas = glob.glob("*.txt")

for index, LCdata in enumerate(random.sample(LCdatas, 1)):
    now1 = datetime.now()
    print("\n", index, LCdata, now1)

    try:
        sts.analyse_lc_from_file(LCdata, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='none',
                                 save_dir=None, overwrite=True, verbose=True)
        now2 = datetime.now()
        print("Elapsed: ", now2 - now1)

    except Exception as e:
        print(f"{e} Error")