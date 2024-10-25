import glob
import star_shadow as sts
from datetime import datetime
import random
import os


if not os.path.exists("StarShadowAnalysis"):
    os.makedirs("StarShadowAnalysis")

LCdatas = glob.glob("./StarShadow/*.txt")

for index, LCdata in enumerate(LCdatas):
    now1 = datetime.now()
    print("\n", index, LCdata, now1)

    try:
        sts.analyse_lc_from_file(LCdata, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='none',
                                 save_dir="StarShadowAnalysis", overwrite=True, verbose=True)
        now2 = datetime.now()
        print("Length: ", len(LCdata), " Elapsed: ", now2 - now1)

    except Exception as e:
        print(f"{e} Error")