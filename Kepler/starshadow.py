import glob
import star_shadow as sts
from datetime import datetime
import random
import os


folder = "StarShadow"
folderAnalysis = folder + "Analysis"
if not os.path.exists(folderAnalysis):
    os.makedirs(folderAnalysis)

LCdatas = glob.glob("./"+folder+"/*.txt")

for index, LCdata in enumerate(LCdatas):
    now1 = datetime.now()
    print("\n", index, LCdata, now1)

    try:
        P = float(os.path.basename(LCdata).split("_")[6][:-4])
        sts.analyse_lc_from_file(LCdata, p_orb=P, i_sectors=None, stage='all', method='fitter', data_id='none',
                                 save_dir=folderAnalysis, overwrite=True, verbose=True)
        now2 = datetime.now()
        print("Length: ", len(LCdata), " Elapsed: ", now2 - now1)

    except Exception as e:
        print(f"{e} Error")
