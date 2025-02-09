import star_shadow as sts
from datetime import datetime
import random
import sys
import os


def run_starshadow_analyse(folderAnalysis, LCdata):
    now1 = datetime.now()
    print("LCdata: ", LCdata, " now: ", now1)

    P = 1/ float(os.path.basename(LCdata).split("_")[5][:-4])
    sts.analyse_lc_from_file(LCdata, p_orb=P, i_sectors=None, stage='all', method='fitter', data_id='none',
                             save_dir=folderAnalysis, overwrite=True, verbose=True)
    now2 = datetime.now()
    print("Length: ", len(LCdata), " Elapsed: ", now2 - now1)


if __name__ == "__main__":
    folderAnalysis = sys.argv[1]
    LCdata = sys.argv[2]
    run_starshadow_analyse(folderAnalysis, LCdata)