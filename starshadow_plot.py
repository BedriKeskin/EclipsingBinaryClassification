import star_shadow as sts
import pandas as pd
import glob
import os


LCdatas = glob.glob("/Users/bedrikeskin/EclipsingBinaryClassification/Tess/StarShadow/*.txt")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        sts.ut.plot_all_from_file(LCdata, load_dir="/Users/bedrikeskin/EclipsingBinaryClassification/Tess/StarShadowAnalysis", save_dir="", show=False)
    
    except Exception as e:
        print(f"{e} Error")