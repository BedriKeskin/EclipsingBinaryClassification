import star_shadow as sts
import glob
import os


folder = "./Tess/"
LCdatas = glob.glob(folder+"StarShadow/*.txt")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        #if os.path.exists(folder + "StarShadowAnalysis/" + os.path.basename(LCdata)[:-4] + "_analysis/" + os.path.basename(LCdata)[:-4] + "_9.hdf5"):
            sts.ut.plot_all_from_file(LCdata, load_dir=folder+"StarShadowAnalysis", save_dir=folder+"StarShadowAnalysis", show=False)
    
    except Exception as e:
        print(f"{e} Error")