import star_shadow as sts
import glob
import os


folder = "./Tess/"
LCdatas = glob.glob(folder+"StarShadow/*.txt")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        # if not os.path.exists(folder + "StarShadowAnalysis/" + os.path.basename(LCdata)[:-4] + "_analysis/" + os.path.basename(LCdata)[:-4] + "_analysis_summary.csv"):
        #     print("Analysis not done yet ", os.path.basename(LCdata)[:-4])
        # else:
            if not os.path.exists(folder + "StarShadowAnalysis/" + os.path.basename(LCdata)[:-4] + "_analysis/" + os.path.basename(LCdata)[:-4] + "_eclipse_analysis_derivatives_h.png_ModelOnly.png"):
                sts.ut.plot_all_from_file(LCdata, load_dir=folder+"StarShadowAnalysis", save_dir=folder+"StarShadowAnalysis", show=False)
                print("newly plotted")
            else:
                print("already plotted")
    
    except Exception as e:
        print(f"{e} Error")