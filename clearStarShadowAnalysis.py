import glob
import shutil
import os

folders = glob.glob("./Tess/StarShadowAnalysis/*")
deleted = 0
print(len(folders))

for index, folder in enumerate(folders):
    print("\n", index, folder)

    #if not os.path.exists(folder + "/" + os.path.basename(folder)[:-9] + "_analysis_summary.csv"):
    if not os.path.exists(folder + "/" + os.path.basename(folder)[:-9] + "_eclipse_analysis_derivatives_h.png_ModelOnly.png"):
    #if not os.path.exists(folder + "/" + os.path.basename(folder) + "_9.hdf5"):
    #if not os.path.exists(folder + "/" + os.path.basename(folder) + "_6_ecl_indices.csv"):
        #shutil.rmtree(folder)
        deleted = deleted + 1
        print("deleted")

print("deleted ", deleted)