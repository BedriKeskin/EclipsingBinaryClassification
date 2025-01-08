import glob
import shutil
import os

folders = glob.glob("StarShadowAnalysis/*")

for index, folder in enumerate(folders):
    print("\n", index, folder)

    if not os.path.exists(folder + "/" + os.path.basename(folder) + "_9.hdf5"):
        #shutil.rmtree(folder)
        print("deleted ", folder)

