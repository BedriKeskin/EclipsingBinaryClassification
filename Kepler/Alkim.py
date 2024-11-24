import glob
import os
import shutil


PNGs = glob.glob("./Alkim./EW/*.png")

for index, png in enumerate(PNGs):
    print("\n", index, png)

    try:
        dosya_adi = os.path.basename(png)
        shutil.copy("./PNG-/"+dosya_adi, './Alkim-/EW/')

    except Exception as e:
        print(f"{e} Error")