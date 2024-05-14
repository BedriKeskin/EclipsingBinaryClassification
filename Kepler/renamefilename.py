import os

for subdir, dirs, files in os.walk('./'):
    for file in files:
        if file.endswith('.png'):
            if len(file[:file.index("_")][3:]) == 7:
                os.rename(file, "KIC0" + file[3:])
