# coding=utf-8

import matplotlib.pyplot as plt
import os
import sys
import datetime

sys.stdout = open("plotLC" + ".out", 'w')
now1 = datetime.datetime.now()
print(now1)

counter = 1
cwd = os.getcwd()

files = [os.path.join(cwd, f) for f in os.listdir(cwd) if f.endswith('.txt')]
print("Dosya sayısı: " + str(len(files)))

for file in files:
    print(str(counter) + " " + os.path.basename(file))
    time = []
    flux = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip().split('\t')
            time.append([float(x)])
            flux.append([float(y)])

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(time, flux, "r-")
            plt.axis('off')
            fig.savefig(os.path.splitext(str(file))[0] + ".png", dpi=50, bbox_inches='tight')
            plt.close(fig)

    counter += 1

now2 = datetime.datetime.now()
print(now2)
print(now2 - now1)