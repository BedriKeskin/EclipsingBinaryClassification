# coding=utf-8

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

now1 = datetime.datetime.now()
print(now1)

count = 1

cwd = os.getcwd()

path = os.path.join(cwd, "WUMa_testLC")
try:
    os.mkdir(path)
except:
    print("Klasor zaten var")

arrayPeriod = np.linspace(0.41, 0.59, 3)
arrayIncl = np.linspace(70.11, 80.22, 3)
arraySma = np.linspace(3.31, 3.91, 4)
arrayQ = np.linspace(0.72, 0.91, 4)
arrayTeffP = np.linspace(7511, 7951, 4)
arrayTeffS = np.linspace(7351, 7451, 4)

for period in arrayPeriod:
    for incl in arrayIncl:
        for sma in arraySma:
            for q in arrayQ:
                for teffP in arrayTeffP:
                    for teffS in arrayTeffS:
                        print(count)
                        b = phoebe.default_binary(contact_binary=True)
                        times = np.linspace(0, period, 100)
                        b.add_dataset('lc', times=times)
                        b["period@binary"] = period * u.day  # Orbital period in t
                        b["incl@binary@orbit@component"] = incl * u.deg
                        b["sma@binary@orbit@component"] = sma * u.solRad  # AU
                        b["q@binary@orbit@component"] = q  # mass ratio
                        b["teff@primary"] = teffP * u.K
                        b["teff@secondary"] = teffS * u.K
                        
                        try:
                            b.run_compute()
                            model = b["fluxes@latest@model"].get_value()
                            noise = np.random.randn(len(times)) / 50.0
                            model = model + noise
                            normflux_model = model / np.average(model)
        
                            #Figure çiz ve kaydet
                            fig, ax = plt.subplots(nrows=1, ncols=1)
                            ax.plot(times, normflux_model, "r-")
                            fig.savefig(path + "/betalyrae" + str(count) + ".png")
                            plt.close(fig)
        
                            #Sentetik Light Curve txt dosyası
                            file2wr = open(path + "/wuma" + str(count) + ".txt", "w")
                            for i, t in enumerate(times):
                                str2wr = "{:.6f}\t{:.4f}\n".format(t, normflux_model[i])
                                file2wr.write(str2wr)
                            file2wr.close()
        
                            #Sentetik Light Curve'un oluşturulduğu parametreler
                            file3wr = open(path + "/wuma" + str(count) + ".par", "w")
                            str3wr = "period: " + str(period) + "\n" + "incl: " + str(incl) + "\n" + "sma: " + str(sma) + "\n" + "q: " + str(q) + "\n" + "teffP: " + str(teffP) + "\n" + "teffS: " + str(teffS)
                            file3wr.write(str3wr)
                            file3wr.close()
                        except:
                            print("Şu parametrelerle sentetik W Uma Light Curve oluşturulamadı: count:" 
                                  + str(count) + " period:" + str(period) + " incl:" + str(incl) + " sma:" + str(sma) + " q:" + str(q) + " teffP:" + str(teffP) + " teffS:" + str(teffS))
        
                        count += 1

now2 = datetime.datetime.now()
print(now2)
print(now2 - now1)

