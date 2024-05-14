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

path = os.path.join(cwd, "Algol_testLC")
try:
    os.mkdir(path)
except:
    print("Klasor zaten var")

arrayPeriod = np.linspace(3.11, 4.09, 2)
arrayIncl = np.linspace(70.9, 80.1, 2)
arraySma = np.linspace(10.9, 20.1, 2)
arrayQ = np.linspace(0.59, 0.81, 3)
arrayTeffP = np.linspace(10990, 13111, 3)
arrayTeffS = np.linspace(5111, 6990, 3)
arrayRequivP = np.linspace(2.51, 2.79, 3)
arrayRequivS = np.linspace(2.81, 3.29, 3)

for period in arrayPeriod:
   for incl in arrayIncl:
       for sma in arraySma:
            for q in arrayQ:
                for teffP in arrayTeffP:
                    for teffS in arrayTeffS:
                        for requivP in arrayRequivP:
                            for requivS in arrayRequivS:
                                print(count)
                                b = phoebe.default_binary()
                                times = np.linspace(0, period, 100)
                                b.add_dataset('lc', times=times)
                                b["period@binary"] = period * u.day  # Orbital period in t
                                b["incl@binary@orbit@component"] = incl * u.deg
                                b["sma@binary@orbit@component"] = sma * u.solRad  # AU
                                b["q@binary@orbit@component"] = q  # mass ratio
                                b["teff@primary"] = teffP * u.K
                                b["teff@secondary"] = teffS * u.K
                                b["requiv@primary@component"] = requivP * u.solRad
                                b["requiv@secondary@component"] = requivS * u.solRad

                                # Albedo
                                b["irrad_frac_refl_bol@primary@component"] = 1.00  # radiative
                                b["irrad_frac_refl_bol@secondary@component"] = 0.5  # convective
                                # Gravity darkening
                                b["gravb_bol@primary@component"] = 1.00  # radiative
                                b["gravb_bol@secondary@component"] = 0.32  # convective
                                # Let's use linear limb darkening law
                                b["ld_mode_bol@primary"] = "manual"
                                b["ld_mode_bol@secondary"] = "manual"

                                try:
                                    b.run_compute()
                                    model = b["fluxes@latest@model"].get_value()
                                    noise = np.random.randn(len(times)) / 300.0
                                    model = model + noise
                                    normflux_model = model / np.average(model)

                                    #Figure çiz ve kaydet
                                    fig, ax = plt.subplots(nrows=1, ncols=1)
                                    ax.plot(times, normflux_model, "r-")
                                    fig.savefig(path + "/algol" + str(count) + ".png")
                                    plt.close(fig)

                                    #Sentetik Light Curve txt dosyası
                                    file2wr = open(path + "/algol" + str(count) + ".txt", "w")
                                    for i, t in enumerate(times):
                                        str2wr = "{:.6f}\t{:.4f}\n".format(t, normflux_model[i])
                                        file2wr.write(str2wr)
                                    file2wr.close()

                                    #Sentetik Light Curve'un oluşturulduğu parametreler
                                    file3wr = open(path + "/algol" + str(count) + ".par", "w")
                                    str3wr = "period: " + str(period) + "\n" + "incl: " + str(incl) + "\n" + "sma: " + str(sma) + "\n" + "q: " + str(q) + "\n" + "teffP: " + str(teffP) + "\n" + "teffS: " + str(teffS) + "\n" + "requivP: " + str(requivP) + "\n" + "requivS: " + str(requivS)
                                    file3wr.write(str3wr)
                                    file3wr.close()
                                except:
                                    print("Şu parametrelerle sentetik Algol Light Curve oluşturulamadı: count:" 
                                          + str(count) + " period:" + str(period) + " incl:" + str(incl) + " sma:" + str(sma) + " q:" + str(q) + " teffP:" + str(teffP) + " teffS:" + str(teffS)) + " requivP: " + str(requivP) + " requivS: " + str(requivS)

                                count += 1

now2 = datetime.datetime.now()
print(now2)
print(now2 - now1)

