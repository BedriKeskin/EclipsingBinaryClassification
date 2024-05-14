import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Range = 1000

x = []
for i in range(Range):
    fi = i / Range
    x.append(fi)


def Gaussian(fi, dk, muk, sigmak):
    return dk * math.exp(-math.pow((fi - muk), 2) / (2 * math.pow(sigmak, 2)))


def Ellipsoidal(fi, Aell, muell):
    return Aell * math.cos(4 * math.pi * (fi - muell))


folder = "PNGMowlaviFit"
if not os.path.exists(folder):
    os.makedirs(folder)

vari_eclipsing_binary = pd.read_csv('../Gaia/vari_eclipsing_binary.csv', index_col=False)
vari_eclipsing_binary = vari_eclipsing_binary.replace('--', np.nan)  # --'leri NaN'a çevir

astype = {"source_id": str, 'global_ranking': float, 'reference_time': float,
          'frequency': float, 'geom_model_reference_level': float,
          'geom_model_gaussian1_phase': float, 'geom_model_gaussian1_sigma': float, 'geom_model_gaussian1_depth': float,
          'geom_model_gaussian2_phase': float, 'geom_model_gaussian2_sigma': float, 'geom_model_gaussian2_depth': float,
          'geom_model_cosine_half_period_amplitude': float, 'geom_model_cosine_half_period_phase': float,
          'model_type': str, 'num_model_parameters': int}

vari_eclipsing_binary = vari_eclipsing_binary.astype(astype)

for index, row in vari_eclipsing_binary.iterrows():
    print("\n", index, row['source_id'])

    C = row['geom_model_reference_level']
    dk1 = row['geom_model_gaussian1_depth']
    muk1 = row['geom_model_gaussian1_phase']
    sigmak1 = row['geom_model_gaussian1_sigma']
    dk2 = row['geom_model_gaussian2_depth']
    muk2 = row['geom_model_gaussian2_phase']
    sigmak2 = row['geom_model_gaussian2_sigma']
    Aell = row['geom_model_cosine_half_period_amplitude']
    muell = row['geom_model_cosine_half_period_phase']

    y = []

    if row['model_type'] == 'TWOGAUSSIANS':  # and index == 346464564620:
        for i in range(Range):
            fi = i / Range
            y.append(C + Gaussian(fi, dk1, muk1, sigmak1) + Gaussian(fi, dk2, muk2, sigmak2))

    elif row['model_type'] == 'TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1' or row['model_type'] == 'TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE2':
        for i in range(Range):
            fi = i / Range
            y.append(C + Gaussian(fi, dk1, muk1, sigmak1) + Gaussian(fi, dk2, muk2, sigmak2) + Ellipsoidal(fi, Aell, muell))

    elif row['model_type'] == 'ONEGAUSSIAN':
        for i in range(Range):
            fi = i / Range
            y.append(C + Gaussian(fi, dk1, muk1, sigmak1))

    elif row['model_type'] == 'ONEGAUSSIAN_WITH_ELLIPSOIDAL':
        for i in range(Range):
            fi = i / Range
            y.append(C + Gaussian(fi, dk1, muk1, sigmak1) + Ellipsoidal(fi, Aell, muell))

    elif row['model_type'] == 'ELLIPSOIDAL':
        for i in range(Range):
            fi = i / Range
            y.append(C + Ellipsoidal(fi, Aell, muell))

    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    y = [i * -1 for i in y]
    ax.plot(x, y, 'k-')
    plt.savefig(folder + '/' + row['source_id'] + '.png')
    plt.close()
