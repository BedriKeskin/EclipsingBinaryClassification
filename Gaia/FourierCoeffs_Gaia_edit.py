# FourierCoeffs_Gaia.csv'nin sonundaki Villanova ID,morph,Villanova Class sütunlarını doldurmak için yazmıştım bu scripti

import pandas as pd
import querySimbad

Kepler = pd.read_csv("/Users/tiga/Documents/EclipsingBinaryClassification/Kepler/FourierCoeffs_Kepler.csv",
                     delim_whitespace=False, index_col=False)
Tess = pd.read_csv("/Users/tiga/Documents/EclipsingBinaryClassification/Tess/FourierCoeffs_TESS.csv",
                   delim_whitespace=False, index_col=False)
file = '/Users/tiga/Documents/EclipsingBinaryClassification/Gaia/FourierCoeffs_Gaia.csv'
Gaia = pd.read_csv(file, delim_whitespace=False, index_col=False)

for index, gaia in Gaia.iterrows():
    KICandTIC = querySimbad.SimbadQuery_GaiaID2_KICandTIC('Gaia DR3 ' + str(gaia['ID']))
    KIC = KICandTIC[0]
    TIC = KICandTIC[1]
    print(f"index {index} Gaia ID: {gaia['ID']} KIC: {KIC} TIC: {TIC}")

    kepler = pd.DataFrame()
    tess = pd.DataFrame()

    if KIC is not None:
        kepler = Kepler.loc[Kepler['ID'] == KIC]

    if TIC is not None:
        tess = Tess.loc[Tess['ID'] == TIC]

    if not kepler.empty:
        print(f"kepler {kepler}")
        gaia['Villanova ID'] = KIC
        gaia['morph'] = kepler['morph']
        gaia['Villanova Class'] = kepler['label']
        Gaia.loc[index] = gaia
    elif not tess.empty:
        print(f"tess {tess}")
        gaia['Villanova ID'] = TIC
        gaia['morph'] = tess['morph']
        gaia['Villanova Class'] = tess['label']
        Gaia.loc[index] = gaia
    else:
        print(f"Error. Gaia ID {gaia['ID']} KIC {KIC} TIC {TIC}. kepler {kepler} tess {tess}.")

Gaia.to_csv(file, index=False)
