# Villanova'nın Kepler ve Tess eclipsing binary star listesini alır, Morph'tan morfolojik Class'ını bulur,
# tabloları alt alta birleştirir, Simbad'ı kullanarak KIC'tan TIC ve Gaia DR3 nosunu veya TIC'tan KIC ve Gaia DR3 nosunu bulur, diske kaydeder.
import pandas as pd
import querySimbad
import datetime
'''
now1 = datetime.datetime.now()
print("Start time: ", now1)


def FindMorphClass(Morph):
    try:
        Morph = float(Morph)
        if Morph <= 0.4:
            return 'Detached'
        elif Morph <= 0.7:
            return 'SemiDetached'
        elif Morph <= 0.8:
            return 'OverContact'
        else:
            return 'Ellipsoidal'
    except ValueError:
        return ''


Villanova_kepler_EB = pd.read_csv('Villanova_kepler_EB.csv', index_col=False, dtype={"KIC": "string"})
KIC = Villanova_kepler_EB["KIC"]
KeplerMorph = Villanova_kepler_EB["morph"]

Villanova_tess_EB = pd.read_csv('Villanova_tess_EB.csv', index_col=False, dtype={"tic__tess_id": "string"})
TIC = Villanova_tess_EB["tic__tess_id"]
TessMorph = Villanova_tess_EB["morph_coeff"]

dataKIC = {'KIC': KIC,
           'KeplerMorph': KeplerMorph,
           'KeplerClass': KeplerMorph.apply(lambda x: FindMorphClass(x)),
           'TIC': "",
           'TessMorph': "",
           'TessClass': "",
           'DR3': ""}

dataTIC = {'KIC': "",
           'KeplerMorph': "",
           'KeplerClass': "",
           'TIC': TIC,
           'TessMorph': TessMorph,
           'TessClass': TessMorph.apply(lambda x: FindMorphClass(x)),
           'DR3': ""}

dfKIC = pd.DataFrame(dataKIC)
dfTIC = pd.DataFrame(dataTIC)

df = pd.concat([dfKIC, pd.DataFrame(dfTIC)], ignore_index=True)

for index, row in df.iterrows():
    print(index)
    if row['KIC'] != "" and row['TIC'] == "" and row['DR3'] == "":
        df.at[index, 'TIC'] = querySimbad.SimbadQueryID('KIC'+row['KIC'], 'TIC')
        df.at[index, 'DR3'] = querySimbad.SimbadQueryID('KIC'+row['KIC'], 'DR3')
    elif row['KIC'] == "" and row['TIC'] != "" and row['DR3'] == "":
        df.at[index, 'KIC'] = querySimbad.SimbadQueryID('TIC'+row['TIC'], 'KIC')
        df.at[index, 'DR3'] = querySimbad.SimbadQueryID('TIC'+row['TIC'], 'DR3')

df.to_csv("MergeKeplerTessGaia.csv", encoding='utf-8', index=False)

now2 = datetime.datetime.now()
print("End time: ", now2)
print("Elapsed time: ", now2 - now1)
'''

MergeKeplerTessGaia = pd.read_csv('MergeKeplerTessGaia.csv', index_col=False,
                                  dtype={"KIC": "string", "TIC": "string", "DR3": "string"})
NonNull = MergeKeplerTessGaia[MergeKeplerTessGaia['DR3'] != ""]  # boş satırları gözardı et
Unique = NonNull.drop_duplicates(subset=["DR3"])
Unique.to_csv(r'MergeKeplerTessGaiaUnique.csv', header=True, index=False, sep=',', mode='w')

