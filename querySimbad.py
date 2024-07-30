# https://github.com/BedriKeskin/SimbadQueryID

import numpy as np
from astroquery.simbad import Simbad


def SimbadQueryID(fromID, toID):
    table = Simbad.query_objectids(fromID)
    if table is not None:
        string = table['ID'].astype(str)
        table.replace_column('ID', string)
        data = table['ID'].data
        array = np.flatnonzero(np.core.defchararray.find(data, toID) != -1)
        if len(array) == 1:
            return data[array[0]].split()[-1]
        else:
            return None
    else:
        return None


def SimbadQuery_GaiaID2_KICandTIC(fromID):
    table = Simbad.query_objectids(fromID)
    if table is not None:
        string = table['ID'].astype(str)
        table.replace_column('ID', string)
        data = table['ID'].data
        arrayKIC = np.flatnonzero(np.core.defchararray.find(data, 'KIC') != -1)
        arrayTIC = np.flatnonzero(np.core.defchararray.find(data, 'TIC') != -1)

        KIC = None
        TIC = None

        if len(arrayKIC) == 1:
            KIC = data[arrayKIC[0]].split()[-1]
            if len(KIC) == 7:
                KIC = "KIC0" + KIC
            elif len(KIC) == 8:
                KIC = "KIC" + KIC
            else:
                print(f"Length of KIC {KIC} is other than 7 or 8. Length: {len(KIC)}")

        if len(arrayTIC) == 1:
            TIC = data[arrayTIC[0]].split()[-1]
            if len(TIC) == 7:
                TIC = "TIC 000" + TIC
            if len(TIC) == 8:
                TIC = "TIC 00" + TIC
            elif len(TIC) == 9:
                TIC = "TIC 0" + TIC
            elif len(TIC) == 10:
                TIC = "TIC " + TIC
            else:
                print(f"Length of TIC {TIC} is other than 8 or 9 or 10. Length: {len(TIC)}")

        return KIC, TIC
    else:
        return None, None


'''
kicID = SimbadQuery_GaiaID2_KICandTIC("Gaia DR3 6384729641558823040")
if kicID is not None:
    print(kicID)
else:
    print("not found")
'''
'''
kicID = SimbadQueryID("Gaia DR3 2102813720380137600", 'KIC')
if kicID is not None:
    print(kicID)
else:
    print("not found")
'''
