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


kicID = SimbadQueryID("TIC 0120684604", 'KIC')
if kicID is not None:
    print(kicID)
else:
    print("not found")
