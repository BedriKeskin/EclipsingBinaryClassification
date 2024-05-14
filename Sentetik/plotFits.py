from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fits_file = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/tess2018292075959-s0004-0000000025155310-0124-s_lc.fits'

fitsdata = fits.getdata(fits_file)
dfFits = pd.DataFrame(np.array(fitsdata).byteswap().newbyteorder())
df_filtered = dfFits[dfFits['PDCSAP_FLUX'] != np.nan]  # boş satırları gözardı et
df_filtered1 = dfFits[df_filtered['QUALITY'] == 0]  # kalitesiz noktaları gözardı et

fig, ax = plt.subplots()
ax.plot(df_filtered1['TIME'], df_filtered1['PDCSAP_FLUX'], 'ko')
plt.show()
