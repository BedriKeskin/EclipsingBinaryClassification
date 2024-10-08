import star_shadow as sts
import pandas as pd
import glob

LCdatas = glob.glob("/Users/bedrikeskin/EclipsingBinaryClassification/Kepler/LCdata/*.csv")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    LC = pd.read_csv(LCdata, delim_whitespace=False, index_col=False)
    LC = LC[:].values
    LC = pd.DataFrame(LC, columns=['index', 'BJD', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err', 'dtr_flux',
                                   'dtr_err', 'blanc'])

    LC = LC.drop('index', axis=1)
    LC = LC.drop('phase', axis=1)
    LC = LC.drop('raw_flux', axis=1)
    LC = LC.drop('raw_err', axis=1)
    LC = LC.drop('corr_flux', axis=1)
    LC = LC.drop('corr_err', axis=1)
    LC = LC.drop('blanc', axis=1)
    LC['BJD'] = LC['BJD'] + 2400000

    LC.to_csv(LCdata + '.txt', sep=' ', index=False, header=False)

    sts.analyse_lc_from_file(LCdata + '.txt', p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='none', save_dir=None, overwrite=False, verbose=True)
