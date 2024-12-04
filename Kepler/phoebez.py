# In[3]:
import phoebe
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import minimize
from matplotlib.pyplot import savefig
import matplotlib

matplotlib.use('TkAgg')

phoebe.list_passband_online_history("Bolometric:900-40000")
phoebe.list_passband_online_history("Johnson:V")

folder = "StarShadow2"

LCdatas = glob.glob("./"+folder+"/*.txt")

for index, LCdata in enumerate(LCdatas):
    print("\n", index, LCdata)

    try:
        times, fluxes, sigmas_lc = np.loadtxt(LCdata, usecols=(0, 1, 2), unpack=True)

        b = phoebe.default_binary()
        b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas_lc)
        b.set_value_all('ld_mode', 'manual')
        b.set_value_all('ld_mode_bol', 'manual')
        b.set_value_all('atm', 'blackbody')
        b.set_value('pblum_mode', 'dataset-scaled')

        b.run_compute(model='default')
        _ = b.plot(x='phase', show=True)

        # Add Differential Evolution solver to the model
        b.add_solver('estimator.ebai', ebai_method='mlp', solver='ebai_mlp')
        print("1 ", b.filter(solver='ebai_mlp'))

        b['phase_bin@ebai_mlp'] = False
        print("2 ", b.filter(solver='ebai_mlp'))

        b.run_solver('ebai_mlp', solution='ebai_mlp_solution')

        try:
            b.adopt_solution('ebai_mlp_solution')
        except Exception as e:
            print(e)

        b.flip_constraint('requivsumfrac', solve_for='requiv@secondary')
        b.flip_constraint('teffratio', solve_for='teff@secondary')
        b.flip_constraint('esinw', solve_for='ecc')
        b.flip_constraint('ecosw', solve_for='per0')

        # Finally, we can adopt the EBAI solution and see how it has improved the model light curve:

        # In[10]:

        print(b.adopt_solution('ebai_mlp_solution'))

        # In[11]:

        b.run_compute(model='ebai_mlp_model')
        _ = b.plot(x='phase', ls='-', legend=True, show=True)


    except Exception as e:
        print(f"{e} Error")
