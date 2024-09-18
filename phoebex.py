import phoebe
from phoebe import u  # units
import numpy as np
from astroquery.simbad import Simbad

# logger = phoebe.logger()

# b = phoebe.default_binary()

result_table = Simbad.query_object("KIC 10855535")
result_table.pprint()