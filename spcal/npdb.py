from importlib.resources import open_binary

import numpy as np

db = np.load(open_binary("spcal.resources", "npdb.npz"), allow_pickle=False)
