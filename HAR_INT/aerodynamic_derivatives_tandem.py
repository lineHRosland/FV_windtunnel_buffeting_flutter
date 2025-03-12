#%%
import numpy as np
import sys
sys.path.append(r"C:\Users\oiseth\Documents\GitHub\w3tp")
import w3t
#import os
import h5py
from matplotlib import pyplot as plt
import time

import pandas as pd

 
tic = time.perf_counter()
plt.close("all")


section_name = "Tandem"
section_height = 0.05
section_width =  0.740
section_length = 2.640

#Load all experiments
h5_file = section_name
f = h5py.File((h5_file + ".hdf5"), "r")
list(f)

#%%
all_ads_gap2D = w3t.AerodynamicDerivatives()
exp0 = w3t.Experiment.fromWTT(f["HAR_INT_MDS_GAP_213D_03_01_01_000"]) #Still-air
exp1 = w3t.Experiment.fromWTT(f["HAR_INT_MDS_GAP_213D_03_01_01_001"]) # In wind
exp0.plot_experiment()
exp1.plot_experiment()
exp0.filt_forces(6,5)
exp1.filt_forces(6,5)
ads, val, expf = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
fig, _ = plt.subplots(4,2,sharex=True)
expf.plot_experiment(fig=fig)
val.plot_experiment(fig=fig)
ads.plot()
all_ads_gap2D.append(ads)

#%%
exp0 = w3t.Experiment.fromWTT(f["HAR_INT_MDS_GAP_213D_03_01_03_000"]) #Still-air
exp1 = w3t.Experiment.fromWTT(f["HAR_INT_MDS_GAP_213D_03_01_03_001"]) # In wind
exp0.plot_experiment()
exp1.plot_experiment()
exp0.filt_forces(6,5)
exp1.filt_forces(6,5)
ads, val, expf = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
fig, _ = plt.subplots(4,2,sharex=True)
expf.plot_experiment(fig=fig)
val.plot_experiment(fig=fig)
ads.plot()
all_ads_gap2D.append(ads)
all_ads_gap2D.plot(mode="decks")
plt.show()