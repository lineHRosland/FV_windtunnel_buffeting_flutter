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


section_name = "HARD_SINGLE"
section_height = 0.05
section_width =  0.740
section_length = 2.640

#Load all experiments
h5_file = section_name
f = h5py.File((h5_file + ".hdf5"), "r")

data_set_groups = list(f)
   
exp0 = w3t.Experiment.fromWTT(f["HAR_INT_SINGLE_03_04_000"])

exp0.plot_experiment()

exp0.harmonic_groups(plot=True)
