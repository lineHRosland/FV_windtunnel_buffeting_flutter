#%%
import numpy as np
import sys
sys.path.append(r"C:\Users\alasm\Masteroppgave\w3tp")
import w3t
#import os
import h5py
from matplotlib import pyplot as plt
import time

import pandas as pd

 
tic = time.perf_counter()
plt.close("all")


section_name = "HARD_SINGLE"
section_height = 0.0667
section_width =  0.365
section_length = 2.68

#Load all experiments
h5_file = section_name
f = h5py.File((h5_file + ".hdf5"), "r")

data_set_groups = list(f)
exps = np.array([])
for data_set_group in data_set_groups:
    exps = np.append(exps,w3t.Experiment.fromWTT(f[data_set_group]))
tests_with_equal_motion = w3t.group_motions(exps)

#%% Obtain aerodynamic derivatives

ads_list = []
val_list = []
expf_list = []

fig_damping, _ = plt.subplots(3,3)
fig_stiffness, _ = plt.subplots(3,3)



all_ads = w3t.AerodynamicDerivatives()
for k1 in [1,2]:
    for k2 in range(len(tests_with_equal_motion[k1])-1):
        exp0 = exps[tests_with_equal_motion[k1][0]]
        exp1 = exps[tests_with_equal_motion[k1][k2+1]]
        exp0.filt_forces(6,5)
        exp1.filt_forces(6,5)
        
        ads, val, expf = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
        ads_list.append(ads)
        val_list.append(val)
        expf_list.append(expf)
        all_ads.append(ads)   
        
        # plot measurements and predictions by ads
        fig, _ = plt.subplots(4,2,sharex=True)
        expf.plot_experiment(fig=fig)
        val.plot_experiment(fig=fig)
all_ads.plot(fig_damping = fig_damping, fig_stiffness=fig_stiffness)

#%%
all_ads = w3t.AerodynamicDerivatives()
for k1 in [3,4]:
    print(k1)
    for k2 in range(len(tests_with_equal_motion[k1])-1):
        exp0 = exps[tests_with_equal_motion[k1][0]]
        exp1 = exps[tests_with_equal_motion[k1][k2+1]]
        exp0.filt_forces(6,5)
        exp1.filt_forces(6,5)

        print("Experiment " + str(tests_with_equal_motion[k1][0]))
        print(exp0.motion_type())
        
        ads, val, expf = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
        ads_list.append(ads)
        val_list.append(val)
        expf_list.append(expf)
        all_ads.append(ads)   
        
        # plot measurements and predictions by ads
        fig, _ = plt.subplots(4,2,sharex=True)
        expf.plot_experiment(fig=fig)
        val.plot_experiment(fig=fig)
all_ads.plot(fig_damping = fig_damping, fig_stiffness=fig_stiffness)
#%%
all_ads = w3t.AerodynamicDerivatives()
for k1 in [5,6]:
    print(k1)
    for k2 in range(len(tests_with_equal_motion[k1])-1):
        exp0 = exps[tests_with_equal_motion[k1][0]]
        exp1 = exps[tests_with_equal_motion[k1][k2+1]]
        exp0.filt_forces(6,5)
        exp1.filt_forces(6,5)

        print("Experiment " + str(tests_with_equal_motion[k1][0]))
        print(exp0.motion_type())
        
        ads, val, expf = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
        ads_list.append(ads)
        val_list.append(val)
        expf_list.append(expf)
        all_ads.append(ads)   
        
        # plot measurements and predictions by ads
        fig, _ = plt.subplots(4,2,sharex=True)
        expf.plot_experiment(fig=fig)
        val.plot_experiment(fig=fig)
all_ads.plot(fig_damping = fig_damping, fig_stiffness=fig_stiffness)


plt.show()
    
    
#%% Plot aerodynamic derivatives