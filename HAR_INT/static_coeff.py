# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:47:47 2022

@author: oiseth
"""
#%%
import numpy as np
import sys
sys.path.append(r"C:\Users\liner\Documents\Github\Masteroppgave\w3tp")
import w3t as w3t
print(w3t.__file__)
import os
import h5py
from matplotlib import pyplot as plt
import time
import pandas as pd
#from scipy import signal as spsp
 #%%
tic = time.perf_counter()
plt.close("all")


section_height = 3.33/100
section_width =  18.3/100
section_length = 2.68

#%% Load all experiments
def load_and_process_static_coeff(h5_input_path, section_name, file_names, filter_order = 6, filter_cutoff_frequency = 2, mode="decks"):
    """Gather, filter, calculate and plot static coeff for."""

    h5_file = os.path.join(h5_input_path, section_name)
    f = h5py.File((h5_file + ".hdf5"), "r")

    exp0 = w3t.Experiment.fromWTT(f[file_names[0]])
    exp1 = w3t.Experiment.fromWTT(f[file_names[1]])

    #exp0ms_single.plot_experiment() #Before filtering
    #plt.gcf().suptitle("Before filtering", fontsize=16)
    #exp6ms_single.plot_experiment() #Before filtering
    #plt.gcf().suptitle("Before filtering", fontsize=16)

    exp0.filt_forces(filter_order, filter_cutoff_frequency)
    exp1.filt_forces(filter_order, filter_cutoff_frequency)

    static_coeff = w3t.StaticCoeff.fromWTT(exp0, exp1, section_width, section_height, section_length)

    static_coeff.plot_drag(mode=mode)
    static_coeff.plot_lift(mode=mode)
    static_coeff.plot_pitch(mode=mode)
    plt.show()
    return exp0,exp1,static_coeff

h5_input_path = r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Python\Ole_sin_kode\HAR_INT\H5F\\"

#%% Load single deck
section_name = "Single_Static"
file_names = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_005"]

exp0 = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="single")[0]
exp1 = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="single")[1]
static_coeff_single = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[2]

#exp0ms_single.plot_experiment() #After filtering
#plt.gcf().suptitle("MUS 2D 0 ms – After filtering", fontsize=16)
#exp6ms_single.plot_experiment() #After filtering
#plt.gcf().suptitle("MUS 2D 6 ms – After filtering", fontsize=16)




#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_2D_Static"
file_names = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_002"]

exp0 = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[0]
exp1 = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[1]
static_coeff_down = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[2]

#exp0ms_single.plot_experiment() #After filtering
#plt.gcf().suptitle("MUS 2D 0 ms – After filtering", fontsize=16)
#exp6ms_single.plot_experiment() #After filtering
#plt.gcf().suptitle("MUS 2D 6 ms – After filtering", fontsize=16)



#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_2D_Static"
file_names = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_001"]

exp0 = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[0]
exp1 = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[1]
static_coeff_up = load_and_process_static_coeff(h5_input_path, section_name, file_names, mode="decks")[2]

#exp0ms_single.plot_experiment() #After filtering
#plt.gcf().suptitle("MUS 2D 0 ms – After filtering", fontsize=16)
#exp6ms_single.plot_experiment() #After filtering
#plt.gcf().suptitle("MUS 2D 6 ms – After filtering", fontsize=16)


#%% Compare all experiments
w3t._scoff.plot_compare_drag(static_coeff_single, static_coeff_up, static_coeff_down)
w3t._scoff.plot_compare_lift(static_coeff_single, static_coeff_up, static_coeff_down)
w3t._scoff.plot_compare_pitch(static_coeff_single, static_coeff_up, static_coeff_down)


plt.show()


static_coeff_down.to_excel(static_coeff_down,section_name = "2D", sheet_name="MUS #" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up.to_excel(static_coeff_down,section_name = "2D", sheet_name='MDS #' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_up.to_excel(static_coeff_down,section_name = "2D", sheet_name='Single #' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)



