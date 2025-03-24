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
import matplotlib as mpl

# Computer Modern Roman without latex
mpl.rcParam/s['text.usetex'] = False  

# Bruk fonten som ligner mest på Computer Modern
mpl.rcParam/s['font.family'] = 'serif'
mpl.rcParam/s['font.serif'] = ['cmr10', 'Computer Modern Roman', 'Times New Roman']
mpl.rcParam/s['mathtext.fontset'] = 'cm' 

# Generelt større og mer lesbar tekst
mpl.rcParam/s.update({
    'font.size': 12,              # Generell tekststørrelse
    'axes.labelsize': 12,         # Aksetitler
    'axes.titlesize': 14,         # Plot-titler
    'legend.fontsize': 12,        # Tekst i legend
    'xtick.labelsize': 12,        # X-tick labels
    'ytick.labelsize': 12         # Y-tick labels
})

#from scipy import signal as spsp

def load_experiments_from_hdf5(h5_input_path, section_name, file_names,  upwind_in_rig=True):
    h5_file = os.path.join(h5_input_path, section_name)
    f = h5py.File(h5_file + ".hdf5", "r")
    exp0 = w3t.Experiment.fromWTT(f[file_names[0]])
    exp1 = w3t.Experiment.fromWTT(f[file_names[1]])

    if upwind_in_rig:
        section_name = "MUS"
    else: section_name = "MDS"
    

    return exp0, exp1


def plot_static_coeff_summary(static_coeff, section_name, wind_speed, mode="decks", upwind_in_rig=True):
    if "MUS" in section_name:
        section_name = section_name.replace("MUS", "MDS")
    elif "MDS" in section_name:
        section_name = section_name.replace("MDS", "MUS")

    static_coeff.plot_drag(mode=mode,upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    static_coeff.plot_lift(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    static_coeff.plot_pitch(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")

    #mean
    static_coeff.plot_drag_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    static_coeff.plot_lift_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    static_coeff.plot_pitch_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")

    plt.tight_layout()
    plt.show()



# Load all experiments
tic = time.perf_counter()
plt.close("all")


section_height = 0.066
section_width =  0.365
section_length_in_rig = 2.68
section_length_on_wall = 2.66

h5_input_path = r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Python\Ole_sin_kode\HAR_INT\H5F\\"

#%% Load single deck
section_name = "Single_Static"
file_names_6 = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_005"] # 6 m/s
file_names_9 = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_004"] # 9 m/s, Vibrations

exp0_single, exp1_single_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_6,  upwind_in_rig=True)
exp0_single, exp1_single_9= load_experiments_from_hdf5(h5_input_path, section_name, file_names_9,  upwind_in_rig=True)

exp0_single.plot_experiment(mode="total") #
plt.gcf().suptitle(f"Single deck - Wind speed: 0 m/s",  y=1.05)
exp1_single_6.plot_experiment(mode="total") #
plt.gcf().suptitle(f"Single deck - Wind speed: 6 m/s",  y=1.05)
exp1_single_9.plot_experiment(mode="total") #
plt.gcf().suptitle(f"Single deck - Wind speed: 9 m/s",  y=1.05)

exp0_single.filt_forces(6, 2)
exp1_single_6.filt_forces(6, 2)
exp1_single_9.filt_forces(6, 2)

exp0_single.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"Single deck - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=1.1)
exp1_single_6.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"Single deck - Wind speed: 6 m/s - With Butterworth low-pass filter",  y=1.1)
exp1_single_9.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"Single deck - Wind speed: 9 m/s - With Butterworth low-pass filter",  y=1.1)
plt.show()


static_coeff_single_6 =w3t.StaticCoeff.fromWTT(exp0_single, exp1_single_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_single_9 = w3t.StaticCoeff.fromWTT(exp0_single, exp1_single_9, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_single_6, section_name, 6, mode="single", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_single_9, section_name, 9, mode="single", upwind_in_rig=True)




#%% Filter and plot ALT 1
#drag
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_6, threshold=0.05, scoff="drag", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"Single deck - Wind speed: 6 m/s",  y=1.05)
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_9, threshold=0.05, scoff="drag", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"Single deck - Wind speed: 9 m/s",  y=1.05)

#lift
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_6, threshold=0.05, scoff="lift", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"Single deck - Wind speed: 6 m/s",  y=1.05)
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_9, threshold=0.05, scoff="lift", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"Single deck - Wind speed: 9 m/s",  y=1.05)

#pitch
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_6, threshold=0.05, scoff="pitch", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"Single deck - Wind speed: 6 m/s",  y=1.05)
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_9, threshold=0.05, scoff="pitch", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"Single deck - Wind speed: 9 m/s",  y=1.05)

#%%  Filter and plot ALT 2

static_coeff_single_6_filtered, static_coeff_single_9_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_single_6,static_coeff_2=static_coeff_single_9,threshold=0.05,  threshold_low=[0.07, 0.05,0.005], threshold_high=[0.019,0.05,0.01],single=True)

plot_static_coeff_summary(static_coeff_single_6_filtered, section_name, 6, mode="single", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_single_9_filtered, section_name, 9, mode="single", upwind_in_rig=True)



############################################################################################################

#print("1D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_1D_Static"
file_names_MDS_1D_6 = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_001"] #6 m/s, vibrations (Ser OK ut)
file_names_MDS_1D_8 = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_002"] # 8 m/s, vibrations
file_names_MDS_1D_10 = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_003"] # 10 m/s



exp0_MDS_1D, exp1_MDS_1D_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_1D_6,  upwind_in_rig=False)
exp0_MDS_1D, exp1_MDS_1D_8= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_1D_8,  upwind_in_rig=False)
exp0_MDS_1D, exp1_MDS_1D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_1D_10,  upwind_in_rig=False)


exp0_MDS_1D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MDS 1D - Wind speed: 0 m/s",  y=1.05)
exp1_MDS_1D_6.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=1.05)
exp1_MDS_1D_8.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=1.05)
exp1_MDS_1D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=1.05)

exp0_MDS_1D.filt_forces(6, 2)
exp1_MDS_1D_6.filt_forces(6, 2)
exp1_MDS_1D_8.filt_forces(6, 2)
exp1_MDS_1D_10.filt_forces(6, 2)

exp0_MDS_1D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MDS 1D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_1D_6.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MDS 1D - Wind speed:) 6 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_1D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MDS 1D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_1D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MDS 1D - Wind speed:) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MDS_1D_6 =w3t.StaticCoeff.fromWTT(exp0_MDS_1D, exp1_MDS_1D_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_1D_8 = w3t.StaticCoeff.fromWTT(exp0_MDS_1D, exp1_MDS_1D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_1D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_1D, exp1_MDS_1D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)


plot_static_coeff_summary(static_coeff_MDS_1D_6, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_8, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_1D_6, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_1D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_1D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_1D_6, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_1D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_1D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_1D_6, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_1D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_1D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=1.05)



#%%  Filter and plot ALT 2
static_coeff_MDS_1D_6_filtered, static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_1D_6, static_coeff_2=static_coeff_MDS_1D_8, static_coeff_3=static_coeff_MDS_1D_10, threshold=0.05, threshold_low=[0.06,0.05,0.05],threshold_med = [0.005,0.03,0.05],threshold_high=[0.05,0.01,0.05],single=False)

plot_static_coeff_summary(static_coeff_MDS_1D_6_filtered, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=False)




#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_1D_Static"
file_names_MUS_1D_5 = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_001"] # 5 m/s
file_names_MUS_1D_8 = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_002"] # 8 m/s, vibrations
file_names_MUS_1D_10 = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_003"] # 10 m/s, vibrations

exp0_MUS_1D, exp1_MUS_1D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_1D_5,  upwind_in_rig=True)
exp0_MUS_1D, exp1_MUS_1D_8 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_1D_8,  upwind_in_rig=True)
exp0_MUS_1D, exp1_MUS_1D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_1D_10,  upwind_in_rig=True)



exp0_MUS_1D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MUS 1D - Wind speed: 0 m/s - ",  y=1.05)
exp1_MUS_1D_5.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MUS 1D - Wind speed: 5 m/s - ",  y=1.05)
exp1_MUS_1D_8.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MUS 1D - Wind speed: 8 m/s - ",  y=1.05)
exp1_MUS_1D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"MUS 1D - Wind speed: 10 m/s - ",  y=1.05)

exp0_MUS_1D.filt_forces(6, 2)
exp1_MUS_1D_5.filt_forces(6, 2)
exp1_MUS_1D_8.filt_forces(6, 2)
exp1_MUS_1D_10.filt_forces(6, 2)

exp0_MUS_1D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MUS 1D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_1D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MUS 1D - Wind speed: 5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_1D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MUS 1D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_1D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"MUS 1D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MUS_1D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_1D, exp1_MUS_1D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_1D_8 = w3t.StaticCoeff.fromWTT(exp0_MUS_1D, exp1_MUS_1D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_1D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_1D, exp1_MUS_1D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_1D_5, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_1D_8, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_1D_10, section_name, 10, mode="decks", upwind_in_rig=True)



#%% Filter and plot ALT 1_MUS_1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_1D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS 1D - Wind speed: 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_1D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS 1D - Wind speed: 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_1D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS 1D - Wind speed: 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_1D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS 1D - Wind speed: 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_1D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS 1D - Wind speed: 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_1D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS 1D - Wind speed: 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_1D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS 1D - Wind speed: 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_1D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS 1D - Wind speed: 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_1D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS 1D - Wind speed: 10 m/s",  y=1.05)


#%%  Filter and plot ALT 2
static_coeff_MUS_1D_5_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_1D_5, static_coeff_2=static_coeff_MUS_1D_8, static_coeff_3=static_coeff_MUS_1D_10, threshold=0.019, threshold_low=[0.06,0.04,0.01],threshold_med = [0.01,0.04,0.01],threshold_high=[0.02,0.04,0.01],single=False)


plot_static_coeff_summary(static_coeff_MUS_1D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_1D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_1D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% Save all experiments to excel
section_name = "1D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_1D_6.to_excel(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_1D_8.to_excel(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_8.to_excel(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "1D_mean"

# Low wind speed
static_coeff_MDS_1D_6.to_excel_mean(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_1D_8.to_excel_mean(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_8.to_excel_mean(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% Save all experiments to excel filtered
section_name = "1D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_1D_6_filtered.to_excel(section_name, sheet_name="MDS - 6 - m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5_filtered.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_1D_8_filtered.to_excel(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_8_filtered.to_excel(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10_filtered.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10_filtered.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "1D_mean_filtered"

# Low wind speed
static_coeff_MDS_1D_6_filtered.to_excel_mean(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_1D_8_filtered.to_excel_mean(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_8_filtered.to_excel_mean(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10_filtered.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "1D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments (MUS vs MDS vs Single) filtered
section_name = "1D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MDS_1D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MDS_1D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MDS_1D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MDS_1D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MDS_1D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, static_coeff_MDS_1D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_1D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MDS_1D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_1D_8,  upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_8,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_8,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_1D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_1D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()

#%% Compare all experiments - only with single deck filtered

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MDS_1D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_1D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_1D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_1D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_1D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_1D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_8_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_8_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()
# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                               static_coeff_MUS_1D_8, static_coeff_MUS_1D_10,
                             scoff = "drag")                        
plt.gcf().suptitle(f"1D: MUS ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_8, static_coeff_MDS_1D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"1D: MDS ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                               static_coeff_MUS_1D_8, static_coeff_MUS_1D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9,static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_8, static_coeff_MDS_1D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                               static_coeff_MUS_1D_8, static_coeff_MUS_1D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_8, static_coeff_MDS_1D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                               static_coeff_MUS_1D_8, static_coeff_MUS_1D_10,
                           scoff = "drag")                        
plt.gcf().suptitle(f"1D: MUS ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_8, static_coeff_MDS_1D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                               static_coeff_MUS_1D_8, static_coeff_MUS_1D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_8, static_coeff_MDS_1D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"1D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                               static_coeff_MUS_1D_8, static_coeff_MUS_1D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_8, static_coeff_MDS_1D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MDS ",  y=1.05)


# %% Compare all experiments (Wind speed) filtered
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                               static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6_filtered,
                               static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered,
                                scoff = "drag")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                               static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered,
                            scoff = "lift")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6_filtered,
                               static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered,
                               scoff = "lift")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                               static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6_filtered,
                               static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MDS ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                               static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6_filtered,
                               static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered,
                              scoff = "drag")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                               static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"1D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6_filtered,
                               static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"1D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                               static_coeff_MUS_1D_8_filtered, static_coeff_MUS_1D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6_filtered,
                               static_coeff_MDS_1D_8_filtered, static_coeff_MDS_1D_10_filtered,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=1.05)




############################################################################################################

#print("2D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_2D_Static"
file_names_MDS_2D_6 = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_002"] #6 m/s
file_names_MDS_2D_8 = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_003"] # 8 m/s, vibrations
file_names_MDS_2D_10 = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_004"] # 10 m/s


exp0_MDS_2D, exp1_MDS_2D_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_2D_6,  upwind_in_rig=False)
exp0_MDS_2D, exp1_MDS_2D_8= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_2D_8,  upwind_in_rig=False)
exp0_MDS_2D, exp1_MDS_2D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_2D_10,  upwind_in_rig=False)



exp0_MDS_2D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MDS_2D_6.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 6 m/s - ",  y=1.05)
exp1_MDS_2D_8.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - ",  y=1.05)
exp1_MDS_2D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MDS_2D.filt_forces(6, 2)
exp1_MDS_2D_6.filt_forces(6, 2)
exp1_MDS_2D_8.filt_forces(6, 2)
exp1_MDS_2D_10.filt_forces(6, 2)

exp0_MDS_2D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_2D_6.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 6 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_2D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_2D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MDS_2D_6 =w3t.StaticCoeff.fromWTT(exp0_MDS_2D, exp1_MDS_2D_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_2D_8 = w3t.StaticCoeff.fromWTT(exp0_MDS_2D, exp1_MDS_2D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_2D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_2D, exp1_MDS_2D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

plot_static_coeff_summary(static_coeff_MDS_2D_6, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_2D_8, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_2D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_2D_6, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_2D_Static, 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_2D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_2D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_2D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_2D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_2D_6, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_2D_Static, 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_2D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_2D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_2D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_2D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_2D_6, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_2D_Static, 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_2D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_2D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_2D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_2D_Static, 10 m/s",  y=1.05)



#%%  Filter and plot ALT 2
static_coeff_MDS_2D_6_filtered,static_coeff_MDS_2D_8_filtered,static_coeff_MDS_2D_10_filtered  = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_2D_6, static_coeff_2=static_coeff_MDS_2D_8, static_coeff_3=static_coeff_MDS_2D_10, threshold=0.05, threshold_low=[0.05,0.05,0.05],threshold_med = [0.0199,0.02,0.01],threshold_high=[0.05,0.01,0.05],single=False)

plot_static_coeff_summary(static_coeff_MDS_2D_6_filtered, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_2D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_2D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=False)



#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_2D_Static"
file_names_MUS_2D_5 = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_001"] # 5 m/s, vibrations
file_names_MUS_2D_8 = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_002"] # 8 m/s, vibrations
file_names_MUS_2D_10 = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_003"] # 10 m/s, vibrations



exp0_MUS_2D, exp1_MUS_2D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_2D_5,  upwind_in_rig=True)
exp0_MUS_2D, exp1_MUS_2D_8 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_2D_8,  upwind_in_rig=True)
exp0_MUS_2D, exp1_MUS_2D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_2D_10,  upwind_in_rig=True)




exp0_MUS_2D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MUS_2D_5.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - ",  y=1.05)
exp1_MUS_2D_8.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - ",  y=1.05)
exp1_MUS_2D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MUS_2D.filt_forces(6, 2)
exp1_MUS_2D_5.filt_forces(6, 2)
exp1_MUS_2D_8.filt_forces(6, 2)
exp1_MUS_2D_10.filt_forces(6, 2)

exp0_MUS_2D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_2D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_2D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_2D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MUS_2D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_2D, exp1_MUS_2D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_2D_8 = w3t.StaticCoeff.fromWTT(exp0_MUS_2D, exp1_MUS_2D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_2D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_2D, exp1_MUS_2D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)


plot_static_coeff_summary(static_coeff_MUS_2D_5, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_2D_8, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_2D_10, section_name, 10, mode="decks", upwind_in_rig=True)



#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_2D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_2D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_2D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_2D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_2D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_2D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_2D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.025, scoff="lift")
plt.suptitle(f"MUS_2D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_2D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_2D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_2D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_2D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_2D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_2D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_2D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_2D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_2D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_2D_Static, 10 m/s",  y=1.05)


#%%  Filter and plot ALT 2
static_coeff_MUS_2D_5_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_2D_5, static_coeff_2=static_coeff_MUS_2D_8, static_coeff_3=static_coeff_MUS_2D_10, threshold=0.019, threshold_low=[0.05,0.025,0.005],threshold_med = [0.04,0.03,0.02],threshold_high=[0.02,0.04,0.01],single=False)


plot_static_coeff_summary(static_coeff_MUS_2D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_2D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_2D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% Save all experiments to excel
section_name = "2D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_2D_6.to_excel(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_5.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_2D_8.to_excel(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_8.to_excel(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_2D_10.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_10.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "2D_mean"

# Low wind speed
static_coeff_MDS_2D_6.to_excel_mean(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_5.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_2D_8.to_excel_mean(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_8.to_excel_mean(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_2D_10.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_10.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% Save all experiments to excel filtered
section_name = "2D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_2D_6_filtered.to_excel(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_5_filtered.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_2D_8_filtered.to_excel(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_8_filtered.to_excel(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_2D_10_filtered.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_10_filtered.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "2D_mean_filtered"

# Low wind speed
static_coeff_MDS_2D_6_filtered.to_excel_mean(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_2D_8_filtered.to_excel_mean(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_8_filtered.to_excel_mean(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_2D_10_filtered.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "2D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments (MUS vs MDS vs Single) filtered
section_name = "2D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MDS_2D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MDS_2D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MDS_2D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MDS_2D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MDS_2D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, static_coeff_MDS_2D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_2D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MDS_2D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_2D_8,  upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_8,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_8,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_2D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_2D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()

#%% Compare all experiments - only with single deck filtered

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_8_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_8_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()
# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                               static_coeff_MUS_2D_8, static_coeff_MUS_2D_10,
                             scoff = "drag")                        
plt.gcf().suptitle(f"2D: MUS ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                               static_coeff_MDS_2D_8, static_coeff_MDS_2D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"2D: MDS ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                               static_coeff_MUS_2D_8, static_coeff_MUS_2D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9,static_coeff_MDS_2D_6,
                               static_coeff_MDS_2D_8, static_coeff_MDS_2D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"2D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                               static_coeff_MUS_2D_8, static_coeff_MUS_2D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                               static_coeff_MDS_2D_8, static_coeff_MDS_2D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MDS  ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                               static_coeff_MUS_2D_8, static_coeff_MUS_2D_10,
                           scoff = "drag")                        
plt.gcf().suptitle(f"2D: MUS ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                               static_coeff_MDS_2D_8, static_coeff_MDS_2D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"2D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                               static_coeff_MUS_2D_8, static_coeff_MUS_2D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                               static_coeff_MDS_2D_8, static_coeff_MDS_2D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"2D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                               static_coeff_MUS_2D_8, static_coeff_MUS_2D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                               static_coeff_MDS_2D_8, static_coeff_MDS_2D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MDS ",  y=1.05)


# %% Compare all experiments (Wind speed) filtered
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                               static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"2D: MUS  ", fontsize=16)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6_filtered,
                               static_coeff_MDS_2D_8_filtered, static_coeff_MDS_2D_10_filtered,
                                scoff = "drag")                        
plt.gcf().suptitle(f"2D: MDS  ", fontsize=16)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                               static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered,
                            scoff = "lift")                        
plt.gcf().suptitle(f"2D: MUS  ", fontsize=16)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6_filtered,
                               static_coeff_MDS_2D_8_filtered, static_coeff_MDS_2D_10_filtered,
                               scoff = "lift")                        
plt.gcf().suptitle(f"2D: MDS  ", fontsize=16)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                               static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MUS ", fontsize=16)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6_filtered,
                               static_coeff_MDS_2D_8_filtered, static_coeff_MDS_2D_10_filtered,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MDS ", fontsize=16)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                               static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"2D: MUS  ", fontsize=16)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6_filtered,
                               static_coeff_MDS_2D_8_filtered, static_coeff_MDS_2D_10_filtered,
                              scoff = "drag")                        
plt.gcf().suptitle(f"2D: MDS  ", fontsize=16)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                               static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"2D: MUS ", fontsize=16)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6_filtered,
                               static_coeff_MDS_2D_8_filtered, static_coeff_MDS_2D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"2D: MDS ", fontsize=16)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                               static_coeff_MUS_2D_8_filtered, static_coeff_MUS_2D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MUS ", fontsize=16)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6_filtered,
                               static_coeff_MDS_2D_8_filtered, static_coeff_MDS_2D_10_filtered,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MDS  ", fontsize=16)







############################################################################################################
#%%
#print("3D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_3D_Static"
file_names_MDS_3D_6 = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_001"] #6 m/s
file_names_MDS_3D_8 = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_002"] # 8 m/s, vibrations
file_names_MDS_3D_10 = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_003"] # 10 m/s


exp0_MDS_3D, exp1_MDS_3D_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_3D_6,  upwind_in_rig=False)
exp0_MDS_3D, exp1_MDS_3D_8= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_3D_8,  upwind_in_rig=False)
exp0_MDS_3D, exp1_MDS_3D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_3D_10,  upwind_in_rig=False)



exp0_MDS_3D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MDS_3D_6.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 6 m/s - ",  y=1.05)
exp1_MDS_3D_8.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - ",  y=1.05)
exp1_MDS_3D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MDS_3D.filt_forces(6, 2)
exp1_MDS_3D_6.filt_forces(6, 2)
exp1_MDS_3D_8.filt_forces(6, 2)
exp1_MDS_3D_10.filt_forces(6, 2)

exp0_MDS_3D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_3D_6.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 6 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_3D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_3D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MDS_3D_6 =w3t.StaticCoeff.fromWTT(exp0_MDS_3D, exp1_MDS_3D_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_3D_8 = w3t.StaticCoeff.fromWTT(exp0_MDS_3D, exp1_MDS_3D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_3D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_3D, exp1_MDS_3D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)


plot_static_coeff_summary(static_coeff_MDS_3D_6, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_3D_8, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_3D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_3D_6, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_3D_Static, 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_3D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_3D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_3D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_3D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_3D_6, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_3D_Static, 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_3D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_3D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_3D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_3D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_3D_6, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_3D_Static, 6 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_3D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_3D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_3D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_3D_Static, 10 m/s",  y=1.05)



#%%  Filter and plot ALT 2
static_coeff_MDS_3D_6_filtered, static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_3D_6, static_coeff_2=static_coeff_MDS_3D_8, static_coeff_3=static_coeff_MDS_3D_10, threshold=0.05, threshold_low=[0.06,0.05,0.05],threshold_med = [0.005,0.03,0.05],threshold_high=[0.05,0.01,0.05],single=False)

plot_static_coeff_summary(static_coeff_MDS_3D_6_filtered, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_3D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_3D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=False)




#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_3D_Static"
file_names_MUS_3D_5 = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_004"] # 5 m/s, vibrations (Finnes en fil for 6 også)
file_names_MUS_3D_8 = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_006"] # 8 m/s, vibrations
file_names_MUS_3D_10 = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_005"] # 10 m/s, vibrations



exp0_MUS_3D, exp1_MUS_3D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_3D_5,  upwind_in_rig=True)
exp0_MUS_3D, exp1_MUS_3D_8 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_3D_8,  upwind_in_rig=True)
exp0_MUS_3D, exp1_MUS_3D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_3D_10,  upwind_in_rig=True)


exp0_MUS_3D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MUS_3D_5.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - ",  y=1.05)
exp1_MUS_3D_8.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - ",  y=1.05)
exp1_MUS_3D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MUS_3D.filt_forces(6, 2)
exp1_MUS_3D_5.filt_forces(6, 2)
exp1_MUS_3D_8.filt_forces(6, 2)
exp1_MUS_3D_10.filt_forces(6, 2)

exp0_MUS_3D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_3D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_3D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_3D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MUS_3D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_3D, exp1_MUS_3D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_3D_8 = w3t.StaticCoeff.fromWTT(exp0_MUS_3D, exp1_MUS_3D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_3D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_3D, exp1_MUS_3D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_3D_5, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_3D_8, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_3D_10, section_name, 10, mode="decks", upwind_in_rig=True)



#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_3D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_3D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_3D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_3D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_3D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_3D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_3D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_3D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_3D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_3D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_3D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_3D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_3D_5, threshold=0.005, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.005, scoff="pitch")
plt.suptitle(f"MUS_3D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_3D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_3D_Static, 8 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_3D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_3D_Static, 10 m/s",  y=1.05)


#%%  Filter and plot ALT 2
static_coeff_MUS_3D_5_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_3D_5, static_coeff_2=static_coeff_MUS_3D_8, static_coeff_3=static_coeff_MUS_3D_10, threshold=0.019, threshold_low=[0.05,0.025,0.005],threshold_med = [0.04,0.03,0.02],threshold_high=[0.02,0.04,0.01],single=False)


plot_static_coeff_summary(static_coeff_MUS_3D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_3D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_3D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% Save all experiments to excel
section_name = "3D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_3D_6.to_excel(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_5.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_3D_8.to_excel(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_8.to_excel(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_3D_10.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_10.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "3D_mean"

# Low wind speed
static_coeff_MDS_3D_6.to_excel_mean(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_5.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_3D_8.to_excel_mean(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_8.to_excel_mean(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_3D_10.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_10.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% Save all experiments to excel filtered
section_name = "3D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_3D_6_filtered.to_excel(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_5_filtered.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_3D_8_filtered.to_excel(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_8_filtered.to_excel(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_3D_10_filtered.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_10_filtered.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "3D_mean_filtered"

# Low wind speed
static_coeff_MDS_3D_6_filtered.to_excel_mean(section_name, sheet_name="MDS - 6 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_3D_8_filtered.to_excel_mean(section_name, sheet_name="MDS - 8 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_8_filtered.to_excel_mean(section_name, sheet_name='MUS - 8 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_3D_10_filtered.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "3D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments (MUS vs MDS vs Single) filtered
section_name = "3D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MDS_3D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MDS_3D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MDS_3D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MDS_3D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MDS_3D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, static_coeff_MDS_3D_8_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_3D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MDS_3D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_3D_8,  upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_8,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_8,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_3D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_3D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()

#%% Compare all experiments - only with single deck filtered

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_8_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_8_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_8_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()
# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                               static_coeff_MUS_3D_8, static_coeff_MUS_3D_10,
                             scoff = "drag")                        
plt.gcf().suptitle(f"3D: MUS ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                               static_coeff_MDS_3D_8, static_coeff_MDS_3D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"3D: MDS ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                               static_coeff_MUS_3D_8, static_coeff_MUS_3D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9,static_coeff_MDS_3D_6,
                               static_coeff_MDS_3D_8, static_coeff_MDS_3D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                               static_coeff_MUS_3D_8, static_coeff_MUS_3D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                               static_coeff_MDS_3D_8, static_coeff_MDS_3D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                               static_coeff_MUS_3D_8, static_coeff_MUS_3D_10,
                           scoff = "drag")                        
plt.gcf().suptitle(f"3D: MUS ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                               static_coeff_MDS_3D_8, static_coeff_MDS_3D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                               static_coeff_MUS_3D_8, static_coeff_MUS_3D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                               static_coeff_MDS_3D_8, static_coeff_MDS_3D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"3D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                               static_coeff_MUS_3D_8, static_coeff_MUS_3D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                               static_coeff_MDS_3D_8, static_coeff_MDS_3D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MDS ",  y=1.05)


# %% Compare all experiments (Wind speed) filtered
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                               static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6_filtered,
                               static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered,
                                scoff = "drag")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                               static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered,
                            scoff = "lift")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6_filtered,
                               static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered,
                               scoff = "lift")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                               static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6_filtered,
                               static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MDS ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                               static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6_filtered,
                               static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered,
                              scoff = "drag")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                               static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"3D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6_filtered,
                               static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"3D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                               static_coeff_MUS_3D_8_filtered, static_coeff_MUS_3D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6_filtered,
                               static_coeff_MDS_3D_8_filtered, static_coeff_MDS_3D_10_filtered,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=1.05)






############################################################################################################
#%%
#print("4D")

#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_4D_Static"
file_names_MDS_4D_55 = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_002"] #5.5 m/s
file_names_MDS_4D_85 = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_003"] # 8.5 m/s, vibrations
file_names_MDS_4D_10 = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_004"] # 10 m/s

exp0_MDS_4D, exp1_MDS_4D_55 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_4D_55,  upwind_in_rig=False)
exp0_MDS_4D, exp1_MDS_4D_85= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_4D_85,  upwind_in_rig=False)
exp0_MDS_4D, exp1_MDS_4D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_4D_10,  upwind_in_rig=False)



exp0_MDS_4D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MDS_4D_55.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - ",  y=1.05)
exp1_MDS_4D_85.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - ",  y=1.05)
exp1_MDS_4D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MDS_4D.filt_forces(6, 2)
exp1_MDS_4D_55.filt_forces(6, 2)
exp1_MDS_4D_85.filt_forces(6, 2)
exp1_MDS_4D_10.filt_forces(6, 2)

exp0_MDS_4D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_4D_55.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_4D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_4D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MDS_4D_55 =w3t.StaticCoeff.fromWTT(exp0_MDS_4D, exp1_MDS_4D_55, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_4D_85 = w3t.StaticCoeff.fromWTT(exp0_MDS_4D, exp1_MDS_4D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_4D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_4D, exp1_MDS_4D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)


plot_static_coeff_summary(static_coeff_MDS_4D_55, section_name, 5.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_4D_85, section_name, 8.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_4D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_4D_55, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_4D_Static, 5.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_4D_85, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_4D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_4D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_4D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_4D_55, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_4D_Static, 5.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_4D_85, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_4D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_4D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_4D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_4D_55, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_4D_Static, 5.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_4D_85, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_4D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_4D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_4D_Static, 10 m/s",  y=1.05)



#%%  Filter and plot ALT 2
static_coeff_MDS_4D_55_filtered, static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_4D_55, static_coeff_2=static_coeff_MDS_4D_85, static_coeff_3=static_coeff_MDS_4D_10, threshold=0.05, threshold_low=[0.06,0.05,0.05],threshold_med = [0.005,0.03,0.05],threshold_high=[0.05,0.01,0.05],single=False)

plot_static_coeff_summary(static_coeff_MDS_4D_55_filtered, section_name, 5.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_4D_85_filtered, section_name, 8.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_4D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=False)



#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_4D_Static"
file_names_MUS_4D_5 = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_003"] # 5 m/s, vibrations 
file_names_MUS_4D_85= ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_004"] # 8.5 m/s, vibrations
file_names_MUS_4D_10 = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_005"] # 10 m/s, vibrations


exp0_MUS_4D, exp1_MUS_4D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_4D_5,  upwind_in_rig=True)
exp0_MUS_4D, exp1_MUS_4D_85 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_4D_85,  upwind_in_rig=True)
exp0_MUS_4D, exp1_MUS_4D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_4D_10,  upwind_in_rig=True)


exp0_MUS_4D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MUS_4D_5.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - ",  y=1.05)
exp1_MUS_4D_85.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - ",  y=1.05)
exp1_MUS_4D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MUS_4D.filt_forces(6, 2)
exp1_MUS_4D_5.filt_forces(6, 2)
exp1_MUS_4D_85.filt_forces(6, 2)
exp1_MUS_4D_10.filt_forces(6, 2)

exp0_MUS_4D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_4D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_4D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_4D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MUS_4D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_4D, exp1_MUS_4D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_4D_85 = w3t.StaticCoeff.fromWTT(exp0_MUS_4D, exp1_MUS_4D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_4D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_4D, exp1_MUS_4D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_4D_5, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_4D_85, section_name, 8.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_4D_10, section_name, 10, mode="decks", upwind_in_rig=True)



#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_4D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_4D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_4D_85, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_4D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_4D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_4D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_4D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_4D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_4D_85, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_4D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_4D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_4D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_4D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_4D_Static, 5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_4D_85, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_4D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_4D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_4D_Static, 10 m/s",  y=1.05)


#%%  Filter and plot ALT 2
static_coeff_MUS_4D_5_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_4D_5, static_coeff_2=static_coeff_MUS_4D_85, static_coeff_3=static_coeff_MUS_4D_10, threshold=0.019, threshold_low=[0.05,0.025,0.005],threshold_med = [0.04,0.03,0.02],threshold_high=[0.02,0.04,0.01],single=False)


plot_static_coeff_summary(static_coeff_MUS_4D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_4D_85_filtered, section_name, 8.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_4D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% Save all experiments to excel
section_name = "4D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_4D_55.to_excel(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_5.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_4D_85.to_excel(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_85.to_excel(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_4D_10.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_10.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "4D_mean"

# Low wind speed
static_coeff_MDS_4D_55.to_excel_mean(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_5.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_4D_85.to_excel_mean(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_85.to_excel_mean(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_4D_10.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_10.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% Save all experiments to excel filtered
section_name = "4D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_4D_55_filtered.to_excel(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_5_filtered.to_excel(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_4D_85_filtered.to_excel(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_85_filtered.to_excel(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_4D_10_filtered.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_10_filtered.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "4D_mean_filtered"

# Low wind speed
static_coeff_MDS_4D_55_filtered.to_excel_mean(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_4D_85_filtered.to_excel_mean(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_85_filtered.to_excel_mean(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_4D_10_filtered.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "4D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments (MUS vs MDS vs Single) filtered
section_name = "4D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MDS_4D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS:  8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MDS_4D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS:  8.5 m/s, MDS:  8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MDS_4D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS:  8.5 m/s, MDS:  8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MDS_4D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS:  8.5 m/s, MDS:  8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MDS_4D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS:  8.5 m/s, MDS:  8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, static_coeff_MDS_4D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS:  8.5 m/s, MDS:  8.5 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_4D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MDS_4D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_4D_85,  upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_85,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_85,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_4D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_4D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()

#%% Compare all experiments - only with single deck filtered

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_85_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_85_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()
# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                               static_coeff_MUS_4D_85, static_coeff_MUS_4D_10,
                             scoff = "drag")                        
plt.gcf().suptitle(f"4D: MUS ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                               static_coeff_MDS_4D_85, static_coeff_MDS_4D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"4D: MDS ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                               static_coeff_MUS_4D_85, static_coeff_MUS_4D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9,static_coeff_MDS_4D_55,
                               static_coeff_MDS_4D_85, static_coeff_MDS_4D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                               static_coeff_MUS_4D_85, static_coeff_MUS_4D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                               static_coeff_MDS_4D_85, static_coeff_MDS_4D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                               static_coeff_MUS_4D_85, static_coeff_MUS_4D_10,
                           scoff = "drag")                        
plt.gcf().suptitle(f"4D: MUS ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                               static_coeff_MDS_4D_85, static_coeff_MDS_4D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                               static_coeff_MUS_4D_85, static_coeff_MUS_4D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                               static_coeff_MDS_4D_85, static_coeff_MDS_4D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"4D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                               static_coeff_MUS_4D_85, static_coeff_MUS_4D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                               static_coeff_MDS_4D_85, static_coeff_MDS_4D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MDS ",  y=1.05)

# %% Compare all experiments (Wind speed) filtered
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                               static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55_filtered,
                               static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered,
                                scoff = "drag")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                               static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered,
                            scoff = "lift")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55_filtered,
                               static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered,
                               scoff = "lift")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                               static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55_filtered,
                               static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MDS ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                               static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55_filtered,
                               static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered,
                              scoff = "drag")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                               static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"4D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55_filtered,
                               static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered,
                              scoff = "lift")                        
plt.gcf().suptitle(f"4D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                               static_coeff_MUS_4D_85_filtered, static_coeff_MUS_4D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55_filtered,
                               static_coeff_MDS_4D_85_filtered, static_coeff_MDS_4D_10_filtered,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=1.05)




############################################################################################################
#%%
#print("5D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_5D_Static"
file_names_MDS_5D_55 = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_001"] # 5.5 m/s
file_names_MDS_5D_85 = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_002"] # 8.5 m/s, vibrations
file_names_MDS_5D_10 = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_003"] # 10 m/s

exp0_MDS_5D, exp1_MDS_5D_55 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_5D_55,  upwind_in_rig=False)
exp0_MDS_5D, exp1_MDS_5D_85= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_5D_85,  upwind_in_rig=False)
exp0_MDS_5D, exp1_MDS_5D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_5D_10,  upwind_in_rig=False)




exp0_MDS_5D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MDS_5D_55.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 5.5 m/s - ",  y=1.05)
exp1_MDS_5D_85.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8.5 m/s - ",  y=1.05)
exp1_MDS_5D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MDS_5D.filt_forces(6, 2)
exp1_MDS_5D_55.filt_forces(6, 2)
exp1_MDS_5D_85.filt_forces(6, 2)
exp1_MDS_5D_10.filt_forces(6, 2)

exp0_MDS_5D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_5D_55.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 5.5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_5D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8.5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MDS_5D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MDS_5D_55 =w3t.StaticCoeff.fromWTT(exp0_MDS_5D, exp1_MDS_5D_55, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_5D_85 = w3t.StaticCoeff.fromWTT(exp0_MDS_5D, exp1_MDS_5D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_5D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_5D, exp1_MDS_5D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

plot_static_coeff_summary(static_coeff_MDS_5D_55, section_name, 5.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_5D_85, section_name, 8.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_5D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_5D_55, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_5D_Static, 5.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_5D_85, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_5D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_5D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_5D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_5D_55, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_5D_Static, 5.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_5D_85, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_5D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_5D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_5D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_5D_55, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_5D_Static, 5.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_5D_85, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_5D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_5D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_5D_Static, 10 m/s",  y=1.05)



#%%  Filter and plot ALT 2
static_coeff_MDS_5D_55_filtered, static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_5D_55, static_coeff_2=static_coeff_MDS_5D_85, static_coeff_3=static_coeff_MDS_5D_10, threshold=0.05, threshold_low=[0.06,0.05,0.05],threshold_med = [0.005,0.03,0.05],threshold_high=[0.05,0.01,0.05],single=False)

plot_static_coeff_summary(static_coeff_MDS_5D_55_filtered, section_name, 5.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_5D_85_filtered, section_name, 8.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_5D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=False)



#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_5D_Static"
file_names_MUS_5D_45 = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_002"] # 4.5 m/s, vibrations 
file_names_MUS_5D_85 = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_003"] # 8.5 m/s, vibrations
file_names_MUS_5D_10 = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_004"] # 10 m/s, vibrations



exp0_MUS_5D, exp1_MUS_5D_45= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_5D_45,  upwind_in_rig=True)
exp0_MUS_5D, exp1_MUS_5D_85 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_5D_85,  upwind_in_rig=True)
exp0_MUS_5D, exp1_MUS_5D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_5D_10,  upwind_in_rig=True)


exp0_MUS_5D.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - ",  y=1.05)
exp1_MUS_5D_45.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 4.5 m/s - ",  y=1.05)
exp1_MUS_5D_85.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 8.5 m/s - ",  y=1.05)
exp1_MUS_5D_10.plot_experiment(mode="total") #
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - ",  y=1.05)

exp0_MUS_5D.filt_forces(6, 2)
exp1_MUS_5D_45.filt_forces(6, 2)
exp1_MUS_5D_85.filt_forces(6, 2)
exp1_MUS_5D_10.filt_forces(6, 2)

exp0_MUS_5D.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 0 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_5D_45.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 4.5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_5D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 8.5 m/s - With Butterworth low-pass filter",  y=1.05)
exp1_MUS_5D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
plt.gcf().suptitle(f"{section_name} (ref excel) 10 m/s - With Butterworth low-pass filter",  y=1.05)
plt.show()


static_coeff_MUS_5D_45 =w3t.StaticCoeff.fromWTT(exp0_MUS_5D, exp1_MUS_5D_45, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_5D_85 = w3t.StaticCoeff.fromWTT(exp0_MUS_5D, exp1_MUS_5D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_5D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_5D, exp1_MUS_5D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_5D_45, section_name, 4.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_5D_85, section_name, 8.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_5D_10, section_name, 10, mode="decks", upwind_in_rig=True)



#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_5D_45, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_5D_Static, 4.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_5D_85, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_5D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_5D_10, threshold=0.0205, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_5D_Static, 10 m/s",  y=1.05)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_5D_45, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_5D_Static, 4.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_5D_85, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_5D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_5D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_5D_Static, 10 m/s",  y=1.05)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_5D_45, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_5D_Static, 4.5 m/s",  y=1.05)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_5D_85, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_5D_Static, 8.5 m/s",  y=1.05)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_5D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_5D_Static, 10 m/s",  y=1.05)


#%%  Filter and plot ALT 2
static_coeff_MUS_5D_45_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_5D_45, static_coeff_2=static_coeff_MUS_5D_85, static_coeff_3=static_coeff_MUS_5D_10, threshold=0.019, threshold_low=[0.05,0.025,0.005],threshold_med = [0.04,0.03,0.02],threshold_high=[0.02,0.04,0.01],single=False)


plot_static_coeff_summary(static_coeff_MUS_5D_45_filtered, section_name, 4.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_5D_85_filtered, section_name, 8.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_5D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% Save all experiments to excel
section_name = "5D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_5D_55.to_excel(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_45.to_excel(section_name, sheet_name='MUS - 4.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_5D_85.to_excel(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_85.to_excel(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_5D_10.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_10.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "5D_mean"

# Low wind speed
static_coeff_MDS_5D_55.to_excel_mean(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_45.to_excel_mean(section_name, sheet_name='MUS - 4.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_5D_85.to_excel_mean(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_85.to_excel_mean(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_5D_10.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_10.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% Save all experiments to excel filtered
section_name = "5D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_5D_55_filtered.to_excel(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_45_filtered.to_excel(section_name, sheet_name='MUS - 4.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_5D_85_filtered.to_excel(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_85_filtered.to_excel(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_5D_10_filtered.to_excel(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_10_filtered.to_excel(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "5D_mean_filtered"

# Low wind speed
static_coeff_MDS_5D_55_filtered.to_excel_mean(section_name, sheet_name="MDS - 5.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_45_filtered.to_excel_mean(section_name, sheet_name='MUS - 4.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_5D_85_filtered.to_excel_mean(section_name, sheet_name="MDS - 8.5 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_85_filtered.to_excel_mean(section_name, sheet_name='MUS - 8.5 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_5D_10_filtered.to_excel_mean(section_name, sheet_name="MDS - 10 m/s" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9 m/s' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "5D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments (MUS vs MDS vs Single) filtered
section_name = "5D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 4.5 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 4.5 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 4.5 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 4.5 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 4.5 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 4.5 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_5D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MDS_5D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_5D_85,  upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_85,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_85,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_5D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_5D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()

#%% Compare all experiments - only with single deck filtered

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_85_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_85_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_85_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()
# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_5D_45,
                               static_coeff_MUS_5D_85, static_coeff_MUS_5D_10,
                             scoff = "drag")                        
plt.gcf().suptitle(f"5D: MUS ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                               static_coeff_MDS_5D_85, static_coeff_MDS_5D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"5D: MDS ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_5D_45,
                               static_coeff_MUS_5D_85, static_coeff_MUS_5D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9,static_coeff_MDS_5D_55,
                               static_coeff_MDS_5D_85, static_coeff_MDS_5D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_5D_45,
                               static_coeff_MUS_5D_85, static_coeff_MUS_5D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                               static_coeff_MDS_5D_85, static_coeff_MDS_5D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_5D_45,
                               static_coeff_MUS_5D_85, static_coeff_MUS_5D_10,
                           scoff = "drag")                        
plt.gcf().suptitle(f"5D: MUS ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                               static_coeff_MDS_5D_85, static_coeff_MDS_5D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_5D_45,
                               static_coeff_MUS_5D_85, static_coeff_MUS_5D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                               static_coeff_MDS_5D_85, static_coeff_MDS_5D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"5D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MUS_5D_45,
                               static_coeff_MUS_5D_85, static_coeff_MUS_5D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, static_coeff_single_9,
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                               static_coeff_MDS_5D_85, static_coeff_MDS_5D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MDS ",  y=1.05)


# %% Compare all experiments (Wind speed) filtered
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45_filtered,
                               static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55_filtered,
                               static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered,
                                scoff = "drag")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45_filtered,
                               static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered,
                            scoff = "lift")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55_filtered,
                               static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered,
                               scoff = "lift")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45_filtered,
                               static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55_filtered,
                               static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MDS ",  y=1.05)

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45_filtered,
                               static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=1.05)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55_filtered,
                               static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered,
                              scoff = "drag")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45_filtered,
                               static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"5D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55_filtered,
                               static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered,
                              scoff = "lift")                        
plt.gcf().suptitle(f"5D: MDS ",  y=1.05)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45_filtered,
                               static_coeff_MUS_5D_85_filtered, static_coeff_MUS_5D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MUS ",  y=1.05)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, static_coeff_single_9_filtered,
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55_filtered,
                               static_coeff_MDS_5D_85_filtered, static_coeff_MDS_5D_10_filtered,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=1.05)

##########################################################################333

static_coeff_list = [static_coeff_1D, static_coeff_2D, static_coeff_3D, static_coeff_4D]
titles = ["1D", "2D", "3D", "4D"]

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharex=True, sharey=True)

for i, static_coeff in enumerate(static_coeff_list):
    static_coeff.plot_drag(mode="decks", upwind_in_rig=True, ax=axs[i])
    axs[i].set_title(f"Gap {titles[i]}")

fig.suptitle("Drag coefficient comparison", fontsize=18)
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # juster plass for suptitle
plt.savefig("overview_drag_subplots.png", dpi=300)
plt.show()