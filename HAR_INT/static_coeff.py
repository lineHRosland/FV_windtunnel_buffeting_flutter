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

def load_experiments_from_hdf5(h5_input_path, section_name, file_names,  upwind_in_rig=True):
    h5_file = os.path.join(h5_input_path, section_name)
    f = h5py.File(h5_file + ".hdf5", "r")
    exp0 = w3t.Experiment.fromWTT(f[file_names[0]])
    exp1 = w3t.Experiment.fromWTT(f[file_names[1]])

    if upwind_in_rig:
        section_name = "MUS"
    else: section_name = "MDS"
    
    #exp0.plot_experiment() #Before filtering
    #plt.gcf().suptitle(f"{section_name} 0 ms – Before filtering", fontsize=16)
    #exp1.plot_experiment() #Before filtering
    #plt.gcf().suptitle(f"{section_name} {wind_speed} ms – Before filtering", fontsize=16)

    return exp0, exp1

def generate_static_coeff_from_experiments(exp0, exp1, section_width, section_height, section_length_in_rig, section_length_on_wall, upwind_in_rig=True):
    exp0.filt_forces(6, 2)
    exp1.filt_forces(6, 2)

    if upwind_in_rig:
        section_name = "MUS"
    else: section_name = "MDS"

    #exp0_single.plot_experiment() #After filtering
    #plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
    #exp1_single_low.plot_experiment() #After filtering
    #plt.gcf().suptitle(f"{section_name} 6 ms – After filtering", fontsize=16)
    #exp1_single_high.plot_experiment() #After filtering
    #plt.gcf().suptitle(f"{section_name} 9 ms – After filtering", fontsize=16)
    #plt.show()
    return w3t.StaticCoeff.fromWTT(exp0, exp1, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=upwind_in_rig)

def plot_static_coeff_summary(static_coeff, section_name, wind_speed, mode="decks", upwind_in_rig=True):
    if "MUS" in section_name:
        section_name = "MDS"
    elif "MDS" in section_name:
        section_name = "MUS"

    static_coeff.plot_drag(mode=mode,upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name}, {wind_speed} m/s", fontsize=16)
    static_coeff.plot_lift(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name}, {wind_speed} m/s", fontsize=16)
    static_coeff.plot_pitch(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name}, {wind_speed} m/s", fontsize=16)

    #mean
    static_coeff.plot_drag_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name}, {wind_speed} m/s", fontsize=16)
    static_coeff.plot_lift_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name}, {wind_speed} m/s", fontsize=16)
    static_coeff.plot_pitch_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    plt.gcf().suptitle(f"{section_name}, {wind_speed} m/s", fontsize=16)

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
file_names_low = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_005"] # 6 ms
file_names_high = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_004"] # 9 ms, Vibrations

exp0_single, exp1_single_low = load_experiments_from_hdf5(h5_input_path, section_name, file_names_low,  upwind_in_rig=True)
exp0_single, exp1_single_high= load_experiments_from_hdf5(h5_input_path, section_name, file_names_high,  upwind_in_rig=True)

static_coeff_single_low = generate_static_coeff_from_experiments(exp0_single, exp1_single_low, section_width, section_height, section_length_in_rig, section_length_on_wall, upwind_in_rig=True)
static_coeff_single_high =  generate_static_coeff_from_experiments(exp0_single, exp1_single_high, section_width, section_height, section_length_in_rig, section_length_on_wall, upwind_in_rig=True)




#%% Filter and plot ALT 1
#drag
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_low, threshold=0.07, scoff="drag", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.07, scoff="drag")
plt.suptitle(f"{section_name}, 6 m/s", fontsize=16, y=1.05)
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_high, threshold=0.019, scoff="drag", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.019, scoff="drag")
plt.suptitle(f"{section_name}, 9 m/s", fontsize=16, y=1.05)

#lift
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_low, threshold=0.05, scoff="lift", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"{section_name}, 6 m/s", fontsize=16, y=1.05)
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_high, threshold=0.05, scoff="lift", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"{section_name}, 9 m/s", fontsize=16, y=1.05)

#pitch
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_low, threshold=0.005, scoff="pitch", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.005, scoff="pitch")
plt.suptitle(f"{section_name}, 6 m/s", fontsize=16, y=1.05)
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_high, threshold=0.01, scoff="pitch", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.01, scoff="pitch")
plt.suptitle(f"{section_name}, 9 m/s", fontsize=16, y=1.05)

#%%  Filter and plot ALT 2

static_coeff_single_low_filtered, static_coeff_single_high_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_single_low,static_coeff_2=static_coeff_single_high,threshold=0.05,  threshold_low=[0.07, 0.05,0.005], threshold_high=[0.019,0.05,0.01],single=True)
idx_inconsistent = np.where(np.isnan(static_coeff_1_filt.drag_coeff[:,0]) != np.isnan(static_coeff_1_filt.drag_coeff[:,1]))[0]
print("Inconsistent NaN filtering between drag_coeff[:,0] and [:,1]:", idx_inconsistent)

plot_static_coeff_summary(static_coeff_single_low_filtered, section_name, 6, mode="single", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_single_high_filtered, section_name, 9, mode="single", upwind_in_rig=True)



############################################################################################################

#print("1D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_1D_Static"
file_names_low = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_001"] #6 ms, vibrations (Ser OK ut)
file_names_med = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_002"] # 8 ms, vibrations
file_names_high = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_003"] # 10 ms


exp0_down, exp1_down_low, static_coeff_down_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks",  wind_speed = 6, upwind_in_rig=False)
exp0_down, exp1_down_med, static_coeff_down_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8, upwind_in_rig=False)
exp0_down, exp1_down_high, static_coeff_down_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=False)


#%% Plot all downwind experiments 

exp0_down.plot_experiment() #After filtering
plt.gcf().suptitle(f"MDS_1D_Static: 0 ms – After filtering", fontsize=16)
exp1_down_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"MDS_1D_Static: 6 ms – After filtering", fontsize=16)
exp1_down_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"MDS_1D_Static: 8 ms – After filtering", fontsize=16)
exp1_down_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"MDS_1D_Static: 10 ms – After filtering", fontsize=16)
plt.show()

#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_down_low, threshold=0.06, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, setUp_type="MDS", threshold=0.06, scoff="drag")
plt.suptitle(f"MDS_1D_Static, 6 m/s", fontsize=16, y=1.08)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_down_med, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, setUp_type="MDS", threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_1D_Static, 8 m/s", fontsize=16, y=1.08)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_down_high, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, setUp_type="MDS", threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_1D_Static, 10 m/s", fontsize=16, y=1.08)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_down_low, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, setUp_type="MDS", threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_1D_Static, 6 m/s", fontsize=16, y=1.08)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_down_med, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, setUp_type="MDS", threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_1D_Static, 8 m/s", fontsize=16, y=1.08)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_down_high, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, setUp_type="MDS", threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_1D_Static, 10 m/s", fontsize=16, y=1.08)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_down_low, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, setUp_type="MDS", threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_1D_Static, 6 m/s", fontsize=16, y=1.08)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_down_med, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, setUp_type="MDS", threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_1D_Static, 8 m/s", fontsize=16, y=1.08)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_down_high, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, setUp_type="MDS", threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_1D_Static, 10 m/s", fontsize=16, y=1.08)



#%%  Filter and plot ALT 2
#drag
static_coeff_1_filtered, static_coeff_2_filtered, static_coeff_3_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_down_low, static_coeff_2=static_coeff_down_med, static_coeff_3=static_coeff_down_high, threshold=0.1, scoff="drag",threshold_low=0.07,threshold_med = 0.05,threshold_high=0.03,single=False)


#lift
static_coeff_1_filtered, static_coeff_2_filtered, static_coeff_3_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_down_low, static_coeff_2=static_coeff_down_med, static_coeff_3=static_coeff_down_high, threshold=0.1, scoff="lift",threshold_low=0.07,threshold_med = 0.05,threshold_high=0.03,single=False)


#pitch
static_coeff_1_filtered, static_coeff_2_filtered, static_coeff_3_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_down_low, static_coeff_2=static_coeff_down_med, static_coeff_3=static_coeff_down_high, threshold=0.1, scoff="pitch",threshold_low=0.07,threshold_med = 0.05,threshold_high=0.03,single=False)






#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_1D_Static"
file_names_low = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_001"] # 5 ms
file_names_med = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_002"] # 8 ms, vibrations
file_names_high = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_003"] # 10 ms, vibrations

exp0_up, exp1_up_low, static_coeff_up_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks", wind_speed = 5, upwind_in_rig=True)
exp0_up, exp1_up_med, static_coeff_up_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8, upwind_in_rig=True)
exp0_up, exp1_up_high, static_coeff_up_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=True)

#%% Plot all upwind experiments

exp0_up.plot_experiment() #After filtering
plt.gcf().suptitle(f"MUS_1D_Static: 0 ms – After filtering", fontsize=16)
exp1_up_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"MUS_1D_Static: 5 ms – After filtering", fontsize=16)
exp1_up_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"MUS_1D_Static: 8 ms – After filtering", fontsize=16)
exp1_up_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"MUS_1D_Static: 10 ms – After filtering", fontsize=16)
plt.show()

#%% Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_up_low, threshold=0.06, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, setUp_type="MUS", threshold=0.06, scoff="drag")
plt.suptitle(f"MUS_1D_Static, 5 m/s", fontsize=16, y=1.08)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_up_med, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, setUp_type="MUS", threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_1D_Static, 8 m/s", fontsize=16, y=1.08)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_up_high, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, setUp_type="MUS", threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_1D_Static, 10 m/s", fontsize=16, y=1.08)

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_up_low, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, setUp_type="MUS", threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_1D_Static, 5 m/s", fontsize=16, y=1.08)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_up_med, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, setUp_type="MUS", threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_1D_Static, 8 m/s", fontsize=16, y=1.08)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_up_high, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, setUp_type="MUS", threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_1D_Static, 10 m/s", fontsize=16, y=1.08)

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_up_low, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, setUp_type="MUS", threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_1D_Static, 5 m/s", fontsize=16, y=1.08)
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_up_med, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, setUp_type="MUS", threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_1D_Static, 8 m/s", fontsize=16, y=1.08)
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_up_high, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, setUp_type="MUS", threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_1D_Static, 10 m/s", fontsize=16, y=1.08)


#%%  Filter and plot ALT 2
#drag
static_coeff_1_filtered, static_coeff_2_filtered, static_coeff_3_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_up_low,static_coeff_2= static_coeff_up_med,static_coeff_3= static_coeff_up_high, threshold=0.1, scoff="drag",threshold_low=0.07,threshold_med = 0.05,threshold_high=0.03,single=False)

#lift
static_coeff_1_filtered, static_coeff_2_filtered, static_coeff_3_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_up_low, static_coeff_2=static_coeff_up_med,static_coeff_3= static_coeff_up_high, threshold=0.1, scoff="lift",threshold_low=0.07,threshold_med = 0.05,threshold_high=0.03,single=False)

#pitch
static_coeff_1_filtered, static_coeff_2_filtered, static_coeff_3_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_up_low, static_coeff_2=static_coeff_up_med, static_coeff_3=static_coeff_up_high, threshold=0.1, scoff="pitch",threshold_low=0.07,threshold_med = 0.05,threshold_high=0.03,single=False)

#%% Save all experiments to excel
section_name = "1D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_down_low.to_excel(section_name, sheet_name="MDS - 6 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel(section_name, sheet_name='MUS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel(section_name, sheet_name="MDS - 8 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel(section_name, sheet_name='MUS - 8 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel(section_name, sheet_name="MDS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel(section_name, sheet_name='MUS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "1D_mean"

# Low wind speed
static_coeff_down_low.to_excel_mean(section_name, sheet_name="MDS - 6 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel_mean(section_name, sheet_name='MUS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel_mean(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel_mean(section_name, sheet_name="MDS - 8 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel_mean(section_name, sheet_name='MUS - 8 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel_mean(section_name, sheet_name="MDS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel_mean(section_name, sheet_name='MUS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "1D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_up_low, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_down_low, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_up_low, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_down_low, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_up_low, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_down_low, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_up_low, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_down_low,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_up_low,  setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_down_low,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_up_low,  setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 6 m/s, MUS: 5 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_down_low,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 6 m/s, MDS: 6 m/s", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_med, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_med,   setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_med,  setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_med,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_med,  setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_med,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_med, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_med,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_med,  setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_med, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_med,  setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 8 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_med,  setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 8 m/s", fontsize=16)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_high, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_high, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_high, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_high, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_high, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_high, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_high, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_high, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_high, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_high, setUp_type ="MDS")
plt.gcf().suptitle(f"Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_high, setUp_type ="MUS")
plt.gcf().suptitle(f"Single: 9 m/s, MUS: 10 m/s", fontsize=16)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_high, setUp_type ="MDS")
plt.figtext(0.5, -0.08,f"Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.show()

# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                             scoff = "drag")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                                scoff = "drag")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                            scoff = "lift")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                               scoff = "lift")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)

#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                           scoff = "drag")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                              scoff = "drag")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                                scoff = "lift")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                                scoff = "lift")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_single_high, static_coeff_up_low,
                               static_coeff_up_med, static_coeff_up_high,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"{section_name} ", fontsize=16)




############################################################################################################

#print("2D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_2D_Static"
file_names_low = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_002"] #6 ms
file_names_med = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_003"] # 8 ms, vibrations
file_names_high = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_004"] # 10 ms


exp0_down, exp1_down_low, static_coeff_down_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks",  wind_speed = 6, upwind_in_rig=False)
exp0_down, exp1_down_med, static_coeff_down_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8, upwind_in_rig=False)
exp0_down, exp1_down_high, static_coeff_down_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=False)


#%% Plot all downwind experiments 

exp0_down.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_down_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 6 ms – After filtering", fontsize=16)
exp1_down_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8 ms – After filtering", fontsize=16)
exp1_down_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()


#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_2D_Static"
file_names_low = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_001"] # 5 ms, vibrations
file_names_med = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_002"] # 8 ms, vibrations
file_names_high = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_003"] # 10 ms, vibrations

exp0_up, exp1_up_low, static_coeff_up_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks", wind_speed = 5, upwind_in_rig=True)
exp0_up, exp1_up_med, static_coeff_up_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8, upwind_in_rig=True)
exp0_up, exp1_up_high, static_coeff_up_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=True)

#%% Plot all upwind experiments

exp0_up.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_up_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 5 ms – After filtering", fontsize=16)
exp1_up_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8 ms – After filtering", fontsize=16)
exp1_up_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()



#%% Save all experiments to excel
section_name = "2D"

# Low wind speed
static_coeff_down_low.to_excel(section_name, sheet_name="MUS - 6 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel(section_name, sheet_name='MDS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel(section_name, sheet_name="MUS - 8 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel(section_name, sheet_name='MDS - 8 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "2D_mean"

# Low wind speed
static_coeff_down_low.to_excel_mean(section_name, sheet_name="MUS - 6 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel_mean(section_name, sheet_name='MDS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel_mean(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel_mean(section_name, sheet_name="MUS - 8 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel_mean(section_name, sheet_name='MDS - 8 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel_mean(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel_mean(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "2D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Low wind speed", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Low wind speed", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Low wind speed", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Low wind speed", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Low wind speed", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}: Low wind speed", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Medium wind speed", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Medium wind speed", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Medium wind speed", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Medium wind speed", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Medium wind speed", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}: Medium wind speed", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: High wind speed", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: High wind speed", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: High wind speed", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: High wind speed", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: High wind speed", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}: High wind speed", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_down_low)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_down_low)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_med)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_med)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_high)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_high)
plt.show()

# %% Compare all experiments (Wind speed)
w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_up_low, static_coeff_up_high,
                               static_coeff_down_low, static_coeff_down_high,
                               label_up="Upwind deck in rig", label_down="Downwind deck on wall",
                               section_name="2D - Plottype 1")

w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_down_low, static_coeff_down_high,
                               static_coeff_up_low, static_coeff_up_high,
                               label_up="Downwind deck in rig", label_down="Upwind deck on wall",
                               section_name="2D - Plottype 2")



############################################################################################################
#%%
print("3D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_3D_Static"
file_names_low = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_001"] #6 ms
file_names_med = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_002"] # 8 ms, vibrations
file_names_high = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_003"] # 10 ms


exp0_down, exp1_down_low, static_coeff_down_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks",  wind_speed = 6, upwind_in_rig=False)
exp0_down, exp1_down_med, static_coeff_down_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8, upwind_in_rig=False)
exp0_down, exp1_down_high, static_coeff_down_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=False)


#%% Plot all downwind experiments 

exp0_down.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_down_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 6 ms – After filtering", fontsize=16)
exp1_down_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8 ms – After filtering", fontsize=16)
exp1_down_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()


#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_3D_Static"
file_names_low = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_004"] # 5 ms, vibrations (Finnes en fil for 6 også)
file_names_med = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_006"] # 8 ms, vibrations
file_names_high = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_005"] # 10 ms, vibrations

exp0_up, exp1_up_low, static_coeff_up_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks", wind_speed = 5, upwind_in_rig=True)
exp0_up, exp1_up_med, static_coeff_up_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8, upwind_in_rig=True)
exp0_up, exp1_up_high, static_coeff_up_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=True)

#%% Plot all upwind experiments

exp0_up.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_up_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 5 ms – After filtering", fontsize=16)
exp1_up_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8 ms – After filtering", fontsize=16)
exp1_up_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()



#%% Save all experiments to excel
section_name = "3D"

# Low wind speed
static_coeff_down_low.to_excel(section_name, sheet_name="MUS - 6 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel(section_name, sheet_name='MDS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel(section_name, sheet_name="MUS - 8 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel(section_name, sheet_name='MDS - 8 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "3D_mean"

# Low wind speed
static_coeff_down_low.to_excel_mean(section_name, sheet_name="MUS - 6 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel_mean(section_name, sheet_name='MDS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel_mean(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel_mean(section_name, sheet_name="MUS - 8 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel_mean(section_name, sheet_name='MDS - 8 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel_mean(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel_mean(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "3D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_up_low) # Upwind in rig  
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_down_low) # Downwind in rig
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_up_low) # Upwind in rig
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_down_low) # Downwind in rig
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_up_low) # Upwind in rig
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_down_low) # Downwind in rig

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_up_low) # Upwind in rig
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_down_low) # Downwind in rig
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_up_low) # Upwind in rig
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_down_low) # Downwind in rig
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_up_low) # Upwind in rig
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_down_low) # Downwind in rig

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_med) # Upwind in rig
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_med) # Downwind in rig
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_med) # Upwind in rig
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_med) # Downwind in rig
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_med) # Upwind in rig
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_med)  # Downwind in rig

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_med) # Upwind in rig
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_med) # Downwind in rig
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_med) # Upwind in rig
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_med) # Downwind in rig
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_med) # Upwind in rig
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_med) # Downwind in rig
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_high) # Upwind in rig
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_high) # Downwind in rig
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_high) # Upwind in rig
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_high) # Downwind in rig
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_high) # Upwind in rig
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_high) # Downwind in rig

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_high) # Upwind in rig
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_high) # Downwind in rig
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_high) # Upwind in rig
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_high) # Downwind in rig
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_high) # Upwind in rig
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_high) # Downwind in rig
plt.show()

# %% Compare all experiments (Wind speed)
w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_up_low, static_coeff_up_high,
                               static_coeff_down_low, static_coeff_down_high,
                               label_up="Upwind deck in rig", label_down="Downwind deck on wall",
                               section_name="2D - Plottype 1")

w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_down_low, static_coeff_down_high,
                               static_coeff_up_low, static_coeff_up_high,
                               label_up="Downwind deck in rig", label_down="Upwind deck on wall",
                               section_name="2D - Plottype 2")
############################################################################################################
#%%
print("4D")

#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_4D_Static"
file_names_low = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_002"] #5.5 ms
file_names_med = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_003"] # 8.5 ms, vibrations
file_names_high = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_004"] # 10 ms


exp0_down, exp1_down_low, static_coeff_down_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks",  wind_speed = 5.5, upwind_in_rig=False)
exp0_down, exp1_down_med, static_coeff_down_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8.5, upwind_in_rig=False)
exp0_down, exp1_down_high, static_coeff_down_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=False)


#%% Plot all downwind experiments 

exp0_down.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_down_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 5.5 ms – After filtering", fontsize=16)
exp1_down_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8.5 ms – After filtering", fontsize=16)
exp1_down_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()


#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_4D_Static"
file_names_low = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_003"] # 5 ms, vibrations 
file_names_med = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_004"] # 8.5 ms, vibrations
file_names_high = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_005"] # 10 ms, vibrations

exp0_up, exp1_up_low, static_coeff_up_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks", wind_speed = 5, upwind_in_rig=True)
exp0_up, exp1_up_med, static_coeff_up_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8.5, upwind_in_rig=True)
exp0_up, exp1_up_high, static_coeff_up_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=True)

#%% Plot all upwind experiments

exp0_up.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_up_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 5 ms – After filtering", fontsize=16)
exp1_up_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8.5 ms – After filtering", fontsize=16)
exp1_up_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()



#%% Save all experiments to excel
section_name = "4D"

# Low wind speed
static_coeff_down_low.to_excel(section_name, sheet_name="MUS - 5.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel(section_name, sheet_name='MDS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel(section_name, sheet_name="MUS - 8.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel(section_name, sheet_name='MDS - 8.5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "4D_mean"

# Low wind speed
static_coeff_down_low.to_excel_mean(section_name, sheet_name="MUS - 5.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel_mean(section_name, sheet_name='MDS - 5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel_mean(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel_mean(section_name, sheet_name="MUS - 8.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel_mean(section_name, sheet_name='MDS - 8.5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel_mean(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel_mean(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "4D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_down_low)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_down_low)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_med)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_med)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_high)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_high)
plt.show()

# %% Compare all experiments (Wind speed)
w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_up_low, static_coeff_up_high,
                               static_coeff_down_low, static_coeff_down_high,
                               label_up="Upwind deck in rig", label_down="Downwind deck on wall",
                               section_name="2D - Plottype 1")

w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_down_low, static_coeff_down_high,
                               static_coeff_up_low, static_coeff_up_high,
                               label_up="Downwind deck in rig", label_down="Upwind deck on wall",
                               section_name="2D - Plottype 2")
############################################################################################################
#%%
print("5D")


#%% Load all downwind experiments (downwind in rig)
section_name = "MUS_5D_Static"
file_names_low = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_001"] # 5.5 ms
file_names_med = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_002"] # 8.5 ms, vibrations
file_names_high = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_003"] # 10 ms


exp0_down, exp1_down_low, static_coeff_down_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks",  wind_speed = 5.5, upwind_in_rig=False)
exp0_down, exp1_down_med, static_coeff_down_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8.5, upwind_in_rig=False)
exp0_down, exp1_down_high, static_coeff_down_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=False)


#%% Plot all downwind experiments 

exp0_down.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_down_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 5.5 ms – After filtering", fontsize=16)
exp1_down_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8.5 ms – After filtering", fontsize=16)
exp1_down_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()


#%% Load all upwind experiments (upwind in rig)

section_name = "MDS_5D_Static"
file_names_low = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_002"] # 4.5 ms, vibrations 
file_names_med = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_003"] # 8.5 ms, vibrations
file_names_high = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_004"] # 10 ms, vibrations

exp0_up, exp1_up_low, static_coeff_up_low = load_and_process_static_coeff(h5_input_path, section_name, file_names_low, mode="decks", wind_speed = 5, upwind_in_rig=True)
exp0_up, exp1_up_med, static_coeff_up_med = load_and_process_static_coeff(h5_input_path, section_name, file_names_med, mode="decks", wind_speed = 8.5, upwind_in_rig=True)
exp0_up, exp1_up_high, static_coeff_up_high = load_and_process_static_coeff(h5_input_path, section_name, file_names_high, mode="decks", wind_speed = 10, upwind_in_rig=True)

#%% Plot all upwind experiments

exp0_up.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 0 ms – After filtering", fontsize=16)
exp1_up_low.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 4.5 ms – After filtering", fontsize=16)
exp1_up_med.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 8.5 ms – After filtering", fontsize=16)
exp1_up_high.plot_experiment() #After filtering
plt.gcf().suptitle(f"{section_name} 10 ms – After filtering", fontsize=16)
plt.show()



#%% Save all experiments to excel
section_name = "5D"

# Low wind speed
static_coeff_down_low.to_excel(section_name, sheet_name="MUS - 5.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel(section_name, sheet_name='MDS - 4.5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel(section_name, sheet_name="MUS - 8.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel(section_name, sheet_name='MDS - 8.5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "5D_mean"

# Low wind speed
static_coeff_down_low.to_excel_mean(section_name, sheet_name="MUS - 5.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_low.to_excel_mean(section_name, sheet_name='MDS - 4.5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_low.to_excel_mean(section_name, sheet_name='Single - 6 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_down_med.to_excel_mean(section_name, sheet_name="MUS - 8.5 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_med.to_excel_mean(section_name, sheet_name='MDS - 8.5 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_down_high.to_excel_mean(section_name, sheet_name="MUS - 10 ms" ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_up_high.to_excel_mean(section_name, sheet_name='MDS - 10 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_high.to_excel_mean(section_name, sheet_name='Single - 9 ms' ,section_width=18.3/100,section_height=3.33/100,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% Compare all experiments (MUS vs MDS vs Single)
section_name = "5D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_low, static_coeff_down_low)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

#Medium wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_med, static_coeff_down_med)
plt.gcf().suptitle(f"{section_name}", fontsize=16)


#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_lift_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_low, static_coeff_up_high, static_coeff_down_high)
plt.gcf().suptitle(f"{section_name}", fontsize=16)

plt.show()

#%% Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_low, static_coeff_down_low)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_low, static_coeff_down_low)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_up_low)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_low, static_coeff_down_low)

#Medium wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_med)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_med)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_med)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_med)
                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_high, static_coeff_down_high)

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_high, static_coeff_down_high)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_up_high)
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_high, static_coeff_down_high)

plt.show()

# %% Compare all experiments (Wind speed)
w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_up_low, static_coeff_up_high,
                               static_coeff_down_low, static_coeff_down_high,
                               label_up="Upwind deck in rig", label_down="Downwind deck on wall",
                               section_name="2D - Plottype 1")

w3t._scoff.plot_compare_drag_wind_speeds(static_coeff_single_low, static_coeff_single_high,
                               static_coeff_down_low, static_coeff_down_high,
                               static_coeff_up_low, static_coeff_up_high,
                               label_up="Downwind deck in rig", label_down="Upwind deck on wall",
                               section_name="2D - Plottype 2")
############################################################################################################




