# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:47:47 2022
Edited 2025
@author: oiseth, linehro
"""
#%% !!!
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
from matplotlib import rc
import copy

# # Computer Modern Roman without latex
# mpl.rcParams['text.usetex'] == True  

# # # Bruk fonten som ligner mest på Computer Modern
# # mpl.rcParams['font.family'] == 'serif'
# # mpl.rcParams['font.serif'] == ['cmr10', 'Computer Modern Roman', 'Times New Roman']
# # mpl.rcParams['mathtext.fontset'] == 'cm' 

# # Generelt større og mer lesbar tekst
# mpl.rcParams.update({
#     'font.size': 12,              # Generell tekststørrelse
#     'axes.labelsize': 12,         # Aksetitler
#     'axes.titlesize': 14,         # Plot-titler
#     'legend.fontsize': 12,        # Tekst i legend
#     'xtick.labelsize': 12,        # X-tick labels
#     'ytick.labelsize': 12         # Y-tick labels
# })


from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']

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

    # fig_d.savefig("AD_5D_C.png", dpi=300, bbox_inches='tight')
    # fig_s.savefig("AD_5D_K.png", dpi=300, bbox_inches='tight')

    static_coeff.plot_drag(mode=mode,upwind_in_rig=upwind_in_rig)
    #plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"_{wind_speed}" + "_drag.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    static_coeff.plot_lift(mode=mode, upwind_in_rig=upwind_in_rig)
    #plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", section_name + f"_{wind_speed}" + "_lift.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    static_coeff.plot_pitch(mode=mode, upwind_in_rig=upwind_in_rig)
    #plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", section_name + f"_{wind_speed}" + "_pitch.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)


    # fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    # fig.suptitle(f"{section_name} - {wind_speed} m/s")
    # # Plot drag, lift, pitch i de tre aksene
    # static_coeff.plot_drag(mode=mode, upwind_in_rig=upwind_in_rig, ax=axs[0])
    # axs[0].set_title("Drag")
    # static_coeff.plot_lift(mode=mode, upwind_in_rig=upwind_in_rig, ax=axs[1])
    # axs[1].set_title("Lift")
    # static_coeff.plot_pitch(mode=mode, upwind_in_rig=upwind_in_rig, ax=axs[2])
    # axs[2].set_title("Pitch")
    # axs[0].set_xticks([-8, -6, -4, -2,0, 2,4, 6, 8])
    # axs[0].set_xlabel(r"$\alpha$")
    # axs[0].set_ylabel(r"$C_D(\alpha)$")
    # axs[0].set_ylim(ymin=0.4,ymax=0.8)
    # axs[0].grid()
    # axs[0].tick_params(labelsize=12)
    # axs[0].legend(fontsize=18)
    # axs[1].set_xticks([-8, -6, -4, -2,0, 2, 4, 6, 8])
    # axs[1].set_xlabel(r"$\alpha$")
    # axs[1].set_ylabel(r"$C_L(\alpha)$")
    # axs[1].set_ylim(ymin=-0.7,ymax=0.7)
    # axs[1].grid()
    # axs[1].tick_params(labelsize=12)
    # axs[1].legend(fontsize=18)
    # axs[2].set_xticks([-8, -6, -4, -2,0, 2, 4, 6, 8])
    # axs[2].set_xlabel(r"$\alpha$")
    # axs[2].set_ylabel(r"$C_M(\alpha)$")
    # axs[2].set_ylim(ymin=-0.15,ymax=0.2)
    # axs[2].grid()
    # axs[2].tick_params(labelsize=12)
    # axs[2].legend(fontsize=18) 
    # # Layout og lagring
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # La plass til suptitle
    # plt.show()
    # plt.close()

    #mean
    static_coeff.plot_drag_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    #plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", section_name + f"_{wind_speed}_mean" + "_drag.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    static_coeff.plot_lift_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    #plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", section_name + f"_{wind_speed}_mean" + "_lift.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    static_coeff.plot_pitch_mean(mode=mode, upwind_in_rig=upwind_in_rig)
    #plt.gcf().suptitle(f"{section_name} - {wind_speed} m/s")
    plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", section_name + f"_{wind_speed}_mean" + "_pitch.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.tight_layout()
    plt.show()


def fix(filtered, unfiltered):
    """
    Replace NaNs and their immediate neighbors in a filtered array with values from an unfiltered array.
 
    Parameters:
    - filtered: The array containing NaN values.
    - unfiltered: The original array with all valid values.
 
    Returns:
    - A new array where NaNs and their neighbors are replaced with values from the unfiltered array.
    """
    # if filtered.shape != unfiltered.shape:
    #     raise ValueError("Filtered and unfiltered arrays must have the same shape.")
 
    result = filtered.copy()
    nan_indices = np.where(np.isnan(filtered))[0]
    indices_to_replace = set()
 
    for idx in nan_indices:
        indices_to_replace.add(idx)
        if idx > 0:
            indices_to_replace.add(idx - 1)
        if idx < len(filtered) - 1:
            indices_to_replace.add(idx + 1)
 
    for idx in indices_to_replace:
        result[idx] = unfiltered[idx]
 
    return result

# Load all experiments
tic = time.perf_counter()
plt.close("all")


section_height = 0.066 #m
section_width =  0.366 #m
section_length_in_rig = 2.68 #m
section_length_on_wall = 2.66 #m

h5_input_path = r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Python\Ole_sin_kode\HAR_INT\H5F\\"

#%% !!!
# Load single deck
section_name = "Single_Static"
file_names_6 = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_005"] # 6 m/s
file_names_9 = ["HAR_INT_SINGLE_02_00_003","HAR_INT_SINGLE_02_00_004"] # 9 m/s, Vibrations

exp0_single, exp1_single_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_6,  upwind_in_rig=True)
exp0_single, exp1_single_9= load_experiments_from_hdf5(h5_input_path, section_name, file_names_9,  upwind_in_rig=True)

# exp0_single.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"Single deck - Wind speed: 0 m/s",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", section_name + "_0" + ".png"))

# exp1_single_6.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"Single deck - Wind speed: 6 m/s",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", section_name + "_6" + ".png"))

# exp1_single_9.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"Single deck - Wind speed: 9 m/s",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", section_name + "_9" + ".png"))

exp0_single.filt_forces(6, 2)
exp1_single_6.filt_forces(6, 2)
exp1_single_9.filt_forces(6, 2)

# exp0_single.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"Single deck - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", section_name + "_0_filter" + ".png"))
# exp1_single_6.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"Single deck - Wind speed: 6 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", section_name + "_6_filter" + ".png"))
# exp1_single_9.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"Single deck - Wind speed: 9 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", section_name + "_9_filter" + ".png"))
# plt.show()
# plt.close()

static_coeff_single_6 =w3t.StaticCoeff.fromWTT(exp0_single, exp1_single_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_single_9 = w3t.StaticCoeff.fromWTT(exp0_single, exp1_single_9, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_single_6, section_name, 6, mode="single", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_single_9, section_name, 9, mode="single", upwind_in_rig=True)


# filter regression
print("reg")
section_name = "Single_Static_reg"
static_coeff_single_9_reg = copy.deepcopy(static_coeff_single_9)
static_coeff_single_6_reg = copy.deepcopy(static_coeff_single_6)

alpha_single_reg, coeff_single_drag_reg = w3t._scoff.poly_estimat(static_coeff_single_9, scoff="drag", single=True)
static_coeff_single_9_reg.drag_coeff[:, 0] = coeff_single_drag_reg / 2
static_coeff_single_9_reg.drag_coeff[:, 1] = coeff_single_drag_reg / 2


alpha_single_reg, coeff_single_lift_reg = w3t._scoff.poly_estimat(static_coeff_single_9, scoff="lift", single=True)
static_coeff_single_9_reg.lift_coeff[:, 0] = coeff_single_lift_reg / 2
static_coeff_single_9_reg.lift_coeff[:, 1] = coeff_single_lift_reg / 2


alpha_single_reg, coeff_single_pitch_reg = w3t._scoff.poly_estimat(static_coeff_single_9, scoff="pitch", single=True)
static_coeff_single_9_reg.pitch_coeff[:, 0] = coeff_single_pitch_reg / 2
static_coeff_single_9_reg.pitch_coeff[:, 1] = coeff_single_pitch_reg / 2

plot_static_coeff_summary(static_coeff_single_9_reg, section_name, 9, mode="single", upwind_in_rig=True)


alpha_single_reg, coeff_single_drag_reg = w3t._scoff.poly_estimat(static_coeff_single_6, scoff="drag", single=True)
static_coeff_single_6_reg.drag_coeff[:, 0] = coeff_single_drag_reg / 2
static_coeff_single_6_reg.drag_coeff[:, 1] = coeff_single_drag_reg / 2


alpha_single_reg, coeff_single_lift_reg = w3t._scoff.poly_estimat(static_coeff_single_6, scoff="lift", single=True)
static_coeff_single_6_reg.lift_coeff[:, 0] = coeff_single_lift_reg / 2
static_coeff_single_6_reg.lift_coeff[:, 1] = coeff_single_lift_reg / 2


alpha_single_reg, coeff_single_pitch_reg = w3t._scoff.poly_estimat(static_coeff_single_6, scoff="pitch", single=True)
static_coeff_single_6_reg.pitch_coeff[:, 0] = coeff_single_pitch_reg / 2
static_coeff_single_6_reg.pitch_coeff[:, 1] = coeff_single_pitch_reg / 2

plot_static_coeff_summary(static_coeff_single_6_reg, section_name, 6, mode="single", upwind_in_rig=True)




section_name = "Single_Static"


#%%
#  Filter and plot ALT 1
#drag
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_6, threshold=0.05, scoff="drag", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"Single deck - Wind speed: 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "Single_6_drag_clean.png"))
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_9, threshold=0.05, scoff="drag", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"Single deck - Wind speed: 9 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "Single_9_drag_clean.png"))

#lift
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_6, threshold=0.05, scoff="lift", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"Single deck - Wind speed: 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "Single_6_lift_clean.png"))
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_9, threshold=0.05, scoff="lift", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"Single deck - Wind speed: 9 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "Single_9_lift_clean.png"))

#pitch
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_6, threshold=0.05, scoff="pitch", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"Single deck - Wind speed: 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "Single_6_pitch_clean.png"))
alpha_single, coeff_single_plot=w3t._scoff.filter(static_coeff_single_9, threshold=0.05, scoff="pitch", single = True)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_single,coeff_single_plot,coeff_down_plot=None, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"Single deck - Wind speed: 9 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "Single_9_pitch_clean.png"))

#%%  !!!
# Filter and plot ALT 2
section_name = "Single_Static_filtered"

static_coeff_single_6_filtered, static_coeff_single_9_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_single_6,static_coeff_2=static_coeff_single_9,threshold=0.02,  threshold_low=[0.05, 0.02,0.004], threshold_high=[0.04,0.03,0.005],single=True)

#plot_static_coeff_summary(static_coeff_single_6_filtered, section_name, 6, mode="single", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_single_9_filtered, section_name, 9, mode="single", upwind_in_rig=True)


#%% !!!
# Summary
static_coeff_single_6_updated = copy.deepcopy(static_coeff_single_6)
static_coeff_single_6_updated.drag_coeff[:, 0] = static_coeff_single_6.drag_coeff[:, 0]
static_coeff_single_6_updated.drag_coeff[:, 1] = static_coeff_single_6.drag_coeff[:, 1]
static_coeff_single_6_updated.lift_coeff[:, 0] = static_coeff_single_6.lift_coeff[:, 0]
static_coeff_single_6_updated.lift_coeff[:, 1] = static_coeff_single_6.lift_coeff[:, 1]
static_coeff_single_6_updated.pitch_coeff[:, 0] = static_coeff_single_6.pitch_coeff[:, 0]
static_coeff_single_6_updated.pitch_coeff[:, 1] = static_coeff_single_6.pitch_coeff[:, 1]

static_coeff_single_9_updated = copy.deepcopy(static_coeff_single_9)
static_coeff_single_9_updated.drag_coeff[:, 0] = static_coeff_single_9_filtered.drag_coeff[:, 0]
static_coeff_single_9_updated.drag_coeff[:, 1] = static_coeff_single_9_filtered.drag_coeff[:, 1]
static_coeff_single_9_updated.lift_coeff[:, 0] = static_coeff_single_9_reg.lift_coeff[:, 0]
static_coeff_single_9_updated.lift_coeff[:, 1] = static_coeff_single_9_reg.lift_coeff[:, 1]
static_coeff_single_9_updated.pitch_coeff[:, 0] = static_coeff_single_9_reg.pitch_coeff[:, 0]
static_coeff_single_9_updated.pitch_coeff[:, 1] = static_coeff_single_9_reg.pitch_coeff[:, 1]

cd_6_mean = static_coeff_single_6_updated.plot_drag_mean( mode ="single", upwind_in_rig=True, ax = None)[0]
cd_9_mean = static_coeff_single_9_updated.plot_drag_mean( mode ="single", upwind_in_rig=True,ax  = None)[0]
alpha= static_coeff_single_9_updated.plot_drag_mean( mode ="single", upwind_in_rig=True,ax  = None)[1]

for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_6_mean[i])
        print(cd_9_mean[i])
#%% !?!!!
def plot_compare_wind_speeds_mean_seperate(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_D(\alpha)$"
        coeff = "drag_coeff"
        min = 0.5#0.4
        max = 0.62#0.58
    elif scoff == "lift":
        axis = r"$C_L(\alpha)$"
        coeff = "lift_coeff"
        min = -0.37#-0.35
        max = 0.42#0.5
    elif scoff == "pitch":
        axis = r"$C_M(\alpha)$"
        coeff = "pitch_coeff"
        min = -0.05#-0.05
        max = 0.11#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)


    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"6 m/s", color = "#2ca02c", alpha = 0.8)
    # ax.plot(unique_alphas_low, downwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        ax.plot(unique_alphas_med, upwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        # ax.plot(unique_alphas_med, downwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"9 m/s", color ="#d62728", alpha = 0.8)
    # ax.plot(unique_alphas_high, downwind_mean_high,
    #             label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

#%% !?!!!
# Compare wind speed
section_name = "Single_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate(static_coeff_single_6_updated, static_coeff_single_9_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_single_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate(static_coeff_single_6_updated, static_coeff_single_9_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_single_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate(static_coeff_single_6_updated, static_coeff_single_9_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_single_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)



############################################################################################################

#print("1D")
#%% !!!
# Load all downwind experiments (downwind in rig)
section_name = "MUS_1D_Static"
file_names_MDS_1D_6 = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_001"] #6 m/s, vibrations (Ser OK ut)
file_names_MDS_1D_8 = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_002"] # 8 m/s, vibrations
file_names_MDS_1D_10 = ["HAR_INT_MUS_GAP_213D_02_02_000","HAR_INT_MUS_GAP_213D_02_02_003"] # 10 m/s



exp0_MDS_1D, exp1_MDS_1D_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_1D_6,  upwind_in_rig=False)
exp0_MDS_1D, exp1_MDS_1D_8= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_1D_8,  upwind_in_rig=False)
exp0_MDS_1D, exp1_MDS_1D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_1D_10,  upwind_in_rig=False)


# exp0_MDS_1D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 1D - Wind speed: 0 m/s",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_1D_0" + ".png"))
exp1_MDS_1D_6.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_1D_6_tidserie" + ".png"), dpi=300)
# exp1_MDS_1D_8.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_1D_8" + ".png"))
exp1_MDS_1D_10.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_1D_10_tidserie" + ".png"), dpi=300)

exp0_MDS_1D.filt_forces(6, 2)
exp1_MDS_1D_6.filt_forces(6, 2)
exp1_MDS_1D_8.filt_forces(6, 2)
exp1_MDS_1D_10.filt_forces(6, 2)

# exp0_MDS_1D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 1D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_1D_0_filter" + ".png"))
exp1_MDS_1D_6.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 1D - Wind speed:) 6 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_1D_6_filter_tidserie" + ".png"), dpi=300)
# exp1_MDS_1D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 1D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_1D_8_filter" + ".png"))
exp1_MDS_1D_10.plot_experiment(mode="decks") #With Butterworth low-pass filter
#plt.gcf().suptitle(f"MDS 1D - Wind speed:) 10 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_1D_10_filter_tidserie" + ".png"), dpi=300)
plt.show()
plt.close()


static_coeff_MDS_1D_6 =w3t.StaticCoeff.fromWTT(exp0_MDS_1D, exp1_MDS_1D_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_1D_8 = w3t.StaticCoeff.fromWTT(exp0_MDS_1D, exp1_MDS_1D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_1D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_1D, exp1_MDS_1D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)


plot_static_coeff_summary(static_coeff_MDS_1D_6, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_8, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_10, section_name, 10, mode="decks", upwind_in_rig=False)

#%%

def plot_three_subplots_with_shared_legend(save_path="three_plots.pdf"):
    # Dummy data
    x = np.linspace(-4, 4, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tanh(x)

    # Create figure and axes for 3 subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=False)

    # Labels and colors
    labels = ["Single deck", "Upstream deck", "Downstream deck"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax in axes:
        ax.plot(x, y1, label=labels[0], color=colors[0])
        ax.plot(x, y2, label=labels[1], color=colors[1])
        ax.plot(x, y3, label=labels[2], color=colors[2])
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    fig.legend(labels, loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0))

    # Adjust layout
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(bottom=0.25)

    # Save figure
    return fig, ax

fig, ax = plot_three_subplots_with_shared_legend("three_plots_with_legend.pdf")
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "label_stat_coeff" + ".png"), dpi=300)

#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_1D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_1D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_1D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_1D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_1D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_1D_low_cm" + ".png"), dpi=300)

fig, ax =w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_1D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_1D_high_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_1D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_1D_high_cl" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_1D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_1D_high_cm" + ".png"), dpi=300)


#%%
#  Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_1D_6, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_1D_6_drag_clean.png"))

alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_1D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_1D_8_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_1D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_1D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_1D_6, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_1D_6_lift_clean.png"))
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_1D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_1D_8_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_1D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_1D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_1D_6, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS 1D - Wind speed: 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_1D_6_pitch_clean.png"))
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_1D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS 1D - Wind speed: 8 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_1D_8_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_1D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_1D_10_pitch_clean.png"))




#%%  
# Filter and plot ALT 2
section_name = "MUS_1D_Static_filtered" 

static_coeff_MDS_1D_6_filtered, static_coeff_MDS_1D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_1D_6, static_coeff_2=static_coeff_MDS_1D_10, threshold=0.1, threshold_low=[0.03,0.03,0.005],threshold_high=[0.03,0.03,0.005],single=False)

plot_static_coeff_summary(static_coeff_MDS_1D_6_filtered, section_name, 6, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_1D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=False)

#%% !!!
#Summary

static_coeff_MDS_1D_6_updated = copy.deepcopy(static_coeff_MDS_1D_6)
static_coeff_MDS_1D_6_updated.drag_coeff[:, 2] = static_coeff_MDS_1D_6.drag_coeff[:, 2]
static_coeff_MDS_1D_6_updated.drag_coeff[:, 3] = static_coeff_MDS_1D_6.drag_coeff[:, 3]
static_coeff_MDS_1D_6_updated.lift_coeff[:, 2] = static_coeff_MDS_1D_6.lift_coeff[:, 2]
static_coeff_MDS_1D_6_updated.lift_coeff[:, 3] = static_coeff_MDS_1D_6.lift_coeff[:, 3]
static_coeff_MDS_1D_6_updated.pitch_coeff[:,2] = static_coeff_MDS_1D_6.pitch_coeff[:,2]
static_coeff_MDS_1D_6_updated.pitch_coeff[:, 3] = static_coeff_MDS_1D_6.pitch_coeff[:,3]
print(static_coeff_MDS_1D_6_updated.drag_coeff[:5,2])
static_coeff_MDS_1D_8_updated = copy.deepcopy(static_coeff_MDS_1D_8)
static_coeff_MDS_1D_8_updated.drag_coeff[:, 2] = np.nan #for mye vibrasjoner
static_coeff_MDS_1D_8_updated.drag_coeff[:, 3] = np.nan #for mye vibrasjoner
static_coeff_MDS_1D_8_updated.lift_coeff[:, 2] = static_coeff_MDS_1D_8.lift_coeff[:, 2]
static_coeff_MDS_1D_8_updated.lift_coeff[:, 3] = static_coeff_MDS_1D_8.lift_coeff[:, 3]
static_coeff_MDS_1D_8_updated.pitch_coeff[:,2] = static_coeff_MDS_1D_8.pitch_coeff[:,2]
static_coeff_MDS_1D_8_updated.pitch_coeff[:, 3] = static_coeff_MDS_1D_8.pitch_coeff[:,3]

static_coeff_MDS_1D_10_updated = copy.deepcopy(static_coeff_MDS_1D_10)
static_coeff_MDS_1D_10_updated.drag_coeff[:, 2] = static_coeff_MDS_1D_10.drag_coeff[:, 2] 
static_coeff_MDS_1D_10_updated.drag_coeff[:, 3] = static_coeff_MDS_1D_10.drag_coeff[:, 3] 
static_coeff_MDS_1D_10_updated.lift_coeff[:, 2] = static_coeff_MDS_1D_10.lift_coeff[:, 2]
static_coeff_MDS_1D_10_updated.lift_coeff[:, 3] = static_coeff_MDS_1D_10.lift_coeff[:, 3]
static_coeff_MDS_1D_10_updated.pitch_coeff[:, 2] = static_coeff_MDS_1D_10.pitch_coeff[:,2]
static_coeff_MDS_1D_10_updated.pitch_coeff[:, 3] = static_coeff_MDS_1D_10.pitch_coeff[:, 3]

cd_6_mean = static_coeff_MDS_1D_6_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True, ax = None)[1]
cd_10_mean = static_coeff_MDS_1D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[1]
alpha= static_coeff_MDS_1D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[2]

for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_6_mean[i])
        print(cd_10_mean[i])

#%% !?!!!
def plot_compare_wind_speeds_mean_seperate1dmds(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,2}(\alpha_2)$"
        coeff = "drag_coeff"
        min = 0.34#0.4
        max = 0.45#0.58
    elif scoff == "lift":
        axis = r"$C_{L,2}(\alpha_2)$"
        coeff = "lift_coeff"
        min = -0.3#-0.35
        max = 0.22#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,2}(\alpha_2)$"
        coeff = "pitch_coeff"
        min = -0.04#-0.05
        max = 0.05#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    # ax.plot(unique_alphas_low, upwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)
    ax.plot(unique_alphas_low, downwind_mean_low,
             label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        # ax.plot(unique_alphas_med, upwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        ax.plot(unique_alphas_med, downwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    # ax.plot(unique_alphas_high, upwind_mean_high,
    #             label=f"9 m/s", color ="#d62728", alpha = 0.8)
    ax.plot(unique_alphas_high, downwind_mean_high,
                label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)

    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MDS1D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate1dmds(static_coeff_MDS_1D_6_updated, static_coeff_MDS_1D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mds_1D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate1dmds(static_coeff_MDS_1D_6_updated, static_coeff_MDS_1D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mds_1D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate1dmds(static_coeff_MDS_1D_6_updated, static_coeff_MDS_1D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mds_1D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)



#%% !!!
# Load all upwind experiments (upwind in rig)

section_name = "MDS_1D_Static"
file_names_MUS_1D_5 = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_001"] # 5 m/s
file_names_MUS_1D_10 = ["HAR_INT_MDS_GAP_213D_02_01_000","HAR_INT_MDS_GAP_213D_02_01_003"] # 10 m/s, vibrations

exp0_MUS_1D, exp1_MUS_1D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_1D_5,  upwind_in_rig=True)
exp0_MUS_1D, exp1_MUS_1D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_1D_10,  upwind_in_rig=True)



# exp0_MUS_1D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 1D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_1D_0" + ".png"))
exp1_MUS_1D_5.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MUS 1D - Wind speed: 5 m/s ",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_1D_5_tidserie" + ".png"),  dpi=300)
exp1_MUS_1D_10.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MUS 1D - Wind speed: 10 m/s ",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_1D_10_tidserie" + ".png"), dpi=300)

exp0_MUS_1D.filt_forces(6, 2)
exp1_MUS_1D_5.filt_forces(6, 2)
exp1_MUS_1D_10.filt_forces(6, 2)

# exp0_MUS_1D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 1D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_1D_0_filter" + ".png"))
exp1_MUS_1D_5.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 1D - Wind speed: 5 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_1D_5_filter_tidserie" + ".png"), dpi=300)
exp1_MUS_1D_10.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 1D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_1D_10_filter_tidserie" + ".png"), dpi=300)
plt.show()
plt.close()



static_coeff_MUS_1D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_1D, exp1_MUS_1D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_1D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_1D, exp1_MUS_1D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_1D_5, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_1D_10, section_name, 10, mode="decks", upwind_in_rig=True)


#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_1D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_1D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_1D_low_cm" + ".png"), dpi=300)

fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_1D_high_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_1D_high_cl" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_1D_high_cm" + ".png"), dpi=300)



#%%
#  Filter and plot ALT 
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_1D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS 1D - Wind speed: 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_1D_6_drag_clean.png"))

alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_1D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_1D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_1D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS 1D - Wind speed: 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_1D_6_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_1D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_1D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_1D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS 1D - Wind speed: 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_1D_6_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_1D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS 1D - Wind speed: 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_1D_10_pitch_clean.png"))





#%% !!!
#  Filter and plot ALT 2
section_name = "MDS_1D_Static_filtered"

static_coeff_MUS_1D_5_filtered, static_coeff_MUS_1D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_1D_5, static_coeff_2=static_coeff_MUS_1D_10, threshold=0.05, threshold_low=[0.08,0.02,0.005],threshold_high=[0.08,0.02,0.005],single=False)


#plot_static_coeff_summary(static_coeff_MUS_1D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_1D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% !!!
#Summary

static_coeff_MUS_1D_5_updated = copy.deepcopy(static_coeff_MUS_1D_5)
static_coeff_MUS_1D_5_updated.drag_coeff[:, 0] = static_coeff_MUS_1D_5.drag_coeff[:, 0]
static_coeff_MUS_1D_5_updated.drag_coeff[:, 1] = static_coeff_MUS_1D_5.drag_coeff[:, 1]
static_coeff_MUS_1D_5_updated.lift_coeff[:, 0] = static_coeff_MUS_1D_5.lift_coeff[:, 0]
static_coeff_MUS_1D_5_updated.lift_coeff[:, 1] = static_coeff_MUS_1D_5.lift_coeff[:, 1]
static_coeff_MUS_1D_5_updated.pitch_coeff[:, 0] = static_coeff_MUS_1D_5.pitch_coeff[:,0]
static_coeff_MUS_1D_5_updated.pitch_coeff[:, 1] = static_coeff_MUS_1D_5.pitch_coeff[:, 1]


static_coeff_MUS_1D_10_updated = copy.deepcopy(static_coeff_MUS_1D_10)
 
static_coeff_MUS_1D_10_updated.drag_coeff[:, 0] = static_coeff_MUS_1D_10_filtered.drag_coeff[:, 0] 
static_coeff_MUS_1D_10_updated.drag_coeff[:, 1] = static_coeff_MUS_1D_10_filtered.drag_coeff[:, 1] 
static_coeff_MUS_1D_10_updated.lift_coeff[:, 0] = static_coeff_MUS_1D_10.lift_coeff[:, 0]
static_coeff_MUS_1D_10_updated.lift_coeff[:, 1] = static_coeff_MUS_1D_10.lift_coeff[:, 1]
static_coeff_MUS_1D_10_updated.pitch_coeff[:, 0] = static_coeff_MUS_1D_10.pitch_coeff[:,0]
static_coeff_MUS_1D_10_updated.pitch_coeff[:, 1] = static_coeff_MUS_1D_10.pitch_coeff[:, 1]


cd_5_mean = static_coeff_MUS_1D_5_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True, ax = None)[0]
cd_10_mean = static_coeff_MUS_1D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[0]
alpha = static_coeff_MUS_1D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[2]

for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_5_mean[i])
        print(cd_10_mean[i])
#%% !?!!!
def plot_compare_wind_speeds_mean_seperate1dmUs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,1}(\alpha_1)$"
        coeff = "drag_coeff"
        min = 0.35#0.4
        max = 0.5#0.58
    elif scoff == "lift":
        axis = r"$C_{L,1}(\alpha_1)$"
        coeff = "lift_coeff"
        min = -0.28#-0.35
        max = 0.42#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,1}(\alpha_1)$"
        coeff = "pitch_coeff"
        min = -0.034#-0.05
        max = 0.09#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    # ax.plot(unique_alphas_low, downwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        ax.plot(unique_alphas_med, upwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        # ax.plot(unique_alphas_med, downwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"10 m/s", color ="#d62728", alpha = 0.8)
    # ax.plot(unique_alphas_high, downwind_mean_high,
    #             label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MUS1D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate1dmUs(static_coeff_MUS_1D_5_updated, static_coeff_MUS_1D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mus_1D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate1dmUs(static_coeff_MUS_1D_5_updated, static_coeff_MUS_1D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mus_1D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate1dmUs(static_coeff_MUS_1D_5_updated, static_coeff_MUS_1D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mus_1D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)



#%%
#  Save all experiments to excel
section_name = "1D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_1D_6.to_excel(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_1D_8.to_excel(section_name, sheet_name="MDS - 8" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10.to_excel(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "1D_mean"

# Low wind speed
static_coeff_MDS_1D_6.to_excel_mean(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_MDS_1D_8.to_excel_mean(section_name, sheet_name="MDS - 8" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10.to_excel_mean(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% 
# Save all experiments to excel filtered
section_name = "1D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_1D_6_filtered.to_excel(section_name, sheet_name="MDS - 6 -" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5_filtered.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10_filtered.to_excel(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10_filtered.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "1D_mean_filtered"

# Low wind speed
static_coeff_MDS_1D_6_filtered.to_excel_mean(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_1D_10_filtered.to_excel_mean(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_1D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% 
# Compare all experiments (MUS vs MDS vs Single)
section_name = "1D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_low_drag" + ".png"))
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_low_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_low_pitch" + ".png"))
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_low_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_1D_5, static_coeff_MDS_1D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_low_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_1D_5_filtered, static_coeff_MDS_1D_6_filtered)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_low_pitch_mean" + ".png"))

# Medium wind speed
# w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()

# # Mean
# w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_1D_8, static_coeff_MDS_1D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()

#%%
section_name = "1D"

#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_high_drag" + ".png"))
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_high_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_high_pitch" + ".png"))

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_high_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_1D_10, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_high_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_1D_10_filtered, static_coeff_MDS_1D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "1D_high_pitch_mean" + ".png"))

plt.show()

plt.close()

#%% 
# Compare all experiments - only with single deck
section_name = "1D"

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_low_drag.png"))
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_low_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_low_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_low_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_low_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_1D_6_filtered,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(  os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_low_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_low_drag_mean.png"))
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_1D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_low_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MUS_1D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(  os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_low_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6, static_coeff_MDS_1D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_low_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MUS_1D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_low_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6, static_coeff_MDS_1D_6_filtered, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_low_pitch_mean.png"))
#Medium wind speed
#w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_med_drag.png"))
#w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_med_drag.png"))
#w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
#plt.savefig( )
#w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
#plt.savefig()
#w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_1D_8,  upwind_in_rig=True)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
#plt.savefig()
#w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
#plt.savefig()

# Mean
#w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_8,upwind_in_rig=True)
#plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_8,upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
#%%                 
section_name = "1D"
                             
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_high_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_high_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_1D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_high_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_high_lift.png") )
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_1D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_high_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_1D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_high_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_1D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_high_drag_mean.png"))
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_high_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_high_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_high_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_1D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_1D_high_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_1D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_1D_high_pitch_mean.png"))
plt.show()

plt.close()


# %% 
# Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6,static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,static_coeff_MUS_1D_10_filtered, scoff = "drag")                        
plt.gcf().suptitle(f"1D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_1D_drag.png"))

# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6,
                               static_coeff_MDS_1D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"1D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_drag.png"))
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                             static_coeff_MUS_1D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_1D_lift.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered,
                               static_coeff_single_9,static_coeff_MDS_1D_6,
                             static_coeff_MDS_1D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_lift.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MUS_1D_5_filtered,
                                static_coeff_MUS_1D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_pitch.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MDS_1D_6_filtered,
                                static_coeff_MDS_1D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_pitch.png"))
#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6,
                               static_coeff_single_9_filtered, static_coeff_MUS_1D_5_filtered,
                                static_coeff_MUS_1D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"1D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_1D_drag_mean.png"))
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6,
                               static_coeff_single_9_filtered, static_coeff_MDS_1D_6,
                                static_coeff_MDS_1D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"1D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_drag_mean.png"))
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MUS_1D_5,
                                static_coeff_MUS_1D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_1D_lift_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MDS_1D_6,
                                static_coeff_MDS_1D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"1D: MDS ",  y=0.95)
plt.savefig(  os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_lift_mean.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MUS_1D_5_filtered,
                                static_coeff_MUS_1D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MUS  ",  y=0.95)
plt.savefig(    os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_1D_pitch_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered,
                               static_coeff_single_9, static_coeff_MDS_1D_6_filtered,
                                static_coeff_MDS_1D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"1D: MDS ",  y=0.95)
plt.savefig(    os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_1D_pitch_mean.png"))


############################################################################################################

#print("2D")


#%% !!!
# Load all downwind experiments (downwind in rig)
section_name = "MUS_2D_Static"
file_names_MDS_2D_6 = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_002"] #6 m/s
#file_names_MDS_2D_8 = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_003"] # 8 m/s, vibrations
file_names_MDS_2D_10 = ["HAR_INT_MUS_GAP_213D_02_00_001","HAR_INT_MUS_GAP_213D_02_00_004"] # 10 m/s


exp0_MDS_2D, exp1_MDS_2D_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_2D_6,  upwind_in_rig=False)
#exp0_MDS_2D, exp1_MDS_2D_8= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_2D_8,  upwind_in_rig=False)
exp0_MDS_2D, exp1_MDS_2D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_2D_10,  upwind_in_rig=False)



# exp0_MDS_2D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 2D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_0.png"))
# exp1_MDS_2D_6.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 2D - Wind speed: 6 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_6.png"))
# #exp1_MDS_2D_8.plot_experiment(mode="total") #
# #plt.gcf().suptitle(f"MDS 2D - Wind speed: 8 m/s ",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_8.png"))
# exp1_MDS_2D_10.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 2D - Wind speed: 10 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_10.png"))

exp0_MDS_2D.filt_forces(6, 2)
exp1_MDS_2D_6.filt_forces(6, 2)
#exp1_MDS_2D_8.filt_forces(6, 2)
exp1_MDS_2D_10.filt_forces(6, 2)

# exp0_MDS_2D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 2D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_0_filter.png"))
# exp1_MDS_2D_6.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 2D - Wind speed: 6 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_6_filter.png"))
# #exp1_MDS_2D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
# #plt.gcf().suptitle(f"MDS 2D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_8_filter.png"))
# exp1_MDS_2D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 2D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_2D_10_filter.png"))
# plt.show()
# plt.close()


static_coeff_MDS_2D_6 =w3t.StaticCoeff.fromWTT(exp0_MDS_2D, exp1_MDS_2D_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

#static_coeff_MDS_2D_8 = w3t.StaticCoeff.fromWTT(exp0_MDS_2D, exp1_MDS_2D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_2D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_2D, exp1_MDS_2D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

plot_static_coeff_summary(static_coeff_MDS_2D_6, section_name, 6, mode="decks", upwind_in_rig=False)
#plot_static_coeff_summary(static_coeff_MDS_2D_8, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_2D_10, section_name, 10, mode="decks", upwind_in_rig=False)

#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_2D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_2D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_2D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_2D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_2D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_2D_low_cm" + ".png"), dpi=300)

fig, ax =w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_2D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_2D_high_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_2D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_2D_high_cl" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_2D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_2D_high_cm" + ".png"), dpi=300)


#%%
#  Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_2D_6, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_2D_Static, 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_2D_6_drag_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_2D_8, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
#plt.suptitle(f"MDS_2D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_2D_8_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_2D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_2D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_2D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_2D_6, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_2D_Static, 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_2D_6_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_2D_8, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
#plt.suptitle(f"MDS_2D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_2D_8_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_2D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_2D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_2D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_2D_6, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_2D_Static, 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_2D_6_pitch_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_2D_8, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MDS_2D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_2D_8_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_2D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_2D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_2D_10_pitch_clean.png"))

#%% !!!
# Summary
static_coeff_MDS_2D_6_updated = copy.deepcopy(static_coeff_MDS_2D_6)
static_coeff_MDS_2D_6_updated.drag_coeff[:, 2] = static_coeff_MDS_2D_6.drag_coeff[:, 2]
static_coeff_MDS_2D_6_updated.drag_coeff[:, 3] = static_coeff_MDS_2D_6.drag_coeff[:, 3]
static_coeff_MDS_2D_6_updated.lift_coeff[:, 2] = static_coeff_MDS_2D_6.lift_coeff[:, 2]
static_coeff_MDS_2D_6_updated.lift_coeff[:, 3] = static_coeff_MDS_2D_6.lift_coeff[:, 3]
static_coeff_MDS_2D_6_updated.pitch_coeff[:, 2] = static_coeff_MDS_2D_6.pitch_coeff[:, 2]
static_coeff_MDS_2D_6_updated.pitch_coeff[:, 3] = static_coeff_MDS_2D_6.pitch_coeff[:, 3]




static_coeff_MDS_2D_10_updated = copy.deepcopy(static_coeff_MDS_2D_10)
 
static_coeff_MDS_2D_10_updated.drag_coeff[:, 2] = static_coeff_MDS_2D_10.drag_coeff[:, 2] 
static_coeff_MDS_2D_10_updated.drag_coeff[:, 3] = static_coeff_MDS_2D_10.drag_coeff[:, 3] 
static_coeff_MDS_2D_10_updated.lift_coeff[:, 2] = static_coeff_MDS_2D_10.lift_coeff[:, 2]
static_coeff_MDS_2D_10_updated.lift_coeff[:, 3] = static_coeff_MDS_2D_10.lift_coeff[:, 3]
static_coeff_MDS_2D_10_updated.pitch_coeff[:, 2] = static_coeff_MDS_2D_10.pitch_coeff[:,2]
static_coeff_MDS_2D_10_updated.pitch_coeff[:, 3] = static_coeff_MDS_2D_10.pitch_coeff[:, 3]



#%% !?!!!
def plot_compare_wind_speeds_mean_seperate2dmDs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,2}(\alpha_2)$"
        coeff = "drag_coeff"
        min = 0.42#0.4
        max = 0.55#0.58
    elif scoff == "lift":
        axis = r"$C_{L,2}(\alpha_2)$"
        coeff = "lift_coeff"
        min = -0.28#-0.35
        max = 0.2#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,2}(\alpha_2)$"
        coeff = "pitch_coeff"
        min = -0.035#-0.05
        max = 0.075
    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    # ax.plot(unique_alphas_low, upwind_mean_low,
    #          label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    ax.plot(unique_alphas_low, downwind_mean_low,
             label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        # ax.plot(unique_alphas_med, upwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        ax.plot(unique_alphas_med, downwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    # ax.plot(unique_alphas_high, upwind_mean_high,
    #             label=f"10 m/s", color ="#d62728", alpha = 0.8)
    ax.plot(unique_alphas_high, downwind_mean_high,
                label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MDS2D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate2dmDs(static_coeff_MDS_2D_6_updated, static_coeff_MDS_2D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mds_2D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate2dmDs(static_coeff_MDS_2D_6_updated, static_coeff_MDS_2D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mds_2D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate2dmDs(static_coeff_MDS_2D_6_updated, static_coeff_MDS_2D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mds_2D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)


#%% !!!
# Load all upwind experiments (upwind in rig)

section_name = "MDS_2D_Static"
file_names_MUS_2D_5 = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_001"] # 5 m/s, vibrations
#file_names_MUS_2D_8 = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_002"] # 8 m/s, vibrations
file_names_MUS_2D_10 = ["HAR_INT_MDS_GAP_213D_02_00_000","HAR_INT_MDS_GAP_213D_02_00_003"] # 10 m/s, vibrations



exp0_MUS_2D, exp1_MUS_2D_5 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_2D_5,  upwind_in_rig=True)
#exp0_MUS_2D, exp1_MUS_2D_8 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_2D_8,  upwind_in_rig=True)
exp0_MUS_2D, exp1_MUS_2D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_2D_10,  upwind_in_rig=True)




# exp0_MUS_2D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 2D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_0.png"))
# exp1_MUS_2D_5.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 2D - Wind speed: 5 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_5.png"))
# #exp1_MUS_2D_8.plot_experiment(mode="total") #
# #plt.gcf().suptitle(f"MUS 2D - Wind speed: 8 m/s ",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_8.png"))
# exp1_MUS_2D_10.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 2D - Wind speed: 10 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_10.png"))

exp0_MUS_2D.filt_forces(6, 2)
exp1_MUS_2D_5.filt_forces(6, 2)
#exp1_MUS_2D_8.filt_forces(6, 2)
exp1_MUS_2D_10.filt_forces(6, 2)

# exp0_MUS_2D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 2D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_0_filter.png"))
# exp1_MUS_2D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 2D - Wind speed: 5 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_5_filter.png"))
# #exp1_MUS_2D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
# #plt.gcf().suptitle(f"MUS 2D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_8_filter.png"))
# exp1_MUS_2D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 2D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_2D_10_filter.png"))
# plt.show()
# plt.close()

static_coeff_MUS_2D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_2D, exp1_MUS_2D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

#static_coeff_MUS_2D_8 = w3t.StaticCoeff.fromWTT(exp0_MUS_2D, exp1_MUS_2D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_2D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_2D, exp1_MUS_2D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)


plot_static_coeff_summary(static_coeff_MUS_2D_5, section_name, 5, mode="decks", upwind_in_rig=True)
#plot_static_coeff_summary(static_coeff_MUS_2D_8, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_2D_10, section_name, 10, mode="decks", upwind_in_rig=True)

#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_2D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_2D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_2D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_2D_low_cm" + ".png"), dpi=300)

fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_2D_high_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_2D_high_cl" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_2D_high_cm" + ".png"), dpi=300)



#%% 
# Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_2D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_2D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_2D_5_drag_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_2D_8, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
#plt.suptitle(f"MUS_2D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_2D_8_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_2D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_2D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_2D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_2D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.025, scoff="lift")
plt.suptitle(f"MUS_2D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_2D_5_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_2D_8, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
#plt.suptitle(f"MUS_2D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_2D_8_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_2D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_2D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_2D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_2D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_2D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_2D_5_pitch_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_2D_8, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MUS_2D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_2D_8_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_2D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_2D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_2D_10_pitch_clean.png"))


#%%  !!!
# Filter and plot ALT 2
section_name = "MDS_2D_Static_filtered"

static_coeff_MUS_2D_5_filtered, static_coeff_MUS_2D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_2D_5, static_coeff_2=static_coeff_MUS_2D_10, threshold=0.05, threshold_low=[0.05,0.025,0.005],threshold_high=[0.05,0.025,0.005],single=False)


#plot_static_coeff_summary(static_coeff_MUS_2D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_2D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)

#%%
#Summary !!!
static_coeff_MUS_2D_10_updated = copy.deepcopy(static_coeff_MUS_2D_10)
static_coeff_MUS_2D_10_filtered_fix = copy.deepcopy(static_coeff_MUS_2D_10_filtered)

static_coeff_MUS_2D_10_filtered_fix_drag= static_coeff_MUS_2D_10_filtered.plot_drag_mean( mode="decks", upwind_in_rig=True)[0]
static_coeff_MUS_2D_10_fix_drag = static_coeff_MUS_2D_10.plot_drag_mean( mode="decks", upwind_in_rig=True)[0]

#static_coeff_MUS_2D_10_filtered_upwind = fix(static_coeff_MUS_2D_10_filtered_fix.drag_coeff[:,0] + static_coeff_MUS_2D_10_filtered_fix.drag_coeff[:,1], static_coeff_MUS_2D_10.drag_coeff[:,0] + static_coeff_MUS_2D_10.drag_coeff[:,1])

static_coeff_MUS_2D_10_filtered_upwind_drag = fix(static_coeff_MUS_2D_10_filtered_fix_drag, static_coeff_MUS_2D_10_fix_drag)

# static_coeff_MUS_2D_10_filtered.drag_coeff[:, 0] = static_coeff_MUS_2D_10_filtered_upwind/2
# static_coeff_MUS_2D_10_filtered.drag_coeff[:, 1] = static_coeff_MUS_2D_10_filtered_upwind/2


# 1. Beregn alpha-grupper (som i plot_drag_mean)
alpha = np.round(static_coeff_MUS_2D_10_updated.pitch_motion * 360 / (2 * np.pi), 1)
unique_alphas = np.sort(np.unique(alpha))

# 2. Lag en mapping: alpha-verdi → mean-verdi (fra tidligere)
alpha_to_mean = dict(zip(unique_alphas, static_coeff_MUS_2D_10_filtered_upwind_drag))

# 3. Gjenoppbygg full array: hvert datapunkt får sin gruppes gjennomsnittsverdi
cd_full = np.array([alpha_to_mean.get(a, np.nan) for a in alpha])



static_coeff_MUS_2D_10_filtered_fix.drag_coeff[:, 0] = cd_full/2
static_coeff_MUS_2D_10_filtered_fix.drag_coeff[:, 1] = cd_full/2


plot_static_coeff_summary(static_coeff_MUS_2D_10_filtered_fix, section_name, 10, mode="decks", upwind_in_rig=True)

 
static_coeff_MUS_2D_10_updated.drag_coeff[:, 0] = static_coeff_MUS_2D_10_filtered_fix.drag_coeff[:, 0] 
static_coeff_MUS_2D_10_updated.drag_coeff[:, 1] = static_coeff_MUS_2D_10_filtered_fix.drag_coeff[:, 1] 
static_coeff_MUS_2D_10_updated.lift_coeff[:, 0] = static_coeff_MUS_2D_10.lift_coeff[:, 0]
static_coeff_MUS_2D_10_updated.lift_coeff[:, 1] = static_coeff_MUS_2D_10.lift_coeff[:, 1]
static_coeff_MUS_2D_10_updated.pitch_coeff[:, 0] = static_coeff_MUS_2D_10.pitch_coeff[:,0]
static_coeff_MUS_2D_10_updated.pitch_coeff[:, 1] = static_coeff_MUS_2D_10.pitch_coeff[:, 1]

static_coeff_MUS_2D_5_updated = copy.deepcopy(static_coeff_MUS_2D_5)
static_coeff_MUS_2D_5_updated.drag_coeff[:, 0] = static_coeff_MUS_2D_5.drag_coeff[:, 0]
static_coeff_MUS_2D_5_updated.drag_coeff[:, 1] = static_coeff_MUS_2D_5.drag_coeff[:, 1]
static_coeff_MUS_2D_5_updated.lift_coeff[:, 0] = static_coeff_MUS_2D_5.lift_coeff[:, 0]
static_coeff_MUS_2D_5_updated.lift_coeff[:, 1] = static_coeff_MUS_2D_5.lift_coeff[:, 1]
static_coeff_MUS_2D_5_updated.pitch_coeff[:, 0] = static_coeff_MUS_2D_5.pitch_coeff[:,0]
static_coeff_MUS_2D_5_updated.pitch_coeff[:, 1] = static_coeff_MUS_2D_5.pitch_coeff[:, 1]
#%% !?!!!
def plot_compare_wind_speeds_mean_seperate2dmUs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,1}(\alpha_1)$"
        coeff = "drag_coeff"
        min = 0.35#0.4
        max = 0.5#0.58
    elif scoff == "lift":
        axis = r"$C_{L,1}(\alpha_1)$"
        coeff = "lift_coeff"
        min = -0.28#-0.35
        max = 0.42#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,1}(\alpha_1)$"
        coeff = "pitch_coeff"
        min = -0.035#-0.05
        max = 0.090#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    # ax.plot(unique_alphas_low, downwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        ax.plot(unique_alphas_med, upwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        # ax.plot(unique_alphas_med, downwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"10 m/s", color ="#d62728", alpha = 0.8)
    # ax.plot(unique_alphas_high, downwind_mean_high,
    #             label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MUS2D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate2dmUs(static_coeff_MUS_1D_5_updated, static_coeff_MUS_1D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mus_2D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate2dmUs(static_coeff_MUS_1D_5_updated, static_coeff_MUS_1D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mus_2D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate2dmUs(static_coeff_MUS_1D_5_updated, static_coeff_MUS_1D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mus_2D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)


#%% 
# Save all experiments to excel
section_name = "2D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_2D_6.to_excel(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_5.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_2D_8.to_excel(section_name, sheet_name="MDS - 8" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_2D_8.to_excel(section_name, sheet_name='MUS - 8' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_2D_10.to_excel(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_10.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "2D_mean"

# Low wind speed
static_coeff_MDS_2D_6.to_excel_mean(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_5.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_2D_8.to_excel_mean(section_name, sheet_name="MDS - 8" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_2D_8.to_excel_mean(section_name, sheet_name='MUS - 8' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_2D_10.to_excel_mean(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_2D_10.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% 
# Save all experiments to excel filtered
section_name = "2D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MUS_2D_5_filtered.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_2D_10_filtered.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "2D_mean_filtered"

# Low wind speed
static_coeff_MUS_2D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_2D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% 
# Compare all experiments (MUS vs MDS vs Single)
section_name = "2D"


#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_low_drag" + ".png"))
w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_low_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_low_pitch" + ".png"))
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_2D_5_filtered, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_low_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_low_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, static_coeff_MDS_2D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_low_pitch_mean" + ".png"))
# #Medium wind speed
# w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_lift(static_coeff_single_6, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_pitch(static_coeff_single_6, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()

# # Mean
# w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# plt.savefig()
# w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()
# w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_2D_8, static_coeff_MDS_2D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# plt.savefig()

#%%
#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_high_drag" + ".png"))
w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_high_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_high_pitch" + ".png"))

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_high_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_high_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_2D_10, static_coeff_MDS_2D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "2D_high_pitch_mean" + ".png"))

plt.show()
plt.close()

#%% 
# Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_2D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_low_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_low_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_low_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_low_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_low_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_low_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_2D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_low_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_2D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_low_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_low_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_low_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_2D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_low_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_2D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_low_pitch_mean.png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_2D_8,  upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_8,upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_8,upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
#%%                                 
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_high_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_high_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_high_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_high_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_2D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_high_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_2D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_high_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_2D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_high_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_high_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_high_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_high_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_2D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_2D_high_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_2D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_2D_high_pitch_mean.png"))
plt.show()
plt.close()

# %% Compare all experiments (Wind speed)


#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                                static_coeff_MUS_2D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"2D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_2D_drag.png"))


# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6,
                                static_coeff_MDS_2D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"2D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_2D_drag.png"))

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                                static_coeff_MUS_2D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_2D_lift.png"))

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9,static_coeff_MDS_2D_6,
                                static_coeff_MDS_2D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"2D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_2D_lift.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                                static_coeff_MUS_2D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_2D_pitch.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                                static_coeff_MDS_2D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_2D_pitch.png"))

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_2D_5_filtered,
                                static_coeff_MUS_2D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"2D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_2D_drag_mean.png"))
# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_2D_6,
                                static_coeff_MDS_2D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"2D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_2D_drag_mean.png"))
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                                static_coeff_MUS_2D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_2D_lift_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                                static_coeff_MDS_2D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"2D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_2D_lift_mean.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_2D_5,
                                static_coeff_MUS_2D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_2D_pitch_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_2D_6,
                                static_coeff_MDS_2D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"2D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_2D_pitch_mean.png"))



############################################################################################################
#print("3D")


#%% !!!
# Load all downwind experiments (downwind in rig)
section_name = "MUS_3D_Static"
file_names_MDS_3D_6 = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_001"] #6 m/s
file_names_MDS_3D_8 = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_002"] # 8 m/s, vibrations
file_names_MDS_3D_10 = ["HAR_INT_MUS_GAP_213D_02_01_000","HAR_INT_MUS_GAP_213D_02_01_003"] # 10 m/s


exp0_MDS_3D, exp1_MDS_3D_6 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_3D_6,  upwind_in_rig=False)
exp0_MDS_3D, exp1_MDS_3D_8= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_3D_8,  upwind_in_rig=False)
exp0_MDS_3D, exp1_MDS_3D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_3D_10,  upwind_in_rig=False)



# exp0_MDS_3D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_0.png"))
# exp1_MDS_3D_6.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 6 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_6.png"))
# exp1_MDS_3D_8.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 8 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_8.png"))
# exp1_MDS_3D_10.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 10 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_10.png"))

exp0_MDS_3D.filt_forces(6, 2)
exp1_MDS_3D_6.filt_forces(6, 2)
exp1_MDS_3D_8.filt_forces(6, 2)
exp1_MDS_3D_10.filt_forces(6, 2)

# exp0_MDS_3D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_0_filter.png"))
# exp1_MDS_3D_6.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 6 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_6_filter.png"))
# exp1_MDS_3D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_8_filter.png"))
# exp1_MDS_3D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 3D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_3D_10_filter.png"))
# plt.show()
# plt.close()


static_coeff_MDS_3D_6 =w3t.StaticCoeff.fromWTT(exp0_MDS_3D, exp1_MDS_3D_6, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_3D_8 = w3t.StaticCoeff.fromWTT(exp0_MDS_3D, exp1_MDS_3D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_3D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_3D, exp1_MDS_3D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)


plot_static_coeff_summary(static_coeff_MDS_3D_6, section_name, 6, mode="decks", upwind_in_rig=False)
#plot_static_coeff_summary(static_coeff_MDS_3D_8, section_name, 8, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_3D_10, section_name, 10, mode="decks", upwind_in_rig=False)

#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_3D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_3D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_3D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_3D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_3D_6, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_3D_low_cm" + ".png"), dpi=300)

fig, ax =w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_3D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_3D_high_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_3D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_3D_high_cl" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_3D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_3D_high_cm" + ".png"), dpi=300)




#%% !!!!!
#fjerne hakk
alpha = np.round(static_coeff_MDS_3D_10.pitch_motion * 360 / (2 * np.pi), 1)
coeff_raw = static_coeff_MDS_3D_10.drag_coeff[:, 2] + static_coeff_MDS_3D_10.drag_coeff[:, 2]  # downwind

# 1. Fit en glatt kurve (uten NaN eller store avvik)
mask_fit = (alpha < 7) & (alpha > -7) & ~np.isnan(coeff_raw)
alpha_fit = alpha[mask_fit]
coeff_fit = coeff_raw[mask_fit]
coeffs = np.polyfit(alpha_fit, coeff_fit, deg=2)
curve = np.polyval(coeffs, alpha)

# 2. Finn hvilke punkter som avviker "for mye"
spread = np.abs(coeff_raw - curve)
threshold = 0.02  # juster etter hvor streng du vil være
mask_good = spread < threshold

# 3. Lag filtrert array
coeff_filtered = coeff_raw.copy()
coeff_filtered[~mask_good] = np.nan

# 4. Del ut tilbake til hver lastcelle
static_coeff_MDS_3D_10.drag_coeff[:, 2] = coeff_filtered / 2
static_coeff_MDS_3D_10.drag_coeff[:, 3] = coeff_filtered / 2
plot_static_coeff_summary(static_coeff_MDS_3D_10, section_name, 10, mode="decks", upwind_in_rig=False)

#%%
# filter regression
print("regRESSION")
section_name = "3D_MUS_Static_reg" # obs egt motsatt
static_coeff_MDS_3D_6_reg = copy.deepcopy(static_coeff_MDS_3D_6)

# alpha = np.round(static_coeff_MDS_3D_6_reg.pitch_motion * 360 / (2 * np.pi), 1)  # eller np.degrees()
# mask = alpha <= 4.0
# static_coeff_MDS_3D_6_reg.pitch_motion = static_coeff_MDS_3D_6_reg.pitch_motion[mask]
# static_coeff_MDS_3D_6_reg.drag_coeff = static_coeff_MDS_3D_6_reg.drag_coeff[mask, :]
# static_coeff_MDS_3D_6_reg.lift_coeff = static_coeff_MDS_3D_6_reg.lift_coeff[mask, :]
# static_coeff_MDS_3D_6_reg.pitch_coeff = static_coeff_MDS_3D_6_reg.pitch_coeff[mask, :]



# alpha_MDS_3D_6_reg, _,coeff_MDS_3D_6_drag_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_6, scoff="drag", single=False)
# static_coeff_MDS_3D_6_reg.drag_coeff[:, 0] = coeff_MDS_3D_6_drag_reg_down / 2
# static_coeff_MDS_3D_6_reg.drag_coeff[:, 1] = coeff_MDS_3D_6_drag_reg_down / 2
# alpha_MDS_3D_6_reg, _,coeff_MDS_3D_6_lift_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_6, scoff="lift", single=False)
# static_coeff_MDS_3D_6_reg.lift_coeff[:, 0] = coeff_MDS_3D_6_lift_reg_down / 2
# static_coeff_MDS_3D_6_reg.lift_coeff[:, 1] = coeff_MDS_3D_6_lift_reg_down / 2
# alpha_MDS_3D_6_reg,_, coeff_MDS_3D_6_pitch_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_6, scoff="pitch", single=False)
# static_coeff_MDS_3D_6_reg.pitch_coeff[:, 0] = coeff_MDS_3D_6_pitch_reg_down / 2
# static_coeff_MDS_3D_6_reg.pitch_coeff[:, 1] = coeff_MDS_3D_6_pitch_reg_down / 2
# plot_static_coeff_summary(static_coeff_MDS_3D_6_reg, section_name, 6, mode="decks", upwind_in_rig=False)

# static_coeff_MDS_3D_8_reg = copy.deepcopy(static_coeff_MDS_3D_8)
# alpha = np.round(static_coeff_MDS_3D_8_reg.pitch_motion * 360 / (2 * np.pi), 1)  # eller np.degrees()
# mask = alpha <= 4.0
# static_coeff_MDS_3D_8_reg.pitch_motion = static_coeff_MDS_3D_8_reg.pitch_motion[mask]
# static_coeff_MDS_3D_8_reg.drag_coeff = static_coeff_MDS_3D_8_reg.drag_coeff[mask, :]
# static_coeff_MDS_3D_8_reg.lift_coeff = static_coeff_MDS_3D_8_reg.lift_coeff[mask, :]
# static_coeff_MDS_3D_8_reg.pitch_coeff = static_coeff_MDS_3D_8_reg.pitch_coeff[mask, :]




# alpha_MDS_3D_8_reg, _,coeff_MDS_3D_8_drag_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_8, scoff="drag", single=False)
# static_coeff_MDS_3D_8_reg.drag_coeff[:, 0] = coeff_MDS_3D_8_drag_reg_down / 2
# static_coeff_MDS_3D_8_reg.drag_coeff[:, 1] = coeff_MDS_3D_8_drag_reg_down / 2
# alpha_MDS_3D_8_reg, _,coeff_MDS_3D_8_lift_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_8, scoff="lift", single=False)
# static_coeff_MDS_3D_8_reg.lift_coeff[:, 0] = coeff_MDS_3D_8_lift_reg_down / 2
# static_coeff_MDS_3D_8_reg.lift_coeff[:, 1] = coeff_MDS_3D_8_lift_reg_down / 2
# alpha_MDS_3D_8_reg,_, coeff_MDS_3D_8_pitch_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_8, scoff="pitch", single=False)
# static_coeff_MDS_3D_8_reg.pitch_coeff[:, 0] = coeff_MDS_3D_8_pitch_reg_down / 2
# static_coeff_MDS_3D_8_reg.pitch_coeff[:, 1] = coeff_MDS_3D_8_pitch_reg_down / 2
# plot_static_coeff_summary(static_coeff_MDS_3D_8_reg, section_name, 8, mode="decks", upwind_in_rig=False)

# static_coeff_MDS_3D_8_reg_new=w3t._scoff.poly_estimat_spess(static_coeff_MDS_3D_8_reg, scoff="drag", single=False)
# plot_static_coeff_summary(static_coeff_MDS_3D_8_reg_new, section_name, 8, mode="decks", upwind_in_rig=False)


# static_coeff_MDS_3D_10_reg = copy.deepcopy(static_coeff_MDS_3D_10)
# static_coeff_MDS_3D_10_reg.pitch_motion = static_coeff_MDS_3D_10_reg.pitch_motion[mask]
# static_coeff_MDS_3D_10_reg.drag_coeff = static_coeff_MDS_3D_10_reg.drag_coeff[mask, :]
# static_coeff_MDS_3D_10_reg.lift_coeff = static_coeff_MDS_3D_10_reg.lift_coeff[mask, :]
# static_coeff_MDS_3D_10_reg.pitch_coeff = static_coeff_MDS_3D_10_reg.pitch_coeff[mask, :]


# alpha_MDS_3D_10_reg, _, coeff_MDS_3D_10_drag_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_10, scoff="drag", single=False)
# static_coeff_MDS_3D_10_reg.drag_coeff[:, 0] = coeff_MDS_3D_10_drag_reg_down / 2
# static_coeff_MDS_3D_10_reg.drag_coeff[:, 1] = coeff_MDS_3D_10_drag_reg_down / 2
# alpha_MDS_3D_10_reg,_, coeff_MDS_3D_10_lift_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_10, scoff="lift", single=False)
# static_coeff_MDS_3D_10_reg.lift_coeff[:, 0] = coeff_MDS_3D_10_lift_reg_down / 2
# static_coeff_MDS_3D_10_reg.lift_coeff[:, 1] = coeff_MDS_3D_10_lift_reg_down / 2
# alpha_MDS_3D_10_reg, _,coeff_MDS_3D_10_pitch_reg_down = w3t._scoff.poly_estimat(static_coeff_MDS_3D_10, scoff="pitch", single=False)
# static_coeff_MDS_3D_10_reg.pitch_coeff[:, 0] = coeff_MDS_3D_10_pitch_reg_down / 2
# static_coeff_MDS_3D_10_reg.pitch_coeff[:, 1] = coeff_MDS_3D_10_pitch_reg_down / 2
# plot_static_coeff_summary(static_coeff_MDS_3D_10_reg, section_name, 10, mode="decks", upwind_in_rig=False)
section_name = "MUS_3D_Static"


#%%
#  Filter and plot ALT 1
print("FILTER 1")

#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_3D_6, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_3D_Static, 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_3D_6_drag_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_3D_8, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
#plt.suptitle(f"MDS_3D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_3D_8_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_3D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_3D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_3D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_3D_6, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_3D_Static, 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_3D_6_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_3D_8, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
#plt.suptitle(f"MDS_3D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_3D_8_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_3D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_3D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_3D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_3D_6, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_3D_Static, 6 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_3D_6_pitch_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_3D_8, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MDS_3D_Static, 8 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_3D_8_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_3D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_3D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_3D_10_pitch_clean.png"))

#%%
#   Filter and plot ALT 2
print("FILTER 2")

section_name = "MUS_3D_Static_filtered"

static_coeff_MDS_3D_6_filtered, static_coeff_MDS_3D_8_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MDS_3D_6, static_coeff_2=static_coeff_MDS_3D_8, threshold=0.05, threshold_low=[0.05,0.03,0.005],threshold_high=[0.04,0.03,0.005],single=False)


plot_static_coeff_summary(static_coeff_MDS_3D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=True)

#%% !!!
# Summary
static_coeff_MDS_3D_6_updated = copy.deepcopy(static_coeff_MDS_3D_6)
# static_coeff_MDS_3D_8_updated = copy.deepcopy(static_coeff_MDS_3D_8)
static_coeff_MDS_3D_10_updated = copy.deepcopy(static_coeff_MDS_3D_10)

# alpha_6_updated = np.round(static_coeff_MDS_3D_6_updated.pitch_motion * 360 / (2 * np.pi), 1)
# mask_6_updated = alpha_6_updated <= 4.00
# alpha_8_updated = np.round(static_coeff_MDS_3D_8_updated.pitch_motion * 360 / (2 * np.pi), 1)
# mask_8_updated = alpha_8_updated <= 4.00
# alpha_10_updated = np.round(static_coeff_MDS_3D_10_updated.pitch_motion * 360 / (2 * np.pi), 1)
# mask_10_updated = alpha_10_updated <= 4.00

# static_coeff_MDS_3D_6_updated.pitch_motion = static_coeff_MDS_3D_6_updated.pitch_motion[mask_6_updated]
# static_coeff_MDS_3D_6_updated.drag_coeff = static_coeff_MDS_3D_6_updated.drag_coeff[mask_6_updated, :]
# static_coeff_MDS_3D_6_updated.lift_coeff = static_coeff_MDS_3D_6_updated.lift_coeff[mask_6_updated, :]
# static_coeff_MDS_3D_6_updated.pitch_coeff = static_coeff_MDS_3D_6_updated.pitch_coeff[mask_6_updated, :]

# static_coeff_MDS_3D_8_updated.pitch_motion = static_coeff_MDS_3D_8_updated.pitch_motion[mask_8_updated]
# static_coeff_MDS_3D_8_updated.drag_coeff = static_coeff_MDS_3D_8_updated.drag_coeff[mask_8_updated, :]
# static_coeff_MDS_3D_8_updated.lift_coeff = static_coeff_MDS_3D_8_updated.lift_coeff[mask_8_updated, :]
# static_coeff_MDS_3D_8_updated.pitch_coeff = static_coeff_MDS_3D_8_updated.pitch_coeff[mask_8_updated, :]

# static_coeff_MDS_3D_10_updated.pitch_motion = static_coeff_MDS_3D_10_updated.pitch_motion[mask_10_updated]
# static_coeff_MDS_3D_10_updated.drag_coeff = static_coeff_MDS_3D_10_updated.drag_coeff[mask_10_updated, :]
# static_coeff_MDS_3D_10_updated.lift_coeff = static_coeff_MDS_3D_10_updated.lift_coeff[mask_10_updated, :]
# static_coeff_MDS_3D_10_updated.pitch_coeff = static_coeff_MDS_3D_10_updated.pitch_coeff[mask_10_updated, :]


# alpha_6_updated = np.round(static_coeff_MDS_3D_6.pitch_motion * 360 / (2 * np.pi), 1)
# mask_6_updated2 = alpha_6_updated <= 4.00
# alpha_8_updated = np.round(static_coeff_MDS_3D_8.pitch_motion * 360 / (2 * np.pi), 1)
# mask_8_updated2 = alpha_8_updated <= 4.00
# alpha_10_updated = np.round(static_coeff_MDS_3D_10.pitch_motion * 360 / (2 * np.pi), 1)
# mask_10_updated2 = alpha_10_updated <= 4.00

# static_coeff_MDS_3D_6.pitch_motion = static_coeff_MDS_3D_6.pitch_motion[mask_6_updated2]
# static_coeff_MDS_3D_6.drag_coeff = static_coeff_MDS_3D_6.drag_coeff[mask_6_updated2, :]
# static_coeff_MDS_3D_6.lift_coeff = static_coeff_MDS_3D_6.lift_coeff[mask_6_updated2, :]
# static_coeff_MDS_3D_6.pitch_coeff = static_coeff_MDS_3D_6.pitch_coeff[mask_6_updated2, :]

# static_coeff_MDS_3D_8.pitch_motion = static_coeff_MDS_3D_8.pitch_motion[mask_8_updated2]
# static_coeff_MDS_3D_8.drag_coeff = static_coeff_MDS_3D_8.drag_coeff[mask_8_updated2, :]
# static_coeff_MDS_3D_8.lift_coeff = static_coeff_MDS_3D_8.lift_coeff[mask_8_updated2, :]
# static_coeff_MDS_3D_8.pitch_coeff = static_coeff_MDS_3D_8.pitch_coeff[mask_8_updated2, :]
# alpha_8_new = np.round(static_coeff_MDS_3D_8_reg_new.pitch_motion * 360 / (2 * np.pi), 1)
# mask_8_new = alpha_8_new <= 4.0
# static_coeff_MDS_3D_8_reg_new.drag_coeff = static_coeff_MDS_3D_8_reg_new.drag_coeff[mask_8_new, :]

# static_coeff_MDS_3D_10.pitch_motion = static_coeff_MDS_3D_10.pitch_motion[mask_10_updated2]
# static_coeff_MDS_3D_10.drag_coeff = static_coeff_MDS_3D_10.drag_coeff[mask_10_updated2, :]
# static_coeff_MDS_3D_10.lift_coeff = static_coeff_MDS_3D_10.lift_coeff[mask_10_updated2, :]
# static_coeff_MDS_3D_10.pitch_coeff = static_coeff_MDS_3D_10.pitch_coeff[mask_10_updated2, :]




static_coeff_MDS_3D_6_updated.drag_coeff[:, 2] = static_coeff_MDS_3D_6.drag_coeff[:, 2]
static_coeff_MDS_3D_6_updated.drag_coeff[:, 3] = static_coeff_MDS_3D_6.drag_coeff[:, 3]
static_coeff_MDS_3D_6_updated.lift_coeff[:, 2] = static_coeff_MDS_3D_6.lift_coeff[:, 2]
static_coeff_MDS_3D_6_updated.lift_coeff[:, 3] = static_coeff_MDS_3D_6.lift_coeff[:, 3]
static_coeff_MDS_3D_6_updated.pitch_coeff[:,2] = static_coeff_MDS_3D_6.pitch_coeff[:, 2]
static_coeff_MDS_3D_6_updated.pitch_coeff[:, 3] = static_coeff_MDS_3D_6.pitch_coeff[:, 3]

# static_coeff_MDS_3D_8_updated.drag_coeff[:, 2] = static_coeff_MDS_3D_8_reg_new.drag_coeff[:, 2]  
# static_coeff_MDS_3D_8_updated.drag_coeff[:, 3] = static_coeff_MDS_3D_8_reg_new.drag_coeff[:, 3]  
# static_coeff_MDS_3D_8_updated.lift_coeff[:, 2] = static_coeff_MDS_3D_8.lift_coeff[:, 2]
# static_coeff_MDS_3D_8_updated.lift_coeff[:, 3] = static_coeff_MDS_3D_8.lift_coeff[:, 3]
# static_coeff_MDS_3D_8_updated.pitch_coeff[:, 2] = static_coeff_MDS_3D_8.pitch_coeff[:,2]
# static_coeff_MDS_3D_8_updated.pitch_coeff[:, 3] = static_coeff_MDS_3D_8.pitch_coeff[:, 3]


static_coeff_MDS_3D_10_updated.drag_coeff[:, 2] = static_coeff_MDS_3D_10.drag_coeff[:, 2] 
static_coeff_MDS_3D_10_updated.drag_coeff[:, 3] = static_coeff_MDS_3D_10.drag_coeff[:, 3] 
static_coeff_MDS_3D_10_updated.lift_coeff[:, 2] = static_coeff_MDS_3D_10.lift_coeff[:, 2]
static_coeff_MDS_3D_10_updated.lift_coeff[:, 3] = static_coeff_MDS_3D_10.lift_coeff[:, 3]
static_coeff_MDS_3D_10_updated.pitch_coeff[:, 2] = static_coeff_MDS_3D_10.pitch_coeff[:,2]
static_coeff_MDS_3D_10_updated.pitch_coeff[:, 3] = static_coeff_MDS_3D_10.pitch_coeff[:, 3]

cd_6_mean = static_coeff_MDS_3D_6_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True, ax = None)[1]
cd_10_mean = static_coeff_MDS_3D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[1]
alpha = static_coeff_MDS_3D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[2]

for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_6_mean[i])
        print(cd_10_mean[i])
#%% !?!!!
def plot_compare_wind_speeds_mean_seperate3dmDs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,2}(\alpha_2)$"
        coeff = "drag_coeff"
        min = 0.42#0.4
        max = 0.55#0.58
    elif scoff == "lift":
        axis = r"$C_{L,2}(\alpha_2)$"
        coeff = "lift_coeff"
        min = -0.3#-0.35
        max = 0.28#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,2}(\alpha_2)$"
        coeff = "pitch_coeff"
        min = -0.035#-0.05
        max = 0.09#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    # ax.plot(unique_alphas_low, upwind_mean_low,
    #          label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    ax.plot(unique_alphas_low, downwind_mean_low,
             label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        # ax.plot(unique_alphas_med, upwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        ax.plot(unique_alphas_med, downwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    # ax.plot(unique_alphas_high, upwind_mean_high,
    #             label=f"10 m/s", color ="#d62728", alpha = 0.8)
    ax.plot(unique_alphas_high, downwind_mean_high,
                label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MDS3D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate3dmDs(static_coeff_MDS_3D_6_updated, static_coeff_MDS_3D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mds_3D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate3dmDs(static_coeff_MDS_3D_6_updated, static_coeff_MDS_3D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mds_3D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate3dmDs(static_coeff_MDS_3D_6_updated, static_coeff_MDS_3D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mds_3D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)

#%%
# Compare windspeed
section_name = "3D_MUS_Static_updated"
w3t._scoff.plot_compare_wind_speeds_mean_seperate(static_coeff_MDS_3D_6_updated, static_coeff_MDS_3D_10_updated,static_coeff_med = static_coeff_MDS_3D_8_updated,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + "_drag.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
w3t._scoff.plot_compare_wind_speeds_mean_seperate(static_coeff_MDS_3D_6_updated, static_coeff_MDS_3D_10_updated,static_coeff_med = static_coeff_MDS_3D_8_updated,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + "_lift.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
w3t._scoff.plot_compare_wind_speeds_mean_seperate(static_coeff_MDS_3D_6_updated, static_coeff_MDS_3D_10_updated,static_coeff_med = static_coeff_MDS_3D_8_updated,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + "_pitch.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)


#%% !!!
# Load all upwind experiments (upwind in rig)

section_name = "MDS_3D_Static"
file_names_MUS_3D_5 = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_004"] # 5 m/s, vibrations (Finnes en fil for 6 også)
file_names_MUS_3D_8 = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_006"] # 8 m/s, vibrations
file_names_MUS_3D_10 = ["HAR_INT_MDS_GAP_213D_02_02_000","HAR_INT_MDS_GAP_213D_02_02_005"] # 10 m/s, vibrations

exp0_MUS_3D, exp1_MUS_3D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_3D_5,  upwind_in_rig=True)
exp0_MUS_3D, exp1_MUS_3D_8 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_3D_8,  upwind_in_rig=True)
exp0_MUS_3D, exp1_MUS_3D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_3D_10,  upwind_in_rig=True)

# exp0_MUS_3D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_0.png"))
# exp1_MUS_3D_5.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 5 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_5.png"))
# exp1_MUS_3D_8.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 8 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_8.png"))
# exp1_MUS_3D_10.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 10 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_10.png"))

exp0_MUS_3D.filt_forces(6, 2)
exp1_MUS_3D_5.filt_forces(6, 2)
exp1_MUS_3D_8.filt_forces(6, 2)
exp1_MUS_3D_10.filt_forces(6, 2)

# exp0_MUS_3D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_0_filter.png"))
# exp1_MUS_3D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 5 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_5_filter.png"))
# exp1_MUS_3D_8.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 8 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_8_filter.png"))
# exp1_MUS_3D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 3D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_3D_10_filter.png"))
# plt.show()
# plt.close()

static_coeff_MUS_3D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_3D, exp1_MUS_3D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_3D_8 = w3t.StaticCoeff.fromWTT(exp0_MUS_3D, exp1_MUS_3D_8, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_3D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_3D, exp1_MUS_3D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_3D_5, section_name, 5, mode="decks", upwind_in_rig=True)
#plot_static_coeff_summary(static_coeff_MUS_3D_8, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_3D_10, section_name, 10, mode="decks", upwind_in_rig=True)


#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_3D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_3D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_3D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_3D_low_cm" + ".png"), dpi=300)

fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_3D_high_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_3D_high_cl" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_3D_high_cm" + ".png"), dpi=300)


#%%
# filter regression
print("regRESSION MDS")
section_name = "3D_MDS_Static_reg"
static_coeff_MUS_3D_5_reg = copy.deepcopy(static_coeff_MUS_3D_5)

alpha_MUS_3D_5_reg, coeff_MUS_3D_5_drag_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_5, scoff="drag", single=False)
static_coeff_MUS_3D_5_reg.drag_coeff[:, 0] = coeff_MUS_3D_5_drag_reg_up / 2
static_coeff_MUS_3D_5_reg.drag_coeff[:, 1] = coeff_MUS_3D_5_drag_reg_up / 2
alpha_MUS_3D_5_reg, coeff_MUS_3D_5_lift_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_5, scoff="lift", single=False)
static_coeff_MUS_3D_5_reg.lift_coeff[:, 0] = coeff_MUS_3D_5_lift_reg_up / 2
static_coeff_MUS_3D_5_reg.lift_coeff[:, 1] = coeff_MUS_3D_5_lift_reg_up / 2
alpha_MUS_3D_5_reg, coeff_MUS_3D_5_pitch_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_5, scoff="pitch", single=False)
static_coeff_MUS_3D_5_reg.pitch_coeff[:, 0] = coeff_MUS_3D_5_pitch_reg_up / 2
static_coeff_MUS_3D_5_reg.pitch_coeff[:, 1] = coeff_MUS_3D_5_pitch_reg_up / 2
plot_static_coeff_summary(static_coeff_MUS_3D_5_reg, section_name, 5, mode="decks", upwind_in_rig=True)

static_coeff_MUS_3D_8_reg = copy.deepcopy(static_coeff_MUS_3D_8)


alpha_MDS_3D_8_reg, coeff_MUS_3D_8_drag_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_8, scoff="drag", single=False)
static_coeff_MUS_3D_8_reg.drag_coeff[:, 0] = coeff_MUS_3D_8_drag_reg_up / 2
static_coeff_MUS_3D_8_reg.drag_coeff[:, 1] = coeff_MUS_3D_8_drag_reg_up / 2
alpha_MDS_3D_8_reg, coeff_MUS_3D_8_lift_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_8, scoff="lift", single=False)
static_coeff_MUS_3D_8_reg.lift_coeff[:, 0] = coeff_MUS_3D_8_lift_reg_up / 2
static_coeff_MUS_3D_8_reg.lift_coeff[:, 1] = coeff_MUS_3D_8_lift_reg_up / 2
alpha_MDS_3D_8_reg, coeff_MUS_3D_8_pitch_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_8, scoff="pitch", single=False)
static_coeff_MUS_3D_8_reg.pitch_coeff[:, 0] = coeff_MUS_3D_8_pitch_reg_up / 2
static_coeff_MUS_3D_8_reg.pitch_coeff[:, 1] = coeff_MUS_3D_8_pitch_reg_up / 2
plot_static_coeff_summary(static_coeff_MUS_3D_8_reg, section_name, 8, mode="decks", upwind_in_rig=True)

static_coeff_MUS_3D_10_reg = copy.deepcopy(static_coeff_MUS_3D_10)

alpha_MUS_3D_10_reg, coeff_MUS_3D_10_drag_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_10, scoff="drag", single=False)
static_coeff_MUS_3D_10_reg.drag_coeff[:, 0] = coeff_MUS_3D_10_drag_reg_up / 2
static_coeff_MUS_3D_10_reg.drag_coeff[:, 1] = coeff_MUS_3D_10_drag_reg_up / 2
alpha_MUS_3D_10_reg, coeff_MUS_3D_10_lift_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_10, scoff="lift", single=False)
static_coeff_MUS_3D_10_reg.lift_coeff[:, 0] = coeff_MUS_3D_10_lift_reg_up / 2
static_coeff_MUS_3D_10_reg.lift_coeff[:, 1] = coeff_MUS_3D_10_lift_reg_up / 2
alpha_MUS_3D_10_reg, coeff_MUS_3D_10_pitch_reg_up,_ = w3t._scoff.poly_estimat(static_coeff_MUS_3D_10, scoff="pitch", single=False)
static_coeff_MUS_3D_10_reg.pitch_coeff[:, 0] = coeff_MUS_3D_10_pitch_reg_up / 2
static_coeff_MUS_3D_10_reg.pitch_coeff[:, 1] = coeff_MUS_3D_10_pitch_reg_up / 2
plot_static_coeff_summary(static_coeff_MUS_3D_10_reg, section_name, 10, mode="decks", upwind_in_rig=True)

section_name = "MDS_3D_Static"

#%% 
print("FILTER 1")

# Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_3D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_3D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_3D_5_drag_clean.png"))

alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_3D_8, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_3D_Static, 8 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_3D_8_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_3D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_3D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_3D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_3D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_3D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_3D_5_lift_clean.png"))
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_3D_8, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_3D_Static, 8 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_3D_8_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_3D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_3D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_3D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_3D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_3D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_3D_5_pitch_clean.png"))
alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_3D_8, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_3D_Static, 8 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_3D_8_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_3D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_3D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_3D_10_pitch_clean.png"))


#%% !!!
#   Filter and plot ALT 2
print("FILTER 2")

section_name = "MDS_3D_Static_filtered"

static_coeff_MUS_3D_5_filtered, static_coeff_MUS_3D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_3D_5, static_coeff_2=static_coeff_MUS_3D_10, threshold=0.05, threshold_low=[0.05,0.03,0.005],threshold_high=[0.04,0.03,0.005],single=False)
static_coeff_MUS_3D_5_filtered, static_coeff_MUS_3D_8_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_3D_5, static_coeff_2=static_coeff_MUS_3D_8, threshold=0.05, threshold_low=[0.05,0.03,0.005],threshold_high=[0.04,0.03,0.005],single=False)


#plot_static_coeff_summary(static_coeff_MUS_3D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
#plot_static_coeff_summary(static_coeff_MUS_3D_8_filtered, section_name, 8, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_3D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)

#%% !!!
# Summary
print("her det skjers")
static_coeff_MUS_3D_5_updated = copy.deepcopy(static_coeff_MUS_3D_5)
static_coeff_MUS_3D_5_updated.drag_coeff[:, 0] = static_coeff_MUS_3D_5.drag_coeff[:, 0]
static_coeff_MUS_3D_5_updated.drag_coeff[:, 1] = static_coeff_MUS_3D_5.drag_coeff[:, 1]
static_coeff_MUS_3D_5_updated.lift_coeff[:, 0] = static_coeff_MUS_3D_5.lift_coeff[:, 0]
static_coeff_MUS_3D_5_updated.lift_coeff[:, 1] = static_coeff_MUS_3D_5.lift_coeff[:, 1]
static_coeff_MUS_3D_5_updated.pitch_coeff[:, 0] = static_coeff_MUS_3D_5.pitch_coeff[:, 0]
static_coeff_MUS_3D_5_updated.pitch_coeff[:, 1] = static_coeff_MUS_3D_5.pitch_coeff[:, 1]

# static_coeff_MUS_3D_8_updated = copy.deepcopy(static_coeff_MUS_3D_8)
# static_coeff_MUS_3D_8_updated.drag_coeff[:, 0] = static_coeff_MUS_3D_8_filtered.drag_coeff[:, 0] #filt 2
# static_coeff_MUS_3D_8_updated.drag_coeff[:, 1] = static_coeff_MUS_3D_8_filtered.drag_coeff[:, 1] #filt 2
# static_coeff_MUS_3D_8_updated.lift_coeff[:, 0] = static_coeff_MUS_3D_8.lift_coeff[:, 0]
# static_coeff_MUS_3D_8_updated.lift_coeff[:, 1] = static_coeff_MUS_3D_8.lift_coeff[:, 1]
# static_coeff_MUS_3D_8_updated.pitch_coeff[:, 0] = static_coeff_MUS_3D_8.pitch_coeff[:, 0]
# static_coeff_MUS_3D_8_updated.pitch_coeff[:, 1] = static_coeff_MUS_3D_8.pitch_coeff[:, 1]


static_coeff_MUS_3D_10_updated = copy.deepcopy(static_coeff_MUS_3D_10)
static_coeff_MUS_3D_10_updated.drag_coeff[:, 0] = static_coeff_MUS_3D_10_filtered.drag_coeff[:, 0] # filt 2
static_coeff_MUS_3D_10_updated.drag_coeff[:, 1] = static_coeff_MUS_3D_10_filtered.drag_coeff[:, 1] # filt 2
static_coeff_MUS_3D_10_updated.lift_coeff[:, 0] = static_coeff_MUS_3D_10.lift_coeff[:, 0]
static_coeff_MUS_3D_10_updated.lift_coeff[:, 1] = static_coeff_MUS_3D_10.lift_coeff[:, 1]
static_coeff_MUS_3D_10_updated.pitch_coeff[:, 0] = static_coeff_MUS_3D_10.pitch_coeff[:, 0]
static_coeff_MUS_3D_10_updated.pitch_coeff[:, 1] = static_coeff_MUS_3D_10.pitch_coeff[:, 1]

cd_5_mean = static_coeff_MUS_3D_5_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True, ax = None)[0]
cd_10_mean = static_coeff_MUS_3D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[0]
alpha = static_coeff_MUS_3D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[2]

for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_5_mean[i])
        print(cd_10_mean[i])

#%% !?!!!
def plot_compare_wind_speeds_mean_seperate3dmUs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,1}(\alpha_1)$"
        coeff = "drag_coeff"
        min = 0.41#0.4
        max = 0.54#0.58
    elif scoff == "lift":
        axis = r"$C_{L,1}(\alpha_1)$"
        coeff = "lift_coeff"
        min = -0.3#-0.35
        max = 0.42#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,1}(\alpha_1)$"
        coeff = "pitch_coeff"
        min = -0.05#-0.05
        max = 0.11#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    # ax.plot(unique_alphas_low, downwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        ax.plot(unique_alphas_med, upwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        # ax.plot(unique_alphas_med, downwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"10 m/s", color ="#d62728", alpha = 0.8)
    # ax.plot(unique_alphas_high, downwind_mean_high,
    #             label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MUS3D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate3dmUs(static_coeff_MUS_3D_5_updated, static_coeff_MUS_3D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mus_3D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate3dmUs(static_coeff_MUS_3D_5_updated, static_coeff_MUS_3D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mus_3D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate3dmUs(static_coeff_MUS_3D_5_updated, static_coeff_MUS_3D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mus_3D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)


#%%
# Compare windspeed
section_name = "3D_MDS_Static_updated"
w3t._scoff.plot_compare_wind_speeds_mean_seperate(static_coeff_MUS_3D_5_updated, static_coeff_MUS_3D_10_updated,static_coeff_med = static_coeff_MUS_3D_8_updated,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + "_drag.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
w3t._scoff.plot_compare_wind_speeds_mean_seperate(static_coeff_MUS_3D_5_updated, static_coeff_MUS_3D_10_updated,static_coeff_med = static_coeff_MUS_3D_8_updated,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + "_lift.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
w3t._scoff.plot_compare_wind_speeds_mean_seperate(static_coeff_MUS_3D_5_updated, static_coeff_MUS_3D_10_updated,static_coeff_med = static_coeff_MUS_3D_8_updated,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + "_pitch.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)

#husk å kommentere inn/ut downwind/upwind!

#%%
#  Save all experiments to excel
section_name = "3D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_3D_6.to_excel(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_5.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_3D_8.to_excel(section_name, sheet_name="MDS - 8" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_3D_8.to_excel(section_name, sheet_name='MUS - 8' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_3D_10.to_excel(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_10.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "3D_mean"

# Low wind speed
static_coeff_MDS_3D_6.to_excel_mean(section_name, sheet_name="MDS - 6" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_5.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_3D_8.to_excel_mean(section_name, sheet_name="MDS - 8" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_3D_8.to_excel_mean(section_name, sheet_name='MUS - 8' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_3D_10.to_excel_mean(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_3D_10.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%%
#  Save all experiments to excel filtered
section_name = "3D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MUS_3D_5_filtered.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_3D_10_filtered.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "3D_mean_filtered"

# Low wind speed
static_coeff_MUS_3D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_3D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% 
# Compare all experiments (MUS vs MDS vs Single)
section_name = "3D"

#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_low_drag" + ".png"))

w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_low_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_low_pitch" + ".png"))
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_3D_5_filtered, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_low_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_low_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, static_coeff_MDS_3D_6)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_low_pitch_mean" + ".png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_3D_8, static_coeff_MDS_3D_8)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s, MDS: 8 m/s", fontsize=16)
#%%

#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_high_drag" + ".png"))

w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_high_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_high_pitch" + ".png"))

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_high_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_high_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_3D_10, static_coeff_MDS_3D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "3D_high_pitch_mean" + ".png"))

plt.show()
plt.close()

#%% 
# Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_3D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_low_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_3D_low_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_low_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_low_lift.png"))


w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_low_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_low_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_3D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_low_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_3D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_3D_low_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_low_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_3D_low_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_3D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_low_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_3D_6, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 6 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_3D_low_pitch_mean.png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_3D_8,  upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_8,upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_8,upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_8, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_8, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8 m/s", fontsize=16)
#%%                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_high_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_high_drag.png"))

w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_high_lift.png"))

w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_high_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_3D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_high_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_3D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_high_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_3D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_high_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_high_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_high_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_high_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_3D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_3D_high_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_3D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_3D_high_pitch_mean.png"))
plt.show()
plt.close()

# %% Compare all experiments (Wind speed)
#drag

# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                                static_coeff_MUS_3D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"3D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_3D_drag.png"))


# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6,
                                static_coeff_MDS_3D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"3D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_3D_drag.png"))

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                                static_coeff_MUS_3D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_3D_lift.png"))

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9,static_coeff_MDS_3D_6,
                                static_coeff_MDS_3D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_3D_lift.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                                static_coeff_MUS_3D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_3D_pitch.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                                static_coeff_MDS_3D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_3D_pitch.png"))

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_3D_5_filtered,
                             static_coeff_MUS_3D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"3D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_3D_drag_mean.png"))

# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_3D_6,
                                static_coeff_MDS_3D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"3D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_3D_drag_mean.png"))
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                                static_coeff_MUS_3D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_3D_lift_mean.png"))


#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                                static_coeff_MDS_3D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"3D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_3D_lift_mean.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_3D_5,
                                static_coeff_MUS_3D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_3D_pitch_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_3D_6,
                                static_coeff_MDS_3D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"3D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_3D_pitch_mean.png"))


############################################################################################################
#print("4D")

#%% !!!
# Load all downwind experiments (downwind in rig)
section_name = "MUS_4D_Static"
file_names_MDS_4D_55 = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_002"] #5.5 m/s
#file_names_MDS_4D_85 = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_003"] # 8.5 m/s, vibrations
file_names_MDS_4D_10 = ["HAR_INT_MUS_GAP_45D_02_00_000","HAR_INT_MUS_GAP_45D_02_00_004"] # 10 m/s

exp0_MDS_4D, exp1_MDS_4D_55 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_4D_55,  upwind_in_rig=False)
#exp0_MDS_4D, exp1_MDS_4D_85= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_4D_85,  upwind_in_rig=False)
exp0_MDS_4D, exp1_MDS_4D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_4D_10,  upwind_in_rig=False)



# exp0_MDS_4D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 4D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_0.png"))
# exp1_MDS_4D_55.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 4D - Wind speed: 5.5 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_55.png"))
# #exp1_MDS_4D_85.plot_experiment(mode="total") #
# #plt.gcf().suptitle(f"MDS 4D - Wind speed: 8.5 m/s ",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_85.png"))
# exp1_MDS_4D_10.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 4D - Wind speed: 10 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_10.png"))
# plt.show()
# plt.close()

exp0_MDS_4D.filt_forces(6, 2)
exp1_MDS_4D_55.filt_forces(6, 2)
#exp1_MDS_4D_85.filt_forces(6, 2)
exp1_MDS_4D_10.filt_forces(6, 2)

# exp0_MDS_4D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 4D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_0_filter.png"))
# exp1_MDS_4D_55.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 4D - Wind speed: 5.5 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_55_filter.png"))
# #exp1_MDS_4D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
# #plt.gcf().suptitle(f"MDS 4D - Wind speed: 8.5 m/s - With Butterworth low-pass filter",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_85_filter.png"))
# exp1_MDS_4D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 4D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_4D_10_filter.png"))
# plt.show()
# plt.close()


static_coeff_MDS_4D_55 =w3t.StaticCoeff.fromWTT(exp0_MDS_4D, exp1_MDS_4D_55, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

#static_coeff_MDS_4D_85 = w3t.StaticCoeff.fromWTT(exp0_MDS_4D, exp1_MDS_4D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_4D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_4D, exp1_MDS_4D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)


plot_static_coeff_summary(static_coeff_MDS_4D_55, section_name, 5.5, mode="decks", upwind_in_rig=False)
#plot_static_coeff_summary(static_coeff_MDS_4D_85, section_name, 8.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_4D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_4D_55, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_4D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_4D_55, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_4D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_4D_55, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_4D_low_cm" + ".png"), dpi=300)

fig, ax =w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_4D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_4D_high_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_4D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_4D_high_cl" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_4D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_4D_high_cm" + ".png"), dpi=300)

#%% 
# Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_4D_55, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_4D_Static, 5.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_4D_55_drag_clean.png"))

#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_4D_85, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
#plt.suptitle(f"MDS_4D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_4D_85_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_4D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_4D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_4D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_4D_55, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_4D_Static, 5.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_4D_55_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_4D_85, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
#plt.suptitle(f"MDS_4D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_4D_85_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_4D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_4D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_4D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_4D_55, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_4D_Static, 5.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_4D_55_pitch_clean.png"))
#, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_4D_85, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MDS_4D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_4D_85_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_4D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_4D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_4D_10_pitch_clean.png"))



#%% !!!
# Summary

static_coeff_MDS_4D_55_updated = copy.deepcopy(static_coeff_MDS_4D_55)
static_coeff_MDS_4D_55_updated.drag_coeff[:, 2] = static_coeff_MDS_4D_55.drag_coeff[:, 2]
static_coeff_MDS_4D_55_updated.drag_coeff[:, 3] = static_coeff_MDS_4D_55.drag_coeff[:, 3]
static_coeff_MDS_4D_55_updated.lift_coeff[:, 2] = static_coeff_MDS_4D_55.lift_coeff[:, 2]
static_coeff_MDS_4D_55_updated.lift_coeff[:, 3] = static_coeff_MDS_4D_55.lift_coeff[:, 3]
static_coeff_MDS_4D_55_updated.pitch_coeff[:, 2] = static_coeff_MDS_4D_55.pitch_coeff[:,2]
static_coeff_MDS_4D_55_updated.pitch_coeff[:, 3] = static_coeff_MDS_4D_55.pitch_coeff[:, 3]

static_coeff_MDS_4D_10_updated = copy.deepcopy(static_coeff_MDS_4D_10)
 
static_coeff_MDS_4D_10_updated.drag_coeff[:, 2] = static_coeff_MDS_4D_10.drag_coeff[:, 2] 
static_coeff_MDS_4D_10_updated.drag_coeff[:, 3] = static_coeff_MDS_4D_10.drag_coeff[:, 3] 
static_coeff_MDS_4D_10_updated.lift_coeff[:, 2] = static_coeff_MDS_4D_10.lift_coeff[:, 2]
static_coeff_MDS_4D_10_updated.lift_coeff[:, 3] = static_coeff_MDS_4D_10.lift_coeff[:, 3]
static_coeff_MDS_4D_10_updated.pitch_coeff[:, 2] = static_coeff_MDS_4D_10.pitch_coeff[:,2]
static_coeff_MDS_4D_10_updated.pitch_coeff[:, 3] = static_coeff_MDS_4D_10.pitch_coeff[:, 3]

cd_55_mean = static_coeff_MDS_4D_55_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True, ax = None)[1]
cd_10_mean = static_coeff_MDS_4D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[1]
alpha = static_coeff_MDS_4D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[2]


for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_55_mean[i])
        print(cd_10_mean[i])
#%% !?!!!
def plot_compare_wind_speeds_mean_seperate4dmDs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,2}(\alpha_2)$"
        coeff = "drag_coeff"
        min = 0.4#0.4
        max = 0.505#0.58
    elif scoff == "lift":
        axis = r"$C_{L,2}(\alpha_2)$"
        coeff = "lift_coeff"
        min = -0.27#-0.35
        max = 0.37#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,2}(\alpha_2)$"
        coeff = "pitch_coeff"
        min = -0.04#-0.05
        max = 0.09#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    # ax.plot(unique_alphas_low, upwind_mean_low,
    #          label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    ax.plot(unique_alphas_low, downwind_mean_low,
             label=f"5,5 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        # ax.plot(unique_alphas_med, upwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        ax.plot(unique_alphas_med, downwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    # ax.plot(unique_alphas_high, upwind_mean_high,
    #             label=f"10 m/s", color ="#d62728", alpha = 0.8)
    ax.plot(unique_alphas_high, downwind_mean_high,
                label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MDS4D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate4dmDs(static_coeff_MDS_4D_55_updated, static_coeff_MDS_4D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mds_4D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate4dmDs(static_coeff_MDS_4D_55_updated, static_coeff_MDS_4D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mds_4D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate4dmDs(static_coeff_MDS_4D_55_updated, static_coeff_MDS_4D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mds_4D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)



#%% !!!
# Load all upwind experiments (upwind in rig)

section_name = "MDS_4D_Static"
file_names_MUS_4D_5 = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_003"] # 5 m/s, vibrations 
#file_names_MUS_4D_85= ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_004"] # 8.5 m/s, vibrations
file_names_MUS_4D_10 = ["HAR_INT_MDS_GAP_45D_02_00_001","HAR_INT_MDS_GAP_45D_02_00_005"] # 10 m/s, vibrations


exp0_MUS_4D, exp1_MUS_4D_5= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_4D_5,  upwind_in_rig=True)
#exp0_MUS_4D, exp1_MUS_4D_85 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_4D_85,  upwind_in_rig=True)
exp0_MUS_4D, exp1_MUS_4D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_4D_10,  upwind_in_rig=True)


# exp0_MUS_4D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 4D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_0.png"))
# exp1_MUS_4D_5.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 4D - Wind speed: 5 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_5.png"))
# #exp1_MUS_4D_85.plot_experiment(mode="total") #
# #plt.gcf().suptitle(f"MUS 4D - Wind speed: 8.5 m/s ",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_85.png"))
# exp1_MUS_4D_10.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 4D - Wind speed: 10 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_10.png"))
# plt.show()
# plt.close()

exp0_MUS_4D.filt_forces(6, 2)
exp1_MUS_4D_5.filt_forces(6, 2)
#exp1_MUS_4D_85.filt_forces(6, 2)
exp1_MUS_4D_10.filt_forces(6, 2)

# exp0_MUS_4D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 4D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_0_filter.png"))
# exp1_MUS_4D_5.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 4D - Wind speed: 5 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_5_filter.png"))
# #exp1_MUS_4D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
# #plt.gcf().suptitle(f"MUS 4D - Wind speed: 8.5 m/s - With Butterworth low-pass filter",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_85_filter.png"))
# exp1_MUS_4D_10.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 4D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_4D_10_filter.png"))
# plt.show()
# plt.close()

static_coeff_MUS_4D_5 =w3t.StaticCoeff.fromWTT(exp0_MUS_4D, exp1_MUS_4D_5, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

#static_coeff_MUS_4D_85 = w3t.StaticCoeff.fromWTT(exp0_MUS_4D, exp1_MUS_4D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_4D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_4D, exp1_MUS_4D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_4D_5, section_name, 5, mode="decks", upwind_in_rig=True)
#plot_static_coeff_summary(static_coeff_MUS_4D_85, section_name, 8.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_4D_10, section_name, 10, mode="decks", upwind_in_rig=True)

#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_4D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_4D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_4D_5, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_4D_low_cm" + ".png"), dpi=300)

fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_4D_high_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_4D_high_cl" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_4D_high_cm" + ".png"), dpi=300)



#%%
#  Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_4D_5, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_4D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_4D_5_drag_clean.png"))

#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_4D_85, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
#plt.suptitle(f"MUS_4D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_4D_85_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_4D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_4D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_4D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_4D_5, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_4D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_4D_5_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_4D_85, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
#plt.suptitle(f"MUS_4D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_4D_85_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_4D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_4D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_4D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_4D_5, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_4D_Static, 5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_4D_5_pitch_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_4D_85, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MUS_4D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_4D_85_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_4D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_4D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_4D_10_pitch_clean.png"))



#%% !!!
#  Filter and plot ALT 2
section_name = "MDS_4D_Static_filtered"

static_coeff_MUS_4D_5_filtered, static_coeff_MUS_4D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_4D_5, static_coeff_2=static_coeff_MUS_4D_10, threshold=0.2, threshold_low=[0.04,0.025,0.005],threshold_high=[0.04,0.025,0.005],single=False)


#plot_static_coeff_summary(static_coeff_MUS_4D_5_filtered, section_name, 5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_4D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)


#%% !!!
#Summary

static_coeff_MUS_4D_5_updated = copy.deepcopy(static_coeff_MUS_4D_5)
static_coeff_MUS_4D_5_updated.drag_coeff[:, 0] = static_coeff_MUS_4D_5.drag_coeff[:, 0]
static_coeff_MUS_4D_5_updated.drag_coeff[:, 1] = static_coeff_MUS_4D_5.drag_coeff[:, 1]
static_coeff_MUS_4D_5_updated.lift_coeff[:, 0] = static_coeff_MUS_4D_5.lift_coeff[:, 0]
static_coeff_MUS_4D_5_updated.lift_coeff[:, 1] = static_coeff_MUS_4D_5.lift_coeff[:, 1]
static_coeff_MUS_4D_5_updated.pitch_coeff[:, 0] = static_coeff_MUS_4D_5.pitch_coeff[:,0]
static_coeff_MUS_4D_5_updated.pitch_coeff[:, 1] = static_coeff_MUS_4D_5.pitch_coeff[:, 1]

static_coeff_MUS_4D_10_updated = copy.deepcopy(static_coeff_MUS_4D_10)
 
static_coeff_MUS_4D_10_updated.drag_coeff[:, 0] = static_coeff_MUS_4D_10_filtered.drag_coeff[:, 0] 
static_coeff_MUS_4D_10_updated.drag_coeff[:, 1] = static_coeff_MUS_4D_10_filtered.drag_coeff[:, 1] 
static_coeff_MUS_4D_10_updated.lift_coeff[:, 0] = static_coeff_MUS_4D_10.lift_coeff[:, 0]
static_coeff_MUS_4D_10_updated.lift_coeff[:, 1] = static_coeff_MUS_4D_10.lift_coeff[:, 1]
static_coeff_MUS_4D_10_updated.pitch_coeff[:, 0] = static_coeff_MUS_4D_10.pitch_coeff[:,0]
static_coeff_MUS_4D_10_updated.pitch_coeff[:, 1] = static_coeff_MUS_4D_10.pitch_coeff[:, 1]

cd_5_mean = static_coeff_MUS_4D_5_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True, ax = None)[0]
cd_10_mean = static_coeff_MUS_4D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[0]
alpha = static_coeff_MUS_4D_10_updated.plot_drag_mean( mode ="decks", upwind_in_rig=True,ax  = None)[2]


for i, vinkel in enumerate(alpha):
    if np.isclose(vinkel, 0):
        print(cd_5_mean[i])
        print(cd_10_mean[i])
#%% !?!!!
def plot_compare_wind_speeds_mean_seperate4dmUs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,1}(\alpha_1)$"
        coeff = "drag_coeff"
        min = 0.45#0.4
        max = 0.65#0.58
    elif scoff == "lift":
        axis = r"$C_{L,1}(\alpha_1)$"
        coeff = "lift_coeff"
        min = -0.32#-0.35
        max = 0.48#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,1}(\alpha_1)$"
        coeff = "pitch_coeff"
        min = -0.05#-0.05
        max = 0.14#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    # ax.plot(unique_alphas_low, downwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        ax.plot(unique_alphas_med, upwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        # ax.plot(unique_alphas_med, downwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"10 m/s", color ="#d62728", alpha = 0.8)
    # ax.plot(unique_alphas_high, downwind_mean_high,
    #             label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MUS4D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate4dmUs(static_coeff_MUS_4D_5_updated, static_coeff_MUS_4D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mus_4D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate4dmUs(static_coeff_MUS_4D_5_updated, static_coeff_MUS_4D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mus_4D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate4dmUs(static_coeff_MUS_4D_5_updated, static_coeff_MUS_4D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mus_4D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)



#%%
#  Save all experiments to excel
section_name = "4D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_4D_55.to_excel(section_name, sheet_name="MDS - 5.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_5.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_4D_85.to_excel(section_name, sheet_name="MDS - 8.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_4D_85.to_excel(section_name, sheet_name='MUS - 8.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_4D_10.to_excel(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_10.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "4D_mean"

# Low wind speed
static_coeff_MDS_4D_55.to_excel_mean(section_name, sheet_name="MDS - 5.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_5.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_4D_85.to_excel_mean(section_name, sheet_name="MDS - 8.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_4D_85.to_excel_mean(section_name, sheet_name='MUS - 8.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_4D_10.to_excel_mean(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_4D_10.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%% 
# Save all experiments to excel filtered
section_name = "4D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MUS_4D_5_filtered.to_excel(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_4D_10_filtered.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "4D_mean_filtered"

# Low wind speed
static_coeff_MUS_4D_5_filtered.to_excel_mean(section_name, sheet_name='MUS - 5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_4D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%% 
# Compare all experiments (MUS vs MDS vs Single)
section_name = "4D"


#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_low_drag" + ".png"))

w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_low_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_low_pitch" + ".png"))
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_4D_5_filtered, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_low_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_low_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, static_coeff_MDS_4D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_low_pitch_mean" + ".png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_4D_85, static_coeff_MDS_4D_85)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
#%%

#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_high_drag" + ".png"))

w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_high_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_high_pitch" + ".png"))

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_high_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_high_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_4D_10, static_coeff_MDS_4D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "4D_high_pitch_mean" + ".png"))

plt.show()
plt.close()

#%%
#  Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_4D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_low_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_low_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_low_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_low_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_low_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_low_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_4D_5_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_low_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_4D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_4D_low_drag_mean.png"))

w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_low_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_low_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_4D_5, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_low_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_4D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_low_pitch_mean.png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_4D_85,  upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_85,upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_85,upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
#%%                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_high_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_high_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_high_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_high_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_4D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_high_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_4D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_high_pitch.png"))



# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_4D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_high_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MDS_4D_high_drag_mean.png"))

w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_high_lift_mean.png"))

w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_high_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_4D_10,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_4D_high_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_4D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_4D_high_pitch_mean.png"))
plt.show()
plt.close()


# %% 
# Compare all experiments (Wind speed)
#drag
# MUS

w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                                static_coeff_MUS_4D_10_filtered,
                             scoff = "drag")                        
plt.gcf().suptitle(f"4D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_4D_drag.png"))


# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55,
                                static_coeff_MDS_4D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"4D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_4D_drag.png"))

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                                static_coeff_MUS_4D_10,
                            scoff = "lift")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_4D_lift.png"))

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9,static_coeff_MDS_4D_55,
                                static_coeff_MDS_4D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_4D_lift.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                                static_coeff_MUS_4D_10,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_4D_pitch.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                                static_coeff_MDS_4D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_4D_pitch.png"))

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_4D_5_filtered,
                                static_coeff_MUS_4D_10_filtered,
                           scoff = "drag")                        
plt.gcf().suptitle(f"4D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_4D_drag_mean.png"))

# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_4D_55,
                                static_coeff_MDS_4D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"4D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_4D_drag_mean.png"))
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                                static_coeff_MUS_4D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_4D_lift_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                                static_coeff_MDS_4D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"4D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_4D_lift_mean.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_4D_5,
                                static_coeff_MUS_4D_10,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_4D_pitch_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_4D_55,
                                static_coeff_MDS_4D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"4D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_4D_pitch_mean.png"))



############################################################################################################
#print("5D")


#%% !!!
# Load all downwind experiments (downwind in rig)
section_name = "MUS_5D_Static"
file_names_MDS_5D_55 = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_001"] # 5.5 m/s
#file_names_MDS_5D_85 = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_002"] # 8.5 m/s, vibrations
file_names_MDS_5D_10 = ["HAR_INT_MUS_GAP_45D_02_01_000","HAR_INT_MUS_GAP_45D_02_01_003"] # 10 m/s

exp0_MDS_5D, exp1_MDS_5D_55 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_5D_55,  upwind_in_rig=False)
#exp0_MDS_5D, exp1_MDS_5D_85= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_5D_85,  upwind_in_rig=False)
exp0_MDS_5D, exp1_MDS_5D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MDS_5D_10,  upwind_in_rig=False)




# exp0_MDS_5D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MDS 5D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_5D_0.png"))
exp1_MDS_5D_55.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MDS 5D - Wind speed: 5.5 m/s ",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_5D_55_tidserie" + ".png"), dpi=300)
# #exp1_MDS_5D_85.plot_experiment(mode="total") #
# #plt.gcf().suptitle(f"MDS 5D - Wind speed: 8.5 m/s ",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_5D_85.png"))
exp1_MDS_5D_10.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MDS 5D - Wind speed: 10 m/s ",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_5D_10_tidserie" + ".png"), dpi=300)
# plt.show()
# plt.close()

exp0_MDS_5D.filt_forces(6, 2)
exp1_MDS_5D_55.filt_forces(6, 2)
#exp1_MDS_5D_85.filt_forces(6, 2)
exp1_MDS_5D_10.filt_forces(6, 2)

# exp0_MDS_5D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 5D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_5D_0_filter.png"))
exp1_MDS_5D_55.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 5D - Wind speed: 5.5 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_5D_55_filter_tidserie" + ".png"), dpi=300)
# #exp1_MDS_5D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
# #plt.gcf().suptitle(f"MDS 5D - Wind speed: 8.5 m/s - With Butterworth low-pass filter",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MDS_5D_85_filter.png"))
exp1_MDS_5D_10.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MDS 5D - Wind speed: 10 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MDS_5D_10_filter_tidserie" + ".png"), dpi=300)
# plt.show()
# plt.close()

static_coeff_MDS_5D_55 =w3t.StaticCoeff.fromWTT(exp0_MDS_5D, exp1_MDS_5D_55, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

#static_coeff_MDS_5D_85 = w3t.StaticCoeff.fromWTT(exp0_MDS_5D, exp1_MDS_5D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

static_coeff_MDS_5D_10 = w3t.StaticCoeff.fromWTT(exp0_MDS_5D, exp1_MDS_5D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=False)

plot_static_coeff_summary(static_coeff_MDS_5D_55, section_name, 5.5, mode="decks", upwind_in_rig=False)
#plot_static_coeff_summary(static_coeff_MDS_5D_85, section_name, 8.5, mode="decks", upwind_in_rig=False)
plot_static_coeff_summary(static_coeff_MDS_5D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_5D_55, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_5D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MDS_5D_55, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_5D_low_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MDS_5D_55, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_5D_low_cm" + ".png"), dpi=300)

fig, ax =w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_5D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_5D_high_cd" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_5D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_5D_high_cl" + ".png"), dpi=300)
fig, ax =w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_5D_10, upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMDS_5D_high_cm" + ".png"), dpi=300)

#%% !!!!!
#fjerne hakk
alpha = np.round(static_coeff_MDS_5D_10.pitch_motion * 360 / (2 * np.pi), 1)
coeff_raw = static_coeff_MDS_5D_10.drag_coeff[:, 2] + static_coeff_MDS_5D_10.drag_coeff[:, 3]  # downwind

# 1. Fit en glatt kurve (uten NaN eller store avvik)
mask_fit = (alpha < 7) & (alpha > -7) & ~np.isnan(coeff_raw)
alpha_fit = alpha[mask_fit]
coeff_fit = coeff_raw[mask_fit]
coeffs = np.polyfit(alpha_fit, coeff_fit, deg=2)
curve = np.polyval(coeffs, alpha)

# 2. Finn hvilke punkter som avviker "for mye"
spread = np.abs(coeff_raw - curve)
threshold = 0.02  # juster etter hvor streng du vil være
mask_good = spread < threshold

# 3. Lag filtrert array
coeff_filtered = coeff_raw.copy()
coeff_filtered[~mask_good] = np.nan

# 4. Del ut tilbake til hver lastcelle
static_coeff_MDS_5D_10.drag_coeff[:, 2] = coeff_filtered / 2
static_coeff_MDS_5D_10.drag_coeff[:, 3] = coeff_filtered / 2
plot_static_coeff_summary(static_coeff_MDS_5D_10, section_name, 10, mode="decks", upwind_in_rig=False)


#%% !!!!!
#fjerne hakk
alpha = np.round(static_coeff_MDS_3D_10.pitch_motion * 360 / (2 * np.pi), 1)
coeff_raw = static_coeff_MDS_3D_10.pitch_coeff[:, 2] + static_coeff_MDS_3D_10.pitch_coeff[:, 2]  # downwind

# 1. Fit en glatt kurve (uten NaN eller store avvik)
mask_fit = (alpha < 7) & (alpha > -7) & ~np.isnan(coeff_raw)
alpha_fit = alpha[mask_fit]
coeff_fit = coeff_raw[mask_fit]
coeffs = np.polyfit(alpha_fit, coeff_fit, deg=1)
curve = np.polyval(coeffs, alpha)

# 2. Finn hvilke punkter som avviker "for mye"
spread = np.abs(coeff_raw - curve)
threshold = 0.008  # juster etter hvor streng du vil være
mask_good = spread < threshold

# 3. Lag filtrert array
coeff_filtered = coeff_raw.copy()
coeff_filtered[~mask_good] = np.nan

# 4. Del ut tilbake til hver lastcelle
static_coeff_MDS_3D_10.pitch_coeff[:, 2] = coeff_filtered / 2
static_coeff_MDS_3D_10.pitch_coeff[:, 3] = coeff_filtered / 2
plot_static_coeff_summary(static_coeff_MDS_3D_10, section_name, 10, mode="decks", upwind_in_rig=False)

#%% 
# Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_5D_55, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_5D_Static, 5.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_5D_55_drag_clean.png"))

#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_5D_85, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="drag")
#plt.suptitle(f"MDS_5D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_5D_85_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_5D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="drag")
plt.suptitle(f"MDS_5D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MDS_5D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_5D_55, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_5D_Static, 5.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_5D_55_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_5D_85, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="lift")
#plt.suptitle(f"MDS_5D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_5D_85_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_5D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="lift")
plt.suptitle(f"MDS_5D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MDS_5D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MDS_5D_55, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_5D_Static, 5.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_5D_55_pitch_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MDS_5D_85, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=False, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MDS_5D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_5D_85_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MDS_5D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=False, threshold=0.05, scoff="pitch")
plt.suptitle(f"MDS_5D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MDS_5D_10_pitch_clean.png"))


#%% !!!
# Summary
static_coeff_MDS_5D_55_updated = copy.deepcopy(static_coeff_MDS_5D_55)
static_coeff_MDS_5D_55_updated.drag_coeff[:, 2] = static_coeff_MDS_5D_55.drag_coeff[:, 2]  
static_coeff_MDS_5D_55_updated.drag_coeff[:, 3] = static_coeff_MDS_5D_55.drag_coeff[:, 3]
static_coeff_MDS_5D_55_updated.lift_coeff[:, 2] = static_coeff_MDS_5D_55.lift_coeff[:, 2]
static_coeff_MDS_5D_55_updated.lift_coeff[:, 3] = static_coeff_MDS_5D_55.lift_coeff[:, 3]
static_coeff_MDS_5D_55_updated.pitch_coeff[:, 2] = static_coeff_MDS_5D_55.pitch_coeff[:, 2]
static_coeff_MDS_5D_55_updated.pitch_coeff[:, 3] = static_coeff_MDS_5D_55.pitch_coeff[:, 3]

static_coeff_MDS_5D_10_updated = copy.deepcopy(static_coeff_MDS_5D_10)
 
static_coeff_MDS_5D_10_updated.drag_coeff[:, 2] = static_coeff_MDS_5D_10.drag_coeff[:, 2] 
static_coeff_MDS_5D_10_updated.drag_coeff[:, 3] = static_coeff_MDS_5D_10.drag_coeff[:, 3] 
static_coeff_MDS_5D_10_updated.lift_coeff[:, 2] = static_coeff_MDS_5D_10.lift_coeff[:, 2]
static_coeff_MDS_5D_10_updated.lift_coeff[:, 3] = static_coeff_MDS_5D_10.lift_coeff[:, 3]
static_coeff_MDS_5D_10_updated.pitch_coeff[:, 2] = static_coeff_MDS_5D_10.pitch_coeff[:,2]
static_coeff_MDS_5D_10_updated.pitch_coeff[:, 3] = static_coeff_MDS_5D_10.pitch_coeff[:, 3]


#%% !?!!!
def plot_compare_wind_speeds_mean_seperate5dmDs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,2}(\alpha_2)$"
        coeff = "drag_coeff"
        min = 0.42#0.4
        max = 0.5#0.58
    elif scoff == "lift":
        axis = r"$C_{L,2}(\alpha_2)$"
        coeff = "lift_coeff"
        min = -0.28#-0.35
        max = 0.33#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,2}(\alpha_2)$"
        coeff = "pitch_coeff"
        min = -0.04#-0.05
        max = 0.1#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    # ax.plot(unique_alphas_low, upwind_mean_low,
    #          label=f"5 m/s", color = "#2ca02c", alpha = 0.8)
    ax.plot(unique_alphas_low, downwind_mean_low,
             label=f"5,5 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        # ax.plot(unique_alphas_med, upwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        ax.plot(unique_alphas_med, downwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    # ax.plot(unique_alphas_high, upwind_mean_high,
    #             label=f"10 m/s", color ="#d62728", alpha = 0.8)
    ax.plot(unique_alphas_high, downwind_mean_high,
                label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MDS5D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate5dmDs(static_coeff_MDS_5D_55_updated, static_coeff_MDS_5D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mds_5D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate5dmDs(static_coeff_MDS_5D_55_updated, static_coeff_MDS_5D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mds_5D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate5dmDs(static_coeff_MDS_5D_55_updated, static_coeff_MDS_5D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mds_5D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)



#%% !!!
# Load all upwind experiments (upwind in rig)

section_name = "MDS_5D_Static"
file_names_MUS_5D_45 = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_002"] # 4.5 m/s, vibrations 
#file_names_MUS_5D_85 = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_003"] # 8.5 m/s, vibrations
file_names_MUS_5D_10 = ["HAR_INT_MDS_GAP_45D_02_01_000","HAR_INT_MDS_GAP_45D_02_01_004"] # 10 m/s, vibrations

import os

for f in file_names_MUS_5D_45 + file_names_MUS_5D_10:
    full_path = os.path.join(h5_input_path, section_name, f + ".h5")
    print(f, "->", os.path.exists(full_path))

exp0_MUS_5D, exp1_MUS_5D_45= load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_5D_45,  upwind_in_rig=True)
#exp0_MUS_5D, exp1_MUS_5D_85 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_5D_85,  upwind_in_rig=True)
exp0_MUS_5D, exp1_MUS_5D_10 = load_experiments_from_hdf5(h5_input_path, section_name, file_names_MUS_5D_10,  upwind_in_rig=True)


# exp0_MUS_5D.plot_experiment(mode="total") #
# plt.gcf().suptitle(f"MUS 5D - Wind speed: 0 m/s ",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_5D_0.png"))
exp1_MUS_5D_45.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MUS 5D - Wind speed: 4.5 m/s ",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_5D_45_tidserie" + ".png"), dpi=300)
# #exp1_MUS_5D_85.plot_experiment(mode="total") #
# #plt.gcf().suptitle(f"MUS 5D - Wind speed: 8.5 m/s ",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_5D_85.png"))
exp1_MUS_5D_10.plot_experiment(mode="decks") #
# plt.gcf().suptitle(f"MUS 5D - Wind speed: 10 m/s ",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_5D_10_tidserie" + ".png"), dpi=300)

exp0_MUS_5D.filt_forces(6, 2)
exp1_MUS_5D_45.filt_forces(6, 2)
#exp1_MUS_5D_85.filt_forces(6, 2)
exp1_MUS_5D_10.filt_forces(6, 2)

# exp0_MUS_5D.plot_experiment(mode="total") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 5D - Wind speed: 0 m/s - With Butterworth low-pass filter",  y=0.95)
# plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_5D_0_filter.png"))
exp1_MUS_5D_45.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 5D - Wind speed: 4.5 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_5D_45_filter_tidserie" + ".png"), dpi=300)
# #exp1_MUS_5D_85.plot_experiment(mode="total") #With Butterworth low-pass filter
# #plt.gcf().suptitle(f"MUS 5D - Wind speed: 8.5 m/s - With Butterworth low-pass filter",  y=0.95)
# #plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\tidsserier", "MUS_5D_85_filter.png"))
exp1_MUS_5D_10.plot_experiment(mode="decks") #With Butterworth low-pass filter
# plt.gcf().suptitle(f"MUS 5D - Wind speed:) 10 m/s - With Butterworth low-pass filter",  y=0.95)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "MUS_5D_10_filter_tidserie" + ".png"), dpi=300)
# plt.show()
# plt.close()

static_coeff_MUS_5D_45 =w3t.StaticCoeff.fromWTT(exp0_MUS_5D, exp1_MUS_5D_45, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

#static_coeff_MUS_5D_85 = w3t.StaticCoeff.fromWTT(exp0_MUS_5D, exp1_MUS_5D_85, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

static_coeff_MUS_5D_10 = w3t.StaticCoeff.fromWTT(exp0_MUS_5D, exp1_MUS_5D_10, section_width, section_height, section_length_in_rig, section_length_on_wall,  upwind_in_rig=True)

plot_static_coeff_summary(static_coeff_MUS_5D_45, section_name, 4.5, mode="decks", upwind_in_rig=True)
#plot_static_coeff_summary(static_coeff_MUS_5D_85, section_name, 8.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_5D_10, section_name, 10, mode="decks", upwind_in_rig=True)


#%%NILS


fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_5D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_5D_low_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_5D_low_cm" + ".png"), dpi=300)

fig, ax = w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_5D_high_cd" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_5D_high_cl" + ".png"), dpi=300)
fig, ax = w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_5D_10, upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "_nilsMUS_5D_high_cm" + ".png"), dpi=300)


#%% !!!
#fjerne hakk
alpha = np.round(static_coeff_MUS_5D_45.pitch_motion * 360 / (2 * np.pi), 1)
coeff_raw = static_coeff_MUS_5D_45.lift_coeff[:, 0] + static_coeff_MUS_5D_45.lift_coeff[:, 1]  # upwind

# 1. Fit en glatt kurve (uten NaN eller store avvik)
mask_fit = (alpha < 4.5) & (alpha > -7) & ~np.isnan(coeff_raw)
alpha_fit = alpha[mask_fit]
coeff_fit = coeff_raw[mask_fit]
coeffs = np.polyfit(alpha_fit, coeff_fit, deg=1)
curve = np.polyval(coeffs, alpha)

# 2. Finn hvilke punkter som avviker "for mye"
spread = np.abs(coeff_raw - curve)
threshold = 0.05  # juster etter hvor streng du vil være
mask_good = spread < threshold

# 3. Lag filtrert array
coeff_filtered = coeff_raw.copy()
coeff_filtered[~mask_good] = np.nan

# 4. Del ut tilbake til hver lastcelle
static_coeff_MUS_5D_45.lift_coeff[:, 0] = coeff_filtered / 2
static_coeff_MUS_5D_45.lift_coeff[:, 1] = coeff_filtered / 2
plot_static_coeff_summary(static_coeff_MUS_5D_45, section_name, 10, mode="decks", upwind_in_rig=True)

#%% !!!
#fjerne hakk
alpha = np.round(static_coeff_MUS_5D_10.pitch_motion * 360 / (2 * np.pi), 1)
coeff_raw = static_coeff_MUS_5D_10.drag_coeff[:, 0] + static_coeff_MUS_5D_10.drag_coeff[:, 1]  # upwind

# 1. Fit en glatt kurve (uten NaN eller store avvik)
mask_fit = (alpha < 7) & (alpha > -7) & ~np.isnan(coeff_raw)
alpha_fit = alpha[mask_fit]
coeff_fit = coeff_raw[mask_fit]
coeffs = np.polyfit(alpha_fit, coeff_fit, deg=2)
curve = np.polyval(coeffs, alpha)

# 2. Finn hvilke punkter som avviker "for mye"
spread = np.abs(coeff_raw - curve)
threshold = 0.04  # juster etter hvor streng du vil være
mask_good = spread < threshold

# 3. Lag filtrert array
coeff_filtered = coeff_raw.copy()
coeff_filtered[~mask_good] = np.nan

# 4. Del ut tilbake til hver lastcelle
static_coeff_MUS_5D_10.drag_coeff[:, 0] = coeff_filtered / 2
static_coeff_MUS_5D_10.drag_coeff[:, 1] = coeff_filtered / 2
plot_static_coeff_summary(static_coeff_MUS_5D_10, section_name, 10, mode="decks", upwind_in_rig=True)


#%% 
# Filter and plot ALT 1
#drag
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_5D_45, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_5D_Static, 4.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_5D_45_drag_clean.png"))

#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_5D_85, threshold=0.05, scoff="drag", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="drag")
#plt.suptitle(f"MUS_5D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_5D_85_drag_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_5D_10, threshold=0.05, scoff="drag", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="drag")
plt.suptitle(f"MUS_5D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\drag", "MUS_5D_10_drag_clean.png"))

#lift
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_5D_45, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_5D_Static, 4.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_5D_45_lift_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_5D_85, threshold=0.05, scoff="lift", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med,upwind_in_rig=True, threshold=0.05, scoff="lift")
#plt.suptitle(f"MUS_5D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_5D_85_lift_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_5D_10, threshold=0.05, scoff="lift", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="lift")
plt.suptitle(f"MUS_5D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\lift", "MUS_5D_10_lift_clean.png"))

#pitch
alpha_low, coeff_plot_up_low, coeff_plot_down_low=w3t._scoff.filter(static_coeff_MUS_5D_45, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_low,coeff_plot_up_low,coeff_plot_down_low, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_5D_Static, 4.5 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_5D_45_pitch_clean.png"))
#alpha_med, coeff_plot_up_med, coeff_plot_down_med=w3t._scoff.filter(static_coeff_MUS_5D_85, threshold=0.05, scoff="pitch", single = False)
#w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_med,coeff_plot_up_med,coeff_plot_down_med, upwind_in_rig=True, threshold=0.05, scoff="pitch")
#plt.suptitle(f"MUS_5D_Static, 8.5 m/s",  y=0.95)
#plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_5D_85_pitch_clean.png"))
alpha_high, coeff_plot_up_high, coeff_plot_down_high=w3t._scoff.filter(static_coeff_MUS_5D_10, threshold=0.05, scoff="pitch", single = False)
w3t._scoff.plot_static_coeff_filtered_out_above_threshold(alpha_high,coeff_plot_up_high,coeff_plot_down_high, upwind_in_rig=True, threshold=0.05, scoff="pitch")
plt.suptitle(f"MUS_5D_Static, 10 m/s",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\without_vibration\pitch", "MUS_5D_10_pitch_clean.png"))


#%%  
# Filter and plot ALT 2
section_name = "MDS_5D_Static_filtered"

static_coeff_MUS_5D_45_filtered, static_coeff_MUS_5D_10_filtered = w3t._scoff.filter_by_reference(static_coeff_1=static_coeff_MUS_5D_45, static_coeff_2=static_coeff_MUS_5D_10, threshold=0.05, threshold_low=[0.1,0.03,0.008],threshold_high=[0.1,0.03,0.008],single=False)
#DRAG DÅRLIG HER!

plot_static_coeff_summary(static_coeff_MUS_5D_45_filtered, section_name, 4.5, mode="decks", upwind_in_rig=True)
plot_static_coeff_summary(static_coeff_MUS_5D_10_filtered, section_name, 10, mode="decks", upwind_in_rig=True)
#%% !!!
#Summary

static_coeff_MUS_5D_45_updated = copy.deepcopy(static_coeff_MUS_5D_45)
static_coeff_MUS_5D_45_updated.drag_coeff[:, 0] = static_coeff_MUS_5D_45.drag_coeff[:, 0]
static_coeff_MUS_5D_45_updated.drag_coeff[:, 1] = static_coeff_MUS_5D_45.drag_coeff[:, 1]
static_coeff_MUS_5D_45_updated.lift_coeff[:, 0] = static_coeff_MUS_5D_45.lift_coeff[:, 0]
static_coeff_MUS_5D_45_updated.lift_coeff[:, 1] = static_coeff_MUS_5D_45.lift_coeff[:, 1]
static_coeff_MUS_5D_45_updated.pitch_coeff[:, 0] = static_coeff_MUS_5D_45.pitch_coeff[:, 0]
static_coeff_MUS_5D_45_updated.pitch_coeff[:, 1] = static_coeff_MUS_5D_45.pitch_coeff[:, 1]

static_coeff_MUS_5D_10_updated = copy.deepcopy(static_coeff_MUS_5D_10)
 
static_coeff_MUS_5D_10_updated.drag_coeff[:, 0] = static_coeff_MUS_5D_10.drag_coeff[:, 0] 
static_coeff_MUS_5D_10_updated.drag_coeff[:, 1] = static_coeff_MUS_5D_10.drag_coeff[:, 1] 
static_coeff_MUS_5D_10_updated.lift_coeff[:, 0] = static_coeff_MUS_5D_10.lift_coeff[:, 0]
static_coeff_MUS_5D_10_updated.lift_coeff[:, 1] = static_coeff_MUS_5D_10.lift_coeff[:, 1]
static_coeff_MUS_5D_10_updated.pitch_coeff[:, 0] = static_coeff_MUS_5D_10.pitch_coeff[:,0]
static_coeff_MUS_5D_10_updated.pitch_coeff[:, 1] = static_coeff_MUS_5D_10.pitch_coeff[:, 1]


#%% !?!!!
def plot_compare_wind_speeds_mean_seperate5dmUs(static_coeff_low, 
                                   static_coeff_high, static_coeff_med = None,
                                    scoff = "", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.63))
    if scoff == "drag":
        axis = r"$C_{D,1}(\alpha_1)$"
        coeff = "drag_coeff"
        min = 0.45#0.4
        max = 0.72#0.58
    elif scoff == "lift":
        axis = r"$C_{L,1}(\alpha_1)$"
        coeff = "lift_coeff"
        min = -0.35#-0.35
        max = 0.53#0.5
    elif scoff == "pitch":
        axis = r"$C_{M,1}(\alpha_1)$"
        coeff = "pitch_coeff"
        min = -0.04#-0.05
        max = 0.12#0.15

    # Calculate unique alpha values (pitch motion in degrees)
    alpha_low = np.round(static_coeff_low.pitch_motion*360/2/np.pi,1)
    unique_alphas_low = np.unique(alpha_low)
    alpha_high = np.round(static_coeff_high.pitch_motion*360/2/np.pi,1)
    unique_alphas_high = np.unique(alpha_high)

    print(np.where(alpha_low == 2.0))

    upwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,0][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,1][alpha_low == val]) for val in unique_alphas_low])
    downwind_mean_low = np.array([np.nanmean(getattr(static_coeff_low, coeff)[:,2][alpha_low == val]) + np.nanmean(getattr(static_coeff_low, coeff)[:,3][alpha_low == val]) for val in unique_alphas_low])
    
    upwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,0][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,1][alpha_high == val]) for val in unique_alphas_high])
    downwind_mean_high = np.array([np.nanmean(getattr(static_coeff_high, coeff)[:,2][alpha_high == val]) + np.nanmean(getattr(static_coeff_high, coeff)[:,3][alpha_high == val]) for val in unique_alphas_high])


    # Plot low wind speed
    ax.plot(unique_alphas_low, upwind_mean_low,
             label=f"4,5 m/s", color = "#2ca02c", alpha = 0.8)
    # ax.plot(unique_alphas_low, downwind_mean_low,
    #          label=f"6 m/s", color = "#2ca02c", alpha = 0.8)


    if static_coeff_med is not None:
        alpha_med = np.round(static_coeff_med.pitch_motion*360/2/np.pi,1)
        unique_alphas_med = np.unique(alpha_med)
        upwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,0][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,1][alpha_med == val]) for val in unique_alphas_med])
        downwind_mean_med = np.array([np.nanmean(getattr(static_coeff_med, coeff)[:,2][alpha_med == val]) + np.nanmean(getattr(static_coeff_med, coeff)[:,3][alpha_med == val]) for val in unique_alphas_med])
        ax.plot(unique_alphas_med, upwind_mean_med,
                    label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)
        # ax.plot(unique_alphas_med, downwind_mean_med,
        #             label=f"8 m/s", color = "#ff7f0e", alpha = 0.8)

    # Plot high wind speed
    ax.plot(unique_alphas_high, upwind_mean_high,
                label=f"10 m/s", color ="#d62728", alpha = 0.8)
    # ax.plot(unique_alphas_high, downwind_mean_high,
    #             label=f"10 m/s", color = "#d62728", alpha = 0.8)

    #ax.grid()
    ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
    ax.set_ylabel(axis, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11,labelspacing=0.3) #loc='upper left',
    ax.grid(True)
    ax.set_xticks([-4,-2, 0,2,  4])
    ax.set_ylim(min,max)
    ax.set_xlim(-4,4)
    
    #ax.set_title(f"Comparison of {scoff} coefficients at different wind speeds")

# Compare wind speed
section_name = "MUS5D_Static_updated_windspeeds"
plot_compare_wind_speeds_mean_seperate5dmUs(static_coeff_MUS_5D_45_updated, static_coeff_MUS_5D_10_updated,static_coeff_med = None,scoff = "drag", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cd_mus_5D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate5dmUs(static_coeff_MUS_5D_45_updated, static_coeff_MUS_5D_10_updated,static_coeff_med = None,scoff = "lift", ax=None)
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , "cl_mus_5D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plot_compare_wind_speeds_mean_seperate5dmUs(static_coeff_MUS_5D_45_updated, static_coeff_MUS_5D_10_updated,static_coeff_med = None,scoff = "pitch", ax=None)  
plt.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" ,  "cm_mus_5D_speed.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)


#%% 
# Save all experiments to excel
section_name = "5D"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MDS_5D_55.to_excel(section_name, sheet_name="MDS - 5.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_45.to_excel(section_name, sheet_name='MUS - 4.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_5D_85.to_excel(section_name, sheet_name="MDS - 8.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_5D_85.to_excel(section_name, sheet_name='MUS - 8.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_5D_10.to_excel(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_10.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "5D_mean"

# Low wind speed
static_coeff_MDS_5D_55.to_excel_mean(section_name, sheet_name="MDS - 5.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_45.to_excel_mean(section_name, sheet_name='MUS - 4.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
#static_coeff_MDS_5D_85.to_excel_mean(section_name, sheet_name="MDS - 8.5" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
#static_coeff_MUS_5D_85.to_excel_mean(section_name, sheet_name='MUS - 8.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MDS_5D_10.to_excel_mean(section_name, sheet_name="MDS - 10" ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=False)
static_coeff_MUS_5D_10.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)


#%%
#  Save all experiments to excel filtered
section_name = "5D_filtered"
#Her er MDS og MUS riktig, så motsatt av våre eksperimenter i excel arket

# Low wind speed
static_coeff_MUS_5D_45_filtered.to_excel(section_name, sheet_name='MUS - 4.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_5D_10_filtered.to_excel(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

section_name = "5D_mean_filtered"

# Low wind speed
static_coeff_MUS_5D_45_filtered.to_excel_mean(section_name, sheet_name='MUS - 4.5' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_6_filtered.to_excel_mean(section_name, sheet_name='Single - 6' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# Medium wind speed
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

# High wind speed
static_coeff_MUS_5D_10_filtered.to_excel_mean(section_name, sheet_name='MUS - 10' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)
static_coeff_single_9_filtered.to_excel_mean(section_name, sheet_name='Single - 9' ,section_width =  section_width,section_height=section_height,section_length_in_rig=2.68, section_length_on_wall=2.66, upwind_in_rig=True)

#%%
#  Compare all experiments (MUS vs MDS vs Single)
section_name = "5D"


#Low wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_6, static_coeff_MUS_5D_45, static_coeff_MDS_5D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_low_drag" + ".png"))

w3t._scoff.plot_compare_lift(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_low_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_low_pitch" + ".png"))
# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_6, static_coeff_MUS_5D_45, static_coeff_MDS_5D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_low_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_low_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, static_coeff_MDS_5D_55)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_low_pitch_mean" + ".png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_5D_85_filtered, static_coeff_MDS_5D_85_filtered)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s, MDS: 8.5 m/s", fontsize=16)

#%%
#High wind speed
w3t._scoff.plot_compare_drag(static_coeff_single_9_filtered, static_coeff_MUS_5D_10, static_coeff_MDS_5D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_high_drag" + ".png"))

w3t._scoff.plot_compare_lift(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_high_lift" + ".png"))
w3t._scoff.plot_compare_pitch(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_high_pitch" + ".png"))

# Mean
w3t._scoff.plot_compare_drag_mean(static_coeff_single_9_filtered, static_coeff_MUS_5D_10, static_coeff_MDS_5D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_high_drag_mean" + ".png"))
w3t._scoff.plot_compare_lift_mean(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_high_lift_mean" + ".png"))
w3t._scoff.plot_compare_pitch_mean(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, static_coeff_MDS_5D_10)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig(os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\ALT", "5D_high_pitch_mean" + ".png"))

plt.show()
plt.close()

#%% 
# Compare all experiments - only with single deck

#Low wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS: 4.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_low_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_6, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_low_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_low_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_low_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_low_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_low_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MUS_5D_45, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_low_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_6, static_coeff_MDS_5D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_low_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_low_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_low_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MUS_5D_45_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MUS:  4.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_low_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_6_filtered, static_coeff_MDS_5D_55, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 6 m/s, MDS: 5.5 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_low_pitch_mean.png"))

# #Medium wind speed
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_5D_85,  upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)

# # Mean
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_85,upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_85,upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_85, upwind_in_rig=True)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 8.5 m/s", fontsize=16)
# w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_85, upwind_in_rig=False)
# plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 8.5 m/s", fontsize=16)
#%%                                               
#High wind speed
w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_high_drag.png"))

w3t._scoff.plot_compare_drag_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_high_drag.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_high_lift.png"))
w3t._scoff.plot_compare_lift_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_high_lift.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MUS_5D_10_filtered, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_high_pitch.png"))
w3t._scoff.plot_compare_pitch_only_single(static_coeff_single_9, static_coeff_MDS_5D_10, upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_high_pitch.png"))

# Mean
w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MUS_5D_10, upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_high_drag_mean.png"))

w3t._scoff.plot_compare_drag_mean_only_single(static_coeff_single_9_filtered, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_high_drag_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_high_lift_mean.png"))
w3t._scoff.plot_compare_lift_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s,  MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_high_lift_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MUS_5D_10_filtered,upwind_in_rig=True)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MUS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMUS", "Single_MUS_5D_high_pitch_mean.png"))
w3t._scoff.plot_compare_pitch_mean_only_single(static_coeff_single_9, static_coeff_MDS_5D_10,upwind_in_rig=False)
plt.gcf().suptitle(f"{section_name}: Single: 9 m/s, MDS: 10 m/s", fontsize=16)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\SingleMDS", "Single_MDS_5D_high_pitch_mean.png"))
plt.show()
plt.close()

# %% Compare all experiments (Wind speed)
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45,
                                static_coeff_MUS_5D_10,
                             scoff = "drag")                        
plt.gcf().suptitle(f"5D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_5D_drag.png"))


# MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55,
                                static_coeff_MDS_5D_10,
                                scoff = "drag")                        
plt.gcf().suptitle(f"5D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_5D_drag.png"))

#lift
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_5D_45_filtered,
                                static_coeff_MUS_5D_10_filtered,
                            scoff = "lift")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_5D_lift.png"))

#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9,static_coeff_MDS_5D_55,
                                static_coeff_MDS_5D_10,
                               scoff = "lift")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_5D_lift.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_5D_45_filtered,
                                static_coeff_MUS_5D_10_filtered,
                              scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_5D_pitch.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                                static_coeff_MDS_5D_10,
                          scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_5D_pitch.png"))

#MEAN
#drag
# MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MUS_5D_45,
                                static_coeff_MUS_5D_10,
                           scoff = "drag")                        
plt.gcf().suptitle(f"5D: MUS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_5D_drag_mean.png"))

# MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6, 
                               static_coeff_single_9_filtered, static_coeff_MDS_5D_55,
                                static_coeff_MDS_5D_10,
                              scoff = "drag")                        
plt.gcf().suptitle(f"5D: MDS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_5D_drag_mean.png"))
#lift
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_5D_45_filtered,
                                static_coeff_MUS_5D_10_filtered,
                                scoff = "lift")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_5D_lift_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                                static_coeff_MDS_5D_10,
                                scoff = "lift")                        
plt.gcf().suptitle(f"5D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_5D_lift_mean.png"))
#pitch
#MUS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MUS_5D_45_filtered,
                                static_coeff_MUS_5D_10_filtered,
                                scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MUS  ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MUS_5D_pitch_mean.png"))
#MDS
w3t._scoff.plot_compare_wind_speeds_mean(static_coeff_single_6_filtered, 
                               static_coeff_single_9, static_coeff_MDS_5D_55,
                                static_coeff_MDS_5D_10,
                               scoff = "pitch")                        
plt.gcf().suptitle(f"5D: MDS ",  y=0.95)
plt.savefig( os.path.join(r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Plots\static\comparison\windspeed", "Single_MDS_5D_pitch_mean.png"))


##########################################################################333


#%% !!!

def plot_compare_distance_mean1(static_coeff_single, static_coeff_1D, static_coeff_2D, static_coeff_3D, static_coeff_4D, static_coeff_5D, scoff="", upwind_in_rig=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.05, 2.63))

    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.sort(np.unique(alpha_single)) 
    alpha_1D = np.round(static_coeff_1D.pitch_motion*360/2/np.pi,1)
    unique_alphas_1D = np.sort(np.unique(alpha_1D))
    alpha_2D = np.round(static_coeff_2D.pitch_motion*360/2/np.pi,1)
    unique_alphas_2D = np.sort(np.unique(alpha_2D))
    alpha_3D = np.round(static_coeff_3D.pitch_motion*360/2/np.pi,1)
    unique_alphas_3D = np.sort(np.unique(alpha_3D))
    alpha_4D = np.round(static_coeff_4D.pitch_motion*360/2/np.pi,1)
    unique_alphas_4D = np.sort(np.unique(alpha_4D))
    alpha_5D = np.round(static_coeff_5D.pitch_motion*360/2/np.pi,1)
    unique_alphas_5D = np.sort(np.unique(alpha_5D))

    if upwind_in_rig: #MUS
        if scoff == "drag":
            ylabel = r"$C_{D, 1}(\alpha_1)$"
            ymin = 0.35
            ymax = 0.605
            cd_upwind_mean_single = np.array([
                np.nanmean(static_coeff_single.drag_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.drag_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single])
            cd_upwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.drag_coeff[:,0][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.drag_coeff[:,1][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D])
            cd_upwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.drag_coeff[:,0][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.drag_coeff[:,1][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D])
            
            cd_upwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.drag_coeff[:,0][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.drag_coeff[:,1][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D])
            cd_upwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.drag_coeff[:,0][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.drag_coeff[:,1][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D])
            cd_upwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.drag_coeff[:,0][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.drag_coeff[:,1][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D])
        elif scoff == "lift":
            ylabel = r"$C_{L, 1}(\alpha_1)$"
            ymin = -0.375
            ymax = 0.43
            cd_upwind_mean_single = np.array([
                np.nanmean(static_coeff_single.lift_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.lift_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single])
            cd_upwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.lift_coeff[:,0][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.lift_coeff[:,1][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D])
            cd_upwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.lift_coeff[:,0][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.lift_coeff[:,1][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D])
            cd_upwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.lift_coeff[:,0][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.lift_coeff[:,1][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D])
            cd_upwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.lift_coeff[:,0][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.lift_coeff[:,1][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D])
            cd_upwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.lift_coeff[:,0][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.lift_coeff[:,1][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D])
        elif scoff == "pitch":
            ylabel = r"$C_{M,1}(\alpha_1)$"
            ymin=-0.05
            ymax=0.105
            cd_upwind_mean_single = np.array([
                np.nanmean(static_coeff_single.pitch_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.pitch_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single])
            cd_upwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.pitch_coeff[:,0][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.pitch_coeff[:,1][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D])
            cd_upwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.pitch_coeff[:,0][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.pitch_coeff[:,1][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D])
            cd_upwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.pitch_coeff[:,0][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.pitch_coeff[:,1][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D])
            cd_upwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.pitch_coeff[:,0][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.pitch_coeff[:,1][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D])

            cd_upwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.pitch_coeff[:,0][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.pitch_coeff[:,1][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D])
        else:
            print(scoff + " Error: Unknown argument: scoff=" + scoff + " Use scoff=drag, lift or pitch" )
            return None

        ax.plot(unique_alphas_1D,cd_upwind_mean_1D,label=(" 1D "), alpha = 0.8)
        ax.plot(unique_alphas_2D,cd_upwind_mean_2D,label=(" 2D "), alpha = 0.8)
        ax.plot(unique_alphas_3D,cd_upwind_mean_3D,label=(" 3D "), alpha = 0.8)
        ax.plot(unique_alphas_4D,cd_upwind_mean_4D,label=(" 4D "), alpha = 0.8)
        ax.plot(unique_alphas_5D,cd_upwind_mean_5D,label=(" 5D "), alpha = 0.8)
        ax.plot(unique_alphas_single, cd_upwind_mean_single,label=("Single"),  alpha = 0.8)
        ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin=-4, xmax=4)
        ax.grid(True)


            
    else: #MDS
        if scoff == "drag":    
            ylabel = r"$C_{D,2}(\alpha_2)$"
            ymin = 0.34
            ymax = 0.62
            cd_downwind_mean_single = np.array([
                np.nanmean(static_coeff_single.drag_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.drag_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single
            ])
            cd_downwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.drag_coeff[:,2][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.drag_coeff[:,3][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D
            ])
            cd_downwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.drag_coeff[:,2][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.drag_coeff[:,3][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D
            ])
            cd_downwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.drag_coeff[:,2][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.drag_coeff[:,3][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D
            ])
            cd_downwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.drag_coeff[:,2][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.drag_coeff[:,3][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D
            ])
            cd_downwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.drag_coeff[:,2][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.drag_coeff[:,3][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D
            ])
        elif scoff == "lift":
            ylabel = r"$C_{L,2}(\alpha_2)$"
            ymin = -0.37
            ymax = 0.37
            cd_downwind_mean_single = np.array([
                np.nanmean(static_coeff_single.lift_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.lift_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single
            ])
            cd_downwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.lift_coeff[:,2][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.lift_coeff[:,3][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D
            ])
            cd_downwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.lift_coeff[:,2][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.lift_coeff[:,3][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D
            ])
            cd_downwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.lift_coeff[:,2][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.lift_coeff[:,3][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D
            ])
            cd_downwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.lift_coeff[:,2][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.lift_coeff[:,3][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D
            ])
            cd_downwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.lift_coeff[:,2][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.lift_coeff[:,3][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D
            ])
        elif scoff == "pitch":
            ylabel = r"$C_{M,2}(\alpha_2)$"
            ymin=-0.05
            ymax=0.105
            cd_downwind_mean_single = np.array([
                np.nanmean(static_coeff_single.pitch_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.pitch_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single
            ])
            cd_downwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.pitch_coeff[:,2][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.pitch_coeff[:,3][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D
            ])
            cd_downwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.pitch_coeff[:,2][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.pitch_coeff[:,3][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D
            ])
            cd_downwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.pitch_coeff[:,2][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.pitch_coeff[:,3][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D
            ])
            cd_downwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.pitch_coeff[:,2][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.pitch_coeff[:,3][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D
            ])

            cd_downwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.pitch_coeff[:,2][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.pitch_coeff[:,3][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D
            ])


        ax.plot(unique_alphas_1D,cd_downwind_mean_1D,label=(" 1D "), alpha = 0.8)
        ax.plot(unique_alphas_2D,cd_downwind_mean_2D,label=(" 2D "), alpha = 0.8)
        ax.plot(unique_alphas_3D,cd_downwind_mean_3D,label=(" 3D "), alpha = 0.8)
        ax.plot(unique_alphas_4D,cd_downwind_mean_4D,label=(" 4D "), alpha = 0.8)
        ax.plot(unique_alphas_5D,cd_downwind_mean_5D,label=(" 5D "), alpha = 0.8)
        ax.plot(unique_alphas_single, cd_downwind_mean_single,label=("Single"),  alpha = 0.8)
        ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin=-4, xmax=4)
        ax.grid(True)
  

    ax.set_xticks([-4,-2,0,2 ,4])
    plt.show()
    return fig, ax


#%% !!!
def plot_compare_distance_mean_oposite(static_coeff_single, static_coeff_1D, static_coeff_2D, static_coeff_3D, static_coeff_4D, static_coeff_5D, scoff="", upwind_in_rig=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.05, 2.63))


    alpha_single = np.round(static_coeff_single.pitch_motion*360/2/np.pi,1)
    unique_alphas_single = np.sort(np.unique(alpha_single)) 
    alpha_1D = np.round(static_coeff_1D.pitch_motion*360/2/np.pi,1)
    unique_alphas_1D = np.sort(np.unique(alpha_1D))
    alpha_2D = np.round(static_coeff_2D.pitch_motion*360/2/np.pi,1)
    unique_alphas_2D = np.sort(np.unique(alpha_2D))
    alpha_3D = np.round(static_coeff_3D.pitch_motion*360/2/np.pi,1)
    unique_alphas_3D = np.sort(np.unique(alpha_3D))
    alpha_4D = np.round(static_coeff_4D.pitch_motion*360/2/np.pi,1)
    unique_alphas_4D = np.sort(np.unique(alpha_4D))
    alpha_5D = np.round(static_coeff_5D.pitch_motion*360/2/np.pi,1)
    unique_alphas_5D = np.sort(np.unique(alpha_5D))

    if upwind_in_rig: #MUS
        if scoff == "drag":
            ylabel = r"$C_{D,2}(\alpha_1)$"
            ymin = 0.3
            ymax = 0.41
            cd_upwind_mean_single = np.array([
                np.nanmean(static_coeff_single.drag_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.drag_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single])
            cd_upwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.drag_coeff[:,2][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.drag_coeff[:,3][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D])
            cd_upwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.drag_coeff[:,2][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.drag_coeff[:,3][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D])
            
            cd_upwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.drag_coeff[:,2][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.drag_coeff[:,3][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D])
            cd_upwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.drag_coeff[:,2][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.drag_coeff[:,3][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D])
            cd_upwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.drag_coeff[:,2][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.drag_coeff[:,3][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D])
        elif scoff == "lift":
            ylabel = r"$C_{L,2}(\alpha_1)$"
            ymin = -0.2
            ymax = 0.11
            cd_upwind_mean_single = np.array([
                np.nanmean(static_coeff_single.lift_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.lift_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single])
            cd_upwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.lift_coeff[:,2][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.lift_coeff[:,3][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D])
            cd_upwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.lift_coeff[:,2][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.lift_coeff[:,3][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D])
            cd_upwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.lift_coeff[:,2][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.lift_coeff[:,3][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D])
            cd_upwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.lift_coeff[:,2][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.lift_coeff[:,3][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D])
            cd_upwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.lift_coeff[:,2][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.lift_coeff[:,3][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D])
        elif scoff == "pitch":
            ylabel = r"$C_{M,2}(\alpha_1)$"
            ymin=-0.03
            ymax=0.0501
            cd_upwind_mean_single = np.array([
                np.nanmean(static_coeff_single.pitch_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.pitch_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single])
            cd_upwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.pitch_coeff[:,2][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.pitch_coeff[:,3][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D])
            cd_upwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.pitch_coeff[:,2][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.pitch_coeff[:,3][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D])
            cd_upwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.pitch_coeff[:,2][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.pitch_coeff[:,3][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D])
            cd_upwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.pitch_coeff[:,2][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.pitch_coeff[:,3][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D])

            cd_upwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.pitch_coeff[:,2][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.pitch_coeff[:,3][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D])
        else:
            print(scoff + " Error: Unknown argument: scoff=" + scoff + " Use scoff=drag, lift or pitch" )
            return None

        ax.plot(unique_alphas_1D,cd_upwind_mean_1D,label=(" 1D "), alpha = 0.8)
        ax.plot(unique_alphas_2D,cd_upwind_mean_2D,label=(" 2D "), alpha = 0.8)
        ax.plot(unique_alphas_3D,cd_upwind_mean_3D,label=(" 3D "), alpha = 0.8)
        ax.plot(unique_alphas_4D,cd_upwind_mean_4D,label=(" 4D "), alpha = 0.8)
        ax.plot(unique_alphas_5D,cd_upwind_mean_5D,label=(" 5D "), alpha = 0.8)
        #ax.plot(unique_alphas_single, cd_upwind_mean_single,label=("Single"),  alpha = 0.8)
        ax.set_xlabel(r"$\alpha_1$ [deg]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin=-4, xmax=4)
        ax.grid(True)

            
    else: #MDS
        if scoff == "drag":    
            ylabel = r"$C_{D,1}(\alpha_2)$"
            ymin = 0.35
            ymax = 0.5
            cd_downwind_mean_single = np.array([
                np.nanmean(static_coeff_single.drag_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.drag_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single
            ])
            cd_downwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.drag_coeff[:,0][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.drag_coeff[:,1][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D
            ])
            cd_downwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.drag_coeff[:,0][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.drag_coeff[:,1][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D
            ])
            cd_downwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.drag_coeff[:,0][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.drag_coeff[:,1][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D
            ])
            cd_downwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.drag_coeff[:,0][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.drag_coeff[:,1][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D
            ])
            cd_downwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.drag_coeff[:,0][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.drag_coeff[:,1][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D
            ])
        elif scoff == "lift":
            ylabel = r"$C_{L,1}(\alpha_2)$"
            ymin = -0.02
            ymax = 0.18
            cd_downwind_mean_single = np.array([
                np.nanmean(static_coeff_single.lift_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.lift_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single
            ])
            cd_downwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.lift_coeff[:,0][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.lift_coeff[:,1][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D
            ])
            cd_downwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.lift_coeff[:,0][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.lift_coeff[:,1][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D
            ])
            cd_downwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.lift_coeff[:,0][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.lift_coeff[:, 1][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D
            ])
            cd_downwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.lift_coeff[:,0][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.lift_coeff[:,1][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D
            ])
            cd_downwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.lift_coeff[:,0][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.lift_coeff[:,1][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D
            ])
        elif scoff == "pitch":
            ylabel = r"$C_{M,1}(\alpha_2)$"
            ymin=0.02
            ymax=0.055
            cd_downwind_mean_single = np.array([
                np.nanmean(static_coeff_single.pitch_coeff[:,0][np.isclose(alpha_single, val, atol=1e-6)]) + np.nanmean(static_coeff_single.pitch_coeff[:,1][np.isclose(alpha_single, val, atol=1e-6)])
                for val in unique_alphas_single
            ])
            cd_downwind_mean_1D = np.array([
                np.nanmean(static_coeff_1D.pitch_coeff[:,0][np.isclose(alpha_1D, val, atol=1e-6)]) + np.nanmean(static_coeff_1D.pitch_coeff[:,1][np.isclose(alpha_1D, val, atol=1e-6)])
                for val in unique_alphas_1D
            ])
            cd_downwind_mean_2D = np.array([
                np.nanmean(static_coeff_2D.pitch_coeff[:,0][np.isclose(alpha_2D, val, atol=1e-6)]) + np.nanmean(static_coeff_2D.pitch_coeff[:,1][np.isclose(alpha_2D, val, atol=1e-6)])
                for val in unique_alphas_2D
            ])
            cd_downwind_mean_3D = np.array([
                np.nanmean(static_coeff_3D.pitch_coeff[:,0][np.isclose(alpha_3D, val, atol=1e-6)]) + np.nanmean(static_coeff_3D.pitch_coeff[:,1][np.isclose(alpha_3D, val, atol=1e-6)])
                for val in unique_alphas_3D
            ])
            cd_downwind_mean_4D = np.array([
                np.nanmean(static_coeff_4D.pitch_coeff[:,0][np.isclose(alpha_4D, val, atol=1e-6)]) + np.nanmean(static_coeff_4D.pitch_coeff[:,1][np.isclose(alpha_4D, val, atol=1e-6)])
                for val in unique_alphas_4D
            ])

            cd_downwind_mean_5D = np.array([
                np.nanmean(static_coeff_5D.pitch_coeff[:,0][np.isclose(alpha_5D, val, atol=1e-6)]) + np.nanmean(static_coeff_5D.pitch_coeff[:,1][np.isclose(alpha_5D, val, atol=1e-6)])
                for val in unique_alphas_5D
            ])


        ax.plot(unique_alphas_1D,cd_downwind_mean_1D,label=(" 1D "), alpha = 0.8)
        ax.plot(unique_alphas_2D,cd_downwind_mean_2D,label=(" 2D "), alpha = 0.8)
        ax.plot(unique_alphas_3D,cd_downwind_mean_3D,label=(" 3D "), alpha = 0.8)
        ax.plot(unique_alphas_4D,cd_downwind_mean_4D,label=(" 4D "), alpha = 0.8)
        ax.plot(unique_alphas_5D,cd_downwind_mean_5D,label=(" 5D "), alpha = 0.8)
        #ax.plot(unique_alphas_single, cd_downwind_mean_single,label=("Single"),  alpha = 0.8)
        ax.set_xlabel(r"$\alpha_2$ [deg]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin=-4, xmax=4)

        ax.grid(True)

    ax.set_xticks([-4,-2,0,2 ,4])
    plt.show()
    return fig, ax



#%% !!!
plt.close()
section_name = ""
# sammenlinge hastigheter
fig, ax = plot_compare_distance_mean1(static_coeff_single_9_updated, static_coeff_MDS_1D_10_updated, static_coeff_MDS_2D_10_updated, static_coeff_MDS_3D_10_updated, static_coeff_MDS_4D_10_updated, static_coeff_MDS_5D_10_updated, scoff="drag", upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cd_mds" + "_dist.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean1(static_coeff_single_9_updated, static_coeff_MDS_1D_10_updated, static_coeff_MDS_2D_10_updated, static_coeff_MDS_3D_10_updated, static_coeff_MDS_4D_10_updated, static_coeff_MDS_5D_10_updated, scoff="lift", upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cl_mds" + "_dist.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean1(static_coeff_single_9_updated, static_coeff_MDS_1D_10_updated, static_coeff_MDS_2D_10_updated, static_coeff_MDS_3D_10_updated, static_coeff_MDS_4D_10_updated, static_coeff_MDS_5D_10_updated, scoff="pitch", upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cm_mds" + "_dist.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax = plot_compare_distance_mean1(static_coeff_single_9_updated, static_coeff_MUS_1D_10_updated, static_coeff_MUS_2D_10_updated, static_coeff_MUS_3D_10_updated, static_coeff_MUS_4D_10_updated, static_coeff_MUS_5D_10_updated, scoff="drag", upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cd_mus" + "_dist.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean1(static_coeff_single_9_updated, static_coeff_MUS_1D_10_updated, static_coeff_MUS_2D_10_updated, static_coeff_MUS_3D_10_updated, static_coeff_MUS_4D_10_updated, static_coeff_MUS_5D_10_updated, scoff="lift", upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cl_mus" + "_dist.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean1(static_coeff_single_9_updated, static_coeff_MUS_1D_10_updated, static_coeff_MUS_2D_10_updated, static_coeff_MUS_3D_10_updated, static_coeff_MUS_4D_10_updated, static_coeff_MUS_5D_10_updated, scoff="pitch", upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cm_mus" + "_dist.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()



#%% !!!
plt.close()
section_name = ""
# sammenlinge hastigheter
fig, ax = plot_compare_distance_mean_oposite(static_coeff_single_9_updated, static_coeff_MDS_1D_10_updated, static_coeff_MDS_2D_10_updated, static_coeff_MDS_3D_10_updated, static_coeff_MDS_4D_10_updated, static_coeff_MDS_5D_10_updated, scoff="drag", upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cd_mds" + "_dist2.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean_oposite(static_coeff_single_9_updated, static_coeff_MDS_1D_10_updated, static_coeff_MDS_2D_10_updated, static_coeff_MDS_3D_10_updated, static_coeff_MDS_4D_10_updated, static_coeff_MDS_5D_10_updated, scoff="lift", upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cl_mds" + "_dist2.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean_oposite(static_coeff_single_9_updated, static_coeff_MDS_1D_10_updated, static_coeff_MDS_2D_10_updated, static_coeff_MDS_3D_10_updated, static_coeff_MDS_4D_10_updated, static_coeff_MDS_5D_10_updated, scoff="pitch", upwind_in_rig=False, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cm_mds" + "_dist2.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax = plot_compare_distance_mean_oposite(static_coeff_single_9_updated, static_coeff_MUS_1D_10_updated, static_coeff_MUS_2D_10_updated, static_coeff_MUS_3D_10_updated, static_coeff_MUS_4D_10_updated, static_coeff_MUS_5D_10_updated, scoff="drag", upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cd_mus" + "_dist2.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean_oposite(static_coeff_single_9_updated, static_coeff_MUS_1D_10_updated, static_coeff_MUS_2D_10_updated, static_coeff_MUS_3D_10_updated, static_coeff_MUS_4D_10_updated, static_coeff_MUS_5D_10_updated, scoff="lift", upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cl_mus" + "_dist2.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

fig, ax  = plot_compare_distance_mean_oposite(static_coeff_single_9_updated, static_coeff_MUS_1D_10_updated, static_coeff_MUS_2D_10_updated, static_coeff_MUS_3D_10_updated, static_coeff_MUS_4D_10_updated, static_coeff_MUS_5D_10_updated, scoff="pitch", upwind_in_rig=True, ax=None)
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave" , section_name + f"cm_mus" + "_dist2.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()


##########################################################################
#%% 
# Save arrays to .npy files
#numpy array
file_path = ".\\Arrays_Static_coeff\\"
#Dictionary of arrays to save
arrays_to_save={
    "cd_single_high": static_coeff_single_9_updated.plot_drag_mean( mode="single", upwind_in_rig=True)[0],
    "cd_alpha_single_high": static_coeff_single_9_updated.plot_drag_mean( mode="single", upwind_in_rig=True)[1],
    "cl_single_high": static_coeff_single_9_updated.plot_lift_mean( mode="single", upwind_in_rig=True)[0],
    "cl_alpha_single_high": static_coeff_single_9_updated.plot_lift_mean( mode="single", upwind_in_rig=True)[1],
    "cm_single_high": static_coeff_single_9_updated.plot_pitch_mean( mode="single", upwind_in_rig=True)[0],
    "cm_alpha_single_high": static_coeff_single_9_updated.plot_pitch_mean( mode="single", upwind_in_rig=True)[1],

    "cd_1D_mds_high_upwind_deck": static_coeff_MDS_1D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[0],
    "cd_1D_mds_high_downwind_deck": static_coeff_MDS_1D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[1],
    "cd_alpha_1D_mds_high": static_coeff_MDS_1D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[2],
    "cl_1D_mds_high_upwind_deck": static_coeff_MDS_1D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[0],
    "cl_1D_mds_high_downwind_deck": static_coeff_MDS_1D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[1],
    "cl_alpha_1D_mds_high": static_coeff_MDS_1D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[2],
    "cm_1D_mds_high_upwind_deck": static_coeff_MDS_1D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[0],
    "cm_1D_mds_high_downwind_deck": static_coeff_MDS_1D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[1],
    "cm_alpha_1D_mds_high": static_coeff_MDS_1D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[2],

    "cd_1D_mus_high_upwind_deck": static_coeff_MUS_1D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[0],
    "cd_1D_mus_high_downwind_deck": static_coeff_MUS_1D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[1],
    "cd_alpha_1D_mus_high": static_coeff_MUS_1D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[2],
    "cl_1D_mus_high_upwind_deck": static_coeff_MUS_1D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[0],
    "cl_1D_mus_high_downwind_deck": static_coeff_MUS_1D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[1],
    "cl_alpha_1D_mus_high": static_coeff_MUS_1D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[2],
    "cm_1D_mus_high_upwind_deck": static_coeff_MUS_1D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[0],
    "cm_1D_mus_high_downwind_deck": static_coeff_MUS_1D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[1],
    "cm_alpha_1D_mus_high": static_coeff_MUS_1D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[2],

    "cd_2D_mds_high_upwind_deck": static_coeff_MDS_2D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[0],
    "cd_2D_mds_high_downwind_deck": static_coeff_MDS_2D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[1],
    "cd_alpha_2D_mds_high": static_coeff_MDS_2D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[2],
    "cl_2D_mds_high_upwind_deck": static_coeff_MDS_2D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[0],
    "cl_2D_mds_high_downwind_deck": static_coeff_MDS_2D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[1],
    "cl_alpha_2D_mds_high": static_coeff_MDS_2D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[2],
    "cm_2D_mds_high_upwind_deck": static_coeff_MDS_2D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[0],
    "cm_2D_mds_high_downwind_deck": static_coeff_MDS_2D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[1],
    "cm_alpha_2D_mds_high": static_coeff_MDS_2D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[2],

    "cd_2D_mus_high_upwind_deck": static_coeff_MUS_2D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[0],
    "cd_2D_mus_high_downwind_deck": static_coeff_MUS_2D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[1],
    "cd_alpha_2D_mus_high": static_coeff_MUS_2D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[2],
    "cl_2D_mus_high_upwind_deck": static_coeff_MUS_2D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[0],
    "cl_2D_mus_high_downwind_deck": static_coeff_MUS_2D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[1],
    "cl_alpha_2D_mus_high": static_coeff_MUS_2D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[2],
    "cm_2D_mus_high_upwind_deck": static_coeff_MUS_2D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[0],
    "cm_2D_mus_high_downwind_deck": static_coeff_MUS_2D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[1],
    "cm_alpha_2D_mus_high": static_coeff_MUS_2D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[2],

    "cd_3D_mds_high_upwind_deck": static_coeff_MDS_3D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[0],
    "cd_3D_mds_high_downwind_deck": static_coeff_MDS_3D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[1],
    "cd_alpha_3D_mds_high": static_coeff_MDS_3D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[2],
    "cl_3D_mds_high_upwind_deck": static_coeff_MDS_3D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[0],
    "cl_3D_mds_high_downwind_deck": static_coeff_MDS_3D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[1],
    "cl_alpha_3D_mds_high": static_coeff_MDS_3D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[2],
    "cm_3D_mds_high_upwind_deck": static_coeff_MDS_3D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[0],
    "cm_3D_mds_high_downwind_deck": static_coeff_MDS_3D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[1],
    "cm_alpha_3D_mds_high": static_coeff_MDS_3D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[2],

    "cd_3D_mus_high_upwind_deck": static_coeff_MUS_3D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[0],
    "cd_3D_mus_high_downwind_deck": static_coeff_MUS_3D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[1],
    "cd_alpha_3D_mus_high": static_coeff_MUS_3D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[2],
    "cl_3D_mus_high_upwind_deck": static_coeff_MUS_3D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[0],
    "cl_3D_mus_high_downwind_deck": static_coeff_MUS_3D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[1],
    "cl_alpha_3D_mus_high": static_coeff_MUS_3D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[2],
    "cm_3D_mus_high_upwind_deck": static_coeff_MUS_3D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[0],
    "cm_3D_mus_high_downwind_deck": static_coeff_MUS_3D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[1],
    "cm_alpha_3D_mus_high": static_coeff_MUS_3D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[2],

    "cd_4D_mds_high_upwind_deck": static_coeff_MDS_4D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[0],
    "cd_4D_mds_high_downwind_deck": static_coeff_MDS_4D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[1],
    "cd_alpha_4D_mds_high": static_coeff_MDS_4D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[2],
    "cl_4D_mds_high_upwind_deck": static_coeff_MDS_4D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[0],
    "cl_4D_mds_high_downwind_deck": static_coeff_MDS_4D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[1],
    "cl_alpha_4D_mds_high": static_coeff_MDS_4D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[2],
    "cm_4D_mds_high_upwind_deck": static_coeff_MDS_4D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[0],
    "cm_4D_mds_high_downwind_deck": static_coeff_MDS_4D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[1],
    "cm_alpha_4D_mds_high": static_coeff_MDS_4D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[2],

    "cd_4D_mus_high_upwind_deck": static_coeff_MUS_4D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[0],
    "cd_4D_mus_high_downwind_deck": static_coeff_MUS_4D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[1],
    "cd_alpha_4D_mus_high": static_coeff_MUS_4D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[2],
    "cl_4D_mus_high_upwind_deck": static_coeff_MUS_4D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[0],
    "cl_4D_mus_high_downwind_deck": static_coeff_MUS_4D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[1],
    "cl_alpha_4D_mus_high": static_coeff_MUS_4D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[2],
    "cm_4D_mus_high_upwind_deck": static_coeff_MUS_4D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[0],
    "cm_4D_mus_high_downwind_deck": static_coeff_MUS_4D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[1],
    "cm_alpha_4D_mus_high": static_coeff_MUS_4D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[2],

    "cd_5D_mds_high_upwind_deck": static_coeff_MDS_5D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[0],
    "cd_5D_mds_high_downwind_deck": static_coeff_MDS_5D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[1],
    "cd_alpha_5D_mds_high": static_coeff_MDS_5D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=False)[2],
    "cl_5D_mds_high_upwind_deck": static_coeff_MDS_5D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[0],
    "cl_5D_mds_high_downwind_deck": static_coeff_MDS_5D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[1],
    "cl_alpha_5D_mds_high": static_coeff_MDS_5D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=False)[2],
    "cm_5D_mds_high_upwind_deck": static_coeff_MDS_5D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[0],
    "cm_5D_mds_high_downwind_deck": static_coeff_MDS_5D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[1],
    "cm_alpha_5D_mds_high": static_coeff_MDS_5D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=False)[2],

    "cd_5D_mus_high_upwind_deck": static_coeff_MUS_5D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[0],
    "cd_5D_mus_high_downwind_deck": static_coeff_MUS_5D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[1],
    "cd_alpha_5D_mus_high": static_coeff_MUS_5D_10_updated.plot_drag_mean( mode="decks", upwind_in_rig=True)[2],
    "cl_5D_mus_high_upwind_deck": static_coeff_MUS_5D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[0],
    "cl_5D_mus_high_downwind_deck": static_coeff_MUS_5D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[1],
    "cl_alpha_5D_mus_high": static_coeff_MUS_5D_10_updated.plot_lift_mean( mode="decks", upwind_in_rig=True)[2],
    "cm_5D_mus_high_upwind_deck": static_coeff_MUS_5D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[0],
    "cm_5D_mus_high_downwind_deck": static_coeff_MUS_5D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[1],
    "cm_alpha_5D_mus_high": static_coeff_MUS_5D_10_updated.plot_pitch_mean( mode="decks", upwind_in_rig=True)[2],
    
}
# %%
print("hei")
for name,array in arrays_to_save.items():
    np.save(os.path.join(file_path, f"{name}.npy"), array)


#%%#
