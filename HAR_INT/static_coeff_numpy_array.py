# -*- coding: utf-8 -*-
"""
Created in March 2025

@author: linehro
"""
#%%
import numpy as np
import sys
sys.path.append(r"C:\Users\liner\Documents\Github\Masteroppgave\w3tp")
import w3t as w3t
import os

import 


# Define the path where arrays are stored
file_path = ".\\Arrays_Static_coeff\\"

# Dictionary to store loaded arrays
arrays_loaded = {}

# List of array names
array_names = [
    "cd_single_high", "cd_alpha_single_high",
    "cd_1D_mus_high_upwind_deck", "cd_alpha_1D_mus_high",
    "cd_2D_mus_high_upwind_deck","cd_alpha_2D_mus_high",
    "cd_3D_mus_high_upwind_deck", "cd_alpha_3D_mus_high",
    "cd_4D_mus_high_upwind_deck", "cd_alpha_4D_mus_high",
    "cd_5D_mus_high_upwind_deck", "cd_alpha_5D_mus_high",

    "cd_1D_mds_high_upwind_deck", "cd_alpha_1D_mds_high",
    "cd_2D_mds_high_upwind_deck", "cd_alpha_2D_mds_high",
    "cd_3D_mds_high_upwind_deck", "cd_alpha_3D_mds_high",
    "cd_4D_mds_high_upwind_deck", "cd_alpha_4D_mds_high",
    "cd_5D_mds_high_upwind_deck", "cd_alpha_5D_mds_high",

    "cd_1D_mus_high_downwind_deck", "cd_alpha_1D_mus_high",
    "cd_2D_mus_high_downwind_deck", "cd_alpha_2D_mus_high",
    "cd_3D_mus_high_downwind_deck", "cd_alpha_3D_mus_high",
    "cd_4D_mus_high_downwind_deck", "cd_alpha_4D_mus_high",
    "cd_5D_mus_high_downwind_deck", "cd_alpha_5D_mus_high",

    "cd_1D_mds_high_downwind_deck", "cd_alpha_1D_mds_high",
    "cd_2D_mds_high_downwind_deck", "cd_alpha_2D_mds_high",
    "cd_3D_mds_high_downwind_deck", "cd_alpha_3D_mds_high",
    "cd_4D_mds_high_downwind_deck", "cd_alpha_4D_mds_high",
    "cd_5D_mds_high_downwind_deck", "cd_alpha_5D_mds_high",



    "cl_single_high", "cl_alpha_single_high",
    "cl_1D_mus_high_upwind_deck", "cl_alpha_1D_mus_high",
    "cl_2D_mus_high_upwind_deck", "cl_alpha_2D_mus_high",
    "cl_3D_mus_high_upwind_deck", "cl_alpha_3D_mus_high",
    "cl_4D_mus_high_upwind_deck", "cl_alpha_4D_mus_high",
    "cl_5D_mus_high_upwind_deck", "cl_alpha_5D_mus_high",

    "cl_1D_mds_high_upwind_deck", "cl_alpha_1D_mds_high",
    "cl_2D_mds_high_upwind_deck", "cl_alpha_2D_mds_high",
    "cl_3D_mds_high_upwind_deck", "cl_alpha_3D_mds_high",
    "cl_4D_mds_high_upwind_deck", "cl_alpha_4D_mds_high",
    "cl_5D_mds_high_upwind_deck", "cl_alpha_5D_mds_high",

    "cl_1D_mus_high_downwind_deck", "cl_alpha_1D_mus_high",
    "cl_2D_mus_high_downwind_deck", "cl_alpha_2D_mus_high",
    "cl_3D_mus_high_downwind_deck", "cl_alpha_3D_mus_high",
    "cl_4D_mus_high_downwind_deck", "cl_alpha_4D_mus_high",
    "cl_5D_mus_high_downwind_deck", "cl_alpha_5D_mus_high",

    "cl_1D_mds_high_downwind_deck", "cl_alpha_1D_mds_high",
    "cl_2D_mds_high_downwind_deck", "cl_alpha_2D_mds_high",
    "cl_3D_mds_high_downwind_deck", "cl_alpha_3D_mds_high",
    "cl_4D_mds_high_downwind_deck", "cl_alpha_4D_mds_high",
    "cl_5D_mds_high_downwind_deck", "cl_alpha_5D_mds_high",



    "cm_single_high", "cm_alpha_single_high",
    "cm_1D_mus_high_upwind_deck", "cm_alpha_1D_mus_high",
    "cm_2D_mus_high_upwind_deck", "cm_alpha_2D_mus_high",
    "cm_3D_mus_high_upwind_deck", "cm_alpha_3D_mus_high",
    "cm_4D_mus_high_upwind_deck", "cm_alpha_4D_mus_high",
    "cm_5D_mus_high_upwind_deck", "cm_alpha_5D_mus_high",

    "cm_1D_mds_high_upwind_deck", "cm_alpha_1D_mds_high",
    "cm_2D_mds_high_upwind_deck", "cm_alpha_2D_mds_high",
    "cm_3D_mds_high_upwind_deck", "cm_alpha_3D_mds_high",
    "cm_4D_mds_high_upwind_deck", "cm_alpha_4D_mds_high",
    "cm_5D_mds_high_upwind_deck", "cm_alpha_5D_mds_high",

    "cm_1D_mus_high_downwind_deck", "cm_alpha_1D_mus_high",
    "cm_2D_mus_high_downwind_deck", "cm_alpha_2D_mus_high",
    "cm_3D_mus_high_downwind_deck", "cm_alpha_3D_mus_high",
    "cm_4D_mus_high_downwind_deck", "cm_alpha_4D_mus_high",
    "cm_5D_mus_high_downwind_deck", "cm_alpha_5D_mus_high",

    "cm_1D_mds_high_downwind_deck", "cm_alpha_1D_mds_high",
    "cm_2D_mds_high_downwind_deck", "cm_alpha_2D_mds_high",
    "cm_3D_mds_high_downwind_deck", "cm_alpha_3D_mds_high",
    "cm_4D_mds_high_downwind_deck", "cm_alpha_4D_mds_high",
    "cm_5D_mds_high_downwind_deck", "cm_alpha_5D_mds_high",]
#High wind velocity chosen for the static coefficient.

# Load each .npy file
for name in array_names:
    arrays_loaded[name] = np.load(os.path.join(file_path, f"{name}.npy"))

# Assign variables directly if you prefer:
cd_single_high = arrays_loaded["cd_single_high"]
cd_alpha_single_high = arrays_loaded["cd_alpha_single_high"]

cd_1D_mus_high_upwind_deck = arrays_loaded["cd_1D_mus_high_upwind_deck"]
cd_alpha_1D_mus_high = arrays_loaded["cd_alpha_1D_mus_high"]

cd_2D_mus_high_upwind_deck = arrays_loaded["cd_2D_mus_high_upwind_deck"]
cd_alpha_2D_mus_high = arrays_loaded["cd_alpha_2D_mus_high"]

cd_3D_mus_high_upwind_deck = arrays_loaded["cd_3D_mus_high_upwind_deck"]
cd_alpha_3D_mus_high = arrays_loaded["cd_alpha_3D_mus_high"]

cd_4D_mus_high_upwind_deck = arrays_loaded["cd_4D_mus_high_upwind_deck"]
cd_alpha_4D_mus_high = arrays_loaded["cd_alpha_4D_mus_high"]

cd_5D_mus_high_upwind_deck = arrays_loaded["cd_5D_mus_high_upwind_deck"]
cd_alpha_5D_mus_high = arrays_loaded["cd_alpha_5D_mus_high"]

cd_1D_mds_high_upwind_deck = arrays_loaded["cd_1D_mds_high_upwind_deck"]
cd_alpha_1D_mds_high = arrays_loaded["cd_alpha_1D_mds_high"]

cd_2D_mds_high_upwind_deck = arrays_loaded["cd_2D_mds_high_upwind_deck"]
cd_alpha_2D_mds_high = arrays_loaded["cd_alpha_2D_mds_high"]

cd_3D_mds_high_upwind_deck = arrays_loaded["cd_3D_mds_high_upwind_deck"]
cd_alpha_3D_mds_high = arrays_loaded["cd_alpha_3D_mds_high"]

cd_4D_mds_high_upwind_deck = arrays_loaded["cd_4D_mds_high_upwind_deck"]
cd_alpha_4D_mds_high = arrays_loaded["cd_alpha_4D_mds_high"]

cd_5D_mds_high_upwind_deck = arrays_loaded["cd_5D_mds_high_upwind_deck"]
cd_alpha_5D_mds_high = arrays_loaded["cd_alpha_5D_mds_high"]

cd_1D_mus_high_downwind_deck = arrays_loaded["cd_1D_mus_high_downwind_deck"]
cd_alpha_1D_mus_high = arrays_loaded["cd_alpha_1D_mus_high"]

cd_2D_mus_high_downwind_deck = arrays_loaded["cd_2D_mus_high_downwind_deck"]
cd_alpha_2D_mus_high = arrays_loaded["cd_alpha_2D_mus_high"]

cd_3D_mus_high_downwind_deck = arrays_loaded["cd_3D_mus_high_downwind_deck"]
cd_alpha_3D_mus_high = arrays_loaded["cd_alpha_3D_mus_high"]

cd_4D_mus_high_downwind_deck = arrays_loaded["cd_4D_mus_high_downwind_deck"]
cd_alpha_4D_mus_high = arrays_loaded["cd_alpha_4D_mus_high"]

cd_5D_mus_high_downwind_deck = arrays_loaded["cd_5D_mus_high_downwind_deck"]
cd_alpha_5D_mds_high = arrays_loaded["cd_alpha_5D_mds_high"]

cd_1D_mds_high_downwind_deck = arrays_loaded["cd_1D_mds_high_downwind_deck"]
cd_alpha_1D_mds_high = arrays_loaded["cd_alpha_1D_mds_high"]

cd_2D_mds_high_downwind_deck = arrays_loaded["cd_2D_mds_high_downwind_deck"]
cd_alpha_2D_mds_high = arrays_loaded["cd_alpha_2D_mds_high"]

cd_3D_mds_high_downwind_deck = arrays_loaded["cd_3D_mds_high_downwind_deck"]
cd_alpha_3D_mds_high = arrays_loaded["cd_alpha_3D_mds_high"]

cd_4D_mds_high_downwind_deck = arrays_loaded["cd_4D_mds_high_downwind_deck"]
cd_alpha_4D_mds_high = arrays_loaded["cd_alpha_4D_mds_high"]

cd_5D_mds_high_downwind_deck = arrays_loaded["cd_5D_mds_high_downwind_deck"]
cd_alpha_5D_mds_high = arrays_loaded["cd_alpha_5D_mds_high"]





cl_single_high = arrays_loaded["cl_single_high"]
cl_alpha_single_high = arrays_loaded["cl_alpha_single_high"]

cl_1D_mus_high_upwind_deck = arrays_loaded["cl_1D_mus_high_upwind_deck"]
cl_alpha_1D_mus_high = arrays_loaded["cl_alpha_1D_mus_high"]

cl_2D_mus_high_upwind_deck = arrays_loaded["cl_2D_mus_high_upwind_deck"]
cl_alpha_2D_mus_high = arrays_loaded["cl_alpha_2D_mus_high"]

cl_3D_mus_high_upwind_deck = arrays_loaded["cl_3D_mus_high_upwind_deck"]
cl_alpha_3D_mus_high = arrays_loaded["cl_alpha_3D_mus_high"]

cl_4D_mus_high_upwind_deck = arrays_loaded["cl_4D_mus_high_upwind_deck"]
cl_alpha_4D_mus_high = arrays_loaded["cl_alpha_4D_mus_high"]

cl_5D_mus_high_upwind_deck = arrays_loaded["cl_5D_mus_high_upwind_deck"]
cl_alpha_5D_mus_high = arrays_loaded["cl_alpha_5D_mus_high"]

cl_1D_mds_high_upwind_deck = arrays_loaded["cl_1D_mds_high_upwind_deck"]
cl_alpha_1D_mds_high = arrays_loaded["cl_alpha_1D_mds_high"]

cl_2D_mds_high_upwind_deck = arrays_loaded["cl_2D_mds_high_upwind_deck"]
cl_alpha_2D_mds_high = arrays_loaded["cl_alpha_2D_mds_high"]

cl_3D_mds_high_upwind_deck = arrays_loaded["cl_3D_mds_high_upwind_deck"]
cl_alpha_3D_mds_high = arrays_loaded["cl_alpha_3D_mds_high"]

cl_4D_mds_high_upwind_deck = arrays_loaded["cl_4D_mds_high_upwind_deck"]
cl_alpha_4D_mds_high = arrays_loaded["cl_alpha_4D_mds_high"]

cl_5D_mds_high_upwind_deck = arrays_loaded["cl_5D_mds_high_upwind_deck"]
cl_alpha_5D_mds_high = arrays_loaded["cl_alpha_5D_mds_high"]

cl_1D_mus_high_downwind_deck = arrays_loaded["cl_1D_mus_high_downwind_deck"]
cl_alpha_1D_mus_high = arrays_loaded["cl_alpha_1D_mus_high"]

cl_2D_mus_high_downwind_deck = arrays_loaded["cl_2D_mus_high_downwind_deck"]
cl_alpha_2D_mus_high = arrays_loaded["cl_alpha_2D_mus_high"]

cl_3D_mus_high_downwind_deck = arrays_loaded["cl_3D_mus_high_downwind_deck"]
cl_alpha_3D_mus_high = arrays_loaded["cl_alpha_3D_mus_high"]

cl_4D_mus_high_downwind_deck = arrays_loaded["cl_4D_mus_high_downwind_deck"]
cl_alpha_4D_mus_high = arrays_loaded["cl_alpha_4D_mus_high"]

cl_5D_mus_high_downwind_deck = arrays_loaded["cl_5D_mus_high_downwind_deck"]
cl_alpha_5D_mds_high = arrays_loaded["cl_alpha_5D_mds_high"]

cl_1D_mds_high_downwind_deck = arrays_loaded["cl_1D_mds_high_downwind_deck"]
cl_alpha_1D_mds_high = arrays_loaded["cl_alpha_1D_mds_high"]

cl_2D_mds_high_downwind_deck = arrays_loaded["cl_2D_mds_high_downwind_deck"]
cl_alpha_2D_mds_high = arrays_loaded["cl_alpha_2D_mds_high"]

cl_3D_mds_high_downwind_deck = arrays_loaded["cl_3D_mds_high_downwind_deck"]
cl_alpha_3D_mds_high = arrays_loaded["cl_alpha_3D_mds_high"]

cl_4D_mds_high_downwind_deck = arrays_loaded["cl_4D_mds_high_downwind_deck"]
cl_alpha_4D_mds_high = arrays_loaded["cl_alpha_4D_mds_high"]

cl_5D_mds_high_downwind_deck = arrays_loaded["cl_5D_mds_high_downwind_deck"]
cl_alpha_5D_mds_high = arrays_loaded["cl_alpha_5D_mds_high"]



cm_single_high = arrays_loaded["cm_single_high"]
cm_alpha_single_high = arrays_loaded["cm_alpha_single_high"]

cm_1D_mus_high_upwind_deck = arrays_loaded["cm_1D_mus_high_upwind_deck"]
cm_alpha_1D_mus_high = arrays_loaded["cm_alpha_1D_mus_high"]

cm_2D_mus_high_upwind_deck = arrays_loaded["cm_2D_mus_high_upwind_deck"]
cm_alpha_2D_mus_high = arrays_loaded["cm_alpha_2D_mus_high"]

cm_3D_mus_high_upwind_deck = arrays_loaded["cm_3D_mus_high_upwind_deck"]
cm_alpha_3D_mus_high = arrays_loaded["cm_alpha_3D_mus_high"]

cm_4D_mus_high_upwind_deck = arrays_loaded["cm_4D_mus_high_upwind_deck"]
cm_alpha_4D_mus_high = arrays_loaded["cm_alpha_4D_mus_high"]

cm_5D_mus_high_upwind_deck = arrays_loaded["cm_5D_mus_high_upwind_deck"]
cm_alpha_5D_mus_high = arrays_loaded["cm_alpha_5D_mus_high"]

cm_1D_mds_high_upwind_deck = arrays_loaded["cm_1D_mds_high_upwind_deck"]
cm_alpha_1D_mds_high = arrays_loaded["cm_alpha_1D_mds_high"]

cm_2D_mds_high_upwind_deck = arrays_loaded["cm_2D_mds_high_upwind_deck"]
cm_alpha_2D_mds_high = arrays_loaded["cm_alpha_2D_mds_high"]

cm_3D_mds_high_upwind_deck = arrays_loaded["cm_3D_mds_high_upwind_deck"]
cm_alpha_3D_mds_high = arrays_loaded["cm_alpha_3D_mds_high"]

cm_4D_mds_high_upwind_deck = arrays_loaded["cm_4D_mds_high_upwind_deck"]
cm_alpha_4D_mds_high = arrays_loaded["cm_alpha_4D_mds_high"]

cm_5D_mds_high_upwind_deck = arrays_loaded["cm_5D_mds_high_upwind_deck"]
cm_alpha_5D_mds_high = arrays_loaded["cm_alpha_5D_mds_high"]

cm_1D_mus_high_downwind_deck = arrays_loaded["cm_1D_mus_high_downwind_deck"]
cm_alpha_1D_mus_high = arrays_loaded["cm_alpha_1D_mus_high"]

cm_2D_mus_high_downwind_deck = arrays_loaded["cm_2D_mus_high_downwind_deck"]
cm_alpha_2D_mus_high = arrays_loaded["cm_alpha_2D_mus_high"]

cm_3D_mus_high_downwind_deck = arrays_loaded["cm_3D_mus_high_downwind_deck"]
cm_alpha_3D_mus_high = arrays_loaded["cm_alpha_3D_mus_high"]

cm_4D_mus_high_downwind_deck = arrays_loaded["cm_4D_mus_high_downwind_deck"]
cm_alpha_4D_mus_high = arrays_loaded["cm_alpha_4D_mus_high"]

cm_5D_mus_high_downwind_deck = arrays_loaded["cm_5D_mus_high_downwind_deck"]
cm_alpha_5D_mus_high = arrays_loaded["cm_alpha_5D_mus_high"]

cm_1D_mds_high_downwind_deck = arrays_loaded["cm_1D_mds_high_downwind_deck"]
cm_alpha_1D_mds_high = arrays_loaded["cm_alpha_1D_mds_high"]

cm_2D_mds_high_downwind_deck = arrays_loaded["cm_2D_mds_high_downwind_deck"]
cm_alpha_2D_mds_high = arrays_loaded["cm_alpha_2D_mds_high"]

cm_3D_mds_high_downwind_deck = arrays_loaded["cm_3D_mds_high_downwind_deck"]
cm_alpha_3D_mds_high = arrays_loaded["cm_alpha_3D_mds_high"]

cm_4D_mds_high_downwind_deck = arrays_loaded["cm_4D_mds_high_downwind_deck"]
cm_alpha_4D_mds_high = arrays_loaded["cm_alpha_4D_mds_high"]

cm_5D_mds_high_downwind_deck = arrays_loaded["cm_5D_mds_high_downwind_deck"]
cm_alpha_5D_mds_high = arrays_loaded["cm_alpha_5D_mds_high"]




# v_range --> [v_min, v_max] (reduced velocity range)
# poly_coeff --> number of ADs x [a0, a1, a2], AD(v) = a0 + a1*v + a2*v^2

# Single order: H_1*, H_2*, H_3*, H_4*, A_1*, A_2*, A_3*, A_4*
# Tandem order: c_z1z1*, c_z1θ1*, c_z1z2*, c_z1θ2*, c_θ1z1*, c_θ1θ1*, c_θ1z2*, c_θ1θ2*, 
#               c_z2z1*, c_z2θ1*, c_z2z2*, c_z2θ2*, c_θ2z1*, c_θ2θ1*, c_θ2z2*, c_θ2θ2*, 
#               k_z1z1*, k_z1θ1*, k_z1z2*, k_z1θ2*, k_θ1z1*, k_θ1θ1*, k_θ1z2*, k_θ1θ2*, 
#               k_z2z1*, k_z2θ1*, k_z2z2*, k_z2θ2*, k_θ2z1*, k_θ2θ1*, k_θ2z2*, k_θ2