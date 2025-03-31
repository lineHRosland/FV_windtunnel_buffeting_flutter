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

#%%
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
    "cm_5D_mds_high_downwind_deck", "cm_alpha_5D_mds_high",
    
    
    "cd_single_high_filtered", "cd_alpha_single_high_filtered",
    "cd_1D_mus_high_upwind_deck_filtered", "cd_alpha_1D_mus_high_filtered",
    "cd_2D_mus_high_upwind_deck_filtered","cd_alpha_2D_mus_high_filtered",
    "cd_3D_mus_high_upwind_deck_filtered", "cd_alpha_3D_mus_high_filtered",
    "cd_4D_mus_high_upwind_deck_filtered", "cd_alpha_4D_mus_high_filtered",

    "cd_1D_mds_high_upwind_deck_filtered", "cd_alpha_1D_mds_high_filtered",
 

    "cd_1D_mus_high_downwind_deck_filtered", "cd_alpha_1D_mus_high_filtered",
    "cd_2D_mus_high_downwind_deck_filtered", "cd_alpha_2D_mus_high_filtered",
    "cd_3D_mus_high_downwind_deck_filtered", "cd_alpha_3D_mus_high_filtered",
    "cd_4D_mus_high_downwind_deck_filtered", "cd_alpha_4D_mus_high_filtered",

    "cd_1D_mds_high_downwind_deck_filtered", "cd_alpha_1D_mds_high_filtered",




    "cl_single_high_filtered", "cl_alpha_single_high_filtered",
    "cl_1D_mus_high_upwind_deck_filtered", "cl_alpha_1D_mus_high_filtered",
    "cl_2D_mus_high_upwind_deck_filtered", "cl_alpha_2D_mus_high_filtered",
    "cl_3D_mus_high_upwind_deck_filtered", "cl_alpha_3D_mus_high_filtered",
    "cl_4D_mus_high_upwind_deck_filtered", "cl_alpha_4D_mus_high_filtered",
    "cl_5D_mus_high_upwind_deck_filtered", "cl_alpha_5D_mus_high_filtered",

    "cl_1D_mds_high_upwind_deck_filtered", "cl_alpha_1D_mds_high_filtered",
   

    "cl_1D_mus_high_downwind_deck_filtered", "cl_alpha_1D_mus_high_filtered",
    "cl_2D_mus_high_downwind_deck_filtered", "cl_alpha_2D_mus_high_filtered",
    "cl_3D_mus_high_downwind_deck_filtered", "cl_alpha_3D_mus_high_filtered",
    "cl_4D_mus_high_downwind_deck_filtered", "cl_alpha_4D_mus_high_filtered",
    "cl_5D_mus_high_downwind_deck_filtered", "cl_alpha_5D_mus_high_filtered",

    "cl_1D_mds_high_downwind_deck_filtered", "cl_alpha_1D_mds_high_filtered",
   

    "cl_1D_mus_high_downwind_deck_filtered", "cl_alpha_1D_mus_high_filtered",
    "cl_2D_mus_high_downwind_deck_filtered", "cl_alpha_2D_mus_high_filtered",
    "cl_3D_mus_high_downwind_deck_filtered", "cl_alpha_3D_mus_high_filtered",
    "cl_4D_mus_high_downwind_deck_filtered", "cl_alpha_4D_mus_high_filtered",
    "cl_5D_mus_high_downwind_deck_filtered", "cl_alpha_5D_mus_high_filtered",
    


    "cm_single_high_filtered", "cm_alpha_single_high_filtered",
    "cm_1D_mus_high_upwind_deck_filtered", "cm_alpha_1D_mus_high_filtered",
    "cm_2D_mus_high_upwind_deck_filtered", "cm_alpha_2D_mus_high_filtered",
    "cm_3D_mus_high_upwind_deck_filtered", "cm_alpha_3D_mus_high_filtered",
    "cm_4D_mus_high_upwind_deck_filtered", "cm_alpha_4D_mus_high_filtered",
    "cm_5D_mus_high_upwind_deck_filtered", "cm_alpha_5D_mus_high_filtered",

    "cm_1D_mds_high_upwind_deck_filtered", "cm_alpha_1D_mds_high_filtered",
    
    "cm_1D_mus_high_downwind_deck_filtered", "cm_alpha_1D_mus_high_filtered",
    "cm_2D_mus_high_downwind_deck_filtered", "cm_alpha_2D_mus_high_filtered",
    "cm_3D_mus_high_downwind_deck_filtered", "cm_alpha_3D_mus_high_filtered",
    "cm_4D_mus_high_downwind_deck_filtered", "cm_alpha_4D_mus_high_filtered",
    "cm_5D_mus_high_downwind_deck_filtered", "cm_alpha_5D_mus_high_filtered",

    "cm_1D_mds_high_downwind_deck_filtered", "cm_alpha_1D_mds_high_filtered",
    ###########################################################################

    "cd_single_low", "cd_alpha_single_low",
    "cd_1D_mus_low_upwind_deck", "cd_alpha_1D_mus_low",
    "cd_2D_mus_low_upwind_deck","cd_alpha_2D_mus_low",
    "cd_3D_mus_low_upwind_deck", "cd_alpha_3D_mus_low",
    "cd_4D_mus_low_upwind_deck", "cd_alpha_4D_mus_low",
    "cd_5D_mus_low_upwind_deck", "cd_alpha_5D_mus_low",

    "cd_1D_mds_low_upwind_deck", "cd_alpha_1D_mds_low",
    "cd_2D_mds_low_upwind_deck", "cd_alpha_2D_mds_low",
    "cd_3D_mds_low_upwind_deck", "cd_alpha_3D_mds_low",
    "cd_4D_mds_low_upwind_deck", "cd_alpha_4D_mds_low",
    "cd_5D_mds_low_upwind_deck", "cd_alpha_5D_mds_low",

    "cd_1D_mus_low_downwind_deck", "cd_alpha_1D_mus_low",
    "cd_2D_mus_low_downwind_deck", "cd_alpha_2D_mus_low",
    "cd_3D_mus_low_downwind_deck", "cd_alpha_3D_mus_low",
    "cd_4D_mus_low_downwind_deck", "cd_alpha_4D_mus_low",
    "cd_5D_mus_low_downwind_deck", "cd_alpha_5D_mus_low",

    "cd_1D_mds_low_downwind_deck", "cd_alpha_1D_mds_low",
    "cd_2D_mds_low_downwind_deck", "cd_alpha_2D_mds_low",
    "cd_3D_mds_low_downwind_deck", "cd_alpha_3D_mds_low",
    "cd_4D_mds_low_downwind_deck", "cd_alpha_4D_mds_low",
    "cd_5D_mds_low_downwind_deck", "cd_alpha_5D_mds_low",



    "cl_single_low", "cl_alpha_single_low",
    "cl_1D_mus_low_upwind_deck", "cl_alpha_1D_mus_low",
    "cl_2D_mus_low_upwind_deck", "cl_alpha_2D_mus_low",
    "cl_3D_mus_low_upwind_deck", "cl_alpha_3D_mus_low",
    "cl_4D_mus_low_upwind_deck", "cl_alpha_4D_mus_low",
    "cl_5D_mus_low_upwind_deck", "cl_alpha_5D_mus_low",

    "cl_1D_mds_low_upwind_deck", "cl_alpha_1D_mds_low",
    "cl_2D_mds_low_upwind_deck", "cl_alpha_2D_mds_low",
    "cl_3D_mds_low_upwind_deck", "cl_alpha_3D_mds_low",
    "cl_4D_mds_low_upwind_deck", "cl_alpha_4D_mds_low",
    "cl_5D_mds_low_upwind_deck", "cl_alpha_5D_mds_low",

    "cl_1D_mus_low_downwind_deck", "cl_alpha_1D_mus_low",
    "cl_2D_mus_low_downwind_deck", "cl_alpha_2D_mus_low",
    "cl_3D_mus_low_downwind_deck", "cl_alpha_3D_mus_low",
    "cl_4D_mus_low_downwind_deck", "cl_alpha_4D_mus_low",
    "cl_5D_mus_low_downwind_deck", "cl_alpha_5D_mus_low",

    "cl_1D_mds_low_downwind_deck", "cl_alpha_1D_mds_low",
    "cl_2D_mds_low_downwind_deck", "cl_alpha_2D_mds_low",
    "cl_3D_mds_low_downwind_deck", "cl_alpha_3D_mds_low",
    "cl_4D_mds_low_downwind_deck", "cl_alpha_4D_mds_low",
    "cl_5D_mds_low_downwind_deck", "cl_alpha_5D_mds_low",



    "cm_single_low", "cm_alpha_single_low",
    "cm_1D_mus_low_upwind_deck", "cm_alpha_1D_mus_low",
    "cm_2D_mus_low_upwind_deck", "cm_alpha_2D_mus_low",
    "cm_3D_mus_low_upwind_deck", "cm_alpha_3D_mus_low",
    "cm_4D_mus_low_upwind_deck", "cm_alpha_4D_mus_low",
    "cm_5D_mus_low_upwind_deck", "cm_alpha_5D_mus_low",

    "cm_1D_mds_low_upwind_deck", "cm_alpha_1D_mds_low",
    "cm_2D_mds_low_upwind_deck", "cm_alpha_2D_mds_low",
    "cm_3D_mds_low_upwind_deck", "cm_alpha_3D_mds_low",
    "cm_4D_mds_low_upwind_deck", "cm_alpha_4D_mds_low",
    "cm_5D_mds_low_upwind_deck", "cm_alpha_5D_mds_low",

    "cm_1D_mus_low_downwind_deck", "cm_alpha_1D_mus_low",
    "cm_2D_mus_low_downwind_deck", "cm_alpha_2D_mus_low",
    "cm_3D_mus_low_downwind_deck", "cm_alpha_3D_mus_low",
    "cm_4D_mus_low_downwind_deck", "cm_alpha_4D_mus_low",
    "cm_5D_mus_low_downwind_deck", "cm_alpha_5D_mus_low",

    "cm_1D_mds_low_downwind_deck", "cm_alpha_1D_mds_low",
    "cm_2D_mds_low_downwind_deck", "cm_alpha_2D_mds_low",
    "cm_3D_mds_low_downwind_deck", "cm_alpha_3D_mds_low",
    "cm_4D_mds_low_downwind_deck", "cm_alpha_4D_mds_low",
    "cm_5D_mds_low_downwind_deck", "cm_alpha_5D_mds_low",
    
    
    "cd_single_low_filtered", "cd_alpha_single_low_filtered",
    "cd_1D_mus_low_upwind_deck_filtered", "cd_alpha_1D_mus_low_filtered",
    "cd_2D_mus_low_upwind_deck_filtered","cd_alpha_2D_mus_low_filtered",
    "cd_3D_mus_low_upwind_deck_filtered", "cd_alpha_3D_mus_low_filtered",
    "cd_4D_mus_low_upwind_deck_filtered", "cd_alpha_4D_mus_low_filtered",

    "cd_1D_mds_low_upwind_deck_filtered", "cd_alpha_1D_mds_low_filtered",
 

    "cd_1D_mus_low_downwind_deck_filtered", "cd_alpha_1D_mus_low_filtered",
    "cd_2D_mus_low_downwind_deck_filtered", "cd_alpha_2D_mus_low_filtered",
    "cd_3D_mus_low_downwind_deck_filtered", "cd_alpha_3D_mus_low_filtered",
    "cd_4D_mus_low_downwind_deck_filtered", "cd_alpha_4D_mus_low_filtered",

    "cd_1D_mds_low_downwind_deck_filtered", "cd_alpha_1D_mds_low_filtered",




    "cl_single_low_filtered", "cl_alpha_single_low_filtered",
    "cl_1D_mus_low_upwind_deck_filtered", "cl_alpha_1D_mus_low_filtered",
    "cl_2D_mus_low_upwind_deck_filtered", "cl_alpha_2D_mus_low_filtered",
    "cl_3D_mus_low_upwind_deck_filtered", "cl_alpha_3D_mus_low_filtered",
    "cl_4D_mus_low_upwind_deck_filtered", "cl_alpha_4D_mus_low_filtered",
    "cl_5D_mus_low_upwind_deck_filtered", "cl_alpha_5D_mus_low_filtered",

    "cl_1D_mds_low_upwind_deck_filtered", "cl_alpha_1D_mds_low_filtered",
   

    "cl_1D_mus_low_downwind_deck_filtered", "cl_alpha_1D_mus_low_filtered",
    "cl_2D_mus_low_downwind_deck_filtered", "cl_alpha_2D_mus_low_filtered",
    "cl_3D_mus_low_downwind_deck_filtered", "cl_alpha_3D_mus_low_filtered",
    "cl_4D_mus_low_downwind_deck_filtered", "cl_alpha_4D_mus_low_filtered",
    "cl_5D_mus_low_downwind_deck_filtered", "cl_alpha_5D_mus_low_filtered",

    "cl_1D_mds_low_downwind_deck_filtered", "cl_alpha_1D_mds_low_filtered",
   

    "cl_1D_mus_low_downwind_deck_filtered", "cl_alpha_1D_mus_low_filtered",
    "cl_2D_mus_low_downwind_deck_filtered", "cl_alpha_2D_mus_low_filtered",
    "cl_3D_mus_low_downwind_deck_filtered", "cl_alpha_3D_mus_low_filtered",
    "cl_4D_mus_low_downwind_deck_filtered", "cl_alpha_4D_mus_low_filtered",
    "cl_5D_mus_low_downwind_deck_filtered", "cl_alpha_5D_mus_low_filtered",
    


    "cm_single_low_filtered", "cm_alpha_single_low_filtered",
    "cm_1D_mus_low_upwind_deck_filtered", "cm_alpha_1D_mus_low_filtered",
    "cm_2D_mus_low_upwind_deck_filtered", "cm_alpha_2D_mus_low_filtered",
    "cm_3D_mus_low_upwind_deck_filtered", "cm_alpha_3D_mus_low_filtered",
    "cm_4D_mus_low_upwind_deck_filtered", "cm_alpha_4D_mus_low_filtered",
    "cm_5D_mus_low_upwind_deck_filtered", "cm_alpha_5D_mus_low_filtered",

    "cm_1D_mds_low_upwind_deck_filtered", "cm_alpha_1D_mds_low_filtered",
    
    "cm_1D_mus_low_downwind_deck_filtered", "cm_alpha_1D_mus_low_filtered",
    "cm_2D_mus_low_downwind_deck_filtered", "cm_alpha_2D_mus_low_filtered",
    "cm_3D_mus_low_downwind_deck_filtered", "cm_alpha_3D_mus_low_filtered",
    "cm_4D_mus_low_downwind_deck_filtered", "cm_alpha_4D_mus_low_filtered",
    "cm_5D_mus_low_downwind_deck_filtered", "cm_alpha_5D_mus_low_filtered",

    "cm_1D_mds_low_downwind_deck_filtered", "cm_alpha_1D_mds_low_filtered",



  ]
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



#Har ikke lagt til linjer for filtrert versjon her...
#Har ikke lagt til linjer for lav vindhastighet her...






# %%
# Data hentet ut:
# static_coeff_single_6
# static_coeff_single_9 #
# static_coeff_single_6_filtered
# static_coeff_single_9_filtered #

# static_coeff_MDS_1D_6
# static_coeff_MDS_1D_8 #for å vise at 8 m/s er dårlig
# static_coeff_MDS_1D_10 # 
# static_coeff_MDS_1D_6_filtered
# static_coeff_MDS_1D_10_filtered #
# static_coeff_MUS_1D_5
# static_coeff_MUS_1D_10 #
# static_coeff_MUS_1D_5_filtered
# static_coeff_MUS_1D_10_filtered #

# static_coeff_MDS_2D_6
# static_coeff_MDS_2D_10 #
# static_coeff_MUS_2D_5
# static_coeff_MUS_2D_10 #
# static_coeff_MUS_2D_5_filtered
# static_coeff_MUS_2D_10_filtered #

# static_coeff_MDS_3D_6
# static_coeff_MDS_3D_10 #
# static_coeff_MUS_3D_5
# static_coeff_MUS_3D_10 #
# static_coeff_MUS_3D_5_filtered
# static_coeff_MUS_3D_10_filtered #

# static_coeff_MDS_4D_55
# static_coeff_MDS_4D_10 #
# static_coeff_MUS_4D_5
# static_coeff_MUS_4D_10 #
# static_coeff_MUS_4D_5_filtered
# static_coeff_MUS_4D_10_filtered #

# static_coeff_MDS_5D_55
# static_coeff_MDS_5D_10
# static_coeff_MUS_5D_45
# static_coeff_MUS_5D_10
# static_coeff_MUS_5D_45_filtered #drag dårlig data her
# static_coeff_MUS_5D_10_filtered #drag dårlig data her
