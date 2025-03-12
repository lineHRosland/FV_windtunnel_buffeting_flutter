# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:07:37 2022

@author: oiseth test
"""
#%%
import numpy as np
import sys
sys.path.append(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Python\Ole_sin_kode\w3tp")
import w3t
import os
import h5py
from matplotlib import pyplot as plt
import time

#%% Get all *.tdms files in directory
files = []
data_path = r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Python\Ole_sin_kode\HAR_INT\data\Single\Harmonic\\"

for file in os.listdir(data_path):
    if file.endswith(".tdms"):
        files.append(file)

section = "Single_Harmonic"
#%% Make hdf5 file containing all experiments
h5_output_path = r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Python\Ole_sin_kode\HAR_INT\H5F\\"
  
h5_file = os.path.join(h5_output_path, section)

ex1 = -0
ex2 = 0
for file in files:
    print(file)
    w3t.tdms2h5_4loadcells(h5_file,(data_path + file),ex1,ex2,wrong_side=False,load_cell_3_and_4_fixed=True)
