# -*- coding: utf-8 -*-
"""
Created in April 2025

@author: linehro
"""
#%%
import numpy as np
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # <- HAR_INT/
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # <- Masteroppgave/
sys.path.append(PARENT_DIR)
from w3tp.w3t import _eigVal 

file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Arrays_AD"

B = 0.365 # m, section width

# Values from FEM-model
ms1 = 50 #kg, vertical
ms2 = 15 #kg, torsion
fs1 = 0.11042 #Hz, vertical MODE 3
fs2 = 0.0979904 #Hz, torsion

ws1 = 2*np.pi*fs1 # rad/s, vertical FØRSTE ITERASJON
ws2 = 2*np.pi*fs2 # rad/s, torsion FØRSTE ITERASJON

zeta = 0.005 # 0.5 %, critical damping
rho = 1.225 # kg/m^3, air density


scale = 1/50

N = 100 # Number of steps in V_list

#%%
#ITERATIVE BIMODAL EIGENVALUE APPROACH
eps = 10 # Konvergensterskel
max_iter = 500  # Maksimalt antall iterasjoner

#%%
#Single deck
if os.path.exists(os.path.join(file_path, "poly_coeff_single.npy")):
    poly_coeff_single = np.load(os.path.join(file_path, "poly_coeff_single.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_single.npy' does not exist in the specified path: {os.path.abspath(file_path)}")
if os.path.exists(os.path.join(file_path, "v_range_single.npy")):
    v_range_single = np.load(os.path.join(file_path, "v_range_single.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_single.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

#print(poly_coeff_single.shape)  # Skal være (8, 3) (8 AD)
#print(v_range_single)           # Skal være (8,2)

damping_ratios_single, omega_all_single, eigvals_all_single, eigvecs_all_single= _eigVal.solve_omega(poly_coeff_single,v_range_single, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N = 100, single = True)

flutter_speed_modes_single =_eigVal.solve_flutter_speed( damping_ratios_single, N = 100, single = True)

print("Flutter speed modes: ", flutter_speed_modes_single)

_eigVal.plot_damping_vs_wind_speed_single(B,v_range_single, damping_ratios_single, omega_all_single, N = 100, single = True)

_eigVal.plot_frequency_vs_wind_speed(B, v_range_single, omega_all_single, N = 100, single = True)

#%%
#Double deck 1D
if os.path.exists(os.path.join(file_path, "poly_coeff_1D.npy")):
    poly_coeff_1D = np.load(os.path.join(file_path, "poly_coeff_1D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_1D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_1D.npy")):
    v_range_1D = np.load(os.path.join(file_path, "v_range_1D.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_1D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

damping_ratios_1D, omega_all_1D, eigvals_all_1D, eigvecs_all_1D, damping_ratios_local_1D, omega_all_local_1D, eigvals_all_local_1D, eigvecs_all_local_1D= _eigVal.solve_omega(poly_coeff_1D,v_range_1D, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N = 100, single = False)

flutter_speed_modes_1D=_eigVal.solve_flutter_speed( damping_ratios_1D, N = 100, single = False)

print("Flutter speed modes 1D: ", flutter_speed_modes_1D)


_eigVal.plot_damping_vs_wind_speed_single(B,v_range_1D, damping_ratios=damping_ratios_1D, omega_all = omega_all_1D, damping_ratios_local = damping_ratios_local_1D, omega_all_local = omega_all_local_1D, N = 100, single = False)

_eigVal.plot_frequency_vs_wind_speed(B, v_range_1D, omega_all=omega_all_1D, omega_all_local = omega_all_local_1D, N = 100, single = False)

#%%
#Double deck 2D
if os.path.exists(os.path.join(file_path, "poly_coeff_2D.npy")):
    poly_coeff_2D = np.load(os.path.join(file_path, "poly_coeff_2D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_2D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_2D.npy")):
    v_range_2D = np.load(os.path.join(file_path, "v_range_2D.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_2D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")


damping_ratios_2D, omega_all_2D, eigvals_all_2D, eigvecs_all_2D, damping_ratios_local_2D, omega_all_local_2D, eigvals_all_local_2D, eigvecs_all_local_2D= _eigVal.solve_omega(poly_coeff_2D,v_range_2D, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N = 100, single = False)

flutter_speed_modes_2D=_eigVal.solve_flutter_speed( damping_ratios_2D, N = 100, single = False)

print("Flutter speed modes 2D: ", flutter_speed_modes_2D)


_eigVal.plot_damping_vs_wind_speed_single(B,v_range_2D, damping_ratios=damping_ratios_2D, omega_all = omega_all_2D, damping_ratios_local = damping_ratios_local_2D, omega_all_local = omega_all_local_2D, N = 100, single = False)

_eigVal.plot_frequency_vs_wind_speed(B, v_range_2D, omega_all=omega_all_2D, omega_all_local = omega_all_local_2D, N = 100, single = False)



#%%
#Double deck 3D
if os.path.exists(os.path.join(file_path, "poly_coeff_3D.npy")):
    poly_coeff_3D = np.load(os.path.join(file_path, "poly_coeff_3D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_3D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_3D.npy")):
    v_range_3D = np.load(os.path.join(file_path, "v_range_3D.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_3D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

#print(poly_coeff_3D.shape)  # Skal være (32, 3) (32 AD)
#print(v_range_3D)           # Skal være f.eks. [min, max]

damping_ratios_3D, omega_all_3D, eigvals_all_3D, eigvecs_all_3D, damping_ratios_local_3D, omega_all_local_3D, eigvals_all_local_3D, eigvecs_all_local_3D= _eigVal.solve_omega(poly_coeff_3D,v_range_3D, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N = 100, single = False)

flutter_speed_modes_3D=_eigVal.solve_flutter_speed( damping_ratios_3D, N = 100, single = False)

print("Flutter speed modes 3D: ", flutter_speed_modes_3D)

_eigVal.plot_damping_vs_wind_speed_single(B,v_range_3D, damping_ratios=damping_ratios_3D, omega_all = omega_all_3D, damping_ratios_local = damping_ratios_local_3D, omega_all_local = omega_all_local_3D, N = 100, single = False)
_eigVal.plot_frequency_vs_wind_speed(B, v_range_3D, omega_all=omega_all_3D, omega_all_local = omega_all_local_3D, N = 100, single = False)


#%%
#Double deck 4D
if os.path.exists(os.path.join(file_path, "poly_coeff_4D.npy")):
    poly_coeff_4D = np.load(os.path.join(file_path, "poly_coeff_4D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_4D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_4D.npy")):
    v_range_4D = np.load(os.path.join(file_path, "v_range_4D.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_4D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

damping_ratios_4D, omega_all_4D, eigvals_all_4D, eigvecs_all_4D, damping_ratios_local_4D, omega_all_local_4D, eigvals_all_local_4D, eigvecs_all_local_4D= _eigVal.solve_omega(poly_coeff_4D,v_range_4D, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N = 100, single = False)

flutter_speed_modes_4D=_eigVal.solve_flutter_speed( damping_ratios_4D, N = 100, single = False)

print("Flutter speed modes 4D: ", flutter_speed_modes_4D)

_eigVal.plot_damping_vs_wind_speed_single(B,v_range_4D, damping_ratios=damping_ratios_4D, omega_all = omega_all_4D, damping_ratios_local = damping_ratios_local_4D, omega_all_local = omega_all_local_4D, N = 100, single = False)
_eigVal.plot_frequency_vs_wind_speed(B, v_range_4D, omega_all=omega_all_4D, omega_all_local = omega_all_local_4D, N = 100, single = False)

#%%
#Double deck 5D
if os.path.exists(os.path.join(file_path, "poly_coeff_5D.npy")):
    poly_coeff_5D = np.load(os.path.join(file_path, "poly_coeff_5D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_5D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_5D.npy")):
    v_range_5D = np.load(os.path.join(file_path, "v_range_5D.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_5D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

damping_ratios_5D, omega_all_5D, eigvals_all_5D, eigvecs_all_5D, damping_ratios_local_5D, omega_all_local_5D, eigvals_all_local_5D, eigvecs_all_local_5D= _eigVal.solve_omega(poly_coeff_5D,v_range_5D, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N = 100, single = False)

flutter_speed_modes_5D=_eigVal.solve_flutter_speed( damping_ratios_5D, N = 100, single = False)

print("Flutter speed modes 5D: ", flutter_speed_modes_5D)

_eigVal.plot_damping_vs_wind_speed_single(B,v_range_5D, damping_ratios=damping_ratios_5D, omega_all = omega_all_5D, damping_ratios_local = damping_ratios_local_5D, omega_all_local = omega_all_local_5D, N = 100, single = False)
_eigVal.plot_frequency_vs_wind_speed(B, v_range_5D, omega_all=omega_all_5D, omega_all_local = omega_all_local_5D, N = 100, single = False)