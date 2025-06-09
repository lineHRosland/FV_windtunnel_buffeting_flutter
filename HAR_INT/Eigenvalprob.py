# -*- coding: utf-8 -*-
"""
Created on Mon Feb 3 08:15:00 2025
Editited spring 2025
@author: Smetoch, Rosland
"""
#%%
import numpy as np
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # <- HAR_INT/
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # <- Masteroppgave/
sys.path.append(PARENT_DIR)
from w3tp.w3t import _eigVal 
from mode_shapes import mode_shape_single
from mode_shapes import mode_shape_two
import matplotlib.pyplot as plt



B = 18.3 # m, section width 
zeta = 0.005 # 5 %, critical damping
rho = 1.25 # kg/m^3, air density 


# Values from FEM-model (Pure structural modes in still air)
m1V = 4440631 #kg, vertical 
m2V = 6683762 #kg, vertical 
m1T = 5582542 #kg m^2, torsion 
f1V = 0.14066587 #Hz, vertical 
f2V = 0.19722386 #Hz, vertical  
f1T = 0.35948062 #Hz, torsion

w1V = 2*np.pi*f1V # rad/s, vertical FØRSTE ITERASJON
w2V = 2*np.pi*f2V # rad/s, vertical FØRSTE ITERASJON
w1T = 2*np.pi*f1T # rad/s, torsion FØRSTE ITERASJON

#ITERATIVE BIMODAL EIGENVALUE APPROACH
eps = 0.0001 # Konvergensterske

# Single
phi_single, x_single = mode_shape_single()

Ms_single, Cs_single, Ks_single = _eigVal.structural_matrices(m1V, m1T, f1V, f1T, zeta, single = True)

# two

phi_two, x_two = mode_shape_two()

Ms_two, Cs_two, Ks_two = _eigVal.structural_matrices(m1V, m1T, f1V, f1T, zeta, single = False)

#  STATIC
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\buffeting\Cae_Kae_updated_stat_coeff.npy"
# Load the saved dictionary
matrices = np.load(file_path, allow_pickle=True).item()

Kae_Single_gen = matrices["Kae_Single"]
Cae_Single_gen = matrices["Cae_Single"]

Kae_1D_gen = matrices["Kae_1D"]
Cae_1D_gen = matrices["Cae_1D"]

Kae_2D_gen = matrices["Kae_2D"]
Cae_2D_gen = matrices["Cae_2D"]

Kae_3D_gen = matrices["Kae_3D"]
Cae_3D_gen = matrices["Cae_3D"]

Kae_4D_gen = matrices["Kae_4D"]
Cae_4D_gen = matrices["Cae_4D"]

Kae_5D_gen = matrices["Kae_5D"]
Cae_5D_gen = matrices["Cae_5D"]

print("Cae_2_gen :", Cae_2D_gen)
print("Cae_3_gen :", Cae_3D_gen)
print("Cae_4_gen :", Cae_4D_gen)

print("kae_2_gen :", Kae_2D_gen)
print("kae_3_gen :", Kae_3D_gen)
print("kae_4_gen :", Kae_4D_gen)
# print
#Cae_5D_gen, Kae_5D_gen = _eigVal.generalize_C_K(Cae_5D, Kae_5D, phi_two, x_two, single=False)

#  AD
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Arrays_AD_k"
if os.path.exists(os.path.join(file_path, "poly_coeff_single.npy")):
    poly_coeff_single = np.load(os.path.join(file_path, "poly_coeff_single.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_single.npy' does not exist in the specified path: {os.path.abspath(file_path)}")
if os.path.exists(os.path.join(file_path, "k_range_single.npy")):
    k_range_single = np.load(os.path.join(file_path, "k_range_single.npy"))
else:
    raise FileNotFoundError(f"The file 'k_range_single.npy' does not exist in the specified path: {os.path.abspath(file_path)}")
if os.path.exists(os.path.join(file_path, "poly_coeff_1D.npy")):
    poly_coeff_1D = np.load(os.path.join(file_path, "poly_coeff_1D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_1D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "k_range_1D.npy")):
    k_range_1D = np.load(os.path.join(file_path, "k_range_1D.npy"))
else:
    raise FileNotFoundError(f"The file 'k_range_1D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "poly_coeff_2D.npy")):
    poly_coeff_2D = np.load(os.path.join(file_path, "poly_coeff_2D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_2D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "k_range_2D.npy")):
    k_range_2D = np.load(os.path.join(file_path, "k_range_2D.npy"))
else:
    raise FileNotFoundError(f"The file 'k_range_2D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "poly_coeff_3D.npy")):
    poly_coeff_3D = np.load(os.path.join(file_path, "poly_coeff_3D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_3D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "k_range_3D.npy")):
    k_range_3D = np.load(os.path.join(file_path, "k_range_3D.npy"))
else:
    raise FileNotFoundError(f"The file 'k_range_3D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")
if os.path.exists(os.path.join(file_path, "poly_coeff_4D.npy")):
    poly_coeff_4D = np.load(os.path.join(file_path, "poly_coeff_4D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_4D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "k_range_4D.npy")):
    k_range_4D = np.load(os.path.join(file_path, "k_range_4D.npy"))
else:
    raise FileNotFoundError(f"The file 'k_range_4D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "poly_coeff_5D.npy")):
    poly_coeff_5D = np.load(os.path.join(file_path, "poly_coeff_5D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_5D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "k_range_5D.npy")):
    k_range_5D = np.load(os.path.join(file_path, "k_range_5D.npy"))
else:
    raise FileNotFoundError(f"The file 'k_range_5D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")



#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################
#%%
#UNSTEADY
alphas = 0.7
#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_flutter(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, static_quasi = False,  Cae_star_gen_STAT=None, Kae_star_gen_STAT=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, alphas, dist="Single deck",  single = True, static_quasi = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, alphas, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')




fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')



#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_flutter(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = False,  Cae_star_gen_STAT=None, Kae_star_gen_STAT=None,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)


#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, alphas, dist="two deck 1D",  single = False, static_quasi = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D,  alphas,dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')



fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_flutter(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,static_quasi = False,  Cae_star_gen_STAT=None, Kae_star_gen_STAT=None,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_2D, eigvecs_list_two_2D, V_list_two_2D, alphas, dist="two deck 2D",  single = False, static_quasi = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D,  alphas,dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_flutter(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = False,  Cae_star_gen_STAT=None, Kae_star_gen_STAT=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D, V_list_two_3D,  alphas,dist="two deck 3D",  single = False, static_quasi = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D,  alphas,dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_flutter(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = False,  Cae_star_gen_STAT=None, Kae_star_gen_STAT=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_4D, eigvecs_list_two_4D, V_list_two_4D,  alphas,dist="two deck 4D",  single = False, static_quasi = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D,  alphas,dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_flutter(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = False,  Cae_star_gen_STAT=None, Kae_star_gen_STAT=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting
fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_5D, eigvecs_list_two_5D, V_list_two_5D, alphas, dist="two deck 5D",  single = False, static_quasi = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_damp" + ".png"), dpi=300)

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D,  alphas,dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_frek" + ".png"), dpi=300)


fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_dof" + ".png"), dpi=300)

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
#%%
# QUASI-STATIC
alphas = 0.5
#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_flutter(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, static_quasi = True, Cae_star_gen_STAT=Cae_Single_gen, Kae_star_gen_STAT=Kae_Single_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, alphas, dist="Single deck",  single = True, static_quasi = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_single_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, alphas, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_single_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_single_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_flutter(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = True,  Cae_star_gen_STAT=Cae_1D_gen, Kae_star_gen_STAT=Kae_1D_gen,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D,  alphas,dist="two deck 1D",  single = False, static_quasi = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_1D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, alphas, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_1D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_1D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_flutter(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,static_quasi = True,  Cae_star_gen_STAT=Cae_2D_gen, Kae_star_gen_STAT=Kae_2D_gen,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting


fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_2D,eigvecs_list_two_2D,V_list_two_2D,  alphas,dist="two deck 2D",  single = False, static_quasi = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_2D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D,  alphas,dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_2D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_2D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_flutter(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = True,  Cae_star_gen_STAT=Cae_3D_gen, Kae_star_gen_STAT=Kae_3D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting


fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D,V_list_two_3D,  alphas,dist="two deck 3D",  single = False, static_quasi = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_3D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D,  alphas,dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_3D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_3D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_flutter(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = True,  Cae_star_gen_STAT=Cae_4D_gen, Kae_star_gen_STAT=Kae_4D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting


fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_4D,eigvecs_list_two_4D,V_list_two_4D, alphas, dist="two deck 4D",  single = False, static_quasi = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_4D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D,  alphas,dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_4D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_4D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_flutter(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, static_quasi = True,  Cae_star_gen_STAT=Cae_5D_gen, Kae_star_gen_STAT=Kae_5D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_5D,eigvecs_list_two_5D,V_list_two_5D,  alphas,dist="two deck 5D",  single = False, static_quasi = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_5D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D,  alphas,dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_5D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "STAT_5D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

# %%
###########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

#%%
# Plotte unsteady vs quasi-static
#  quasi-static
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Buffeting\Cae_Kae_as_AD.npy"
# Load the saved dictionary
matrices = np.load(file_path, allow_pickle=True).item()

Single_k = matrices["Kae_Single"]
Single_c = matrices["Cae_Single"]

Static_1D_k = matrices["Kae_1D"]
Static_1D_c = matrices["Cae_1D"]

Static_2D_k = matrices["Kae_2D"]
Static_2D_c = matrices["Cae_2D"]

Static_3D_k = matrices["Kae_3D"]
Static_3D_c = matrices["Cae_3D"]

Static_4D_k = matrices["Kae_4D"]
Static_4D_c = matrices["Cae_4D"]

Static_5D_k = matrices["Kae_5D"]
Static_5D_c = matrices["Cae_5D"]
#Cae_5D_gen, Kae_5D_gen = _eigVal.generalize_C_K(Cae_5D, Kae_5D, phi_two, x_two, single=False)

def from_poly_k(poly_k, k_range, vred, damping_ad = True):
   
    if vred == 0:
        vred = 1e-10 # Prevent division by zero
        
    uit_step = lambda k,kc: 1./(1 + np.exp(-2*20*(k-kc)))
    fit = lambda p,k,k1c,k2c : np.polyval(p,k)*uit_step(k,k1c)*(1-uit_step(k,k2c)) + np.polyval(p,k1c)*(1-uit_step(k,k1c)) + np.polyval(p,k2c)*(uit_step(k,k2c))

    if damping_ad == True:
        ad_value = np.abs(vred)*fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
    else:
        ad_value = np.abs(vred)**2*fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
   
    #ad_value = fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
 
    return float(ad_value)
def AD_two(poly_coeff, k_range, Vred_global, B):

    Vred_global = float(Vred_global) 

    # AD
    # Damping derivatives (indices 0–15)

    c_z1z1 = from_poly_k(poly_coeff[0], k_range[0], Vred_global, damping_ad=True)
    c_z1θ1 = from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True)
    c_z1z2 = from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=True)
    c_z1θ2 = from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=True)
    c_θ1z1 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True)
    c_θ1θ1 = from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True)
    c_θ1z2 = from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=True)
    c_θ1θ2 = from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=True)
    c_z2z1 = from_poly_k(poly_coeff[8], k_range[8],Vred_global, damping_ad=True)
    c_z2θ1 = from_poly_k(poly_coeff[9], k_range[9],Vred_global, damping_ad=True)
    c_z2z2 = from_poly_k(poly_coeff[10], k_range[10],Vred_global, damping_ad=True)
    c_z2θ2 = from_poly_k(poly_coeff[11], k_range[11],Vred_global, damping_ad=True)
    c_θ2z1 = from_poly_k(poly_coeff[12], k_range[12],Vred_global, damping_ad=True)
    c_θ2θ1 = from_poly_k(poly_coeff[13], k_range[13],Vred_global, damping_ad=True)
    c_θ2z2 = from_poly_k(poly_coeff[14], k_range[14],Vred_global, damping_ad=True)
    c_θ2θ2 = from_poly_k(poly_coeff[15], k_range[15],Vred_global, damping_ad=True)

    # Stiffness derivatives (indices 16–31)
    k_z1z1 = from_poly_k(poly_coeff[16], k_range[16],Vred_global, damping_ad=False)
    k_z1θ1 = from_poly_k(poly_coeff[17], k_range[17],Vred_global, damping_ad=False)
    k_z1z2 = from_poly_k(poly_coeff[18], k_range[18],Vred_global, damping_ad=False)
    k_z1θ2 = from_poly_k(poly_coeff[19], k_range[19],Vred_global, damping_ad=False)
    k_θ1z1 = from_poly_k(poly_coeff[20], k_range[20],Vred_global, damping_ad=False)
    k_θ1θ1 = from_poly_k(poly_coeff[21], k_range[21],Vred_global, damping_ad=False)
    k_θ1z2 = from_poly_k(poly_coeff[22], k_range[22],Vred_global, damping_ad=False)
    k_θ1θ2 = from_poly_k(poly_coeff[23], k_range[23],Vred_global, damping_ad=False)
    k_z2z1 = from_poly_k(poly_coeff[24], k_range[24],Vred_global, damping_ad=False)
    k_z2θ1 = from_poly_k(poly_coeff[25], k_range[25],Vred_global, damping_ad=False)
    k_z2z2 = from_poly_k(poly_coeff[26], k_range[26],Vred_global, damping_ad=False)
    k_z2θ2 = from_poly_k(poly_coeff[27], k_range[27],Vred_global, damping_ad=False)
    k_θ2z1 = from_poly_k(poly_coeff[28], k_range[28],Vred_global, damping_ad=False)
    k_θ2θ1 = from_poly_k(poly_coeff[29], k_range[29],Vred_global, damping_ad=False)
    k_θ2z2 = from_poly_k(poly_coeff[30], k_range[30],Vred_global, damping_ad=False)
    k_θ2θ2 = from_poly_k(poly_coeff[31], k_range[31],Vred_global, damping_ad=False)
    
    Cae_star = np.array([
         [c_z1z1,    c_z1θ1,   c_z1z2,   c_z1θ2],
         [ c_θ1z1,   c_θ1θ1,   c_θ1z2,   c_θ1θ2],
         [c_z2z1,    c_z2θ1,   c_z2z2,   c_z2θ2],
         [c_θ2z1,    c_θ2θ1,    c_θ2z2,  c_θ2θ2]
    ])
    Kae_star = np.array([
         [k_z1z1,     k_z1θ1,  k_z1z2,  k_z1θ2],
         [ k_θ1z1,    k_θ1θ1,  k_θ1z2,  k_θ1θ2],
         [k_z2z1,     k_z2θ1,  k_z2z2,  k_z2θ2],
         [ k_θ2z1,    k_θ2θ1,  k_θ2z2,  k_θ2θ2]
    ])

    return Cae_star, Kae_star
def AD_single(poly_coeff, k_range, Vred_global,  B):
    
    Vred_global = float(Vred_global) 

    # AD
    # Evaluate aerodynamic damping derivatives
    H1 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True)
    H2 = from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True)
    A1 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True)
    A2 = from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True)

        
    # Evaluate aerodynamic stiffness derivatives
    H3 = from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=False)
    H4 = from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=False)
    A3 = from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=False)
    A4 = from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=False)

    Cae_star = np.array([
         [H1,       H2],
         [ A1,    A2]
     ])
    Kae_star = np.array([
         [H4,       H3],
         [ A4,   A3]
     ])        
    return Cae_star, Kae_star



from scipy.interpolate import interp1d
#%%
# Single 
v = np.linspace(0, 4, 100)

i = 0
AD_single_damping = np.zeros((100, 2, 2))
AD_single_stiffness = np.zeros((100, 2, 2))
for vi in v:
    AD_single_damping[i], AD_single_stiffness[i] = AD_single(poly_coeff_single,k_range_single,  vi, B)
    i += 1

Vcr = 84.724853515625 # m/s, critical wind speed
omega_cr = 1.452509128159339 # rad/s, critical frequency
K = omega_cr * B / Vcr

y_vals_c2 = 1/v * AD_single_damping[:,1,1]
f_interp2 = interp1d(1/v, y_vals_c2, kind='linear', fill_value="extrapolate")
y_K_c2 = f_interp2(K)

y_vals_c1 = 1/v * AD_single_damping[:,1,0]
f_interp1 = interp1d(1/v, y_vals_c1, kind='linear', fill_value="extrapolate")
y_K_c1 = f_interp1(K)

y_vals_ka = (1/v)**2 * AD_single_stiffness[:,0,1]
f_interpa = interp1d(1/v, y_vals_ka, kind='linear', fill_value="extrapolate")
y_K_ka = f_interpa(K)

y_vals_kh = (1/v)**2 * AD_single_stiffness[:,1,1]
f_interph = interp1d(1/v, y_vals_kh, kind='linear', fill_value="extrapolate")
y_K_kh = f_interph(K)


# Single order: H_1*, H_2*, H_3*, H_4*, A_1*, A_2*, A_3*, A_4*

fig, axs = plt.subplots(2, 2, figsize=(8, 4))  
axs[0,0].plot(v, 1/v*AD_single_damping[:,1,0], label=r"Unsteady")
axs[0,0].plot(v, np.full_like(1/v, Single_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[0,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1, label="Kcr")
axs[0,0].plot(1/K, y_K_c1, 'ro', color='#d62728',  markersize=4)
axs[0,0].plot(1/K, Single_c[1,1], 'ro', color='#d62728', markersize=4)
axs[0,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,0].set_ylabel(r"$K \cdot A_1^*$", fontsize=14)
axs[0,0].tick_params(labelsize=14)

axs[0,1].plot(v, 1/v*AD_single_damping[:,1,1])
axs[0,1].plot(v, np.full_like(1/v, Single_c[1,1]), alpha=0.8)
axs[0,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[0,1].plot(1/K, y_K_c2, 'ro', color='#d62728',  markersize=4)
axs[0,1].plot(1/K, Single_c[1,1], 'ro', color='#d62728',  markersize=4)
axs[0,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,1].set_ylabel(r"$K \cdot A_2^*$", fontsize=14)
axs[0,1].tick_params(labelsize=14)

axs[1,0].plot(v, (1/v)**2*AD_single_stiffness[:,0,1])
axs[1,0].plot(v, np.full_like(1/v, Single_k[0,1]), alpha=0.8)
axs[1,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,0].plot(1/K, y_K_ka, 'ro',color='#d62728',  markersize=4)
axs[1,0].plot(1/K, Single_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,0].set_ylabel(r"$K^2 \cdot H_3^*$", fontsize=14)
axs[1,0].tick_params(labelsize=14)

axs[1,1].plot(v, (1/v)**2*AD_single_stiffness[:,1,1])
axs[1,1].plot(v, np.full_like((1/v), Single_k[1,1]), alpha=0.8)
axs[1,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,1].plot(1/K, y_K_kh, 'ro',color='#d62728',  markersize=4)
axs[1,1].plot(1/K, Single_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,1].set_ylabel(r"$K^2 \cdot A_3^*$", fontsize=14)
axs[1,1].tick_params(labelsize=14)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,              
           fontsize=14,
           frameon=False,
           bbox_to_anchor=(0.5, -0.1))  
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "fitte_single.png"), dpi=300, bbox_inches='tight')


plt.show()



#%%
# 1D

# Tandem order: c_z1z1*, c_z1θ1*, c_z1z2*, c_z1θ2*, c_θ1z1*, c_θ1θ1*, c_θ1z2*, c_θ1θ2*, 
#               c_z2z1*, c_z2θ1*, c_z2z2*, c_z2θ2*, c_θ2z1*, c_θ2θ1*, c_θ2z2*, c_θ2θ2*, 
#               k_z1z1*, k_z1θ1*, k_z1z2*, k_z1θ2*, k_θ1z1*, k_θ1θ1*, k_θ1z2*, k_θ1θ2*, 
#               k_z2z1*, k_z2θ1*, k_z2z2*, k_z2θ2*, k_θ2z1*, k_θ2θ1*, k_θ2z2*, k_θ2θ2*
 
Vcr = 31.890869140625# m/s, critical wind speed
omega_cr = 2.154673716266362 # rad/s, critical frequency
K = omega_cr * B / Vcr

v = np.linspace(0, 4, 100)


i = 0
AD_1D_damping = np.zeros((100, 4, 4))
AD_1D_stiffness = np.zeros((100, 4, 4))
for vi in v:
    AD_1D_damping[i], AD_1D_stiffness[i] = AD_two(poly_coeff_1D,k_range_1D,  vi, B)
    i += 1

y_vals_c2 = 1/v * AD_1D_damping[:,1,1]
f_interp2 = interp1d(1/v, y_vals_c2, kind='linear', fill_value="extrapolate")
y_K_c2 = f_interp2(K)

y_vals_c1 = 1/v * AD_1D_damping[:,1,0]
f_interp1 = interp1d(1/v, y_vals_c1, kind='linear', fill_value="extrapolate")
y_K_c1 = f_interp1(K)

y_vals_c3 = 1/v * AD_1D_damping[:,3,3]
f_interp3 = interp1d(1/v, y_vals_c3, kind='linear', fill_value="extrapolate")
y_K_c3 = f_interp3(K)

y_vals_c4 = 1/v * AD_1D_damping[:,3,2]
f_interp4 = interp1d(1/v, y_vals_c4, kind='linear', fill_value="extrapolate")
y_K_c4 = f_interp4(K)

y_vals_ka = (1/v)**2 * AD_1D_stiffness[:,0,1]
f_interpa = interp1d(1/v, y_vals_ka, kind='linear', fill_value="extrapolate")
y_K_ka = f_interpa(K)

y_vals_kh = (1/v)**2 * AD_1D_stiffness[:,1,1]
f_interph = interp1d(1/v, y_vals_kh, kind='linear', fill_value="extrapolate")
y_K_kh = f_interph(K)

y_vals_ka1 = (1/v)**2 * AD_1D_stiffness[:,2,3]
f_interpa1 = interp1d(1/v, y_vals_ka1, kind='linear', fill_value="extrapolate")
y_K_ka1 = f_interpa1(K)

y_vals_kh1 = (1/v)**2 * AD_1D_stiffness[:,3,3]
f_interph1 = interp1d(1/v, y_vals_kh1, kind='linear', fill_value="extrapolate")
y_K_kh1 = f_interph1(K)

fig, axs = plt.subplots(4, 2, figsize=(8, 8))  
axs[0,0].plot(v, 1/v*AD_1D_damping[:,1,0], label=r"Unsteady")
axs[0,0].plot(v, np.full_like(1/v, Static_1D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[0,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1, label="Vcr")
axs[0,0].plot(1/K, y_K_c1, 'ro',color='#d62728',  markersize=4)
axs[0,0].plot(1/K, Static_1D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[0,0].set_xlabel(r"$V_{red}$", fontsize=14
)
axs[0,0].set_ylabel(r"$K \cdot c_{\theta 1 z1}^*$", fontsize=14
)
axs[0,0].tick_params(labelsize=14)

axs[1,0].plot(v, 1/v*AD_1D_damping[:,1,1])
axs[1,0].plot(v, np.full_like(1/v, Static_1D_c[1,1]), alpha=0.8)
axs[1,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,0].plot(1/K, y_K_c2, 'ro',color='#d62728',  markersize=4)
axs[1,0].plot(1/K, Static_1D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[1,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,0].set_ylabel(r"$K \cdot c_{\theta 1\theta 1}^*$", fontsize=14)
axs[1,0].tick_params(labelsize=14)

axs[2,0].plot(v, 1/v*AD_1D_damping[:,3,2], label=r"Unsteady")
axs[2,0].plot(v, np.full_like(1/v, Static_1D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[2,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,0].plot(1/K, y_K_c4, 'ro',color='#d62728',  markersize=4)
axs[2,0].plot(1/K, Static_1D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[2,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,0].set_ylabel(r"$K \cdot c_{\theta 2 z2}^*$", fontsize=14)
axs[2,0].tick_params(labelsize=14)

axs[3,0].plot(v, 1/v*AD_1D_damping[:,3,3])
axs[3,0].plot(v, np.full_like(1/v, Static_1D_c[1,1]), alpha=0.8)
axs[3,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,0].plot(1/K, y_K_c3, 'ro',color='#d62728',  markersize=4)
axs[3,0].plot(1/K, Static_1D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[3,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,0].set_ylabel(r"$K \cdot c_{\theta 2\theta 2}^*$", fontsize=14)
axs[3,0].tick_params(labelsize=14)

axs[0,1].plot(v, (1/v)**2*AD_1D_stiffness[:,0,1])
axs[0,1].plot(v, np.full_like(1/v, Static_1D_k[0,1]), alpha=0.8)
axs[0,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[0,1].plot(1/K, y_K_ka, 'ro',color='#d62728',  markersize=4)
axs[0,1].plot(1/K, Static_1D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[0,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,1].set_ylabel(r"$K^2 \cdot k_{z1 \theta 1}^*$", fontsize=14)
axs[0,1].tick_params(labelsize=14)

axs[1,1].plot(v, (1/v)**2*AD_1D_stiffness[:,1,1])
axs[1,1].plot(v, np.full_like((1/v), Static_1D_k[1,1]), alpha=0.8)
axs[1,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,1].plot(1/K, y_K_kh, 'ro',color='#d62728',  markersize=4)
axs[1,1].plot(1/K, Static_1D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,1].set_ylabel(r"$K^2 \cdot k_{\theta 1 \theta 1}^*$", fontsize=14)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(v, (1/v)**2*AD_1D_stiffness[:,2,3])
axs[2,1].plot(v, np.full_like(1/v, Static_1D_k[0,1]), alpha=0.8)
axs[2,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,1].plot(1/K, y_K_ka1, 'ro',color='#d62728',  markersize=4)
axs[2,1].plot(1/K, Static_1D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[2,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,1].set_ylabel(r"$K^2 \cdot k_{z2 \theta 2}^*$", fontsize=14)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(v, (1/v)**2*AD_1D_stiffness[:,3,3])
axs[3,1].plot(v, np.full_like((1/v), Static_1D_k[1,1]), alpha=0.8)
axs[3,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,1].plot(1/K, y_K_kh1, 'ro',color='#d62728',  markersize=4)
axs[3,1].plot(1/K, Static_1D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[3,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,1].set_ylabel(r"$K^2 \cdot k_{\theta 2 \theta 2}^*$", fontsize=14)
axs[3,1].tick_params(labelsize=14)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,              
           fontsize=14,
           frameon=False,
           bbox_to_anchor=(0.5, -0.05))  
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "fitte_1D.png"), dpi=300, bbox_inches='tight')

plt.show()

#%%
# 2D
Vcr = 40.43505859375# m/s, critical wind speed
omega_cr = 2.0946407671740137  # rad/s, critical frequency
K = omega_cr * B / Vcr



v = np.linspace(0, 4, 100)


i = 0
AD_2D_damping = np.zeros((100, 4, 4))
AD_2D_stiffness = np.zeros((100, 4, 4))
for vi in v:
    AD_2D_damping[i], AD_2D_stiffness[i] = AD_two(poly_coeff_2D,k_range_2D,  vi, B)
    i += 1

y_vals_c2 = 1/v * AD_2D_damping[:,1,1]
f_interp2 = interp1d(1/v, y_vals_c2, kind='linear', fill_value="extrapolate")
y_K_c2 = f_interp2(K)

y_vals_c1 = 1/v * AD_2D_damping[:,1,0]
f_interp1 = interp1d(1/v, y_vals_c1, kind='linear', fill_value="extrapolate")
y_K_c1 = f_interp1(K)

y_vals_c3 = 1/v * AD_2D_damping[:,3,3]
f_interp3 = interp1d(1/v, y_vals_c3, kind='linear', fill_value="extrapolate")
y_K_c3 = f_interp3(K)

y_vals_c4 = 1/v * AD_2D_damping[:,3,2]
f_interp4 = interp1d(1/v, y_vals_c4, kind='linear', fill_value="extrapolate")
y_K_c4 = f_interp4(K)

y_vals_ka = (1/v)**2 * AD_2D_stiffness[:,0,1]
f_interpa = interp1d(1/v, y_vals_ka, kind='linear', fill_value="extrapolate")
y_K_ka = f_interpa(K)

y_vals_kh = (1/v)**2 * AD_2D_stiffness[:,1,1]
f_interph = interp1d(1/v, y_vals_kh, kind='linear', fill_value="extrapolate")
y_K_kh = f_interph(K)

y_vals_ka1 = (1/v)**2 * AD_2D_stiffness[:,2,3]
f_interpa1 = interp1d(1/v, y_vals_ka1, kind='linear', fill_value="extrapolate")
y_K_ka1 = f_interpa1(K)

y_vals_kh1 = (1/v)**2 * AD_2D_stiffness[:,3,3]
f_interph1 = interp1d(1/v, y_vals_kh1, kind='linear', fill_value="extrapolate")
y_K_kh1 = f_interph1(K)

fig, axs = plt.subplots(4, 2, figsize=(8, 8))  
axs[0,0].plot(v, 1/v*AD_2D_damping[:,1,0], label=r"Unsteady")
axs[0,0].plot(v, np.full_like(1/v, Static_2D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[0,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1, label="Vcr")
axs[0,0].plot(1/K, y_K_c1, 'ro',color='#d62728',  markersize=4)
axs[0,0].plot(1/K, Static_2D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[0,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,0].set_ylabel(r"$K \cdot c_{\theta 1 z1}^*$", fontsize=14)
axs[0,0].tick_params(labelsize=14)

axs[1,0].plot(v, 1/v*AD_2D_damping[:,1,1])
axs[1,0].plot(v, np.full_like(1/v, Static_2D_c[1,1]), alpha=0.8)
axs[1,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,0].plot(1/K, y_K_c2, 'ro',color='#d62728',  markersize=4)
axs[1,0].plot(1/K, Static_2D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[1,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,0].set_ylabel(r"$K \cdot c_{\theta 1\theta 1}^*$", fontsize=14)
axs[1,0].tick_params(labelsize=14)

axs[2,0].plot(v, 1/v*AD_2D_damping[:,3,2], label=r"Unsteady")
axs[2,0].plot(v, np.full_like(1/v, Static_2D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[2,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,0].plot(1/K, y_K_c4, 'ro',color='#d62728',  markersize=4)
axs[2,0].plot(1/K, Static_2D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[2,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,0].set_ylabel(r"$K \cdot c_{\theta 2 z2}^*$", fontsize=14)
axs[2,0].tick_params(labelsize=14)

axs[3,0].plot(v, 1/v*AD_2D_damping[:,3,3])
axs[3,0].plot(v, np.full_like(1/v, Static_2D_c[1,1]), alpha=0.8)
axs[3,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,0].plot(1/K, y_K_c3, 'ro',color='#d62728',  markersize=4)
axs[3,0].plot(1/K, Static_2D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[3,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,0].set_ylabel(r"$K \cdot c_{\theta 2\theta 2}^*$", fontsize=14)
axs[3,0].tick_params(labelsize=14)

axs[0,1].plot(v, (1/v)**2*AD_2D_stiffness[:,0,1])
axs[0,1].plot(v, np.full_like(1/v, Static_2D_k[0,1]), alpha=0.8)
axs[0,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[0,1].plot(1/K, y_K_ka, 'ro',color='#d62728',  markersize=4)
axs[0,1].plot(1/K, Static_2D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[0,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,1].set_ylabel(r"$K^2 \cdot k_{z1 \theta 1}^*$", fontsize=14)
axs[0,1].tick_params(labelsize=14)

axs[1,1].plot(v, (1/v)**2*AD_2D_stiffness[:,1,1])
axs[1,1].plot(v, np.full_like((1/v), Static_2D_k[1,1]), alpha=0.8)
axs[1,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,1].plot(1/K, y_K_kh, 'ro',color='#d62728',  markersize=4)
axs[1,1].plot(1/K, Static_2D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,1].set_ylabel(r"$K^2 \cdot k_{\theta 1 \theta 1}^*$", fontsize=14)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(v, (1/v)**2*AD_2D_stiffness[:,2,3])
axs[2,1].plot(v, np.full_like(1/v, Static_2D_k[0,1]), alpha=0.8)
axs[2,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,1].plot(1/K, y_K_ka1, 'ro',color='#d62728',  markersize=4)
axs[2,1].plot(1/K, Static_2D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[2,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,1].set_ylabel(r"$K^2 \cdot k_{z2 \theta 2}^*$", fontsize=14)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(v, (1/v)**2*AD_2D_stiffness[:,3,3])
axs[3,1].plot(v, np.full_like((1/v), Static_2D_k[1,1]), alpha=0.8)
axs[3,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,1].plot(1/K, y_K_kh1, 'ro',color='#d62728',  markersize=4)
axs[3,1].plot(1/K, Static_2D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[3,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,1].set_ylabel(r"$K^2 \cdot k_{\theta 2 \theta 2}^*$", fontsize=14)
axs[3,1].tick_params(labelsize=14)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,              
           fontsize=14,
           frameon=False,
           bbox_to_anchor=(0.5, -0.05))  
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "fitte_2D.png"), dpi=300, bbox_inches='tight')

plt.show()

#%%
# 3D

Vcr = 53.93597412109375# m/s, critical wind speed
omega_cr = 1.9602677552302152 # rad/s, critical frequency
K = omega_cr * B / Vcr

v = np.linspace(0, 4, 100)


i = 0
AD_3D_damping = np.zeros((100, 4, 4))
AD_3D_stiffness = np.zeros((100, 4, 4))
for vi in v:
    AD_3D_damping[i], AD_3D_stiffness[i] = AD_two(poly_coeff_3D,k_range_3D,  vi, B)
    i += 1

y_vals_c2 = 1/v * AD_3D_damping[:,1,1]
f_interp2 = interp1d(1/v, y_vals_c2, kind='linear', fill_value="extrapolate")
y_K_c2 = f_interp2(K)

y_vals_c1 = 1/v * AD_3D_damping[:,1,0]
f_interp1 = interp1d(1/v, y_vals_c1, kind='linear', fill_value="extrapolate")
y_K_c1 = f_interp1(K)

y_vals_c3 = 1/v * AD_3D_damping[:,3,3]
f_interp3 = interp1d(1/v, y_vals_c3, kind='linear', fill_value="extrapolate")
y_K_c3 = f_interp3(K)

y_vals_c4 = 1/v * AD_3D_damping[:,3,2]
f_interp4 = interp1d(1/v, y_vals_c4, kind='linear', fill_value="extrapolate")
y_K_c4 = f_interp4(K)

y_vals_ka = (1/v)**2 * AD_3D_stiffness[:,0,1]
f_interpa = interp1d(1/v, y_vals_ka, kind='linear', fill_value="extrapolate")
y_K_ka = f_interpa(K)

y_vals_kh = (1/v)**2 * AD_3D_stiffness[:,1,1]
f_interph = interp1d(1/v, y_vals_kh, kind='linear', fill_value="extrapolate")
y_K_kh = f_interph(K)

y_vals_ka1 = (1/v)**2 * AD_3D_stiffness[:,2,3]
f_interpa1 = interp1d(1/v, y_vals_ka1, kind='linear', fill_value="extrapolate")
y_K_ka1 = f_interpa1(K)

y_vals_kh1 = (1/v)**2 * AD_3D_stiffness[:,3,3]
f_interph1 = interp1d(1/v, y_vals_kh1, kind='linear', fill_value="extrapolate")
y_K_kh1 = f_interph1(K)

fig, axs = plt.subplots(4, 2, figsize=(8, 8))  
axs[0,0].plot(v, 1/v*AD_3D_damping[:,1,0], label=r"Unsteady")
axs[0,0].plot(v, np.full_like(1/v, Static_3D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[0,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1, label="Vcr")
axs[0,0].plot(1/K, y_K_c1, 'ro',color='#d62728',  markersize=4)
axs[0,0].plot(1/K, Static_3D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[0,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,0].set_ylabel(r"$K \cdot c_{\theta 1 z1}^*$", fontsize=14)
axs[0,0].tick_params(labelsize=14)

axs[1,0].plot(v, 1/v*AD_3D_damping[:,1,1])
axs[1,0].plot(v, np.full_like(1/v, Static_3D_c[1,1]), alpha=0.8)
axs[1,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,0].plot(1/K, y_K_c2, 'ro',color='#d62728',  markersize=4)
axs[1,0].plot(1/K, Static_3D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[1,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,0].set_ylabel(r"$K \cdot c_{\theta 1\theta 1}^*$", fontsize=14)
axs[1,0].tick_params(labelsize=14)

axs[2,0].plot(v, 1/v*AD_3D_damping[:,3,2], label=r"Unsteady")
axs[2,0].plot(v, np.full_like(1/v, Static_3D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[2,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,0].plot(1/K, y_K_c4, 'ro',color='#d62728',  markersize=4)
axs[2,0].plot(1/K, Static_3D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[2,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,0].set_ylabel(r"$K \cdot c_{\theta 2 z2}^*$", fontsize=14)
axs[2,0].tick_params(labelsize=14)

axs[3,0].plot(v, 1/v*AD_3D_damping[:,3,3])
axs[3,0].plot(v, np.full_like(1/v, Static_3D_c[1,1]), alpha=0.8)
axs[3,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,0].plot(1/K, y_K_c3, 'ro',color='#d62728',  markersize=4)
axs[3,0].plot(1/K, Static_3D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[3,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,0].set_ylabel(r"$K \cdot c_{\theta 2\theta 2}^*$", fontsize=14)
axs[3,0].tick_params(labelsize=14)

axs[0,1].plot(v, (1/v)**2*AD_3D_stiffness[:,0,1])
axs[0,1].plot(v, np.full_like(1/v, Static_3D_k[0,1]), alpha=0.8)
axs[0,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[0,1].plot(1/K, y_K_ka, 'ro',color='#d62728',  markersize=4)
axs[0,1].plot(1/K, Static_3D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[0,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,1].set_ylabel(r"$K^2 \cdot k_{z1 \theta 1}^*$", fontsize=14)
axs[0,1].tick_params(labelsize=14)

axs[1,1].plot(v, (1/v)**2*AD_3D_stiffness[:,1,1])
axs[1,1].plot(v, np.full_like((1/v), Static_3D_k[1,1]), alpha=0.8)
axs[1,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,1].plot(1/K, y_K_kh, 'ro',color='#d62728',  markersize=4)
axs[1,1].plot(1/K, Static_3D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,1].set_ylabel(r"$K^2 \cdot k_{\theta 1 \theta 1}^*$", fontsize=14)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(v, (1/v)**2*AD_3D_stiffness[:,2,3])
axs[2,1].plot(v, np.full_like(1/v, Static_3D_k[0,1]), alpha=0.8)
axs[2,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,1].plot(1/K, y_K_ka1, 'ro',color='#d62728',  markersize=4)
axs[2,1].plot(1/K, Static_3D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[2,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,1].set_ylabel(r"$K^2 \cdot k_{z2 \theta 2}^*$", fontsize=14)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(v, (1/v)**2*AD_3D_stiffness[:,3,3])
axs[3,1].plot(v, np.full_like((1/v), Static_3D_k[1,1]), alpha=0.8)
axs[3,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,1].plot(1/K, y_K_kh1, 'ro',color='#d62728',  markersize=4)
axs[3,1].plot(1/K, Static_3D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[3,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,1].set_ylabel(r"$K^2 \cdot k_{\theta 2 \theta 2}^*$", fontsize=14)
axs[3,1].tick_params(labelsize=14)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,              
           fontsize=14,
           frameon=False,
           bbox_to_anchor=(0.5, -0.05))  
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "fitte_3D.png"), dpi=300, bbox_inches='tight')

plt.show()


#%%
# 4D


Vcr = 55.8602294921875# m/s, critical wind speed
omega_cr = 1.9493400472910092 # rad/s, critical frequency
K = omega_cr * B / Vcr

v = np.linspace(0, 4, 100)


i = 0
AD_4D_damping = np.zeros((100, 4, 4))
AD_4D_stiffness = np.zeros((100, 4, 4))
for vi in v:
    AD_4D_damping[i], AD_4D_stiffness[i] = AD_two(poly_coeff_4D,k_range_4D,  vi, B)
    i += 1

y_vals_c2 = 1/v * AD_4D_damping[:,1,1]
f_interp2 = interp1d(1/v, y_vals_c2, kind='linear', fill_value="extrapolate")
y_K_c2 = f_interp2(K)

y_vals_c1 = 1/v * AD_4D_damping[:,1,0]
f_interp1 = interp1d(1/v, y_vals_c1, kind='linear', fill_value="extrapolate")
y_K_c1 = f_interp1(K)

y_vals_c3 = 1/v * AD_4D_damping[:,3,3]
f_interp3 = interp1d(1/v, y_vals_c3, kind='linear', fill_value="extrapolate")
y_K_c3 = f_interp3(K)

y_vals_c4 = 1/v * AD_4D_damping[:,3,2]
f_interp4 = interp1d(1/v, y_vals_c4, kind='linear', fill_value="extrapolate")
y_K_c4 = f_interp4(K)

y_vals_ka = (1/v)**2 * AD_4D_stiffness[:,0,1]
f_interpa = interp1d(1/v, y_vals_ka, kind='linear', fill_value="extrapolate")
y_K_ka = f_interpa(K)

y_vals_kh = (1/v)**2 * AD_4D_stiffness[:,1,1]
f_interph = interp1d(1/v, y_vals_kh, kind='linear', fill_value="extrapolate")
y_K_kh = f_interph(K)

y_vals_ka1 = (1/v)**2 * AD_4D_stiffness[:,2,3]
f_interpa1 = interp1d(1/v, y_vals_ka1, kind='linear', fill_value="extrapolate")
y_K_ka1 = f_interpa1(K)

y_vals_kh1 = (1/v)**2 * AD_4D_stiffness[:,3,3]
f_interph1 = interp1d(1/v, y_vals_kh1, kind='linear', fill_value="extrapolate")
y_K_kh1 = f_interph1(K)

fig, axs = plt.subplots(4, 2, figsize=(8, 8))  
axs[0,0].plot(v, 1/v*AD_4D_damping[:,1,0], label=r"Unsteady")
axs[0,0].plot(v, np.full_like(1/v, Static_4D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[0,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1, label="Vcr")
axs[0,0].plot(1/K, y_K_c1, 'ro',color='#d62728',  markersize=4)
axs[0,0].plot(1/K, Static_4D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[0,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,0].set_ylabel(r"$K \cdot c_{\theta 1 z1}^*$", fontsize=14)
axs[0,0].tick_params(labelsize=14)

axs[1,0].plot(v, 1/v*AD_4D_damping[:,1,1])
axs[1,0].plot(v, np.full_like(1/v, Static_4D_c[1,1]), alpha=0.8)
axs[1,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,0].plot(1/K, y_K_c2, 'ro',color='#d62728',  markersize=4)
axs[1,0].plot(1/K, Static_4D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[1,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,0].set_ylabel(r"$K \cdot c_{\theta 1\theta 1}^*$", fontsize=14)
axs[1,0].tick_params(labelsize=14)

axs[2,0].plot(v, 1/v*AD_4D_damping[:,3,2], label=r"Unsteady")
axs[2,0].plot(v, np.full_like(1/v, Static_4D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[2,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,0].plot(1/K, y_K_c4, 'ro',color='#d62728',  markersize=4)
axs[2,0].plot(1/K, Static_4D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[2,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,0].set_ylabel(r"$K \cdot c_{\theta 2 z2}^*$", fontsize=14)
axs[2,0].tick_params(labelsize=14)

axs[3,0].plot(v, 1/v*AD_4D_damping[:,3,3])
axs[3,0].plot(v, np.full_like(1/v, Static_4D_c[1,1]), alpha=0.8)
axs[3,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,0].plot(1/K, y_K_c3, 'ro',color='#d62728',  markersize=4)
axs[3,0].plot(1/K, Static_4D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[3,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,0].set_ylabel(r"$K \cdot c_{\theta 2\theta 2}^*$", fontsize=14)
axs[3,0].tick_params(labelsize=14)

axs[0,1].plot(v, (1/v)**2*AD_4D_stiffness[:,0,1])
axs[0,1].plot(v, np.full_like(1/v, Static_4D_k[0,1]), alpha=0.8)
axs[0,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[0,1].plot(1/K, y_K_ka, 'ro',color='#d62728',  markersize=4)
axs[0,1].plot(1/K, Static_4D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[0,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,1].set_ylabel(r"$K^2 \cdot k_{z1 \theta 1}^*$", fontsize=14)
axs[0,1].tick_params(labelsize=14)

axs[1,1].plot(v, (1/v)**2*AD_4D_stiffness[:,1,1])
axs[1,1].plot(v, np.full_like((1/v), Static_4D_k[1,1]), alpha=0.8)
axs[1,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,1].plot(1/K, y_K_kh, 'ro',color='#d62728',  markersize=4)
axs[1,1].plot(1/K, Static_4D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,1].set_ylabel(r"$K^2 \cdot k_{\theta 1 \theta 1}^*$", fontsize=14)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(v, (1/v)**2*AD_4D_stiffness[:,2,3])
axs[2,1].plot(v, np.full_like(1/v, Static_4D_k[0,1]), alpha=0.8)
axs[2,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,1].plot(1/K, y_K_ka1, 'ro',color='#d62728',  markersize=4)
axs[2,1].plot(1/K, Static_4D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[2,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,1].set_ylabel(r"$K^2 \cdot k_{z2 \theta 2}^*$", fontsize=14)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(v, (1/v)**2*AD_4D_stiffness[:,3,3])
axs[3,1].plot(v, np.full_like((1/v), Static_4D_k[1,1]), alpha=0.8)
axs[3,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,1].plot(1/K, y_K_kh1, 'ro',color='#d62728',  markersize=4)
axs[3,1].plot(1/K, Static_4D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[3,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,1].set_ylabel(r"$K^2 \cdot k_{\theta 2 \theta 2}^*$", fontsize=14)
axs[3,1].tick_params(labelsize=14)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,              
           fontsize=14,
           frameon=False,
           bbox_to_anchor=(0.5, -0.05))  
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "fitte_4D.png"), dpi=300, bbox_inches='tight')

plt.show()


#%%
# 5D

Vcr = 62.259796142578125# m/s, critical wind speed
omega_cr = 1.7808098966420758 # rad/s, critical frequency
K = omega_cr * B / Vcr

v = np.linspace(0, 4, 100)


i = 0
AD_5D_damping = np.zeros((100, 4, 4))
AD_5D_stiffness = np.zeros((100, 4, 4))
for vi in v:
    AD_5D_damping[i], AD_5D_stiffness[i] = AD_two(poly_coeff_5D,k_range_5D,  vi, B)
    i += 1

y_vals_c2 = 1/v * AD_5D_damping[:,1,1]
f_interp2 = interp1d(1/v, y_vals_c2, kind='linear', fill_value="extrapolate")
y_K_c2 = f_interp2(K)

y_vals_c1 = 1/v * AD_5D_damping[:,1,0]
f_interp1 = interp1d(1/v, y_vals_c1, kind='linear', fill_value="extrapolate")
y_K_c1 = f_interp1(K)

y_vals_c3 = 1/v * AD_5D_damping[:,3,3]
f_interp3 = interp1d(1/v, y_vals_c3, kind='linear', fill_value="extrapolate")
y_K_c3 = f_interp3(K)

y_vals_c4 = 1/v * AD_5D_damping[:,3,2]
f_interp4 = interp1d(1/v, y_vals_c4, kind='linear', fill_value="extrapolate")
y_K_c4 = f_interp4(K)

y_vals_ka = (1/v)**2 * AD_5D_stiffness[:,0,1]
f_interpa = interp1d(1/v, y_vals_ka, kind='linear', fill_value="extrapolate")
y_K_ka = f_interpa(K)

y_vals_kh = (1/v)**2 * AD_5D_stiffness[:,1,1]
f_interph = interp1d(1/v, y_vals_kh, kind='linear', fill_value="extrapolate")
y_K_kh = f_interph(K)

y_vals_ka1 = (1/v)**2 * AD_5D_stiffness[:,2,3]
f_interpa1 = interp1d(1/v, y_vals_ka1, kind='linear', fill_value="extrapolate")
y_K_ka1 = f_interpa1(K)

y_vals_kh1 = (1/v)**2 * AD_5D_stiffness[:,3,3]
f_interph1 = interp1d(1/v, y_vals_kh1, kind='linear', fill_value="extrapolate")
y_K_kh1 = f_interph1(K)

fig, axs = plt.subplots(4, 2, figsize=(8, 8))  
axs[0,0].plot(v, 1/v*AD_5D_damping[:,1,0], label=r"Unsteady")
axs[0,0].plot(v, np.full_like(1/v, Static_5D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[0,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1, label="Vcr")
axs[0,0].plot(1/K, y_K_c1, 'ro',color='#d62728',  markersize=4)
axs[0,0].plot(1/K, Static_5D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[0,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,0].set_ylabel(r"$K \cdot c_{\theta 1 z1}^*$", fontsize=14)
axs[0,0].tick_params(labelsize=14)

axs[1,0].plot(v, 1/v*AD_5D_damping[:,1,1])
axs[1,0].plot(v, np.full_like(1/v, Static_5D_c[1,1]), alpha=0.8)
axs[1,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,0].plot(1/K, y_K_c2, 'ro',color='#d62728',  markersize=4)
axs[1,0].plot(1/K, Static_5D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[1,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,0].set_ylabel(r"$K \cdot c_{\theta 1\theta 1}^*$", fontsize=14)
axs[1,0].tick_params(labelsize=14)

axs[2,0].plot(v, 1/v*AD_5D_damping[:,3,2], label=r"Unsteady")
axs[2,0].plot(v, np.full_like(1/v, Static_5D_c[1,0]),  label=r"Quasi-static", alpha=0.8)
axs[2,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,0].plot(1/K, y_K_c4, 'ro',color='#d62728',  markersize=4)
axs[2,0].plot(1/K, Static_5D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[2,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,0].set_ylabel(r"$K \cdot c_{\theta 2 z2}^*$", fontsize=14)
axs[2,0].tick_params(labelsize=14)

axs[3,0].plot(v, 1/v*AD_5D_damping[:,3,3])
axs[3,0].plot(v, np.full_like(1/v, Static_5D_c[1,1]), alpha=0.8)
axs[3,0].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,0].plot(1/K, y_K_c3, 'ro',color='#d62728',  markersize=4)
axs[3,0].plot(1/K, Static_5D_c[1,1], 'ro',color='#d62728',  markersize=4)
axs[3,0].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,0].set_ylabel(r"$K \cdot c_{\theta 2\theta 2}^*$", fontsize=14)
axs[3,0].tick_params(labelsize=14)

axs[0,1].plot(v, (1/v)**2*AD_5D_stiffness[:,0,1])
axs[0,1].plot(v, np.full_like(1/v, Static_5D_k[0,1]), alpha=0.8)
axs[0,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[0,1].plot(1/K, y_K_ka, 'ro',color='#d62728',  markersize=4)
axs[0,1].plot(1/K, Static_5D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[0,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[0,1].set_ylabel(r"$K^2 \cdot k_{z1 \theta 1}^*$", fontsize=14)
axs[0,1].tick_params(labelsize=14)

axs[1,1].plot(v, (1/v)**2*AD_5D_stiffness[:,1,1])
axs[1,1].plot(v, np.full_like((1/v), Static_5D_k[1,1]), alpha=0.8)
axs[1,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[1,1].plot(1/K, y_K_kh, 'ro',color='#d62728',  markersize=4)
axs[1,1].plot(1/K, Static_5D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[1,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[1,1].set_ylabel(r"$K^2 \cdot k_{\theta 1 \theta 1}^*$", fontsize=14)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(v, (1/v)**2*AD_5D_stiffness[:,2,3])
axs[2,1].plot(v, np.full_like(1/v, Static_5D_k[0,1]), alpha=0.8)
axs[2,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[2,1].plot(1/K, y_K_ka1, 'ro',color='#d62728',  markersize=4)
axs[2,1].plot(1/K, Static_5D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[2,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[2,1].set_ylabel(r"$K^2 \cdot k_{z2 \theta 2}^*$", fontsize=14)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(v, (1/v)**2*AD_5D_stiffness[:,3,3])
axs[3,1].plot(v, np.full_like((1/v), Static_5D_k[1,1]), alpha=0.8)
axs[3,1].axvline(x=1/K, color='black', linestyle='--', linewidth=1)
axs[3,1].plot(1/K, y_K_kh1, 'ro',color='#d62728',  markersize=4)
axs[3,1].plot(1/K, Static_5D_k[0,0], 'ro',color='#d62728',  markersize=4)
axs[3,1].set_xlabel(r"$V_{red}$", fontsize=14)
axs[3,1].set_ylabel(r"$K^2 \cdot k_{\theta 2 \theta 2}^*$", fontsize=14)
axs[3,1].tick_params(labelsize=14)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,              
           fontsize=14,
           frameon=False,
           bbox_to_anchor=(0.5, -0.05))  
plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "fitte_5D.png"), dpi=300, bbox_inches='tight')

plt.show()


######################################################################################
######################################################################################
######################################################################################
#%%
def plot_flutter_mode_shape(eigvec_flutter, Phi, x, labels=["V1", "T1", "V2", "T2"]):
    """
    Plots the real, imaginary, and absolute parts of the physical flutter mode shape.

    Parameters:
    -----------
    eigvec_flutter : np.ndarray, shape (n_modes,)
        Eigenvector at flutter (complex modal amplitudes).
    Phi : np.ndarray, shape (n_points, n_dof, n_modes)
        Mode shapes along the span.
    x : np.ndarray, shape (n_points,)
        Spanwise locations (normalized or in meters).
    labels : list of str
        DOF labels for the components (used for legend).
    """
    u_real = []
    u_imag = []
    u_abs = []

    for i in range(len(x)):
        u_complex = Phi[i] @ eigvec_flutter  # shape: (n_dof,)
        u_real.append(np.real(u_complex))
        u_imag.append(np.imag(u_complex))
        u_abs.append(np.abs(u_complex))

    u_real = np.array(u_real)
    u_imag = np.array(u_imag)
    u_abs = np.array(u_abs)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for i, label in enumerate(labels):
        axs[0].plot(x, u_real[:, i], label=label)
        axs[1].plot(x, u_imag[:, i], label=label)
        axs[2].plot(x, u_abs[:, i], label=label)


    axs[0].set_ylabel("Real")
    axs[1].set_ylabel("Imag")
    axs[2].set_ylabel("Abs")
    axs[2].set_xlabel("Fraction of the span length [x/L]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

last_damping = np.array(damping_list_two_1D[-1])  
idx_mode_flutter = np.nanargmin(last_damping)
flutter_index = np.argmin(np.abs(np.array(V_list_two_1D) - Vcritical_two_1D))
print(flutter_index)  # indeks der flutter skjer
eigvec_flutter = eigvecs_list_two_1D[flutter_index, idx_mode_flutter]  # f.eks. mode med negativ demping
plot_flutter_mode_shape(eigvec_flutter, phi_two, x_two, labels=["V1", "T1", "V2", "T2"])
