 
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



B = 18.3 # m, section width 


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

zeta = 0.005 # 5 %, critical damping
rho = 1.25 # kg/m^3, air density 

#ITERATIVE BIMODAL EIGENVALUE APPROACH
eps = 0.0001 # Konvergensterske

# Single
phi_single, x_single = mode_shape_single()

Ms_single, Cs_single, Ks_single = _eigVal.structural_matrices(m1V, m1T, f1V, f1T, zeta, single = True)

# two

phi_two, x_two = mode_shape_two()
print("phi_two shape: ", phi_two.shape)
print("phi_two[0]: ", phi_two[0])
Ms_two, Cs_two, Ks_two = _eigVal.structural_matrices(m1V, m1T, f1V, f1T, zeta, single = False)

#  BUFFETING
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Buffeting\Cae_Kae.npy"
# Load the saved dictionary
matrices = np.load(file_path, allow_pickle=True).item()

Kae_Single = matrices["Kae_Single"]
Cae_Single = matrices["Cae_Single"]
Cae_Single_gen, Kae_Single_gen = _eigVal.generalize_C_K(Cae_Single, Kae_Single, phi_single, x_single)

Kae_1D = matrices["Kae_1D"]
Cae_1D = matrices["Cae_1D"]
Cae_1D_gen, Kae_1D_gen = _eigVal.generalize_C_K(Cae_1D, Kae_1D, phi_two, x_two, single=False)
print("Kae_1D_gen", 37.6*37.6*Kae_1D_gen)

Kae_2D = matrices["Kae_2D"]
Cae_2D = matrices["Cae_2D"]
Cae_2D_gen, Kae_2D_gen = _eigVal.generalize_C_K(Cae_2D, Kae_2D, phi_two, x_two, single=False)

Kae_3D = matrices["Kae_3D"]
Cae_3D = matrices["Cae_3D"]
Cae_3D_gen, Kae_3D_gen = _eigVal.generalize_C_K(Cae_3D, Kae_3D, phi_two, x_two, single=False)

Kae_4D = matrices["Kae_4D"]
Cae_4D = matrices["Cae_4D"]
Cae_4D_gen, Kae_4D_gen = _eigVal.generalize_C_K(Cae_4D, Kae_4D, phi_two, x_two, single=False)

Kae_5D = matrices["Kae_5D"]
Cae_5D = matrices["Cae_5D"]
Cae_5D_gen, Kae_5D_gen = _eigVal.generalize_C_K(Cae_5D, Kae_5D, phi_two, x_two, single=False)

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




#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_omega(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, dist="Single deck",  single = True, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, dist="Single deck", single = True)

_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)

#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_omega(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, dist="two deck 1D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, dist="two deck 1D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)

#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_omega(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_2D, eigvecs_list_two_2D, V_list_two_2D, dist="two deck 2D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D, dist="two deck 2D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_omega(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D, V_list_two_3D, dist="two deck 3D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D, dist="two deck 3D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_omega(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_4D, eigvecs_list_two_4D, V_list_two_4D, dist="two deck 4D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D, dist="two deck 4D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_omega(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting
_eigVal.plot_damping_vs_wind_speed(damping_list_two_5D, eigvecs_list_two_5D, V_list_two_5D, dist="two deck 5D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D, dist="two deck 5D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)


# BUFFETING

#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_omega(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, buffeting = True, Cae_star_gen_BUFF=Cae_Single_gen, Kae_star_gen_BUFF=Kae_Single_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, dist="Single deck",  single = True, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, dist="Single deck", single = True)

_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)

#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_omega(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_1D_gen, Kae_star_gen_BUFF=Kae_1D_gen,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, dist="two deck 1D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, dist="two deck 1D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)

#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_omega(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,buffeting = True,  Cae_star_gen_BUFF=Cae_2D_gen, Kae_star_gen_BUFF=Kae_2D_gen,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_2D, eigvecs_list_two_2D, V_list_two_2D, dist="two deck 2D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D, dist="two deck 2D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_omega(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_3D_gen, Kae_star_gen_BUFF=Kae_3D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D, V_list_two_3D, dist="two deck 3D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D, dist="two deck 3D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_omega(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_4D_gen, Kae_star_gen_BUFF=Kae_4D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_4D, eigvecs_list_two_4D, V_list_two_4D, dist="two deck 4D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D, dist="two deck 4D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_omega(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_5D_gen, Kae_star_gen_BUFF=Kae_5D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting
_eigVal.plot_damping_vs_wind_speed(damping_list_two_5D, eigvecs_list_two_5D, V_list_two_5D, dist="two deck 5D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D, dist="two deck 5D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)

# %%
