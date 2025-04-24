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
from mode_shapes import mode_shape_single
from mode_shapes import mode_shape_twin


file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Arrays_AD"

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

zeta = 0.005 # 0.5 %, critical damping
rho = 1.25 # kg/m^3, air density ??


V_list = np.linspace(0, 100, 200) # m/s


#ITERATIVE BIMODAL EIGENVALUE APPROACH
eps = 1e-4 # Konvergensterske

# Single
x_single = mode_shape_single(full_matrix=True)[1]

Ms_single, Cs_single, Ks_single = _eigVal.structural_matrices(m1V, m1T, f1V, f1T, zeta, single = True)

Ms_single_massnorm, Cs_single_massnorm, Ks_single_massnorm = _eigVal.structural_matrices_massnorm(f1V, f1T, zeta, single=True)



# Twin

x_twin = mode_shape_twin(full_matrix=True)[1]

Ms_twin, Cs_twin, Ks_twin = _eigVal.structural_matrices(m1V, m1T, f1V, f1T, zeta, single = False)

Ms_twin_massnorm, Cs_twin_massnorm, Ks_twin_massnorm = _eigVal.structural_matrices_massnorm(f1V, f1T, zeta, single=False)


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



#Solve for eigenvalues and eigenvectors
damping_ratios_single, omega_all_single, eigvals_all_single, eigvecs_all_single = _eigVal.solve_omega(poly_coeff_single, Ms_single,Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, zeta,  eps, V_list,  x_single, single = True)


#Flutter
flutter_speed_modes_single, flutter_idx_modes_single =_eigVal.solve_flutter_speed( damping_ratios_single, V_list, single = True)

if flutter_speed_modes_single[0] is not None:
    print(f"Flutter speed mode 1 - single deck: {flutter_speed_modes_single[0]:.2f}")
else:
    print("Flutter speed mode 1 - single deck: Not found")

if flutter_speed_modes_single[1] is not None:   
    print(f"Flutter speed mode 2 - single deck: {flutter_speed_modes_single[1]:.2f}")
else:
    print("Flutter speed mode 2 - single deck: Not found")


#Plotting
_eigVal.plot_damping_vs_wind_speed_single(damping_ratios_single,V_list, dist="Single deck",  single = True)

_eigVal.plot_frequency_vs_wind_speed(B,  v_range_single, V_list, omega_all_single, dist="Single deck", single = True)

_eigVal.plot_flutter_mode_shape(eigvecs_all_single,flutter_idx_modes_single, dist="Single deck", single = True)

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



damping_ratios_1D, omega_all_1D, eigvals_all_1D, eigvecs_all_1D = _eigVal.solve_omega(poly_coeff_1D, Ms_twin, Ms_twin, Cs_twin, Ks_twin, f1V, f1T, B, rho, zeta,  eps, V_list,  x_twin, single = False)


flutter_speed_modes_1D, flutter_idx_modes_1D=_eigVal.solve_flutter_speed( damping_ratios_1D, V_list, single = False)

if flutter_speed_modes_1D[0] is not None:
    print(f"Flutter speed mode 1 - 1D: {flutter_speed_modes_1D[0]:.2f}")
else:
    print("Flutter speed mode 1 - 1D: Not found")

if flutter_speed_modes_1D[1] is not None:   
    print(f"Flutter speed mode 2 - 1D: {flutter_speed_modes_1D[1]:.2f}")
else:
    print("Flutter speed mode 2 - 1D: Not found")

if flutter_speed_modes_1D[2] is not None:   
    print(f"Flutter speed mode 3 - 1D: {flutter_speed_modes_1D[2]:.2f}")
else:
    print("Flutter speed mode 3 - 1D: Not found")

if flutter_speed_modes_1D[3] is not None:   
    print(f"Flutter speed mode 4 - 1D: {flutter_speed_modes_1D[3]:.2f}")
else:
    print("Flutter speed mode 4 - 1D: Not found")



_eigVal.plot_damping_vs_wind_speed_single(damping_ratios_1D,  V_list, dist="1D", single = False)

_eigVal.plot_frequency_vs_wind_speed(B, v_range_1D, V_list, omega_all=omega_all_1D,  dist="1D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_all_1D,flutter_idx_modes_1D, dist="1D", single = False)

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



damping_ratios_2D, omega_all_2D, eigvals_all_2D, eigvecs_all_2D = _eigVal.solve_omega(poly_coeff_2D, Ms, Cs, Ks,  f1V, f1T, B, rho, zeta,  eps, V_list, single = False)


flutter_speed_modes_2D, flutter_idx_modes_2D=_eigVal.solve_flutter_speed( damping_ratios_2D, V_list, single = False)

if flutter_speed_modes_2D[0] is not None:
    print(f"Flutter speed mode 1 - 2D: {flutter_speed_modes_2D[0]:.2f}")
else:
    print("Flutter speed mode 1 - 2D: Not found")

if flutter_speed_modes_2D[1] is not None:   
    print(f"Flutter speed mode 2 - 2D: {flutter_speed_modes_2D[1]:.2f}")
else:
    print("Flutter speed mode 2 - 2D: Not found")

if flutter_speed_modes_2D[2] is not None:   
    print(f"Flutter speed mode 3 - 2D: {flutter_speed_modes_2D[2]:.2f}")
else:
    print("Flutter speed mode 3 - 2D: Not found")

if flutter_speed_modes_2D[3] is not None:   
    print(f"Flutter speed mode 4 - 2D: {flutter_speed_modes_2D[3]:.2f}")
else:
    print("Flutter speed mode 4 - 2D: Not found")


_eigVal.plot_damping_vs_wind_speed_single(damping_ratios_2D,  V_list, dist="2D", single = False)

_eigVal.plot_frequency_vs_wind_speed(B, v_range_2D, V_list, omega_all=omega_all_2D, dist="2D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_all_2D,flutter_idx_modes_2D, dist="2D", single = False)


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



damping_ratios_3D, omega_all_3D, eigvals_all_3D, eigvecs_all_3D = _eigVal.solve_omega(poly_coeff_3D, Ms, Cs, Ks, f1V, f1T, B, rho, zeta,  eps, V_list, single = False)


flutter_speed_modes_3D, flutter_idx_modes_3D=_eigVal.solve_flutter_speed( damping_ratios_3D, V_list, single = False)

if flutter_speed_modes_3D[0] is not None:
    print(f"Flutter speed mode 1 - 3D: {flutter_speed_modes_3D[0]:.2f}")
else:
    print("Flutter speed mode 1 - 3D: Not found")

if flutter_speed_modes_3D[1] is not None:   
    print(f"Flutter speed mode 2 - 3D: {flutter_speed_modes_3D[1]:.2f}")
else:
    print("Flutter speed mode 2 - 3D: Not found")

if flutter_speed_modes_3D[2] is not None:   
    print(f"Flutter speed mode 3 - 3D: {flutter_speed_modes_3D[2]:.2f}")
else:
    print("Flutter speed mode 3 - 3D: Not found")

if flutter_speed_modes_3D[3] is not None:   
    print(f"Flutter speed mode 4 - 3D: {flutter_speed_modes_3D[3]:.2f}")
else:
    print("Flutter speed mode 4 - 3D: Not found")

_eigVal.plot_damping_vs_wind_speed_single(damping_ratios_3D, V_list,  dist="3D",single = False)
_eigVal.plot_frequency_vs_wind_speed(B, v_range_3D, V_list, omega_all=omega_all_3D,dist="3D",single = False)
_eigVal.plot_flutter_mode_shape(eigvecs_all_3D,flutter_idx_modes_3D, dist="3D", single = False)


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

damping_ratios_4D, omega_all_4D, eigvals_all_4D, eigvecs_all_4D = _eigVal.solve_omega(poly_coeff_4D, Ms, Cs, Ks,  f1V, f1T, B, rho, zeta,  eps, V_list, single = False)

flutter_speed_modes_4D, flutter_idx_modes_4D=_eigVal.solve_flutter_speed( damping_ratios_4D, V_list, single = False)

if flutter_speed_modes_4D[0] is not None:
    print(f"Flutter speed mode 1 - 4D: {flutter_speed_modes_4D[0]:.2f}")
else:
    print("Flutter speed mode 1 - 4D: Not found")

if flutter_speed_modes_4D[1] is not None:   
    print(f"Flutter speed mode 2 - 4D: {flutter_speed_modes_4D[1]:.2f}")
else:
    print("Flutter speed mode 2 - 4D: Not found")

if flutter_speed_modes_4D[2] is not None:   
    print(f"Flutter speed mode 3 - 4D: {flutter_speed_modes_4D[2]:.2f}")
else:
    print("Flutter speed mode 3 - 4D: Not found")

if flutter_speed_modes_4D[3] is not None:   
    print(f"Flutter speed mode 4 - 4D: {flutter_speed_modes_4D[3]:.2f}")
else:
    print("Flutter speed mode 4 - 4D: Not found")

_eigVal.plot_damping_vs_wind_speed_single(damping_ratios_4D,V_list,dist="4D", single = False)
_eigVal.plot_frequency_vs_wind_speed(B, v_range_4D, V_list,omega_all=omega_all_4D,   dist="4D", single = False)
_eigVal.plot_flutter_mode_shape(eigvecs_all_4D,flutter_idx_modes_4D, dist="4D", single = False)

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

damping_ratios_5D, omega_all_5D, eigvals_all_5D, eigvecs_all_5D = _eigVal.solve_omega(poly_coeff_5D, Ms, Cs, Ks, f1V, f1T, B, rho, zeta,  eps, V_list, single = False)


flutter_speed_modes_5D, flutter_idx_modes_5D=_eigVal.solve_flutter_speed( damping_ratios_5D, V_list, single = False)

if flutter_speed_modes_5D[0] is not None:
    print(f"Flutter speed mode 1 - 5D: {flutter_speed_modes_5D[0]:.2f}")
else:
    print("Flutter speed mode 1 - 5D: Not found")

if flutter_speed_modes_5D[1] is not None:   
    print(f"Flutter speed mode 2 - 5D: {flutter_speed_modes_5D[1]:.2f}")
else:
    print("Flutter speed mode 2 - 5D: Not found")

if flutter_speed_modes_5D[2] is not None:   
    print(f"Flutter speed mode 3 - 5D: {flutter_speed_modes_5D[2]:.2f}")
else:
    print("Flutter speed mode 3 - 5D: Not found")

if flutter_speed_modes_5D[3] is not None:   
    print(f"Flutter speed mode 4 - 5D: {flutter_speed_modes_5D[3]:.2f}")
else:
    print("Flutter speed mode 4 - 5D: Not found")

_eigVal.plot_damping_vs_wind_speed_single(damping_ratios_5D,V_list,  dist="5D",single = False)
_eigVal.plot_frequency_vs_wind_speed(B, v_range_5D, V_list, omega_all=omega_all_5D, dist="5D", single = False)
_eigVal.plot_flutter_mode_shape(eigvecs_all_5D,flutter_idx_modes_5D, dist="5D", single = False)
