 
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

print("w1V, w2V, w1T: ", w1V, w2V, w1T)
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
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Buffeting\Cae_Kae_updated_stat_coeff.npy"
# Load the saved dictionary
matrices = np.load(file_path, allow_pickle=True).item()

Kae_Single_gen = matrices["Kae_Single"]
Cae_Single_gen = matrices["Cae_Single"]

Kae_1D_gen = matrices["Kae_1D"]
Cae_1D_gen = matrices["Cae_1D"]
print("Kae_1D_gen", 37.6*37.6*Kae_1D_gen)

Kae_2D_gen = matrices["Kae_2D"]
Cae_2D_gen = matrices["Cae_2D"]

Kae_3D_gen = matrices["Kae_3D"]
Cae_3D_gen = matrices["Cae_3D"]

Kae_4D_gen = matrices["Kae_4D"]
Cae_4D_gen = matrices["Cae_4D"]

Kae_5D_gen = matrices["Kae_5D"]
Cae_5D_gen = matrices["Cae_5D"]
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



##########################################################################
######################################################################################
######################################################################################

#AD

#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_flutter(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, dist="Single deck",  single = True, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')


# fig, ax =_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
# fig.tight_layout()
# fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_single_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')



#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_flutter(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#%%
#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, dist="two deck 1D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

# fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
# fig.tight_layout()
# fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_1D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_flutter(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_2D, eigvecs_list_two_2D, V_list_two_2D, dist="two deck 2D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

# fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
# fig.tight_layout()
# fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_flutter(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D, V_list_two_3D, dist="two deck 3D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

# fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
# fig.tight_layout()
# fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_3D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_flutter(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_4D, eigvecs_list_two_4D, V_list_two_4D, dist="two deck 4D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

# fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
# fig.tight_layout()
# fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_4D_flutter_dof" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_flutter(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting
fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_5D, eigvecs_list_two_5D, V_list_two_5D, dist="two deck 5D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_damp" + ".png"), dpi=300)

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_frek" + ".png"), dpi=300)

fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

# fig, ax = _eigVal.plot_flutter_mode_shape(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
# fig.tight_layout()
# fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "AD_5D_flutter_dof" + ".png"), dpi=300)

######################################################################################
######################################################################################
######################################################################################

# BUFFETING

#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_flutter(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, buffeting = True, Cae_star_gen_BUFF=Cae_Single_gen, Kae_star_gen_BUFF=Kae_Single_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, dist="Single deck",  single = True, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_single_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_single_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_single_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_single_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_flutter(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_1D_gen, Kae_star_gen_BUFF=Kae_1D_gen,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, dist="two deck 1D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_1D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_1D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_1D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_flutter(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,buffeting = True,  Cae_star_gen_BUFF=Cae_2D_gen, Kae_star_gen_BUFF=Kae_2D_gen,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting


fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_2D,eigvecs_list_two_2D,V_list_two_2D, dist="two deck 2D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_flutter(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_3D_gen, Kae_star_gen_BUFF=Kae_3D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting


fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D,V_list_two_3D, dist="two deck 3D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_3D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_3D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_3D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_flutter(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_4D_gen, Kae_star_gen_BUFF=Kae_4D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting


fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_4D,eigvecs_list_two_4D,V_list_two_4D, dist="two deck 4D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_4D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_4D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_4D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_flutter(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_5D_gen, Kae_star_gen_BUFF=Kae_5D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting

fig, ax = _eigVal.plot_damping_vs_wind_speed(damping_list_two_5D,eigvecs_list_two_5D,V_list_two_5D, dist="two deck 5D",  single = False, buffeting = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_5D_flutter_damp" + ".png"), dpi=300, bbox_inches='tight')

fig, ax = _eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_5D_flutter_frek" + ".png"), dpi=300, bbox_inches='tight')


fig, ax =_eigVal.plot_flutter_mode_shape_top(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_5D_flutter_dof_top" + ".png"), dpi=300, bbox_inches='tight')

fig, ax =_eigVal.plot_flutter_mode_shape_bunn(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)
fig.tight_layout()
fig.savefig(os.path.join(r"C:\Users\liner\OneDrive - NTNU\NTNU\12 semester\Plot\Masteroppgave", "Buff_2D_flutter_dof_bunn" + ".png"), dpi=300, bbox_inches='tight')

# %%
############################################################################################################
######################################################################################
######################################################################################

#%%
# Plotte AD mot tilsvarende buffeting

#  BUFFETING
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Buffeting\Cae_Kae_as_AD.npy"
# Load the saved dictionary
matrices = np.load(file_path, allow_pickle=True).item()

Single_k = matrices["Kae_Single"]
Single_c = matrices["Cae_Single"]

Buff_1D_k = matrices["Kae_1D"]
Buff_1D_c = matrices["Cae_1D"]

Buff_2D_k = matrices["Kae_2D"]
Buff_2D_c = matrices["Cae_2D"]

Buff_3D_k = matrices["Kae_3D"]
Buff_3D_c = matrices["Cae_3D"]

Buff_4D_k = matrices["Kae_4D"]
Buff_4D_c = matrices["Cae_4D"]

Buff_5D_k = matrices["Kae_5D"]
Buff_5D_c = matrices["Cae_5D"]
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

k = np.linspace(k_range_single[0,0], k_range_single[0,1], 10)

i = 0
AD_single_damping = np.zeros((10, 2, 2))
AD_single_stiffness = np.zeros((10, 2, 2))
for ki in k:
    AD_single_damping[i], AD_single_stiffness[i] = AD_single(poly_coeff_single,k_range_single,  1/ki, B)
    i += 1

Vcr = 84.50000000745058 # m/s, critical wind speed
omega_cr = 1.4569030259030422 # rad/s, critical frequency
K = omega_cr * B / Vcr

# y-verdier for kurven du har plottet
y_vals_c = k * AD_single_damping[:,1,1]
f_interp = interp1d(k, y_vals_c, kind='linear', fill_value="extrapolate")
y_K_c = f_interp(K)

y_vals_k = k**2 * AD_single_stiffness[:,0,0]
f_interp = interp1d(k, y_vals_k, kind='linear', fill_value="extrapolate")
y_K_k = f_interp(K)

plt.figure(figsize=(10, 6))
plt.title(" Single Damping")
# plt.plot(k, k*AD_single_damping[:,0,0],  label=r"$H_1*$ Unsteady")
plt.plot(k, k*AD_single_damping[:,1,1],  label=r"$A_2^*$ Unsteady")
# plt.plot(k, np.full_like(k, Single_c[0,0]), linestyle = "-.",label=r"$H_1*$ Quasi-static", alpha = 0.8)
plt.plot(k,np.full_like(k,Single_c[1,1]), linestyle = "-.",label=r"$A_2^*$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K, y_K_c, 'ro', color='black')
plt.plot(K,Single_c[1,1], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$ K*AD$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.title(" Single Stiffness")
plt.plot(k, k**2*AD_single_stiffness[:,0,0],label=r"$H_4^*$ Unsteady")
# plt.plot(k, k**2*AD_single_stiffness[:,1,1], label=r"$A_3*$ Unsteady")
plt.plot(k, np.full_like(k,Single_k[0,0]),linestyle = "-.", label=r"$H_4^*$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Single_k[1,1]), linestyle = "-.",label=r"$A_3*$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K,y_K_k, 'ro', color='black')
plt.plot(K,Single_k[0,0], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$K^2 * AD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


#%%
# 1D

k = np.linspace(k_range_1D[0,0], k_range_1D[0,1], 10)

i = 0
AD_1D_damping = np.zeros((10, 4, 4))
AD_1D_stiffness = np.zeros((10, 4, 4))
for ki in k:
    AD_1D_damping[i], AD_1D_stiffness[i] = AD_two(poly_coeff_1D,k_range_1D,  1/ki, B)
    i += 1

Vcr = 53.50000000745058# m/s, critical wind speed
omega_cr = 1.9649638980089759 # rad/s, critical frequency
K = omega_cr * B / Vcr

# y-verdier for kurven du har plottet
y_vals_c = k * AD_1D_damping[:,3,3]
f_interp = interp1d(k, y_vals_c, kind='linear', fill_value="extrapolate")
y_K_c = f_interp(K)

y_vals_k = k**2 * AD_1D_stiffness[:,2,2]
f_interp = interp1d(k, y_vals_k, kind='cubic', fill_value="extrapolate")
y_K_k = f_interp(K)

plt.figure(figsize=(10, 6))
plt.title(" 1D Damping")
# plt.plot(k, k*AD_1D_damping[:,0,0],  label=r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_1D_damping[:,1,1],  label=r"r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_1D_damping[:,2,2],  label=r"$c_{z2z2}$ Unsteady")
plt.plot(k, k*AD_1D_damping[:,3,3],  label=r"$c_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k, Buff_1D_c[0,0]), linestyle = "-.",label=r"$c_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_1D_c[1,1]), linestyle = "-.",label=r"$c_{θ1θ1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_1D_c[2,2]), linestyle = "--",label=r"$c_{z2z2}$ Quasi-static", alpha = 0.8)
plt.plot(k,np.full_like(k,Buff_1D_c[3,3]),linestyle = "--", label=r"$c_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K, y_K_c, 'ro', color='black')
plt.plot(K,Buff_1D_c[3,3], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$ K*AD$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.title(" 1D Stiffness")
# plt.plot(k, k**2*AD_1D_stiffness[:,0,0],label=r"$k_{z1z1}$ Unsteady")

# plt.plot(k, k**2*AD_1D_stiffness[:,1,1],  label=r"$k_{z1z1}$ Unsteady")
plt.plot(k, k**2*AD_1D_stiffness[:,2,2], label=r"$k_{z2z2}$ Unsteady")
# plt.plot(k, k**2*AD_1D_stiffness[:,3,3], label=r"$k_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k,Buff_1D_k[0,0]),linestyle = "-.", label=r"$k_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_1D_k[1,1]), linestyle = "-.",label=r"$k_{θ1θ1}$ Quasi-static", alpha = 0.8)
plt.plot(k, np.full_like(k,Buff_1D_k[2,2]), linestyle = "--",label=r"$k_{z2z2}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_1D_k[3,3]),linestyle = "--", label=r"$k_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K,y_K_k, 'ro', color='black')
plt.plot(K,Buff_1D_k[2,2], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$K^2 * AD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()



#%%
# 2D

k = np.linspace(k_range_2D[0,0], k_range_2D[0,1], 10)

i = 0
AD_2D_damping = np.zeros((10, 4, 4))
AD_2D_stiffness = np.zeros((10, 4, 4))
for ki in k:
    AD_2D_damping[i], AD_2D_stiffness[i] = AD_two(poly_coeff_2D,k_range_2D,  1/ki, B)
    i += 1

Vcr = 53.50000000745058# m/s, critical wind speed
omega_cr = 1.9649638980089759 # rad/s, critical frequency
K = omega_cr * B / Vcr

# y-verdier for kurven du har plottet
y_vals_c = k * AD_2D_damping[:,3,3]
f_interp = interp1d(k, y_vals_c, kind='linear', fill_value="extrapolate")
y_K_c = f_interp(K)

y_vals_k = k**2 * AD_2D_stiffness[:,2,2]
f_interp = interp1d(k, y_vals_k, kind='cubic', fill_value="extrapolate")
y_K_k = f_interp(K)

plt.figure(figsize=(10, 6))
plt.title(" 2D Damping")
# plt.plot(k, k*AD_2D_damping[:,0,0],  label=r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_2D_damping[:,1,1],  label=r"r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_2D_damping[:,2,2],  label=r"$c_{z2z2}$ Unsteady")
plt.plot(k, k*AD_2D_damping[:,3,3],  label=r"$c_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k, Buff_2D_c[0,0]), linestyle = "-.",label=r"$c_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_2D_c[1,1]), linestyle = "-.",label=r"$c_{θ1θ1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_2D_c[2,2]), linestyle = "--",label=r"$c_{z2z2}$ Quasi-static", alpha = 0.8)
plt.plot(k,np.full_like(k,Buff_2D_c[3,3]),linestyle = "--", label=r"$c_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K, y_K_c, 'ro', color='black')
plt.plot(K,Buff_2D_c[3,3], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$ K*AD$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.title(" 2D Stiffness")
# plt.plot(k, k**2*AD_2D_stiffness[:,0,0],label=r"$k_{z1z1}$ Unsteady")
# plt.plot(k, k**2*AD_2D_stiffness[:,1,1],  label=r"$k_{z1z1}$ Unsteady")
plt.plot(k, k**2*AD_2D_stiffness[:,2,2], label=r"$k_{z2z2}$ Unsteady")
# plt.plot(k, k**2*AD_2D_stiffness[:,3,3], label=r"$k_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k,Buff_2D_k[0,0]),linestyle = "-.", label=r"$k_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_2D_k[1,1]), linestyle = "-.",label=r"$k_{θ1θ1}$ Quasi-static", alpha = 0.8)
plt.plot(k, np.full_like(k,Buff_2D_k[2,2]), linestyle = "--",label=r"$k_{z2z2}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_2D_k[3,3]),linestyle = "--", label=r"$k_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K,y_K_k, 'ro', color='black')
plt.plot(K,Buff_2D_k[2,2], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$K^2 * AD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()
#%%
# 3D

k = np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10)
i = 0
AD_3D_damping = np.zeros((10, 4, 4))
AD_3D_stiffness = np.zeros((10, 4, 4))
for ki in k:
    AD_3D_damping[i], AD_3D_stiffness[i] = AD_two(poly_coeff_3D,k_range_3D,  1/ki, B)
    i += 1


Vcr = 53.50000000745058# m/s, critical wind speed
omega_cr = 1.9649638980089759 # rad/s, critical frequency
K = omega_cr * B / Vcr

# y-verdier for kurven du har plottet
y_vals_c = k * AD_3D_damping[:,3,3]
f_interp = interp1d(k, y_vals_c, kind='linear', fill_value="extrapolate")
y_K_c = f_interp(K)

y_vals_k = k**2 * AD_3D_stiffness[:,2,2]
f_interp = interp1d(k, y_vals_k, kind='cubic', fill_value="extrapolate")
y_K_k = f_interp(K)

plt.figure(figsize=(10, 6))
plt.title(" 3D Damping")
# plt.plot(k, k*AD_3D_damping[:,0,0],  label=r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_3D_damping[:,1,1],  label=r"r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_3D_damping[:,2,2],  label=r"$c_{z2z2}$ Unsteady")
plt.plot(k, k*AD_3D_damping[:,3,3],  label=r"$c_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k, Buff_3D_c[0,0]), linestyle = "-.",label=r"$c_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_3D_c[1,1]), linestyle = "-.",label=r"$c_{θ1θ1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_3D_c[2,2]), linestyle = "--",label=r"$c_{z2z2}$ Quasi-static", alpha = 0.8)
plt.plot(k,np.full_like(k,Buff_3D_c[3,3]),linestyle = "--", label=r"$c_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K, y_K_c, 'ro', color='black')
plt.plot(K,Buff_3D_c[3,3], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$ K*AD$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.title(" 3D Stiffness")
# plt.plot(k, k**2*AD_3D_stiffness[:,0,0],label=r"$k_{z1z1}$ Unsteady")
# plt.plot(k, k**2*AD_3D_stiffness[:,1,1],  label=r"$k_{z1z1}$ Unsteady")
plt.plot(k, k**2*AD_3D_stiffness[:,2,2], label=r"$k_{z2z2}$ Unsteady")
# plt.plot(k, k**2*AD_3D_stiffness[:,3,3], label=r"$k_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k,Buff_3D_k[0,0]),linestyle = "-.", label=r"$k_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_3D_k[1,1]), linestyle = "-.",label=r"$k_{θ1θ1}$ Quasi-static", alpha = 0.8)
plt.plot(k, np.full_like(k,Buff_3D_k[2,2]), linestyle = "--",label=r"$k_{z2z2}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_3D_k[3,3]),linestyle = "--", label=r"$k_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K,y_K_k, 'ro', color='black')
plt.plot(K,Buff_3D_k[2,2], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$K^2 * AD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


#%%
# 4D

k = np.linspace(k_range_4D[0,0], k_range_4D[0,1], 10)
i = 0
AD_4D_damping = np.zeros((10, 4, 4))
AD_4D_stiffness = np.zeros((10, 4, 4))
for ki in k:
    AD_4D_damping[i], AD_4D_stiffness[i] = AD_two(poly_coeff_4D,k_range_4D,  1/ki, B)
    i += 1


Vcr = 53.50000000745058# m/s, critical wind speed
omega_cr = 1.9649638980089759 # rad/s, critical frequency
K = omega_cr * B / Vcr

# y-verdier for kurven du har plottet
y_vals_c = k * AD_4D_damping[:,3,3]
f_interp = interp1d(k, y_vals_c, kind='linear', fill_value="extrapolate")
y_K_c = f_interp(K)

y_vals_k = k**2 * AD_4D_stiffness[:,2,2]
f_interp = interp1d(k, y_vals_k, kind='cubic', fill_value="extrapolate")
y_K_k = f_interp(K)

plt.figure(figsize=(10, 6))
plt.title(" 4D Damping")
# plt.plot(k, k*AD_4D_damping[:,0,0],  label=r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_4D_damping[:,1,1],  label=r"r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_4D_damping[:,2,2],  label=r"$c_{z2z2}$ Unsteady")
plt.plot(k, k*AD_4D_damping[:,3,3],  label=r"$c_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k, Buff_4D_c[0,0]), linestyle = "-.",label=r"$c_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_4D_c[1,1]), linestyle = "-.",label=r"$c_{θ1θ1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_4D_c[2,2]), linestyle = "--",label=r"$c_{z2z2}$ Quasi-static", alpha = 0.8)
plt.plot(k,np.full_like(k,Buff_4D_c[3,3]),linestyle = "--", label=r"$c_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K, y_K_c, 'ro', color='black')
plt.plot(K,Buff_4D_c[3,3], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$ K*AD$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.title(" 4D Stiffness")
# plt.plot(k, k**2*AD_4D_stiffness[:,0,0],label=r"$k_{z1z1}$ Unsteady")
# plt.plot(k, k**2*AD_4D_stiffness[:,1,1],  label=r"$k_{z1z1}$ Unsteady")
plt.plot(k, k**2*AD_4D_stiffness[:,2,2], label=r"$k_{z2z2}$ Unsteady")
# plt.plot(k, k**2*AD_4D_stiffness[:,3,3], label=r"$k_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k,Buff_4D_k[0,0]),linestyle = "-.", label=r"$k_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_4D_k[1,1]), linestyle = "-.",label=r"$k_{θ1θ1}$ Quasi-static", alpha = 0.8)
plt.plot(k, np.full_like(k,Buff_4D_k[2,2]), linestyle = "--",label=r"$k_{z2z2}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_4D_k[3,3]),linestyle = "--", label=r"$k_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K,y_K_k, 'ro', color='black')
plt.plot(K,Buff_4D_k[2,2], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$K^2 * AD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


#%%
# 5D

k = np.linspace(k_range_5D[0,0], k_range_5D[0,1], 10)
i = 0
AD_5D_damping = np.zeros((10, 4, 4))
AD_5D_stiffness = np.zeros((10, 4, 4))
for ki in k:
    AD_5D_damping[i], AD_5D_stiffness[i] = AD_two(poly_coeff_5D,k_range_5D,  1/ki, B)
    i += 1


Vcr = 53.50000000745058# m/s, critical wind speed
omega_cr = 1.9649638980089759 # rad/s, critical frequency
K = omega_cr * B / Vcr

# y-verdier for kurven du har plottet
y_vals_c = k * AD_5D_damping[:,3,3]
f_interp = interp1d(k, y_vals_c, kind='linear', fill_value="extrapolate")
y_K_c = f_interp(K)

y_vals_k = k**2 * AD_5D_stiffness[:,2,2]
f_interp = interp1d(k, y_vals_k, kind='cubic', fill_value="extrapolate")
y_K_k = f_interp(K)

plt.figure(figsize=(10, 6))
plt.title(" 5D Damping")
# plt.plot(k, k*AD_5D_damping[:,0,0],  label=r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_5D_damping[:,1,1],  label=r"r"$c_{z1z1}$ Unsteady")
# plt.plot(k, k*AD_5D_damping[:,2,2],  label=r"$c_{z2z2}$ Unsteady")
plt.plot(k, k*AD_5D_damping[:,3,3],  label=r"$c_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k, Buff_5D_c[0,0]), linestyle = "-.",label=r"$c_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_5D_c[1,1]), linestyle = "-.",label=r"$c_{θ1θ1}$ Quasi-static", alpha = 0.8)
# plt.plot(k,np.full_like(k,Buff_5D_c[2,2]), linestyle = "--",label=r"$c_{z2z2}$ Quasi-static", alpha = 0.8)
plt.plot(k,np.full_like(k,Buff_5D_c[3,3]),linestyle = "--", label=r"$c_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K, y_K_c, 'ro', color='black')
plt.plot(K,Buff_5D_c[3,3], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$: K*AD$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.title(" 5D Stiffness")
# plt.plot(k, k**2*AD_5D_stiffness[:,0,0],label=r"$k_{z1z1}$ Unsteady")
# plt.plot(k, k**2*AD_5D_stiffness[:,1,1],  label=r"$k_{z1z1}$ Unsteady")
plt.plot(k, k**2*AD_5D_stiffness[:,2,2], label=r"$k_{z2z2}$ Unsteady")
# plt.plot(k, k**2*AD_5D_stiffness[:,3,3], label=r"$k_{θ2θ2}$ Unsteady")
# plt.plot(k, np.full_like(k,Buff_5D_k[0,0]),linestyle = "-.", label=r"$k_{z1z1}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_5D_k[1,1]), linestyle = "-.",label=r"$k_{θ1θ1}$ Quasi-static", alpha = 0.8)
plt.plot(k, np.full_like(k,Buff_5D_k[2,2]), linestyle = "--",label=r"$k_{z2z2}$ Quasi-static", alpha = 0.8)
# plt.plot(k, np.full_like(k,Buff_5D_k[3,3]),linestyle = "--", label=r"$k_{θ2θ2}$ Quasi-static", alpha = 0.8)
plt.axvline(x=K, color='black', linestyle='--', linewidth=1, label="Kcr")
plt.plot(K,y_K_k, 'ro', color='black')
plt.plot(K,Buff_5D_k[2,2], 'ro', color='black')
plt.xlabel(r"$K$", fontsize=16)
plt.ylabel(r"$K^2 * AD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
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
