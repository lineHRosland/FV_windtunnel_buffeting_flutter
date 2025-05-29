 
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
file_path = r"C:\Users\liner\Documents\Github\Masteroppgave\HAR_INT\Buffeting\Cae_Kae_ny.npy"
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


####################################3
#%%
# Plotte Kae og Cae
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
def cae_kae_two(poly_coeff, k_range, Vred_global, B):

    Vred_global = float(Vred_global) 

    # AD
    # Damping derivatives (indices 0–15)

    c_z1z1 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True)
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
         [c_z1z1,       B * c_z1θ1,       c_z1z2,       B * c_z1θ2],
         [B * c_θ1z1,   B**2 * c_θ1θ1,   B * c_θ1z2,   B**2 * c_θ1θ2],
         [c_z2z1,       B * c_z2θ1,       c_z2z2,       B * c_z2θ2],
         [B * c_θ2z1,   B**2 * c_θ2θ1,   B * c_θ2z2,   B**2 * c_θ2θ2]
    ])
    Kae_star = np.array([
         [k_z1z1,       B * k_z1θ1,       k_z1z2,       B * k_z1θ2],
         [B * k_θ1z1,   B**2 * k_θ1θ1,   B * k_θ1z2,   B**2 * k_θ1θ2],
         [k_z2z1,       B * k_z2θ1,       k_z2z2,       B * k_z2θ2],
         [B * k_θ2z1,   B**2 * k_θ2θ1,   B * k_θ2z2,   B**2 * k_θ2θ2]
    ])

    return Cae_star, Kae_star

def generalize_C_K(C, K, Phi, x, single=True):
    N = len(x)
    n_modes = Phi.shape[2]
    Cae_star_gen = np.zeros((n_modes, n_modes))
    Kae_star_gen = np.zeros((n_modes, n_modes))

    for i in range(N-1): 
        dx = x[i+1] - x[i] 
        phi_L = Phi[i] # shape (n_dof, n_modes)
        phi_R = Phi[i+1]
        # Damping
        C_int = 0.5 * (phi_L.T @ C @ phi_L + phi_R.T @ C @ phi_R)
        Cae_star_gen += C_int * dx
        # Stiffness
        K_int = 0.5 * (phi_L.T @ K @ phi_L + phi_R.T @ K @ phi_R)
        Kae_star_gen += K_int * dx
    
    return Cae_star_gen, Kae_star_gen

data4 = np.load('mode4_data.npz')
data15 = np.load('mode15_data.npz')
mode_4 = data4['mode']
mode_15 = data15['mode']
x = data4['x']
N = mode_4.shape[0]
Phi = np.zeros((N, 4, 4))
for i in range(N):
    Phi[i, 0, 0] = mode_4[i, 2]  # zz
    Phi[i, 1, 1] = mode_15[i, 3] # θθ

    Phi[i, 2, 2] = mode_4[i, 2]  # zz
    Phi[i, 3, 3] = mode_15[i, 3] # θθ
    
k = np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10)
i = 0
Cae_3D_gen_AD = np.zeros((10, 4, 4))
Kae_3D_gen_AD = np.zeros((10, 4, 4))
for ki in k:
    Cae_3D_AD, Kae_3D_AD = cae_kae_two(poly_coeff_3D,k_range_3D,  ki, B)
    Cae_3D_gen_AD[i], Kae_3D_gen_AD[i] = generalize_C_K(Cae_3D_AD, Kae_3D_AD, Phi, x, single=False)
    i += 1

plt.figure(figsize=(10, 6))
plt.title("Cae 3D deck")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Cae_3D_gen[0,0], label="Cae_3D_gen, z1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Kae_3D_gen[1,1], label="Cae_3D_gen, \theta1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Cae_3D_gen[2,2], label="Cae_3D_gen, z2")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Kae_3D_gen[3,3], label="Cae_3D_gen, \theta2")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Cae_3D_gen_AD[:,0,0],linestyle = "--",  label="Cae_AD, z1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Cae_3D_gen_AD[:,1,1], linestyle = "--", label="Cae_AD, \theta1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Cae_3D_gen_AD[:,2,2], linestyle = "--", label="Cae_AD, z2")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Cae_3D_gen_AD[:,3,3], linestyle = "--", label="Cae_AD, \theta2")

plt.xlabel("k")
plt.ylabel("Cae")
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.title("Kae 3D deck")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Kae_3D_gen[0,0], label="Kae_3D_gen, z1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Kae_3D_gen[1,1], label="Kae_3D_gen, \theta1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Kae_3D_gen[2,2], label="Kae_3D_gen, z2")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), (np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10))**2*Kae_3D_gen[3,3], label="Kae_3D_gen, \theta2")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Kae_3D_gen_AD[:,3,3], linestyle = "--", label="Kae_AD, z1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Kae_3D_gen_AD[:,1,1], linestyle = "--", label="Kae_AD, \theta1")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Kae_3D_gen_AD[:,2,2], linestyle = "--", label="Kae_AD, z2")
plt.plot(np.linspace(k_range_3D[0,0], k_range_3D[0,1], 10), Kae_3D_gen_AD[:,3,3], linestyle = "--", label="Kae_AD, \theta2")


plt.xlabel("k")
plt.ylabel("Kae")
plt.legend()
plt.show()



#%%
#Single deck

#Solve for eigenvalues and eigenvectors
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_flutter(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, dist="Single deck",  single = True, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, dist="Single deck", single = True)

_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)

#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_flutter(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, dist="two deck 1D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, dist="two deck 1D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)

#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_flutter(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_2D, eigvecs_list_two_2D, V_list_two_2D, dist="two deck 2D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D, dist="two deck 2D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_flutter(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D, V_list_two_3D, dist="two deck 3D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D, dist="two deck 3D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_flutter(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_4D, eigvecs_list_two_4D, V_list_two_4D, dist="two deck 4D",  single = False, buffeting = False)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D, dist="two deck 4D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_flutter(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = False,  Cae_star_gen_BUFF=None, Kae_star_gen_BUFF=None,  verbose=False)

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
V_list_single, omega_list_single, damping_list_single, eigvecs_list_single, eigvals_list_single, omegacritical_single, Vcritical_single = _eigVal.solve_flutter(poly_coeff_single, k_range_single, Ms_single, Cs_single, Ks_single, f1V, f1T, B, rho, eps, phi_single, x_single, single = True, buffeting = True, Cae_star_gen_BUFF=Cae_Single_gen, Kae_star_gen_BUFF=Kae_Single_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_single, Vcritical_single)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_single, eigvecs_list_single, V_list_single, dist="Single deck",  single = True, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_single, omega_list_single, dist="Single deck", single = True)

_eigVal.plot_flutter_mode_shape(eigvecs_list_single, damping_list_single, V_list_single, Vcritical_single, omegacritical_single, dist="Single deck", single = True)

#%%
#Double deck 1D

V_list_two_1D, omega_list_two_1D, damping_list_two_1D, eigvecs_list_two_1D, eigvals_list_two_1D, omegacritical_two_1D, Vcritical_two_1D = _eigVal.solve_flutter(poly_coeff_1D,k_range_1D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_1D_gen, Kae_star_gen_BUFF=Kae_1D_gen,  verbose=True)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_1D, Vcritical_two_1D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_1D,eigvecs_list_two_1D,V_list_two_1D, dist="two deck 1D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_1D, omega_list_two_1D, dist="two deck 1D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_1D, damping_list_two_1D, V_list_two_1D, Vcritical_two_1D, omegacritical_two_1D, dist="two deck 1D", single = False)

#%%
#Double deck 2D
V_list_two_2D, omega_list_two_2D, damping_list_two_2D, eigvecs_list_two_2D, eigvals_list_two_2D,omegacritical_two_2D, Vcritical_two_2D = _eigVal.solve_flutter(poly_coeff_2D,k_range_2D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False,buffeting = True,  Cae_star_gen_BUFF=Cae_2D_gen, Kae_star_gen_BUFF=Kae_2D_gen,   verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_2D, Vcritical_two_2D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_2D, eigvecs_list_two_2D, V_list_two_2D, dist="two deck 2D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_2D, omega_list_two_2D, dist="two deck 2D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_2D, damping_list_two_2D, V_list_two_2D, Vcritical_two_2D, omegacritical_two_2D, dist="two deck 2D", single = False)

#%%
#Double deck 3D
V_list_two_3D, omega_list_two_3D, damping_list_two_3D, eigvecs_list_two_3D,eigvals_list_two_3D, omegacritical_two_3D, Vcritical_two_3D = _eigVal.solve_flutter(poly_coeff_3D, k_range_3D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_3D_gen, Kae_star_gen_BUFF=Kae_3D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_3D, Vcritical_two_3D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_3D,eigvecs_list_two_3D, V_list_two_3D, dist="two deck 3D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_3D, omega_list_two_3D, dist="two deck 3D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_3D, damping_list_two_3D, V_list_two_3D, Vcritical_two_3D, omegacritical_two_3D, dist="two deck 3D", single = False)


#%%
#Double deck 4D
V_list_two_4D, omega_list_two_4D, damping_list_two_4D, eigvecs_list_two_4D, eigvals_list_two_4D,omegacritical_two_4D, Vcritical_two_4D = _eigVal.solve_flutter(poly_coeff_4D, k_range_4D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_4D_gen, Kae_star_gen_BUFF=Kae_4D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_4D, Vcritical_two_4D)

#Plotting

_eigVal.plot_damping_vs_wind_speed(damping_list_two_4D, eigvecs_list_two_4D, V_list_two_4D, dist="two deck 4D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_4D, omega_list_two_4D, dist="two deck 4D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_4D, damping_list_two_4D, V_list_two_4D, Vcritical_two_4D, omegacritical_two_4D, dist="two deck 4D", single = False)

#%%
#Double deck 5D
V_list_two_5D, omega_list_two_5D, damping_list_two_5D, eigvecs_list_two_5D,eigvals_list_two_5D,omegacritical_two_5D, Vcritical_two_5D = _eigVal.solve_flutter(poly_coeff_5D, k_range_5D, Ms_two, Cs_two, Ks_two, f1V, f1T, B, rho, eps, phi_two, x_two, single = False, buffeting = True,  Cae_star_gen_BUFF=Cae_5D_gen, Kae_star_gen_BUFF=Kae_5D_gen,  verbose=False)

#Flutter
print("Omega_cr, V_cr: ",omegacritical_two_5D, Vcritical_two_5D)

#Plotting
_eigVal.plot_damping_vs_wind_speed(damping_list_two_5D, eigvecs_list_two_5D, V_list_two_5D, dist="two deck 5D",  single = False, buffeting = True)

_eigVal.plot_frequency_vs_wind_speed(V_list_two_5D, omega_list_two_5D, dist="two deck 5D", single = False)

_eigVal.plot_flutter_mode_shape(eigvecs_list_two_5D, damping_list_two_5D, V_list_two_5D, Vcritical_two_5D, omegacritical_two_5D, dist="two deck 5D", single = False)

# %%
