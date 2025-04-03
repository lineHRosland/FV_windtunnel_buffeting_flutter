# -*- coding: utf-8 -*-
"""
Created in April 2025

@author: linehro
"""

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
ms1 = 1000 #kg, vertical
ms2 = 2000 #kg, torsion
fs1 = 10 #Hz, vertical
fs2 = 20 #Hz, torsion

ws1 = 2*np.pi*fs1 # rad/s, vertical FØRSTE ITERASJON
ws2 = 2*np.pi*fs2 # rad/s, torsion FØRSTE ITERASJON

zeta = 0.005 # 0.5 %, critical damping
rho = 1.225 # kg/m^3, air density


scale = 1/50

N = 100 # Number of steps in V_list

#%%
#ITERATIVE BIMODAL EIGENVALUE APPROACH
eps = 1e-3  # Konvergensterskel
max_iter = 10  # Maksimalt antall iterasjoner
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

#print(poly_coeff_single.shape)  # Skal være (8, 3)
#print(v_range_single)           # Skal være (8,2)

flutter_speed, damping_ratios, min_damping_ratios, eigvals_all, eigvecs_all, Vred_list=_eigVal.solve_flutter_single(poly_coeff_single, v_range_single, m1, m2, f1, f2, B, rho, zeta, max_iter, eps, N)

_eigVal.plot_damping_vs_wind_speed_single(flutter_speed, Vred_list, damping_ratios)


#%%

if os.path.exists(os.path.join(file_path, "poly_coeff_3D.npy")):
    poly_coeff_3D = np.load(os.path.join(file_path, "poly_coeff_3D.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_3D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_3D.npy")):
    v_range_3D = np.load(os.path.join(file_path, "v_range_3D.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_3D.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

#print(poly_coeff_3D.shape)  # Skal være (32, 3)
#print(v_range_3D)           # Skal være f.eks. [min, max]



# Print the results
print("Mass Matrix (M):")
print(Ms)

print("\nStructural Damping Matrix (C_struc):")
print(Cs)

print("\nStructural Stiffness Matrix (K_struc):")
print(Ks)

# Call the function to evaluate aerodynamic matrices
#vertical
C_aero1, K_aero1, V_all1 = _eigVal.cae_kae_twin(poly_coeff_single, v_range_single, B, f1, 100)
#torsion
C_aero2, K_aero2, V_all2 = _eigVal.cae_kae_twin(poly_coeff_single, v_range_single, B, f2, 100)


# Print the aerodynamic damping and stiffness matrices
#print("Aerodynamic Damping Matrices (C_aero):")
#print(C_aero1[0])

#print("Aerodynamic Damping Matrices (C_aero):")
#print(C_aero2[0])

#print("\nAerodynamic Stiffness Matrices (K_aero):")
#print(K_aero1)

# ALTERNATIVE 1: WITHOUT ITERATION




# ALTERNATIVE 2: WITH ITERATION

# V = 0, wi(V=0) egenfrekvens i still air
# Det betyr at den egenfrekvensen 𝜔  du bruker, egentlig bør avhenge av vindhastigheten V.

#1. 
    # Bruk FEM-verdier for 𝜔1, w2. 
    # Evaluer matrisene 𝑀,𝐶+𝐶𝑎𝑒,𝐾+𝐾𝑎𝑒  for mange ulike vindhastigheter V. 
    # Løs generalisert egenverdiproblem. Sjekk hvor dempingen blir null → det er flutterhastigheten. 
    # Plot Demping 𝜁 eller Re(𝜆) som funksjon av V
#2. Med iterasjon (for nøyaktig fluttergrense)
    # Når du har funnet omtrentlig flutterhastighet, kan du i ettertid lage et lite iterativt skript for mer nøyaktig justering i akkurat det området.
    # Start med w0.
    # Finn Cae, Kae med w0
    # Løs egenverdiproblemet, finn ny w, w1
    # Sjekk om w1 er lik w0. 
    # Repeter til w1 er lik w0 med ønsket nøyaktighet.



# finn w iterativt, deretter finn demping for ulike vindhastigheter
# Du løser egenverdiproblemet for hver V, og sjekker dempingen. Egenverdiproblemet gir egenverdier og egenvektorer.
#     Egenverdiene er komplekse og trengs for å regne ut dempingen.
#     Egenvektorene er komplekse og gir fase informasjon. Dette er viktig å kommentere.
# Man får en frekvens per løsningn av egenverdiprob for en gitt vindhastighet.
    
# #CLOSED FORM
# # Constants
# I = #FEM
# mikro = rho * (B/2)**2/I
# nu = rho * (B/2)**4/I
# D = G_h1alpha2/np.sqrt(G_h1h1*G_alpha2alpha2) #G: mode shapes

# # Isolated modes
# w1_bar = ws1*np.sqrt(1-mikro(w/ws1)**2*H4) 