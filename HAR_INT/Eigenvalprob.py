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
m1 = 1000 #kg, vertical
m2 = 2000 #kg, torsion
f1 = 10 #Hz, vertical
f2 = 20 #Hz, torsion


zeta = 0.005 # 0.5 %, critical damping


if os.path.exists(os.path.join(file_path, "poly_coeff_single.npy")):
    poly_coeff_single = np.load(os.path.join(file_path, "poly_coeff_single.npy"))
else:
    raise FileNotFoundError(f"The file 'poly_coeff_single.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

if os.path.exists(os.path.join(file_path, "v_range_single.npy")):
    v_range_single = np.load(os.path.join(file_path, "v_range_single.npy"))
else:
    raise FileNotFoundError(f"The file 'v_range_single.npy' does not exist in the specified path: {os.path.abspath(file_path)}")

#print(poly_coeff_single.shape)  # Skal være (8, 3)
#print(v_range_single)           # Skal være f.eks. [min, max]




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

# Call the function to evaluate aerodynamic matrices
#vertical
C_aero1_single, K_aero1_single, V_all1_single = _eigVal.cae_kae_single(poly_coeff_single, v_range_single, B, f1, 100)
#torsion
C_aero2_single, K_aero2_single, V_all2_single = _eigVal.cae_kae_single(poly_coeff_single, v_range_single, B, f2, 100)

# Call the function to compute structural matrices
Ms, Cs, Ks = _eigVal.structural_matrices(m1, m2, f1, f2, zeta)

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