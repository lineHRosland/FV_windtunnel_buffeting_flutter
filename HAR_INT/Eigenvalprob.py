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

flutter_speed, damping_ratios, min_damping_ratios, eigvals_all, eigvecs_all, Vred_list=_eigVal.solve_flutter_single(poly_coeff_single, v_range_single, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N)

_eigVal.plot_damping_vs_wind_speed_single(flutter_speed, Vred_list, damping_ratios)
_eigVal.plot_eigenvalues_over_v(Vred_list, eigvals_all)

_eigVal.plot_flutter_results(Vred_list, eigvals_all)

_eigVal.plot_flutter_modes(Vred_list, eigvals_all, split_modes=True, labels=None)

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

#print(poly_coeff_3D.shape)  # Skal være (32, 3)
#print(v_range_3D)           # Skal være f.eks. [min, max]

flutter_speed_local, damping_ratios_local, eigvals_all_local, eigvecs_all_local, V_local, flutter_speed_global, damping_ratios_global, eigvals_all_global, eigvecs_all_global, V_global= _eigVal.solve_flutter_single(poly_coeff_3D, v_range_3D, ms1, ms2, fs1, fs2, B, rho, zeta, max_iter, eps, N)


#%%
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


###############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def uncoupled_freq_damping(w_s, H4_star, A3_star, mu):
    """
    Beregn uncoupled (modifiserte) frekvenser og demping for vertikal og torsjon
    """
    w_bar_1 = w_s[0] * np.sqrt(1 - (mu * H4_star)**2)
    w_bar_2 = w_s[1] * np.sqrt(1 - (mu * A3_star)**2)

    return w_bar_1, w_bar_2

def uncoupled_damping(xi_s, w_s, w_bar, H1_star, A2_star, mu):
    """
    Beregn modifisert demping (ξ̄) for uncoupled modes
    """
    xi_bar_1 = xi_s[0] * (w_s[0] / w_bar[0]) - 0.5 * mu * (H1_star / w_bar[0])
    xi_bar_2 = xi_s[1] * (w_s[1] / w_bar[1]) - 0.5 * mu * (A2_star / w_bar[1])
    return xi_bar_1, xi_bar_2

# Eksempelverdier (kan erstattes med reelle)
w_s = [10, 20]         # struktur-frekvenser (Hz)
xi_s = [0.005, 0.005]  # struktur-demping
mu = 0.03              # masseparameter
H4_star = 0.1
A3_star = 0.1
H1_star = 0.2
A2_star = 0.15

# Steg 1: uncoupled frekvens og damping
w_bar_1, w_bar_2 = uncoupled_freq_damping(w_s, H4_star, A3_star, mu)
xi_bar_1, xi_bar_2 = uncoupled_damping(xi_s, w_s, (w_bar_1, w_bar_2), H1_star, A2_star, mu)

results = {
    "Uncoupled modifisert frekvens ω̄1": w_bar_1,
    "Uncoupled modifisert frekvens ω̄2": w_bar_2,
    "Uncoupled demping ξ̄1": xi_bar_1,
    "Uncoupled demping ξ̄2": xi_bar_2
}

import ace_tools as tools; tools.display_dataframe_to_user(name="Uncoupled Flutter Resultater", dataframe=pd.DataFrame([results]))


# %%
