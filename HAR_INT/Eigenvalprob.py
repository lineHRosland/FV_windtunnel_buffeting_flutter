
import numpy as np
import os
import sys
sys.path.append(r"C:\Users\liner\Documents\Github\Masteroppgave\w3tp")
import w3t as w3t

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

print(poly_coeff_single.shape)  # Skal være (8, 3)
print(v_range_single)           # Skal være f.eks. [min, max]



# Call the function to evaluate aerodynamic matrices
C_aero, K_aero, V_all = w3t.cae_kae_single(poly_coeff_single, v_range_single, B, 100)

# Print the aerodynamic damping and stiffness matrices
print("Aerodynamic Damping Matrices (C_aero):")
print(C_aero[0])

print("Aerodynamic Damping Matrices (C_aero):")
print(C_aero[4])

print("\nAerodynamic Stiffness Matrices (K_aero):")
print(K_aero)
