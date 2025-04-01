
import numpy as np
import os

file_path = "./Arrays_AD"
print("Gjeldende arbeidsmappe:", os.getcwd())


# Values from FEM-model
m1 = 1000 #kg, vertical
m2 = 2000 #kg, torsion
f1 = 10 #Hz, vertical
f2 = 20 #Hz, torsion


zeta = 0.005 # 0.5 %, critical damping

# Stiffness
k1 = (2*np.pi*f1)**2*m1 #  (Vertical)
k2 = (2*np.pi*f2)**2*m2 #  (Torsion)

# Damping
c1 = 2*zeta*np.sqrt(k1*m1) # (vertical)
c2 = 2*zeta*np.sqrt(k2*m2) # (torsion)

# Structural matrices, diagonal matrices in modal coordss
Ms = np.diag([m1, m2]) #Mass matrix
Ks = np.diag([k1, k2]) #Stiffness matrix
Cs = np.diag([c1, c2]) #Damping matrix

# Single deck
def get_aero_matrices_single(V, B, poly_coeff, f1, f2):
    """
    Calculates aerodynamic damping and stiffness matrices (2DOF: vertical and torsion)
    for a given wind speed V, using reduced velocity and polynomial AD coefficients.

    Parameters:
    -----------
    V : float
        Wind speed [m/s]
    B : float
        Section width [m]
    rho : float
        Air density [kg/m^3]
    poly_coeff : ndarray
        Polynomial coefficients of aerodynamic derivatives, shape (8, 3)
    f1 : float
        Natural frequency of vertical mode [Hz]
    f2 : float
        Natural frequency of torsional mode [Hz]

    Returns:
    --------
    C_aero : ndarray
        2x2 aerodynamic damping matrix
    K_aero : ndarray
        2x2 aerodynamic stiffness matrix
    """

    # Reduced velocities
    Vr1 = V / (f1 * B)
    Vr2 = V / (f2 * B)

    # Evaluate aerodynamic derivatives at reduced velocities
    # For simplicity: use Vr1 (vert) for H* and Vr2 (tors) for A*
    H1_star = np.polyval(poly_coeff[0], Vr1)
    H2_star = np.polyval(poly_coeff[1], Vr1)
    H3_star = np.polyval(poly_coeff[2], Vr1)
    H4_star = np.polyval(poly_coeff[3], Vr1)

    A1_star = np.polyval(poly_coeff[4], Vr2)
    A2_star = np.polyval(poly_coeff[5], Vr2)
    A3_star = np.polyval(poly_coeff[6], Vr2)
    A4_star = np.polyval(poly_coeff[7], Vr2)

    # Aerodynamic matrices
    C_aero = np.array([
        [H1_star, B * H2_star],
        [B * A1_star, B**2 * A2_star]
    ])

    K_aero = np.array([
        [H4_star, B*H3_star],
        [B*A4_star,B**2*A3_star]
    ])

    return C_aero, K_aero



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
