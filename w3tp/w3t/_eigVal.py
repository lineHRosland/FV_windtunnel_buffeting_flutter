# -*- coding: utf-8 -*-
"""
Created in April 2025

@author: linehro
"""

import numpy as np
from scipy import linalg as spla
import matplotlib.pyplot as plt



def solve_eigvalprob(M_struc, C_struc, K_struc, C_aero, K_aero):
    """
    Løser generalisert eigenverdiproblem for gitt system.
    
    [ -λ² M + λ(C - Cae) + (K - Kae) ] φ = 0

    Parameters:
    -----------
    M : ndarray
        Mass matrix
    C_struc : ndarray
        Structural damping matrix
    K_struc : ndarray
        Structural stiffness matrix
    C_aero : ndarray
        Aerodynamic damping matrix (dimensional)
    K_aero : ndarray
        Aerodynamic stiffness matrix (dimensional)

    Returns:
    --------
    eigvals : ndarray
        Eigenvalues λ (complex), shape (n_dof*2,)
    """   
    C = C_struc - C_aero
    K = K_struc - K_aero

    # State-space A-matrix
    A = np.block([
        [np.zeros_like(M_struc), np.eye(M_struc.shape[0])],
        [-np.linalg.inv(M_struc) @ K, -np.linalg.inv(M_struc) @ C]
    ])

    # Solve the eigenvalue problem  
    eigvals, eigvecs = spla.eig(A)

    return eigvals, eigvecs

def structural_matrices(m1, m2, f1, f2, zeta, single = True):
    """
    Construct structural mass, damping, and stiffness matrices for single or twin-deck systems.

    Parameters:
    -----------
    m1 : float
        Modal mass for vertical motion (kg).
    m2 : float
        Modal mass for torsional motion (kg).
    f1 : float
        Natural frequency for vertical mode (Hz).
    f2 : float
        Natural frequency for torsional mode (Hz).
    zeta : float
        Damping ratio (assumed equal for both modes).
    single : bool, optional
        If True, returns 2x2 matrices (single-deck).
        If False, returns 4x4 block-diagonal matrices (twin-deck).

    Returns:
    --------
    Ms : ndarray
        Mass matrix (2x2 or 4x4).
    Cs : ndarray
        Damping matrix (2x2 or 4x4).
    Ks : ndarray
        Stiffness matrix (2x2 or 4x4).
    """
    # Stiffness
    k1 = (2 * np.pi * f1) ** 2 * m1  #  (Vertical)
    k2 = (2 * np.pi * f2) ** 2 * m2  #  (Torsion)
    
    # Damping
    c1 = 2 * zeta * m1 * np.sqrt(k1 / m1)  # (Vertical)
    c2 = 2 * zeta * m2 * np.sqrt(k2 / m2)  # (Torsion)

    # Structural matrices, diagonal matrices in modal coordinates
    Ms = np.array([[m1,0],[0,m2]])  # Mass matrix
    Cs = np.array([[c1,0],[0,c2]])  # Damping matrix
    Ks = np.array([[k1,0],[0,k2]])  # Stiffness matrix

    if not single:
        Ms = np.array([[m1,0,0,0],[0,m2,0,0],[0,0,m1,0],[0,0,0,m2]])  # Mass matrix
        Cs = np.array([[c1,0,0,0],[0,c2,0,0],[0,0,c1,0],[0,0,0,c2]])  # Damping matrix
        Ks = np.array([[k1,0,0,0],[0,k2,0,0],[0,0,k1,0],[0,0,0,k2]])  # Stiffness matrix

    return Ms, Cs, Ks


def cae_kae_single(poly_coeff, Vred_global, B):
    """
    Evaluates the 8 aerodynamic derivatives for single-deck bridges.

    Parameters:
    -----------
    poly_coeff : ndarray, shape (8, 3)
        Polynomial coefficients for H1-H4 and A1-A4 (aerodynamic derivatives).
    V : float
        Reduced velocity (non-dimensional).
    B : float
        Section width (m).

    Returns:
    --------
    C_ae_star : ndarray, shape (2, 2)
        Dimensionless aerodynamic damping matrix.
    K_ae_star : ndarray, shape (2, 2)
        Dimensionless aerodynamic stiffness matrix.
    """

    Vred_global = float(Vred_global) 
    #Damping and stiffness matrices
    C_aeN_star = np.zeros((2, 2)) #Dimensionless
    K_aeN_star = np.zeros((2, 2)) #Dimensionless

    # AD
    H1, H2, H3, H4 = np.polyval(poly_coeff[0], Vred_global), np.polyval(poly_coeff[1][::-1], Vred_global), np.polyval(poly_coeff[2][::-1], Vred_global), np.polyval(poly_coeff[3][::-1], Vred_global)
    A1, A2, A3, A4 = np.polyval(poly_coeff[4], Vred_global), np.polyval(poly_coeff[5][::-1], Vred_global), np.polyval(poly_coeff[6][::-1], Vred_global), np.polyval(poly_coeff[7][::-1], Vred_global)

    C_aeN_star = np.array([
        [H1,       B * H2],
        [B * A1,   B**2 * A2]
    ])
    K_aeN_star = np.array([
        [H4,       B * H3],
        [B * A4,   B**2 * A3]
    ])
        
    return C_aeN_star, K_aeN_star




def cae_kae_twin(poly_coeff, Vred_global, B):
    """
    Evaluates all 32 aerodynamic derivatives at given reduced velocities.

    Parameters:
    -----------
    poly_coeff : ndarray, shape (32, 3)
        Polynomial coefficients for each aerodynamic derivative (2nd order).
    V : ndarray, shape (32,)
        Reduced velocity for each derivative.
    B : float
        Section width (m).

    Returns:
    --------
    C_ae_star : ndarray, shape (4, 4)
        Non-dimensional aerodynamic damping matrix.
    K_ae_star : ndarray, shape (4, 4)
        Non-dimensional aerodynamic stiffness matrix.
    """

    Vred_global = float(Vred_global) 
    # AD
    c_z1z1, c_z1θ1, c_z1z2, c_z1θ2 = np.polyval(poly_coeff[0][::-1],Vred_global), np.polyval(poly_coeff[1][::-1],Vred_global), np.polyval(poly_coeff[2][::-1],Vred_global), np.polyval(poly_coeff[3][::-1],Vred_global)
    c_θ1z1, c_θ1θ1, c_θ1z2, c_θ1θ2 = np.polyval(poly_coeff[4][::-1],Vred_global), np.polyval(poly_coeff[5][::-1],Vred_global), np.polyval(poly_coeff[6][::-1],Vred_global), np.polyval(poly_coeff[7][::-1],Vred_global)
    c_z2z1, c_z2θ1, c_z2z2, c_z2θ2 = np.polyval(poly_coeff[8][::-1],Vred_global), np.polyval(poly_coeff[9][::-1],Vred_global), np.polyval(poly_coeff[10][::-1],Vred_global), np.polyval(poly_coeff[11][::-1],Vred_global)
    c_θ2z1, c_θ2θ1, c_θ2z2, c_θ2θ2 = np.polyval(poly_coeff[12][::-1],Vred_global), np.polyval(poly_coeff[13][::-1],Vred_global), np.polyval(poly_coeff[14][::-1],Vred_global), np.polyval(poly_coeff[15][::-1],Vred_global)
    k_z1z1, k_z1θ1, k_z1z2, k_z1θ2 = np.polyval(poly_coeff[16][::-1],Vred_global), np.polyval(poly_coeff[17][::-1],Vred_global), np.polyval(poly_coeff[18][::-1],Vred_global), np.polyval(poly_coeff[19][::-1],Vred_global)
    k_θ1z1, k_θ1θ1, k_θ1z2, k_θ1θ2 = np.polyval(poly_coeff[20][::-1],Vred_global), np.polyval(poly_coeff[21][::-1],Vred_global), np.polyval(poly_coeff[22][::-1],Vred_global), np.polyval(poly_coeff[23][::-1],Vred_global)
    k_z2z1, k_z2θ1, k_z2z2, k_z2θ2 = np.polyval(poly_coeff[24][::-1],Vred_global), np.polyval(poly_coeff[25][::-1],Vred_global), np.polyval(poly_coeff[26][::-1],Vred_global), np.polyval(poly_coeff[27][::-1],Vred_global)
    k_θ2z1, k_θ2θ1, k_θ2z2, k_θ2θ2 = np.polyval(poly_coeff[28][::-1],Vred_global), np.polyval(poly_coeff[29][::-1],Vred_global), np.polyval(poly_coeff[30][::-1],Vred_global), np.polyval(poly_coeff[31][::-1],Vred_global)

    C_ae_star = np.array([
        [c_z1z1,       B * c_z1θ1,       c_z1z2,       B * c_z1θ2],
        [B * c_θ1z1,   B**2 * c_θ1θ1,   B * c_θ1z2,   B**2 * c_θ1θ2],
        [c_z2z1,       B * c_z2θ1,       c_z2z2,       B * c_z2θ2],
        [B * c_θ2z1,   B**2 * c_θ2θ1,   B * c_θ2z2,   B**2 * c_θ2θ2]
    ])

    K_ae_star = np.array([
        [k_z1z1,       B * k_z1θ1,       k_z1z2,       B * k_z1θ2],
        [B * k_θ1z1,   B**2 * k_θ1θ1,   B * k_θ1z2,   B**2 * k_θ1θ2],
        [k_z2z1,       B * k_z2θ1,       k_z2z2,       B * k_z2θ2],
        [B * k_θ2z1,   B**2 * k_θ2θ1,   B * k_θ2z2,   B**2 * k_θ2θ2]
    ])

    return C_ae_star, K_ae_star


def solve_omega(poly_coeff, m1, m2, f1, f2, B, rho, zeta, eps, N = 100, single = True):
    '''
    Solves flutter analysis using an iterative method for either single-deck (2DOF) or twin-deck (4DOF).
    Returns global results (and optionally local for twin-deck).

    Parameters
    ----------
    poly_coeff : ndarray
        Aerodynamic derivatives as polynomial coefficients (32, 3).

    m1, m2, f1, f2 : float
        Masses and natural frequencies for the two modes.
    B : float
        Bridge deck width.
    rho : float
        Air density.
    zeta : float
        Structural damping ratio.
    eps : float
        Convergence tolerance.
    N : int, optional
        Number of points in wind velocity range.
    single : bool, optional
        True for single-deck (2DOF), False for twin-deck (4DOF).

    Returns
    -------
    damping_ratios, omega_all, eigvals_all, eigvecs_all : list
        Global results for all modes.

    '''

    if single:
        Ms, Cs, Ks = structural_matrices(m1, m2, f1, f2, zeta, single = True)
        n_modes = 2 # 2 modes for single deck
        n = 2
    else:
        Ms, Cs, Ks = structural_matrices(m1, m2, f1, f2, zeta, single = False)
        n_modes = 4 # 4 modes for twin deck
        n = 4


    eigvals_all = np.empty((N, n_modes), dtype=complex)
    eigvecs_all = np.empty((N, n_modes), dtype=object) 
    damping_ratios  = np.empty((N, n_modes))
    omega_all = np.empty((N, n_modes))

    V_list = np.linspace(0, 180, N) #m/s

    #skip_mode = [False] * n_modes   # skip mode once flutter is detected

    for i, V in enumerate(V_list):
        if single:
            omega_old = np.array([2*np.pi*f1, 2*np.pi*f2])
        else:
            omega_old = np.array([2*np.pi*f1, 2*np.pi*f2, 2*np.pi*f1, 2*np.pi*f2]) #Først brudekke 1, deretter brudekke 2
        
        damping_old = [zeta]*n_modes
        eigvec_old = [None] * n_modes

        
       
        #print(f"Wind speed iteration {i+1}: V = {V} m/s")

        for j in range(n_modes): # 4 modes for twin deck, 2 modes for single deck
            #print(f"Mode iteration {j+1}")

            # if skip_mode[j]:
            #     omega_all[j].append(np.nan)
            #     damping_ratios[j].append(np.nan)
            #     eigvals_all[j].append(np.nan)
            #     eigvecs_all[j].append(np.nan)
            #     continue  # Go to next mode if flutter is detected

            if single:
                Vred_global = V/(omega_old[j]*B) # reduced velocity for global
            else:
                Vred_global = V/(omega_old[j]*B)

            converge = False
            while converge != True:
                #print("omega_ref", omega_old[j])

                if single:
                    C_star, K_star = cae_kae_single(poly_coeff, Vred_global, B)
                else:
                    C_star, K_star = cae_kae_twin(poly_coeff, Vred_global, B)

                C_aero = 0.5 * rho * B**2 * omega_old[j] * C_star
                K_aero = 0.5 * rho * B**2 * omega_old[j]**2 * K_star

               

                if np.isclose(V, 0.0): # still air
                    # Egenverdiene og egenvektorene kommer i riktig rekkefølge
                    # Save which DOFs that dominate the mode shape
                    if single: 
                        dominant_dofs = [0, 1]  # z, θ
                    else:
                        dominant_dofs = [0, 1, 2, 3]  # z1, θ1, z2, θ2

                    print("C_tot", Cs-C_aero)
                    print("K_tot", Ks-K_aero)
 
                    C_aero = np.zeros_like(Ms)  # Set aerodynamic damping to zero
                    K_aero = np.zeros_like(Ms)  # Set aerodynamic stiffness to zero

                    eigvals, eigvecs = solve_eigvalprob(Ms, Cs, Ks, C_aero, K_aero)
                    eigvals_pos = eigvals[np.imag(eigvals) > 0]
                    eigvecs_pos = eigvecs[:, np.imag(eigvals) > 0]

                    I = np.eye(Ms.shape[0])  # Identity matrix
                    lambda_j = -damping_old[j] * omega_old[j] + 1j * omega_old[j] * np.sqrt(1 - damping_old[j]**2)
                    omega_all[i,j]=omega_old[j]  # or np.nan
                    damping_ratios[i,j]=damping_old[j]  # or np.nan
                    eigvals_all[i,j]=lambda_j  # or np.nan
                    eigvecs_all[i,j]=eigvecs[:n, j] # or I[:, j]


                    converge = True
                else:
                    eigvals, eigvecs = solve_eigvalprob(Ms, Cs, Ks, C_aero, K_aero)

                    # Keep only the eigenvalues with positive imaginary part (complex conjugate pairs)
                    eigvals_pos = eigvals[np.imag(eigvals) > 0]
                    eigvecs_pos = eigvecs[:, np.imag(eigvals) > 0]  


                    # omega_pos = np.imag(eigvals_pos)
                    # damping_pos = -np.real(eigvals_pos) / np.abs(eigvals_pos)

                    # # Find correct mode j ??
                    # if eigvec_old[j] is None:
                    #     score = np.abs(omega_pos - omega_old[j]) +5 *np.abs(damping_pos - damping_old[j])
                    #     idx2 = np.argmin(score)
                    # else:
                    #     # Calculate similarity with previous eigenvector
                    #     # similarities = [np.abs(np.dot(eigvec_old[j].conj().T, eigvecs_pos[:, k])) for k in range(eigvecs_pos.shape[1])]
                    #     # idx2 = np.argmax(similarities)
                    #     # Calculate similarity with previous damping and frequency
                    #     score = np.abs(omega_pos - omega_old[j]) + 5*np.abs(damping_pos - damping_old[j]) # Kan kanskje vurdere å vekte de ulikt ??
                    #     idx2 = np.argmin(score)
                    
                    # score = (
                    #     np.abs(omega_pos - omega_old[j]) +
                    #     2 * np.abs(damping_pos - damping_old[j]) -
                    #     10 * np.abs(eigvecs_pos[dominant_dofs[j], :])
                    # )
                    # idx = np.argmin(score)

                    best_idx = None
                    max_val = -np.inf

                    # For mode j, finn den vektoren som ligner mest på tidligere dominant frihetsgrad
                    for idx in range(eigvecs_pos.shape[1]): 
                        mag = np.abs(eigvecs_pos[dominant_dofs[j], idx])
                        if mag > max_val:
                            max_val = mag
                            best_idx = idx

                    λj = eigvals_pos[best_idx]
                    φj = eigvecs_pos[:n, best_idx]
                        # eigvecs_pos[:, j] = 4 komponenter i single-deck → skyldes at du henter hele state-vektoen (inkl. hastighet)



                    # plt.plot(np.abs(φj), label=f"Mode {j+1} at V={V:.1f}")
                    # plt.legend()
                    # plt.show()

                    omega_new = np.imag(λj)
                    damping_new = -np.real(λj) / np.abs(λj)
                        
                    # Check if the mode is converged 
                    if np.abs(omega_new - omega_old[j]) < eps:
                        # # When flutter has occurred, we don't need to increase the speed further for this mode
                        # tmp_damping = damping_ratios[j] + [damping_new]
                        # sign_changes = np.where(np.diff(np.sign(tmp_damping)) != 0)[0]
                        # if len(sign_changes) >= 2: # Stop the plotting of mode when curve shifts after flutter. A little bit after the flutter speed.
                        #     omega_all[j].append(np.nan)
                        #     damping_ratios[j].append(np.nan)
                        #     eigvals_all[j].append(np.nan)
                        #     eigvecs_all[j].append(np.nan)
                        #     skip_mode[j] = True
                        # else: 
                        #     omega_all[j].append(omega_new)              
                        #     damping_ratios[j].append(damping_new)      
                        #     eigvals_all[j].append(λj)                   
                        #     eigvecs_all[j].append(φj) 

                        omega_all[i,j]=omega_new    
                        damping_ratios[i,j]=damping_new
                        eigvals_all[i,j]=λj                
                        eigvecs_all[i,j]=φj
                        converge = True              

                    else: # For the next iteration, use the new values as the old ones
                        omega_old[j] = omega_new
                        damping_old[j] = damping_new
                        eigvec_old[j] = φj


    return damping_ratios, omega_all, eigvals_all, eigvecs_all

def solve_flutter_speed(damping_ratios, N = 100, single = True):
    """
    Finds the flutter speed for each mode where the damping becomes negative.

    Parameters:
    -----------
    damping_ratios : list of arrays
        Damping ratios for each mode (shape: list of length n_modes with N elements each).
    N : int
        Number of wind speed steps.
    single : bool
        True if single-deck (2 modes), False if twin-deck (4 modes).

    Returns:
    --------
    flutter_speed_modes : list
        List of flutter speeds (None if not observed).
    V_list : ndarray
        Wind speed vector used in calculation.
    """
    n_modes = 2 if single else 4
    flutter_speed_modes = np.full(n_modes, np.nan)
    flutter_idx_modes = np.full(n_modes, np.nan)
    V_list = np.linspace(0, 180, N)

    for j in range(n_modes):
        for i, V in enumerate(V_list):
            if damping_ratios[i, j] < 0 and np.isnan(flutter_speed_modes[j]):
                flutter_speed_modes[j] = V
                flutter_idx_modes[j] = i
                break

    if np.all(np.isnan(flutter_speed_modes)):
        print("Ingen flutter observert for noen moder!")
        return None
    return flutter_speed_modes, flutter_idx_modes
     

     
def plot_damping_vs_wind_speed_single(B, Vred_defined, damping_ratios, omega_all,  dist="Fill in dist", N = 100, single = True):
    """
    Plot damping ratios as a function of wind speed, and mark AD-validity range.

    Parameters:
    -----------
    Vred_defined : list or array
        Reduced velocity validity intervals for ADs.
    damping_ratios : list of arrays
        Global damping ratios per mode.
    damping_ratios_local : list of arrays
        Local damping ratios per mode (only valid inside AD range).
    omega_all : list of arrays
        Global angular frequencies per mode.
    omega_all_local : list of arrays
        Local angular frequencies per mode (used in AD range).
    B : float
        Deck width.
    N : int
        Number of wind speed steps.
    single : bool
        Whether single-deck (2 modes) or twin-deck (4 modes).
    """

    V_list = np.linspace(0, 180, N)  # m/s
    linestyles = [(0,(5,10)), '--', ':', (0,(3,5,1,5))]
    colors = ['blue', 'red', 'green', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    plt.figure(figsize=(10, 6))

    if single:
        n_modes = 2 
        title = f"Damping vs. wind speed - {dist}"
    else:
        n_modes = 4
        title = f"Damping vs. wind speed - {dist}"

    for j in range(n_modes):
        target_value = np.min(Vred_defined[:, 1])
        idx = np.argmax(V_list / (omega_all[:, j] * B) >= target_value)
        plt.plot(V_list[:idx], damping_ratios[:idx, j], label=labels[j], color=colors[j],  linestyle=linestyles[j], linewidth=2.5)
        plt.plot(V_list, damping_ratios[:, j], color=colors[j], linestyle=linestyles[j], linewidth=2.5)

    plt.axhline(0, linestyle="--", color="grey", linewidth=1.1, label="Critical damping")
    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Damping ratio [-]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.ylim(-0.06, 0.06)
    plt.xlim(0,)
    plt.show()

def plot_frequency_vs_wind_speed(B, Vred_defined, omega_all, dist="Fill in dist", N = 100, single = True):
    """
    Plots natural frequencies as a function of wind speed, marking valid AD regions.

    Parameters:
    -----------
    Vred_defined : list or array
        Reduced velocity validity interval (e.g. from polynomial fit).
    omega_all : list of arrays
        Global angular frequencies (from V_list loop).
    omega_all_local : list of arrays
        Local angular frequencies (from Vred_local loop).
    B : float
        Deck width.
    N : int
        Number of points.
    single : bool
    """
    linestyles = [(0,(5,10)), '--', ':', (0,(3,5,1,5))]
    colors = ['blue', 'red', 'green', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    frequencies = omega_all / (2 * np.pi)       # Convert to Hz
    V_list = np.linspace(0, 180, N)             # Wind speed [m/s]

    plt.figure(figsize=(10, 6))

    if single:
        n_modes = 2 
        title = f"Natural frequencies vs wind speed - {dist}"
    else:
        n_modes = 4
        title = f"Natural frequencies vs wind speed - {dist}"

    for j in range(n_modes):
        target_value = np.min(Vred_defined[:, 1])
        idx = np.argmax(V_list / (omega_all[:, j] * B) >= target_value)
        plt.plot(V_list[:idx], frequencies[:idx, j], label=labels[j], color=colors[j],  linestyle=linestyles[j], linewidth=2.5)
        plt.plot(V_list, frequencies[:, j], color=colors[j], linestyle=linestyles[j], linewidth=2.5)

    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Natural frequency [Hz]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    #plt.ylim(0.12, 0.16)
    plt.xlim(0,150)
    plt.show()

def plot_flutter_mode_shape(eigvecs_all, flutter_idx_modes, dist="Fill in dist", single = True):
    
    flutter_idx = np.nanmin(flutter_idx_modes) 
    critical_mode = np.nanargmin(flutter_idx_modes)
    flutter_vec = eigvecs_all[int(flutter_idx),critical_mode] # Velg riktig mode ved flutter


    # 2 subplot: én for amplitudene, én for fasevinklene. De deler x-aksen
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    if single:
        dofs = ["V1", "T1"]  # DOF labels for single-deck
        n_modes = 2
    else:
        dofs = ["V1", "T1", "V2", "T2"]
        n_modes = 4      

    abs_vec = np.abs(flutter_vec) # absoluttverdien av alle komponentene i egenvektoren
    max_idx = np.argmax(abs_vec) # Finner hvilken komponent som har størst amplitude

    #  Normaliserer vektoren slik at den dominante komponenten blir 1 + 0j 
    normalized_vec = flutter_vec / flutter_vec[max_idx]
    magnitudes = np.abs(normalized_vec)
    phases = np.angle(normalized_vec, deg=True)

    dof_labels = [f"{dofs[i]}" for i in range(n_modes)]

    ax[0].bar(dof_labels, magnitudes, color='blue', width = 0.5)
    ax[1].bar(dof_labels, phases, color='orange', width = 0.5)

    ax[0].set_ylabel(r"|$\Phi$| [-]")
    ax[0].set_ylim(0, 1.1)
    ax[1].set_ylabel(r"∠$\Phi$ [deg]")
    ax[1].set_ylim(-180, 180)
    ax[1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    ax[0].set_title(f"Magnitude and phase of normalized eigenvector at flutter wind speed - {dist}", fontsize=14)
    ax[1].set_xlabel("DOFs")

        # for i in range(n_modes):
        #    ax[0].text(i, magnitudes[i] + 0.05, f"{magnitudes[i]:.2f}", ha='center', fontsize=9)
        #    ax[1].text(i, phases[i] + 10*np.sign(phases[i]), f"{phases[i]:.1f}°", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    if np.all(np.isnan(flutter_idx_modes)):
        print("Ingen flutter observert for noen moder!")
        return None



#def plot_compare_with_single():

#def plot_compare_with_dist():