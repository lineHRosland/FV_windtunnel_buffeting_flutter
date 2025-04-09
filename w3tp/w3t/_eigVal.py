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
        Ms = np.block([
        [Ms, np.zeros((2,2))],
        [np.zeros((2,2)), Ms]
                  ])
        Cs = np.block([
            [Cs, np.zeros((2,2))],
            [np.zeros((2,2)), Cs]
                    ])
        Ks = np.block([
            [Ks, np.zeros((2,2))],
            [np.zeros((2,2)), Ks]
                    ])

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
    H1, H2, H3, H4 = np.polyval(poly_coeff[0], Vred_global), np.polyval(poly_coeff[1], Vred_global), np.polyval(poly_coeff[2], Vred_global), np.polyval(poly_coeff[3], Vred_global)
    A1, A2, A3, A4 = np.polyval(poly_coeff[4], Vred_global), np.polyval(poly_coeff[5], Vred_global), np.polyval(poly_coeff[6], Vred_global), np.polyval(poly_coeff[7], Vred_global)

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
    c_z1z1, c_z1θ1, c_z1z2, c_z1θ2 = np.polyval(poly_coeff[0],Vred_global), np.polyval(poly_coeff[1],Vred_global), np.polyval(poly_coeff[2],Vred_global), np.polyval(poly_coeff[3],Vred_global)
    c_θ1z1, c_θ1θ1, c_θ1z2, c_θ1θ2 = np.polyval(poly_coeff[4],Vred_global), np.polyval(poly_coeff[5],Vred_global), np.polyval(poly_coeff[6],Vred_global), np.polyval(poly_coeff[7],Vred_global)
    c_z2z1, c_z2θ1, c_z2z2, c_z2θ2 = np.polyval(poly_coeff[8],Vred_global), np.polyval(poly_coeff[9],Vred_global), np.polyval(poly_coeff[10],Vred_global), np.polyval(poly_coeff[11],Vred_global)
    c_θ2z1, c_θ2θ1, c_θ2z2, c_θ2θ2 = np.polyval(poly_coeff[12],Vred_global), np.polyval(poly_coeff[13],Vred_global), np.polyval(poly_coeff[14],Vred_global), np.polyval(poly_coeff[15],Vred_global)
    k_z1z1, k_z1θ1, k_z1z2, k_z1θ2 = np.polyval(poly_coeff[16],Vred_global), np.polyval(poly_coeff[17],Vred_global), np.polyval(poly_coeff[18],Vred_global), np.polyval(poly_coeff[19],Vred_global)
    k_θ1z1, k_θ1θ1, k_θ1z2, k_θ1θ2 = np.polyval(poly_coeff[20],Vred_global), np.polyval(poly_coeff[21],Vred_global), np.polyval(poly_coeff[22],Vred_global), np.polyval(poly_coeff[23],Vred_global)
    k_z2z1, k_z2θ1, k_z2z2, k_z2θ2 = np.polyval(poly_coeff[24],Vred_global), np.polyval(poly_coeff[25],Vred_global), np.polyval(poly_coeff[26],Vred_global), np.polyval(poly_coeff[27],Vred_global)
    k_θ2z1, k_θ2θ1, k_θ2z2, k_θ2θ2 = np.polyval(poly_coeff[28],Vred_global), np.polyval(poly_coeff[29],Vred_global), np.polyval(poly_coeff[30],Vred_global), np.polyval(poly_coeff[31],Vred_global)

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
    V_list : np.ndarray
        Wind velocity range used for global solution.
    '''

    if single:
        Ms, Cs, Ks = structural_matrices(m1, m2, f1, f2, zeta, single = True)
        n_modes = 2 # 2 modes for single deck
    else:
        Ms, Cs, Ks = structural_matrices(m1, m2, f1, f2, zeta, single = False)
        n_modes = 4 # 4 modes for twin deck


    eigvals_all = [[] for _ in range(n_modes)]
    eigvecs_all = [[] for _ in range(n_modes)]
    damping_ratios = [[] for _ in range(n_modes)]
    omega_all = [[] for _ in range(n_modes)]

    V_list = np.linspace(0, 160, N) #m/s

    skip_mode = [False] * n_modes   # skip mode once flutter is detected

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

            if skip_mode[j]:
                omega_all[j].append(np.nan)
                damping_ratios[j].append(np.nan)
                eigvals_all[j].append(np.nan)
                eigvecs_all[j].append(np.nan)
                continue  # Go to next mode if flutter is detected

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

                #print("C_aero", C_aero)
                #print("K_aero", K_aero)
                
                eigvals, eigvecs = solve_eigvalprob(Ms, Cs, Ks, C_aero, K_aero)

                
                #print("eigvals", eigvals)

                if np.all(np.imag(eigvals) == 0):
                        print(f"At wind speed {V} m/s, no complex eigenvalues found for mode {j+1}.")
                        converge = True
                        I = np.eye(Ms.shape[0])  # Identity matrix
                        lambda_j = -damping_old[j] * omega_old[j] + 1j * omega_old[j] * np.sqrt(1 - damping_old[j]**2)

                        omega_all[j].append(omega_old[j])  # or np.nan
                        damping_ratios[j].append(damping_old[j])  # or np.nan
                        eigvals_all[j].append(lambda_j)  # or np.nan
                        eigvecs_all[j].append(I[:, j])
                else:
                    # Keep only the eigenvalues with positive imaginary part (complex conjugate pairs)
                    eigvals_pos = eigvals[np.imag(eigvals) > 0]
                    eigvecs_pos = eigvecs[:, np.imag(eigvals) > 0]  
                    omega_pos = np.imag(eigvals_pos)
                    damping_pos = -np.real(eigvals_pos) / np.abs(eigvals_pos)


                    # Find correct mode j ??
                    if eigvec_old[j] is None:
                        score = np.abs(omega_pos - omega_old[j]) +np.abs(damping_pos - damping_old[j])
                        idx = np.argmin(score)
                    else:
                        # Calculate similarity with previous eigenvector
                        similarities = [np.abs(np.dot(eigvec_old[j].conj().T, eigvecs_pos[:, k])) for k in range(eigvecs_pos.shape[1])]
                        idx = np.argmax(similarities)
                        # Calculate similarity with previous damping and frequency
                        score = np.abs(omega_pos - omega_old[j]) + np.abs(damping_pos - damping_old[j]) # Kan kanskje vurdere å vekte de ulikt ??
                        idx2 = np.argmin(score)
                    
                        # Check if the two checks selected different indices
                        print("idx", idx) 
                        print("idx2", idx2) 

                    λj = eigvals_pos[idx]
                    φj = eigvecs_pos[:, idx]

                    omega_new = np.imag(λj)
                    damping_new = -np.real(λj) / np.abs(λj)
                    
                    # Check if the mode is converged 
                    if np.abs(omega_new - omega_old[j]) < eps:
                        converge = True
                    
                        # When flutter has occurred, we don't need to increase the speed further for this mode
                        tmp_damping = damping_ratios[j] + [damping_new]
                        sign_changes = np.where(np.diff(np.sign(tmp_damping)) != 0)[0]
                        if len(sign_changes) >= 2: # Stop the plotting of mode when curve shifts after flutter. A little bit after the flutter speed.
                            omega_all[j].append(np.nan)
                            damping_ratios[j].append(np.nan)
                            eigvals_all[j].append(np.nan)
                            eigvecs_all[j].append(np.nan)
                            skip_mode[j] = True
                        else: 
                            omega_all[j].append(omega_new)              
                            damping_ratios[j].append(damping_new)      
                            eigvals_all[j].append(λj)                   
                            eigvecs_all[j].append(φj)                   

                    else: # For the next iteration, use the new values as the old ones
                        omega_old[j] = omega_new
                        damping_old[j] = damping_new
                        eigvec_old[j] = φj

    damping_array = np.array(damping_ratios).T #shape (N, n_modes)
    omega_array = np.array(omega_all).T
    eigvals_array = np.array(eigvals_all).T
    eigvecs_array = np.array(eigvecs_all, dtype=object).T

    return damping_array, omega_array, eigvals_array, eigvecs_array

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
    flutter_speed_modes = [None] * n_modes
    flutter_idx_modes = [None] * n_modes
    V_list = np.linspace(0, 160, N)

    for j in range(n_modes):
        for i, V in enumerate(V_list):
            if damping_ratios[i, j] < 0 and flutter_speed_modes[j] is None:
                flutter_speed_modes[j] = V
                flutter_idx_modes[j] = i
                break

    if all(fs is None for fs in flutter_speed_modes):
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

    V_list = np.linspace(0, 160, N)  # m/s
    linestyles = ['-', '--', ':', '-.']
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
        plt.plot(V_list[:idx], damping_ratios[:idx, j], label=labels[j], color=colors[j],  linestyle=linestyles[j], linewidth=1.5)
        plt.plot(V_list, damping_ratios[:, j], color=colors[j], linestyle=linestyles[j], linewidth=1.5)

    plt.axhline(0, linestyle="--", color="black", linewidth=1.1, label="Critical damping")
    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Damping ratio [-]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.ylim(-0.1, 0.1)
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
    linestyles = ['-', '--', ':', '-.']
    colors = ['blue', 'red', 'green', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    frequencies = omega_all / (2 * np.pi)       # Convert to Hz
    V_list = np.linspace(0, 160, N)             # Wind speed [m/s]

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
        plt.plot(V_list[:idx], frequencies[:idx, j], label=labels[j], color=colors[j],  linestyle=linestyles[j], linewidth=1.5)
        plt.plot(V_list, frequencies[:, j], color=colors[j], linestyle=linestyles[j], linewidth=1.5)

    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Natural frequency [Hz]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.ylim(0.12, 0.16)
    plt.show()

    ############################################

def plot_flutter_mode_shape(eigvecs_all, flutter_idx, dist="Fill in dist"):
    n_dof = len(eigvecs_all[0])  # Antall DOF (2 for single deck, 4 for twin deck)
    # DOF 0 og 1 = vertikal og torsjon for fremste bro
    # DOF 2 og 3 = vertikal og torsjon for bakre bro

    for j in range(n_dof):
        flutter_vec = eigvecs_all[flutter_idx[j],j]

    fig, ax = plt.subplots()
    
    # Plott hver komponent av egenvektoren
    dof_labels = [f"DOF {i+1}" for i in range(n_dof)]
    magnitudes = np.abs(flutter_vec)  # størrelsen på utslagene
    phases = np.angle(flutter_vec)    # faseforskjeller

    ax.bar(dof_labels, magnitudes, color='skyblue')
    ax.set_ylabel("Amplitude [-]")
    ax.set_title(f"Vibrasjonsform ved flutter - {dist}", fontsize=16)
    ax.grid(True)

    for i, phase in enumerate(phases):
        ax.text(i, magnitudes[i] + 0.01, f"fase={np.round(np.degrees(phase),1)}°", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()


#def plot_compare_with_single():

#def plot_compare_with_dist():

# def plot_flutter_mode_shape(phi, labels=['$z$', r'$\theta$'], title='Flutter mode shape'):
#     """
#     Plot vertical and torsional components of the flutter mode shape.

#     Parameters
#     ----------
#     phi : ndarray of shape (2,)
#         Eigenvector (real or complex) corresponding to flutter mode.
#     labels : list of str
#         Labels for z and θ components.
#     title : str
#         Plot title.
#     """
#     phi = np.array(phi).flatten()

#     # Normaliser til maks 1
#     phi_norm = phi / np.max(np.abs(phi))

#     z = phi_norm[0]
#     theta = phi_norm[1]

#     fig, ax = plt.subplots(figsize=(8,4))

#     # Plott realdel og imaginærdel
#     ax.bar([0, 1], [np.real(z), np.real(theta)], width=0.3, label='Real part', color='tab:blue', alpha=0.7)
#     ax.bar([0.4, 1.4], [np.imag(z), np.imag(theta)], width=0.3, label='Imag part', color='tab:orange', alpha=0.7)

#     ax.set_xticks([0.2, 1.2])
#     ax.set_xticklabels(labels, fontsize=13)
#     ax.set_ylabel("Amplitude (normalized)", fontsize=13)
#     ax.set_title(title, fontsize=14)
#     ax.legend()
#     ax.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()
# def plot_flutter_mode_shape_twin(phi, labels=['$z_1$', r'$\theta_1$', '$z_2$', r'$\theta_2$'], title='Flutter mode shape'):
#     """
#     Plot vertical and torsional components of the flutter mode shape for twin-deck.

#     Parameters
#     ----------
#     phi : ndarray of shape (4,)
#         Eigenvector (real or complex) corresponding to flutter mode.
#     labels : list of str
#         DOF labels.
#     title : str
#         Plot title.
#     """
#     phi = np.array(phi).flatten()
#     phi_norm = phi / np.max(np.abs(phi))

#     fig, ax = plt.subplots(figsize=(10,4))

#     ax.bar(np.arange(4)-0.15, np.real(phi_norm), width=0.3, label='Real part', color='tab:blue', alpha=0.7)
#     ax.bar(np.arange(4)+0.15, np.imag(phi_norm), width=0.3, label='Imag part', color='tab:orange', alpha=0.7)

#     ax.set_xticks(np.arange(4))
#     ax.set_xticklabels(labels, fontsize=13)
#     ax.set_ylabel("Amplitude (normalized)", fontsize=13)
#     ax.set_title(title, fontsize=14)
#     ax.legend()
#     ax.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()

# # Order DOFs [z1, tetha1, z2, tetha2]
# # Each eigenvec has 4 components, corresponding to the two modes of the two decks.

# def plot_mode_shape(phi, dist='TBD', title='Flutter mode shape', show_phase=True):
#     """
#     Plot vertical and torsional components of a flutter mode shape (eigenvector) for twin-deck bridge.

#     Parameters:
#     -----------
#     phi : ndarray
#         Eigenvector with 4 components [z1, θ1, z2, θ2]
#     dist : str
#         Deck separation label (e.g. "1D", "2D")
#     title : str
#         Plot title
#     show_phase : bool
#         Whether to plot real & imag parts separately (True) or just magnitude (False)
#     """
#     assert len(phi) == 4, "Eigenvector must have 4 components for twin-deck [z1, θ1, z2, θ2]"

#     # Normaliser slik at maksverdi = 1 (bruk modulus hvis kompleks)
#     scale = np.max(np.abs(phi))
#     phi_norm = phi / scale

#     # Pakk ut
#     z1, theta1, z2, theta2 = phi_norm

#     fig, axs = plt.subplots(1, 1, figsize=(8, 4))
#     x = ['z₁', 'θ₁', 'z₂', 'θ₂']
    
#     if show_phase:
#         axs.plot(x, np.real(phi_norm), 'o-', label='Real part')
#         axs.plot(x, np.imag(phi_norm), 's--', label='Imag part')
#     else:
#         axs.bar(x, np.abs(phi_norm), color='gray', label='Magnitude')

#     axs.axhline(0, color='black', linestyle='--', linewidth=0.8)
#     axs.set_ylabel('Normalized amplitude [-]')
#     axs.set_title(f"{title} ({dist})")
#     axs.legend()
#     axs.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # use:
# phi_flutter = eigvecs_all[j][-1]  # siste lagrede egenvektor for mode j
# plot_mode_shape(phi_flutter, dist='2D')


# def plot_flutter_mode_shape(eigvecs_all, damping_ratios, V_list, mode_labels=["z1", "θ1", "z2", "θ2"]):
#     """
#     Plot mode shape (eigenvector components) at flutter onset.

#     Parameters
#     ----------
#     eigvecs_all : list of arrays
#         List with eigenvectors (complex) per mode, length N.
#     damping_ratios : list of arrays
#         Corresponding damping values (real), same shape.
#     V_list : array
#         Wind speed values corresponding to each result.
#     mode_labels : list of str
#         Labels for the degrees of freedom, default: ["z1", "θ1", "z2", "θ2"].
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     n_modes = len(damping_ratios)
#     for j in range(n_modes):
#         damping = np.array(damping_ratios[j])
#         flutter_idx = np.argmax(damping < 0)  # Første gang dempingen går under null
#         if flutter_idx == 0:
#             continue  # hopp over hvis den er negativ helt fra start

#         φ_flutter = eigvecs_all[j][flutter_idx]
#         φ_flutter = φ_flutter / np.max(np.abs(φ_flutter))  # normaliser til maks = 1

#         plt.figure(figsize=(8, 5))
#         plt.plot(np.real(φ_flutter), 'o-', label='Real part')
#         plt.plot(np.imag(φ_flutter), 's--', label='Imag part')
#         plt.xticks(np.arange(len(φ_flutter)), mode_labels)
#         plt.title(f"Mode shape at flutter (mode {j+1}) – V = {V_list[flutter_idx]:.2f} m/s")
#         plt.ylabel("Normalized amplitude")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
# def plot_mode_shape_bar(phi, mode_labels=["z1", "θ1", "z2", "θ2"], title="Flutter mode shape"):
#     """
#     Bar plot for mode shape (real and imaginary parts).

#     Parameters
#     ----------
#     phi : array
#         Complex eigenvector.
#     mode_labels : list
#         Labels for each degree of freedom.
#     """
#     phi = phi / np.max(np.abs(phi))  # normalize
#     real = np.real(phi)
#     imag = np.imag(phi)

#     x = np.arange(len(phi))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.bar(x - width/2, real, width, label='Real part', alpha=0.7)
#     ax.bar(x + width/2, imag, width, label='Imag part', alpha=0.7)
    
#     ax.set_xticks(x)
#     ax.set_xticklabels(mode_labels)
#     ax.set_ylabel("Normalized amplitude")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()