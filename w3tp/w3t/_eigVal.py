# -*- coding: utf-8 -*-
"""
Created in April 2025

@author: linehro
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def Uf(Um, scale):
    """
    The function takes the model wind speed and the scale factor as inputs 
    and returns the full scale wind speed.
    """
    return Um * 1/np.sqrt(scale)


def solve_eigvalprob(M_struc, C_struc, K_struc, C_aero, K_aero):
    """
    Løser generalisert eigenverdiproblem for gitt system.
    
    [ -λ² M + λ(C + Cae) + (K + Kae) ] φ = 0

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
    n = M_struc.shape[0] # n = 2: single deck, n = 4: twin deck

    
    print
    A = -la.block_diag(M_struc, np.eye(n))
    B = np.block([
        [C_struc + C_aero, K_struc + K_aero],
        [-np.eye(n), np.zeros((n, n))]
    ])

    eigvals, eigvec = la.eig(B, A)
    return eigvals, eigvec

def structural_matrices_single(m1, m2, f1, f2, zeta):
    """
    Calculate structural matrices for a two-degree-of-freedom system.

    Parameters:
    -----------
    m1 : float
        Mass of the first degree of freedom (kg).
    m2 : float
        Mass of the second degree of freedom (kg).
    f1 : float
        Natural frequency of the first degree of freedom (Hz).
    f2 : float
        Natural frequency of the second degree of freedom (Hz).
    zeta : float
        Damping ratio.

    Returns:
    --------
    Ms : ndarray of shape (N, 2, 2)
        Mass matrix.
    Ks : ndarray of shape (N, 2, 2)
        Stiffness matrix.
    Cs : ndarray of shape (N, 2, 2)
        Damping matrix.
    """
    # Stiffness
    k1 = (2 * np.pi * f1) ** 2 * m1  #  (Vertical)
    k2 = (2 * np.pi * f2) ** 2 * m2  #  (Torsion)

    # Damping
    c1 = 2 * zeta * m1 * np.sqrt(k1 * m1)  # (Vertical)
    c2 = 2 * zeta * m2 * np.sqrt(k2 * m2)  # (Torsion)

    # Structural matrices, diagonal matrices in modal coordinates
    Ms = np.array([[m1,0],[0,m2]])  # Mass matrix
    Cs = np.array([[c1,0],[0,c2]])  # Damping matrix
    Ks = np.array([[k1,0],[0,k2]])  # Stiffness matrix


    return Ms, Cs, Ks


def cae_kae_single(poly_coeff, V, B):
    """structural_matrices
    Evaluates all 8 aerodynamic derivatives individually across their own reduced velocity range.

    Parameters:
    -----------
    poly_coeff : ndarray
        Polynomial coefficients, shape (8, 3)
    v_all : ndarray
        Reduced velocity range per AD, shape (8, 2)
    B : float
        Section width (m)
    f : float
        Frequency (Hz)
        Gathered from FEM-model
    rho : float
        Air density (kg/m^3)
    N : int
        Number of points per AD
    Returns:
    --------
    C_all : ndarray of shape (N, 2, 2)
        Aerodynamic damping matrices
    K_all : ndarray of shape (N, 2, 2)
        Aerodynamic stiffness matrices
    """

    
    #Damping and stiffness matrices
    C_aeN_star = np.zeros((2, 2)) #Dimensionless
    K_aeN_star = np.zeros((2, 2)) #Dimensionless

    # AD
    H1, H2, H3, H4 = np.polyval(poly_coeff[0], V), np.polyval(poly_coeff[1], V), np.polyval(poly_coeff[2], V), np.polyval(poly_coeff[3], V)
    A1, A2, A3, A4 = np.polyval(poly_coeff[4], V), np.polyval(poly_coeff[5], V), np.polyval(poly_coeff[6], V), np.polyval(poly_coeff[7], V)

    C_aeN_star = np.array([
        [H1,       B * H2],
        [B * A1,   B**2 * A2]
    ])
    K_aeN_star = np.array([
        [H4,       B * H3],
        [B * A4,   B**2 * A3]
    ])
        
    return C_aeN_star, K_aeN_star

def solve_flutter_single(poly_coeff, v_all, m1, m2, f1, f2, B, rho, zeta, max_iter, eps, N = 100):
    """
    Parameters:
    -----------
    M_struc : ndarray
        Structural mass matrix
    C_struc : ndarray
        Structural damping matrix
    K_struc : ndarray
        Structural stiffness matrix
    C_aero : ndarray of shape (N, n, n)
        Aerodynamic damping matrices (per wind speed)
    K_aero : ndarray of shape (N, n, n)
        Aerodynamic stiffness matrices (per wind speed)
    V_all : ndarray of shape (1, N) or (N,)
        Wind speeds corresponding to C_aero/K_aero
    """
    Ms, Cs, Ks = structural_matrices_single(m1, m2, f1, f2, zeta)

    # Sjekk at alle AD-er har samme hastighetsintervall
    if not np.allclose(v_all, v_all[0], atol=1e-8):
        raise ValueError("OBS: AD-ene har forskjellige reduced velocity-intervaller!"
                         "Denne funksjonen forutsetter at alle bruker samme v_range.")
           
    Vred_list = np.linspace(v_all[0][0] ,v_all[0][1], N)
    
    eigvals_all = []
    eigvecs_all = []
    damping_ratios = []
    omega=[]
    flutter_speed = None

    for i, V in enumerate(Vred_list):
        omega_old = 2* np.pi * f1 #Angular frequency (rad/s) ??f1

        for _ in range(max_iter):
            #Beregn nye Cae og Kae for denne omega
            C_aero_single_star, K_aero_single_star  = cae_kae_single(poly_coeff, V, B)
            C_aero_single_iter = 0.5 * rho * B**2*omega_old*C_aero_single_star
            K_aero_single_iter = 0.5 * rho * B**2*omega_old**2*K_aero_single_star

            eigvals, eigvec = solve_eigvalprob(Ms, Cs, Ks, C_aero_single_iter, K_aero_single_iter)
            eigvals_all.append(eigvals)
            eigvecs_all.append(eigvec)


            # Finn ny omega fra kritisk mode (minste demping)
            damping = -np.real(eigvals) / np.abs(eigvals)
            idx = np.argmin(damping)
            omega_new = np.abs(np.imag(eigvals[idx]))


            # Brudd dersom konvergert
            if np.abs(omega_new - omega_old) < eps:
                min_damping = np.min(damping) # Mest utstabile mode
                omega.append(omega_new)
                break
            else: omega_old = omega_new

        damping_ratios.append(damping)

        if min_damping < 0 and flutter_speed is None:
            flutter_speed = V  # Første gang vi får negativ demping


    if flutter_speed is None:
        print("Ingen flutter observert i gitt vindhastighetsintervall!")


    return flutter_speed, damping_ratios, omega, eigvals_all, eigvecs_all, Vred_list

def plot_damping_vs_wind_speed_single(flutter_speed, Vred_list, damping_ratios, omega, B):
    """
    Plott dempingforhold som funksjon av vindhastighet, og marker flutterhastigheten.
    
    Parameters:
    ----------
    flutter_speed : float
        Flutterhastighet (kritisk vindhastighet).
    Vred_list : ndarray
        Liste over vindhastigheter (reduced velocities).
    damping_ratios : list
        Liste over dempingsforhold for hver vindhastighet.
    """
    plt.figure(figsize=(10, 6))

    if flutter_speed is not None:
        plt.scatter(flutter_speed, 0, color='r', label=f"Flutter Speed: {flutter_speed*omega*B:.2f} m/s")
    
    plt.plot(Vred_list*omega*B, damping_ratios[:,0], label="Damping Ratio 1", color='blue')
    plt.plot(Vred_list*omega*B, damping_ratios[:,1], label="Damping Ratio 2", color='red')
    plt.plot(Vred_list*omega*B, damping_ratios[:,2], label="Damping Ratio 3", color='yellow')
    plt.plot(Vred_list*omega*B, damping_ratios[:,3], label="Damping Ratio 4", color='green')

    plt.axhline(0, linestyle="--", color="gray", label="Critical Damping Line")
    plt.xlabel("Vindhastighet [m/s]")
    plt.ylabel("Dempingforhold")
    plt.title("Flutteranalyse: Dempingforhold i funksjon av vindhastighet")
    plt.legend()
    plt.grid(True)
    plt.show()





def plot_frequency_vs_wind_speed_singles(Vred_list, omega, B):
    """
    Plotter frekvens og demping for hver mode over vindhastighet.

    Parameters:
    -----------
    Vred_list : array
        Liste over reduserte vindhastigheter.
    eigvals_all : array
        Array av egenverdier med shape (N, n_modes).
    """

    frequencies = omega/(np.pi*2)  # Frekvens i Hz
  
    plt.figure(figsize=(10, 6))


    plt.plot(Vred_list*omega*B, frequencies[:,0], label="Freq 1", color='blue')
    plt.plot(Vred_list*omega*B, frequencies[:, 1], label="Freq 2", color='red')
    plt.plot(Vred_list*omega*B, frequencies[:, 2], label="Freq 3", color='yellow')
    plt.plot(Vred_list*omega*B, frequencies[:, 3], label="Freq 4", color='green')

    plt.xlabel("Vindhastighet [m/s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend()
    plt.grid(True)
    plt.show()

def structural_matrices_twin(m1, m2, f1, f2, zeta):
    """
    Calculate structural matrices for a two-degree-of-freedom system.

    Parameters:
    -----------
    m1 : float
        Mass of the first degree of freedom (kg).
    m2 : float
        Mass of the second degree of freedom (kg).
    f1 : float
        Natural frequency of the first degree of freedom (Hz).
    f2 : float
        Natural frequency of the second degree of freedom (Hz).
    zeta : float
        Damping ratio.

    Returns:
    --------
    Ms : ndarray of shape (N, 4, 4)
        Mass matrix.
    Ks : ndarray of shape (N, 4, 4)
        Stiffness matrix.
    Cs : ndarray of shape (N, 4, 4)
        Damping matrix.
    """
    # Stiffness
    k1 = (2 * np.pi * f1) ** 2 * m1  #  (Vertical)
    k2 = (2 * np.pi * f2) ** 2 * m2  #  (Torsion)

    # Damping
    c1 = 2 * zeta * m1 * np.sqrt(k1 * m1)  # (Vertical)
    c2 = 2 * zeta * m2 * np.sqrt(k2 * m2)  # (Torsion)

    # One bridge deck, two degrees of freedom
    Ms_single = np.diag([m1, m2])  # Mass matrix
    Cs_single = np.diag([c1, c2])  # Damping matrix
    Ks_single = np.diag([k1, k2])  # Stiffness matrix


    Ms = np.block([
        [Ms_single, np.zeroes((2,2))],
        [np.zeros((2,2)), Ms_single]
                  ])
    Cs = np.block([
        [Cs_single, np.zeros((2,2))],
        [np.zeros((2,2)), Cs_single]
                  ])
    Ks = np.block([
        [Ks_single, np.zeros((2,2))],
        [np.zeros((2,2)), Ks_single]
                  ])
    
    return Ms, Cs, Ks

def cae_kae_twin(poly_coeff, V, B):
    """
    Evaluates all 32 aerodynamic derivatives individually across their own reduced velocity range.

    Parameters:
    -----------
    poly_coeff : ndarray
        Polynomial coefficients, shape (32, 3)
    v_range : ndarray
        Reduced velocity range per AD, shape (32, 2)
    B : float
        Section width (m)
    f : float
        Frequency (Hz)
        Gathered from FEM-model
    rho : float
        Air density (kg/m^3)
    
    N : int
        Number of points per AD
    Returns:
    --------
    C_all : ndarray of shape (N, 2, 2)
        Aerodynamic damping matrices
    K_all : ndarray of shape (N, 2, 2)
        Aerodynamic stiffness matrices
    """
    


    # AD
    c_z1z1, c_z1θ1, c_z1z2, c_z1θ2 = np.polyval(poly_coeff[0],V[0]), np.polyval(poly_coeff[1],V[1]), np.polyval(poly_coeff[2],V[2]), np.polyval(poly_coeff[3],V[3])
    c_θ1z1, c_θ1θ1, c_θ1z2, c_θ1θ2 = np.polyval(poly_coeff[4],V[4]), np.polyval(poly_coeff[5],V[5]), np.polyval(poly_coeff[6],V[6]), np.polyval(poly_coeff[7],V[7])
    c_z2z1, c_z2θ1, c_z2z2, c_z2θ2 = np.polyval(poly_coeff[8],V[8]), np.polyval(poly_coeff[9],V[9]), np.polyval(poly_coeff[10],V[10]), np.polyval(poly_coeff[11],V[11])
    c_θ2z1, c_θ2θ1, c_θ2z2, c_θ2θ2 = np.polyval(poly_coeff[12],V[12]), np.polyval(poly_coeff[13],V[13]), np.polyval(poly_coeff[14],V[14]), np.polyval(poly_coeff[15],V[15])
    k_z1z1, k_z1θ1, k_z1z2, k_z1θ2 = np.polyval(poly_coeff[16],V[16]), np.polyval(poly_coeff[17],V[17]), np.polyval(poly_coeff[18],V[18]), np.polyval(poly_coeff[19],V[19])
    k_θ1z1, k_θ1θ1, k_θ1z2, k_θ1θ2 = np.polyval(poly_coeff[20],V[20]), np.polyval(poly_coeff[21],V[21]), np.polyval(poly_coeff[22],V[22]), np.polyval(poly_coeff[23],V[23])
    k_z2z1, k_z2θ1, k_z2z2, k_z2θ2 = np.polyval(poly_coeff[24],V[24]), np.polyval(poly_coeff[25],V[25]), np.polyval(poly_coeff[26],V[26]), np.polyval(poly_coeff[27],V[27])
    k_θ2z1, k_θ2θ1, k_θ2z2, k_θ2θ2 = np.polyval(poly_coeff[28],V[28]), np.polyval(poly_coeff[29],V[29]), np.polyval(poly_coeff[30],V[30]), np.polyval(poly_coeff[31],V[31])

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

def solve_flutter_twin(poly_coeff, v_all, m1, m2, f1, f2, B, rho, zeta, max_iter, eps, N = 100):
    """
    Parameters:
    -----------
    M_struc : ndarray
        Structural mass matrix
    C_struc : ndarray
        Structural damping matrix
    K_struc : ndarray
        Structural stiffness matrix
    C_aero : ndarray of shape (N, n, n)
        Aerodynamic damping matrices (per wind speed)
    K_aero : ndarray of shape (N, n, n)
        Aerodynamic stiffness matrices (per wind speed)
    V_all : ndarray of shape (1, N) or (N,)
        Wind speeds corresponding to C_aero/K_aero
    """
    Ms, Cs, Ks = structural_matrices_twin(m1, m2, f1, f2, zeta)

    # Sjekk at alle AD-er har samme hastighetsintervall
    if not np.allclose(v_all, v_all[0], atol=1e-8):
        print("OBS: AD-ene har forskjellige reduced velocity-intervaller! ")
        print("Global differs local, where local is the shortest interval")
       

    vmin_global = np.min(v_all[:, 0])
    vmax_global = np.max(v_all[:, 1])
    V_common = np.linspace(vmax_global, vmin_global, N)
    V_local = np.zeros((32, N))
    V_global = np.zeros((32, N))
    for i in range(32):
        v_min_red, v_max_red = v_all[i]
        V_local[i] = np.linspace(v_min_red ,v_max_red, N)
        V_global[i] = V_common

    
    eigvals_all_local = []
    eigvecs_all_local = []
    damping_ratios_local = []
    flutter_speed_local = None

    eigvals_all_global = []
    eigvecs_all_global = []
    damping_ratios_global = []
    flutter_speed_global = None

    #Global
    for i, V in enumerate(V_local):

        omega_old_local = 2* np.pi * f1 #Angular frequency (rad/s) ??f1

        for _ in range(max_iter):
            #Beregn nye Cae og Kae for denne omega
            C_aero_twin_star_local, K_aero_twin_star_local  = cae_kae_twin(poly_coeff, V, B)
            C_aero_twin__local = 0.5 * rho * B**2*omega_old_local*C_aero_twin_star_local
            K_aero_twin_local = 0.5 * rho * B**2*omega_old_local**2*K_aero_twin_star_local

            eigvals_local, eigvec_local = solve_eigvalprob(Ms, Cs, Ks, C_aero_twin__local, K_aero_twin_local)
            eigvals_all_local.append(eigvals_local)
            eigvecs_all_local.append(eigvec_local)


            # Finn ny omega fra kritisk mode (minste demping)
            damping_local = -np.real(eigvals_local) / np.abs(eigvals_local)
            idx_local = np.argmin(damping_local)
            omega_new_local = np.abs(np.imag(eigvals_local[idx_local]))


            # Brudd dersom konvergert
            if np.abs(omega_new_local - omega_old_local) < eps:
                min_damping_local = np.min(damping_local)
                break
            else: omega_old_local = omega_new_local

        damping_ratios_local.append(damping_local)

        if min_damping_local < 0 and flutter_speed_local is None:
            flutter_speed_local = V  # Første gang vi får negativ demping

        print("eigvals_local", eigvals_local)
        print("eigvec_local", eigvec_local)        
        print("damping_local", damping_local)
        print("min_damping_local", min_damping_local)
    
    for i, V in enumerate(V_global):

        omega_old_global = 2* np.pi * f1

        for _ in range(max_iter):
            #Beregn nye Cae og Kae for denne omega
            C_aero_twin_star_global, K_aero_twin_star_global  = cae_kae_twin(poly_coeff, V, B)
            C_aero_twin__global = 0.5 * rho * B**2*omega_old_global*C_aero_twin_star_global
            K_aero_twin_global = 0.5 * rho * B**2*omega_old_global**2*K_aero_twin_star_global

            eigvals_global, eigvec_global = solve_eigvalprob(Ms, Cs, Ks, C_aero_twin__global, K_aero_twin_global)
            eigvals_all_global.append(eigvals_global)
            eigvecs_all_global.append(eigvec_global)


            # Finn ny omega fra kritisk mode (minste demping)
            damping_global = -np.real(eigvals_global) / np.abs(eigvals_global)
            idx_global = np.argmin(damping_global)
            omega_new_global = np.abs(np.imag(eigvals_global[idx_global]))


            # Brudd dersom konvergert
            if np.abs(omega_new_global - omega_old_global) < eps:
                min_damping_global = np.min(damping_global)
                break
            else: omega_old_global = omega_new_global
        
        damping_ratios_global.append(damping_global)

        if min_damping_global < 0 and flutter_speed_global is None:
            flutter_speed_global = V
        
        print("eigvals_global", eigvals_global)
        print("eigvec_global", eigvec_global)
        print("damping_global", damping_global)
        print("min_damping_global", min_damping_global)


    if flutter_speed_local is None:
        print("Ingen flutter observert i gitt vindhastighetsintervall!")
    
    if flutter_speed_global is None:
        print("Ingen flutter observert i gitt vindhastighetsintervall!")


    return flutter_speed_local, damping_ratios_local, eigvals_all_local, eigvecs_all_local, V_local, flutter_speed_global, damping_ratios_global, eigvals_all_global, eigvecs_all_global, V_global
