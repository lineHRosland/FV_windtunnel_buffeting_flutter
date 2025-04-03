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

    print("Ms", M_struc)
    print("Cs", C_struc)
    print("Ks", K_struc)

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
    min_damping_ratios = []
    damping_ratios = []
    flutter_speed = None

    for i, V in enumerate(Vred_list):
        # V = Vred_list[i]  

        omega_old = 2* np.pi * f1 #Angular frequency (rad/s) ??f1

        for _ in range(max_iter):
            #Beregn nye Cae og Kae for denne omega
            C_aero_single_star, K_aero_single_star  = cae_kae_single(poly_coeff, V, B)
            C_aero_single_iter = 0.5 * rho * B**2*omega_old*C_aero_single_star
            K_aero_single_iter = 0.5 * rho * B**2*omega_old**2*K_aero_single_star

            eigvals, eigvec = solve_eigvalprob(Ms[i], Cs[i], Ks[i], C_aero_single_iter[i], K_aero_single_iter[i])
            eigvals_all.append(eigvals)
            eigvecs_all.append(eigvec)


            # Finn ny omega fra kritisk mode (minste demping)
            damping = -np.real(eigvals) / np.abs(eigvals)
            idx = np.argmin(damping)
            omega_new = np.abs(np.imag(eigvals[idx]))


            # Brudd dersom konvergert
            if np.abs(omega_new - omega_old) < eps:
                min_damping = np.min(damping) # Mest utstabile mode
                break
            else: omega_old = omega_new

        damping_ratios.append(damping)
        min_damping_ratios.append(min_damping)   

        if min_damping < 0 and flutter_speed is None:
            flutter_speed = V  # Første gang vi får negativ demping

        print("eigvals", eigvals)
        print("eigvec", eigvec)        
        print("damping", damping)
        print("min_damping", min_damping)

    if flutter_speed is None:
        print("Ingen flutter observert i gitt vindhastighetsintervall!")


    return flutter_speed, damping_ratios, min_damping_ratios, eigvals_all, eigvecs_all, Vred_list

def plot_damping_vs_wind_speed_single(flutter_speed, Vred_list, damping_ratios):
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
    plt.plot(Vred_list, damping_ratios, label="Damping Ratio", color='b')
    plt.axhline(0, linestyle="--", color="gray", label="Critical Damping Line")
    plt.scatter(flutter_speed, 0, color='r', label=f"Flutter Speed: {flutter_speed:.2f} m/s")
    plt.xlabel("Vindhastighet [m/s]")
    plt.ylabel("Dempingforhold")
    plt.title("Flutteranalyse: Dempingforhold i funksjon av vindhastighet")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_eigenvalues_over_v(Vred_list, eigvals_all):
    """
    Plott egenverdier (reelle og imaginære deler) i funksjon av vindhastighet.
    
    Parameters:
    ----------
    Vred_list : ndarray
        Liste over vindhastigheter (reduced velocities).
    eigvals_all : list
        Liste med alle egenverdier for hver hastighet (output fra solve_eigvalprob).
    """
    plt.figure(figsize=(10, 6))
    
    # Ekstraher reelle og imaginære deler av egenverdiene
    real_parts = [np.real(eigvals) for eigvals in eigvals_all]
    imag_parts = [np.imag(eigvals) for eigvals in eigvals_all]

    # Plot reelle og imaginære deler
    plt.plot(Vred_list, real_parts, label="Reelle deler", color='b')
    plt.plot(Vred_list, imag_parts, label="Imaginære deler", color='r', linestyle='--')
    
    plt.xlabel("Vindhastighet [m/s]")
    plt.ylabel("Egenverdi (Reell og Imaginær del)")
    plt.title("Egenverdier i funksjon av vindhastighet")
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

def cae_kae_twin(poly_coeff, V_loc, V_glob, B):
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
    
    
    #Damping and stiffness matrices
    C_ae_local_star = np.zeros((4, 4))
    K_ae_local_star = np.zeros((4, 4))
    C_ae_global_star = np.zeros((4, 4))
    K_ae_global_star = np.zeros((4, 4))

    # AD
    c_z1z1, c_z1θ1, c_z1z2, c_z1θ2 = np.polyval(poly_coeff[0],V_loc), np.polyval(poly_coeff[1],V_loc), np.polyval(poly_coeff[2],V_loc), np.polyval(poly_coeff[3],V_loc)
    c_θ1z1, c_θ1θ1, c_θ1z2, c_θ1θ2 = np.polyval(poly_coeff[4],V_loc), np.polyval(poly_coeff[5],V_loc), np.polyval(poly_coeff[6],V_loc), np.polyval(poly_coeff[7],V_loc)
    c_z2z1, c_z2θ1, c_z2z2, c_z2θ2 = np.polyval(poly_coeff[8],V_loc), np.polyval(poly_coeff[9],V_loc), np.polyval(poly_coeff[10],V_loc), np.polyval(poly_coeff[11],V_loc)
    c_θ2z1, c_θ2θ1, c_θ2z2, c_θ2θ2 = np.polyval(poly_coeff[12],V_loc), np.polyval(poly_coeff[13],V_loc), np.polyval(poly_coeff[14],V_loc), np.polyval(poly_coeff[15],V_loc)
    k_z1z1, k_z1θ1, k_z1z2, k_z1θ2 = np.polyval(poly_coeff[16],V_loc), np.polyval(poly_coeff[17],V_loc), np.polyval(poly_coeff[18],V_loc), np.polyval(poly_coeff[19],V_loc)
    k_θ1z1, k_θ1θ1, k_θ1z2, k_θ1θ2 = np.polyval(poly_coeff[20],V_loc), np.polyval(poly_coeff[21],V_loc), np.polyval(poly_coeff[22],V_loc), np.polyval(poly_coeff[23],V_loc)
    k_z2z1, k_z2θ1, k_z2z2, k_z2θ2 = np.polyval(poly_coeff[24],V_loc), np.polyval(poly_coeff[25],V_loc), np.polyval(poly_coeff[26],V_loc), np.polyval(poly_coeff[27],V_loc)
    k_θ2z1, k_θ2θ1, k_θ2z2, k_θ2θ2 = np.polyval(poly_coeff[28],V_loc), np.polyval(poly_coeff[29],V_loc), np.polyval(poly_coeff[30],V_loc), np.polyval(poly_coeff[31],V_loc)

    # AD_global
    c_z1z1_glob, c_z1θ1_glob, c_z1z2_glob, c_z1θ2_glob = np.polyval(poly_coeff[0],V_glob), np.polyval(poly_coeff[1],V_glob), np.polyval(poly_coeff[2],V_glob), np.polyval(poly_coeff[3],V_glob)
    c_θ1z1_glob, c_θ1θ1_glob, c_θ1z2_glob, c_θ1θ2_glob = np.polyval(poly_coeff[4],V_glob), np.polyval(poly_coeff[5],V_glob), np.polyval(poly_coeff[6],V_glob), np.polyval(poly_coeff[7],V_glob)
    c_z2z1_glob, c_z2θ1_glob, c_z2z2_glob, c_z2θ2_glob = np.polyval(poly_coeff[8],V_glob), np.polyval(poly_coeff[9],V_glob), np.polyval(poly_coeff[10],V_glob), np.polyval(poly_coeff[11],V_glob)
    c_θ2z1_glob, c_θ2θ1_glob, c_θ2z2_glob, c_θ2θ2_glob = np.polyval(poly_coeff[12],V_glob), np.polyval(poly_coeff[13],V_glob), np.polyval(poly_coeff[14],V_glob), np.polyval(poly_coeff[15],V_glob)
    k_z1z1_glob, k_z1θ1_glob, k_z1z2_glob, k_z1θ2_glob = np.polyval(poly_coeff[16],V_glob), np.polyval(poly_coeff[17],V_glob), np.polyval(poly_coeff[18],V_glob), np.polyval(poly_coeff[19],V_glob)
    k_θ1z1_glob, k_θ1θ1_glob, k_θ1z2_glob, k_θ1θ2_glob = np.polyval(poly_coeff[20],V_glob), np.polyval(poly_coeff[21],V_glob), np.polyval(poly_coeff[22],V_glob), np.polyval(poly_coeff[23],V_glob)
    k_z2z1_glob, k_z2θ1_glob, k_z2z2_glob, k_z2θ2_glob = np.polyval(poly_coeff[24],V_glob), np.polyval(poly_coeff[25],V_glob), np.polyval(poly_coeff[26],V_glob), np.polyval(poly_coeff[27],V_glob)
    k_θ2z1_glob, k_θ2θ1_glob, k_θ2z2_glob, k_θ2θ2_glob = np.polyval(poly_coeff[28],V_glob), np.polyval(poly_coeff[29],V_glob), np.polyval(poly_coeff[30],V_glob), np.polyval(poly_coeff[31],V_glob)

    C_ae_global_star = np.array([
        [c_z1z1_glob,       B * c_z1θ1_glob,       c_z1z2_glob,       B * c_z1θ2_glob],
        [B * c_θ1z1_glob,   B**2 * c_θ1θ1_glob,   B * c_θ1z2_glob,   B**2 * c_θ1θ2_glob],
        [c_z2z1_glob,       B * c_z2θ1_glob,       c_z2z2_glob,       B * c_z2θ2_glob],
        [B * c_θ2z1_glob,   B**2 * c_θ2θ1_glob,   B * c_θ2z2_glob,   B**2 * c_θ2θ2_glob]
    ])
    K_ae_global_star = np.array([
        [k_z1z1_glob,       B * k_z1θ1_glob,       k_z1z2_glob,       B * k_z1θ2_glob],
        [B * k_θ1z1_glob,   B**2 * k_θ1θ1_glob,   B * k_θ1z2_glob,   B**2 * k_θ1θ2_glob],
        [k_z2z1_glob,       B * k_z2θ1_glob,       k_z2z2_glob,       B * k_z2θ2_glob],
        [B * k_θ2z1_glob,   B**2 * k_θ2θ1_glob,   B * k_θ2z2_glob,   B**2 * k_θ2θ2_glob]
    ])

    C_ae_local_star = np.array([
        [c_z1z1,       B * c_z1θ1,       c_z1z2,       B * c_z1θ2],
        [B * c_θ1z1,   B**2 * c_θ1θ1,   B * c_θ1z2,   B**2 * c_θ1θ2],
        [c_z2z1,       B * c_z2θ1,       c_z2z2,       B * c_z2θ2],
        [B * c_θ2z1,   B**2 * c_θ2θ1,   B * c_θ2z2,   B**2 * c_θ2θ2]
    ])

    K_ae_local_star = np.array([
        [k_z1z1,       B * k_z1θ1,       k_z1z2,       B * k_z1θ2],
        [B * k_θ1z1,   B**2 * k_θ1θ1,   B * k_θ1z2,   B**2 * k_θ1θ2],
        [k_z2z1,       B * k_z2θ1,       k_z2z2,       B * k_z2θ2],
        [B * k_θ2z1,   B**2 * k_θ2θ1,   B * k_θ2z2,   B**2 * k_θ2θ2]
    ])

    return C_ae_local_star, K_ae_local_star, C_ae_global_star, K_ae_global_star

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
        print("OBS: AD-ene har forskjellige reduced velocity-intervaller! Cae_global differs from Cae_local and Kae_global differs from Kae_local.")
    

    vmin_global = np.min(v_range[:, 0])
    vmax_global = np.max(v_range[:, 1])
    V_common = np.linspace(vmax_global, vmin_global, N)

    AD_interp = []
    AD_global = np.zeros((32, N))


    for i in range(32):
        v_min_red, v_max_red = v_range[i]
        V_local = np.linspace(v_min_red ,v_max_red, N)
        AD_local = np.polyval(poly_coeff[i], V_local)

        interp = np.interp(V_common, V_local, AD_local, left=np.nan, right=np.nan)
        AD_interp.append(interp)
        # Could extend this to include an other option than setting NaN for out of bounds values
        
        AD_global[i] = np.polyval(poly_coeff[i], V_common)

    
    eigvals_all = []
    eigvecs_all = []
    min_damping_ratios = []
    damping_ratios = []
    flutter_speed = None

    for i, V in enumerate(Vred_list):
        # V = Vred_list[i]  

        omega_old = 2* np.pi * f1 #Angular frequency (rad/s) ??f1

        for _ in range(max_iter):
            #Beregn nye Cae og Kae for denne omega
            C_aero_single_star, K_aero_single_star  = cae_kae_single(poly_coeff, V, B)
            C_aero_single_iter = 0.5 * rho * B**2*omega_old*C_aero_single_star
            K_aero_single_iter = 0.5 * rho * B**2*omega_old**2*K_aero_single_star

            eigvals, eigvec = solve_eigvalprob(Ms[i], Cs[i], Ks[i], C_aero_single_iter[i], K_aero_single_iter[i])
            eigvals_all.append(eigvals)
            eigvecs_all.append(eigvec)


            # Finn ny omega fra kritisk mode (minste demping)
            damping = -np.real(eigvals) / np.abs(eigvals)
            idx = np.argmin(damping)
            omega_new = np.abs(np.imag(eigvals[idx]))


            # Brudd dersom konvergert
            if np.abs(omega_new - omega_old) < eps:
                min_damping = np.min(damping) # Mest utstabile mode
                break
            else: omega_old = omega_new

        damping_ratios.append(damping)
        min_damping_ratios.append(min_damping)   

        if min_damping < 0 and flutter_speed is None:
            flutter_speed = V  # Første gang vi får negativ demping

        print("eigvals", eigvals)
        print("eigvec", eigvec)        
        print("damping", damping)
        print("min_damping", min_damping)

    if flutter_speed is None:
        print("Ingen flutter observert i gitt vindhastighetsintervall!")


    return flutter_speed, damping_ratios, eigvals_all, eigvecs_all
