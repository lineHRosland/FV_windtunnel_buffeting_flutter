# -*- coding: utf-8 -*-
"""
Created in April 2025

@author: linehro
"""

import numpy as np


def structural_matrices(m1, m2, f1, f2, zeta):
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
    Ms : ndarray
        Mass matrix.
    Ks : ndarray
        Stiffness matrix.
    Cs : ndarray
        Damping matrix.
    """
    # Stiffness
    k1 = (2 * np.pi * f1) ** 2 * m1  #  (Vertical)
    k2 = (2 * np.pi * f2) ** 2 * m2  #  (Torsion)

    # Damping
    c1 = 2 * zeta * np.sqrt(k1 * m1)  # (vertical)
    c2 = 2 * zeta * np.sqrt(k2 * m2)  # (torsion)

    # Structural matrices, diagonal matrices in modal coordinates
    Ms = np.diag([m1, m2])  # Mass matrix
    Ks = np.diag([k1, k2])  # Stiffness matrix
    Cs = np.diag([c1, c2])  # Damping matrix

    return Ms, Cs, Ks

def cae_kae_twin(poly_coeff, v_range, B, f, N=100):
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
    N : int
        Number of points per AD
    Returns:
    --------
    C_all : ndarray of shape (N, 2, 2)
        Aerodynamic damping matrices
    K_all : ndarray of shape (N, 2, 2)
        Aerodynamic stiffness matrices
    """
    # Sjekk at alle AD-er har samme hastighetsintervall
    if not np.allclose(v_range, v_range[0], atol=1e-8):
        print("OBS: AD-ene har forskjellige reduced velocity-intervaller! Cae_global differs from Cae_local and Kae_global differs from Kae_local.")
    

    vmin_global = np.min(v_range[:, 0])
    vmax_global = np.max(v_range[:, 1])
    V_common = np.linspace(f * B / vmax_global, f * B / vmin_global, N)

    AD_interp = []
    AD_global = np.zeros((32, N))


    for i in range(32):
        v_min_red, v_max_red = v_range[i]
        V_local = np.linspace(f*B/v_min_red ,f*B/ v_max_red, N)
        AD_local = np.polyval(poly_coeff[i], V_local)

        interp = np.interp(V_common, V_local, AD_local, left=np.nan, right=np.nan)
        AD_interp.append(interp)
        # Could extend this to include an other option than setting NaN for out of bounds values
        
        AD_global[i] = np.polyval(poly_coeff[i], V_common)

    
    #Damping and stiffness matrices
    C_ae_local = np.zeros((N, 4, 4))
    K_ae_local = np.zeros((N, 4, 4))
    C_ae_global = np.zeros((N, 4, 4))
    K_ae_global = np.zeros((N, 4, 4))


    for i in range(N):
        # AD
        c_z1z1, c_z1θ1, c_z1z2, c_z1θ2 = AD_interp[0, i], AD_interp[1, i], AD_interp[2, i], AD_interp[3, i]
        c_θ1z1, c_θ1θ1, c_θ1z2, c_θ1θ2 = AD_interp[4, i], AD_interp[5, i], AD_interp[6, i], AD_interp[7, i]
        c_z2z1, c_z2θ1, c_z2z2, c_z2θ2 = AD_interp[8, i], AD_interp[9, i], AD_interp[10, i], AD_interp[11, i]
        c_θ2z1, c_θ2θ1, c_θ2z2, c_θ2θ2 = AD_interp[12, i], AD_interp[13, i], AD_interp[14, i], AD_interp[15, i]
        k_z1z1, k_z1θ1, k_z1z2, k_z1θ2 = AD_interp[16, i], AD_interp[17, i], AD_interp[18, i], AD_interp[19, i]
        k_θ1z1, k_θ1θ1, k_θ1z2, k_θ1θ2 = AD_interp[20, i], AD_interp[21, i], AD_interp[22, i], AD_interp[23, i]
        k_z2z1, k_z2θ1, k_z2z2, k_z2θ2 = AD_interp[24, i], AD_interp[25, i], AD_interp[26, i], AD_interp[27, i]
        k_θ2z1, k_θ2θ1, k_θ2z2, k_θ2θ2 = AD_interp[28, i], AD_interp[29, i], AD_interp[30, i], AD_interp[31, i]

        # AD_global
        c_z1z1_glob, c_z1θ1_glob, c_z1z2_glob, c_z1θ2_glob = AD_global[0, i], AD_global[1, i], AD_global[2, i], AD_global[3, i]
        c_θ1z1_glob, c_θ1θ1_glob, c_θ1z2_glob, c_θ1θ2_glob = AD_global[4, i], AD_global[5, i], AD_global[6, i], AD_global[7, i]
        c_z2z1_glob, c_z2θ1_glob, c_z2z2_glob, c_z2θ2_glob = AD_global[8, i], AD_global[9, i], AD_global[10, i], AD_global[11, i]
        c_θ2z1_glob, c_θ2θ1_glob, c_θ2z2_glob, c_θ2θ2_glob = AD_global[12, i], AD_global[13, i], AD_global[14, i], AD_global[15, i]
        k_z1z1_glob, k_z1θ1_glob, k_z1z2_glob, k_z1θ2_glob = AD_global[16, i], AD_global[17, i], AD_global[18, i], AD_global[19, i]
        k_θ1z1_glob, k_θ1θ1_glob, k_θ1z2_glob, k_θ1θ2_glob = AD_global[20, i], AD_global[21, i], AD_global[22, i], AD_global[23, i]
        k_z2z1_glob, k_z2θ1_glob, k_z2z2_glob, k_z2θ2_glob = AD_global[24, i], AD_global[25, i], AD_global[26, i], AD_global[27, i]
        k_θ2z1_glob, k_θ2θ1_glob, k_θ2z2_glob, k_θ2θ2_glob = AD_global[28, i], AD_global[29, i], AD_global[30, i], AD_global[31, i]

        C_ae_global[i] = np.array([
            [c_z1z1_glob,       B * c_z1θ1_glob,       c_z1z2_glob,       B * c_z1θ2_glob],
            [B * c_θ1z1_glob,   B**2 * c_θ1θ1_glob,   B * c_θ1z2_glob,   B**2 * c_θ1θ2_glob],
            [c_z2z1_glob,       B * c_z2θ1_glob,       c_z2z2_glob,       B * c_z2θ2_glob],
            [B * c_θ2z1_glob,   B**2 * c_θ2θ1_glob,   B * c_θ2z2_glob,   B**2 * c_θ2θ2_glob]
        ])
        K_ae_global[i] = np.array([
            [k_z1z1_glob,       B * k_z1θ1_glob,       k_z1z2_glob,       B * k_z1θ2_glob],
            [B * k_θ1z1_glob,   B**2 * k_θ1θ1_glob,   B * k_θ1z2_glob,   B**2 * k_θ1θ2_glob],
            [k_z2z1_glob,       B * k_z2θ1_glob,       k_z2z2_glob,       B * k_z2θ2_glob],
            [B * k_θ2z1_glob,   B**2 * k_θ2θ1_glob,   B * k_θ2z2_glob,   B**2 * k_θ2θ2_glob]
        ])

        C_ae_local[i] = np.array([
            [c_z1z1,       B * c_z1θ1,       c_z1z2,       B * c_z1θ2],
            [B * c_θ1z1,   B**2 * c_θ1θ1,   B * c_θ1z2,   B**2 * c_θ1θ2],
            [c_z2z1,       B * c_z2θ1,       c_z2z2,       B * c_z2θ2],
            [B * c_θ2z1,   B**2 * c_θ2θ1,   B * c_θ2z2,   B**2 * c_θ2θ2]
        ])

        K_ae_local[i] = np.array([
            [k_z1z1,       B * k_z1θ1,       k_z1z2,       B * k_z1θ2],
            [B * k_θ1z1,   B**2 * k_θ1θ1,   B * k_θ1z2,   B**2 * k_θ1θ2],
            [k_z2z1,       B * k_z2θ1,       k_z2z2,       B * k_z2θ2],
            [B * k_θ2z1,   B**2 * k_θ2θ1,   B * k_θ2z2,   B**2 * k_θ2θ2]
        ])

    return C_ae_local, K_ae_local, C_ae_global, K_ae_global, V_common

def cae_kae_single(poly_coeff, v_range, B, f, N=100):
    """
    Evaluates all 8 aerodynamic derivatives individually across their own reduced velocity range.

    Parameters:
    -----------
    poly_coeff : ndarray
        Polynomial coefficients, shape (8, 3)
    v_range : ndarray
        Reduced velocity range per AD, shape (8, 2)
    B : float
        Section width (m)
    f : float
        Frequency (Hz)
    N : int
        Number of points per AD
    Returns:
    --------
    C_all : ndarray of shape (N, 2, 2)
        Aerodynamic damping matrices
    K_all : ndarray of shape (N, 2, 2)
        Aerodynamic stiffness matrices
    """
    # Sjekk at alle AD-er har samme hastighetsintervall
    if not np.allclose(v_range, v_range[0], atol=1e-8):
        raise ValueError("OBS: AD-ene har forskjellige reduced velocity-intervaller!"
                         "Denne funksjonen forutsetter at alle bruker samme v_range.")
    
    AD_N = np.zeros((8, N))
    V_N = np.zeros((8, N))

    for i in range(8):
        v_min_red, v_max_red = v_range[i]
        V_i = np.linspace(f*B/v_min_red ,f*B/ v_max_red, N)
        AD_i = np.polyval(poly_coeff[i], V_i)
        V_N[i] = V_i
        AD_N[i] = AD_i
    
    #Damping and stiffness matrices
    C_aeN = np.zeros((N, 2, 2))
    K_aeN = np.zeros((N, 2, 2))

    for i in range(N):
        # AD
        H1, H2, H3, H4 = AD_N[0, i], AD_N[1, i], AD_N[2, i], AD_N[3, i]
        A1, A2, A3, A4 = AD_N[4, i], AD_N[5, i], AD_N[6, i], AD_N[7, i]

        C_aeN[i] = np.array([
            [H1,       B * H2],
            [B * A1,   B**2 * A2]
        ])
        K_aeN[i] = np.array([
            [H4,       B * H3],
            [B * A4,   B**2 * A3]
        ])

    return C_aeN, K_aeN, V_N


#TOD0: Ad plotting functions for Cae and Kae
