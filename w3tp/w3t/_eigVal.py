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

    return Ms, Ks, Cs


def cae_kae_single(poly_coeff, v_range, B, N=100):
    """
    Evaluates all 8 aerodynamic derivatives individually across their own reduced velocity range.

    Parameters:
    -----------
    poly_coeff : ndarray
        Polynomial coefficients, shape (8, 3)
    v_range : ndarray
        Reduced velocity range per AD, shape (8, 2)
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
        v_min, v_max = v_range[i]
        V_i = np.linspace(v_min, v_max, N)
        AD_i = np.polyval(poly_coeff[i], V_i)
        V_N[i] = V_i
        AD_N[i] = AD_i
    
    #Damping and stiffness matrices
    C_aeN = np.zeros((N, 2, 2))
    K_aeN = np.zeros((N, 2, 2))

    for i in range(N):
        # Aerodynamic matrices
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