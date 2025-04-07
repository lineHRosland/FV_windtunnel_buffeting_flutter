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


def solve_omega(poly_coeff,v_all, m1, m2, f1, f2, B, rho, zeta, max_iter, eps, N = 100, single = True):
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

    if single:
        Ms, Cs, Ks = structural_matrices_single(m1, m2, f1, f2, zeta)
        n_modes = 2 # 2 modes for single deck
    else:
        Ms, Cs, Ks = structural_matrices_twin(m1, m2, f1, f2, zeta)
        n_modes = 4 # 4 modes for twin deck

        eigvals_all_local = [[] for _ in range(n_modes)]
        eigvecs_all_local = [[] for _ in range(n_modes)]
        damping_ratios_local = [[] for _ in range(n_modes)]
        omega_all_local = [[] for _ in range(n_modes)]

    eigvals_all = [[] for _ in range(n_modes)]
    eigvecs_all = [[] for _ in range(n_modes)]
    damping_ratios = [[] for _ in range(n_modes)]
    omega_all = [[] for _ in range(n_modes)]


    V_list = np.linspace(0, 80, N) #m/s

    for i, V in enumerate(V_list):
        omega_old = np.array([2*np.pi*f1, 2*np.pi*f2])
        damping_old = [zeta, zeta]
        eigvec_old = [None] * n_modes

        for j in range(n_modes): # 4 modes for twin deck, 2 modes for single deck

            Vred_global = [V/(omega_old[j]*B)] * 32  # reduced velocity for global
                # Formatet på Vred_global må matche Vred_local

            for _ in range(max_iter):
                print("omega_ref", omega_old[j])
                    
                #Beregn nye Cae og Kae for denne omega[i]
                if single:
                    C_star, K_star = cae_kae_single(poly_coeff, Vred_global, B)
                else:
                    C_star, K_star = cae_kae_twin(poly_coeff, Vred_global, B)

                C_aero_single_iter = 0.5 * rho * B**2 * omega_old[j] * C_star
                K_aero_single_iter = 0.5 * rho * B**2 * omega_old[j]**2 * K_star
                
                eigvals, eigvecs = solve_eigvalprob(Ms, Cs, Ks, C_aero_single_iter, K_aero_single_iter)
                # Komplekse konjugate egenverdier er to sider av samme svingende bevegelse.
                # De beskriver samme mode, bare i motfase – og inneholder nøyaktig samme fysiske informasjon
                
                print("eigvals", eigvals)

                if np.all(np.imag(eigvals) == 0):
                    print("OBS: Ingen komplekse egenverdier - systemet oscillerer ikke!")

                
                # Behold kun én av hvert konjugatpar
                eigvals_pos = eigvals[np.imag(eigvals) > 0]
                eigvecs_pos = eigvecs[:, np.imag(eigvals) > 0]  # hver kolonne er en φ

                # Beregn damping og omega for alle positive eigvals
                omega_pos = np.imag(eigvals_pos)
                damping_pos = -np.real(eigvals_pos) / np.abs(eigvals_pos)

                # Finn λ for denne spesifikke moden j ??
                # Beregn projeksjon:
                similarities = [np.abs(np.dot(eigvec_old[j].conj().T, eigvecs_pos[:, k])) for k in range(eigvecs_pos.shape[1])]
                idx1 = np.argmax(similarities)
                # Beregn score med w og damping for alle modene
                score = np.abs(omega_pos - omega_old[j]) + 5 * np.abs(damping_pos - damping_old[j])
                idx2 = np.argmin(score)

                print("idx1", idx1)
                print("idx2", idx2)

                λj = eigvals_pos[idx1]
                φj = eigvecs_pos[:, idx1]

                omega_new = np.imag(λj)
                damping_new = -np.real(λj) / np.abs(λj)
                
                # Sjekk konvergens for mode
                if np.abs(omega_new - omega_old[j]) < eps:
                    omega_all[j].append(omega_new)              # lagre alle modenes omega
                    damping_ratios[j].append(damping_new)       # lagre alle modenes damping
                    eigvals_all[j].append(λj)                   # lagre alle modenes egenverdier
                    eigvecs_all[j].append(φj)                   # lagre alle modenes egenvektorer
                    break
                else: # Oppdater dersom w ikke har konvergert
                    omega_old[j] = omega_new
                    damping_old = damping_new
                    eigvec_old[j] = φj
    if not single:
        Vred= np.linspace(np.max(v_all[:, 0]), np.min(v_all[:, 1]), N) #m/s reduce

        for i, V in enumerate(Vred): 
            omega_old = np.array([2*np.pi*f1, 2*np.pi*f2])
            damping_old = [zeta, zeta]
            eigvec_old = [None] * n_modes

            for j in range(n_modes): # 4 modes for twin deck, 2 modes for single deck
                V_red_local = [V]* 32
                for _ in range(max_iter):
                    print("omega_ref", omega_old[j])
                        
                    #Beregn nye Cae og Kae for denne omega[i]
                    if single:
                        C_star, K_star = cae_kae_single(poly_coeff, V_red_local, B)
                    else:
                        C_star, K_star = cae_kae_twin(poly_coeff, V_red_local, B)

                    C_aero_single_iter = 0.5 * rho * B**2 * omega_old[j] * C_star
                    K_aero_single_iter = 0.5 * rho * B**2 * omega_old[j]**2 * K_star
                    
                    eigvals, eigvecs = solve_eigvalprob(Ms, Cs, Ks, C_aero_single_iter, K_aero_single_iter)
                   
                    # Behold kun én av hvert konjugatpar
                    eigvals_pos = eigvals[np.imag(eigvals) > 0]
                    eigvecs_pos = eigvecs[:, np.imag(eigvals) > 0]  # hver kolonne er en φ

                    # Beregn damping og omega for alle positive eigvals
                    omega_pos = np.imag(eigvals_pos)
                    damping_pos = -np.real(eigvals_pos) / np.abs(eigvals_pos)

                    # Finn λ for denne spesifikke moden j ??
                    # Beregn projeksjon:
                    similarities = [np.abs(np.dot(eigvec_old[j].conj().T, eigvecs_pos[:, k])) for k in range(eigvecs_pos.shape[1])]
                    idx1 = np.argmax(similarities)
                    # Beregn score med w og damping for alle modene
                    score = np.abs(omega_pos - omega_old[j]) + 5 * np.abs(damping_pos - damping_old[j])
                    idx2 = np.argmin(score)

                    print("idx1", idx1)
                    print("idx2", idx2)

                    λj = eigvals_pos[idx1]
                    φj = eigvecs_pos[:, idx1]

                    omega_new = np.imag(λj)
                    damping_new = -np.real(λj) / np.abs(λj)
                    
                    # Sjekk konvergens for mode
                    if np.abs(omega_new - omega_old[j]) < eps:
                        omega_all_local[j].append(omega_new)              # lagre alle modenes omega
                        damping_ratios_local[j].append(damping_new)       # lagre alle modenes damping
                        eigvals_all_local[j].append(λj)                   # lagre alle modenes egenverdier
                        eigvecs_all_local[j].append(φj)                   # lagre alle modenes egenvektorer
                        break
                    else: # Oppdater dersom w ikke har konvergert
                        omega_old[j] = omega_new
                        damping_old = damping_new
                        eigvec_old[j] = φj    

    return damping_ratios, omega_all, eigvals_all, eigvecs_all, V_list, damping_ratios_local, omega_all_local, eigvals_all_local, eigvecs_all_local, V_red_local


def solve_flutter_speed( damping_ratios, N = 100, single = True):

    if single:
        n_modes = 2
    else:
        n_modes = 4

    flutter_speed_modes = [None] * n_modes  # én per mode
    V_list = np.linspace(0, 80, N) #m/s
    damping = np.array(damping_ratios).T  # shape (N, 4) eller (N, 2)

    for i, V in enumerate(V_list):
        for j in range(n_modes): #  modes for single deck
            if damping[i,j] < 0 and flutter_speed_modes[j] is None:
                flutter_speed_modes[j] = V
    
    if all(fs is None for fs in flutter_speed_modes):
        return f"Ingen flutter observert for noen moder!"
    else:
        return flutter_speed_modes, V_list
     

     
def plot_damping_vs_wind_speed_single( V_list, Vred_defined, damping_ratios, damping_ratios_local, omega_all, omega_all_local, B, N = 100, single = True):
    """
    Plot damping ratios as a function of wind speed, and mark regions where ADs are defined.

    Parameters:
    ----------
    V_defined : list of tuples
        Each tuple (min, max) defines the valid velocity range for ADs (typically one per mode).
    V_list : ndarray
        List of wind speeds [m/s].
    damping_ratios : list of arrays
        List of damping ratios for each mode.
    omega_all : list of arrays
        List of angular frequencies (rad/s) for each mode.
    B : float
        Deck width.
    single : bool
        Whether plotting is for single-deck (2 modes) or twin-deck (4 modes).
    """
    omega = np.array(omega_all).T           # shape (N, 2/4)
    damping_ratios = np.array(damping_ratios).T  # shape (N, 2/4)


    colors = ['blue', 'red', 'green', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    if single:
        n_modes = 2 
        V_min = Vred_defined[0][0]
        V_max = Vred_defined[0][1]
    else:
        n_modes = 4
        omega_local = np.array(omega_all_local).T           # shape (N, 4)
        damping_ratios_local = np.array(damping_ratios_local).T  # shape (N, 4)

        Vred_local = np.zeros((32, N))
        for i, V in enumerate(Vred_defined):
            Vred_local[i] = np.linspace(Vred_defined[i][0], Vred_defined[i][1], N) #m/s reduced


    plt.figure(figsize=(10, 6))

    for j in range(n_modes):
        V_glob = V_list /(omega_all[:,j] * B)

        # Finn indeksene der V_glob er innenfor AD-definert område
        inside_idx = np.where((V_glob >= V_min) & (V_glob <= V_max))[0]
        outside_idx = np.where((V_glob < V_min) | (V_glob > V_max))[0]

        # Plot gyldig område som heltrukket
        if inside_idx.size > 0:
            plt.plot(V_effective[inside_idx], damping_ratios[j][inside_idx],
                     color=colors[j], label=labels[j], linewidth=2)

        # Plot ugyldig område som stipla
        if outside_idx.size > 0:
            plt.plot(V_effective[outside_idx], damping_ratios[j][outside_idx],
                     color=colors[j], linestyle='--', linewidth=1)


    if global ikke begynner før min og ikke slutter etter max:

        plt.plot(V_list*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue')
        plt.plot(V_list*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red')

        if not single:
            plt.plot(V_list*omega_3*B, damping_ratios[:,0], label="$\lambda_1$", color='blue')
            plt.plot(V_list*omega_4*B, damping_ratios[:,1], label="$\lambda_2$", color='red')

    elif global starter før min og slutter etter max:

        plt.plot(V_list[min index:max index]*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue')
        plt.plot(V_list[min index:max index]*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red')
        plt.plot(V_list*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue', linestyle="--")
        plt.plot(V_list*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red', linestyle="--")
    
    elif global starter før min og ikke slutter etter max:
         plt.plot(V_list[min index:]*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue')
        plt.plot(V_list[min index:]*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red')
        plt.plot(V_list*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue', linestyle="--")
        plt.plot(V_list*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red', linestyle="--")

    else:
        plt.plot(V_list[:max index]*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue')
        plt.plot(V_list[:max index]*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red')
        plt.plot(V_list*omega_1*B, damping_ratios[:,0], label="$\lambda_1$", color='blue', linestyle="--")
        plt.plot(V_list*omega_2*B, damping_ratios[:,1], label="$\lambda_2$", color='red', linestyle="--")


    plt.axhline(0, linestyle="--", color="gray", label="Critical Damping Line")
    plt.xlabel("Vindhastighet  [m/s]")
    plt.ylabel("Dempingforhold")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_frequency_vs_wind_speed_singles(Vred_list, omega_all, B):
    """
    Plotter frekvens og demping for hver mode over vindhastighet.

    Parameters:
    -----------
    Vred_list : array
        Liste over reduserte vindhastigheter.
    eigvals_all : array
        Array av egenverdier med shape (N, n_modes).
    """

  
    plt.figure(figsize=(10, 6))

    omega = np.array(omega_all).T           # shape (N, 2)
    frequencies = omega/(2*np.pi)  # shape (N, 2)

    omega_1 = omega[:,0]
    omega_2 = omega[:,1]

    plt.plot(Vred_list*omega_1*B, frequencies[:,0], label="$\lambda_1$", color='blue')
    plt.plot(Vred_list*omega_2*B, frequencies[:, 1], label="$\lambda_2$", color='red')
 
    plt.xlabel("Vindhastighet  [m/s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_damping_vs_wind_speed_twin(Vred_local, Vred_global, damping_ratios_local, damping_ratios_global, omega_all_local, omega_all_global, B):
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

    omega_local = np.array(omega_all_local).T           # shape (N, 2)
    damping_ratios_local = np.array(damping_ratios_local).T  # shape (N, 2)
    omega_1_local = omega_local[:,0]
    omega_2_local = omega_local[:,1]

    omega_global = np.array(omega_all_global).T           # shape (N, 2)
    damping_ratios_global = np.array(damping_ratios_global).T  # shape (N, 2)
    omega_1_global = omega_global[:,0]
    omega_2_global = omega_global[:,1]

    plt.plot(Vred_local*omega_1_local*B, damping_ratios_local[:,0], label="$\lambda_1$", color='blue')
    plt.plot(Vred_local*omega_2_local*B, damping_ratios_local[:,1], label="$\lambda_2$", color='red')

    plt.plot(Vred_global*omega_1_global*B, damping_ratios_global[:,0], label="$\lambda_1$", color='blue', linestyle='--')
    plt.plot(Vred_global*omega_2_global*B, damping_ratios_global[:,1], label="$\lambda_2$", color='red', linestyle='--')  

    plt.axhline(0, linestyle="--", color="gray", label="Critical Damping Line")
    plt.xlabel("Vindhastighet  [m/s]")
    plt.ylabel("Dempingforhold")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_frequency_vs_wind_speed_twin(Vred_local, Vred_global,omega_all_local, omega_all_global, B):
    """
    Plotter frekvens og demping for hver mode over vindhastighet.

    Parameters:
    -----------
    Vred_list : array
        Liste over reduserte vindhastigheter.
    eigvals_all : array
        Array av egenverdier med shape (N, n_modes).
    """

  
    plt.figure(figsize=(10, 6))

    omega_local = np.array(omega_all_local).T           # shape (N, 2)
    frequencies_local = omega_local/(2*np.pi)  # shape (N, 2)
    omega_1_local = omega_local[:,0]
    omega_2_local = omega_local[:,1]

    omega_global = np.array(omega_all_global).T           # shape (N, 2)
    frequencies_global = omega_global/(2*np.pi)  # shape (N, 2)
    omega_1_global = omega_global[:,0]
    omega_2_global = omega_global[:,1]

    plt.plot(Vred_local*omega_1_local*B, frequencies_local[:,0], label="$\lambda_1$", color='blue')
    plt.plot(Vred_local*omega_2_local*B, frequencies_local[:, 1], label="$\lambda_2$", color='red')

    plt.plot(Vred_global*omega_1_global*B, frequencies_global[:,0], label="$\lambda_1$", color='blue', linestyle='--')
    plt.plot(Vred_global*omega_2_global*B, frequencies_global[:, 1], label="$\lambda_2$", color='red', linestyle='--')
 
    plt.xlabel("Vindhastighet  [m/s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend()
    plt.grid(True)
    plt.show()

