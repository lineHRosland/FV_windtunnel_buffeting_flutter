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
    The function takes the model wind speed [m/s] and the scale factor as inputs 
    and returns the full scale wind speed [m/s].
    """
    return Um * 1/np.sqrt(scale)


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
    n = M_struc.shape[0] # n = 2: single deck, n = 4: twin deck

    
    print
    A = -la.block_diag(M_struc, np.eye(n))
    B = np.block([
        [C_struc - C_aero, K_struc - K_aero],
        [-np.eye(n), np.zeros((n, n))]
    ])

    eigvals, eigvec = la.eig(B, A)
    return eigvals, eigvec

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
    c1 = 2 * zeta * m1 * np.sqrt(k1 * m1)  # (Vertical)
    c2 = 2 * zeta * m2 * np.sqrt(k2 * m2)  # (Torsion)

    # Structural matrices, diagonal matrices in modal coordinates
    Ms = np.array([[m1,0],[0,m2]])  # Mass matrix
    Cs = np.array([[c1,0],[0,c2]])  # Damping matrix
    Ks = np.array([[k1,0],[0,k2]])  # Stiffness matrix

    if not single:
        Ms = np.block([
        [Ms, np.zeroes((2,2))],
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


def cae_kae_single(poly_coeff, V, B):
    """
    Evaluates the 8 aerodynamic derivatives for single-deck bridges.

    Parameters:
    -----------
    poly_coeff : ndarray, shape (8, 3)
        Polynomial coefficients for H1–H4 and A1–A4 (aerodynamic derivatives).
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

    
    #Damping and stiffness matrices
    C_aeN_star = np.zeros((2, 2)) #Dimensionless
    K_aeN_star = np.zeros((2, 2)) #Dimensionless

    # AD
    H1, H2, H3, H4 = np.polyval(poly_coeff[0], V[0]), np.polyval(poly_coeff[1], V[1]), np.polyval(poly_coeff[2], V[2]), np.polyval(poly_coeff[3], V[3])
    A1, A2, A3, A4 = np.polyval(poly_coeff[4], V[4]), np.polyval(poly_coeff[5], V[5]), np.polyval(poly_coeff[6], V[6]), np.polyval(poly_coeff[7], V[7])

    C_aeN_star = np.array([
        [H1,       B * H2],
        [B * A1,   B**2 * A2]
    ])
    K_aeN_star = np.array([
        [H4,       B * H3],
        [B * A4,   B**2 * A3]
    ])
        
    return C_aeN_star, K_aeN_star




def cae_kae_twin(poly_coeff, V, B):
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
    '''
    Løser flutteranalyse med iterativ metode for enten single-deck (2DOF) eller twin-deck (4DOF).
    Returnerer både global og lokal løsning (kun twin-deck).

    Parameters
    ----------
    poly_coeff : ndarray
        Aerodynamiske deriverte som polynomkoeffisienter (32, 3)
    v_all : ndarray
        Redusert hastighetsintervall for hver AD (32, 2)
    m1, m2, f1, f2 : float
        Masse og frekvenser for de to modene
    B : float
        Bredde av broseksjon
    rho : float
        Lufttetthet
    zeta : float
        Strukturell demping
    max_iter : int
        Maks antall iterasjoner per mode
    eps : float
        Toleranse for konvergens
    N : int
        Antall punkter i vindhastighetsintervall
    single : bool
        True for single-deck, False for twin-deck

    Returns
    -------
    damping_ratios, omega_all, eigvals_all, eigvecs_all : list
        Globale resultater (alle)
    V_list : np.ndarray
        Vindhastighetsliste brukt for global løsning
    damping_ratios_local, omega_all_local, eigvals_all_local, eigvecs_all_local : list
        Lokale resultater
    V_red_local : np.ndarray
        Felles redusert hastighetsintervall 
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

    eigvals_all_local = [[] for _ in range(n_modes)]
    eigvecs_all_local = [[] for _ in range(n_modes)]
    damping_ratios_local = [[] for _ in range(n_modes)]
    omega_all_local = [[] for _ in range(n_modes)]


    V_list = np.linspace(0, 80, N) #m/s

    for i, V in enumerate(V_list):
        omega_old = np.array([2*np.pi*f1, 2*np.pi*f2])
        damping_old = [zeta, zeta]
        eigvec_old = [None] * n_modes
        print("Vindhastighet iterasjon nr. " + str(i+1) + "som tilsvarer " + str(V) + " m/s")

        for j in range(n_modes): # 4 modes for twin deck, 2 modes for single deck
            print("Modeiterasjon nr. " + str(j+1))

            if single:
                Vred_global = [V/(omega_old[j]*B)] * 8  # reduced velocity for global
                    # Formatet på Vred_global må matche Vred_local
            else:
                Vred_global = [V/(omega_old[j]*B)] * 32

            for k in range(max_iter):
                print("Iterasjon nr. " + str(k+1) + " for mode " + str(j+1))
                
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
                if eigvec_old[j] is None:
                    # Første iterasjon – bruk nærmeste i frekvens
                    score = np.abs(omega_pos - omega_old[j]) + 5 * np.abs(damping_pos - damping_old[j])
                    idx = np.argmin(score)
                else:
                    # Beregn projeksjon mot forrige egenvektor
                    if eigvecs_pos.shape[1] == 0:
                        print("Ingen komplekse egenverdier — hopper over denne iterasjonen.")
                        continue   
                    else:
                        similarities = [np.abs(np.dot(eigvec_old[j].conj().T, eigvecs_pos[:, k])) for k in range(eigvecs_pos.shape[1])]
                        idx = np.argmax(similarities)

                    score = np.abs(omega_pos - omega_old[j]) + 5 * np.abs(damping_pos - damping_old[j])
                    idx2 = np.argmin(score)
                
                    print("idx", idx)
                    print("idx2", idx2)

                λj = eigvals_pos[idx]
                φj = eigvecs_pos[:, idx]

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
                    damping_old[j] = damping_new
                    eigvec_old[j] = φj
    
     
    Vred_local= np.linspace(np.max(v_all[:, 0]), np.min(v_all[:, 1]), N) #m/s reduce

    for i, V in enumerate(Vred_local): 
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
                    damping_old[j] = damping_new
                    eigvec_old[j] = φj    

    return damping_ratios, omega_all, eigvals_all, eigvecs_all, damping_ratios_local, omega_all_local, eigvals_all_local, eigvecs_all_local

def solve_flutter_speed( damping_ratios, N = 100, single = True):
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
    V_list = np.linspace(0, 80, N)
    damping = np.array(damping_ratios).T  # Shape (N, n_modes)

    for j in range(n_modes):
        for i, V in enumerate(V_list):
            if damping[i, j] < 0 and flutter_speed_modes[j] is None:
                flutter_speed_modes[j] = V
                break

    if all(fs is None for fs in flutter_speed_modes):
        print("Ingen flutter observert for noen moder!")
        return None
    return flutter_speed_modes
     

     
def plot_damping_vs_wind_speed_single(B, Vred_defined, damping_ratios, omega_all, damping_ratios_local = None,  omega_all_local = None,  dist="Fill in dist",N = 100, single = True):
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
    damping_ratios = np.array(damping_ratios).T  # shape (N, 2/4)
    omega_all = np.array(omega_all).T  # shape (N, n_modes)

    V_list = np.linspace(0, 80, N) #m/s

    colors = ['blue', 'red', 'green', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    plt.figure(figsize=(10, 6))

    if single:
        n_modes = 2 
        title = f"Demping vs. vindhastighet - {dist}"

    else:
        n_modes = 4
        title = f"Demping vs. vindhastighet - {dist}"


    omega_local = np.array(omega_all_local).T           # shape (N, 4)
    damping_ratios_local = np.array(damping_ratios_local).T  # shape (N, 4)
    
    Vred_defined = np.array(Vred_defined)
    Vred_local= np.linspace(np.max(Vred_defined[:, 0]), np.min(Vred_defined[:, 1]), N) 
        
    for j in range(n_modes):
        plt.plot(Vred_local*omega_local[:,j]*B, damping_ratios_local[:,j], label=labels[j], color=colors[j])
        plt.plot(V_list, damping_ratios[:,j], color=colors[j], linestyle="--")

    plt.axhline(0, linestyle="--", color="gray", linewidth=0.8, label="Kritisk demping")
    plt.xlabel("Vindhastighet [m/s]")
    plt.ylabel("Dempingsforhold")
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_frequency_vs_wind_speed(B, Vred_defined, omega_all, omega_all_local = None, dist="Fill in dist",N = 100, single = True):
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
    colors = ['blue', 'red', 'green', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    omega = np.array(omega_all).T           # shape (N, 2)
    frequencies = omega/(2*np.pi)  # shape (N, 2)
    V_list = np.linspace(0, 80, N) #m/s

    plt.figure(figsize=(10, 6))

    if single:
        n_modes = 2 
        title = f"Egenfrekvenser vs vindhastighet - {dist}"
    else:
        n_modes = 4
        title = f"Egenfrekvenser vs vindhastighet - {dist}"
    
    omega_local = np.array(omega_all_local).T           # shape (N, 4)
    frequencies_local = omega_local/(2*np.pi)           # shape (N, 4)

    Vred_defined = np.array(Vred_defined)
    Vred_local= np.linspace(np.max(Vred_defined[:, 0]), np.min(Vred_defined[:, 1]), N) 

    for j in range(n_modes):
        plt.plot(Vred_local*omega_local[:,j]*B, frequencies_local[:,j], label=labels[j], color=colors[j])
        plt.plot(V_list, frequencies[:,j], color=colors[j], linestyle="--")
 
    plt.xlabel("Vindhastighet [m/s]")
    plt.ylabel("Egenfrekvens [Hz]")
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
