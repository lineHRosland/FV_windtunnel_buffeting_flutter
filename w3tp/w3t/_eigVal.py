"""
Created in April 2025

@author: linehro
"""

import numpy as np
from scipy import linalg as spla
import matplotlib.pyplot as plt
import time
from mode_shapes import mode_shape_single
from mode_shapes import mode_shape_twin

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

 
    # C_ae_star_gen = np.zeros((2, 2))
    # K_ae_star_gen = np.zeros((2, 2))
 
    # for i in range(N-1):
    #     dx = x[i+1] - x[i] # discretized length
 
    #     # Damping
    #     C_integrand_left  = Phi[i].T @ C_ae_star @ Phi[i]
    #     C_integrand_right = Phi[i+1].T @ C_ae_star @ Phi[i+1]
    #     C_ae_star_gen += 0.5 * (C_integrand_left + C_integrand_right) * dx
 
    #     # Stiffness
    #     K_integrand_left = Phi[i].T @ K_ae_star @ Phi[i]
    #     K_integrand_right = Phi[i+1].T @ K_ae_star @ Phi[i+1]
    #     K_ae_star_gen += 0.5 * (K_integrand_left + K_integrand_right) * dx

def generalize_global(C, K, Phi):
    """
    Generalizes full global aerodynamic matrices C and K using modal matrix Phi.

    Parameters:
    -----------
    C : ndarray, shape (n, n)
        Full aerodynamic damping matrix (global DOFs).
    K : ndarray, shape (n, n)
        Full aerodynamic stiffness matrix (global DOFs).
    Phi : ndarray, shape (n, n)
        Modal matrix (columns are mode shapes, usually mass-normalized).

    Returns:
    --------
    C_gen : ndarray, shape (n, n)
        Generalized (modal) damping matrix.
    K_gen : ndarray, shape (n, n)
        Generalized (modal) stiffness matrix.
    """
    C_gen = Phi.T @ C @ Phi # shape: Single - (2, 2) or Twin - (4, 4)
    K_gen = Phi.T @ K @ Phi
    return C_gen, K_gen


def normalize_modes(Phi, M):
    """
    Masse-normaliserer modeshapes for 2 DOF (single-deck) eller 4 DOF (twin-deck).
    
    Parameters:
    -----------
    Phi : ndarray of shape (N, n, n)
        Modalmatriser langs broen. n = 2 for single-deck, n = 4 for twin-deck.
    M : ndarray of shape (n, n)
        Massematrise i fysisk DOF-rom (typisk diagonal).
    
    Returns:
    --------
    Phi_norm : ndarray of shape (N, n, n)
        Masse-normaliserte modeshapes.
    """
    N, n, _ = Phi.shape
    Phi_norm = np.zeros_like(Phi)

    for i in range(N):
        for j in range(n):  # én mode om gangen
            phi_j = Phi[i][:, j]  # shape: (n,)
            m_j = phi_j.T @ M @ phi_j
            Phi_norm[i][:, j] = phi_j / np.sqrt(m_j)
        
    return Phi_norm

def global_massnorm_modeshapes(M, single = True):

    if single:
        n = 2  
        Phi = mode_shape_single(full_matrix=True)[0]

    else:
        n = 4
        Phi= mode_shape_twin(full_matrix=True)[0]
    
    # Phi shape: (n_nodes, n_dof, n_modes)
    # M shape: (n_dof, n_dof) (2x2 or 4x4)
    
    #Phi_massNorm = normalize_modes(Phi, M) 

    Phi_global = Phi.reshape(-1,n) 

    return Phi_global



def structural_matrices_massnorm(f1, f2, zeta, single=True):
    """
    Return mass-normalized structural matrices:
    M = I, K = diag(omega^2), C = diag(2*zeta*omega)
    """
    ω1 = 2 * np.pi * f1
    ω2 = 2 * np.pi * f2

    Ms = np.eye(2)
    Ks = np.diag([ω1**2, ω2**2])
    Cs = np.diag([2 * zeta * ω1, 2 * zeta * ω2])

    if not single:
        Ms = np.eye(4)
        Ks = np.kron(np.eye(2), Ks)
        Cs = np.kron(np.eye(2), Cs)

    return Ms, Cs, Ks

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



def cae_kae_single(poly_coeff, Vred_global, Phi, x, B):
    """
    Evaluates generalized aerodynamic damping and stiffness matrices for a single-deck bridge.

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
    C_ae_star: ndarray, shape (2, 2)
    K_ae_star: ndarray, shape (2, 2)
    """

    Vred_global = float(Vred_global) 

    # AD
    H1, H2, H3, H4 = np.polyval(poly_coeff[0], Vred_global), np.polyval(poly_coeff[1][::-1], Vred_global), np.polyval(poly_coeff[2][::-1], Vred_global), np.polyval(poly_coeff[3][::-1], Vred_global)
    A1, A2, A3, A4 = np.polyval(poly_coeff[4], Vred_global), np.polyval(poly_coeff[5][::-1], Vred_global), np.polyval(poly_coeff[6][::-1], Vred_global), np.polyval(poly_coeff[7][::-1], Vred_global)

    AA = np.zeros((8, 2, 2))
    AA[0, 0, 0] = H1
    AA[1, 0, 1] = B * H2
    AA[2, 1, 0] = B * A1
    AA[3, 1, 1] = B**2 * A2
    AA[4, 0, 0] = H4
    AA[5, 0, 1] = B * H3
    AA[6, 1, 0] = B * A4
    AA[7, 1, 1] = B**2 * A3

    AA_tilde = np.zeros((8, 2, 2))  # Generalisert aero-matriser
    phiphi_p = np.transpose(Phi, [1, 0, 2])  # (DOF, mode, x)

    length = x[-1] - x[0]  # Lengde på bro
    nlength = len(x)  # Antall punkter
    beam = np.linspace(0, length, nlength)  # x-akse for integrasjon

    for k in range(8):
        integrand = np.zeros((len(beam),2, 2))
        for m in range(len(beam)):
            integrand[m] = phiphi_p[:, :, m].T @ AA[k] @ phiphi_p[:, :, m]
        for i in range(2):
            for j in range(2):
                AA_tilde[k, i, j] = np.trapz(integrand[:, i, j], beam)
    
    A1 = np.sum(AA_tilde[4:8], axis=0)  # stiffness part
    A2 = np.sum(AA_tilde[0:4], axis=0)  # damping part

    # Generaliserte aero-matriser
    Cae_star_gen = np.sum(AA_tilde[0:4], axis=0)
    Kae_star_gen = np.sum(AA_tilde[4:8], axis=0)

    # # Per meter
    # C_ae_star_per_m = np.array([
    #     [H1,       B * H2],
    #     [B * A1,   B**2 * A2]
    # ])
    # K_ae_star_per_m = np.array([
    #     [H4,       B * H3],
    #     [B * A4,   B**2 * A3]
    # ])

    # 66 segments
    # standard_segment_length = 19 meters
    # end_segment_length = 14 meters

    # dxs = np.diff(x) # shape: (66,)
    # N = len(x) 

    # segment_lengths = [dxs[0]] + list(dxs)

    # C_blocks = [C_ae_star_per_m * dx for dx in segment_lengths]  # liste med 66 stk (2x2)
    # K_blocks = [K_ae_star_per_m * dx for dx in segment_lengths]

    # C_aero_global = spla.block_diag(*C_blocks)  # Shape (2N, 2N)
    # K_aero_global = spla.block_diag(*K_blocks)

    # C_blocks = [np.zeros_like(C_ae_star_per_m) for _ in range(N)]
    # for i, dx in enumerate(dxs):
    #     C_seg = C_ae_star_per_m * dx
    #     C_blocks[i] += 0.5 * C_seg
    #     C_blocks[i+1] += 0.5 * C_seg
    
    # C_aero_global = spla.block_diag(*C_blocks) 

    # K_blocks = [np.zeros_like(K_ae_star_per_m) for _ in range(N)]
    # for i, dx in enumerate(dxs):
    #     K_seg = K_ae_star_per_m * dx
    #     K_blocks[i] += 0.5 * K_seg
    #     K_blocks[i+1] += 0.5 * K_seg

    # K_aero_global = spla.block_diag(*K_blocks)
    
        
    return Cae_star_gen, Kae_star_gen




def cae_kae_twin(poly_coeff, Vred_global, Phi, x, B):
    """
    Evaluates the generalized aerodynamic damping and stiffness matrices for a twin-deck bridge.

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
    C_ae_star: ndarray, shape (4, 4)
    K_ae_star: ndarray, shape (4, 4)
    """

    Vred_global = float(Vred_global) 
    # AD
    AD = [np.polyval(poly[::-1], Vred_global) for poly in poly_coeff]  # reversed for polyval

    # Initialiser 32 matriser (16 for C, 16 for K)
    AA = np.zeros((32, 4, 4))

    # Demping: fyll inn de 16 første
    AA[0, 0, 0] = AD[0]
    AA[1, 0, 1] = B * AD[1]
    AA[2, 0, 2] = AD[2]
    AA[3, 0, 3] = B * AD[3]

    AA[4, 1, 0] = B * AD[4]
    AA[5, 1, 1] = B**2 * AD[5]
    AA[6, 1, 2] = B * AD[6]
    AA[7, 1, 3] = B**2 * AD[7]

    AA[8, 2, 0] = AD[8]
    AA[9, 2, 1] = B * AD[9]
    AA[10, 2, 2] = AD[10]
    AA[11, 2, 3] = B * AD[11]

    AA[12, 3, 0] = B * AD[12]
    AA[13, 3, 1] = B**2 * AD[13]
    AA[14, 3, 2] = B * AD[14]
    AA[15, 3, 3] = B**2 * AD[15]

    # Stivhet: fyll inn de 16 neste
    AA[16, 0, 0] = AD[16]
    AA[17, 0, 1] = B * AD[17]
    AA[18, 0, 2] = AD[18]
    AA[19, 0, 3] = B * AD[19]

    AA[20, 1, 0] = B * AD[20]
    AA[21, 1, 1] = B**2 * AD[21]
    AA[22, 1, 2] = B * AD[22]
    AA[23, 1, 3] = B**2 * AD[23]

    AA[24, 2, 0] = AD[24]
    AA[25, 2, 1] = B * AD[25]
    AA[26, 2, 2] = AD[26]
    AA[27, 2, 3] = B * AD[27]

    AA[28, 3, 0] = B * AD[28]
    AA[29, 3, 1] = B**2 * AD[29]
    AA[30, 3, 2] = B * AD[30]
    AA[31, 3, 3] = B**2 * AD[31]

    AA_tilde = np.zeros((32, 4, 4))  # Generalisert aero-matriser
    phiphi_p = np.transpose(Phi, [1, 0, 2])  # (DOF, mode, x)

    length = x[-1] - x[0]  # Lengde på bro
    nlength = len(x)  # Antall punkter
    beam = np.linspace(0, length, nlength)  # x-akse for integrasjon

    for k in range(32):
        integrand = np.zeros((len(beam),4, 4))
        for m in range(len(beam)):
            integrand[m] = phiphi_p[:, :, m].T @ AA[k] @ phiphi_p[:, :, m]
        for i in range(4):
            for j in range(4):
                AA_tilde[k, i, j] = np.trapz(integrand[:, i, j], beam)

    # Generaliserte aero-matriser
    Cae_star_gen = np.sum(AA_tilde[0:16], axis=0)
    Kae_star_gen = np.sum(AA_tilde[16:8], axis=0)


    # C_ae_star_per_m = np.array([
    #     [c_z1z1,       B * c_z1θ1,       c_z1z2,       B * c_z1θ2],
    #     [B * c_θ1z1,   B**2 * c_θ1θ1,   B * c_θ1z2,   B**2 * c_θ1θ2],
    #     [c_z2z1,       B * c_z2θ1,       c_z2z2,       B * c_z2θ2],
    #     [B * c_θ2z1,   B**2 * c_θ2θ1,   B * c_θ2z2,   B**2 * c_θ2θ2]
    # ])

    # K_ae_star_per_m = np.array([
    #     [k_z1z1,       B * k_z1θ1,       k_z1z2,       B * k_z1θ2],
    #     [B * k_θ1z1,   B**2 * k_θ1θ1,   B * k_θ1z2,   B**2 * k_θ1θ2],
    #     [k_z2z1,       B * k_z2θ1,       k_z2z2,       B * k_z2θ2],
    #     [B * k_θ2z1,   B**2 * k_θ2θ1,   B * k_θ2z2,   B**2 * k_θ2θ2]
    # ])

    # dxs = np.diff(x) # shape: (66,)
    # N = len(x) 

    # segment_lengths = [dxs[0]] + list(dxs)

    # C_blocks = [C_ae_star_per_m * dx for dx in segment_lengths]  # liste med 66 stk (2x2)
    # K_blocks = [K_ae_star_per_m * dx for dx in segment_lengths]

    # C_aero_global = spla.block_diag(*C_blocks)  # Shape (2N, 2N)
    # K_aero_global = spla.block_diag(*K_blocks)

    # C_blocks = [np.zeros_like(C_ae_star_per_m) for _ in range(N)]
    # for i, dx in enumerate(dxs):
    #     C_seg = C_ae_star_per_m * dx
    #     C_blocks[i] += 0.5 * C_seg
    #     C_blocks[i+1] += 0.5 * C_seg
    
    # C_aero_global = spla.block_diag(*C_blocks) 

    # K_blocks = [np.zeros_like(K_ae_star_per_m) for _ in range(N)]
    # for i, dx in enumerate(dxs):
    #     K_seg = K_ae_star_per_m * dx
    #     K_blocks[i] += 0.5 * K_seg
    #     K_blocks[i+1] += 0.5 * K_seg

    # K_aero_global = spla.block_diag(*K_blocks)


    return Cae_star_gen, Kae_star_gen



def solve_omega(poly_coeff, Ms_not_massnorm, Ms, Cs, Ks, f1, f2, B, rho, zeta, eps, Vs,  x, single = True):
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

    # # empty() creates an array of the given shape and type, EVERY ELEMENT MUST BE INITIALIZED LATER
    # eigvals_all = np.empty((len(V_list), n_modes), dtype=complex)
    # eigvecs_all = np.empty((len(V_list), n_modes), dtype=object) 
    # damping_ratios  = np.empty((len(V_list), n_modes))
    # omega_all = np.empty((len(V_list), n_modes))

    #skip_mode = [False] * n_modes   # skip mode once flutter is detected

    # damping_z_list = [] if single or (not single and n_modes >= 1) else None
    # V_damping_z = []
    # V_damping_theta = []
    # damping_theta_list = [] if single or (not single and n_modes >= 1) else None

    # damping_old = [zeta]*n_modes
    # eigvec_old = [None] * n_modes

    if single:
        n_modes = 2 # 2 modes for single deck
        omega0 = np.array([2*np.pi*f1, 2*np.pi*f2])

        #dominant_dofs = [0, 1]  # z, θ
    else:   
        n_modes = 4 # 4 modes for twin deck
        omega0 = np.array([2*np.pi*f1, 2*np.pi*f2, 2*np.pi*f1, 2*np.pi*f2]) #Først brudekke 1, deretter brudekke 2
        #dominant_dofs = [0, 1, 2, 3]  # z1, θ1, z2, θ2
   
    # Flutter detection
    Vcritical = []
    omegacritical = []

    # Global results
    V = [] # Wind speed, m/s
    omega = [] # Frequency, rad/s
    damping = [] # Damping ratio
    eigvecs = [] # Eigenvectors 

    eigvals0, eigvecs0 = solve_eigvalprob(Ms, Cs, Ks,  np.zeros_like(Cs), np.zeros_like(Ks)) 

    # Sorter etter imaginærdel
    sort_idx = np.argsort(np.imag(eigvals))
    eigvals0_sorted = eigvals0[sort_idx]
    eigvecs0_sorted = eigvecs0[:, sort_idx] 
   
    
    V.append(0.0) # V = 0.0 m/s
    omega.append(np.imag(eigvals0_sorted[:2*n_modes])) #konjugatpar
    damping.append(np.real(eigvals0_sorted[:2*n_modes])) #konjugatpar, ikke normalisert 
    eigvecs.append(eigvecs0_sorted[:, :2*n_modes]) 


                # if np.isclose(V, 0.0): # still air
                #     # Egenverdiene og egenvektorene kommer i riktig rekkefølge
                #     # Save which DOFs that dominate the mode shape
                    

                #     Cae_gen = np.zeros_like(Ms)
                #     Kae_gen = np.zeros_like(Ms)
                    
                #     # Keep only the eigenvalues with positive imaginary part (complex conjugate pairs)
                #     eigvals_pos = eigvals[np.imag(eigvals) > 0]
                #     eigvecs_pos = eigvecs[:, np.imag(eigvals) > 0]  

                #     λj = eigvals_pos[j]
                #     φj = eigvecs_pos[:n_modes, j]

                #     omega_all[iterWind,j]= np.imag(λj) 
                #     damping_ratios[iterWind,j]= -np.real(λj) / np.abs(λj)
                #     eigvals_all[iterWind,j]= λj  
                #     eigvecs_all[iterWind,j]= φj
                #     eigvec_old[j] = φj

                #     converge = True

    ResMat = np.zeros((2, 2 * Ms.shape[0]))
        # Første rad (ResMat[0, :]) skal lagre imag(λ) → altså frekvensene (ω) for alle modene. (Omega: np.imag(λj))
        # Andre rad (ResMat[1, :]) skal lagre real(λ) → altså "rå" demping (ζ) for alle modene. (Damping:-np.real(λj) / np.abs(λj))
        # Når du løser et 2. ordens differensialligningssystem (som flutterproblemet), får du 2 × n_modes egenverdier.
        # For hver fysisk mode får du én positiv og én negativ imaginærdel (komplekse konjugatpar).
        # Derfor trenger du dobbelt så mange plasser.

    pos = 1 # Allerede lagt inn verdier for V = 0.0 m/s, så starter på 1

    stopWind = 0
    iterWind = 0
    V = 1 # Initial wind speed, m/s
    dV = 1 # Hvor mye vi øker vindhastighet per iterasjon. 

    while (iterWind < 1000 and stopWind == 0): # iterer over vindhastigheter
    #stopper etter 1000 forsøk hvis ikke flutter er funnet.

        for j in range(n_modes): # 4 modes for twin deck, 2 modes for single deck

            # if skip_mode[j]:
            #     omega_all[j].append(np.nan)
            #     damping_ratios[j].append(np.nan)
            #     eigvals_all[j].append(np.nan)
            #     eigvecs_all[j].append(np.nan)
            #     continue  # Go to next mode if flutter is detected

            #omegacritical: omega0[j]

            stopFreq = 0
            iterFreq = 0 # Teller hvor mange frekvens-iterasjoner

            omegacr = omega0[j] # Startverdi for omegacr (kritisk frekvens) er den naturlige frekvensen til modusen

            Vred = V/(omegacr*B) # reduced velocity 

            print(f"Wind speed iteration {iterWind+1}: V = {V} m/s")

            while (iterFreq < 10 and stopFreq == 0): # iterer over frekvens-iterasjoner
    
                #if V > 62: print(f"  Iteration {iter+1} for mode {j+1}")
               

                if single:
                    Cae_star_gen, Kae_star_gen = cae_kae_single(poly_coeff, Vred,x, B)
                else:
                    Cae_star_gen, Kae_star_gen = cae_kae_twin(poly_coeff, Vred,x, B)

                Cae_gen = 0.5 * rho * B**2 * omegacr * Cae_star_gen
                Kae_gen = 0.5 * rho * B**2 * omegacr**2 * Kae_star_gen

                eigvals, eigvecs = solve_eigvalprob(Ms, Cs, Ks, Cae_gen, Kae_gen)

                # Sorter etter imaginærdel
                sort_idx = np.argsort(np.imag(eigvals))
                eigvals_sorted = eigvals[sort_idx]
                eigvecs_sorted = eigvecs[:, sort_idx]

                # Ta positiv halvdel
                eigvals_sorted = eigvals_sorted[len(eigvals_sorted)//2:]
                eigvecs_sorted = eigvecs_sorted[:, len(eigvals_sorted)//2:]


                if single:
                    best_idx = np.argmin(np.abs(np.imag(eigvals_sorted) - omega[-1][n_modes + j]))    
                else:
                    best_idx = np.argmin(np.abs(np.imag(eigvals_sorted) - omega[-1][n_modes + j]) + 10 * np.abs(np.real(eigvals_sorted) - damping[-1][n_modes + j]))
                        

                domega = omegacr - np.imag(eigvals_sorted[best_idx])
                omegacr = np.imag(eigvals_sorted[best_idx])

                    # # Min gamle versjon 
                    # # Keep only the eigenvalues with positive imaginary part (complex conjugate pairs)
                    # threshold = 1e-6  
                    # imag_mask = np.imag(eigvals) > threshold
                    # eigvals_pos = eigvals[imag_mask]
                    # eigvecs_pos = eigvecs[:, imag_mask]  
                

                    # if eigvals_pos.size == 0:
                    #     print(f"Ingen komplekse egenverdier ved V = {V:.2f} m/s, mode {j+1}. Skipper mode.")
                    #     omega_all[iterWindi, j] = np.nan
                    #     damping_ratios[iterWind, j] = np.nan
                    #     eigvals_all[iterWind, j] = np.nan
                    #     eigvecs_all[iterWind, j] = None
                    #     break

                    # omega_pos = np.imag(eigvals_pos)
                    # damping_pos = -np.real(eigvals_pos) / np.abs(eigvals_pos)

                    #print(f"\nV = {V:.2f} m/s, mode {j+1}")
                    #print("Egenverdier (λ):", eigvals)

                

                    
                        #if V < 120:
                        #    best_idx = np.argmax(np.abs(eigvecs_pos[dominant_dofs[j], :]))


                    # λj = eigvals_pos[best_idx]
                    # φj = eigvecs_pos[:n_modes, best_idx]
                        # eigvecs_pos[:, j] = 4 komponenter i single-deck → skyldes at du henter hele state-vektoen (inkl. hastighet)



                    # plt.plot(np.abs(φj), label=f"Mode {j+1} at V={V:.1f}")
                    # plt.legend()
                    # plt.show()

                    # omega_new = np.imag(λj)
                    # damping_new = -np.real(λj) / np.abs(λj)
                        
                    # Check if the mode is converged 
                    
                if np.abs(domega) < eps and omegacr <= 0.0:
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
                            
                        #print("C: (-)", Cs - Cae_gen)
                        #print("K: (-)", Ks - Kae_gen)


                        #C_ratio = np.abs(Cae_gen) / np.abs(Cs - Cae_gen)
                        #print(f"V = {V:.2f} m/s → C_aero / C_total ratio:\n{C_ratio}")


                        # if j == 0 and V==100:  
                        #     plt.plot(V_damping_z, [(Cs[0,0] - d[0,0]) for d in damping_z_list], label='Cs-Cae z')
                        #     plt.plot(V_damping_z, [(d[0,0]) for d in damping_z_list], label='Cae z')
                        #     plt.axhline(Cs[0,0], color='grey', linestyle='--', label='Cs z')
                        #     if n_modes ==4:
                        #         plt.plot(V_damping_z, [(Cs[2,2] - d[2,2]) for d in damping_theta_list], label='Cs-Cae z2')
                        #         plt.plot(V_damping_z, [(d[2,2]) for d in damping_theta_list], label='Cae z2')

                            
                        #     plt.legend()
                        #     plt.title('Sammenligning av demping (z DOF)')
                        #     plt.xlabel("Vindhastighet [m/s]")
                        #     plt.ylabel("Demping [Ns/m]")
                        #     plt.grid(True)
                        #     plt.tight_layout()
                        #     plt.show()
                        #     plt.close()

                        # if j == 1 and V==100: 
                        #     plt.plot(V_damping_theta, [(Cs[1,1] - d[1,1]) for d in damping_z_list], label='Cs-Cae rot')
                        #     plt.plot(V_damping_theta, [(d[1,1]) for d in damping_z_list], label='Cae rot')
                        #     plt.axhline(Cs[1,1], color='grey', linestyle='--', label='Cs rot')
                        #     if n_modes ==4:
                        #         plt.plot(V_damping_z, [(Cs[3,3] - d[3,3]) for d in damping_theta_list], label='Cs-Cae rot2')
                        #         plt.plot(V_damping_z, [(d[3,3]) for d in damping_theta_list], label='Cae rot2')
                        #     plt.legend()
                        #     plt.title('Sammenligning av demping (rot. DOF)')
                        #     plt.xlabel("Vindhastighet [m/s]")
                        #     plt.ylabel("Demping [Ns/m]")
                        #     plt.grid(True)
                        #     plt.tight_layout()
                        #     plt.show()
                        #     plt.close()
                        
                        # if V == 100:
                        #     fig, axes = plt.subplots(n_modes, n_modes, figsize=(4 * n_modes, 3 * n_modes), sharex=True)
                        #     fig.suptitle("Cae", fontsize=16)
                        #     for i in range(n_modes):
                        #         for j in range(n_modes):
                        #             ax = axes[i, j] if n_modes > 1 else axes
                        #             if j == 0:
                        #                 ax.plot(V_damping_z, [(d[i,j]) for d in damping_z_list])
                        #             if j == 1:
                        #                 ax.plot(V_damping_theta, [(d[i,j]) for d in damping_theta_list])
                        #             if not single:
                        #                 if j == 2:
                        #                     ax.plot(V_damping_z, [(d[i,j]) for d in damping_z_list])
                        #                 if j == 3:
                        #                     ax.plot(V_damping_theta, [(d[i,j]) for d in damping_theta_list])
                        #             #ax.set_ylabel(i,j)
                        #             ax.grid(True)
                        #     plt.tight_layout(rect=[0, 0, 1, 0.96])
                        #     plt.show()


                    stopFreq = 1 # Stopper frekvens-iterasjonen hvis vi har funnet flutter
                    
                        V -= dV
                        dV *= 0.5 #Hvis flutter er funnet, går tilbake litt i hastighet, for å gjøre søket mer presist.
                        omega_all[iterWind,j]=omega_new    
                        damping_ratios[iterWind,j]=damping_new
                        eigvals_all[iterWind,j]=λj                
                        eigvecs_all[iterWind,j]=φj

                        if j ==0:
                            damping_z_list.append(Cae_gen.copy())  # legg inn her
                            V_damping_z.append(V)

                        if j == 1:
                            damping_theta_list.append(Cae_gen.copy())
                            V_damping_theta.append(V)

                        converge = True              

                    else: # For the next iteration, use the new values as the old ones
                        # Timeout check
                        iterFreq += 1

                        if time.time() - start_time > timeout:
                            print(f"WARNING: Convergence timeout at V = {V:.2f} m/s, mode {j+1}. Setting results to NaN.")
                            omega_all[iterWind, j] = np.nan
                            damping_ratios[iterWind, j] = np.nan
                            eigvals_all[iterWind, j] = np.nan
                            eigvecs_all[iterWind, j] = None
                            #print("iteration:", iter)
                            break

                        omega_old[j] = omega_new
                        damping_old[j] = damping_new
                        eigvec_old[j] = φj
                    
                    if iterFreq == 10:
                        print(f"WARNING: Frequancy iteration has not converged for V = {V:.2f} m/s, mode {j+1}. Setting results to NaN.")
                        stopWind = 1
                        converge = True


            posres = np.where(np.abs(np.imag(eigvals)) == omegacr)[0] # posisjon til rett egenverdi) må hentes når du er ferdig med å iterere på w
            # ResMat har 2 rader: første for ω, andre for ξ
            # Hvis flere treff på ωcr, skriver de inn flere kolonner
            if len(posres) > 2: # Hvis vi fant flere egenverdier som matcher ωcr
                ResMat[0, 2*v-1:2*v-1+len(posres)] = np.imag(eigvals[posres])
                ResMat[1, 2*v-1:2*v-1+len(posres)] = np.real(eigvals[posres])
                v = v + len(posres)//2 - 1
            else:
                ResMat[0, 2*v-1:2*v] = np.imag(eigvals[posres])
                ResMat[1, 2*v-1:2*v] = np.real(eigvals[posres])
        
        stabind = np.max(ResMat[1, ResMat[1,:] != 0]) # største reel del, brukes til å sjekke om det finnes noen uastabile egenverdier
        iterWind += 1
        if stabind > 0: #flutter observert
            V -= 0.5 * dV
            dV *= 0.5
        else: #system stabilt
            V[pos] = V
            omega[:, pos] = ResMat[0, :]
            damping[:, pos] = ResMat[1, :]
            pos += 1
            V += dV

        if np.abs(stabind) < 1e-7: # hvis stabind = 0, akkurat ved flutter!
        # kritisk frekvens og vindhastighet lagres
            stopWind = 1
            Demping = np.max(ResMat[1, :])
            posCR = np.argmax(ResMat[1, :])
            OmegaCR = ResMat[0, posCR]
            VCR = V

        ResMat=zeros(2,2*size(MM,1))
        omega(:,pos)=ResMat(1,:)'
        damping(:,pos)=ResMat(2,:)'

    return damping_ratios, omega_all, eigvals_all, eigvecs_all

def solve_flutter_speed(damping_ratios, V_list, single = True):
    """
    Finds the flutter speed for each mode where the damping becomes negative.

    Parameters:
    -----------
    damping_ratios : list of arrays
        Damping ratios for each mode (shape: list of length n_modes with N elements each).
    V_list
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
     

     
def plot_damping_vs_wind_speed_single(damping_ratios, V_list, dist="Fill in dist",  single = True):
    """
    Plot damping ratios as a function of wind speed, and mark AD-validity range.

    Parameters:
    -----------
    
    damping_ratios : list of arrays
        Global damping ratios per mode.
    omega_all : list of arrays
        Global angular frequencies per mode..
    """

    markers = ['o', 's', '^', 'x']
    markersizes = [2.6, 2.4, 3, 3]
    colors = ['blue', 'green', 'red', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    plt.figure(figsize=(10, 6))

    if single:
        n_modes = 2 
        title = f"Damping vs. wind speed - {dist}"
    else:
        n_modes = 4
        title = f"Damping vs. wind speed - {dist}"

    for j in range(n_modes):
        plt.plot(V_list, damping_ratios[:, j], color=colors[j], marker=markers[j],  markersize=markersizes[j],  linestyle='None', label=labels[j])

    plt.axhline(0, linestyle="--", color="grey", linewidth=1.1, label="Critical damping")
    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Damping ratio [-]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.ylim(-0.05, )
    plt.xlim(0,)
    plt.show()

def plot_frequency_vs_wind_speed(B, Vred_defined, V_list, omega_all,   dist="Fill in dist", single = True):
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
 
    single : bool
    """
    markers = ['o', 's', '^', 'x']
    markersizes = [2.6, 2.4, 3, 3]
    colors = ['blue', 'green', 'red', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    frequencies = omega_all / (2 * np.pi)       # Convert to Hz

    plt.figure(figsize=(10, 6))

    if single:
        n_modes = 2 
        title = f"Natural frequencies vs wind speed - {dist}"
    else:
        n_modes = 4
        title = f"Natural frequencies vs wind speed - {dist}"

    for j in range(n_modes):
        plt.plot(V_list, frequencies[:, j], color=colors[j], marker = markers[j], markersize = markersizes[j], label = labels[j], linestyle='None')

    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Natural frequency [Hz]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    #plt.ylim(0.12, 0.16)
    plt.xlim(0,V_list[-1])
    plt.show()

def plot_flutter_mode_shape(eigvecs_all, flutter_idx_modes, dist="Fill in dist", single = True):
    
    flutter_idx = np.nanmin(flutter_idx_modes) 
    critical_mode = np.nanargmin(flutter_idx_modes)
    flutter_vec = eigvecs_all[int(flutter_idx),critical_mode] # Velg riktig mode ved flutter


    # 2 subplot: én for amplitudene, én for fasevinklene. De deler x-aksen
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    dofs = ["V1", "T1", "V2", "T2"]

    if single:
        n_modes = 2
    else:
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
    ax[1].set_ylim(-100, 100)
    ax[1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    ax[0].set_title(f"Magnitude and phase of normalized eigenvector at flutter wind speed - {dist}", fontsize=14)
    ax[1].set_xlabel("DOFs")

    for i in range(n_modes):
        if abs(magnitudes[i]) > 1e-3: 
            ax[0].text(i, magnitudes[i] + 0.05, f"{magnitudes[i]:.2f}", ha='center', fontsize=9)

        if abs(phases[i]) > 1: 
            ax[1].text(i, phases[i] + 10*np.sign(phases[i]), f"{phases[i]:.1f}°", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    if np.all(np.isnan(flutter_idx_modes)):
        print("Ingen flutter observert for noen moder!")
        return None



#def plot_compare_with_single():

# def plot_compare_with_dist(V_list, flutter_idx_modes1, flutter_idx_modes2,flutter_idx_modes3,flutter_idx_modes4,flutter_idx_modes5,damping_ratios1, damping_ratios2, damping_ratios3, damping_ratios4, damping_ratios5, dist="Fill in dist", single = True):

#     colors = ['blue', 'green', 'red', 'orange', 'purple']
#     labels = ['1D', '2D', '3D', '4D', '5D']

#     plt.figure(figsize=(10, 6))
#     damping_1D = damping_ratios1[:,np.nanargmin(flutter_idx_modes1)] # Velg riktig mode ved flutter
#     damping_2D = damping_ratios2[:,np.nanargmin(flutter_idx_modes2)] 
#     damping_3D = damping_ratios3[:,np.nanargmin(flutter_idx_modes3)] 
#     damping_4D = damping_ratios4[:,np.nanargmin(flutter_idx_modes4)] 
#     damping_5D = damping_ratios5[:,np.nanargmin(flutter_idx_modes5)] 

#     damping = [damping_1D, damping_2D, damping_3D, damping_4D, damping_5D]

#     for j in range(len(damping)):
#         plt.plot(V_list, damping[j], color=colors[j], label=labels[j])

    
    