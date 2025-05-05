

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

def from_poly_k(poly_k, k_range, vred, damping_ad = True):
    if vred == 0:
        vred = 1e-10
    uit_step = lambda k,kc: 1./(1 + np.exp(-2*20*(k-kc)))
    fit = lambda p,k,k1c,k2c : np.polyval(p,k)*uit_step(k,k1c)*(1-uit_step(k,k2c)) + np.polyval(p,k1c)*(1-uit_step(k,k1c)) + np.polyval(p,k2c)*(uit_step(k,k2c))

    if damping_ad == True:
        ad_value = np.abs(vred)*fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
    else:
        ad_value = np.abs(vred)**2*fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
   
    #ad_value = fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
 
    return float(ad_value)
 

def cae_kae_single(poly_coeff, k_range, Vred_global, Phi, x, B):
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
    H1, H2, H3, H4 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True), from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True), from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=False), from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=False)
    A1, A2, A3, A4 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True), from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True), from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=False), from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=False)

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
    phiphi_p = np.transpose(Phi, [1, 2, 0])  # (DOF, mode, x)

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
        
    return Cae_star_gen, Kae_star_gen

def cae_kae_twin(poly_coeff, k_range, Vred_global, Phi, x, B):
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
    c_z1z1, c_z1θ1, c_z1z2, c_z1θ2 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True), from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True), from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=True), from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=True)
    c_θ1z1, c_θ1θ1, c_θ1z2, c_θ1θ2 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True), from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True), from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=True), from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=True)
    c_z2z1, c_z2θ1, c_z2z2, c_z2θ2 = from_poly_k(poly_coeff[8], k_range[8],Vred_global, damping_ad=True), from_poly_k(poly_coeff[9], k_range[9],Vred_global, damping_ad=True), from_poly_k(poly_coeff[10], k_range[10],Vred_global, damping_ad=True), from_poly_k(poly_coeff[11], k_range[11],Vred_global, damping_ad=True)
    c_θ2z1, c_θ2θ1, c_θ2z2, c_θ2θ2 = from_poly_k(poly_coeff[12], k_range[12],Vred_global, damping_ad=True), from_poly_k(poly_coeff[13], k_range[13],Vred_global, damping_ad=True), from_poly_k(poly_coeff[14], k_range[14],Vred_global, damping_ad=True), from_poly_k(poly_coeff[15], k_range[15],Vred_global, damping_ad=True)
    k_z1z1, k_z1θ1, k_z1z2, k_z1θ2 = from_poly_k(poly_coeff[16], k_range[16],Vred_global, damping_ad=False), from_poly_k(poly_coeff[17], k_range[17],Vred_global, damping_ad=False), from_poly_k(poly_coeff[18], k_range[18],Vred_global, damping_ad=False), from_poly_k(poly_coeff[19], k_range[19],Vred_global, damping_ad=False)
    k_θ1z1, k_θ1θ1, k_θ1z2, k_θ1θ2 = from_poly_k(poly_coeff[20], k_range[20],Vred_global, damping_ad=False), from_poly_k(poly_coeff[21], k_range[21],Vred_global, damping_ad=False), from_poly_k(poly_coeff[22], k_range[22],Vred_global, damping_ad=False), from_poly_k(poly_coeff[23], k_range[23],Vred_global, damping_ad=False)
    k_z2z1, k_z2θ1, k_z2z2, k_z2θ2 = from_poly_k(poly_coeff[24], k_range[24],Vred_global, damping_ad=False), from_poly_k(poly_coeff[25], k_range[25],Vred_global, damping_ad=False), from_poly_k(poly_coeff[26], k_range[26],Vred_global, damping_ad=False), from_poly_k(poly_coeff[27], k_range[27],Vred_global, damping_ad=False)
    k_θ2z1, k_θ2θ1, k_θ2z2, k_θ2θ2 = from_poly_k(poly_coeff[28], k_range[28],Vred_global, damping_ad=False), from_poly_k(poly_coeff[29], k_range[29],Vred_global, damping_ad=False), from_poly_k(poly_coeff[30], k_range[30],Vred_global, damping_ad=False), from_poly_k(poly_coeff[31], k_range[31],Vred_global, damping_ad=False)
    

    # Initialiser 32 matriser (16 for C, 16 for K)
    AA = np.zeros((32, 4, 4))

    # Demping: fyll inn de 16 første
    AA[0, 0, 0] = c_z1z1
    AA[1, 0, 1] = B * c_z1θ1
    AA[2, 0, 2] = c_z1z2
    AA[3, 0, 3] = B * c_z1θ2

    AA[4, 1, 0] = B * c_θ1z1
    AA[5, 1, 1] = B**2 *c_θ1θ1
    AA[6, 1, 2] = B *  c_θ1z2
    AA[7, 1, 3] = B**2 * c_θ1θ2

    AA[8, 2, 0] = c_z2z1
    AA[9, 2, 1] = B *c_z2θ1
    AA[10, 2, 2] = c_z2z2
    AA[11, 2, 3] = B * c_z2θ2

    AA[12, 3, 0] = B *c_θ2z1
    AA[13, 3, 1] = B**2 * c_θ2θ1
    AA[14, 3, 2] = B * c_θ2z2
    AA[15, 3, 3] = B**2 * c_θ2θ2

    # Stivhet: fyll inn de 16 neste
    AA[16, 0, 0] = k_z1z1
    AA[17, 0, 1] = B * k_z1θ1
    AA[18, 0, 2] = k_z1z2
    AA[19, 0, 3] = B * k_z1θ2

    AA[20, 1, 0] = B * k_θ1z1
    AA[21, 1, 1] = B**2 * k_θ1θ1
    AA[22, 1, 2] = B *k_θ1z2
    AA[23, 1, 3] = B**2 * k_θ1θ2

    AA[24, 2, 0] = k_z2z1
    AA[25, 2, 1] = B * k_z2θ1
    AA[26, 2, 2] =k_z2z2
    AA[27, 2, 3] = B *  k_z2θ2

    AA[28, 3, 0] = B * k_θ2z1
    AA[29, 3, 1] = B**2 * k_θ2θ1
    AA[30, 3, 2] = B *k_θ2z2
    AA[31, 3, 3] = B**2 * k_θ2θ2

    AA_tilde = np.zeros((32, 4, 4))  # Generalisert aero-matriser
    phiphi_p = np.transpose(Phi, [1, 2, 0])  # (DOF, mode, x)

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
    Kae_star_gen = np.sum(AA_tilde[16:32], axis=0)
    return Cae_star_gen, Kae_star_gen


def solve_omega(poly_coeff,k_range, Ms, Cs, Ks, f1, f2, B, rho, eps, Phi, x, single = True, verbose=True):
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
        n_modes = 2 # 2 modes for single deck
        omega_old = np.array([2*np.pi*f1, 2*np.pi*f2])
        #dominant_dofs = [0, 1]  # z, θ
    else:   
        n_modes = 4 # 4 modes for twin deck
        omega_old = np.array([2*np.pi*f1, 2*np.pi*f2, 2*np.pi*f1, 2*np.pi*f2]) #Først brudekke 1, deretter brudekke 2
        #dominant_dofs = [0, 1, 2, 3]  # z1, θ1, z2, θ2
   
    # Flutter detection
    Vcritical = None # Critical wind speed
    omegacritical = None # Critical frequency


    stopWind = False
    iterWind = 0
    maxIterWind = 400
    V = 1.0 # Initial wind speed, m/s
    dV = 0.5 # Hvor mye vi øker vindhastighet per iterasjon. 

    # # Global results
    V_list = [] # Wind speed, m/s

    skip_mode = [False] * n_modes
    
    zeta = 0.005 # Damping ratio for the structure (assumed equal for both modes)

    # empty() creates an array of the given shape and type, EVERY ELEMENT MUST BE INITIALIZED LATER
    omega_all = np.zeros((maxIterWind, n_modes))
    damping_ratios = np.zeros((maxIterWind, n_modes))
    eigvals_all = np.zeros((maxIterWind, n_modes), dtype=complex)
    eigvecs_all = np.empty((maxIterWind, n_modes), dtype=object)

    V_list.append(0.0) # V = 0.0 m/s

    for j_mode in range(n_modes):
        eigvecs_all[0, j_mode] = np.nan
        eigvals_all[0,j_mode] = np.nan

        omega_all[0,j_mode] = omega_old[j_mode] # Startverdi for omega er den naturlige frekvensen til modusen
        damping_ratios [0,j_mode] = zeta
  
    velocity_counter = 1

    while (iterWind < maxIterWind and not stopWind): # iterer over vindhastigheter
        
        if verbose:
            print(f"Wind speed iteration {iterWind+1}: V = {V} m/s")

        for j in range(n_modes): # 4 modes for twin deck, 2 modes for single deck
            if skip_mode[j]:
                continue
            
            if verbose:
                print("Mode:", j+1)
          
            stopFreq = False
            iterFreq = 0 # Teller hvor mange frekvens-iterasjoner
            maxIterFreq = 1000 # Maks antall iterasjoner for frekvens



            while (iterFreq < maxIterFreq and not stopFreq): # iterer over frekvens-iterasjoner           

                Vred = V/(omega_old[j]*B) # reduced velocity 

                if single:
                    Cae_star_gen, Kae_star_gen = cae_kae_single(poly_coeff, k_range, Vred, Phi, x, B)

                else:
                    Cae_star_gen, Kae_star_gen = cae_kae_twin(poly_coeff,k_range,  Vred,Phi, x, B)

                Cae_gen = 0.5 * rho * B**2 * omega_old[j] * Cae_star_gen
                Kae_gen = 0.5 * rho * B**2 * omega_old[j]**2 * Kae_star_gen

                eigvalsV, eigvecsV = solve_eigvalprob(Ms, Cs, Ks, Cae_gen, Kae_gen)


                eigvals_pos = eigvalsV[np.imag(eigvalsV) > 0]
                eigvecs_pos = eigvecsV[:, np.imag(eigvalsV) > 0]  

                
                if eigvals_pos.size == 0:
                    if verbose:
                        print(f"Ingen komplekse egenverdier ved V = {V:.2f} m/s, mode {j+1}. Skipper mode.")
                    omega_all[velocity_counter, j] = np.nan
                    damping_ratios[velocity_counter, j] = np.nan
                    eigvals_all[velocity_counter, j] = np.nan
                    eigvecs_all[velocity_counter, j] = None
                    break


                if single:
                    best_idx = np.argmin(np.abs(np.imag(eigvals_pos) - omega_old[j]))
                    λj = eigvals_pos[best_idx]
                    φj = eigvecs_pos[:n_modes, best_idx]
                    omega_new = np.imag(λj)
                    damping_new = -np.real(λj) / np.abs(λj)
                else:
                    # dominance_scores = np.array([np.abs(eigvecs_pos[j, idx]) for idx in range(eigvecs_pos.shape[1])])
                    # prev_lambda = eigvals_all[velocity_counter - 1, j] 

                    # if V < 60:
                    #     best_idx = np.argmax(dominance_scores) # Velg den største egenvektoren
                    # else:
                    #     best_idx = np.argmin(
                    #             np.abs(np.imag(eigvals_pos) - omega_old[j])
                    #             + 10 * np.abs(np.real(eigvals_pos) - np.real(prev_lambda))
                    # #     )
                    # mode_order = []
                    # for d in range(n_modes):
                    #     best_mode = np.argmax(np.abs(eigvecs_pos[d, :]))  # Hvilken DOF er størst i denne modusen?
                    #     mode_order.append(best_mode)  # Lagre dette

                    # λj = eigvals_pos[mode_order[j]]
                    # φj = eigvecs_pos[:n_modes, mode_order[j]]
                    # omega_new = np.imag(λj)
                    # damping_new = -np.real(λj) / np.abs(λj)

                    # print("mode_order =", mode_order)
                    # print("unique =",mode_order[j])

                    # -- Start: Robust og entydig mapping mellom DOFs og egenverdier --

                    n_eig = eigvecs_pos.shape[1]  # Antall egenverdier/moder
                    available = set(range(n_eig))  # Hvilke indekser er fortsatt "ledig" å bruke
                    mode_order = []

                    # For hver DOF (altså fysisk mode), finn den egenmoden som er mest dominant der,
                    # men sørg for at ingen mode brukes to ganger
                    for dof in range(n_modes):
                        best_score = -np.inf
                        best_idx = -1
                        for idx in available:
                            score = np.abs(eigvecs_pos[dof, idx])  # Sjekk hvor dominant DOF `dof` er i egenvektor `idx`
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                        if best_idx != -1:
                            mode_order.append(best_idx)
                            available.remove(best_idx)  # Fjern så den ikke brukes igjen

                    # Nå har du en unik mapping: j'te DOF → riktig indeks i eigvals_pos
                    λj = eigvals_pos[mode_order[j]]
                    φj = eigvecs_pos[:n_modes, mode_order[j]]

                    # Beregn ny omega og demping
                    omega_new = np.imag(λj)
                    damping_new = -np.real(λj) / np.abs(λj)

                    # Debug-print for kontroll
                    if verbose:
                        print("mode_order =", mode_order)
                        print("unique =", mode_order[j])

                # if verbose:
                #     print(f"[DEBUG] IterFreq {iterFreq}, omega_old = {omega_old[j]:.4f}, omega_new = {omega_new:.4f}, diff = {np.abs(omega_old[j] - omega_new):.4e}")

                # if verbose:
                #     print(f"  IterFreq {iterFreq}: omega_old = {omega_old[j]:.5f}, omega_new = {omega_new:.5f}, diff = {np.abs(omega_old[j] - omega_new):.2e}")
                #     print(f"  V_red = {Vred:.5f}")

                                            
                if np.abs(omega_old[j] - omega_new) < eps or omega_old[j] <= 0.0: # omega har konvergert, jippi
                    omega_all[velocity_counter, j] = omega_new
                    damping_ratios[velocity_counter, j] = damping_new
                    eigvals_all[velocity_counter, j] = λj
                    eigvecs_all[velocity_counter, j] = φj
                    stopFreq = True # Stopper frekvens-iterasjonen hvis vi har funnet flutter

                iterFreq += 1
                omega_old[j] = omega_new

                if iterFreq == 1000:
                    if verbose:
                        print(f"WARNING: Frequancy iteration has not converged for V = {V:.2f} m/s, mode {j+1}. Setting results to NaN.")
                    omega_all[velocity_counter, j] = np.nan
                    damping_ratios[velocity_counter, j] = np.nan
                    eigvals_all[velocity_counter, j] = np.nan
                    eigvecs_all[velocity_counter, j] = None
                    break

         


            if damping_new < 0: # flutter finnes, må finne nøyaktig flutterhastighe
                for other_j in range(n_modes): #Flutter funnet på en mode, fokuser kunn på denne moden, og stopper til slutt hastighetiterasjonen.
                    if other_j != j:
                        skip_mode[other_j] = True

                if verbose:
                    print(f"Flutter detected at V = {V:.2f} m/s, iterWind = {iterWind}, j = {j}")

                if np.abs(damping_new) < 0.0000001: #Akkurat ved flutter
                    if verbose:
                        print(f"Flutter converged at V ≈ {V:.5f} m/s")
                    omegacritical = omega_new
                    Vcritical = V
                    skip_mode[j] = True  

                
            else: #system stabilt
                omega_all[velocity_counter, j] = omega_new
                damping_ratios[velocity_counter, j] = damping_new
                eigvals_all[velocity_counter, j] = λj   
                eigvecs_all[velocity_counter, j] = φj

            if dV < 1e-8:
                if verbose:
                    print(f"Stopping refinement: dV too small ({dV:.2e}). Flutter converged at V ≈ {V:.5f} m/s")
                Vcritical = V
                omegacritical = omega_new
                skip_mode = [True] * n_modes
                stopWind = True
        
        if all(skip_mode):
            stopWind = True

        if all(not s for s in skip_mode):  
            V_list.append(V)
            V += dV
            velocity_counter += 1
        elif not all(skip_mode):  # flutter har startet, men ikke funnet nøyaktig
            V -= 0.5 * dV
            dV *= 0.5

        iterWind +=1


    # Truncate arrays to actual size
    omega_all = omega_all[:velocity_counter, :]
    damping_ratios = damping_ratios[:velocity_counter, :]
    eigvals_all = eigvals_all[:velocity_counter, :]
    eigvecs_all = eigvecs_all[:velocity_counter, :]

        
    return V_list, omega_all, damping_ratios, eigvecs_all, eigvals_all, omegacritical, Vcritical 

def plot_damping_vs_wind_speed(damping_ratios, V_list, dist="Fill in dist", single=True):
    """
    Plot damping ratios as a function of wind speed.

    Parameters:
    -----------
    damping_ratios : ndarray, shape (Nvind, n_modes)
        Damping ratios for each mode and wind speed.
    V_list : list or ndarray
        Wind speed values.
    dist : str
        Description of deck distance or test case.
    single : bool
        True if single-deck (2 modes), False if twin-deck (4 modes).
    """
    markers = ['o', 's', '^', 'x']
    markersizes = [2.6, 2.4, 3, 3]
    colors = ['blue', 'green', 'red', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    plt.figure(figsize=(10, 6))
    
    n_modes = 2 if single else 4
    title = f"Damping vs. wind speed - {dist}"

    for j in range(n_modes):
        plt.plot(
            V_list,
            damping_ratios[:, j],
            color=colors[j],
            marker=markers[j],
            markersize=markersizes[j],
            linestyle='None',
            label=labels[j]
        )

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
    plt.xlim(0, )
    plt.show()


def plot_frequency_vs_wind_speed(V_list, omega_list, dist="Fill in dist", single=True):
    """
    Plots natural frequencies as a function of wind speed.

    Parameters:
    -----------
    V_list : list
        Wind speeds.
    omega_list : ndarray
        Angular frequencies (shape: N x n_modes).
    dist : str
        Description for plot title.
    single : bool
        True for 2DOF (single-deck), False for 4DOF (twin-deck).
    """

    markers = ['o', 's', '^', 'x']
    markersizes = [2.6, 2.4, 3, 3]
    colors = ['blue', 'green', 'red', 'orange']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    n_modes = 2 if single else 4
    title = f"Natural frequencies vs wind speed - {dist}"

    omega_array = np.array(omega_list)
    frequencies = omega_array / (2 * np.pi)  # Convert from rad/s to Hz

    plt.figure(figsize=(10, 6))
    for j in range(n_modes):
        plt.plot(
            V_list,
            frequencies[:, j],
            color=colors[j],
            marker=markers[j],
            markersize=markersizes[j],
            linestyle='None',
            label=labels[j]
        )

    plt.xlabel("Wind speed [m/s]", fontsize=16)
    plt.ylabel("Natural frequency [Hz]", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.xlim(0, V_list[-1])
    plt.show()

def plot_flutter_mode_shape(eigvecs_all, damping_list, V_list, Vcritical, omegacritical, dist="Fill in dist", single=True):
    if Vcritical is None or omegacritical is None:
        print("Ingen flutter observert!")
        return

    if single:
        n_modes = 2
        dofs = ["V1", "T1"]
    else:
        n_modes = 4
        dofs = ["V1", "T1", "V2", "T2"]
    

    idx_flutter = np.argmin(np.abs(np.array(V_list) - Vcritical))

 
    damping_array = np.array(damping_list)  # shape (Nvind, n_modes)
    last_damping = damping_array[idx_flutter]  # siste vindhastighet synlig
    idx_mode_flutter = np.argmin(last_damping)  # den moden som først får negativ demping

    # Hent ut tilhørende egenvektor
    flutter_vec = eigvecs_all[idx_flutter][idx_mode_flutter]

    # 2 subplot: magnituder og faser
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    abs_vec = np.abs(flutter_vec)
    max_idx = np.argmax(abs_vec)
    normalized_vec = flutter_vec / flutter_vec[max_idx]

    magnitudes = np.abs(normalized_vec)
    phases = np.angle(normalized_vec, deg=True)

    ax[0].bar(dofs, magnitudes, color='blue', width=0.5)
    ax[1].bar(dofs, phases, color='orange', width=0.5)

    ax[0].set_ylabel(r"|$\Phi$| [-]")
    ax[0].set_ylim(0, 1.1)
    ax[1].set_ylabel(r"∠$\Phi$ [deg]")
    ax[1].set_ylim(-230, 230)
    ax[1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    ax[0].set_title(f"Magnitude and phase of normalized eigenvector at flutter wind speed - {dist}", fontsize=14)
    ax[1].set_xlabel("DOFs")

    for i in range(n_modes):
        if abs(magnitudes[i]) > 1e-3:
            ax[0].text(i, magnitudes[i] + 0.02, f"{magnitudes[i]:.2f}", ha='center', fontsize=9)
        if abs(phases[i]) > 1:
            ax[1].text(i, phases[i] + 10*np.sign(phases[i]), f"{phases[i]:.1f}°", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

