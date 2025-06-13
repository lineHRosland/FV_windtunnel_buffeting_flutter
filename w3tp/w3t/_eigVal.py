# -*- coding: utf-8 -*-
"""
Created on Mon Feb 3 08:15:00 2025
Editited spring 2025
@author: Smetoch, Rosland
"""
import numpy as np
from scipy import linalg as spla
import matplotlib.pyplot as plt
import time
from mode_shapes import mode_shape_single
from mode_shapes import mode_shape_two
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']



def solve_eigvalprob(M_struc, C_struc, K_struc, C_aero, K_aero):
    """
    Solves the generalized complex eigenvalue problem for an aeroelastic system.

    The system is formulated as a second-order differential equation:
        [ -λ² M + λ(C - C_aero) + (K - K_aero) ] φ = 0

    This equation is recast into first-order state-space form and solved numerically
    to obtain the eigenvalues of the coupled structural-aerodynamic system.

    Parameters:
    -----------
    M_struc : np.ndarray
        Structural mass matrix
    C_struc : np.ndarray
        Structural damping matrix
    K_struc : np.ndarray
        Structural stiffness matrix
    C_aero : np.ndarray
        Aerodynamic damping matrix (dimensional)
    K_aero : np.ndarray
        Aerodynamic stiffness matrix (dimensional)

    Returns:
    --------
    eigvals : np.ndarray
        Complex eigenvalues λ, shape (2 * n_dof,)
    eigvecs : np.ndarray
        Corresponding right eigenvectors, shape (2 * n_dof, 2 * n_dof)
    """
    # Effective system matrices
  
    C = C_struc - C_aero
    K = K_struc - K_aero

    print(C,K)

    # Construct state-space system matrix A
    A = np.block([
        [np.zeros_like(M_struc), np.eye(M_struc.shape[0])],
        [-np.linalg.inv(M_struc) @ K, -np.linalg.inv(M_struc) @ C]
    ])

    # Solve the eigenvalue problem  
    eigvals, eigvecs = spla.eig(A)

    return eigvals, eigvecs

def generalize_C_K(C, K, Phi, x):
    """
    Generalizes the aerodynamic damping and stiffness matrices to modal coordinates
    by performing numerical integration over the spanwise distribution of mode shapes.

    Parameters:
    -----------
    C : np.ndarray (n_dof x n_dof)
        Aerodynamic damping matrix in physical degrees of freedom (DOFs).
    K : np.ndarray (n_dof x n_dof)
        Aerodynamic stiffness matrix in physical DOFs.
    Phi : np.ndarray (n_points x n_dof x n_modes)
        Mode shape matrix, evaluated at discrete points along the bridge span.
    x : np.ndarray (n_points,)
        Physical x-coordinates corresponding to the spanwise locations.
    single : bool, optional (default=True)
        If True, only the first N nodes are used (single-deck configuration).
        If False, both upstream and downstream decks are processed (two-deck).

    Returns:
    --------
    C_gen : np.ndarray (n_modes x n_modes)
        Generalized aerodynamic damping matrix in modal coordinates.
    K_gen : np.ndarray (n_modes x n_modes)
        Generalized aerodynamic stiffness matrix in modal coordinates.
    """
    N = len(x)

    n_modes = Phi.shape[2]

    Cae_star_gen = np.zeros((n_modes, n_modes))
    Kae_star_gen = np.zeros((n_modes, n_modes))

    for i in range(N-1): 
        dx = x[i+1] - x[i] 
        phi_L = Phi[i] # shape (n_dof, n_modes)
        phi_R = Phi[i+1]
        # Damping
        C_int = 0.5 * (phi_L.T @ C @ phi_L + phi_R.T @ C @ phi_R)
        Cae_star_gen += C_int * dx
        # Stiffness
        K_int = 0.5 * (phi_L.T @ K @ phi_L + phi_R.T @ K @ phi_R)
        Kae_star_gen += K_int * dx
    

    
    return Cae_star_gen, Kae_star_gen
    

 
def structural_matrices(m1, m2, f1, f2, zeta, single = True):
    """
    Constructs structural mass, damping, and stiffness matrices for either single-deck 
    (2-DOF) or two-deck (4-DOF) bridge systems in modal coordinates.

    Parameters:
    -----------
    m1 : float
        Modal mass for the vertical mode [kg].
    m2 : float
        Modal mass for the torsional mode [kg].
    f1 : float
        Natural frequency of the vertical mode [Hz].
    f2 : float
        Natural frequency of the torsional mode [Hz].
    zeta : float
        Structural damping ratio (assumed equal for both modes).
    single : bool, optional (default=True)
        If True, returns 2x2 matrices corresponding to a single-deck system.
        If False, returns 4x4 block-diagonal matrices representing two identical decks.

    Returns:
    --------
    Ms : np.ndarray
        Structural mass matrix (2x2 or 4x4).
    Cs : np.ndarray
        Structural damping matrix (2x2 or 4x4).
    Ks : np.ndarray
        Structural stiffness matrix (2x2 or 4x4).
    """
    # Stiffness
    k1 = (2 * np.pi * f1) ** 2 * m1  #  (Vertical)
    k2 = (2 * np.pi * f2) ** 2 * m2  #  (Torsion)
    
    # Damping
    c1 = 2 * zeta * m1 * np.sqrt(k1 / m1)  # (Vertical)
    c2 = 2 * zeta * m2 * np.sqrt(k2 / m2)  # (Torsion)

    # 2x2 structural matrices (single deck)
    Ms = np.array([[m1,0],[0,m2]])  # Mass matrix
    Cs = np.array([[c1,0],[0,c2]])  # Damping matrix
    Ks = np.array([[k1,0],[0,k2]])  # Stiffness matrix

    # Expand to 4x4 block-diagonal matrices for two-deck system
    if not single:
        Ms = np.array([[m1,0,0,0],[0,m2,0,0],[0,0,m1,0],[0,0,0,m2]])  # Mass matrix
        Cs = np.array([[c1,0,0,0],[0,c2,0,0],[0,0,c1,0],[0,0,0,c2]])  # Damping matrix
        Ks = np.array([[k1,0,0,0],[0,k2,0,0],[0,0,k1,0],[0,0,0,k2]])  # Stiffness matrix

    return Ms, Cs, Ks

def from_poly_k(poly_k, k_range, vred, damping_ad = True):
    """
    Evaluates an aerodynamic derivative based on reduced velocity using a smoothed polynomial approximation.

    Parameters:
    -----------
    poly_k : np.ndarray
        Polynomial coefficients (e.g., degree 2) for the aerodynamic derivative.
    k_range : tuple (float, float)
        Valid reduced frequency range (k_min, k_max) over which the polynomial was fitted.
    vred : float
        Reduced velocity V / (ω B). Will be internally converted to reduced frequency via k = 1 / vred.
    damping_ad : bool, optional (default=True)
        If True, multiplies the polynomial result by |v_red| (linear scaling for damping terms).
        If False, multiplies by |v_red|² (quadratic scaling for stiffness terms).

    Returns:
    --------
    ad_value : float
        Evaluated aerodynamic derivative value at the given reduced velocity.
    """
    if vred == 0:
        vred = 1e-10 # Prevent division by zero
        
    uit_step = lambda k,kc: 1./(1 + np.exp(-2*20*(k-kc)))
    fit = lambda p,k,k1c,k2c : np.polyval(p,k)*uit_step(k,k1c)*(1-uit_step(k,k2c)) + np.polyval(p,k1c)*(1-uit_step(k,k1c)) + np.polyval(p,k2c)*(uit_step(k,k2c))

    if damping_ad == True:
        ad_value = np.abs(vred)*fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
    else:
        ad_value = np.abs(vred)**2*fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
   
    #ad_value = fit(poly_k,np.abs(1/vred),k_range[0],k_range[1])
 
    return float(ad_value)
 

def cae_kae_single(poly_coeff, k_range, Vred_global,  B):
    """
    Computes the generalized aerodynamic damping and stiffness matrices for a 
    single-deck bridge cross-section based on fitted polynomial aerodynamic derivatives.

    Parameters:
    -----------
    poly_coeff : np.ndarray, shape (8, 3)
        Polynomial coefficients for the aerodynamic derivatives:
        [H1, H2, H3, H4, A1, A2, A3, A4], where:
            - H1, H2: damping terms related to vertical DOF
            - A1, A2: damping terms related to torsional DOF
            - H3, H4: stiffness terms related to vertical DOF
            - A3, A4: stiffness terms related to torsional DOF
    k_range : list of tuples, length 8
        Valid reduced frequency ranges for each polynomial, used in smoothing.
    Vred_global : float
        Reduced velocity \( V / (B \omega) \) for the system.
    B : float
        Section width (m), used for non-dimensional scaling of derivatives.

    Returns:
    --------
    C_ae_star : np.ndarray, shape (2, 2)
        Generalized aerodynamic damping matrix (modal form).
    K_ae_star : np.ndarray, shape (2, 2)
        Generalized aerodynamic stiffness matrix (modal form).
    """
    Vred_global = float(Vred_global) 

    # AD
    # Evaluate aerodynamic damping derivatives
    H1 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True)
    H2 = from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True)
    A1 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True)
    A2 = from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True)

        
    # Evaluate aerodynamic stiffness derivatives
    H3 = from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=False)
    H4 = from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=False)
    A3 = from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=False) 
    A4 = from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=False) 

    Cae_star = np.array([
         [H1,       B * H2],
         [B * A1,   B**2 * A2]
     ])
    Kae_star = np.array([
         [H4,       B * H3],
         [B * A4,   B**2 * A3]
     ])        

     
    return Cae_star, Kae_star

def cae_kae_two(poly_coeff, k_range, Vred_global, B):
    """
    Computes the generalized aerodynamic damping and stiffness matrices for a 
    two-deck bridge configuration using fitted polynomial representations of 
    aerodynamic derivatives.

    Parameters:
    -----------
    poly_coeff : np.ndarray, shape (32, 3)
        Polynomial coefficients for 32 aerodynamic derivatives:
        - Indices 0 - 15: damping-related (multiplied by V_red)
        - Indices 16 - 31: stiffness-related (multiplied by V_red²)
    k_range : list of tuples
        Valid reduced frequency ranges (min, max) for each derivative.
    Vred_global : float
        Global reduced velocity \( V / (B \omega) \).
    B : float
        Section width (m).

    Returns:
    --------
    Cae_star : np.ndarray, shape (4, 4)
        Generalized aerodynamic damping matrix.
    Kae_star : np.ndarray, shape (4, 4)
        Generalized aerodynamic stiffness matrix.

    DOFs are ordered as: [z₁, θ₁, z₂, θ₂]
    where:
        - z₁, θ₁: vertical and torsional DOFs for the upstream deck
        - z₂, θ₂: vertical and torsional DOFs for the downstream deck
    """

    Vred_global = float(Vred_global) 

    # AD
    # Damping derivatives (indices 0–15)

    c_z1z1 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True)
    c_z1θ1 = from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True)
    c_z1z2 = from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=True)
    c_z1θ2 = from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=True)
    c_θ1z1 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True)
    c_θ1θ1 = from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True)
    c_θ1z2 = from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=True)
    c_θ1θ2 = from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=True)
    c_z2z1 = from_poly_k(poly_coeff[8], k_range[8],Vred_global, damping_ad=True)
    c_z2θ1 = from_poly_k(poly_coeff[9], k_range[9],Vred_global, damping_ad=True)
    c_z2z2 = from_poly_k(poly_coeff[10], k_range[10],Vred_global, damping_ad=True)
    c_z2θ2 = from_poly_k(poly_coeff[11], k_range[11],Vred_global, damping_ad=True)
    c_θ2z1 = from_poly_k(poly_coeff[12], k_range[12],Vred_global, damping_ad=True)
    c_θ2θ1 = from_poly_k(poly_coeff[13], k_range[13],Vred_global, damping_ad=True)
    c_θ2z2 = from_poly_k(poly_coeff[14], k_range[14],Vred_global, damping_ad=True)
    c_θ2θ2 = from_poly_k(poly_coeff[15], k_range[15],Vred_global, damping_ad=True)

    # Stiffness derivatives (indices 16–31)
    k_z1z1 = from_poly_k(poly_coeff[16], k_range[16],Vred_global, damping_ad=False)
    k_z1θ1 = from_poly_k(poly_coeff[17], k_range[17],Vred_global, damping_ad=False)
    k_z1z2 = from_poly_k(poly_coeff[18], k_range[18],Vred_global, damping_ad=False)
    k_z1θ2 = from_poly_k(poly_coeff[19], k_range[19],Vred_global, damping_ad=False)
    k_θ1z1 = from_poly_k(poly_coeff[20], k_range[20],Vred_global, damping_ad=False)
    k_θ1θ1 = from_poly_k(poly_coeff[21], k_range[21],Vred_global, damping_ad=False)
    k_θ1z2 = from_poly_k(poly_coeff[22], k_range[22],Vred_global, damping_ad=False)
    k_θ1θ2 = from_poly_k(poly_coeff[23], k_range[23],Vred_global, damping_ad=False)
    k_z2z1 = from_poly_k(poly_coeff[24], k_range[24],Vred_global, damping_ad=False)
    k_z2θ1 = from_poly_k(poly_coeff[25], k_range[25],Vred_global, damping_ad=False)
    k_z2z2 = from_poly_k(poly_coeff[26], k_range[26],Vred_global, damping_ad=False)
    k_z2θ2 = from_poly_k(poly_coeff[27], k_range[27],Vred_global, damping_ad=False)
    k_θ2z1 = from_poly_k(poly_coeff[28], k_range[28],Vred_global, damping_ad=False)
    k_θ2θ1 = from_poly_k(poly_coeff[29], k_range[29],Vred_global, damping_ad=False)
    k_θ2z2 = from_poly_k(poly_coeff[30], k_range[30],Vred_global, damping_ad=False)
    k_θ2θ2 = from_poly_k(poly_coeff[31], k_range[31],Vred_global, damping_ad=False)
    
    Cae_star = np.array([
         [c_z1z1,       B * c_z1θ1,       c_z1z2,       B * c_z1θ2],
         [B * c_θ1z1,   B**2 * c_θ1θ1,   B * c_θ1z2,   B**2 * c_θ1θ2],
         [c_z2z1,       B * c_z2θ1,       c_z2z2,       B * c_z2θ2],
         [B * c_θ2z1,   B**2 * c_θ2θ1,   B * c_θ2z2,   B**2 * c_θ2θ2]
    ])
 
    Kae_star = np.array([
         [k_z1z1,       B * k_z1θ1,       k_z1z2,       B * k_z1θ2],
         [B * k_θ1z1,   B**2 * k_θ1θ1,   B * k_θ1z2,   B**2 * k_θ1θ2],
         [k_z2z1,       B * k_z2θ1,       k_z2z2,       B * k_z2θ2],
         [B * k_θ2z1,   B**2 * k_θ2θ1,   B * k_θ2z2,   B**2 * k_θ2θ2]
    ])

    return Cae_star, Kae_star


def solve_flutter(poly_coeff,k_range, Ms, Cs, Ks,  f1, f2, B, rho, eps, 
                Phi, x, single = True, static_quasi = False, 
                Cae_star_STAT = None, Kae_star_STAT=None, verbose=True):
    """
    Solves the aeroelastic eigenvalue problem to determine flutter onset by iterating
    over wind speed and frequency for either a single-deck (2 DOF) or two-deck (4 DOF) bridge.

    Parameters
    ----------
    poly_coeff : np.ndarray, shape (8, 3) or (32, 3)
        Polynomial coefficients for aerodynamic derivatives (single or two-deck).
    k_range : list of tuple(float, float)
        Valid reduced frequency range for each aerodynamic derivative.
    Ms : np.ndarray
        Structural mass matrix.
    Cs : np.ndarray
        Structural damping matrix.
    Ks : np.ndarray
        Structural stiffness matrix.
    f1, f2 : float
        Natural frequencies of vertical and torsional modes [Hz].
    B : float
        Deck width [m].
    rho : float
        Air density [kg/m³].
    eps : float
        Frequency convergence tolerance (rad/s).
    Phi : np.ndarray
        Mode shape matrix (N x DOF x modes).
    x : np.ndarray
        Array of physical node coordinates along the span.
    single : bool, optional
        True for single-deck analysis (2 DOF). False for two-deck (4 DOF). Default is True.
    static_quasi : bool, optional
        If True, uses precomputed quasi-static matrices instead of aerodynamic derivatives.
    Cae_star_STAT : np.ndarray
        Precomputed generalized aerodynamic damping matrix for quasi-static.
    Kae_star_STAT : np.ndarray
        Precomputed generalized aerodynamic stiffness matrix for quasi-static.
    verbose : bool, optional
        If True, prints convergence info and dominant DOF tracking.

    Returns
    -------
    V_list : list of float
        Wind speeds [m/s] at each evaluation point.
    omega_all : np.ndarray
        Damped natural frequencies at each wind speed (shape: N_wind x n_modes).
    damping_ratios : np.ndarray
        Damping ratios at each wind speed (same shape as omega_all).
    eigvecs_all : np.ndarray
        Normalized eigenvectors at each wind speed (N_wind x n_modes).
    eigvals_all : np.ndarray
        Eigenvalues (complex λ = sigma + iω) for each mode and wind speed.
    omegacritical : float
        Critical flutter frequency (rad/s).
    Vcritical : float
        Critical flutter wind speed (m/s).

    Notes
    -----
    - Flutter is detected when the real part of an eigenvalue becomes negative.
    - The procedure stops once flutter is detected and refined below a damping threshold.
    - Frequency iteration is performed per mode and wind speed.
    - Dominant DOF tracking is used to match eigenvalues across iterations.
    """
    if single:
        n_modes = 2 # 2 modes for single deck
        omega_old = np.array([2*np.pi*f1, 2*np.pi*f2])
    else:   
        n_modes = 4 # 4 modes for two deck
        omega_old = np.array([2*np.pi*f1, 2*np.pi*f2, 2*np.pi*f1, 2*np.pi*f2]) 
    
    if static_quasi and (Cae_star_STAT is None or Kae_star_STAT is None):
        raise ValueError("Quasi-static analysis selected but aerodynamic matrices are not provided.")

   
    # Flutter detection
    Vcritical = None # Critical wind speed
    omegacritical = None # Critical frequency

    stopWind = False
    iterWind = 0
    maxIterWind = 1500
    V = 1.0 # Initial wind speed, m/s
    dV = 0.5 
    flutter_detected = False
    # # Global results
    V_list = [] # Wind speed, m/s

    skip_mode = [False] * n_modes
    
    zeta = 0.005 # Damping ratio for the structure 

    omega_all = np.full((maxIterWind, n_modes), np.nan)
    damping_ratios = np.full((maxIterWind, n_modes), np.nan)
    eigvals_all = np.full((maxIterWind, n_modes), np.nan + 1j*np.nan, dtype=complex)
    eigvecs_all = np.full((maxIterWind, n_modes), None, dtype=object)


    V_list.append(0.0) 

    for j_mode in range(n_modes):
        eigvecs_all[0, j_mode] = np.eye(n_modes)[:, j_mode] 
        eigvals_all[0,j_mode] = np.nan  

        omega_all[0,j_mode] = omega_old[j_mode] 
        damping_ratios [0,j_mode] = zeta
  
    velocity_counter = 1

    while (iterWind < maxIterWind and not stopWind): 
        
        if verbose:
            print(f"Wind speed iteration {iterWind+1}: V = {V} m/s")

        for j in range(n_modes): # 4 modes for two deck, 2 modes for single deck
            if skip_mode[j]:
                continue
            
            if verbose:
                print("Mode:", j+1)
          
            stopFreq = False
            iterFreq = 0 
            maxIterFreq = 1000 

            if static_quasi: 
                maxIterFreq = 1 # Quasi-static case only needs one iteration

            while (iterFreq < maxIterFreq and not stopFreq):
                Vred = V/(omega_old[j]*B) # reduced velocity 

                if single:
                    Cae_star_AD, Kae_star_AD= cae_kae_single(poly_coeff, k_range, Vred,  B)
                    Cae_star_gen_AD, Kae_star_gen_AD = generalize_C_K(Cae_star_AD, Kae_star_AD, Phi, x) 
                else:
                    Cae_star_AD, Kae_star_AD = cae_kae_two(poly_coeff,k_range,  Vred, B)
                    Cae_star_gen_AD, Kae_star_gen_AD = generalize_C_K(Cae_star_AD, Kae_star_AD, Phi, x) 

                if static_quasi:
                    Cae_star_gen_STAT,Kae_star_gen_STAT = generalize_C_K(Cae_star_STAT, Kae_star_STAT, Phi, x)

                    Cae_gen = V* Cae_star_STAT
                    Kae_gen = V**2* Kae_star_STAT

                else:      
                    Cae_gen = 0.5 * rho * B**2 * omega_old[j] * Cae_star_gen_AD
                    Kae_gen = 0.5 * rho * B**2 * omega_old[j]**2 * Kae_star_gen_AD

                eigvalsV, eigvecsV = solve_eigvalprob(Ms, Cs, Ks, Cae_gen, Kae_gen)


                eigvals_pos = eigvalsV[np.imag(eigvalsV) > 0]
                eigvecs_pos = eigvecsV[:, np.imag(eigvalsV) > 0]  


                if eigvals_pos.size == 0:
                    if verbose:
                        print(f"No complex eigenvalues at V = {V:.2f} m/s, mode {j+1}. Skipping mode.")
                    if not flutter_detected:
                        omega_all[velocity_counter, j] = np.nan
                        damping_ratios[velocity_counter, j] = np.nan
                        eigvals_all[velocity_counter, j] = np.nan
                        eigvecs_all[velocity_counter, j] = None
                    break
                
          
                if single:
                    best_idx = np.argmin(np.abs(np.imag(eigvals_pos) - omega_old[j]))
                else:
                    dominance_scores = np.array([np.abs(eigvecs_pos[j, idx]) for idx in range(eigvecs_pos.shape[1])])
                    prev_lambda = eigvals_all[velocity_counter - 1, j] 

                    if V < 10:
                        best_idx = np.argmax(dominance_scores) 
                    else:
                        best_idx = np.argmin(
                                np.abs(np.imag(eigvals_pos) - omega_old[j])
                                + 10 * np.abs(np.real(eigvals_pos) - np.real(prev_lambda))
                     )
                
                λj = eigvals_pos[best_idx]
                φj = eigvecs_pos[:n_modes, best_idx] #Choose the first n_modes eigenvectors (exclude lambda*phi)
                omega_new = np.imag(λj)
                damping_new = -np.real(λj) / np.abs(λj)

                if np.abs(omega_old[j] - omega_new) < eps or omega_old[j] <= 0.0:
                    if not flutter_detected:
                        omega_all[velocity_counter, j] = omega_new
                        damping_ratios[velocity_counter, j] = damping_new
                        eigvals_all[velocity_counter, j] = λj
                        eigvecs_all[velocity_counter, j] = φj
                    stopFreq = True 
                else:
                    if static_quasi:
                        if not flutter_detected:
                            omega_all[velocity_counter, j] = omega_new
                            damping_ratios[velocity_counter, j] = damping_new
                            eigvals_all[velocity_counter, j] = λj
                            eigvecs_all[velocity_counter, j] = φj

                iterFreq += 1
                omega_old[j] = omega_new

                if iterFreq == 1000:
                    if verbose:
                        print(f"WARNING: Frequancy iteration has not converged for V = {V:.2f} m/s, mode {j+1}. Setting results to NaN.")
                    if not flutter_detected:
                        omega_all[velocity_counter, j] = np.nan
                        damping_ratios[velocity_counter, j] = np.nan
                        eigvals_all[velocity_counter, j] = np.nan
                        eigvecs_all[velocity_counter, j] = None
                    break

            if damping_new < 0:
                flutter_detected = True
                for other_j in range(n_modes):
                    if other_j != j:
                        skip_mode[other_j] = True
                if verbose:
                    print(f"Flutter detected at V = {V:.2f} m/s, iterWind = {iterWind}, j = {j}")
                if np.abs(damping_new) < 0.0000001:
                    if verbose:
                        print(f"Flutter converged at V ≈ {V:.5f} m/s")
                    omegacritical = omega_new
                    Vcritical = V
                    V_list.append(V)
                    velocity_counter += 1

                    omega_all[velocity_counter, j] = omega_new
                    damping_ratios[velocity_counter, j] = damping_new
                    eigvals_all[velocity_counter, j] = λj   
                    eigvecs_all[velocity_counter, j] = φj
                    for other_j in range(n_modes):
                        if other_j != j:
                            omega_all[velocity_counter, other_j] = np.nan
                            damping_ratios[velocity_counter, other_j] =np.nan
                            eigvals_all[velocity_counter, other_j] = np.nan 
                            eigvecs_all[velocity_counter, other_j] =np.nan

                    skip_mode[j] = True  
                    stopWind = True

            else: 
                skip_mode = [False]*n_modes # Reset skip_mode for all modes
                if not flutter_detected:
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


        if all(not s for s in skip_mode):  
            if not flutter_detected:
                V_list.append(V)
                velocity_counter += 1
            V += dV

        elif not all(skip_mode): 
            V -= 0.5 * dV
            dV *= 0.5

        iterWind +=1

    # Truncate arrays to actual size
    omega_all = omega_all[:velocity_counter, :]
    damping_ratios = damping_ratios[:velocity_counter, :]
    eigvals_all = eigvals_all[:velocity_counter, :]
    eigvecs_all = eigvecs_all[:velocity_counter, :]


    return V_list, omega_all, damping_ratios, eigvecs_all, eigvals_all, omegacritical, Vcritical 


def plot_damping_vs_wind_speed(damping_ratios, eigvecs_all, V_list, alphas,
             dist="Fill in dist", single=True, static_quasi=False):
    """
    Plots the modal damping ratios as a function of wind speed and identifies
    the dominant degree of freedom (DOF) per mode through color encoding.

    Parameters:
    -----------
    damping_ratios : np.ndarray, shape (N_wind, n_modes)
        Modal damping ratios for each mode as a function of wind speed.
    eigvecs_all : np.ndarray, shape (N_wind, n_modes)
        Eigenvectors associated with each mode and wind speed (used to identify dominant DOF).
    V_list : list or np.ndarray, shape (N_wind,)
        List of wind speeds [m/s].
    dist : str
        Descriptor of the test case (e.g., "3D separation", "two-deck 2.5B").
    single : bool
        True for single-deck (2 DOFs), False for two-deck (4 DOFs).
    static_quasi : bool
        If True, the y-axis is constrained for quasi-static  (very low damping).
    """

    mode_labels = [r"$\Phi_{z1}$", r"$\Phi_{\theta1}$", r"$\Phi_{z2}$", r"$\Phi_{\theta2}$"]
    n_modes = 2 if single else 4

    fig, ax = plt.subplots(figsize=(5, 3))

    # Plot each mode's damping ratio at each wind speed
    for j in range(n_modes):
        V_mode = []
        ζ_mode = []
        for i in range(len(V_list)):
            φj = eigvecs_all[i, j]
            if φj is None or np.isnan(φj).any():
                continue
            dominant_dof_idx = np.argmax(np.abs(φj))
            V_mode.append(V_list[i])
            ζ_mode.append(damping_ratios[i, j])
        ax.plot(
                V_mode,
                ζ_mode, alpha=alphas, label=mode_labels[j]
            )

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # Visuals and annotation
    ax.set_xlabel(r"$V$ [m/s]", fontsize=16)
    ax.set_ylabel(r"$\zeta$ [-]", fontsize=16)
    #ax.set_title(f"Damping ratio vs wind speed — {dist}", fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.01,)
    ax.legend(fontsize=14, loc='upper left')

    ax.set_xlim(0, )
    ax.set_ylim(-0.001, )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()
    return fig, ax


def plot_damping__freq_vs_wind_speed(damping_ratios, eigvecs_all, V_list, omega_list, alphas,
             dist="Fill in dist", single=True, static_quasi=False):
    
    mode_labels = [r"$\Phi_{z}$", r"$\Phi_{\theta}$", r"$\Phi_{z2}$", r"$\Phi_{\theta2}$"]
    n_modes = 2 if single else 4

    fig, ax = plt.subplots(2, 1, figsize=(5,6), sharex=True)

    # Plot each mode's damping ratio at each wind speed
    for j in range(n_modes):
        V_mode = []
        ζ_mode = []
        for i in range(len(V_list)):
            φj = eigvecs_all[i, j]
            if φj is None or np.isnan(φj).any():
                continue
            dominant_dof_idx = np.argmax(np.abs(φj))
            V_mode.append(V_list[i])
            ζ_mode.append(damping_ratios[i, j])
        ax[0].plot(
                V_mode,
                ζ_mode, alpha=alphas, label=mode_labels[j]
            )

    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    

    # Visuals and annotation
    # ax.set_xlabel(r"$V$ [m/s]", fontsize=16)
    ax[0].set_ylabel(r"$\zeta$ [-]", fontsize=16)
    #ax.set_title(f"Damping ratio vs wind speed — {dist}", fontsize=18)
    ax[0].set_ylim(-0.01,)
    ax[0].legend(fontsize=14, loc='upper left')

    omega_array = np.array(omega_list)
    frequencies = omega_array / (2 * np.pi)  

    for j in range(n_modes):
        ax[1].plot(
            V_list,
            frequencies[:, j],
            alpha = alphas, label=mode_labels[j]
        )

    ax[1].set_xlabel(r"$V$ [m/s]", fontsize=16)
    ax[1].set_ylabel(r"$f$ [Hz]", fontsize=16)
    ax[1].legend(fontsize=14,loc='center left')
    ax[1].set_xlim(0, V_list[-1])

    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    plt.xticks(fontsize=14)
    ax[0].tick_params(labelsize=14)
    ax[1].tick_params(labelsize=14)
    plt.show()
    return fig, ax


def plot_frequency_vs_wind_speed(V_list, omega_list, alphas, dist="Fill in dist", single=True):
    """
    Plots the damped natural frequencies as a function of wind speed.

    Parameters
    ----------
    V_list : list or np.ndarray
        Wind speed values [m/s].
    omega_list : np.ndarray
        Angular frequencies (shape: N_wind x n_modes), given in rad/s.
    dist : str
        Description of the bridge configuration or test case (e.g. "3D separation").
    single : bool, optional
        If True, assumes 2 DOF system (single-deck); if False, assumes 4 DOF (two-deck).
    """


    labels = [r"$\Phi_{z1}$", r"$\Phi_{\theta1}$", r"$\Phi_{z2}$", r"$\Phi_{\theta2}$"]
    n_modes = 2 if single else 4
    title = f"Natural frequencies vs wind speed - {dist}"

    omega_array = np.array(omega_list)
    frequencies = omega_array / (2 * np.pi)  

    fig, ax = plt.subplots(figsize=(5, 3))
    for j in range(n_modes):
        plt.plot(
            V_list,
            frequencies[:, j],
            alpha = alphas, label=labels[j]
        )



    plt.xlabel(r"$V$ [m/s]", fontsize=16)
    plt.ylabel(r"$f$ [Hz]", fontsize=16)
    #plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14,loc='center left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.tight_layout()
    plt.xlim(0, V_list[-1])
    plt.show()
    return fig, ax



def plot_flutter_mode_shape(eigvecs_all, damping_list, V_list, Vcritical, omegacritical, dist="Fill in dist", single=True):
    if Vcritical is None or omegacritical is None:
        print("No flutter found!")
        return

    if single:
        n_modes = 2
        dofs = [r"$\phi_{z}$", r"$\phi_{\theta}$"]
    else:
        n_modes = 4
        dofs = [r"$\phi_{z1}$", r"$\phi_{\theta1}$", r"$\phi_{z2}$", r"$\phi_{\theta2}$"]
    

    last_damping = np.array(damping_list[-1])  
    idx_mode_flutter = np.nanargmin(last_damping)
    flutter_vec = eigvecs_all[-1][idx_mode_flutter]

    fig, ax = plt.subplots(2, 1, figsize=(5,6), sharex=True)

    abs_vec = np.abs(flutter_vec)
    max_idx = np.argmax(abs_vec)
    normalized_vec = flutter_vec / flutter_vec[max_idx]

    magnitudes = np.abs(normalized_vec)
    phases = np.angle(normalized_vec, deg=True)

    for i in range(n_modes):
        if magnitudes[i] < 1e-2:
            phases[i] = np.nan  # eller sett til 0 hvis du vil være eksplisitt

    colors = ['#9467bd', '#17becf', '#e377c2', '#bcbd22'][:n_modes]



    ax[0].bar(dofs, magnitudes, width=0.4, color =colors)
    ax[1].bar(dofs, phases, width=0.4, color = colors)

    ax[0].set_ylabel(r"|$\Phi_{\theta}$| [-]",fontsize=16)
    ax[0].set_ylim(0, 1.1)
    ax[1].set_ylabel(r"$\angle \Phi_{\theta}$ [deg]",fontsize=16)
    ax[1].set_ylim(-230, 230)
    
    ax[1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    #ax[0].set_title(f"Magnitude and phase of normalized eigenvector at flutter wind speed - {dist}", fontsize=14)
    ax[1].set_xlabel("DOFs",fontsize=16)


    for i in range(n_modes):

        if abs(magnitudes[i]) > 1e-2: 
            ax[0].text(i, magnitudes[i] + 0.02, f"{magnitudes[i]:.2f}", ha='center', fontsize=14)
        if abs(phases[i]) > 1:
            va = 'bottom' if phases[i] > 0 else 'top'
            offset = 10 if phases[i] > 0 else -10
            ax[1].text(i, phases[i] + offset, f"{phases[i]:.0f}", ha='center', va=va, fontsize=14)

    #fig.subplots_adjust(hspace=2.0)

    plt.xticks(fontsize=14)
    ax[0].tick_params(labelsize=14)
    ax[1].tick_params(labelsize=14)
    plt.show()

    return fig, ax

