

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

# def generalize_global(C, K, Phi):
#     """
#     Generalizes full global aerodynamic matrices C and K using modal matrix Phi.

#     Parameters:
#     -----------
#     C : ndarray, shape (n, n)
#         Full aerodynamic damping matrix (global DOFs).
#     K : ndarray, shape (n, n)
#         Full aerodynamic stiffness matrix (global DOFs).
#     Phi : ndarray, shape (n, n)
#         Modal matrix (columns are mode shapes, usually mass-normalized).

#     Returns:
#     --------
#     C_gen : ndarray, shape (n, n)
#         Generalized (modal) damping matrix.
#     K_gen : ndarray, shape (n, n)
#         Generalized (modal) stiffness matrix.
#     """
#     C_gen = Phi.T @ C @ Phi # shape: Single - (2, 2) or Twin - (4, 4)
#     K_gen = Phi.T @ K @ Phi
#     return C_gen, K_gen


# def normalize_modes(Phi, M):
#     """
#     Masse-normaliserer modeshapes for 2 DOF (single-deck) eller 4 DOF (twin-deck).
    
#     Parameters:
#     -----------
#     Phi : ndarray of shape (N, n, n)
#         Modalmatriser langs broen. n = 2 for single-deck, n = 4 for twin-deck.
#     M : ndarray of shape (n, n)
#         Massematrise i fysisk DOF-rom (typisk diagonal).
    
#     Returns:
#     --------
#     Phi_norm : ndarray of shape (N, n, n)
#         Masse-normaliserte modeshapes.
#     """
#     N, n, _ = Phi.shape
#     Phi_norm = np.zeros_like(Phi)

#     for i in range(N):
#         for j in range(n):  # én mode om gangen
#             phi_j = Phi[i][:, j]  # shape: (n,)
#             m_j = phi_j.T @ M @ phi_j
#             Phi_norm[i][:, j] = phi_j / np.sqrt(m_j)
        
#     return Phi_norm

# def global_massnorm_modeshapes(M, single = True):

#     if single:
#         n = 2  
#         Phi = mode_shape_single(full_matrix=True)[0]

#     else:
#         n = 4
#         Phi= mode_shape_twin(full_matrix=True)[0]
    
#     # Phi shape: (n_nodes, n_dof, n_modes)
#     # M shape: (n_dof, n_dof) (2x2 or 4x4)
    
#     #Phi_massNorm = normalize_modes(Phi, M) 

#     Phi_global = Phi.reshape(-1,n) 

#     return Phi_global



# def structural_matrices_massnorm(f1, f2, zeta, single=True):
#     """
#     Return mass-normalized structural matrices:
#     M = I, K = diag(omega^2), C = diag(2*zeta*omega)
#     """
#     ω1 = 2 * np.pi * f1
#     ω2 = 2 * np.pi * f2

#     Ms = np.eye(2)
#     Ks = np.diag([ω1**2, ω2**2])
#     Cs = np.diag([2 * zeta * ω1, 2 * zeta * ω2])

#     if not single:
#         Ms = np.eye(4)
#         Ks = np.kron(np.eye(2), Ks)
#         Cs = np.kron(np.eye(2), Cs)

#     return Ms, Cs, Ks

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

    # Vreds = np.linspace(0.1, 20, 200)
    # AD_H1 = [from_poly_k(poly_coeff[0], k_range[0], v) for v in Vreds]
    # plt.plot(Vreds, AD_H1); plt.title("H1 vs Vred"); plt.show()
    # AD_H2 = [from_poly_k(poly_coeff[1], k_range[1], v) for v in Vreds]
    # plt.plot(Vreds, AD_H2); plt.title("H2 vs Vred"); plt.show()
    # AD_H3 = [from_poly_k(poly_coeff[2], k_range[2], v) for v in Vreds]
    # plt.plot(Vreds, AD_H3); plt.title("H3 vs Vred"); plt.show()
    # AD_H4 = [from_poly_k(poly_coeff[3], k_range[3], v) for v in Vreds]
    # plt.plot(Vreds, AD_H4); plt.title("H4 vs Vred"); plt.show()
    # AD_A1 = [from_poly_k(poly_coeff[4], k_range[4], v) for v in Vreds]
    # plt.plot(Vreds, AD_A1); plt.title("A1 vs Vred"); plt.show()
    # AD_A2 = [from_poly_k(poly_coeff[5], k_range[5], v) for v in Vreds]
    # plt.plot(Vreds, AD_A2); plt.title("A2 vs Vred"); plt.show()
    # AD_A3 = [from_poly_k(poly_coeff[6], k_range[6], v) for v in Vreds]
    # plt.plot(Vreds, AD_A3); plt.title("A3 vs Vred"); plt.show()
    # AD_A4 = [from_poly_k(poly_coeff[7], k_range[7], v) for v in Vreds]
    # plt.plot(Vreds, AD_A4); plt.title("A4 vs Vred"); plt.show()



    Vred_global = float(Vred_global) 

    # AD
    H1, H2, H3, H4 = from_poly_k(poly_coeff[0], k_range[0],Vred_global, damping_ad=True), from_poly_k(poly_coeff[1], k_range[1],Vred_global, damping_ad=True), from_poly_k(poly_coeff[2], k_range[2],Vred_global, damping_ad=False), from_poly_k(poly_coeff[3], k_range[3],Vred_global, damping_ad=False)
    A1, A2, A3, A4 = from_poly_k(poly_coeff[4], k_range[4],Vred_global, damping_ad=True), from_poly_k(poly_coeff[5], k_range[5],Vred_global, damping_ad=True), from_poly_k(poly_coeff[6], k_range[6],Vred_global, damping_ad=False), from_poly_k(poly_coeff[7], k_range[7],Vred_global, damping_ad=False)

    # print("H1 at Vred=5:", from_poly_k(poly_coeff[0], k_range[0], 5))
    # print("H2 at Vred=5:", from_poly_k(poly_coeff[1], k_range[1], 5))
    # print("H3 at Vred=5:", from_poly_k(poly_coeff[2], k_range[2], 5))
    # print("H4 at Vred=5:", from_poly_k(poly_coeff[3], k_range[3], 5))
    # print("A1 at Vred=5:", from_poly_k(poly_coeff[4], k_range[4], 5))
    # print("A2 at Vred=5:", from_poly_k(poly_coeff[5], k_range[5], 5))
    # print("A3 at Vred=5:", from_poly_k(poly_coeff[6], k_range[6], 5))
    # print("A4 at Vred=5:", from_poly_k(poly_coeff[7], k_range[7], 5))
    
    # H1, H2, H3, H4 = np.polyval(poly_coeff[0][::-1], Vred_global), np.polyval(poly_coeff[1][::-1], Vred_global), np.polyval(poly_coeff[2][::-1], Vred_global), np.polyval(poly_coeff[3][::-1], Vred_global)
    # A1, A2, A3, A4 = np.polyval(poly_coeff[4][::-1], Vred_global), np.polyval(poly_coeff[5][::-1], Vred_global), np.polyval(poly_coeff[6][::-1], Vred_global), np.polyval(poly_coeff[7][::-1], Vred_global)


    # print(f"H1* = {H1:.3f}, H2* = {H2:.3f}, H3* = {H3:.3f}, H4* = {H4:.3f}")
    # print(f"A1* = {A1:.3f}, A2* = {A2:.3f}, A3* = {A3:.3f}, A4* = {A4:.3f}")
    # print("---")

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
                else:
                    dominance_scores = np.array([np.abs(eigvecs_pos[j, idx]) for idx in range(eigvecs_pos.shape[1])])
                    prev_lambda = eigvals_all[velocity_counter - 1, j] 

                    if V < 60:
                        best_idx = np.argmax(dominance_scores) # Velg den største egenvektoren
                    else:
                        best_idx = np.argmin(
                                np.abs(np.imag(eigvals_pos) - omega_old[j])
                                + 10 * np.abs(np.real(eigvals_pos) - np.real(prev_lambda))
                            )

                λj = eigvals_pos[best_idx]
                φj = eigvecs_pos[:n_modes, best_idx]
                omega_new = np.imag(λj)
                damping_new = -np.real(λj) / np.abs(λj)

                # if verbose:
                #     print(f"[DEBUG] IterFreq {iterFreq}, omega_old = {omega_old[j]:.4f}, omega_new = {omega_new:.4f}, diff = {np.abs(omega_old[j] - omega_new):.4e}")

                if verbose:
                    print(f"  IterFreq {iterFreq}: omega_old = {omega_old[j]:.5f}, omega_new = {omega_new:.5f}, diff = {np.abs(omega_old[j] - omega_new):.2e}")
                    print(f"  V_red = {Vred:.5f}")

                                            
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

                # if Vcritical_guess is not None and abs(V - Vcritical_guess) < 1e-6:
                #     count_same += 1
                # else:
                #     count_same = 0
                #     Vcritical_guess = V

                # if count_same > 5: # hvis vi har vært på samme hastighet i 5 iterasjoner, så gidder vi ikke mer
                #     if verbose:
                #         print(f"Flutter converged at V ≈ {V:.5f} m/s")
                #     omegacritical = omega_new
                #     Vcritical = V
                #     skip_mode[j] = True
                #     stopWind = True

                
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

def plot_flutter_mode_shape(eigvecs_all, omega_list, V_list, Vcritical, omegacritical, dist="Fill in dist", single=True):
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
    freq_flutter = np.array(omega_list[idx_flutter]) / (2 * np.pi)

    # Finn moden som har frekvens nærmest den kritiske
    idx_mode_flutter = np.argmin(np.abs(freq_flutter - omegacritical / (2 * np.pi)))

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
        if abs(magnitudes[i]) > 1e-3 or magnitudes[i] !=1:
            ax[0].text(i, magnitudes[i] + 0.02, f"{magnitudes[i]:.2f}", ha='center', fontsize=9)
        if abs(phases[i]) > 1:
            ax[1].text(i, phases[i] + 10*np.sign(phases[i]), f"{phases[i]:.1f}°", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

#def solve_omega

    

    #skip_mode = [False] * n_modes   # skip mode once flutter is detected

    # damping_z_list = [] if single or (not single and n_modes >= 1) else None
    # V_damping_z = []
    # V_damping_theta = []
    # damping_theta_list = [] if single or (not single and n_modes >= 1) else None

  


    # omega = [] # Frequency, rad/s
    # damping = [] # Damping ratio
    # eigvecs = [] # Eigenvectors 

    # damping_old = [0.005]*n_modes
    # eigvec_old = [None] * n_modes

    # # Sorter etter imaginærdel
    # sort_idx = np.argsort(np.imag(eigvals0))
    # eigvals0_sorted = eigvals0[sort_idx]
    # eigvecs0_sorted = eigvecs0[:, sort_idx] 
   
    

    # omega.append(np.imag(eigvals0_sorted[:2*n_modes])) #konjugatpar
    # damping.append(np.real(eigvals0_sorted[:2*n_modes])) #konjugatpar, ikke normalisert 
    # eigvecs.append(eigvecs0_sorted[:, :2*n_modes]) 




                # if np.isclose(V, 0.0): # still air
                #     # Egenverdiene og egenvektorene kommer i riktig rekkefølge
                #     # Save which DOFs that dominate the mode shape
                    

                #     Cae_gen = np.zeros_like(Ms)
                #     Kae_gen = np.zeros_like(Ms)
                    

        #idxResMat = 0
                #ResMat = np.zeros((2, 2 * Ms.shape[0]))


              #     eigvec_old[j] = φj

                #     converge = True

   
        # Første rad (ResMat[0, :]) skal lagre imag(λ) → altså frekvensene (ω) for alle modene. (Omega: np.imag(λj))
        # Andre rad (ResMat[1, :]) skal lagre real(λ) → altså "rå" demping (ζ) for alle modene. (Damping:-np.real(λj) / np.abs(λj))
        # Når du løser et 2. ordens differensialligningssystem (som flutterproblemet), får du 2 × n_modes egenverdier.
        # For hver fysisk mode får du én positiv og én negativ imaginærdel (komplekse konjugatpar).
        # Derfor trenger du dobbelt så mange plasser.


            # if skip_mode[j]:
            #     omega_all[j].append(np.nan)
            #     damping_ratios[j].append(np.nan)
            #     eigvals_all[j].append(np.nan)
            #     eigvecs_all[j].append(np.nan)
            #     continue  # Go to next mode if flutter is detected

            #omegacritical: omega0[j]


                # eigvals_pos = eigvalsV[np.imag(eigvalsV) > 0]
                # sort_idx_pos = np.argsort(np.imag(eigvals_pos))
                # eigvals_sortedpos = eigvals_pos[sort_idx_pos]

                # sort_idx = np.argsort(np.imag(eigvalsV))
                # eigvals_sorted = eigvalsV[sort_idx]
                # eigvecs_sorted = eigvecsV[:, sort_idx]


                # # Sorter etter imaginærdel
                # sort_idx = np.argsort(np.imag(eigvalsV))
                # eigvals_sorted = eigvalsV[sort_idx]
                # eigvecs_sorted = eigvecsV[:, sort_idx]

                # # Ta positiv halvdel
                # eigvals_sortedpos = eigvals_sorted[len(eigvals_sorted)//2:]


                # if single:
                #     best_idx = np.argmin(np.abs(np.imag(eigvals_sortedpos) - omegacr))    
                # else:
                #     best_idx = np.argmin(np.abs(np.imag(eigvals_sortedpos) - omegacr) + 10 * np.abs(np.real(eigvals_sortedpos) - damping[-1][n_modes + j]))
                        
               # domega = omegacr - np.imag(eigvals_sortedpos[best_idx])
                # omegacr = np.imag(eigvals_sortedpos[best_idx])

                # domega = omegacr - np.imag(eigvals_sortedpos[j])
                # omegacr = np.imag(eigvals_sortedpos[j])

                    # # Min gamle versjon 
                    # # Keep only the eigenvalues with positive imaginary part (complex conjugate pairs)
                    # threshold = 1e-6  
                    # imag_mask = np.imag(eigvals) > threshold
                    # eigvals_pos = eigvals[imag_mask]
                    # eigvecs_pos = eigvecs[:, imag_mask]  
                


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


                
                        # omega_all[iterWind,j]=omega_new    
                        # damping_ratios[iterWind,j]=damping_new
                        # eigvals_all[iterWind,j]=λj                
                        # eigvecs_all[iterWind,j]=φj

                        # if j ==0:
                        #     damping_z_list.append(Cae_gen.copy())  # legg inn her
                        #     V_damping_z.append(V)

                        # if j == 1:
                        #     damping_theta_list.append(Cae_gen.copy())
                        #     V_damping_theta.append(V)

                        # if time.time() - start_time > timeout:
                        #     print(f"WARNING: Convergence timeout at V = {V:.2f} m/s, mode {j+1}. Setting results to NaN.")
                        #     omega_all[iterWind, j] = np.nan
                        #     damping_ratios[iterWind, j] = np.nan
                        #     eigvals_all[iterWind, j] = np.nan
                        #     eigvecs_all[iterWind, j] = None
                        #     #print("iteration:", iter)
                        #     break

                        # omega_old[j] = omega_new
                        # damping_old[j] = damping_new
                        # eigvec_old[j] = φj
                    # ResMat[0, idxResMat : idxResMat + 2] = [np.nan, np.nan]
                    # ResMat[1, idxResMat : idxResMat + 2] = [np.nan, np.nan]
                    # idxResMat += 2                   

            # # OK, hvilken egenverdi var det egentlig som hadde denne ω?
            # posres = np.where(np.isclose(np.imag(eigvals_sorted), omegacr, atol=1e-6))[0] # posisjon til rett egenverdi) må hentes når du er ferdig med å iterere på w
            # # ResMat har 2 rader: første for ω, andre for ξ
            # # Hvis flere treff på ωcr, skriver de inn flere kolonner
            
            # print("\n--- Debug info ---")
            # print(f"iterWind = {iterWind}, j = {j}")
            # print(f"Current idxResMat = {idxResMat}")
            # print(f"posres = {posres}")
            # print(f"len(posres) = {len(posres)}")
            # print(f"eigvals_sorted[posres] = {eigvals_sorted[posres]}")
            # print(f"Remaining space in ResMat = {ResMat.shape[1] - idxResMat}")
            # print("------------------\n")
                        
            
            # if len(posres) > 2: # Hvis vi fant flere egenverdier som matcher ωcr
            # #det finnes flere løsninger
            #     ResMat[0, idxResMat : idxResMat + len(posres)] = np.imag(eigvals_sorted[posres])
            #     ResMat[1, idxResMat : idxResMat + len(posres)] = np.real(eigvals_sorted[posres])
            #     idxResMat += len(posres)  # Hopper videre
            # else:
            #     ResMat[0, idxResMat : idxResMat + 2] = np.imag(eigvals_sorted[posres])
            #     ResMat[1, idxResMat : idxResMat + 2] = np.real(eigvals_sorted[posres])
            #     idxResMat += 2  # Konjugatpar ⇒ 2 verdier
        
        # stabind = np.max(ResMat[1, ResMat[1,:] != 0]) # største reelle del, brukes til å sjekke om det finnes noen uastabile egenverdier
        # iterWind += 1
        # if stabind > 0: #flutter finnes, må finne nøyaktig flutterhastighet
        #     print(f"Flutter detected at V = {V:.2f} m/s, iterWind = {iterWind}, j = {j}")
        #     if Vcritical_guess is not None and abs(V - Vcritical_guess) < 1e-6:
        #         count_same += 1
        #     else:
        #         count_same = 0
        #         Vcritical_guess = V

        #     if count_same > 2: # hvis vi har vært på samme hastighet i 5 iterasjoner, så gidder vi ikke mer
        #         print(f"Flutter converged at V ≈ {V:.5f} m/s")
        #         posCR = np.argmax(ResMat[1, :])
        #         omegacritical = ResMat[0, posCR]
        #         Vcritical = V
        #         stopWind = True
        #     else:
        #         V -= 0.5 * dV
        #         dV *= 0.5


            # V_list.append(V)
            # omega.append(ResMat[0, :])
            # damping.append(ResMat[1, :])
            # eigvecs.append(eigvecs_sorted[:, :2*n_modes])


        # if np.abs(stabind) < 1e-5: # hvis stabind = 0, akkurat ved flutter!
        # # kritisk frekvens og vindhastighet lagres
        #     posCR = np.argmax(ResMat[1, :])
        #     omegacritical = ResMat[0, posCR]
        #     Vcritical = V
        #     stopWind = True

#return V_list, omega, damping, eigvecs, omegacritical, Vcritical



# def solve_flutter_speed(damping_ratios, V_list, single = True):
#     """
#     Finds the flutter speed for each mode where the damping becomes negative.

#     Parameters:
#     -----------
#     damping_ratios : list of arrays
#         Damping ratios for each mode (shape: list of length n_modes with N elements each).
#     V_list
#     single : bool
#         True if single-deck (2 modes), False if twin-deck (4 modes).

#     Returns:
#     --------
#     flutter_speed_modes : list
#         List of flutter speeds (None if not observed).
#     V_list : ndarray
#         Wind speed vector used in calculation.
#     """
#     n_modes = 2 if single else 4
#     flutter_speed_modes = np.full(n_modes, np.nan)
#     flutter_idx_modes = np.full(n_modes, np.nan)

#     for j in range(n_modes):
#         for i, V in enumerate(V_list):
#             if damping_ratios[i, j] < 0 and np.isnan(flutter_speed_modes[j]):
#                 flutter_speed_modes[j] = V
#                 flutter_idx_modes[j] = i
#                 break

#     if np.all(np.isnan(flutter_speed_modes)):
#         print("Ingen flutter observert for noen moder!")
#         return None
#     return flutter_speed_modes, flutter_idx_modes
     

     
def plot_damping_vs_wind_speed0(damping_list,omega_list, V_list, dist="Fill in dist",  single = True):
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

    damping_array = []
    for damp, omega in zip(damping_list, omega_list):
        # Beregn -Re(λ)/|λ| for positiv halvdel
        lambda_pos = 1j * omega + damp  # rekonstruerer λ (den ble splittet i ulike lister i solve_omega)
        xi = -np.real(lambda_pos) / np.abs(lambda_pos)
        damping_array.append(xi)

    damping_array = np.array(damping_array)  # (Nvind, 2*n_modes)


    for j in range(n_modes):
        plt.plot(V_list, damping_array[:, n_modes+j], color=colors[j], marker=markers[j],  markersize=markersizes[j],  linestyle='None', label=labels[j])

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


def plot_frequency_vs_wind_speed0(V_list, omega_list,   dist="Fill in dist", single = True):
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

    if single:
        n_modes = 2 
        title = f"Natural frequencies vs wind speed - {dist}"
    else:
        n_modes = 4
        title = f"Natural frequencies vs wind speed - {dist}"

    omega_array = np.array(omega_list) 
    frequencies = omega_array[:, n_modes:] / (2 * np.pi)       # Convert to Hz, velger kun positiv del av konjugatparet

    plt.figure(figsize=(10, 6))

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



def plot_flutter_mode_shape0(eigvecs_all, omega_list, V_list, Vcritical, omegacritical, dist="Fill in dist", single = True):
    
    if Vcritical is None or omegacritical is None:
        print("Ingen flutter observert!")
        return

    if single:
        n_modes = 2
        dofs = ["V1", "T1"]

    else:
        n_modes = 4  
        dofs = ["V1", "T1", "V2", "T2"]
    
    # Finn hvilken vindhastighet som tilsvarer Vcritical
    idx_flutter = np.argmin(np.abs(np.array(V_list) - Vcritical))

    # Finn hvilken mode som flutterer
    freq_flutter = np.array(omega_list[idx_flutter])
    freq_flutter = freq_flutter[n_modes:] / (2*np.pi)  # kun positive modes (Hz)

    idx_mode_flutter = np.argmin(np.abs(freq_flutter - omegacritical/(2*np.pi)))

    flutter_vec = eigvecs_all[idx_flutter][:, n_modes + idx_mode_flutter]


    # 2 subplot: én for amplitudene, én for fasevinklene. De deler x-aksen
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)


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

    
    