"""
Created in April 2025

@author: linehro & alicjaas
"""

import numpy as np

data4 = np.load('mode4_data.npz')
data15 = np.load('mode15_data.npz')

x = data4['x']
mode_4 = data4['mode']
mode_15 = data15['mode']

length = x[-1] - x[0]

#Single deck
def mode_shape_single(full_matrix=True):
    """
    Constructs mode shape matrices for a single-deck bridge using vertical (mode 4)
    and torsional (mode 15) modes. Mode data is loaded internally from .npz files.

    Files:
        mode_4.npz  : Contains mode shape data for vertical mode (key = 'mode')
        mode_15.npz : Contains mode shape data for torsional mode (key = 'mode')

    Notes:
        - Each mode array has shape (N, 6), where N is the number of nodes.
        - Each row contains [x, y, z, dx, dy, dz] for a single node.
        - The x array contains the corresponding physical positions along the bridge.
        - x ranges from -654 to 654.

    Parameters:
        full_matrix (bool): 
            If True, populate full 2x2 matrix with mode coupling terms.
            If False, populate only the diagonal (pure modes).

    Returns:
        phi_single (np.ndarray): Array of shape (N, 2, 2), the generalized mode shape matrices
                                 at each node, with degrees of freedom [z, θ].
        N (int): Number of nodes along the bridge span.
        x (np.ndarray): 1D array of length N containing the physical x-coordinates.

    """
    data4 = np.load('mode4_data.npz')
    data15 = np.load('mode15_data.npz')
    mode_4 = data4['mode']
    mode_15 = data15['mode']
    x = data4['x']

    N = mode_4.shape[0]
    phi_single = np.zeros((N, 2, 2))

    # # Extract vertical and rotational components from mode data
    # phi_z_1V = mode_4[:, 2]      # Vertical displacement (z) from mode 4
    # phi_theta_1V = mode_4[:, 3]  # Rotation (theta) from mode 4
    # phi_z_1T = mode_15[:, 2]     # Vertical from torsional mode
    # phi_theta_1T = mode_15[:, 3] # Rotation from torsional mode

    # for i, (z1V, t1V, z1T, t1T) in enumerate(zip(phi_z_1V, phi_theta_1V, phi_z_1T, phi_theta_1T)):
    #     phi_single[i, 0, 0] = z1V
    #     phi_single[i, 1, 1] = t1T

    #     if full_matrix:
    #         phi_single[i, 1, 0] = z1T
    #         phi_single[i, 0, 1] = t1V
    # phi_single.shape: (n_nodes, n_DOF,n_modes)

    # # mode 0: vertical mode (mode 4)
    # phi_single[0, 0, :] = mode_4[:, 2]  # zz
    # phi_single[0, 1, :] = mode_4[:, 3]  # θz

    # # mode 1: torsional mode (mode 15)
    # phi_single[1, 0, :] = mode_15[:, 2]  # zθ
    # phi_single[1, 1, :] = mode_15[:, 3]  # θθ

    for i in range(N):
        phi_single[i, 0, 0] = mode_4[i, 2]  # zz, vertical
        phi_single[i, 0, 1] = mode_4[i, 3]  # θz, rotation
        phi_single[i, 1, 0] = mode_15[i, 2] # zθ
        phi_single[i, 1, 1] = mode_15[i, 3] # θθ


    return phi_single, x

#Double deck
def mode_shape_twin(full_matrix=True):
    """
    Constructs mode shape matrices for a twin-deck bridge using vertical (mode 4)
    and torsional (mode 15) modes. Mode data is loaded internally from .npz files.

    Files:
        mode4_data.npz   : Contains mode shape data for vertical mode (key = 'mode')
        mode_15_data.npz : Contains mode shape data for torsional mode (key = 'mode')

    Notes:
        - Each mode array has shape (N, 6), where N is the number of nodes along the bridge.
        - Each row contains [x, y, z, dx, dy, dz] for a single node.
        - The x array contains the corresponding physical positions along the bridge.
        - x ranges from -654 to 654.

    Parameters:
        full_matrix (bool): 
            If True, populates full 4x4 matrices with coupling terms between modes.
            If False, only diagonal terms are populated.

    Returns:
        phi_double (np.ndarray): Array of shape (N, 4, 4), containing the mode shape matrices
                                 at each node. DOFs are ordered as [z1, θ1, z2, θ2].
        N (int): Number of nodes.
        x (np.ndarray): 1D array of length N containing the physical x-coordinates.

    """
    data4 = np.load('mode4_data.npz')
    data15 = np.load('mode15_data.npz')
    mode_4 = data4['mode']
    mode_15 = data15['mode']
    x = data4['x']


    N = mode_4.shape[0]
    phi_double = np.zeros((N, 4, 4))

    # # Extract relevant DOFs from each mode
    # phi_z_1V = mode_4[:, 2]       # vertical deck 1
    # phi_theta_1V = mode_4[:, 3]   # torsion deck 1
    # phi_z_1T = mode_15[:, 2]      # vertical (torsion mode) deck 1
    # phi_theta_1T = mode_15[:, 3]  # torsion (torsion mode) deck 1

    # for i, (z1V, t1V, z1T, t1T) in enumerate(zip(phi_z_1V, phi_theta_1V, phi_z_1T, phi_theta_1T)):
    #     # Deck 1
    #     phi_double[i, 0, 0] = z1V
    #     phi_double[i, 1, 1] = t1T

    #     # Deck 2 (mirrored)
    #     phi_double[i, 2, 2] = z1V
    #     phi_double[i, 3, 3] = t1T

    #     if full_matrix:
    #         phi_double[i, 1, 0] = z1T
    #         phi_double[i, 0, 1] = t1V 
    #         phi_double[i, 3, 2] = z1T
    #         phi_double[i, 2, 3] = t1V

    # # mode 0: vertical mode (mode 4)
    # phi_double[0, 0, :] = mode_4[:, 2]  # zz
    # phi_double[0, 1, :] = mode_4[:, 3]  # θz

    # # mode 1: torsional mode (mode 15)
    # phi_double[1, 0, :] = mode_15[:, 2]  # zθ
    # phi_double[1, 1, :] = mode_15[:, 3]  # θθ

    # # mode 0: vertical mode (mode 4)
    # phi_double[2, 2, :] = mode_4[:, 2]  # zz
    # phi_double[2, 3, :] = mode_4[:, 3]  # θz

    # # mode 1: torsional mode (mode 15)
    # phi_double[3, 2, :] = mode_15[:, 2]  # zθ
    # phi_double[3, 3, :] = mode_15[:, 3]  # θθ

    for i in range(N):
        phi_double[i, 0, 0] = mode_4[i, 2]  # zz, vertical
        phi_double[i, 0, 1] = mode_4[i, 3]  # θz, rotation
        phi_double[i, 1, 0] = mode_15[i, 2] # zθ
        phi_double[i, 1, 1] = mode_15[i, 3] # θθ
        phi_double[i, 2, 2] = mode_4[i, 2]  # zz, vertical
        phi_double[i, 2, 3] = mode_4[i, 3]  # θz, rotation
        phi_double[i, 3, 2] = mode_15[i, 2] # zθ
        phi_double[i, 3, 3] = mode_15[i, 3] # θθ


    # phi_double.shape: (n_nodes, n_DOF,n_modes)


    return phi_double, x