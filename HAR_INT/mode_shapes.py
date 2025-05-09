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
def mode_shape_single():
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


    for i in range(N):
        phi_single[i, 0, 0] = mode_4[i, 2]  # zz, vertical
        phi_single[i, 1, 0] = mode_4[i, 3]  # θz, rotation
        phi_single[i, 0, 1] = mode_15[i, 2] # zθ
        phi_single[i, 1, 1] = mode_15[i, 3] # θθ


    return phi_single, x

#Double deck
def mode_shape_twin():
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
    phi_double = np.zeros((2*N, 4, 4))

    for i in range(N):
        phi_double[i, 0, 0] = mode_4[i, 2]  # zz, vertical
        phi_double[i, 1, 0] = mode_4[i, 3]  # θz, rotation
        phi_double[i, 0, 1] = mode_15[i, 2] # zθ
        phi_double[i, 1, 1] = mode_15[i, 3] # θθ
        
    for i in range(N):
        j = i + N
        phi_double[j, 2, 2] = mode_4[i, 2]  # zz, vertical
        phi_double[j, 3, 2] = mode_4[i, 3]  # θz, rotation
        phi_double[j, 2, 3] = mode_15[i, 2] # zθ
        phi_double[j, 3, 3] = mode_15[i, 3] # θθ

    # phi_double.shape: (n_nodes, n_DOF,n_modes)

    return phi_double, x