
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
    Constructs the generalized mode shape matrix for a single-deck bridge section,
    using the dominant vertical (mode 4) and torsional (mode 15) modes.

    Mode shape data is loaded from NumPy .npz files containing arrays of nodal information.

    Files:
        mode4_data.npz  : Contains vertical mode shape (key = 'mode', shape = (N, 6))
        mode15_data.npz : Contains torsional mode shape (key = 'mode', shape = (N, 6))

    Format:
        Each mode shape array has shape (N, 6), where N is the number of bridge nodes.
        Each row corresponds to a node and contains:
        [x, y, z, dx, dy, dz] — i.e., position and displacement vectors.
        Only the vertical displacement (z) and torsional rotation (θ ≈ dy) are used.

    Returns:
        phi_single (np.ndarray): Generalized mode shape array of shape (N, 2, 2), where
                                 phi_single[i] contains the 2x2 mode shape matrix at node i.
                                 DOFs are ordered as [z, θ].
        x (np.ndarray): 1D array of nodal x-coordinates (length N).
    """
    data4 = np.load('mode4_data.npz')
    data15 = np.load('mode15_data.npz')
    mode_4 = data4['mode']
    mode_15 = data15['mode']
    x = data4['x']

    N = mode_4.shape[0]
    phi_single = np.zeros((N, 2, 2))


    for i in range(N):
        phi_single[i, 0, 0] = mode_4[i, 2]  # zz
        phi_single[i, 1, 1] = mode_15[i, 3] # θθ

        # Cross-coupling terms are neglected (set to zero)
        #phi_single[i, 1, 0] = mode_4[i, 3] 
        #phi_single[i, 0, 1] = mode_15[i, 2]

    return phi_single, x
 
#Double deck
def mode_shape_two():
    """
    Constructs the generalized mode shape matrix for a two-deck bridge configuration,
    using the dominant vertical (mode 4) and torsional (mode 15) modes. The same mode
    shapes are applied to both decks.

    Mode shape data is loaded from NumPy .npz files containing arrays of nodal information.

    Files:
        mode4_data.npz  : Contains vertical mode shape (key = 'mode', shape = (N, 6))
        mode15_data.npz : Contains torsional mode shape (key = 'mode', shape = (N, 6))

    Format:
        Each mode shape array has shape (N, 6), where N is the number of nodes.
        Each row corresponds to a node and contains:
        [x, y, z, dx, dy, dz] — i.e., position and displacement vectors.
        Only the vertical displacement (z) and torsional rotation (θ ≈ dy) are used.

    Returns:
        phi_double (np.ndarray): Generalized mode shape array of shape (N, 4, 4), where
                                 phi_double[i] contains the 4x4 mode shape matrix at node i.
                                 DOFs are ordered as [z₁, θ₁, z₂, θ₂].
        x (np.ndarray): 1D array of nodal x-coordinates (length N).
    """
    data4 = np.load('mode4_data.npz')
    data15 = np.load('mode15_data.npz')
    mode_4 = data4['mode']
    mode_15 = data15['mode']
    x = data4['x']


    N = mode_4.shape[0]
    phi_double = np.zeros((N, 4, 4))

    for i in range(N):
        phi_double[i, 0, 0] = mode_4[i, 2]  # zz
        phi_double[i, 1, 1] = mode_15[i, 3] # θθ

        phi_double[i, 2, 2] = mode_4[i, 2]  # zz
        phi_double[i, 3, 3] = mode_15[i, 3] # θθ

        # All off-diagonal terms are set to zero
        #phi_double[i, 1, 0] = mode_4[i, 3] 
        #phi_double[i, 0, 1] = mode_15[i, 2]
            
        #phi_double[i, 3, 2] = mode_4[i, 3]  
        #phi_double[i, 2, 3] = mode_15[i, 2]

    # phi_double.shape: (n_nodes, n_DOF,n_modes)

    return phi_double, x