import numpy as np


def build_rotation_minimizing_frame(norm, pos):
    """Transform given position to a rotation-minimized frame
    which is orientated along the 'norm'

    parameters
    ----------------
    norm: (3,) ndarray
        unit normal vector of a plane
    pos: (N, 3) ndarray
        reference grid positions of yz plane

    returns
    ----------------
    trans: (N, 3) ndarray
        grid positions of transformed plane

    functions
    ----------------
    phi_rotation_matrix: (phi: float) -> ndarray
        rotation matrix along phi
    theta_rotation_matrix: (theta: float, axis: ndarray) -> ndarray
        Rodrigues rotation matrix along theta of norm axis
    """

    def phi_rotation_matrix(phi):
        """Return 2D rotational matrix in 3D"""
        return np.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        )

    def theta_rotation_matrix(theta, axis):
        """Return Rodrigues rotation matrix"""
        return np.array(
            [
                [
                    np.cos(theta) + axis[0] ** 2.0 * (1 - np.cos(theta)),
                    axis[0] * axis[1] * (1 - np.cos(theta))
                    - axis[2] * np.sin(theta),
                    axis[0] * axis[2] * (1 - np.cos(theta))
                    + axis[1] * np.sin(theta),
                ],
                [
                    axis[1] * axis[0] * (1 - np.cos(theta))
                    + axis[2] * np.sin(theta),
                    np.cos(theta) + axis[1] ** 2.0 * (1 - np.cos(theta)),
                    axis[1] * axis[2] * (1 - np.cos(theta))
                    - axis[0] * np.sin(theta),
                ],
                [
                    axis[2] * axis[0] * (1 - np.cos(theta))
                    - axis[1] * np.sin(theta),
                    axis[2] * axis[1] * (1 - np.cos(theta))
                    + axis[0] * np.sin(theta),
                    np.cos(theta) + axis[2] ** 2.0 * (1 - np.cos(theta)),
                ],
            ]
        )

    # Get theta for rotatioal matrix along z axis
    if norm[0] > 0:
        phi = np.arctan(norm[1] / norm[0])
    elif norm[1] >= 0 and norm[0] < 0:
        phi = np.pi + np.arctan(norm[1] / norm[0])
    elif norm[1] < 0 and norm[0] < 0:
        phi = -np.pi + np.arctan(norm[1] / norm[0])
    elif norm[1] >= 0 and norm[0] == 0:
        phi = np.pi / 2.0
    elif norm[1] < 0 and norm[0] == 0:
        phi = -np.pi / 2.0
    phi += 2 * np.pi if phi < 0 else 0

    # Get projection of the norm onto x-axis
    axis0 = np.matmul(phi_rotation_matrix(phi), np.array([1, 0, 0]))
    axis1 = np.matmul(phi_rotation_matrix(phi), np.array([0, -1, 0]))
    nProj = norm[0] * axis0[0] + norm[1] * axis0[1]

    # Get phi for rotational matrix along the norm
    margin = np.sqrt(norm[0] ** 2.0 + norm[1] ** 2.0)
    if nProj > 0:
        theta = np.arctan(norm[2] / margin)
    elif norm[2] >= 0 and nProj < 0:
        theta = np.pi + np.arctan(norm[2] / margin)
    elif norm[2] < 0 and nProj < 0:
        theta = -np.pi + np.arctan(norm[2] / margin)
    elif norm[2] >= 0 and nProj == 0:
        theta = np.pi / 2.0
    elif norm[2] < 0 and nProj == 0:
        theta = -np.pi / 2.0
    theta += 2 * np.pi if theta < 0 else 0

    # Get psi holding rotation-minimizing-frame
    psi = np.pi / 2.0 - np.sign(norm[2]) * phi

    # Apply rotational matrices
    trans = np.matmul(phi_rotation_matrix(phi), pos.T).T
    trans = np.matmul(theta_rotation_matrix(theta, axis1), trans.T).T
    trans = np.matmul(theta_rotation_matrix(psi, norm), trans.T).T

    return trans
