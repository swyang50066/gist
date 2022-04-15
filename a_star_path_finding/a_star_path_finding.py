import numpy as np


class Node:
    """Arguments container for node connection"""

    def __init__(self, parent=None, pos=None):
        self.parent = parent
        self.pos = pos

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos


def is_boundary(pos, shape):
    """Return boolean for checking boundary

    Parameters
    ----------------
    pos: (N,) list, tuple or ndarray
        position to check boundary condition

    Returns
    ----------------
    boolean: boolean
        Boundary condition checker
    """
    # Check dimensionality
    ndim = len(shape)

    # Check boundary condition
    if pos[0] == -1 or pos[0] == shape[0]:
        return True
    if pos[1] == -1 or pos[1] == shape[1]:
        return True
    if ndim == 3:
        if pos[2] == -1 or pos[2] == shape[2]:
            return True

    return False


def calc_distance(p, q, ndim=3, scale=None):
    """Return Euclidean displitance between 'p' and 'q'

    Parameters
    ----------------
    p: (N,) list, tuple or ndarray
        start position of displitance measure
    q: (N,) list, tuple or ndarray
        end position of displitance measure
    (optional) ndim: integer
        specified input dimension
    (optional) scale: (N,) list, tuple or ndarray
        scale vector giving weights for each axes

    Returns
    ----------------
    displitance: float
        Euclidean displitance between 'p' and 'q'
    """
    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0], q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0], q[1] - p[1], q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]), scale[1]*(q[1] - p[1])])
        else:
            dr = np.array(
                [
                    scale[0]*(q[0] - p[0]),
                    scale[1]*(q[1] - p[1]),
                    scale[2]*(q[2] - p[2]),
                ]
            )

    # Euclidean displitance
    distance = np.sqrt(np.sum(dr**2.))

    return distance


def smooth_path(path):
    """Smooth path between pivots"""
    # Get input dimensionality
    ndim = len(path[0])

    for _ in range(2):
        if len(path) < 3:
            continue

        index = 2
        while index < len(path):
            alpha, beta, gamma = path[index-2:index+1]

            # Remove voxel if tri-connection is too shrinked
            if ndim == 2 and calc_distance(alpha, gamma, ndim=ndim) <= 1.5:
                path.remove(beta)
            elif ndim == 3 and calc_distance(alpha, gamma) <= 1.8:
                path.remove(beta)
            else:
                index += 1

    return path


def a_star_path_finding(p, q, barrier):
    """Reun A* algorithm (a greedy path finding algorithm)

    Parameters
    ----------------
    p: (N,) list
        start position
    q: (N,) list
        end position
    barrier: (H, W) or (H, W, D) ndarray
        barrier map marking forbidden positions

    Returns
    ----------------
    path: (N, 2) or (N, 3) list
        shortest path from 'p' to 'q'
    """
    # Convert 'barrier' to be a ndarray
    barrier = np.array(barrier)

    # Check input dimensionality
    ndim = len(barrier.shape)

    # Displacements (Use Moore's neighborhood)
    if ndim == 2:
        displi = [-1, 0, 1, -1, 1, -1, 0, 1]
        displj = [-1, -1, -1, 0, 0, 1, 1, 1]
    elif ndim == 3:
        displi = [
            -1, 0, 1, -1, 0, 1, -1, 0, 1, 
            -1, 0, 1, -1, 1, -1, 0, 1,
            -1, 0, 1, -1, 0, 1, -1, 0, 1
        ]
        displj = [
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        displk = [
            -1, -1, -1, 0, 0, 0, 1, 1, 1,
            -1, -1, -1, 0, 0, 1, 1, 1,
            -1, -1, -1, 0, 0, 0, 1, 1, 1,
        ]
    else:
        print("THE ALGORITHM ENCOUNTERED INVALID INPUT DIMENSION!!!")
        raise ValueError

    # Initialize containers
    dept_node = Node(None, p)
    arrv_node = Node(None, q)
    open_list, close_list = [dept_node], []

    # Run A* algorithm
    while open_list:
        # Set current node
        curr_index, curr_node = 0, open_list[0]

        # Substitude curr_node to a node in the open_list with larger f
        for index, open_node in enumerate(open_list):
            if open_node.f < curr_node.f:
                curr_index, curr_node = index, open_node

        # Deliver the curr_node from open_list to close_list
        open_list.pop(curr_index)
        close_list.append(curr_node)

        # Varify the algorithm found the arrv_node
        if curr_node == arrv_node:
            path = []
            while curr_node is not None:
                path.append(curr_node.pos)

                # look parent node
                curr_node = curr_node.parent

            # Smooth path
            path = smooth_path(path)

            return path[::-1]

        # Search neighboring nodes
        children = []
        if ndim == 2:
            for di, dj in zip(displi, displj):
                ii, jj = (curr_node.pos[0] + di, curr_node.pos[1] + dj)

                # Check domain environment
                if is_boundary((ii, jj), barrier.shape):
                    continue
                if barrier[ii, jj]:
                    continue

                # Append child
                children.append(Node(curr_node, (ii, jj)))
        elif ndim == 3:
            for di, dj, dk in zip(displi, displj, displk):
                ii, jj, kk = (
                    curr_node.pos[0] + di,
                    curr_node.pos[1] + dj,
                    curr_node.pos[2] + dk,
                )

                # Check domain environment
                if is_boundary((ii, jj, kk), barrier.shape):
                    continue
                if barrier[ii, jj, kk]:
                    continue

                # Append child
                children.append(Node(curr_node, (ii, jj, kk)))

        # Expand siblings
        for child in children:
            if child in close_list:
                continue

            # Update cost (f = g + h)
            child.g = curr_node.g + 1
            child.h = calc_distance(child.pos, arrv_node.pos, ndim=ndim)**2.
            child.f = child.g + child.h

            # Append child in the open_list if it has larger g
            still_open_list = [
                open_node
                for open_node in open_list
                if child == open_node and child.g > open_node.g
            ]
            if len(still_open_list) > 0:
                continue

            # Update open_list
            open_list.append(child)
