from collections import deque, defaultdict

import numpy as np


# Displacements (Moore's neighborhood)
displi = [
    -1,  0,  1, -1,  0,  1, -1,  0,  1,
    -1,  0,  1, -1,      1, -1,  0,  1,
    -1,  0,  1, -1,  0,  1, -1,  0,  1
]
displj = [ 
    1,  1,  1,  1,  1,  1,  1,  1,  1,
    0,  0,  0,  0,      0,  0,  0,  0,
    -1, -1, -1, -1, -1, -1, -1, -1, -1
]
displk = [ 
    1,  1,  1,  0,  0,  0, -1, -1, -1,
    1,  1,  1,  0,      0, -1, -1, -1,
    1,  1,  1,  0,  0,  0, -1, -1, -1
]


def is_boundary(pos, shape):
    """Return boolean for checking boundary

    Parameters
    ----------------
    pos: (3,) list, tuple or ndarray
        position to check boundary condition

    Returns
    ----------------
    boolean: boolean
        Boundary condition checker
    """
    # Check boundary condition
    if pos[0] == -1 or pos[0] == shape[0]:
        return True
    if pos[1] == -1 or pos[1] == shape[1]:
        return True
    if pos[2] == -1 or pos[2] == shape[2]:
        return True

    return False


def box_kernel(domain, pos, shape):
    """Return box domain with 'size'

    Parameters
    ----------------
    domain: (H, W, D) ndarray
        input domain
    pos: (3,) list, tuple or ndarray
        center position of local box domain
    (optional) shape: (3,) list, tuple or ndarray
        shape of output domain (lW, lD, lH)

    Returns
    ----------------
    box: (lW, lD, lH) ndarray
        local box domain
    """
    # Get domain dimensions
    height, width, depth = domain.shape
    lh, lw, ld = shape

    # Set half size of edges
    window = (
        slice(max(pos[0] - lh // 2, 0), min(pos[0] + lh // 2 + 1, height)),
        slice(max(pos[1] - lw // 2, 0), min(pos[1] + lw // 2 + 1, width)),
        slice(max(pos[2] - ld // 2, 0), min(pos[2] + ld // 2 + 1, depth)),
    )

    # Extract local box domain
    box = domain[window]

    return box


def build_tree_node(mark):
    """Extract 'tip' and 'junction' nodes

    Parameters
    ----------------
    mark: (H, W, D) ndarray
        volumetric marked data

    Returns
    ----------------
    tip: (H, W, D) ndarray
        volumetric marked tip nodes
    junction: (H, W, D) ndarray
        volumetric marked junction nodes
    """
    # Declare tip and junction
    tip, junction = (np.zeros_like(mark), np.zeros_like(mark))

    for i, j, k in np.argwhere(mark != 0):
        # Select a class
        cls = mark[i, j, k]

        # Open box kernel
        box = box_kernel(mark, (i, j, k), (5, 5, 5))
        box = np.uint8(box == cls)

        # Check kernel environment
        if box.shape != (5, 5, 5):
            continue

        # Get shell and face domains
        shell, face = box.copy(), box.copy()
        shell[2, 2, 2], face[1:4, 1:4, 1:4] = 0, 0

        # Verify tip node
        neighbor1 = region_growing(shell)
        neighbor2 = region_growing(face)
        if (
            3 <= np.sum(box) <= 4
            and np.max(neighbor1) == 1
            and np.max(neighbor2) == 1
        ):
            tip[i, j, k] = cls

        # Check kernel for junction
        if np.sum(face) < 3:
            continue
        if np.sum(box[1:4, 1:4, 1:4]) < 4:
            continue

        # Verify junction node
        neighbor = region_growing(face)
        adjacent = box_kernel(junction, (i, j, k), (3, 3, 3))
        if np.sum(adjacent) == 0 and np.max(neighbor) >= 3:
            junction[i, j, k] = cls

    return tip, junction


def region_growing(mark):
    """Do region-growth

    Parameters
    ----------------
    mark: (H, W, D) ndarray
        volumetric marked data

    Returns
    ----------------
    region: (H, W, D) ndarray
        volumetric labeled data
    """
    # Declare domain of region
    region = np.zeros(mark.shape, dtype=np.int8)

    # Grow region
    cls = 1
    for i, j, k in np.argwhere(mark != 0):
        # skip visited location
        if region[i, j, k] != 0:
            continue
        else:
            region[i, j, k] = cls

        # Initial seed
        query = deque([(cls, i, j, k)])

        # Search neighboring pixels(or voxels)
        while len(query) > 0:
            # class, position
            cls, iq, jq, kq = query.popleft()

            for di, dj, dk in zip(displi, displj, displk):
                ii, jj, kk = (iq + di, jq + dj, kq + dk)

                # Check environment
                if is_boundary((ii, jj, kk), mark.shape):
                    continue
                if mark[ii, jj, kk] == 0:
                    continue
                if region[ii, jj, kk] != 0:
                    continue

                # Mark pixel
                region[ii, jj, kk] = cls

                # Append next step
                query.append((cls, ii, jj, kk))

        # Update class
        cls += 1

    return region


def find_neighboring_node(pos, node):
    """Return boolean for checking existence of neighboring node"""
    for di, dj, dk in zip(displi, displj, displk):
        ii, jj, kk = pos[0] + di, pos[1] + dj, pos[2] + dk

        # Check boundary condition
        if is_boundary((ii, jj, kk), node.shape):
            continue
        if node[ii, jj, kk] > 0:
            return (ii, jj, kk)

    return None


def find_connected_path(p, q, region, visited):
    """Return branch connecting two nodes 'p' and 'q'"""
    # Declare connection
    path = [p]

    while q not in path:
        # Initiate current step
        step, (i, j, k) = region[tuple(path[-1])], path[-1]

        for di, dj, dk in zip(displi, displj, displk):
            ii, jj, kk = i + di, j + dj, k + dk

            # Check domain environment
            if is_boundary((ii, jj, kk), region.shape):
                continue
            if region[ii, jj, kk] == 0:
                continue
            if region[ii, jj, kk] >= step:
                continue
            if len(path) > 2 and visited[ii, jj, kk] != 0:
                continue

            # Update connection
            path.append((ii, jj, kk))

            break

        # Warn no path found
        if path[-1] == (i, j, k):
            break

    # Filter invalid path
    path = np.array(path)
    num_visited = np.sum(visited[path[:, 0], path[:, 1], path[:, 2]])
    if num_visited > 2 or num_visited == len(path):
        return None
    else:
        return path


def skel2graph(skel):
    """Find tip and junction from input skeletonized volume and
    return graph involving connectivity between nodes.

    Parameters
    ----------------
    skel: (H, W, D) ndarray
        skeletonized volumetric data (use scikit-image/skeletonize_3d)

    Returns
    ----------------
    graph: dictionary
        a dictionary involving node connectivity.
        key is pair of two node points and
        its indicating value is path connecting them

        e.g., graph = {((xp, yp, zp), (xq, yq, zq)):
                            [(xp, yp, zp), (x1, y1, z1), ... (xq, yq, zq)], ...}
    """
    # Extract tip and junction node
    tip, junction = build_tree_node(skel)

    # Define node volume
    node = np.uint8(tip + junction > 0)

    # initialize graph container
    graph = defaultdict(list)

    # Define indicator of 'visited'
    visited = np.zeros_like(skel)

    # Search linkage of tree-map
    for pivot in np.argwhere(junction > 0):
        # Reset local marking volume
        mark = np.zeros_like(skel).astype(np.int16)
        mark[tuple(pivot)] = 1

        # Initialize query
        query = deque([(2,) + tuple(pivot)])  # (step, x, y, z)

        # Search connected nodes
        while len(query) > 0:
            # Get query
            step, i, j, k = query.popleft()

            # Search neighboring nodes
            for di, dj, dk in zip(displi, displj, displk):
                ii, jj, kk = i + di, j + dj, k + dk

                # Check current location is out of domain
                if is_boundary((ii, jj, kk), node.shape):
                    continue
                if skel[ii, jj, kk] == 0:
                    continue
                if mark[ii, jj, kk] != 0:
                    continue

                # Update mark domain
                mark[ii, jj, kk] = step

                # Verify existence of neighboring node
                neighbor = find_neighboring_node((ii, jj, kk), node)

                if neighbor == None or neighbor == tuple(pivot):
                    # Append next query
                    query.append((step + 1, ii, jj, kk))
                else:
                    # Search neighbor node leaping adjacent junction
                    if (tuple(pivot), neighbor) in graph:
                        if junction[neighbor] and step == 2:
                            # Append next query
                            query.append((step + 1, ii, jj, kk))
                        else:
                            continue

                    # Mark neighbor
                    mark[neighbor] = step + 1

                    # Find path connecting nodes
                    path = find_connected_path(
                        neighbor, tuple(pivot), mark, visited
                    )

                    # Update 'visited'
                    if not isinstance(path, np.ndarray):
                        continue
                    else:
                        visited[path[:, 0], path[:, 1], path[:, 2]] = 1

                    # Append element into the graph
                    graph[(tuple(pivot), neighbor)] = path[::-1]

                    # if the connection is junction to junction pair
                    if (
                        junction[neighbor]
                        and not graph[(neighbor, tuple(pivot))]
                    ):
                        graph[(neighbor, tuple(pivot))] = path

    return graph
