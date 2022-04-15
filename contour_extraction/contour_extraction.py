from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt


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

    return False


def find_contour(region):
    """Get curvature variables of centerline

    Parameters
    ----------------
    region: (H, W) ndarray
        Input region

    Returns
    ----------------
    contour: (N, 2) ndarray
        Contour points (sorted in clock-wise direction)

    Note
    ----------------
    The contour points are sorted in clock-wise
    """
    # Clock-wise displacements (8 neighbors)
    displi = [-1, -1, -1, 0, 1, 1, 1, 0]
    displj = [-1, 0, 1, 1, 1, 0, -1, -1]

    # Get edge  pixels
    edge = np.uint8(distance_transform_edt(region) == 1)

    # Declare mark domain
    mark = np.zeros_like(edge)

    # Get dimensions
    height, width = edge.shape
    center = (height // 2, width // 2)

    # Initialize query with the position in the row-wise order
    # So, seaching initially starts at right-middle position
    pos = np.argwhere(edge == 1)[0]
    query = deque([(2, pos[0], pos[1])])

    # Search connected component
    contour = []
    while query:
        # Pop current position
        start, i, j = query.popleft()

        # Roll displacements for it starts at
        # the next position of previous component
        dis = displi[start:] + displi[:start]
        djs = displj[start:] + displj[:start]

        # Find connected component in clock-wise
        for end, (di, dj) in enumerate(zip(dis, djs)):
            iq, jq = i + di, j + dj

            # Check domain
            if is_boundary((iq, jq), (height, width)):
                continue
            if not edge[iq, jq]:
                continue
            if mark[iq, jq]:
                continue
            else:
                mark[iq, jq] = 1

            # Update query
            query.append(((start + end + 5) % 8, iq, jq))

            # Append sequential component
            contour.append((iq, jq))
            break

        # Make contour a closet
        if not query:
            contour.append(contour[0])

    return contour
