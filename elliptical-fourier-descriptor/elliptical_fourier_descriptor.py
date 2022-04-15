import numpy as np
from scipy.ndimage import distance_transform_edt


def calc_euclidean_distance(p, q, scale=None):
    """Return Euclidean distance between 'p' and 'q'
    Parameters
    ----------------
    p: (N,) list, tuple or ndarray
        start position of distance measure
    q: (N,) list, tuple or ndarray
        end position of distance measure
    (optional) scale: (N,) list, tuple or ndarray
        scaling weights for each dimensions
    Returns
    ----------------
    dists: (N,) float
        Euclidean distance between 'p' and 'q'
    """
    # Vectorize inputs
    if not isinstance(p, type(np.ndarray)):
        p = np.array(p)
    if not isinstance(q, type(np.ndarray)):
        q = np.array(q)

    # Get distance
    if not scale:
        return np.sqrt(np.sum((q - p) ** 2.0))
    else:
        return np.sqrt(np.sum((scale * (q - p)) ** 2.0))


def get_sharpness_threshold(contour, interval=10, eta=0.99):
    """Measure sharpness of contour"""
    # Make cyclic list of contour
    if len(contour) < 10 * interval:
        interval = max(3, len(contour) // 10)

    cycle = contour[:-1]
    cycle = cycle[-interval:] + cycle + cycle[:interval]

    # Search maximum/minimum curvatures
    maxCurvature, minCurvature = 0, len(contour)
    for k in range(len(cycle[: -2 * interval])):
        # Get pivot points
        prev = np.array([cycle[k][0], cycle[k][1], 0])
        curr = np.array([cycle[k + interval][0], cycle[k + interval][1], 0])
        next = np.array(
            [cycle[k + 2 * interval][0], cycle[k + 2 * interval][1], 0]
        )

        # Displacement vectors
        vec1 = np.cross(prev - curr, next - curr)
        vec2 = np.cross(vec1, prev - curr)
        vec3 = np.cross(vec1, next - curr)

        # Corresponding lengths
        arc1 = calc_euclidean_distance(vec1, (0, 0, 0))
        arc2 = calc_euclidean_distance(next, curr)
        arc3 = calc_euclidean_distance(prev, curr)

        # Ignore stright line (i.e., infinite curvature)
        if not arc1:
            continue

        # Get displacement vector from 'curr' to curvature center
        delta = 0.5 * (vec2 * arc2**2.0 - vec3 * arc3**2.0) / arc1**2.0

        # Get curvature radius
        radius = calc_euclidean_distance(delta, (0, 0, 0))

        # Update minimum/maximum
        if radius < minCurvature:
            minCurvature = radius
        if radius > maxCurvature:
            maxCurvature = radius

    # Apply reduction on filtering threshold
    if maxCurvature:
        eta = eta - 0.02 * (1 - minCurvature / maxCurvature)

    return eta


def elliptical_fourier_contour(contour, THRESHOLD=0.95):
    """Get elliptical Fourier descriptor of shape

    Parameters
    ----------------
    contour: (N, 2) ndarray
        Contour points 
    (optional) threshold: float
        Threshold value for low-pass filering

    Returns
    ----------------
    ifft_contour: (N, 2) ndarray
        Ellptical Fourier descriptor components
    """
    # Do Fourier transformation
    fft_contour = np.fft.fft2(contour)

    # Get wavelet amplitudes
    amplitudes = np.abs(np.array(fft_contour)[:, 0])
    norm = np.sum(amplitudes)

    # Get filter threshold
    THRESHOLD = get_sharpness_threshold(contour)

    # Apply low-pass filter
    cumulative = 0
    orders = np.argsort(amplitudes)[::-1]
    for index, order in enumerate(orders):
        cumulative += amplitudes[order] / norm

        if cumulative > THRESHOLD:
            fft_contour[orders[index:]] = 0
            break

    # Do inverse-Fourier transformation
    ifft_contour = np.fft.ifft2(fft_contour).real

    return ifft_contour
