import  numpy           as  np
from    scipy.ndimage       import  distance_transform_edt


def gaussian_kernel(sigma, size=5, ndim=3):
    ''' Build Gaussian kernel

        Parameters
        ----------------
        sigma: integer
            discreted variance of Gaussian kernel
        (optional) size: (2,) or (3,) list, tuple or ndarray
            size of box domain (W, H) or (W, D, H)
        (optional) ndim:
            specified input dimension

        Returns
        ----------------
        kernel: (W, H) or (W, D, H) ndarray
            Discreted Gaussian kernel
    '''
    # Open kernal domain
    if ndim == 2:
        # Get domain dimensions
        shape = domain.shape
        width, height = size[0] // 2, size[1] // 2

        kernel = np.zeros(size)
        kernel[width, height] = 1
    else:
        # Get domain dimensions
        shape = domain.shape
        width, depth, height = size[0] // 2, size[1] // 2, size[2] // 2

        kernel = np.zeros(size)
        kernel[width, depth, height] = 1

    # Build discretized gaussian kernel
    kernel = distance_transform_edt(1 - kernel)
    kernel = (
        np.exp(-.5*kernel**2. / sigma**2.)
        / np.sqrt(2*np.pi*sigma**2.)
    )

    # Normalize kernel
    kernel /= np.sum(kernel)

    return kernel

