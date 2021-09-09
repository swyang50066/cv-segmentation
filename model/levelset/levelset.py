import  numpy       as  np
from    scipy.ndimage.filters   import  gaussian_filter


def __funcGradientIndicator(x, alpha=2, sigma=1):
    ''' Return gradient indicator
    '''
    # Apply gaussian kernel
    x = gaussian_filter(x, sigma)

    # Calc gradients
    pad = np.pad(x, pad_width=1, mode="edge")
    fx = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])
    fy = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])
    grad = np.sqrt(fx**2. + fy**2.)

    return 1./(1. + grad**alpha)


def _calcGradient(x):
    ''' Calculate gradient components
    '''
    # Calc gradients
    pad = np.pad(x, pad_width=1, mode="edge")
    grads = (.5*(pad[2:, 1:-1] - pad[:-2, 1:-1]),
             .5*(pad[1:-1, 2:] - pad[1:-1, :-2]))

    return grads


def _calcCurvature(phi, eta=1.e-8):
    ''' Returns the curvature of a level set 'phi'.
    '''
    # Calcuate partial derivates
    pad = np.pad(phi, pad_width=1, mode='edge')
    fx = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])
    fy = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])
    fxx = pad[2:, 1:-1] - 2.*phi + pad[:-2, 1:-1]
    fyy = pad[1:-1, 2:] - 2.*phi + pad[1:-1, :-2]
    fxy = .25*(pad[2:, 2:]  + pad[:-2, :-2] -
               pad[:-2, 2:] - pad[2:, :-2])

    # Get curvature
    kappa = ((fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2)
             / (np.power(fx**2. + fy**2., 1.5) + eta))

    return kappa


class LevelSet(object):
    ''' Morphological Active Contours without Edges (MorphACWE)
        Active contours without edges implemented with morphological operators. It
        can be used to segment objects in images and volumes without well defined
        borders. It is required that the inside of the object looks different on
        average than the outside (i.e., the inner area of the object should be
        darker or lighter than the outer area on average).
    
    Parameters
    ----------------
    image: (H, W) ndarray
        Input image
    seed: (H, W) ndarray
        Seed map including serial numbers of each classes

    maxIter: integer
        Maximum iterations
    dt: float
        Artifical time displacement as a integration step   
    velosity: float
        Variation rate of level-set deformation
    threshold: float
        Threshold value for foreground identification 

    Returns
    ----------------
    region : (H, W) ndarray
        Segmentation region
    '''
    def __init__(self, maxIter=100, dt=.5, velocity=.5, threshold=.5):
        # Parameters
        self.maxIter = maxIter

        self.dt = dt
        self.velosity = velosity
        self.threshold = threshold

    def run(self, image, seed):
        # Initialize level-set
        phi = np.zeros_like(seed, dtype=np.int16)
        phi[seed > 0], phi[seed == 0] = 2, -2

        # Get stopping function
        indicator = _funcGradientIndicator(image)

        # Calculate gradient of indicator  
        fx, fy = _calcGradient(indicator)

        # Optimization
        for i in range(self.maxIter):
            # Calculate gradient of level-set
            gx, gy = _calcGradient(phi)
            grad = np.sqrt(gx**2. + gy**2.)

            # Calculate curvature
            kappa = _calcCurvature(phi)

            # Calculate energy terms
            smoothing = indicator * kappa * grad
            balloon = indicator * self.velosity * grad
            attachment = fx*gx + fy*gy

            # Calculate displacement
            dphi = self.dt*(smoothing + balloon + attachment)

            # Update level-set
            phi = phi + dphi

        return 1. * (phi > self.threshold)
