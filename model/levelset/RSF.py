import  sys
sys.path.append("../../utils")

import  numpy               as np
from    scipy.ndimage       import  convolve, distance_transform_edt

from    utils.gaussian_kernel       import  getGaussianKernel


def _funcHeavyside(x, eps=1.):
    ''' Return a value of the H(x); heavyside function
    '''
    return .5*(1 + (2./np.pi*np) * np.arctan(x/eps))


def _funcDelta(x, eps=1.):
    ''' Return a value of the delta(x); delta function
    '''
    return eps / (x**2. + eps**2.) / np.pi


def _calcLaplacian(x):
    ''' Apply Laplacian operator
    '''
    # Laplacian kernel
    LAPLACIAN = np.array([[0.,  1., 0.],
                          [1., -4., 1.],
                          [0.,  1., 0.]])

    return convolve(x, LAPLACIAN, mode="nearest")


def _calcGradient(x):
    ''' Return gradient map
    '''
    # Pad input domain
    pad = np.pad(x, pad_width=1, mode="edge")

    # Calculate derivatives
    gradx = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])
    grady = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])

    return (gradx, grady)


def _calcCurvature(u, eta=1.e-8):
    ''' Return curvature energy
    '''
    # Pad input domain
    pad = np.pad(u, pad_width=1, mode="edge")

    # Calculate derivatives
    fx = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])
    fy = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])
    fxx = pad[2:, 1:-1] - 2.*u + pad[:-2, 1:-1]
    fyy = pad[1:-1, 2:] - 2.*u + pad[1:-1, :-2]
    fxy = .25*(pad[2:, 2:]  + pad[:-2, :-2]
               - pad[:-2, 2:] - pad[2:, :-2])

    return ((fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2)
            / (np.power(fx**2. + fy**2., 1.5) + eta))


class RSF(object):
    ''' Minimization of Region scalable fitting energy

    Parameters
    ----------------
    image: (H, W) ndarray
        input image.
    seed: (H, W) ndarray
        input seed.

    Returns
    ----------------
    Region (H, W) ndarray
        segmentation label
    '''
    def __init__(self, maxIter=500, dt=.1, tol=1.e-4,
                       mu=1., nu=.3e-3, lambda1=1., lambda2=1.,
                       threshold=0.):
        # Model parameter
        self.mu = mu
        self.nu = nu
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.maxIter = 500
        self.dt = dt
        self.tol = tol
        
        self.threshold = threshold

    def getInitLevelSet(self, seed):
        ''' Initialize level-set as signed distance function
        '''
        # Make seed be float type
        seed = seed.astype(np.float)
        
        # Get signed distance function based on seed map
        phi = (distance_transform_edt(seed)
               - distance_transform_edt(1-seed)
               + distance_transform_edt(seed-.5))

        return phi

    def getRevLevelSet(self, u, phi, sigma=3):
        ''' Evolve level-set
        '''
        # Curvature
        curvature = _calcCurvature(phi)
        Ecurv = self.nu*_funcDelta(phi)*curvature

        # Regularization
        laplacian = _calcLaplacian(phi)
        Ereg = self.mu*(laplacian - curvature);

        # Build scaling kernel
        #height, width = (4*sigma + 1, 4*sigma + 1)
        #KERNEL = np.ones((height, width), dtype=np.float64)
        #KERNEL = KERNEL / height / width
        KERNEL = getGaussianKernel(sigma, (4*sigma+1, 4*sigma+1), ndim=2)

        # Region scalable fitting energy
        region = _funcHeavyside(phi)
        
        KIH1 = convolve(region*u, KERNEL, mode="nearest")
        KH1 = convolve(region, KERNEL, mode="nearest")
        f1 = KIH1 / KH1
        
        KIH2 = convolve((1. - region)*u, KERNEL, mode="nearest")
        KH2 = convolve((1. - region), KERNEL, mode="nearest")
        f2 = KIH2 / KH2
        
        R1 = (self.lambda1 - self.lambda2)*u*y;
        R2 = convolve(self.lambda1*f1, KERNEL, mode="nearest")
        R3 = convolve(self.lambda2*f1*f1 - self.lambda2*f2*f2, KERNEL, mode="nearest")
        Ersf = -_funcDelta(phi)*(R1 - 2.*R2*u + R3)

        # Evolve level-set
        revPhi = phi + self.dt*(Ecurv + Ereg + Ersf)
        
        return revPhi

    def run(self, image, seed):
        # Convert input image format to be a float container
        image = np.array(image, dtype=np.float32)
        image = (image - image.min())/(image.max() - image.min())

        # Initialize level-set
        phi = self.getInitLevelSet(seed)

        for _ in range(self.maxIter):
            # keep previous level
            prevPhi = phi.copy()

            # Update level-set function
            phi = self.getRevLevelSet(image, phi)

            # Evaluate mean-square energy to confine convergence
            mse = np.sqrt(((phi-prevPhi)**2.).mean())
            if mse < self.tol:
                break

        # Return positive levels as region
        region = np.uint8(phi > self.threshold)
        
        return region

