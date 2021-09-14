import  numpy               as  np
from    scipy.ndimage       import  convolve, gaussian_filter


def _funcGradientIndicator(x, alpha=2, sigma=1):
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


def _funcDiracDelta(x, epsilon=1.5):
    ''' Return value of Dirac delta function
    '''
    f = .5*(1. + np.cos(np.pi*x/epsilon))/epsilon
    b = np.uint8(np.logical_and(x <= epsilon, x >= epsilon))

    return f*b


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


def _divergence(u, v):
    ''' Return divergence
    '''
    # Calculate gradient
    ux, _ = _calcGradient(u)
    _, vy = _calcGradient(v)
    
    return ux + vy 


def DRLSE(object):
    ''' Distance Regularized Level-set Evolution

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
    def __init__(self, maxIter=500, refinement=10, dt=.1, tol=1.e-4,
                       mu=.2, lambda0=5, alpha=-3.,
                       threshold=0.):
        # Model parameter
        self.mu = mu
        self.lambda0 = lambda0
        self.alpha = alpha

        self.maxIter = maxIter
        self.refinement = refinement
        self.dt = dt
        self.tol = tol

        self.threshold = threshold

    def __getDistRegP2(self, phi):
        ''' Return distance regularizer with double-well potential
        '''
        # Get coefficients
        gradx, grady = _calcGradient(phi)
        s = np.sqrt(gradx**2. + grady**2.)

        a = np.uint8(np.logical_and(s >= 0, s <= 1))
        b = np.uint8(s > 1)

        # Calculate derivative of double-well potential term
        ps = a*np.sin(2.*np.pi*s)/(2.*np.pi) + b*(s - 1.)

        # Calculate distance regularizer
        dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))

        return _divergence(dps*gradx - gradx, dps*grady - phi_y) + _calcLaplacian(phi)

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

    def getRevLevelSet(self, g, v, phi, mode="iteration", eps=1.e-10):
        ''' Evolve level-set
        '''
        # Get Dirac-delta of level-set function
        dirac = _funcDiracDelta(phi)
        
        # Area
        if mode != "refinement":
            Earea = dirac*g  
        else:
            Earea = 0.

        # Edge
        gradx, grady = _calcGradient(phi)
        s = np.sqrt(gradx**2. + grady**2.)
        norm = (gradx/(s + eps), grady/(s + eps))
        curvature = _calcCurvature(phi)

        Eedge = dirac*(v[0]*norm[0] + v[1]*norm[1]) + dirac*g*curvature

        # Distance regularization
        Ereg = self.__getDistRegP2(phi)

        # Evolve level-set
        revPhi = phi + self.dt*(self.mu*Ereg + self.lambda0*Eedge + self.alpha*Earea)

        return revPhi

    def run(self, image, seed, sigma=1):
        # Convert input image format to be a float container
        image = np.array(image, dtype=np.float32)
        image = (image - image.min())/(image.max() - image.min())

        # Get gradient indicator and its gradient
        g = _funcGradientIndicator(image)
        v = _calcGradient(g)

        # Initialize level-set
        phi = self.getInitLevelSet(seed)

        for _ in range(self.maxIter):
            # keep previous level
            prevPhi = phi.copy()

            # Update level-set function
            phi = self.getRevLevelSet(g, v, phi)

            # Evaluate mean-square energy to confine convergence
            mse = np.sqrt(((phi-prevPhi)**2.).mean())
            if mse < self.tol:
                break

        # Refine zero-level contour by further evolution with 'alpha=0'
        for _ in range(self.refinement):
            phi = self.getRevLevelSet(image, phi, mode="refine") 

        # Return positive levels as region
        region = np.uint8(phi > self.threshold)

        return region
