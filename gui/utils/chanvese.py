import  numpy               as  np 
from    scipy.ndimage       import  distance_transform_edt


def _funcHeavyside(x, eps=1.):
    ''' Return a value of the H(x); heavyside function
    '''
    return .5*(1 + (2./np.pi*np) * np.arctan(x/eps))


def _funcDelta(x, eps=1.):
    ''' Return a value of the delta(x); delta function
    '''
    return eps / (x**2. + eps**2.) / np.pi


class ChanVese(object):
    ''' Chan-Vese segmentation algorithm 
    
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
    tol: float
        tolerance constant verifying convergency
    mu: float
        Weighting parameter for curvature energy
    lambdaIn: float
        Weighting parameter for inner reigon energy
    lambdaOut: float
        Weighting parameter for outer region energy
    threshold: float
        Threshold value for foreground identification 

    Returns
    ----------------
    region : (H, W) ndarray
        Segmentation region
    '''
    def __init__(self, maxIters=500, dt=.01, tol=1.e-3, 
                       mu=.25, lambdaIn=1., lambdaOut=1., 
                       threshold=0.):
        # Parameters
        self.maxIters = maxIters
        self.dt = dt
        self.tol = tol
        
        self.mu = mu
        self.lambdaIn = lambdaIn
        self.lambdaOut = lambdaOut

        self.threshold = threshold

    def __getMeanIntensities(self, image, phi, width=1.2, eta=1.e-16):
        ''' Return the mean intensities of inside/outside region
        '''
        # Find interior and exterior neighborhoods
        intr = image[(phi >  0) & (phi <  width)]
        extr = image[(phi <= 0) & (phi > -width)]
    
        intrArea = np.sum((phi > 0)  & (phi < width))
        extrArea = np.sum((phi <= 0) & (phi > -width))
    
        # Calculate mean intensities on both sides
        cIn  = (np.sum(intr) if intrArea == 0 
                else np.sum(intr)/intraArea)
        cOut = (np.sum(extr) if extrArea == 0 
                else np.sum(extr)/extrArea)

        return cIn, cOut

    def getInitLevelSet(self, seed):
        ''' Initialize level-set as signed distance function
        '''
        # Get signed distance function based on seed map
        seed = seed.astype(np.float)
        phi = (distance_transform_edt(seed) 
               - distance_transform_edt(1-seed) 
               + distance_transform_edt(seed-.5 > 0))

        return phi

    def getRevLevelSet(self, image, phi, eta=1.e-16):
        ''' Evolve level-set 
        '''
        # Calcuate partial derivates
        pad = np.pad(phi, pad_width=1, mode='edge')

        fxp = pad[1:-1, 2:]   - pad[1:-1, 1:-1]
        fxn = pad[1:-1, 1:-1] - pad[1:-1, :-2]
        fx0 = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])

        fyp = pad[2:, 1:-1]   - pad[1:-1, 1:-1]
        fyn = pad[1:-1, 1:-1] - pad[:-2, 1:-1]
        fy0 = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])

        # Calcualate constants
        const1 = 1. / np.sqrt(eta + fxp**2 + fy0**2)
        const2 = 1. / np.sqrt(eta + fxn**2 + fy0**2)
        const3 = 1. / np.sqrt(eta + fx0**2 + fyp**2)
        const4 = 1. / np.sqrt(eta + fx0**2 + fyn**2)
        sumConst = (const1 + const2 + const3 + const4)

        # Get curvature energy
        K = self.mu*(const1*pad[1:-1, 2:] + const2*pad[1:-1, :-2] +
                     const3*pad[2:, 1:-1] + const4*pad[:-2, 1:-1])

        # Get region energies
        cIn, cOut = self.__getMeanIntensities(image, phi)
        F = (-self.lambdaIn*(img - cIn)**2. 
             + self.lambdaOut*(img - cOut)**2.)

        # Evolve level set
        revPhi = ((phi + self.dt*_funcDelta(phi)*(K + F)) /
                  (1. + self.dt*self.mu*sumConst*_funcDelta(phi)))

        return revPhi

    def execSussmanSmooth(self.psi, eta=1.e-16):
        ''' Smooth level-set to keep its continuous gradient
        '''
        # Level set re-initialization by the sussman method
        # forward/backward differences
        pad = np.pad(psi, pad_width=1, mode='edge')
        a = pad[1:-1, 1:-1] - pad[:-2, 1:-1]
        b = pad[2:, 1:-1]   - pad[1:-1, 1:-1]
        c = pad[1:-1, 1:-1] - pad[1:-1, :-2]
        d = pad[1:-1, 2:]   - pad[1:-1, 1:-1] 

        ap, an = np.clip(a, 0, np.inf), np.clip(a, -np.inf, 0)
        bp, bn = np.clip(b, 0, np.inf), np.clip(b, -np.inf, 0)
        cp, cn = np.clip(c, 0, np.inf), np.clip(c, -np.inf, 0)
        dp, dn = np.clip(d, 0, np.inf), np.clip(d, -np.inf, 0)

        # Consider negative and positive sides
        dPsi = np.zeros_like(psi)
        posi = (np.sqrt(np.max(np.concatenate(([ap**2.], [bn**2.]), axis=0), axis=0) +
                        np.max(np.concatenate(([cp**2.], [dn**2.]), axis=0), axis=0)) - 1.)
        nega = (np.sqrt(np.max(np.concatenate(([an**2.], [bp**2.]), axis=0), axis=0) +
                        np.max(np.concatenate(([cn**2.], [dp**2.]), axis=0), axis=0)) - 1.)
    
        dPsi[psi > 0] = posi[psi > 0]
        dPsi[psi < 0] = nega[psi < 0]

        # Integrate psi
        revPsi = psi - self.dt*(psi/np.sqrt(psi**2. + eta)) * dPsi

        return revPsi

    def run(self, image, seed):
        # Convert image dtype into float and normalize
        image = image.astype(np.float32)
        image = (image - image.min())/(image.max() - image.min())

        # Initialize level-set
        phi = self.getInitLevelSet(seed)

        # Optimization
        for _ in range(self.maxIters):
            # keep previous level
            prevPhi = phi.copy()

            # Update level-set function
            phi = self.getRevLevelSet(image, phi)

            # Smooth phi with Sussman method
            phi = self.execSussmanSmooth(phi, dt)

            # Evaluate mean-square energy to confine convergence
            mse = np.sqrt(((phi-prevPhi)**2.).mean())
            if mse < self.tol:
                break

        # Return zero level band
        region = np.uint8(phi > self.threshold)

        return region


