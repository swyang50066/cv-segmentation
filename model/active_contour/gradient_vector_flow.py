import  sys
sys.path.append("../../utils")      # Use predefined utility functions 

import  cv2
import  numpy       as  np
from    scipy.ndimage       import  convolve, gaussian_filter
from    scipy.interpolate   import  splprep, splev, RectBivariateSpline

from    contour     import  getContour


## Sobel kernels
# Here, we are going to use image coordinates matching
# array coordinates like (x:height, y:width)
SOBEL_X = np.array([[ 1.,  2.,  1.],
                         [ 0.,  0.,  0.],
                         [-1., -2., -1.]])
SOBEL_Y = np.array([[ 1.,  0., -1.],
                         [ 2.,  0., -2.],
                         [ 1.,  0., -1.]])


# Laplacian kernel
LAPLACIAN = np.array([[0.,  1., 0.],
                      [1., -4., 1.],
                      [0.,  1., 0.]])


def _calcLaplacian(x):
    ''' Apply Laplacian operator 
    '''
    return convolve(x, LAPLACIAN, mode="nearest")


def _calcGradient(x):
    ''' Return gradient map
    '''
    ## Apply gaussian filter
    #x = gaussian_filter(x, sigma=1)
    
    # Pad input domain
    pad = np.pad(x, pad_width=1, mode="edge")

    # Calculate derivatives
    gradx = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])
    grady = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])

    return (gradx, grady)


def _getEdgeMap(u, sigma=2):
    ''' Return image edge map by  using Sobel operator
    '''
    # Apply gaussian filter
    u = gaussian_filter(u, sigma=sigma)
    
    # Extract image edges with Sobel filter
    fx = convolve(u, SOBEL_X, mode="nearest")
    fy = convolve(u, SOBEL_Y, mode="nearest")

    return np.sqrt(.5*(fx**2. +  fy**2.))


class GVFSnake(object):
    ''' Gradient vector flow based snake evolution for image segmentation

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
    def __init__(self, alpha=.01, beta=.1, gamma=.01, 
                       maxIter=1000,
                       maxDispl=1.,
                       eps=.1,
                       period=10):
        # Model parameters
        self.alpha = alpha    # continuity parameter
        self.beta = beta      # smoothness parameter
        self.gamma = gamma    # artificial time step

        # Numerical parameters
        self.maxIter = int(maxIter)
        self.maxDispl = float(maxDispl)
        self.eps = eps

        # Evolution history
        self.period = period
        self.xhistory = [0] * period    # container of snake energies for evolution
        self.yhistory = [0] * period    

    def getGradientVectorFlow(self, fx, fy, CLF=.25, mu=1., dx=1., dy=1., maxIter=1000, eps=1.e-6):
        ''' Return gradient vector flow
        '''
        # Set artificial time step under CFL restriction
        dt = CLF*dx*dy/mu
        
        # Set coefficients
        b = fx**2. + fy**2.
        c1, c2 = b*fx, b*fy
    
        # Optimize gradient vector flow
        currGVF = (fx, fy)    # initially set the vector flow to be gradient of edge map
        for i in range(maxIter):
            # Evolve flow
            nextGVF = ((1. - b*dt)*currGVF[0] + CLF*_calcLaplacian(currGVF[0]) + c1*dt,
                       (1. - b*dt)*currGVF[1] + CLF*_calcLaplacian(currGVF[1]) + c2*dt)
            
            # Update flow
            delta = np.sqrt((currGVF[0] - nextGVF[0])**2. 
                            + (currGVF[1] - nextGVF[1])**2.)
            if np.mean(delta) < eps:
                break
            else:
                currGVF = nextGVF
    
        return currGVF

    def run(self, image, seed):
        # Convert input image format to be a float container
        image = np.array(image, dtype=np.float32)
        image = (image - image.min())/(image.max() - image.min())

        # Get input dimensions
        if len(image.shape) == 2:
            height, width = image.shape
        elif len(image.shape) == 3:
            height, width, _ = image.shape
            image = np.mean(image, axis=-1)

        # Get contour from seed region
        contour = np.array(getContour(seed), dtype=np.float32)

        # Initialize snake pivots
        tck, _ = splprep(contour.T, s=0)
        snake = splev(np.linspace(0, 1, 2*len(contour)), tck)
        snake = np.array(snake).T.astype(np.float32)
        snake = np.array(contour).astype(np.float32)

        # Discretize snake
        xx, yy = snake[:, 0], snake[:, 1]
        for p in range(self.period):
            self.xhistory[p] = np.zeros(len(snake), dtype=np.float32)
            self.yhistory[p] = np.zeros(len(snake), dtype=np.float32)

        # Evaluate image energies
        edge = _getEdgeMap(image)

        # Evaluate gradient of edge map
        gradx, grady = _calcGradient(edge)
    
        # Get gradient vector flow (i.e., GVF)
        GVF = self.getGradientVectorFlow(gradx, grady)

        # Get continous GVF
        xinterp = RectBivariateSpline(np.arange(height),
                                      np.arange(width),
                                      GVF[0],
                                      kx=2, ky=2, s=0)
        yinterp = RectBivariateSpline(np.arange(height),
                                      np.arange(width),
                                      GVF[1],
                                      kx=2, ky=2, s=0)

        # Build snake shape matrix
        matrix = np.eye(len(snake), dtype=float)
        a = (np.roll(matrix, -1, axis=0)
             + np.roll(matrix, -1, axis=1)
             - 2.*matrix)  # second order derivative, central difference
        b = (np.roll(matrix, -2, axis=0)
             + np.roll(matrix, -2, axis=1)
             - 4.*np.roll(matrix, -1, axis=0)
             - 4.*np.roll(matrix, -1, axis=1)
             + 6.*matrix)  # fourth order derivative, central difference
        A = -self.alpha*a + self.beta*b

        # Make inverse matrix needed for the numerical scheme
        inv = np.linalg.inv(A + self.gamma*matrix).astype(np.float32)

        # Do optimization
        for step in range(self.maxIter):
            # Get point-wise energy values
            fx = xinterp(xx, yy, grid=False).astype(np.float32)
            fy = yinterp(xx, yy, grid=False).astype(np.float32)

            # Evaluate new snake
            xn = inv @ (self.gamma*xx + fx)
            yn = inv @ (self.gamma*yy + fy)

            # Confine displacements
            dx = self.maxDispl*np.tanh(xn - xx)
            dy = self.maxDispl*np.tanh(yn - yy)

            # Update snake
            xx, yy = xx + dx, yy +dy

            # Verify nermerical convergency
            # Update histories
            index = step % (self.period + 1)
            if index < self.period:
                self.xhistory[index] = xx
                self.yhistory[index] = yy
            else:
                delta = np.max(np.abs(np.array(self.xhistory) - xx)
                               + np.abs(np.array(self.yhistory) - yy), axis=1)
                if np.min(delta) < self.eps: 
                    break

        # Remark region of snake contour into pixel map
        seed = cv2.fillConvexPoly(seed, points=snake.astype(np.int), color=1)
        
        return seed, np.stack([xx, yy], axis=1)
