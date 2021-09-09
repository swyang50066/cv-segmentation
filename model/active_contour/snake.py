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
SOBEL_X = .125*np.array([[ 1.,  2.,  1.],
                         [ 0.,  0.,  0.],
                         [-1., -2., -1.]])
SOBEL_Y = .125*np.array([[ 1.,  0., -1.],
                         [ 2.,  0., -2.],
                         [ 1.,  0., -1.]])


def _calcEdgeEnergy(u):
    ''' Return image edge energy by using Sobel operator
    '''
    # Extract image edges with Sobel filter
    gradx = convolve(u, SOBEL_X, mode="constant")
    grady = convolve(u, SOBEL_Y, mode="constant")

    '''
    # Pad input domain
    pad = np.pad(u, pad_width=1, mode="edge")
   
    # Calculate derivatives
    fx = .5*(pad[2:, 1:-1] - pad[:-2, 1:-1])
    fy = .5*(pad[1:-1, 2:] - pad[1:-1, :-2])

    return -(fx**2. + fy**2.)
    '''
    return (gradx**2. + grady**2.)

def _calcScaleEnergy(u, sigma=3):
    ''' Return image scale energy
    '''
    # Apply gaussian filter
    u = gaussian_filter(u, sigma=sigma)

    # Pad input domain
    pad = np.pad(u, pad_width=1, mode="edge")

    # Calculate derivatives
    fxx = pad[2:, 1:-1] - 2.*u + pad[:-2, 1:-1]
    fyy = pad[1:-1, 2:] - 2.*u + pad[1:-1, :-2]

    return -(fxx**2. + fyy**2.)


def _calcCurvatureEnergy(u, eta=1.e-8):
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


class Snake(object):
    ''' Snake Algotirhm of Active-Contour Based Image Segmentation

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
                       wline=0., wedge=1., wscale=0., wcurv=0.,
                       maxIter=1000,
                       maxDispl=1.,
                       eps=.1,
                       period=10):
        # Model parameters
        self.alpha = alpha    # continuity parameter
        self.beta = beta      # smoothness parameter
        self.gamma = gamma    # artificial time step

        # Contraint weights
        self.wline = wline    # weight of line functional
        self.wedge = wedge    # weight of edge functional
        self.wscale = wscale    # weight of scale functional
        self.wcurv = wcurv    # weight of curvature funtional

        # Numerical parameters
        self.maxIter = int(maxIter)
        self.maxDispl = float(maxDispl)
        self.eps = eps

        # Evolution history
        self.period = period
        self.xhistory = [0] * period    # container of snake energies for evolution
        self.yhistory = [0] * period    

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
        Eedge = _calcEdgeEnergy(image)
        Escale = _calcScaleEnergy(image)
        Ecurv = _calcCurvatureEnergy(image)

        # Get total image energy
        Etot = (self.wline*image + self.wedge*Eedge
                + self.wscale*Escale + self.wcurv*Ecurv)
            
        # Get continuous image field
        interp = RectBivariateSpline(np.arange(height), 
                                     np.arange(width),
                                     Etot,
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
            fx = interp(xx, yy, dx=1, grid=False).astype(np.float32)
            fy = interp(xx, yy, dy=1, grid=False).astype(np.float32)

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
                distance = np.min(np.max(np.abs(np.array(self.xhistory) - xx)
                                         + np.abs(np.array(self.yhistory) - yy), axis=1))
                if distance < self.eps: break

        # Remark region of snake contour into pixel map
        seed = cv2.fillConvexPoly(seed, points=snake.astype(np.int), color=1)

        return seed, np.stack([xx, yy], axis=1)
