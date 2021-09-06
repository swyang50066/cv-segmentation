from    itertools       import  cycle

import  numpy           as  np
from    scipy.ndimage   import  binary_erosion, binary_dilation


class Operator(object):
    def __init__(self, iterable):
        """ Call functions from the iterable each time it is called.
        """
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        value = next(self.funcs)
        
        return value(*args, **kwargs)


class MorphChanVese(object):
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
    (optional) numSmoothing : interger
        Number of applying smoothing operator (larger values lead to smoother
        segmentations)
    lambda1 : float, optional
        Weighting parameter for inner region energy
    lambda2 : float, optional
        Weighting parameter for outer region energy
    
    Returns
    ----------------
    region : (H, W) ndarray
        Segmentation region
    '''
    def __init__(self, maxIter=200, numSmoothing=1, lambda1=1., lambda2=2.):
        # Parameters
        self.maxIter = maxIter
        self.numSmoothing = numSmoothing
        
        self.labmda1 = lambda1
        self.labmda2 = lambda2

        # SI and IS operators for 2D and 3D.
        self.kernel2d = [np.eye(3),
                         np.array([[0, 1, 0]]*3),
                         np.flipud(np.eye(3)),
                         np.rot90([[0, 1, 0]]*3)]

        self.kernel3d = [np.zeros((3, 3, 3)) for i in range(9)]
        self.kernel3d[0][:, :, 1] = 1
        self.kernel3d[1][:, 1, :] = 1
        self.kernel3d[2][1, :, :] = 1
        self.kernel3d[3][:, [0, 1, 2], [0, 1, 2]] = 1
        self.kernel3d[4][:, [0, 1, 2], [2, 1, 0]] = 1
        self.kernel3d[5][[0, 1, 2], :, [0, 1, 2]] = 1
        self.kernel3d[6][[0, 1, 2], :, [2, 1, 0]] = 1
        self.kernel3d[7][[0, 1, 2], [0, 1, 2], :] = 1
        self.kernel3d[8][[0, 1, 2], [2, 1, 0], :] = 1

    def optSupInf(self, u):
        ''' SI operator
        '''
        if np.ndim(u) == 2:
            kernels = self.kernel2d
        elif np.ndim(u) == 3:
            kernels = self.kernel3d
        else:
            raise ValueError("'u' has an invalid number of dimensions")

        erosions = [binary_erosion(u, kernel).astype(np.int8)
                    for kernel in kernels]

        return np.stack(erosions, axis=0).max(0)

    def optInfSup(self, u):
        ''' IS operator
        '''
        if np.ndim(u) == 2:
            kernels = self.kernel2d
        elif np.ndim(u) == 3:
            kernels = self.kernel3d
        else:
            raise ValueError("'u' has an invalid number of dimensions")

        dilations = [binary_dilation(u, kernel).astype(np.int8)
                     for kernel in kernels]

        return np.stack(dilations, axis=0).min(0)

    def run(self, image, seed, eta=1.e-8):
        # Initialize level-set
        phi = np.zeros_like(seed, dtype=np.int16)
        phi[seed > 0], phi[seed == 0] = 2, -2

        # Define region area
        region = np.int8(phi > 0)

        # Declare curvature operator
        curvop = Operator([lambda u: self.optSupInf(self.optInfSup(u)),   # SIoIS
                           lambda u: self.optInfSup(self.optSupInf(u))])  # ISoSI

        for _ in range(self.maxIter):
            # Evaluate area strengths
            c0 = (image*(1 - region)).sum()/float((1 - region).sum() + eta)
            c1 = (image*region).sum()/float(region.sum() + eta)

            # Find edges
            gradx, grady = np.gradient(region)
            edge = abs(gradx) + abs(grady)

            # Find contour
            contour = edge*(self.lambda1*(image - c1)**2. 
                            - self.lambda2*(image - c0)**2.)

            # Update area
            region[contour < 0], region[contour > 0] = 1, 0

            # Smoothing
            for _ in range(self.numSmoothing):
                region = curvop(region)

        return region

