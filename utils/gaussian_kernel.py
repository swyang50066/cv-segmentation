import  numpy   as  np
from    scipy.ndimage       import  distance_transform_edt


def getGaussianKernel(sigma, size, ndim=3):
    ''' Build Gaussian kernel
        Parameters
        ----------------
        sigma: integer
            discreted variance of Gaussian kernel
        (optional) size: (2,) or (3,) list, tuple or ndarray
            size of box domain (H, W) or (H, W, D)
        (optional) ndim:
            specified input dimension
        Returns
        ----------------
        kernel: (H, W) or (H, W, D) ndarray
            Discreted Gaussian kernel
    '''
    # Open kernal domain
    if ndim == 2:
        kernel = np.zeros(size)
        kernel[size[0]//2, size[1]//2] = 1
    else:
        kernel = np.zeros(size)    
        kernel[size[0]//2, size[1]//2, size[2]//2] = 1

    # Build discretized gaussian kernel
    kernel = distance_transform_edt(1 - kernel)        
    kernel = (np.exp(-.5*kernel**2./sigma**2.)
              / np.power(2*np.pi, ndim/2.)
              / sigma**ndim)
    

    return kernel
