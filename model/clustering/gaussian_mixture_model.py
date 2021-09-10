from    collections     import  defaultdict

import  numpy       as  np
from    sklearn.cluster         import  KMeans


def _funcGaussian(x, mu, sigma):
    ''' Return Gaussian probability at given parameter 'x'
    '''
    return (np.exp(-.5*(x - mu)**2./sigma**2.) /
            np.sqrt(2.*np.pi*sigma**2.))


class GMM(object):
    ''' Gaussian mixture model based clustering of image domain

    Parameters
    ----------------
    image: (H, W) ndarray
        input image.

    Returns
    ----------------
    region: (H, W) ndarray
        segmentation region
    prior: (M,) list, tuple or ndarray
        priori of Gaussian basis
    theta: (M, 2) list, tuple or ndarray
        paramter of Gaussian basis

    Note
    ----------------
    The distribution of Gaussian mixture q(x) is to be
        q(x) = Sum_i (prior_i * gaussian(x| theta))
    '''
    def __init__(self, numCls=4, vmin=16, vmax=255, maxIter=100, tol=1.e-4):
        # Model parameters
        self.numCls = numCls    # number of clustering group 
      
        self.vmin = vmin
        self.vmax = vmax
        self.maxIter = maxIter
        self.tol = tol
   
    def run(self, image): 
        # Vecorize input image
        vec = image[(image > self.vmin) & (image < self.vmax)].astype(np.float32).ravel()
        size = np.float32(len(vec))

        # Initialize clusters via k-means clustering
        label = KMeans(n_clusters=self.numCls, 
                       n_init=10, 
                       random_state=931016    # its my birth day :)
                      ).fit(vec.reshape(-1, 1)).labels_
    
        # Declare probablities
        theta = defaultdict(tuple)
        prior = np.zeros(self.numCls, dtype=np.float32)
        likelihood = np.zeros((self.numCls, len(vec)), dtype=np.float32)
        posterior = np.zeros((self.numCls, len(vec)), dtype=np.float32)
        marginal = np.zeros(len(vec), dtype=np.float32)

        # Initialize probabilities 
        for cls in range(self.numCls):
            # Set inital parameter
            mu, sigma = (np.mean(vec[label == cls]),
                         np.std(vec[label == cls]))
            theta[cls] = (mu, sigma)

            # Set initial prior
            prior[cls] = len(label[label == cls])/len(label)
            
            # Get initial likelihood
            likelihood[cls] = _funcGaussian(vec, mu, sigma)        

            # Add marginal
            marginal += likelihood[cls]*prior[cls]

        # EM optimization
        for _ in range(self.maxIter):
            # Posterior
            posterior = likelihood*prior[..., np.newaxis] / marginal
            
            # Maximization
            currMgn = marginal.copy()
            nextMgn = np.zeros(len(vec), dtype=np.float32)
            for cls in range(self.numCls):
                # Update parameter
                mu = np.sum(posterior[cls]*vec)/np.sum(posterior[cls])
                sigma = np.sum(posterior[cls]*(vec - theta[cls][0])**2.)
                sigma = np.sqrt(sigma/np.sum(posterior[cls]))
                theta[cls] = (mu, sigma)

                # Update probabilities
                prior[cls] = np.sum(posterior[cls])/size
                likelihood[cls] = _funcGaussian(vec, mu, sigma)

                # Update marginal
                nextMgn += likelihood[cls]*prior[cls]

            # Verify convergence
            mse = np.sqrt(np.sum((currMgn - nextMgn)**2.))
            if mse < self.tol: 
                break
            else:
                marginal = nextMgn

        # Cluster image domain
        mapping = np.zeros((self.numCls, 256))
        for n in range(self.numCls):
            mapping[n] = _funcGaussian(np.arange(256), theta[n][0], theta[n][1])
        mapping = np.argmax(mapping, axis=0)

        # Mapping image to label
        region = mapping[image.ravel()].reshape(image.shape)

        return region, prior, theta
