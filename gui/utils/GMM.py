import  cv2

import  numpy                   as  np
from    sklearn.cluster         import  KMeans


def gaussianMixtureModel(
                    stack,
                    LOWER=-400, UPPER=1400, WING=50, 
                    numComp=15, 
                    maxIters=100, 
                    tol=1.e-4):
    ''' Gaussian-mixture model which approximates probabilistic distribution
        of feature map

        Parameters
        ----------------
        feature: (N,) list, tuple or ndarray 
            feature map of input data

        Returns
        ----------------
        prior: (M,) list, tuple or ndarray
            prior probabilities of Gaussian components
        theta: (M, 2) list, tuple or ndarray
            paramter space of Gaussian components 

        functions
        ----------------
        _funcGaussian: (x: float, mu: float, sigma: float) -> float 
            Calculate gaussian probability with parameter 'x'

        Note
        ----------------
        The final approximated distribution q(x) comes from
            q(x) = Sum_i (prior_i * gaussian(x|theta))
    '''
    def _funcGaussian(x, mu, sigma):
        ''' Return Gaussian probability at given parameter 'x'        
        '''
        return (np.exp(-.5*(x - mu)**2./sigma**2.) / 
                np.sqrt(2.*np.pi*sigma**2.))

    # Get feature map
    stack = cv2.GaussianBlur(stack, (3, 3), 1)
    feature = stack[stack.shape[0]//2].ravel()
    feature = feature[(feature > LOWER) & (feature < UPPER)]

    # Set initial labels
    label = KMeans(
                n_clusters=numComp, 
                n_init=10, 
                random_state=931016
                ).fit(feature.reshape(-1, 1)).labels_

    # Declare probablities
    marginal = np.zeros_like(feature)
    theta, prior, lkh, post = {}, {}, {}, {}
    
    # Initialize probabilities 
    for n in range(numComp):
        # Model parameters
        mu, sigma = (np.mean(feature[label == n]),
                     np.std(feature[label == n]))
        theta[n] = (mu, sigma)
        
        # Probabilities
        prior[n] = len(label[label == n])/len(label)
        lkh[n] = _funcGaussian(feature, theta[n][0], theta[n][1])        

        # Marginal
        marginal += lkh[n]*prior[n]

    for _ in range(maxIters):
        # Posterior
        post = {n:lkh[n]*prior[n]/marginal for n in range(numComp)}
        
        # Maximization
        prev = marginal.copy()
        marginal = np.zeros_like(feature)
        for n in range(numComp):
            # Update parameter
            mu = np.sum(post[n]*feature)/np.sum(post[n])
            sigma = np.sum(post[n]*(feature - theta[n][0])**2.)
            sigma = np.sqrt(sigma/np.sum(post[n]))
            theta[n] = (mu, sigma)

            # Update probabilities
            prior[n] = np.sum(post[n])/len(feature)
            lkh[n] = _funcGaussian(feature, mu, sigma)

            # Update marginal
            marginal += lkh[n]*prior[n]

        # Check convergency
        mse = np.sqrt(np.sum((prev - marginal)**2.))
        if mse < tol: break

    # Calculate gaussian probabilities  
    probs = [prior[n] * 
             _funcGaussian(theta[n][0], theta[n][0], theta[n][1])
             for n in range(numComp)]
    maxProb = np.array(probs).max()    

    # Find ROI HU range
    order = np.argsort([theta[n][0] for n in range(numComp)])
    for i in range(numComp):
        # Validate probability
        if theta[order[i+1]][0] < -4.e2: 
            continue
        if theta[order[i+1]][1] > 5.e1: 
            continue
        if probs[order[i+1]] < .1*maxProb: 
            continue

        minHU = int(theta[order[i+1]][0] - 
                    3*theta[order[i+1]][1] - WING)
        break
    for i in range(numComp):
        # Validate probability
        if theta[order[numComp-i-1]][0] > 1.3e3: 
            continue
        if theta[order[numComp-i-1]][1] > 1.e2: 
            continue
        if probs[order[numComp-i-1]] < .1*maxProb: 
            continue

        maxHU = int(theta[order[numComp-i-1]][0] + 
                    3*theta[order[numComp-i-1]][1] + WING)
        break 
    
    # Convert HU to gray levels
    voxel = stack.copy()
    voxel[voxel > maxHU], voxel[voxel < minHU] = maxHU, minHU
    voxel = 255. * (voxel - minHU) / (maxHU - minHU)
    
    return voxel
