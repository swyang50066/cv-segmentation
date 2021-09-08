import  numpy    as  np
from    scipy.ndimage   import  convolve


# Neumann neighborhoods
displi = [-1, 1, 0, 0]
displj = [0, 0, -1, 1]


def _funcLogGaussianEnergy(x, pos, theta):
    ''' Return logarithmic Gibb's energy
    '''
    return (np.log(np.sqrt(2*np.pi*theta[1])) 
            + .5*((x[pos] - theta[0])**2)/theta[1])


def _sgnUnion(c1, c2):
    ''' Return negativity of c1 and c2
    '''
    return -1 if c1 == c2 else 1


def _cooling(step, temp, initTemp, mode="gaussian"):
    ''' Return current temperature
    '''
    if mode == "exp":
        return 0
    elif mode == "gaussian":
        return initTemp / (1 + np.log(1+step))
    else:
        return -1


class SimulatedAnnealing(object):
    ''' Grow-cut segmentation.
    Parameters
    ----------------
    image: (H, W) ndarray
        Input image.
    seed: (H, W) ndarray
        Input seed.
    Returns
    -------
    region: (H, W) ndarray
        Segmentation region.
    '''
    def __init__(self, maxIter=100, initTemp=5.e4):
        # Paremters
        self.maxIter = maxIter
        self.initTemp = initTemp

        # Generate numpy random seed
        np.random.seed(931016)

    def getInitPrior(self, image, seed, classes):
        ''' Return initial prior of seed labels
        '''
        # Evaluate prior
        thetas = [(np.mean(image[seed == cls]), 
                   np.var(image[seed == cls])) for cls in classes]

        return thetas
   
    def getInitEnergy(self, image, seed, thetas, beta=100):
        ''' Return Gibb's energy from initial distribution
        '''
        # Neumann neighborhoods
        displi = [-1, 1, 0, 0]
        displj = [0, 0, -1, 1]

        # Padding input seed
        pad = np.pad(seed, pad_width=1, mode="edge")
        
        # Calculate Gibb's energy 
        energy = 0
        for i, j in np.argwhere(seed > 0):
            # Get current seed class
            cls = seed[i, j]

            # Singleton energy
            energy += _funcLogGaussianEnergy(image, (i, j), thetas[cls-1])

            # Doubleton energy
            for di, dj in zip(displi, displj):
                ii, jj = i + di, j + dj
                
                # Append energy
                energy += beta*_sgnUnion(cls, pad[ii+1, jj+1])

        return energy

    def getActiveNode(self, region, eps=1.e-13):
        ''' Return positions to be updated
        '''
        # Get number of neighbors
        weight = convolve(region, np.ones((3, 3)), mode="constant")

        # Set filtering conditions
        logic0 = np.uint8(np.logical_and(region > 0, weight != 9))
        logic1 = np.uint8(np.logical_and(region == 0, weight != 0))

        # Get active nodes
        nodes = np.argwhere(np.logical_or(logic0, logic1))

        return nodes

    def getDeltaEnergy(self, image, region, pos, neighbor, thetas, beta=100):
        ''' Return purturbation of energy by randon walk
        '''
        # Get current class
        cls = region[pos]

        # Evaluate singleton energies
        prevEnergy = _funcLogGaussianEnergy(image, pos, thetas[cls-1])
        nextEnergy = _funcLogGaussianEnergy(image, pos, thetas[neighbor-1])
        deltaEnergy = nextEnergy - prevEnergy

        # Evaluate doubleton energies
        # Larger 'beta' yields more homogeneous classification
        pad = np.pad(region, pad_width=1, mode='edge')
        for di, dj in zip(displi, displj):
            ii, jj = pos[0] + di, pos[1] + dj

            deltaEnergy += beta*(_sgnUnion(neighbor, pad[ii+1, jj+1]) -
                                 _sgnUnion(cls, pad[ii+1, jj+1]))

        return deltaEnergy

    def run(self, image, seed):
        # Feed region from input seed
        region = seed.copy()

        # Get list of labeling class
        classes = np.unique(seed)[1:]    # List out background

        # Get initial priors
        thetas = self.getInitPrior(image, seed, classes)

        # Evaluate initial potentials
        energy = self.getInitEnergy(image, seed, thetas)

        # Optimization
        step, temp = 0, self.initTemp
        while step < self.maxIter:
            # Get positions to be updated
            nodes = self.getActiveNode(region)

            # Perturbe static distribution
            for i, j in nodes:               
                # Randomly choice a random class except for previous one
                samples = np.setdiff1d(classes, region[i, j])
                neighbor = np.random.choice(samples, 1)[0]

                # Get energy change
                deltaEnergy = self.getDeltaEnergy(image, region, (i, j), neighbor, thetas)

                # Metropolice criteria
                zeta = np.random.rand()
                if (deltaEnergy <= 0 or
                    (deltaEnergy > 0 and 
                     np.exp(-deltaEnergy / temp) < zeta)):
                    # Update region
                    region[i, j] = neighbor

                    # Update total energy
                    energy += deltaEnergy

            # Cool down temperature
            temp = _cooling(step, temp, self.initTemp)
            
            # Update step
            step += 1

        return region

