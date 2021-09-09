import  numpy       as  np

from    scipy                   import  sparse
from    scipy.sparse.linalg     import  spsolve


class RandomWalkerPriorModel(object):
    ''' Markov random field based image segmentation algorithm by Grady (2005, 2006)
        Grady first proposed this algorithm and improved it using prior model in next study.
        The function has referred to the module of 'scikit-image/segmentation/random_walker',
        but discriminately includes prior model (You are able to apply original 'random_walker' 
        by setting 'gamma' parameter to be 0).

    Parameters
    ----------
    image: (H, W) or (H, W, C) ndarray
        input image (the third dimension is recognized as image channel)
    seed: (H, W) ndarray
        seed map including seed points of label

    Returns
    ----------
    seed: (H, W) ndarray
        Segmentation regions
    '''
    def __init__(self, numCls=2, beta=1e3, sigma=1.e2, gamma=1.e-2):
        # Set number of label classes
        self.numCls = numCls

        # Model parameters
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def __getGraphEdge(self, shape):
        ''' Return list of edge pairs
        '''
        # Assign indice
        height, width, depth = shape
        vertex = np.arange(height*width*depth).reshape(shape)        

        # List edges
        dEdges = np.vstack((vertex[..., :-1].ravel(), 
                            vertex[..., 1:].ravel()))
        wEdges = np.vstack((vertex[:, :-1].ravel(), 
                            vertex[:, 1:].ravel()))
        hEdges = np.vstack((vertex[:-1].ravel(), 
                            vertex[1:].ravel()))
        edges = np.hstack((dEdges, wEdges, hEdges))

        return edges

    def __getEdgeWeight(self, value, eps=1.e-10):
        ''' Return listo of edge weights
        '''
        # Evaluate gradient of features
        grads = np.hstack([np.diff(value[..., 0], axis=ax).ravel()
                           for ax in [2, 1, 0] if value.shape[ax] > 1])
        
        if value.shape[-1] > 1:    # Multi-channels
            for ch in range(value.shape[-1]):
                grads += np.hstack([np.diff(value[..., ch+1], axis=ax).ravel()
                                    for ax in [2, 1, 0] if value.shape[ax] > 1], 
                                   axis=0)
    
        # Calculate scaling parameters
        rho = np.max(grad**2.)
        beta = self.beta / np.sqrt(value.shape[-1])
        
        # Evaluate weights
        weights = -(np.exp(-self.beta*grads*grads/rho) + eps)

        return weights

    def buildLaplacian(self, value):
        ''' Return Laplacian of probabilities
        '''
        # Get edge pairs and corresponding weight
        edges = self.__getGraphEdge(value.shape[:3])
        weights = self.__getEdgeWeight(value)

        # Extract dimensions
        numNode = edges.shape[1]
        ith, jth = edges.ravel(), edges[::-1].ravel()
    
        # Build sparce Laplacian
        laplacian = np.hstack((weights, weights))
        laplacian = sparse.coo_matrix((laplacian, (ith, jth)),
                                      shape=(numNode, numNode))
        laplacian.setdiag(-np.ravel(laplacian.sum(axis=0)))
        laplacian = laplacian.tocsr()
    
        return laplacian

    def buildPriori(self, value, label):
        ''' Return priori matrices
        '''
        # Reformat inputs
        intensities = value[..., 0].ravel()
        onehot = np.uint8(label[label > 0]) - 1
        
        # Construct matrix of dim(mark x classes)
        matMark = np.eye(self.numCls)[onehot]
        
        # Construct matrix of dim(256 x mark)
        markValue = np.uint8(intensities[label > 0])
        matValue = np.vstack([markValue - i for i in np.arange(256)])
        matValue = np.exp(-matValue**2./self.sigma)

        # Calcuate matrix of dim(256 x classes)
        matProb = np.matmul(matValue, matMark)
        norm = np.sum(matProb, axis=0)
        norm[norm == 0] = np.inf
        matProb = matProb/norm

        # Construct intensity one-hot
        unmarkValue = np.uint8(intensities[label == 0])
        lambdas = np.eye(256)[unmarkValue]
        lambdas = np.matmul(lambdas, matProb)

        # Build priori system
        numNode = lambdas.shape[0]
        matLambda = sparse.coo_matrix((numNode, numNode))
        matLambda.setdiag(np.sum(lambdas, axis=1))
        matLambda = self.gamma*matLambda.tocsr()

        return matLambda, lambdas

    def buildLinearSystem(self, value, label):
        ''' Return matrices A, b in ODE system (Ax = b)
        '''
        # Partitioning marked and unmarked nodes 
        index = np.arange(label.size)
        mark, unmark = label > 0, label == 0
        markIndex, unmarkIndex = index[mark], index[unmark]

        # Build laplacian matrix
        laplacian = self.buildLaplacian(value)
    
        # Get priori matrices
        matLambda, lambdas = self.buildPriori(value, label)

        # Extract linear system
        row = laplacian[unmarkIndex, :]
        partition = row[:, unmarkIndex]
        residue = -row[:, markIndex]
   
        # Make mark probabilities
        rhs = np.eye(self.numCls)[label[mark].astype(np.uint8)-1]
        rhs = sparse.csc_matrix(rhs)
        rhs = residue.dot(rhs).toarray()

        # Add prior model
        rhs = rhs + lambdas
        partition = partition + matLambda
    
        return partition, rhs

    def run(self, image, seed):
        # Regularize shape of inputs to be 4D array of which each axes denotes
        # first three are spatial (third axis is dummy dimension for 2D) and last is channels
        value = np.atleast_3d(image.astype(np.float))[..., np.newaxis]
        label = np.atleast_3d(seed.astype(np.float))[..., np.newaxis].ravel()

        # Build linear system 
        laplacian, residue = self.buildLinearSystem(value, label)

        # Define classes
        classes = np.arange(1, self.numCls+1)

        # Solve ODE and vote maxium prior
        prob = spsolve(laplacian, residue)
        prob = np.argmax(prob[:, classes-1], axis=1)
        prob = np.piecewise(prob, [prob > -1], [lambda x: classes[x]])

        # Update seed
        seed[seed == 0] = prob

        return seed


