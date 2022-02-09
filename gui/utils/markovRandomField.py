import  numpy                   as  np

from    scipy                   import  sparse
from    scipy.sparse.linalg     import  spsolve


def _makeGraphEdge(shape):
    """ Return edge pair of given 'shape'
    """
    # Assign indice
    width, height, depth = shape
    vertex = np.arange(width*height*depth).reshape(shape)

    # List edges
    dEdge = np.vstack((vertex[..., :-1].ravel(), vertex[..., 1:].ravel()))
    hEdge = np.vstack((vertex[:, :-1].ravel(), vertex[:, 1:].ravel()))
    vEdge = np.vstack((vertex[:-1].ravel(), vertex[1:].ravel()))
    edge = np.hstack((dEdge, hEdge, vEdge))

    return edge


def _getPrior(value, label, numCls, sigma, gamma):
    """ Return priori matrices
    """    
    # Construct matrix of dim(mark x classes)
    mark    = label[label > 0]
    matMark = np.eye(numCls)[mark.astype(np.uint8)-1]

    # Construct matrix of dim(256 x mark)
    val     = value[..., 0].ravel()
    markVal = val[label > 0]    
    matVal  = np.vstack([markVal-i for i in np.arange(256)/255.])
    matVal  = np.exp(-matVal**2./sigma)

    # Calcuate matrix of dim(256 x classes)
    matProb  = np.matmul(matVal, matMark)
    
    marginal = np.sum(matProb, axis=0)
    marginal[marginal == 0] = np.inf
    matProb  = matProb/marginal

    # Construct intensity one-hot
    unmarkVal = val[label == 0]
    onehot    = np.eye(256)[255*(unmarkVal).astype(np.uint8)]
    lambdas   = np.matmul(onehot, matProb)

    numNode   = lambdas.shape[0]
    matLambda = sparse.coo_matrix((numNode, numNode))
    matLambda.setdiag(np.sum(lambdas, axis=1))
    matLambda = gamma*matLambda.tocsr()

    return matLambda, lambdas


def _getWeight(value, beta, eps=1.e-10):
    """ Return weight values
    """
    # Evaluate gradient of features
    grad = np.hstack([np.diff(value[..., 0], axis=ax).ravel()
                      for ax in [2, 1, 0] if value.shape[ax] > 1])
    for ch in range(value.shape[-1]):
        if value.shape[-1] == 1: break

        grad += np.hstack([np.diff(value[..., ch+1], axis=ax).ravel()
                           for ax in [2, 1, 0] if value.shape[ax] > 1], axis=0)
    grad2 = grad * grad
    
    # Evaluate weights
    rho  = grad2.max()
    #beta = beta / np.sqrt(value.shape[-1])
    #weight = -(np.exp(-beta * grad2 / rho) + eps)
    weight = -(np.exp(-beta*grad2/(10*np.std(value))) + eps)

    return weight


def _buildLaplacian(value, beta):
    """ Return Laplacian of probabilities
    """
    # Calculate edge pairs and corresponding weight
    edge   = _makeGraphEdge(value.shape[:3])
    weight = _getWeight(value, beta)

    # Extract dimensions
    numNode = edge.shape[1]
    ith, jth = edge.ravel(), edge[::-1].ravel()
    
    # Build sparce Laplacian
    laplacian = np.hstack((weight, weight))
    laplacian = sparse.coo_matrix((laplacian, (ith, jth)),
                                  shape=(numNode, numNode))
    laplacian.setdiag(-np.ravel(laplacian.sum(axis=0)))
    laplacian = laplacian.tocsr()
    
    return laplacian


def _buildLinearSystem(value, label, beta, sigma, gamma):
    """ Return matrices of A, b in ODE system (Ax = b)
    """
    # Partitioning marked and unmarked nodes 
    index, numCls = np.arange(label.size), int(label.max()) 
    mark, unmark = label > 0, label == 0
    markIdx, unmarkIdx = index[mark], index[unmark]

    # Build laplacian matrix
    laplacian = _buildLaplacian(value, beta)
    
    # Get priori matrices
    matLambda, lambdas = _getPrior(value, label, numCls, sigma, gamma)

    # Extract linear system
    row = laplacian[unmarkIdx, :]
    partition = row[:, unmarkIdx]
    residue   = -row[:, markIdx]
   
    # Make mark probabilities
    rhs  = np.eye(numCls)[label[mark].astype(np.uint8)-1]
    rhs  = sparse.csc_matrix(rhs)
    rhs  = residue.dot(rhs).toarray()

    # Add prior model
    rhs = rhs + lambdas
    partition = partition + matLambda
    
    return partition, rhs


def markovRandomField(im, mk,
                      classes=np.arange(1, 3),
                      beta=1e3, sigma=1.e2, gamma=1.e-2):
    """
    This is an algorithm for registration of volumentric cardiac CT.
        Grady (2005, 2006) first proposed this numerical scheme, and
        Main algorithm has referred to 'random_walker' in open-library of 'skimage'.
        The algorithm conducts 2D/3D registration, considering bayes prior based on marks.

    * Arguments
    ----------
    vx: ndarray, int or float
        input data 
    mk: ndarray, int
        label data, ranging '0:background' to the number of classes.
        The following algorithm recognize zero markers as training nodes.

    * Returns
    ----------
    mk: ndarray, int
        Updated label data by this algorithm.

    * Note
    ----------
    We named shape parameters as
        In 2-D, (xlen, ylen) = (width, height) 
        In 3-D, (xlen, ylen, zlen) = (width, height, depth)   
        Others, None

    * Referenece
    ----------
    [1] Leo Grady, Random walks for image segmentation, 
        IEEE Trans Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.
        :DOI:`10.1109/TPAMI.2006.233`

    * Example
    ----------
    None
    """
    # Regularize shape of inputs to be 4D array of which each axes denotes
    # first three are spatial (third axis is dummy dimension for 2D) and last is channels
    value = np.atleast_3d(im.astype(np.float))[..., np.newaxis] #/ 255.
    label = np.atleast_3d(mk.astype(np.float))[..., np.newaxis].ravel()
   
    # Build linear system 
    laplacian, residue = _buildLinearSystem(value, label, beta, sigma, gamma)

    # Solve ODE and vote maxium prior
    prob = spsolve(laplacian, residue)
    prob = np.argmax(prob[:, classes-1], axis=1)
    prob = np.piecewise(prob, [prob > -1], [lambda x: classes[x]])
 
    # Update label
    mk[mk == 0] = prob

    overlay = np.zeros(mk.shape + (3,), dtype=np.uint8)

    pos = np.argwhere(mk == 1)
    for i, j in pos:
        overlay[i, j] = (255, 0, 255)

    return mk, overlay


