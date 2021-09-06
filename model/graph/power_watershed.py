import  numpy                   as  np

from    scipy                   import  sparse
from    scipy.sparse.linalg     import  spsolve 
from    skimage.measure         import  label       as  classifier
import  networkx                as  nx


def _makeGraphPair(shape):
    """ Return graph pair (vertex, edge) of given 'shape'
    """
    # Assign indice
    width, height, depth = shape
    vertex = np.arange(width*height*depth).reshape(shape)

    # List edges
    dEdge = np.vstack((vertex[..., :-1].ravel(), vertex[..., 1:].ravel()))
    hEdge = np.vstack((vertex[:, :-1].ravel(), vertex[:, 1:].ravel()))
    vEdge = np.vstack((vertex[:-1].ravel(), vertex[1:].ravel()))
    edge  = np.hstack((dEdge, hEdge, vEdge))

    return vertex, edge


def _getWeight(value, beta=1, eps=1e-10):
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
    weight = np.exp(-beta * grad2)
    
    # Normalize weigths
    weight = (weight - weight.min())/(weight.max()-weight.min())
    #weight = np.round(weight, 20)

    return weight


def _buildLaplacian(nodes, edges, weights):
    """ Return Laplacian of probabilities
    """
    # Extract dimensions
    numNode = nodes.size
    ith, jth = edges.ravel(), edges[::-1].ravel()

    # Build sparce Laplacian
    laplacian = np.hstack((weights, weights))
    laplacian = sparse.coo_matrix((laplacian, (ith, jth)),
                                  shape=(numNode, numNode))
    laplacian.setdiag(-np.ravel(laplacian.sum(axis=0)))
    laplacian = laplacian.tocsr()

    return laplacian


def _buildLinearSystem(nodes, edges, labels, weights):
    """ Return matrices of A, b in ODE system (Ax = b)
    """
    # Partitioning marked and unmarked nodes 
    index, numCls = np.arange(labels.size), int(labels.max())
    mark, unmark = labels > 0, labels == 0
    markIdx, unmarkIdx = index[mark], index[unmark]

    # Build laplacian matrix
    laplacian = _buildLaplacian(nodes, edges, weights)

    # Extract linear system
    row = laplacian[unmarkIdx, :]
    partition = row[:, unmarkIdx]
    residue   = -row[:, markIdx]

    # Make mark probabilities
    rhs  = np.eye(numCls)[labels[mark].astype(np.uint8)-1]
    rhs  = sparse.csc_matrix(rhs)
    rhs  = residue.dot(rhs).toarray()

    return partition, rhs

import  cv2
import  timeit
def powerWatershed(vx, mk):
    """
    This is an algorithm for

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
    None

    * Referenece
    ----------
    [1] None 

    * Example
    ----------
    None
    """
    
    for k in range(vx.shape[2]):
        # Regularize shape of inputs to be 4D array of which each axes denotes
        # first three are spatial (third axis is dummy dimension for 2D) and last is channels
        value = np.atleast_3d(vx[..., k].astype(np.float))[..., np.newaxis]
        label = np.atleast_3d(mk[..., k].astype(np.float))[..., np.newaxis]
        #label[label == 7] = 8       
 
        # Make graph structure
        vertex, edge = _makeGraphPair(value.shape[:3])
       
        weight = _getWeight(value)

        seed = label[..., 0].copy()
        #prob = np.eye(9)[label.ravel().astype(np.uint8)][:, 1:]
        unseen = np.zeros_like(seed)
        levels = np.unique(weight)[::-1]
        print(levels)
        for i, level in enumerate(levels[:2]):
            print(i, len(levels))
            index = weight == level
            graph = nx.Graph()
            graph.add_nodes_from(np.unique(edge[:, index]))
            graph.add_edges_from(edge[:, index].T)
            nxcc = list(nx.connected_components(graph))

            a = timeit.default_timer()
            for idx, cc in enumerate(nxcc):
                mark = np.isin(vertex, list(cc))
                jointSeed = np.unique(seed[mark == 1])
                maxJointSeed = jointSeed.max()
                adjacent  = unseen[mark == 1].max()                

                if maxJointSeed == 0:
                    unseen[mark] = max(1, adjacent)
                elif len(jointSeed) == 2 and min(jointSeed) == 0:
                    seed[mark & (seed == 0)] = maxJointSeed
                    if adjacent != 0: 
                        seed[unseen == adjacent] = maxJointSeed

                elif len(jointSeed) > 2 and min(jointSeed) == 0:
                    print(jointSeed)
                    nodes   = vertex[mark]
                    
                    edges   = np.array(list(graph.edges(vertex[mark]))).T
                    edic    = {a: b for a, b in zip(nodes, np.argsort(nodes))}
                    edges   = np.vectorize(edic.get)(edges)
                    
                    labels  = seed[mark].ravel()
                    weights = -level * np.ones_like(edges[0, :])

                    # Build linear system 
                    laplacian, residue = _buildLinearSystem(nodes, edges, labels, weights)

                    # Solve ODE and vote maxium prior
                    classes = jointSeed[1:].astype(np.uint8)
                    prob = spsolve(laplacian, residue)
                    prob = np.argmax(prob[:, classes-1], axis=1)
                    prob = np.piecewise(prob, [prob > -1], [lambda x: classes[x]])
                    seed[mark & (seed == 0)] = prob


            b = timeit.default_timer()
            print('time-elapse', b-a)
        mk[:, :, k:k+1] = seed
    
    return mk 
