import  numpy       as  np
import  maxflow

    
def funcGradientIndicator(x, alpha=2.):
    ''' Return gradient indicator
    '''
    return 1/(1 + x**alpha)


class GraphCut(object):
    ''' Graph-cut algorith

    Parameters
    ----------------
    image: (H, W) ndarray
        Input image
    seed: (H, W) ndarray
        seed map, background are assigned with 1 and foreground is 2

    Returns
    ----------------
    region: (H, W) ndarray
        Segmentation region
    '''
    def __init__(self, MAXIMUM=100000):
        # Parameters
        self.MAXIMUM = MAXIMUM

    def __setEdgeWeight(self, image, graph):
        """ Set edge weight into the graph
        """
        # Assign indice
        vertex = np.arange(image.size).reshape(image.shape)

        # List edges
        hEdges = np.vstack((vertex[:-1].ravel(), 
                            vertex[1:].ravel()))
        wEdges = np.vstack((vertex[:, :-1].ravel(), 
                            vertex[:, 1:].ravel()))
        edges = np.hstack((hEdges, wEdges))

        # Evaluate gradient of features
        grads = np.hstack([np.diff(image, axis=ax).ravel() 
                           for ax in [0, 1]])
        grads = funcGradientIndicator(grads)

        # Update edge weight
        for (i, j), grad in zip(edges.T, grads):
            graph.add_edge(i, j, grad, grad)

        return graph

    def getInitLevelSet(self, image, seed):
        ''' Get initial level-set function 
        '''
        # Declare level-set
        phi = .5*np.ones_like(image)

        # Get seed positions
        background, foreground = (np.argwhere(seed == 1),
                                  np.argwhere(seed == 2))

        # Assign seed to level-set
        phi[tuple(background)], phi[tuple(foreground)] = 0., 1.

        return phi

    def populateGraph(self, image, phi):
        ''' Populate graph components
        '''
        # Get input dimensions
        height, width = image.shape
        size = height*width

        # Initialize graph
        graph = maxflow.Graph[float](size, size) 
        graph.add_nodes(size)

        # Update unary term
        for index, pos in enumerate(np.ndindex(height, width)):
            if phi[tuple(pos)] == .0:    # Background
                graph.add_tedge(index, self.MAXIMUM, 0)
            elif phi[tuple(pos)] == 1.:    # Foreground
                graph.add_tedge(index, 0, self.MAXIMUM)
            else: # Transition
                graph.add_tedge(index, 0, 0)

        # Update edge weights
        graph = self.__setEdgeWeight(image, graph)
        
        return graph

    def cutGraph(self, image, graph):
        ''' Cut graph with maxflow search
        '''
        # Get input shape
        height, width = image.shape

        # Cut graph
        graph.maxflow()

        # Get label
        label = np.zeros_like(image)
        for index in range(height*width):
            if graph.get_segment(index) == 1:
                label[index//width, index%width] = 1
        
        return label

    def run(self, image, seed):
        # Get initial level-set function
        phi = self.getInitLevelSet(image, seed)

        # Populate node-edge connections
        graph = self.populateGraph(image, phi)

        # Cut graph
        label = self.cutGraph(image, graph)

        return label


