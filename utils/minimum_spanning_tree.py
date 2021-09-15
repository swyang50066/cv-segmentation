import  heapq
from    collections     import  deque, defaultdict

import  numpy   as  np


class Node(object):
    ''' Graph node object
    '''
    def __init__(self, parent=None, 
                       level=None, 
                       index=None, 
                       weight=None):
        # Instances
        self.parent = parent        # Parent node object
        self.children = list()      # List of child node objects
        self.level = level          # Depth of tree (root: 0)
        self.index = index          # unique node index labeled in graph
        self.weight = weight        # edge-weight with parent node 

    def addChild(self, child):
        ''' Append children
        '''
        self.children.append(child)


class GraphUndirectedWeighted(object):
    ''' Undirected/Weighted Graph Structure Object
    '''
    def __init__(self, x):
        # Get input shape
        height, width = x.shape

        # Get node list and build vertex map with unique label
        self.nodes = np.arange(height*width)
        self.vertex = self.nodes.reshape((height, width))
        
        # Get edge pairs
        self.edges = self.__getGraphEdge(x)
        
        # Get edge weights
        self.weights = self.__getEdgeWeight(x)

    def __call__(self):
        ''' Return graph structure
        '''
        return self.buildGraph(self.edges, self.weights)

    def __getGraphEdge(self, x):
        ''' Return list of edge pairs
        '''
        # List axis-wise edge pairs
        top2bottom = np.vstack((self.vertex[:-1].ravel(),
                                self.vertex[1:].ravel()))
        left2right = np.vstack((self.vertex[:, :-1].ravel(),
                                self.vertex[:, 1:].ravel()))
        #### Add corn2corn

        # Concatenate edge pairs (sorted in the row-wise order)
        edges = np.hstack((top2bottom, left2right))

        return np.array(edges).T

    def __getEdgeWeight(self, u, beta=1., eps=1.e-10):
        ''' Return list of edge weights
        '''
        # Compute intensity difference 
        grads = np.hstack([np.abs(np.diff(u, axis=axis).ravel())
                           for axis in range(2)])
        
        # Normalize intensity difference
        grads = (grads - grads.min())/(grads.max() - grads.min())

        # Evaluate weights
        weights = -np.exp(-beta*grads**2.) + eps

        return np.array(weights).T

    def buildGraph(self, edges, weights):
        ''' Build graph structure
        '''
        # Build graph containing connectivity between nodes
        graph = defaultdict(list)
        for weight, (n1, n2) in zip(weights, edges):
            if (n2, weight) not in graph[n1]:
                graph[n1].append((weight, n1, n2))
            if (n1, weight) not in graph[n2]:
                graph[n2].append((weight, n2, n1))

        # Sort value to be arranged in the ascending order with weight
        for node, neighbors in graph.items():
            # Heap sorting
            heapq.heapify(neighbors)
            
            # Rearrange neighbors
            graph[node] = neighbors

        return graph


def MinimumSpanningTree(graph, initIndex=None):
    ''' Build mininum spanning tree using Kruskal/Prim algorithm
    '''
    # Set valid root node
    if not initIndex or initIndex >= len(graph):
        initIndex = np.random.randint(len(graph))
    
    # Initialize root node
    root = Node(parent=None,
                level=0,
                index=initIndex,
                weight=None)

    # Declare 'visited' indicator
    visited = [0] * len(graph)
    visited[initIndex] = 1

    # Define container of leaf nodes
    leaves = list()

    # Build mininum spanning tree
    adjacency = [edge + (root,) for edge in graph[initIndex]]
    while adjacency:
        # Select a minimum weighted edge from query
        weight, n1, n2, parent = heapq.heappop(adjacency)

        # Find next edge query
        for edge in graph[n2]:
            # Skip visited node 
            if visited[edge[2]]:
                continue
            else:
                visited[edge[2]] = parent.level+1
                
            # Define branch node
            #   root -> branch1 -> branch2 -> ... -> leaf
            branch = Node(parent=parent,
                          level=parent.level+1,
                          index=n2,
                          weight=weight)

            # Append child node to parent node
            parent.addChild(branch)

            # Push heap node
            heapq.heappush(adjacency, edge + (branch,))
        
        # Check if the current node is leaf node
        if len(parent.children) and parent.level == 0:
            visited[n2] = 2    # mark first branch nodes
        elif not parent.children:
            # Define leaf node
            leaf = Node(parent=parent,
                        level=parent.level+1,
                        index=n2,
                        weight=weight)

            # Append leaf node
            leaves.append(leaf)

    return leaves
