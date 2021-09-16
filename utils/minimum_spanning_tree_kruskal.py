import    copy        

import  heapq
from    collections     import  deque, defaultdict

import  numpy   as  np


class Node(object):
    ''' Tree node linkage
    '''
    def __init__(self, parent=None, 
                       level=None, 
                       index=None, 
                       weight=None,
                       label=None):
        # Instances
        self.parent = parent        # Parent node object
        self.children = list()      # List of child node objects
        

        self.level = level          # Depth of tree (root: 0)
        self.index = index          # unique node index labeled in tree
        self.weight = weight        # edge-weight with parent node 
        self.label = label          # unique label of connected nodes

    def upstream(self):
        ''' Return root node
        '''
        if self.parent == None:
            return self
        else:
            root = self.parent
            while root.parent:
                root = root.parent
            return root

    def downstream(self):
        ''' Return leaf nodes of current branch (not the whole tree)
        '''
        branches, leaves = self.children, list()
        while branches:
            stream = list()

            # Check if the stream reaches leaf node
            for branch in branches:
                if not branch.children:
                    leaves.append(branch)
                else:
                    stream += branch.children

            # Downward
            branches = stream

        return leaves

    def addChild(self, child):
        ''' Append children
        '''
        self.children.append(child)

    def setLabel(self, label):
        ''' Set label on connected components
        '''
        # Change current label
        self.label = label

        # Get root node
        root = self.upstream()

        # Propagate new label til leaf node
        branches = root.children
        while branches:
            stream = list()

            # Find next branches
            for branch in branches:
                # Set label
                branch.label = label

                # Append stream
                stream += branch.children

                print('\t >>>> ', len(stream))

            # Downward
            branches = stream


class UnionFind(object):
    ''' This is a set operator that is kind of disjoint set
    '''
    def union(self, x, y):
        ''' Unify two branches
        '''
        # Find mergee and merger
        if self.find(y) < self.find(x):
            xx, yy = x, y
        else:
            xx, yy = y, x

        # Merge two branches
        yy.addChild(xx)

        xx.setLabel(yy.label)

        # List upstream nodes from merger
        geneaology = list()
        while xx.parent:
            geneaology.append(xx)
        geneaology = geneaology[::-1] # ancestor to descendant

        # Reverse stream
        for g in range(len(geneaology)):
            if g+1 < len(geneaology):
                geneaology[g].parent = geneaology[g+1]
            else:
                geneaology[g].parent = yy

        return xx, yy

    def find(self, x):
        ''' Return level of node
        '''
        if x.level:
            return x.level
        else:
            level = int()
            while x.parent:
                level += 1
                x = x.parent

            # Assign level
            x.level = level

            return level


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

    def __call__(self, output="graph"):
        ''' Return graph structure
        '''
        if output == "graph":
            return self.buildGraph(self.edges, self.weights)
        if output == "connection":
            return self.getSortedWeightedConn(self.edges, self.weights)

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

        return np.array(edges, dtype=np.int32).T

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

    def getSortedWeightedConn(self, edges, weights):
        conns = np.hstack([weights[..., np.newaxis], edges])
        conns = sorted(conns, key=lambda x: x[0])

        return conns


def KruskalMST(graph, conns):
    ''' Build mininum spanning tree using Kruskal's algorithm
    '''
    # Initialize node list
    nodes = [Node(parent=None, label=i, index=i) 
             for i in range(len(graph))]

    # UnionFind Operation; UFO
    ufo = UnionFind()

    # Build mininum spanning tree
    query = deque(conns)
    while query:
        # Pop minimum weighted connection from query
        weight, n1, n2 = query.popleft()
        n1, n2 = int(n1), int(n2)

        # Unify nodes
        if nodes[n1].label != nodes[n2].label:
            nodes[n1], nodes[n2] = ufo.union(nodes[n1], nodes[n2])
           
        labels = [node.label for node in nodes]
        print(len(np.unique(labels)))

    print("Hello")

    # Find leaf nodes
    #root = nodes[0].upstream()
    leaves = list()#root.downstream()

    return leaves
