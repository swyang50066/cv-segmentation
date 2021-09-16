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
        root.label = label

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

            # Downward
            branches = stream


class UnionFind(object):
    ''' This is a set operator that is kind of disjoint set
    '''
    def union(self, x, y):
        ''' Unify two branches
        '''
        # Find mergee and merger
        order = str()
        if self.find(y) < self.find(x):
            order = "forward"
            xx, yy = x, y
        else:
            order = "backward"
            xx, yy = y, x

        # Merge two branches
        yy.addChild(xx)

        xx.setLabel(yy.label)

        if not xx.parent:
            xx.parent = yy
            if order == "forward":
                return xx, yy
            else:
                return yy, xx

        # List upstream nodes from merger
        geneaology = deque([xx])
        a = xx.parent
        while a.parent:
            geneaology.appendleft(a)
            a = a.parent

        abc = [g.index for g in geneaology]
        

        # Reverse stream
        cba = list()
        while geneaology:
            g = geneaology.popleft()
            if len(geneaology) > 0:
                g.parent = geneaology[0]
            else:
                g.parent = yy
        
        if order == "forward":
            return xx, yy
        else:
            return yy, xx

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
        
        print("DDVC \t\t\t", nodes[n1].index, nodes[n2].index, 
                             nodes[n1].label, nodes[n2].label)


    labels = [node.label for node in nodes]
    indice = [node.index for node in nodes]
    roots = [node.upstream().index for node in nodes]
    print("labels", labels)
    print("indice", indice)
    print("roots", roots)

    for i, node in enumerate(nodes):
        branch = list()
        while node.parent:
            branch.append(node.parent.index)
            node = node.parent

        print("\t >>> [%d]"%i, branch)

    # Find leaf nodes
    root = nodes[1].upstream()
    leaves = root.downstream()

    return leaves


import cv2
import matplotlib.pyplot as plt

def drawMSF(x, leaves):
    # Get input shape
    height, width = x.shape
    
    # Gray-scale
    gray = 255*(x - x.min())/(x.max() - x.min())

    # Define plot components
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    # Plot MSF
    axes.imshow(gray, cmap="Greys")
    for leaf in leaves:
        # Mark leaf node
        xa, ya = leaf.index//width, leaf.index%width
        axes.scatter(ya, xa, color="Blue", s=20, marker="s")

        while leaf.parent:
            # Mark branch node
            a, b = leaf.parent.index, leaf.index
        
            xa, ya = a//width, a%width
            xb, yb = b//width, b%width
        
            axes.scatter(ya, xa, color="Red", s=20)
            axes.arrow(ya, xa, yb-ya, xb-xa, color="lime", head_width=.05, head_length=.1)

            # Upstream
            leaf = leaf.parent

            # Mark root node
            if leaf.parent == None:
                axes.scatter(ya, xa, color="Green", s=20, marker="D")

    plt.show()
    plt.clf()

# Load sample image
imsample = np.array([[0, 0, 1, 0, 0],
                     [0, 1, 4, 1, 0],
                     [1, 4, 6, 4, 1],
                     [0, 1, 4, 1, 0],
                     [0, 0, 1, 0, 0]])

# Build undirected/weighted image graph
graph = GraphUndirectedWeighted(imsample)(output="graph")
conns = GraphUndirectedWeighted(imsample)(output="connection")

# Find minimum spanning tree with Kruskal's algorithm
leaves = KruskalMST(graph, conns)

# Draw MSF
drawMSF(imsample, leaves)
