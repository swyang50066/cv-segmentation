from    collections     import  deque

import  numpy       as  np
from    scipy.ndimage       import  convolve


## Sobel kernels
# Here, we are going to use image coordinates matching
# array coordinates like (x:height, y:width)
SOBEL_X = np.array([[ 1.,  2.,  1.],
                    [ 0.,  0.,  0.],
                    [-1., -2., -1.]])
SOBEL_Y = np.array([[ 1.,  0., -1.],
                    [ 2.,  0., -2.],
                    [ 1.,  0., -1.]])
    
  
# Neighbor Kernel  
FANCE = np.array([[True, True,  True], 
                  [True, False, True], 
                  [True, True,  True]])


def _getEdgeMap(u, sigma=2):
    ''' Return image edge map by  using Sobel operator
    '''
    # Extract image edges with Sobel filter
    fx = convolve(u, SOBEL_X, mode="nearest")
    fy = convolve(u, SOBEL_Y, mode="nearest")

    return np.sqrt(.5*(fx**2. +  fy**2.))


def toboggan(x):
    ''' Toboggan algorithm for image clustering
    '''
    # Displacements (row-wise)
    displi = [-1, -1, -1,  0, 0,  1, 1, 1]
    displj = [-1,  0,  1, -1, 1, -1, 0, 1]

    # Get input shape
    height, width = x.shape

    # Get edge map
    edge = _getEdgeMap(x).astype(np.int32) 
    pad = np.pad(edge, pad_width=1, mode="edge")
 
    # Get levels of edge map (descending order) 
    levels = np.sort(np.unique(edge))[::-1]

    # Do toboggan clustering
    cls, mark = 1, np.zeros_like(x, dtype=np.int32)
    for level in levels:
        # Get level positions
        pos = np.argwhere(np.logical_and(edge == level, mark == 0))
       
        for i, j in pos:
            # Make search query
            if not mark[i, j]:
                mark[i, j] = cls
                query = deque([(i, j, cls)])
            else:
                continue
           
            while len(query) > 0:
                # Pop current position
                iq, jq, cls = query.popleft()

                # Verify domain               
                if iq == -1 or iq == height:
                    continue
                if jq == -1 or jq == width:
                    continue

                # Find adjacent neighbors not clustered 
                neighbors = pad[iq:iq+3, jq:jq+3][FANCE].ravel()
                
                # Find the sharpest gradient 
                if not len(neighbors[neighbors <= edge[iq, jq]]):
                    continue
                else:
                    slide = np.argmin(neighbors)

                # Slide slop
                ii, jj = iq + displi[slide], jq + displj[slide] 
                if not mark[ii, jj]:
                    # Update mark
                    mark[ii, jj] = cls

                    # Append new queue
                    query.append((ii, jj, cls))
                else:
                    # Merge segments
                    mark[mark == cls] = mark[ii, jj]

            # Update class
            cls += 1

    return mark


def superpixel(img, lbl, 
               sigma=0, numCls=1.e5, eps=1.e-30):
    """ Return components of image graph based on superpixel segmentation
    """
    # Generate superpixel with 'SLIC' algorithm
    mark = toboggan(img)
    _, mark = np.unique(mark, return_inverse=True)
    mark = mark.reshape(512, 512)

    # Pre-build graph conponent
    classes = np.unique(mark) 
    nodes = np.array([np.mean(img[mark == cls]) for cls in classes])
   
    ctr  = np.array([np.mean(np.nonzero(mark == cls), axis=1) for cls in classes]) 

    right = np.vstack([mark[:-1, :].ravel(), mark[1:, :].ravel()])
    down  = np.vstack([mark[:, :-1].ravel(), mark[:, 1:].ravel()])
    edges = np.unique(np.hstack([right, down]), axis=1)
    edges = edges[:, edges[0, :] != edges[1, :]]
    edges = np.unique(np.sort(edges, axis=0), axis=1)
    
    # Merge analogous neighboring sites
    merge = -np.ones_like(mark)
    for i, j in edges.T:
        if np.abs(nodes[i] - nodes[j]) < sigma:
            nodes[i], nodes[j] = (np.mean(img[(mark == i) | (mark == j)]), 
                                  np.mean(img[(mark == i) | (mark == j)]))

            if merge[mark == i][0] != -1 and merge[mark == j][0] != -1:
                merge[merge == j] = merge[mark == i][0]
            elif merge[mark == i][0] != -1 and merge[mark == j][0] == -1:
                merge[mark == j] = merge[mark == i][0]
            elif merge[mark == i][0] == -1 and merge[mark == j][0] == -1:
                merge[(mark == i) | (mark == j)] = i
    merge[merge == -1] = mark[merge == -1]   
    _, merge = np.unique(merge, return_inverse=True)
    merge = merge.reshape(512, 512)

    return mark


