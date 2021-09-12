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
    # Extract image pairs with Sobel filter
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


class Superpixel(object):
    ''' Superpixel algorithm of image segmentation
        This function regions fragments of image domain 
        generated from clustering algorithm (e.g., toboggan, slic). 

    Parameters
    ----------------
    image: (H, W) ndarray
        input image.
    
    Returns
    ----------------
    region (H, W) ndarray
        segmentation label
    '''
    def __init__(self, method="toboggan", sigma=16):
        # Model parameter
        self.sigma = sigma
    
        self.method = method

    def run(self, image):
        # Get input shape
        height, width = image.shape
        
        # Cluster image domain
        if self.method == "toboggan":
            # Apply toboggan algorithm
            fragment = toboggan(image)

            # Extract fragment classes
            _, fragment = np.unique(fragment, return_inverse=True)
            fragment = fragment.reshape(height, width)
            classes = np.unique(fragment)
        elif self.method == "slic":
            return 0
        else:
            raise ValueError("Wrong Clustering Method IS ENCOUNTERED!")

        # Pre-build graph nodes
        nodes = np.array([np.mean(image[fragment == cls]) for cls in classes])
  
        # Find adjacent fragment pairs 
        left2right = np.vstack([fragment[:-1, :].ravel(), 
                                fragment[1:, :].ravel()])
        top2bottom = np.vstack([fragment[:, :-1].ravel(), 
                                fragment[:, 1:].ravel()])
        
        pairs = np.unique(np.hstack([left2right, top2bottom]), axis=1)
        pairs = pairs[:, pairs[0, :] != pairs[1, :]]
        pairs = np.unique(np.sort(pairs, axis=0), axis=1)
   
        # Merge same-kind adjacet fragments
        region = -np.ones_like(fragment)
        for i, j in pairs.T:
            if np.abs(nodes[i] - nodes[j]) < self.sigma:
                # Get merged region
                merge = image[np.logical_or(fragment == i, fragment == j)]
                
                # Change node property with merged region thing
                nodes[i], nodes[j] = (np.mean(merge), np.mean(merge)) 

                # Merge fragment pair to be an identical region
                if (region[fragment == i][0] != -1 and 
                    region[fragment == j][0] != -1):
                    region[fragment == i] = region[fragment == j][0]
                elif (region[fragment == i][0] != -1 and 
                      region[fragment == j][0] == -1):
                    region[fragment == j] = region[fragment == i][0]
                elif (region[fragment == i][0] == -1 and 
                      region[fragment == j][0] == -1):
                    region[(fragment == i) | (fragment == j)] = i
       
        # Remark unmerged fragments
        region[region == -1] = fragment[region == -1]   
        _, region = np.unique(region, return_inverse=True)
        region = region.reshape(height, width)

        return region


