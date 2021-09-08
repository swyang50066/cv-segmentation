import  cv2
import  numpy                       as  np

from    skimage.segmentation        import  slic
from    skimage.segmentation        import  mark_boundaries
from    toboggan                    import  toboggan

import  matplotlib.pyplot           as  plt

def superpixel(img, lbl, 
               sigma=0, numCls=1.e5, eps=1.e-30):
    """ Return components of image graph based on superpixel segmentation
    """
    # Generate superpixel with 'SLIC' algorithm
    #mark = slic(img, n_segments=numCls, compactness=eps)
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
    
    """
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

    '''
    bdy = 255*mark_boundaries(img, merge) 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.imshow(bdy.astype(np.uint8))
    plt.show()
    '''

    # Make components    
    classes = np.unique(merge)
    nodes = np.array([np.mean(img[merge == cls]) for cls in classes])

    right = np.vstack([merge[:-1, :].ravel(), merge[1:, :].ravel()])
    down  = np.vstack([merge[:, :-1].ravel(), merge[:, 1:].ravel()])
    edges = np.unique(np.hstack([right, down]), axis=1)
    edges = edges[:, edges[0, :] != edges[1, :]]
    edges = np.unique(np.sort(edges, axis=0), axis=1)
    """

    return mark, nodes, edges, ctr

