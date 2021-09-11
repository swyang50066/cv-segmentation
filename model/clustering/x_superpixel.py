from    collections     import  deque

import  numpy                       as  np


def toboggan(img):
    dis = [-1, -1, -1,  0, 0,  1, 1, 1]
    djs = [-1,  0,  1, -1, 1, -1, 0, 1]
    fance = np.array([[True, True,  True], 
                      [True, False, True], 
                      [True, True,  True]])

    xgrad = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    ygrad = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    grad   = np.sqrt(.5*(xgrad**2. + ygrad**2.)).astype(np.int32)
    levels = np.sort(np.unique(grad))[::-1]

    cls  = 1
    mark = np.zeros_like(grad)
    for level in levels:
        pos = np.argwhere((grad == level) & (mark == 0))
        
        for i, j in pos:
            mark[i, j] = cls
            query = deque()
            query.append((i, j, cls))
            while len(query) > 0:    
                iq, jq, cls = query.popleft()
               
                if iq == 0 or iq == 511 or jq == 0 or jq == 511: 
                    break

                ngb = grad[iq-1:iq+2, jq-1:jq+2][fance].ravel()
                ckp = mark[iq-1:iq+2, jq-1:jq+2][fance].ravel()
               
                if len(ngb[ngb <= grad[iq, jq]]) == 0:
                    break
 
                slide = np.argmin(ngb <= grad[iq, jq])

                ii, jj = iq + dis[slide], jq + djs[slide] 

                if mark[ii, jj] == 0:
                    query.append((ii, jj, cls))
                    mark[ii, jj] = cls
                else:
                    mark[mark == cls] = mark[ii, jj]
                    break

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

