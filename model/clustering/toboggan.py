import  cv2
import  numpy                       as  np
from    collections                 import  deque


def toboggan(img):
    dis = [-1, -1, -1,  0, 0,  1, 1, 1]
    djs = [-1,  0,  1, -1, 1, -1, 0, 1]
    fance = np.array([[True, True,  True], 
                      [True, False, True], 
                      [True, True,  True]])

    xgrad = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    ygrad = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    grad   = np.sqrt(xgrad**2. + ygrad**2.).astype(np.int32)
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
