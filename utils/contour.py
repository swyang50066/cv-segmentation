from    collections     import  deque

import  numpy       as  np
from    scipy.ndimage       import  distance_transform_edt


def _isBoundary(pos, shape):
    ''' Return boolean for checking boundary

        Parameters
        ----------------
        pos: (3,) list, tuple or ndarray
            position to check boundary condition

        Returns
        ----------------
        boolean: boolean
            Boundary condition checker
    '''
    # Check boundary condition
    if pos[0] == -1 or pos[0] == shape[0]:
        return True
    if pos[1] == -1 or pos[1] == shape[1]:
        return True

    return False


def getContour(region, bClosed=True):
    ''' Get curvature variables of centerline

        Parameters
        ----------------
        region: (H, W) ndarray
            Input region

        Returns
        ----------------
        contour: (N, 2) ndarray
            Contour points (sorted in clock-wise direction)
        (optional) bClosed: boolean
            Determine the output list of contours to be closed or open
            (closed mean the first and last element is same)

        Note
        ----------------
        The contour points are sorted in clock-wise
    '''
    # Clock-wise displacements (Moore's neighborhood)
    displi = [-1, -1, -1, 0, 1, 1, 1, 0]
    displj = [-1, 0, 1, 1, 1, 0, -1, -1]

    # Get edges
    edge = np.uint8(distance_transform_edt(region) == 1)

    # Get dimensions
    height, width = edge.shape
    center = (height//2, width//2)

    # Initialize query with the position in the row-wise order
    # So, contour seaching starts from right-middle neighbor node
    xpos, ypos = np.argwhere(edge == 1)[0]
    query = deque([(2, xpos, ypos)])    # (start neighbor index, x position, y position)

    # Declare mark domain
    mark = np.zeros_like(edge)

    # Search connected component
    contour = []
    while query:
        # Pop current position
        start, i, j = query.popleft()

        # Roll displacements for it starts at 
        # the next position of previous component
        dis = displi[start:] + displi[:start]
        djs = displj[start:] + displj[:start]

        # Find connected component in clock-wise
        for end, (di, dj) in enumerate(zip(dis, djs)):
            iq, jq = i + di, j + dj

            # Check domain
            if _isBoundary((iq, jq), (height, width)):
                continue
            if not edge[iq, jq]:
                continue
            if mark[iq, jq]:
                continue
            else:
                mark[iq, jq] = 1

            # Update query
            query.append(((start+end+5)%8, iq, jq))

            # Append sequential component 
            contour.append((iq, jq))
            break

        # Make contour a closet
        if not query and bClosed:
            contour.append(contour[0])

    return contour

