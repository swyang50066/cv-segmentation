import  numpy           as  np
from    collections     import  deque


class GrowRegion(object):
    ''' Region-growth algorithm

    Parameters
    ----------------
    image: (H, W) ndarray
        Input image
    seed: (H, W) ndarray
        Input seed
    (optional) threshold: integer or float
        Treshold value for image segmentation

    Returns
    ----------------
    region: (H, W) ndarray
        Segmentation region
    '''
    def __init__(self, threshold=0):
        # Parameters
        self.threshold = threshold

    def run(self, image, seed):
        # Get input shape
        height, width = image.shape        

        # Displacements (8 neighbors)
        displi = [-1, -1, -1, 0, 0, 1, 1, 1]
        displj = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Initialize region domain
        region = np.uint8(seed > 0)

        # Grow region 
        query = deque(np.argwhere(seed > 0))

        while len(query) > 0:
            # Pop current position
            i, j = query.popleft()

            # Search neighboring pixels
            for di, dj in zip(displi, displj):
                iq, jq = i + di, j + dj

                # Check domain
                if iq == -1 or iq == height:
                    continue
                if jq == -1 or jq == width:
                    continue
                if image[iq, jq] <= self.threshold:
                    continue
                if region[iq, jq] != 0:
                    continue

                # Update domain      
                region[iq, jq] = 1

                # Append next query
                query.append((iq, jq))

        return region

