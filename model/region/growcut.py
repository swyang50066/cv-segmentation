import  numpy       as  np
from    scipy.ndimage   import  distance_transform_edt      as  dist

    
def _funcStrengthReduction(p, q, norm=255):
    ''' Return strength reduction term 
    '''
    return 1 - np.sqrt((p - q)**2.)/np.sqrt(norm**2.)


class GrowCut(object):
    """ Grow-cut segmentation.

    Parameters
    ----------------
    image: (H, W) ndarray
        Input image.
    seed: (H, W) ndarray
        Input seed.
    state: (3, H, W) ndarray
        Initial state contains (label, strength, and feature) in the order. 
        The strength represents the certainty of the state, for example 
        '1' is a hard seed value that remains constant throughout segmentation.
    (optional) maxIter: integer
        The maximum number of automata iterations to allow.
    (optional) windowSize : integer
        Size of the neighborhood window.
    
    Returns
    -------
    region: (H, W) ndarray
        Segmentation region.
    """
    def __init__(self, t1=6, t2=6, maxIter=100, windowSize=5):
        # Parameters
        self.t1 = t1
        self.t2 = t2

        self.maxIter = maxIter
        self.windowSize = windowSize
        self.wing = (windowSize - 1)//2

    def openWindow(self, pos, shape):
        ''' open local window
        '''
        hslice = slice(max(0, pos[0]-self.wing), 
                       min(pos[0]+self.wing+1, shape[0]))
        wslice = slice(max(0, pos[1]-self.wing), 
                       min(pos[1]+self.wing+1, shape[1]))
        window = (hslice, wslice)
        
        return window

    def run(self, image, seed, t1=6, t2=6):
        # Get image parameters
        image = image.astype(np.float64)
        
        # Get input shape
        height, width = image.shape

        # Build state container
        currState = np.vstack([seed[np.newaxis, ...],
                               1.*(seed > 0)[np.newaxis, ...],
                               image[np.newaxis, ...]])

        # Declare maps
        nextState = currState.copy()
        currPrevent = np.ones_like(image)

        # Grow cut
        count, bRunning = 0, True
        while bRunning:
            # Declare next prevent
            nextPrevent = np.ones_like(image)

            # Search domain
            for i, j in np.ndindex(height, width):
                # Check valid pixel
                if image[i, j] == 0: continue
        
                # Get window indice
                window = self.openWindow((i, j), (height, width))
    
                # Get cell feature
                strength_p = currState[1][i, j]
                feature_p = currState[2][i, j]

                # Get neighbor features
                label_q = currState[0][window]
                strength_q = currState[1][window]
                feature_q = currState[2][window]

                # Check valid attack
                if not np.sum(strength_q): continue

                # Calculate strength reduction factor
                indicator = _funcStrengthReduction(feature_p, feature_q)
                
                # Update cell state by valid attacks
                '''
                First: 
                    a cell that has too many enemies (> T1) around it
                    is prohibited to attack its neighbors. 
                Second: 
                    a cell that has enemies (> T2) around it
                    is forced to be occupied by the weakest of it’s enemies, 
                    no matter the cell’s strength. 
                
                The enemies number is defined by the maximum number of 
                a enermy class cells (no unlabelled)
                '''
                mask = (indicator*strength_q)*currPrevent[window] > strength_p
                if np.any(mask):
                    # Smoothing condition
                    if np.sum(mask) > self.t1:
                        nextPrevent[i, j] = 0

                    nextState[0][i, j] = label_q[mask][0]
                    nextState[1][i, j] = (indicator*strength_q)[mask][0]
                
            # Count the number of iteration
            if count > self.maxIter:
                bRunning = False
            else:
                count += 1

            # Check change of state
            if not np.sum(nextState[0] != currState[0]):
                bRunning = False

            # Update state  
            currState = nextState.copy()
            currPrevent = nextPrevent.copy()

        return currState[0]
