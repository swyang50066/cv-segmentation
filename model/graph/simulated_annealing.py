import  os
import  glob
import  cv2
import  numpy    as  np

from    skimage.measure     import  label

def _getDisplacement(numNgb=4):
    ''' Return displacements of 'numNgb' neighbors
    '''
    if numNgb == 8:
        return np.argwhere(np.ones((3,3))) - [1, 1]
    elif numNgb == 4:
        return np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def _sgnUnion(c1, c2):
    ''' Return negativity of c1 and c2
    '''
    return -1 if c1 == c2 else 1


def _coolingSchedular(step, temp, initTemp, mode='gaussian'):
    ''' Return current temperature
    '''
    if mode == 'exp':
        return 0
    elif mode == 'gaussian':
        return initTemp / (1 + np.log(1+step))
    else:
        return -1


def _funcLogGaussianEnergy(x, pos, mu, var):
    ''' Return logarithmic Gibb's energy
    '''
    return np.log(np.sqrt(2*np.pi*var)) + .5*((x[pos] - mu)**2)/var


def _getClassParams(img, lbl):
    ''' Return class distribution
    '''
    classes = np.delete(np.unique(lbl), 0)
    theta = {cls: [np.mean(img[lbl == cls]),
                  np.var(img[lbl == cls])] for cls in classes}

    for key, value in theta.items():
        print(key, value)

    return theta


def _getInitEnergy(img, lbl, theta, beta=100):
    ''' Return initial Gibb's energy
    '''
    energy = 0
    pos = np.argwhere(lbl > 0)
    pad = np.pad(lbl, pad_width=1, mode='edge')
    for i, j in pos:
        # Read class params
        cls = lbl[i,j]
        mu, var = theta[cls][0], theta[cls][1]

        # Singleton energy
        energy += _funcLogGaussianEnergy(img, (i,j), mu, var)

        # Doubleton energy
        for dx, dy in _getDisplacement():
            xi, yj = i+dx, j+dy
            energy += beta*_sgnUnion(cls, pad[xi+1, yj+1])

    return energy


def _getDeltaEnergy(img, lbl, pos, ngbCls, theta, beta=100):
    ''' Return purturbation of energy by randon walk
    '''
    # Get parameters of each classes
    cls = lbl[pos]

    muPrev, varPrev = theta[cls][0], theta[cls][1]
    muNext, varNext = theta[ngbCls][0], theta[ngbCls][1]

    # Evaluate singleton energies
    prevEnergy  = _funcLogGaussianEnergy(img, pos, muPrev, varPrev)
    nextEnergy  = _funcLogGaussianEnergy(img, pos, muNext, varNext)
    deltaEnergy = nextEnergy - prevEnergy

    # Evaluate doubleton energies
    # Larger 'beta' yields more homogeneous classification
    pad = np.pad(lbl, pad_width=1, mode='edge')
    for dx, dy in _getDisplacement():
        xi, yj = pos[0]+dx, pos[1]+dy

        deltaEnergy += beta*(_sgnUnion(ngbCls, pad[xi+1, yj+1]) -
                             _sgnUnion(cls, pad[xi+1, yj+1]))

    return deltaEnergy


def _Idonthowtonamethisfunction(lbl): 
    ''' Return positions to be updated
    '''
    coord = np.argwhere(np.ones_like(lbl))
    pad = np.pad(lbl, pad_width=1, mode='edge')
    trial = [[i,j] for i, j in coord 
             if (0 < pad[i+1,j+1] < 7 and np.sum(pad[i:i+3, j:j+3])/pad[i+1,j+1] != 9) or
                (pad[i+1,j+1] == 7 and 0 < np.min(pad[i:i+3, j:j+3]) < 7) ]

    return trial 


def simulatedAnnealing(img, lbl,
                       maxIter=10, initTemp=50000, eps=1.e-4):
    # Generate numpy random seed
    np.random.seed(931016)

    # Get label information
    theta = _getClassParams(img, lbl)

    # Evaluate initial potentials
    energy = _getInitEnergy(img, lbl, theta)

    # Optimization
    step = 0
    web  = lbl.copy()
    classes = np.delete(np.unique(lbl), 0)
    while step < maxIter:
        print(step+1, maxIter)
        
        # Get temperature
        temp  = _coolingSchedular(step, 0, initTemp)
        
        for i, j in _Idontknowhowtonamethisfunction(web):
            # Randomly choice a random class except for previous one
            sample = np.setdiff1d(classes, web[i,j])
            ngbCls = np.random.choice(sample, 1)[0]

            # Evaluate perturbation
            deltaEnergy = _getDeltaEnergy(img, web, (i,j), ngbCls, theta)

            # Metropolice criteria
            zeta = np.random.rand()
            if (deltaEnergy <= 0 or
                (deltaEnergy > 0 and np.exp(-deltaEnergy / temp) < zeta)):
                web[i, j]  = ngbCls
                energy    += deltaEnergy

        # Update iterations
        temp  = _coolingSchedular(step, temp, initTemp)
        step += 1

    return web


def _readData(case):
    bgr = cv2.imread(case).astype(np.uint8)

    # Extract intensity, class map
    #   'LA':1, 'LV':2, 'RA':3, 
    #   'RV':4, 'Ao':5, 'LV Wall':6, 'myocardium':7
    img = bgr[:, :, 0]
    lbl = (1 * (bgr[:, :, 1] - bgr[:, :, 2] == 248).astype(np.uint8) +
           2 * (bgr[:, :, 1] - bgr[:, :, 2] == 112).astype(np.uint8) +      
           3 * (bgr[:, :, 1] - bgr[:, :, 2] == 32).astype(np.uint8) +      
           4 * (bgr[:, :, 2] - bgr[:, :, 1] == 32).astype(np.uint8) +      
           5 * (bgr[:, :, 2] - bgr[:, :, 1] == 112).astype(np.uint8) +      
           6 * (bgr[:, :, 2] - bgr[:, :, 1] == 248).astype(np.uint8))       
    myo = ((bgr[:, :, 0] > 0) & 
           (bgr[:, :, 1] - bgr[:, :, 2] != 248) & 
           (bgr[:, :, 1] - bgr[:, :, 2] != 112) & 
           (bgr[:, :, 1] - bgr[:, :, 2] != 32) & 
           (bgr[:, :, 2] - bgr[:, :, 1] != 32) & 
           (bgr[:, :, 2] - bgr[:, :, 1] != 112) & 
           (bgr[:, :, 2] - bgr[:, :, 1] != 248)).astype(np.uint8)
   
    # Detect central cardiac area 
    bg    = label((myo > 0).astype(np.uint8), connectivity=1)
    area  = [np.sum(bg == i+1) for i in range(bg.max())]
    lbl  += 7 * (bg == np.argsort(area)[::-1][0]+1).astype(np.uint8)

    binary = (lbl > 0).astype(np.uint8)
    contours, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Fill holes
    for contour in contours:
        x = np.zeros_like(binary)
        cv2.fillPoly(x, pts=[contour], color=[1,1,1])
        lbl += 7 * ((lbl == 0) & (x > 0)).astype(np.uint8)


    return img, lbl


def _saveData(img, lbl):
    result = np.zeros((512, 512, 3))
    #result[:, :, :] = np.expand_dims(img, axis=-1)  
 
    # Mark labels
    for x, y in np.argwhere(lbl == 1):
        result[x, y, 1], result[x, y, 2] = 255, 7
    for x, y in np.argwhere(lbl == 2):
        result[x, y, 1], result[x, y, 2] = 127, 15
    for x, y in np.argwhere(lbl == 3):
        result[x, y, 1], result[x, y, 2] = 63, 31
    for x, y in np.argwhere(lbl == 4):
        result[x, y, 1], result[x, y, 2] = 31, 63
    for x, y in np.argwhere(lbl == 5):
        result[x, y, 1], result[x, y, 2] = 15, 127
    for x, y in np.argwhere(lbl == 6):
        result[x, y, 1], result[x, y, 2] = 7, 255
    for x, y in np.argwhere(lbl == 7):
        result[x, y, 1], result[x, y, 2] = 93, 10

    cv2.imwrite('./result.bmp', result)


if __name__=='__main__':
    # Read data
    case = './sample3.bmp'
    img, lbl = _readData(case)

    # Run markov-random-walk
    #lbl = simulatedAnnealing(img, lbl)

    # Save data
    _saveData(img, lbl)
