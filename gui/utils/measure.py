from    collections     import  deque

import  scipy
import  numpy           as  np


def isBoundary(pos, shape, ndim=3):
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
    if ndim == 3:
        if pos[2] == -1 or pos[2] == shape[2]:
            return True

    return False


def getBoxDomain(domain, pos, size):
    ''' Return box domain with 'size'

        Parameters
        ----------------
        domain: (W, D, H) ndarray
            global domain
        pos: (3,) list, tuple or ndarray
            center position of local box domain
        (optional) size: (3,) list, tuple or ndarray
            size of box domain (lW, lD, lH)
            
        Returns
        ----------------
        box: (lW, lD, lH) ndarray
            local box domain   
    '''
    # Get domain dimensions
    shape = domain.shape
    width, depth, height = (size[0]//2,
                            size[1]//2,
                            size[2]//2)

    # Set half size of edges
    wmin, wmax = (max(pos[0]-width, 0),
                  min(pos[0]+width+1, shape[0]))
    dmin, dmax = (max(pos[1]-depth, 0),
                  min(pos[1]+depth+1, shape[1]))
    hmin, hmax = (max(pos[2]-height, 0),
                  min(pos[2]+height+1, shape[2]))
    window = (slice(wmin, wmax),
              slice(dmin, dmax),
              slice(hmin, hmax))

    # Extract local box domain
    box = domain[window]

    return box


def getGaussianKernel(sigma, size=5, ndim=3):
    ''' Build Gaussian kernel

        Parameters
        ----------------
        sigma: integer
            discreted variance of Gaussian kernel
        (optional) size: (2,) or (3,) list, tuple or ndarray
            size of box domain (W, H) or (W, D, H)
        (optional) ndim:
            specified input dimension

        Returns
        ----------------
        kernel: (W, H) or (W, D, H) ndarray
            Discreted Gaussian kernel
    '''
    # Open kernal domain
    if ndim == 2:
        # Get domain dimensions
        shape = domain.shape
        width, height = (size[0]//2,
                         size[1]//2)

        kernel = np.zeros(size)
        kernel[width, height] = 1
    else:
        # Get domain dimensions
        shape = domain.shape
        width, depth, height = (size[0]//2,
                                size[1]//2,
                                size[2]//2)

        kernel = np.zeros(size)
        kernel[width, depth, height] = 1

    # Build discretized gaussian kernel
    kernel = distance_transform_edt(1 - kernel)
    kernel = (np.exp(-.5*kernel**2./sigma**2.)
              /np.sqrt(2*np.pi*sigma**2.))

    # Normalize kernel
    kernel /= np.sum(kernel)

    return kernel


def getDistance(p, q, ndim=3, scale=None):
    ''' Return Euclidean distance between 'p' and 'q'

        Parameters
        ----------------
        p: (N,) list, tuple or ndarray
            start position of distance measure
        q: (N,) list, tuple or ndarray
            end position of distance measure
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes

        Returns
        ----------------
        distance: float
            Euclidean distance between 'p' and 'q'        
    '''
    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1],
                           q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1])])
        else:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1]),
                           scale[2]*(q[2] - p[2])])

    # Euclidean distance
    distance = np.sqrt(np.sum(dr**2.))

    return distance


def getMahaDistance(p, q, invCov, ndim=3, scale=None):
    ''' Return Mahalanobis distance between 'p' and 'q'

        Parameters
        ----------------
        p: (N,) list, tuple or ndarray
            start position of distance measure
        q: (N,) list, tuple or ndarray
            end position of distance measure
        invCov: (N, N) list, ndarray
            inverse of covariance matrix
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes

        Returns
        ----------------
        distance: float
            Mahalanobis distance between 'p' and 'q'        
    '''
    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1],
                           q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1])])
        else:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1]),
                           scale[2]*(q[2] - p[2])])

    # Mahalanobis distance
    distance = np.matmul(invCov, dr)
    distance = np.sqrt(np.matmul(dr, distance))

    return distance


def getManhattanDistance(p, q, ndim=3, scale=None):
    ''' Return Manhattan distance between 'p' and 'q'

        Parameters
        ----------------
        p: (N,) list, tuple or ndarray
            start position of distance measure
        q: (N,) list, tuple or ndarray
            end position of distance measure
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes

        Returns
        ----------------
        distance: float
            Manhattan distance between 'p' and 'q'        
    '''
    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1],
                           q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1])])
        else:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1]),
                           scale[2]*(q[2] - p[2])])

    # Manhattan distance
    distance = np.sum(np.abs(dr))

    return distance


def getMinkowskiDistance(p, q, degree=2, ndim=3, scale=None):
    ''' Return Minkowski distance between 'p' and 'q'

        Parameters
        ----------------
        p: (N,) list, tuple or ndarray
            start position of distance measure
        q: (N,) list, tuple or ndarray
            end position of distance measure
        (optional) degree: integer
            degree of distance measure
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes

        Returns
        ----------------
        distance: float
            Minkowski distance between 'p' and 'q'        
    '''
    # Invalid degree
    if not degree:
        return getManhattanDistance(p, q, ndim=ndim, scale=scale)
    elif degree == 2:
        return getDistance(p, q, ndim=ndim, scale=scale)

    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1],
                           q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1])])
        else:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1]),
                           scale[2]*(q[2] - p[2])])

    # Minkowski distance
    distance = np.power(np.sum(np.power(dr, degree)), 1./degree)

    return distance


def getProjDistance(p, q, norm, scale=None, bVertical=False):
    ''' Return projection distance between p' and 'q' onto 'norm'

        Parameters
        ----------------
        p: (N,) list, tuple or ndarray
            start position of distance measure
        q: (N,) list, tuple or ndarray
            end position of distance measure
        invCov: (N, N) list, ndarray
            inverse of covariance matrix
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes
        (optional) bVertical: boolean
            boolean whether to calculate vertical distance

        Returns
        ----------------
        distance: float
            Projection distance between 'p' and 'q' onto 'norm'        
    '''
    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1],
                           q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1])])
        else:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1]),
                           scale[2]*(q[2] - p[2])])

    # Projected distance
    distance = np.sum(norm*dr)

    if bVertical:
        # Calculate vertical distance
        Euclid = getDistance(p, q, ndim=ndim, scale=scale)
        vertical = np.sqrt(Euclid**2. - distance**2.)

        return distance, vertical
    else:
        return distance


def getNormVector(p, q, ndim=3, scale=None):
    ''' Return unit norm from 'p' to 'q'
        
        Parameters
        ----------------
        p: (N,) list, tuple or ndarray
            start position of distance measure
        q: (N,) list, tuple or ndarray
            end position of distance measure
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes
        
        Returns
        ----------------
        norm: (N,) list, tuple or ndarray
            unit norm from 'p' to 'q'
    '''
    # Check whether input types are matched
    if not isinstance(p, type(q)):
        p, q = np.array(p), np.array(q)

    # Displacement
    if isinstance(scale, type(None)):
        if ndim == 2:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1]])
        else:
            dr = np.array([q[0] - p[0],
                           q[1] - p[1],
                           q[2] - p[2]])
    else:
        if ndim == 2:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1])])
        else:
            dr = np.array([scale[0]*(q[0] - p[0]),
                           scale[1]*(q[1] - p[1]),
                           scale[2]*(q[2] - p[2])])

    # Unit normal vector
    norm = dr / np.sqrt(np.sum(dr**2.))

    return norm


def getInterAngle(u, v, ndim=3, scale=None):
    ''' Return intervection angle between 'u' to 'v'
        
        Parameters
        ----------------
        u: (N,) list, tuple or ndarray
            first vector
        v: (N,) list, tuple or ndarray
            second vector
        (optional) ndim: integer
            specified input dimension
        (optional) scale: (N,) list, tuple or ndarray
            scale vector giving weights for each axes
        
        Returns
        ----------------
        angle: float
            intersection angle in radian
    '''
    # Check whether input types are matched
    if not isinstance(u, type(v)):
        u, v = np.array(u), np.array(v)

    # Apply scale factor
    if isinstance(scale, type(None)):
        u, v = scale*u, scale*v

    # Normalize vectors
    u = getNormVector(np.zeros(len(u)), u, ndim=ndim, scale=None)
    v = getNormVector(np.zeros(len(v)), v, ndim=ndim, scale=None)

    # Get intersection angle
    angle = np.arccos(np.sum(u*v))

    return angle


def getPlaneParams(points):
    ''' Get plane parameters

        Parameters
        ----------------
        points: (3, 3) ndarray
            three in-plane positions

        Returns
        ----------------
        params: (4, ) list
            coefficient and constant of plane equation
            (a*x + b*y + c*z = d)

        Note
        ----------------
        The norm calculated in this method do not consider the direction
        whether it is inwarding or outward of the plane

    '''
    # Check input type
    if not isinstance(points, type(np.empty(1))):
        points = np.array(points)

    # Get displacement vectors
    vec1 = points[0] - points[1]
    vec2 = points[2] - points[1]

    # Get plane norm
    norm = np.cross(vec1, vec2)
    norm = getNormVector((0, 0, 0), norm)

    # Calculate constant of plane equation
    const = -np.sum(norm*points)

    # Plane parameter space (in order of a, b, c, d)
    params = [norm[0], norm[1], norm[2], const]

    return params


def getCurveParams(centers, interval=3):
    ''' Get curvature variables of centerline

        Parameters
        ----------------
        centers: (N, 3) ndarray
            center positions of barycenterline
        (optional) interval: integer
            pointwise interval used as scale variable for
            curvature estimation

        Returns
        ----------------
        curve: (N, 2)
            pointwise curvature center and its radius
        traj: (N, 3)
            pointwise curve vectors of tangent, normal, binormal

        Note
        ----------------
        This function understands the 'centers' are given in
        in the direction from proximal to distal.
    '''
    # Declare containers
    curve = []

    # Get centerline parameters
    for k in range(len(centers[:-2*interval])):
        # Get three pivot points 
        if k < interval:
            curr, next = np.array([
                                centers[k],
                                centers[k+interval]])
            prev = curr + (curr - next)
        elif k >= len(centers) - interval:
            prev, curr = np.array([
                                centers[k-interval],
                                centers[k]])
            next = curr + (curr - prev)
        else:
            prev, curr, next = np.array([
                                    centers[k],
                                    centers[k+interval],
                                    centers[k+2*interval]])

        # Displacement vectors
        vec1 = np.cross(prev-curr, next-curr)
        vec2 = np.cross(vec1, prev-curr)
        vec3 = np.cross(vec1, next-curr)

        # Corresponding lengths
        arc1 = getDistance(vec1, (0,0,0))
        arc2 = getDistance(next, curr)
        arc3 = getDistance(prev, curr)

        # Stright line (i.e., infinite curvature)
        if not arc1:
            tangent = getNormVector(prev, next)
            curve.append((curr, tangent))
            continue

        # Get displacement vector from 'curr' to curvature center
        delta = .5*(vec2*arc2**2. - vec3*arc3**2.)/arc1**2.

        # Unit curvature vector
        normal = getNormVector((0, 0, 0), delta)

        # Unit tangent vector
        tangent = np.cross(vec1, normal)
        tangent = getNormVector((0, 0, 0), tangent)
        
        # Unit binormal vector
        binormal = np.cross(normal, tangent)

        # Append parameters
        curve.append((curr, tangent))

    return curve


def calcBarycenter(verts, normals):
    ''' Calculate barycenter of a profile

        Parameters
        ----------------
        verts: (N, 3) ndarray
            vertex positions of the profile
        normals: (N, 3) ndarray
            unit normal vectors at the node cusp

        Returns
        ----------------
        barycenter: (3, ) ndarray
            barycenter position of the profile 
 
        Note
        ----------------
        For the convention of mesh parameters,
        the Instances of both 'verts' and 'normals' are paired 
        in the order of array index. 

        This algorithm finds barycenter at which total angular momentum
        of node diversity (with normals) is minimized
    '''
    # Vectorize inputs
    verts, normals = np.array(verts), np.array(normals)

    # Calculate matrix element    
    xx = np.sum(normals[:, 1]**2. + normals[:, 2]**2.)
    xy = -np.sum(normals[:, 0]*normals[:, 1])
    xz = -np.sum(normals[:, 0]*normals[:, 2])
    yy = np.sum(normals[:, 0]**2. + normals[:, 2]**2.)
    yz = -np.sum(normals[:, 1]*normals[:, 2])
    zz = np.sum(normals[:, 0]**2. + normals[:, 1]**2.)
    
    ii = np.sum(
            verts[:, 0]*(normals[:, 1]**2. + normals[:, 2]**2.)
            - verts[:, 1]*normals[:, 0]*normals[:, 1]
            - verts[:, 2]*normals[:, 0]*normals[:, 2])
    jj = np.sum(
            verts[:, 1]*(normals[:, 0]**2. + normals[:, 2]**2.)
            - verts[:, 0]*normals[:, 0]*normals[:, 1]
            - verts[:, 2]*normals[:, 1]*normals[:, 2])
    kk = np.sum(
            verts[:, 2]*(normals[:, 0]**2. + normals[:, 1]**2.)
            - verts[:, 0]*normals[:, 0]*normals[:, 2]
            - verts[:, 1]*normals[:, 1]*normals[:, 2])
    
    # Build linear system
    A = np.array([[xx, xy, xz],
                  [xy, yy, yz],
                  [xz, yz, zz]])
    b = np.array([ii, jj, kk])

    # Solve ODE to get barycenter
    barycenter = scipy.linalg.solve(A, b)

    # Calculate cut-plane radius
    radius = np.sum((verts - barycenter)**2, axis=1)
    radius = np.max(np.sqrt(radius))

    return barycenter, radius


def transformGridCoord(norm, pos):
    ''' Calculate transformed grid coordinates of 'norm' plane

        parameters
        ----------------
        norm: (3,) ndarray
            unit normal vector of a plane
        pos: (N, 3) ndarray
            reference grid positions of yz plane

        returns
        ----------------
        trans: (N, 3) ndarray
            grid positions of transformed plane
 
        functions
        ----------------
        _matRotPhi: (phi: float) -> ndarray
            rotation matrix along phi
        _matRotTheta: (theta: float, axis: ndarray) -> ndarray
            Rodrigues rotation matrix along theta of norm axis
    '''
    def _matRotPhi(phi):
        ''' Return 2D rotational matrix in 3D
        '''
        return np.array(
            [[np.cos(phi), -np.sin(phi), 0],
             [np.sin(phi),  np.cos(phi), 0],
             [          0,            0, 1]])


    def _matRotTheta(theta, axis):
        ''' Return Rodrigues rotation matrix
        '''
        return np.array(
            [[np.cos(theta) + axis[0]**2.*(1 - np.cos(theta)),
              axis[0]*axis[1]*(1 - np.cos(theta)) - axis[2]*np.sin(theta),
              axis[0]*axis[2]*(1 - np.cos(theta)) + axis[1]*np.sin(theta)],
             [axis[1]*axis[0]*(1 - np.cos(theta)) + axis[2]*np.sin(theta),
              np.cos(theta) + axis[1]**2.*(1 - np.cos(theta)),
              axis[1]*axis[2]*(1 - np.cos(theta)) - axis[0]*np.sin(theta)],
             [axis[2]*axis[0]*(1 - np.cos(theta)) - axis[1]*np.sin(theta),
              axis[2]*axis[1]*(1 - np.cos(theta)) + axis[0]*np.sin(theta),
              np.cos(theta) + axis[2]**2.*(1 - np.cos(theta))]])
    
    # Get theta for rotatioal matrix along z axis
    if norm[0] > 0:
        phi = np.arctan(norm[1]/norm[0])
    elif norm[1] >= 0 and norm[0] < 0:
        phi = np.pi + np.arctan(norm[1]/norm[0])
    elif norm[1] < 0 and norm[0] < 0:
        phi = -np.pi + np.arctan(norm[1]/norm[0])
    elif norm[1] >= 0 and norm[0] == 0:
        phi = np.pi/2.
    elif norm[1] < 0 and norm[0] == 0:
        phi = -np.pi/2.
    phi += 2*np.pi if phi < 0 else 0

    # Get projection of the norm onto x-axis
    axis0 = np.matmul(_matRotPhi(phi), np.array([1,  0, 0]))
    axis1 = np.matmul(_matRotPhi(phi), np.array([0, -1, 0]))
    nProj = norm[0]*axis0[0] + norm[1]*axis0[1]

    # Get phi for rotational matrix along the norm
    margin = np.sqrt(norm[0]**2. + norm[1]**2.)
    if nProj > 0:
        theta = np.arctan(norm[2]/margin)
    elif norm[2] >= 0 and nProj < 0:
        theta = np.pi + np.arctan(norm[2]/margin)
    elif norm[2] <  0 and nProj < 0:
        theta = -np.pi + np.arctan(norm[2]/margin)
    elif norm[2] >= 0 and nProj == 0:
        theta = np.pi/2.
    elif norm[2] <  0 and nProj == 0:
        theta = -np.pi/2.
    theta += 2*np.pi if theta < 0 else 0 

    # Get psi for minimized rotational axis
    psi = np.pi/2. - np.sign(norm[2])*phi

    # Apply rotational matrices
    trans = np.matmul(_matRotPhi(phi), pos.T).T
    trans = np.matmul(_matRotTheta(theta, axis1), trans.T).T
    trans = np.matmul(_matRotTheta(psi, norm), trans.T).T
    
    return trans

