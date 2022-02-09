import  cv2
import  numpy       as np
import  scipy.interpolate           

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  *
from    PyQt5.QtWidgets         import  *
 
from    file        import  File


class MarkUpCanvas(object):
    def __init__(self):
        super().__init__()

    def setCursorPosition(self, x, y):
        ''' set current cursor position in image coordinates
        '''
        # Set position
        self.cursorPos = (x, y)

    def getImageWithCursor(self, a=.9, b=.4, c=.1):
        ''' Return overlayered image with blusher circle
        '''
        # Get slice image
        imslice = self.image[self.axialIndex].copy()
        imoverlay = self.seedOverlay[self.axialIndex].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax
        
        imslice = np.uint8(255.*((imslice - self.valmin)
                    / (self.valmax - self.valmin))**self.gamma)

        # Make overlayered 
        value = cv2.addWeighted(imslice, a,
                                imoverlay, b,
                                c)

        # Mark blusher circle
        cv2.circle(value,
                   self.cursorPos,
                   self.blusherSize,
                   (255, 255, 255), -1)

        return value

    def getImageWithMark(self, markers, invisible, 
                               markerSize=3, a=.9, b=.4, c=.1):
        ''' Return overlayered image with mark-up position
        '''
        # Get slice image
        imslice = self.image[self.axialIndex].copy()
        imoverlay = self.seedOverlay[self.axialIndex].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax

        imslice = np.uint8(255.*((imslice - self.valmin)
                    / (self.valmax - self.valmin))**self.gamma)

        # Make overlayered 
        value = cv2.addWeighted(imslice, a,
                                imoverlay, b,
                                c)
        
        # Draw lines        
        for markerIndex, (axialIndex, inst) in enumerate(markers["dot"]):
            # Skip invisible marker
            if markerIndex in invisible["dot"]:
                continue
            
            # Skip marker on other slice
            if axialIndex != self.axialIndex:
                continue

            # Get point and text
            dot, text = inst

            # Display pivots
            cv2.circle(value,
                       dot,
                       markerSize,
                       (255, 192, 203), -1)

            # Display texts
            cv2.putText(value,
                        text,
                        dot,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)


        # Draw lines        
        for markerIndex, (axialIndex, line) in enumerate(markers["line"]):
            # Skip invisible marker
            if markerIndex in invisible["line"]:
                continue

            # Skip marker on other slice
            if axialIndex != self.axialIndex:
                continue

            # Get line points
            pivot, qivot = line
        
            if pivot == qivot == None:
                # Display pivots
                cv2.circle(value,
                           self.cursorPos,
                           markerSize,
                           (255, 192, 203), -1)
            elif pivot != None and qivot == None:
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)

                # Display connecting line
                cv2.line(value,
                         pivot,
                         self.cursorPos,
                         color=(255, 192, 203),
                         thickness=1)
            else:
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)
                cv2.circle(value,
                           qivot,
                           markerSize,
                           (255, 192, 203), -1)

                # Display connecting line
                cv2.line(value,
                         pivot,
                         qivot,
                         color=(255, 192, 203),
                         thickness=1)

        # Draw arc
        for markerIndex, (axialIndex, arc) in enumerate(markers["arc"]):
            # Skip invisible marker
            if markerIndex in invisible["arc"]:
                continue
            
            # Skip marker on other slice
            if axialIndex != self.axialIndex:
                continue

            # Get arc pivots
            pivot, node, qivot = arc

            if pivot == node == qivot == None:
                # Display pivots
                cv2.circle(value,
                           self.cursorPos,
                           markerSize,
                           (255, 192, 203), -1)

            elif (pivot != None and 
                  node == qivot == None):
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)
            
                # Display connecting line
                cv2.line(value,
                         pivot,
                         self.cursorPos,
                         color=(255, 192, 203),
                         thickness=1)
            elif (pivot != None and 
                  node != None and 
                  qivot == None):
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)
                cv2.circle(value,
                           node,
                           markerSize,
                           (255, 192, 203), -1)

                # Display connecting line
                cv2.line(value,
                         pivot,
                         node,
                         color=(255, 192, 203),
                         thickness=1)
                cv2.line(value,
                         node,
                         self.cursorPos,
                         color=(255, 192, 203),
                         thickness=1)
            else:
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)
                cv2.circle(value,
                           node,
                           markerSize,
                           (255, 192, 203), -1)
                cv2.circle(value,
                           qivot,
                           markerSize,
                           (255, 192, 203), -1)

                # Display connecting line
                cv2.line(value,
                         pivot,
                         node,
                         color=(255, 192, 203),
                         thickness=1)
                cv2.line(value,
                         node,
                         qivot,
                         color=(255, 192, 203),
                         thickness=1)

        # Draw boxs        
        for markerIndex, (axialIndex, box) in enumerate(markers["box"]):
            # Skip invisible marker
            if markerIndex in invisible["box"]:
                continue    

            # Skip marker on other slice
            if axialIndex != self.axialIndex:
                continue

            # Get line points
            pivot, qivot = box

            if pivot == qivot == None:
                # Display pivots
                cv2.circle(value,
                           self.cursorPos,
                           markerSize,
                           (255, 192, 203), -1)
            elif pivot != None and qivot == None:
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)

                # Display connecting line
                cv2.rectangle(value,
                              pivot,
                              self.cursorPos,
                              color=(255, 192, 203),
                              thickness=1)
            else:
                # Display pivots
                cv2.circle(value,
                           pivot,
                           markerSize,
                           (255, 192, 203), -1)
                cv2.circle(value,
                           qivot,
                           markerSize,
                           (255, 192, 203), -1)

                # Display connecting line
                cv2.rectangle(value,
                              pivot,
                              qivot,
                              color=(255, 192, 203),
                              thickness=1)

        # Draw spline contour    
        for markerIndex, (axialIndex, curve) in enumerate(markers["curve"]):
            # Skip invisible marker
            if markerIndex in invisible["curve"]:
                continue

            # Skip marker on other slice
            if axialIndex != self.axialIndex:
                continue

            # Make sequence
            if curve[-1] == None:
                if self.cursorPos != curve[-2]:
                    sequence = curve[:-1] + [self.cursorPos]
                else:
                    sequence = curve[:-1]
            else:
                sequence = curve

            if len(curve) > 4: 
                # Get positions
                x, y = np.array(sequence).T
            
                # Calculate weights of cumulative weightance 
                distance = np.sqrt((x[:-1] - x[1:])**2.
                                   +(y[:-1] - y[1:])**2.)
                weight = np.concatenate(([0], distance.cumsum()))

                # Spline interpolation
                spline, u = scipy.interpolate.splprep([x, y], u=weight, s=0)
                interp = np.linspace(weight[0], weight[-1], 200)
                xInterp, yInterp = scipy.interpolate.splev(interp, spline)

                # Get points
                points = [np.array([xInterp, yInterp], dtype=np.int32).T]
            elif 2 < len(curve) < 5:
                # Get positions
                x, y = np.array(sequence).T

                # Krogh interpolation
                xInterp = np.linspace(x[0], x[-1], 200)
                yInterp = scipy.interpolate.krogh_interpolate(x, y, xInterp)

                # Get points
                points = [np.array([xInterp, yInterp], dtype=np.int32).T]
            else:
                # Get points
                points = [np.array(sequence)]

            # Display pivots
            cv2.polylines(value, 
                          points, 
                          False, 
                          (255, 192, 203))

        return value       
 
    def getImageWithLocalBox(self, box, markerSize=3, a=.9, b=.4, c=.1):
        ''' Return overlayered image with mark-up position
        '''
        # Get slice image
        imslice = self.image[self.axialIndex].copy()
        imoverlay = self.seedOverlay[self.axialIndex].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax

        imslice = np.uint8(255.*((imslice - self.valmin)
                    / (self.valmax - self.valmin))**self.gamma)

        # Make overlayered 
        value = cv2.addWeighted(imslice, a,
                                imoverlay, b,
                                c)

        # Draw local domain
        if not box:
            # Display pivots
            cv2.circle(value,
                self.cursorPos,
                markerSize,
                (0, 128, 0), -1)
        elif len(box) == 1:
            # Shade outside
            outside = 255*np.ones_like(value)
            slices = (slice(min(box[0][1], self.cursorPos[1]),
                            max(box[0][1], self.cursorPos[1])),
                      slice(min(box[0][0], self.cursorPos[0]),
                            max(box[0][0], self.cursorPos[0])))
            outside[slices] = (0, 0, 0)
            value = cv2.addWeighted(imslice, 1.,
                                    outside, .4,
                                    0)   
               
            # Display pivots
            cv2.circle(value,
                       box[0],
                       markerSize,
                       (0, 128, 0), -1)

            # Display connecting line
            cv2.rectangle(value,
                          box[0],
                          self.cursorPos,
                          color=(0, 128, 0),
                          thickness=1)
        else:
            # Shade outside
            outside = 255*np.ones_like(value)
            slices = (slice(min(box[0][1], box[1][1]),
                            max(box[0][1], box[1][1])),
                      slice(min(box[0][0], box[1][0]),
                            max(box[0][0], box[1][0])))
            outside[slices] = (0, 0, 0)
            value = cv2.addWeighted(imslice, 1.,
                                    outside, .4,
                                    0)   
               
            # Display pivots
            cv2.circle(value,
                       box[0],
                       markerSize,
                       (0, 128, 0), -1)
            cv2.circle(value,
                       box[1],
                        markerSize,
                        (0, 128, 0), -1)

            # Display connecting line
            cv2.rectangle(value,
                          box[0],
                          box[1],
                          color=(0, 128, 0),
                          thickness=1)

        return value


class PlaneViewCanvas(object):
    def __init__(self):
        super().__init__()

    def getImageWithSeed(self, a=.9, b=.4, c=.1):
        ''' Return a*self.image + b*seedOverlay + c
        '''
        # Get slice image
        imslice = self.image[self.axialIndex].copy()
        imoverlay = self.seedOverlay[self.axialIndex].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax
        
        imslice = np.uint8(255.*((imslice - self.valmin)
                    / (self.valmax - self.valmin))**self.gamma)

        return cv2.addWeighted(imslice, a,
                               imoverlay, b,
                               c)

    def getImageWithSegment(self, a=.9, b=.4, c=.1):
        ''' Return a*self.image + b*segmentOverlay + c
        '''
        # Get slice image
        imslice = self.image[self.axialIndex].copy()
        imoverlay = self.segmentOverlay[self.axialIndex].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax
        
        imslice = np.uint8(255.*((imslice - self.valmin)
                    / (self.valmax - self.valmin))**self.gamma)

        return cv2.addWeighted(imslice, a,
                               imoverlay, b,
                               c)

    def getImageWithCoronal(self, a=.9, b=.4, c=.1):
        ''' Return a*self.image + b*segmentOverlay + c
        '''
        # Get slice image
        imslice = self.image[:, self.coronalIndex, :].copy()
        imoverlay = self.seedOverlay[:, self.coronalIndex, :].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax
        
        imslice = np.uint8(255.*((imslice - self.valmin) 
                    / (self.valmax - self.valmin))**self.gamma)

        return cv2.addWeighted(imslice, a,
                               imoverlay, b,
                               c)
                
    def getImageWithSagittal(self, a=.9, b=.4, c=.1):
        ''' Return a*self.image + b*segmentOverlay + c
        '''
        # Get slice image
        imslice = self.image[:, :, self.sagittalIndex].copy()
        imoverlay = self.seedOverlay[:, :, self.sagittalIndex].copy()

        # Normalization
        imslice[imslice < self.valmin] = self.valmin
        imslice[imslice > self.valmax] = self.valmax
        
        imslice = np.uint8(255.*((imslice - self.valmin) 
                    / (self.valmax - self.valmin))**self.gamma)

        return cv2.addWeighted(imslice, a,
                               imoverlay, b,
                               c)


class CPRViewCanvas(object):
    def __init__(self):
        super().__init__()

    def getImageWithCPRView(self, a=.9, b=.4, c=.1, bRotation=False):
        ''' Return CPR view slice
        '''
        # Initialize CPR view
        if isinstance(self.CPRVolume, type(None)):
            return np.zeros((256, 512, 3), dtype=np.uint8)
            
        # Get CPR volume shape
        height, width, depth = self.CPRVolume.shape

        # Build maps
        cpr, lbl = (np.zeros((width, depth, 3), dtype=np.uint8),
                    np.zeros((width, depth, 3), dtype=np.uint8))

        # Extract cut plane
        if bRotation:
            cpr[...] = np.expand_dims(
                                self.RotVolume[height//2], 
                                axis=-1)
            lbl[..., 0] = 255*self.RotRegion[height//2] 
        else:
            cpr[...] = np.expand_dims(
                                self.CPRVolume[height//2], 
                                axis=-1)
            lbl[..., 0] = 255*self.CPRRegion[height//2]
                               

        # Draw guide lines
        cv2.line(cpr,
                 (0, width//2),
                 (depth, width//2),
                 color=(0, 0, 255),
                 thickness=1)
        cv2.line(cpr,
                 (self.CPRIndex, 0),
                 (self.CPRIndex, width),
                 color=(255, 0, 0),
                 thickness=1)

        return cv2.addWeighted(cpr, a,
                               lbl, b,
                               c)

    def getImageWithCrossSection(self, a=.9, b=.4, c=.1):
        ''' Return cross section slice
        '''
        # Initialize CPR view
        if isinstance(self.CPRVolume, type(None)):
            return np.zeros((256, 512, 3), dtype=np.uint8)
            
        # Get CPR volume shape
        height, width, depth = self.CPRVolume.shape
            
        # Build maps
        cs, lbl = (np.zeros((height, width, 3), dtype=np.uint8),
                   np.zeros((height, width, 3), dtype=np.uint8))

        # Extract cut plane
        cs[...] = np.expand_dims(
                            self.CPRVolume[..., self.CPRIndex], 
                            axis=-1)
        lbl[..., 0] = 255*self.CPRRegion[..., self.CPRIndex] 

        return cv2.addWeighted(cs, a,
                               lbl, b,
                               c)
    
class LandmarkCanvas(object):
    def __init__(self):
        super().__init__()

    def getImageWithLandmark(self):
        ''' Return landmark overview
        '''
        # Get slice image
        imslice = np.zeros((256, 256, 3), dtype=np.uint8)
        imslice[:, :] = (0, 165, 255)

        return imslice


class GeometryCanvas(object):
    def __init__(self):
        super().__init__()
     
    def getImageWithGeometry(self):
        ''' Return geometry overview
        '''
        # Get slice image
        imslice = np.zeros((256, 256, 3), dtype=np.uint8)
        imslice[:, :] = (0, 165, 255)

        return imslice


class SeedCanvas(object):
    def __init__(self):
        super().__init__()

    def addSeed(self, x, y):
        ''' Add seed points (x, y are from image coordinates)
        '''
        # Get seed colors
        blue, green, red = self.colorSeed[self.typeSeed]
       
        # Non-specified color 
        if (blue, green, red) == 0: return 0

        # Draw circle
        cv2.circle(self.seedOverlay[self.axialIndex], 
                   (x, y), 
                   self.blusherSize, 
                   (blue, green, red), -1)

        # Update seed
        self.seed[self.axialIndex][
            (self.seedOverlay[self.axialIndex][..., 1] == green) &
            (self.seedOverlay[self.axialIndex][..., 2] == red)
            ] = self.typeSeed

    def removeSeed(self, x, y):
        ''' Remove seed points (x, y are from image coordinate)
        '''
        # Get seed colors
        blue, green, red = self.colorSeed[self.typeSeed]
       
        # Non-specified color 
        if (blue, green, red) == 0: return 0

        # Erase circle
        cv2.circle(self.seedOverlay[self.axialIndex], 
                   (x, y),
                   self.blusherSize,
                   (0, 0, 0), -1)

        # Update seed
        self.seed[self.axialIndex][
            (self.seedOverlay[self.axialIndex][..., 1] == 0) &
            (self.seedOverlay[self.axialIndex][..., 2] == 0)] = 0

    def clearSeed(self):
        ''' Reset seed
        '''
        # Clear seed domain
        self.seed[self.axialIndex][...] = 0.
        self.seedOverlay[self.axialIndex][...] = 0.


class Canvas(
        File, 
        MarkUpCanvas, 
        PlaneViewCanvas,
        CPRViewCanvas,
        LandmarkCanvas,
        GeometryCanvas,
        SeedCanvas):
    def __init__(self):
        super().__init__()
        # Pixelmap layers
        layers = File.importImage('./icons/watermelon.png')
        self.image = layers[0]
        self.seedOverlay = layers[1]
        self.segmentOverlay = layers[2]
     
        # Seed parameters
        self.typeSeed = 1
        self.colorSeed = {1: (0, 255, 0)}
        self.seed = np.zeros_like(layers[1])[..., 0]

        # Label
        self.label = np.zeros_like(self.seed)
        
        # CPRs
        self.medialAngle = 0
        self.CPRVolume = None
        self.CPRRegion = None
 
        # Spacing
        self.spacing = np.array([1., 1., 1.])
 
        # Thresholding 
        self.gamma = 1
        self.valmin = self.image.min()
        self.valmax = self.image.max()    

        # Initial index
        self.axialIndex = 0
        self.coronalIndex = layers[0][0].shape[0] // 2
        self.sagittalIndex = layers[0][0].shape[1] // 2
        self.CPRIndex = 0

        # Initial blusher size 
        self.blusherSize = 1

    def getQPixmap(self, cvimage):
        ''' Get Qt pixel map
        '''
        # Get input shape and convert to color map
        height, width, bytePerPixel = cvimage.shape
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)

        # Get Qt image 
        bytePerLine = 3*width
        qimage = QImage(cvimage.data,
                        width, height,
                        bytePerLine,
                        QImage.Format_RGB888)

        # Qt pixmap
        qpixmap = QPixmap.fromImage(qimage)

        return qpixmap
