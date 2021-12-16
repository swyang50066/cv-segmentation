import  cv2
import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *

from    utils.measure           import  getDistance, getInterAngle


class AxialViewMouse(object):
    def __init__(self):
        super().__init__()

    def mouseAxialPress(self, event):
        ''' Activate if mouse is pressed
        '''
        # Change mouse status 
        self.bMousePress = True

        # Search maximum marker index
        maxMarkerCode = 0
        if self.markUpTag:
            # Get tree items
            numChild = self.annotationTreeItem.childCount()

            for index in range(numChild):
                markUpItem = self.annotationTreeItem.child(index)

                # Skip non mark-up items
                if "Marker" not in markUpItem.text(0):
                    continue

                # Get marker code                        
                markerCode = int(markUpItem.text(0).split("Marker")[-1])

                # Update maximum marker index
                if markerCode > maxMarkerCode:
                    maxMarkerCode = markerCode

        # Seed labeling
        if event.button() == 1:    # Left click
            # Set current operator
            self.mouseStatus = "left"
               
            # Actions 
            if not self.markUpTag and not self.bLocalHistogram:
                # Add seed points
                self.canvas.addSeed(int(event.x()/self.scaleFactor), 
                                    int(event.y()/self.scaleFactor))

                # Display added seed
                axialQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSeed())
                coronalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithCoronal())
                sagittalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSagittal())
                
                self.axialQLabel.setPixmap(axialQPixmap)
                self.coronalQLabel.setPixmap(coronalQPixmap)
                self.sagittalQLabel.setPixmap(sagittalQPixmap)
            elif self.markUpTag == "dot":
                # Append dots
                pivot = (int(event.x()/self.scaleFactor),
                         int(event.y()/self.scaleFactor))
                self.markers["dot"].append((self.canvas.axialIndex, 
                                            (pivot, "Text")))
 
                # Update history 
                if self.bMarkUpUndo:
                    self.markUpUndos = []
                    self.bMarkUpUndo = False 
                self.markUpHistory.append(
                                    (maxMarkerCode+1, 
                                     len(self.markers["dot"])-1, 
                                     "dot", 
                                     self.markers["dot"][-1]))
           
                # Update marker hash map
                self.markerHashMap[("dot", maxMarkerCode+1)] = (
                    len(self.markers["dot"])-1)

                # Update overview            
                overviewTreeItem = QTreeWidgetItem()
                overviewTreeItem.setText(0, "Marker%s"%str(maxMarkerCode+1))
                overviewTreeItem.setText(1, "Dot")
                overviewTreeItem.setText(2, "On")
                overviewTreeItem.setText(3, "Text")
                overviewTreeItem.setFlags(Qt.ItemIsEnabled |
                                          Qt.ItemIsSelectable |
                                          Qt.ItemIsUserCheckable)
                overviewTreeItem.setCheckState(2, Qt.Checked)
                self.annotationTreeItem.insertChild(0, overviewTreeItem)
                
                # Display mouse status
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)
            elif self.markUpTag == "line":
                # Append lines
                if self.lineMarkUpCount == 0:
                    pivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.markers["line"].append((self.canvas.axialIndex, 
                                                 (pivot, None)))

                    # Add count
                    self.lineMarkUpCount += 1
                elif self.lineMarkUpCount == 1:
                    pivot = self.markers["line"][-1][1][0]
                    qivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.markers["line"][-1] = ((self.canvas.axialIndex, 
                                                 (pivot, qivot)))
                
                    # Update history 
                    if self.bMarkUpUndo:
                        self.markUpUndos = []
                        self.bMarkUpUndo = False 
                    self.markUpHistory.append(
                                        (maxMarkerCode+1, 
                                         len(self.markers["line"])-1,
                                         "line", 
                                         self.markers["line"][-1]))
                    
                    # Add count
                    self.lineMarkUpCount += 1
                
                    # Update marker hash map
                    self.markerHashMap[("line", maxMarkerCode+1)] = (
                        len(self.markers["line"])-1)
               
                    # Get distance in mm
                    distance = getDistance(pivot, 
                                           qivot, 
                                           ndim=2, 
                                           scale=self.canvas.spacing) 
                    distance = np.round(distance, 1)

                    # Update overview            
                    overviewTreeItem = QTreeWidgetItem()
                    overviewTreeItem.setText(0, 
                                        "Marker%s"%str(maxMarkerCode+1))
                    overviewTreeItem.setText(1, "Line")
                    overviewTreeItem.setText(2, "On")
                    overviewTreeItem.setText(3, "%s mm"%str(distance))
                    overviewTreeItem.setFlags(Qt.ItemIsEnabled |
                                                 Qt.ItemIsSelectable |
                                                 Qt.ItemIsUserCheckable)
                    overviewTreeItem.setCheckState(2, Qt.Checked)
                    self.annotationTreeItem.insertChild(0, overviewTreeItem)
                
                # Display mouse status
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)

                # Reset count
                self.lineMarkUpCount = self.lineMarkUpCount % 2
            elif self.markUpTag == "arc":
                # Append arcs
                if self.arcMarkUpCount == 0:
                    pivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.markers["arc"].append((self.canvas.axialIndex,
                                                (pivot, None, None)))

                    # Add count
                    self.arcMarkUpCount += 1
                elif self.arcMarkUpCount == 1:
                    pivot = self.markers["arc"][-1][1][0]
                    node = (int(event.x()/self.scaleFactor),
                            int(event.y()/self.scaleFactor))
                    self.markers["arc"][-1] = ((self.canvas.axialIndex,
                                                (pivot, node, None)))

                    # Add count
                    self.arcMarkUpCount += 1
                elif self.arcMarkUpCount == 2:
                    pivot = self.markers["arc"][-1][1][0]
                    node = self.markers["arc"][-1][1][1]
                    qivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.markers["arc"][-1] = ((self.canvas.axialIndex,
                                                (pivot, node, qivot)))

                    # Update history 
                    if self.bMarkUpUndo:
                        self.markUpUndos = []
                        self.bMarkUpUndo = False 
                    self.markUpHistory.append(
                                        (maxMarkerCode+1, 
                                         len(self.markers["arc"])-1,
                                         "arc", 
                                         self.markers["arc"][-1]))
                    
                    # Add count
                    self.arcMarkUpCount += 1
                    
                    # Update marker hash map
                    self.markerHashMap[("arc", maxMarkerCode+1)] = (
                        len(self.markers["arc"])-1)

                    # Get angle in degree
                    u = (pivot[0] - node[0], pivot[1] - node[1])
                    v = (qivot[0] - node[0], qivot[1] - node[1])
                    angle = getInterAngle(u, v, 
                                          ndim=2, 
                                          scale=self.canvas.spacing)
                    angle = np.round(180.*angle/np.pi, 1)

                    # Update overview            
                    overviewTreeItem = QTreeWidgetItem()
                    overviewTreeItem.setText(0, 
                                        "Marker%s"%str(maxMarkerCode+1))
                    overviewTreeItem.setText(1, "Arc")
                    overviewTreeItem.setText(2, "On")
                    overviewTreeItem.setText(3, "%s degree"%str(angle))
                    overviewTreeItem.setFlags(Qt.ItemIsEnabled |
                                                 Qt.ItemIsSelectable |
                                                 Qt.ItemIsUserCheckable)
                    overviewTreeItem.setCheckState(2, Qt.Checked)
                    self.annotationTreeItem.insertChild(0, overviewTreeItem)

                # Display mouse status
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)

                # Reset count
                self.arcMarkUpCount = self.arcMarkUpCount % 3
            elif self.markUpTag == "box":
                # Append boxs
                if self.boxMarkUpCount == 0:
                    pivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.markers["box"].append((self.canvas.axialIndex,
                                                (pivot, None)))

                    # Add count
                    self.boxMarkUpCount += 1
                elif self.boxMarkUpCount == 1:
                    pivot = self.markers["box"][-1][1][0]
                    qivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.markers["box"][-1] = ((self.canvas.axialIndex,
                                                (pivot, qivot)))

                    # Update history 
                    if self.bMarkUpUndo:
                        self.markUpUndos = []
                        self.bMarkUpUndo = False 
                    self.markUpHistory.append(
                                        (maxMarkerCode+1, 
                                         len(self.markers["box"])-1,
                                         "box", 
                                         self.markers["box"][-1]))
                    
                    # Add count
                    self.boxMarkUpCount += 1
                    
                    # Update marker hash map
                    self.markerHashMap[("box", maxMarkerCode+1)] = (
                        len(self.markers["box"])-1)

                    # Get area in cm2
                    area = abs(pivot[0] - qivot[0])*abs(pivot[1] - qivot[1])
                    area = np.round(area/100., 2)

                    # Update overview            
                    overviewTreeItem = QTreeWidgetItem()
                    overviewTreeItem.setText(0, 
                                        "Marker%s"%str(maxMarkerCode+1))
                    overviewTreeItem.setText(1, "Box")
                    overviewTreeItem.setText(2, "On")
                    overviewTreeItem.setText(3, "%s cm2"%str(area))
                    overviewTreeItem.setFlags(Qt.ItemIsEnabled |
                                                 Qt.ItemIsSelectable |
                                                 Qt.ItemIsUserCheckable)
                    overviewTreeItem.setCheckState(2, Qt.Checked)
                    self.annotationTreeItem.insertChild(0, overviewTreeItem)

                # Display mouse status
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)

                # Reset count
                self.boxMarkUpCount = self.boxMarkUpCount % 2
            elif self.markUpTag == "curve":
                # Add curve pivots
                pivot = (int(event.x()/self.scaleFactor),
                         int(event.y()/self.scaleFactor))
                if self.curveMarkUpCount == 0:
                    self.curveMarkUpCount += 1
                    self.markers["curve"].append((self.canvas.axialIndex,
                                                  [pivot, None]))
                else:
                    # Append marker
                    axialIndex, marker = self.markers["curve"][-1]
                    marker = marker[:-1] + [pivot, None]
                
                    self.markers["curve"][-1] = (axialIndex, marker)

                # Display mouse status
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)
            elif self.bLocalHistogram:
                # Append lines
                if self.localHistogramCount == 0:
                    # Reset local box domain
                    if self.localBox: self.localBox = []

                    pivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.localBox.append(pivot)

                    # Add count
                    self.localHistogramCount += 1
                elif self.localHistogramCount == 1:
                    qivot = (int(event.x()/self.scaleFactor),
                             int(event.y()/self.scaleFactor))
                    self.localBox.append(qivot)

                    # Add count
                    self.localHistogramCount += 1

                # Display mouse status
                localBoxQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithLocalBox(self.localBox))
                self.axialQLabel.setPixmap(localBoxQPixmap)

                # Reset count
                self.localHistogramCount = self.localHistogramCount % 2
        elif event.button() == 2 and not self.bLocalHistogram:    # Right click       
            if not self.markUpTag:
                # Set current operator
                self.mouseStatus = "right"
            
                # React
                self.canvas.removeSeed(int(event.x()/self.scaleFactor), 
                                       int(event.y()/self.scaleFactor))
            
                # Display added seed
                axialQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSeed())
                coronalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithCoronal())
                sagittalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSagittal())

                self.axialQLabel.setPixmap(axialQPixmap)
                self.coronalQLabel.setPixmap(coronalQPixmap)
                self.sagittalQLabel.setPixmap(sagittalQPixmap)
            elif self.markUpTag == "curve":
                # Reset curve mark-up status
                self.curveMarkUpCount = 0

                # Remove dummy point
                axialIndex, marker = self.markers["curve"][-1]
                self.markers["curve"][-1] = (axialIndex, marker[:-1])

                # Update history 
                if self.bMarkUpUndo:
                    self.markUpUndos = []
                    self.bMarkUpUndo = False
                self.markUpHistory.append(
                                    (maxMarkerCode+1,
                                     len(self.markers["curve"])-1,
                                     "curve",
                                     self.markers["curve"][-1]))

                # Update marker hash map
                self.markerHashMap[("curve", maxMarkerCode+1)] = (
                    len(self.markers["curve"])-1)

                # Update overview            
                overviewTreeItem = QTreeWidgetItem()
                overviewTreeItem.setText(
                                    0,"Marker%s"%str(maxMarkerCode+1))
                overviewTreeItem.setText(1, "Curve")
                overviewTreeItem.setText(2, "On")
                overviewTreeItem.setFlags(Qt.ItemIsEnabled |
                                          Qt.ItemIsSelectable |
                                          Qt.ItemIsUserCheckable)
                overviewTreeItem.setCheckState(2, Qt.Checked)
                self.annotationTreeItem.insertChild(0, overviewTreeItem)

                # Display mouse status
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)
        elif event.button() == 4:    # wheel click
            # Set current operator
            self.mouseStatus = "wheel"
            
            # Set pin point
            self.xpin = event.x()
            self.ypin = event.y()

    def mouseAxialMove(self, event):
        ''' Activate dragging actions
        '''
        # Seed labeling
        if self.bMousePress and not self.bLocalHistogram:
            # Do labeling
            if self.mouseStatus == "left" and not self.markUpTag:
                # Add seed
                self.canvas.addSeed(int(event.x()/self.scaleFactor),
                                    int(event.y()/self.scaleFactor))

                # Display added seed
                axialQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSeed())
                coronalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithCoronal())
                sagittalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSagittal())

                self.axialQLabel.setPixmap(axialQPixmap)
                self.coronalQLabel.setPixmap(coronalQPixmap)
                self.sagittalQLabel.setPixmap(sagittalQPixmap)
            elif self.mouseStatus == "right":
                # Remove seed 
                self.canvas.removeSeed(int(event.x()/self.scaleFactor),
                                       int(event.y()/self.scaleFactor))

                # Display removed seed
                axialQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSeed())
                self.axialQLabel.setPixmap(axialQPixmap)
            elif self.mouseStatus == "wheel":
                # Set move point                
                xmove = (self.axialScrollArea.horizontalScrollBar().value()
                         - (event.x() - self.xpin))
                ymove = (self.axialScrollArea.verticalScrollBar().value()
                         - (event.y() - self.ypin))

                # Clip scroll ranges
                xmax = self.axialScrollArea.horizontalScrollBar().maximum()
                ymax = self.axialScrollArea.verticalScrollBar().maximum()
                xmove = int(np.clip(xmove, 1, xmax-1))
                ymove = int(np.clip(ymove, 1, ymax-1))

                # Move scroll area
                self.axialScrollArea.horizontalScrollBar().setValue(xmove)
                self.axialScrollArea.verticalScrollBar().setValue(ymove)

                # Update pin points
                self.xpin = event.x()
                self.ypin = event.y()
        elif not self.bMousePress:    # Mouse movement
            # Set cursor position
            self.canvas.setCursorPosition(int(event.x()/self.scaleFactor),
                                          int(event.y()/self.scaleFactor))
            
            # Display mouse status
            if not self.markUpTag and not self.bLocalHistogram:
                circleQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithCursor())
                self.axialQLabel.setPixmap(circleQPixmap)
            elif self.markUpTag and not self.bLocalHistogram:
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers,
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)
            else:
                localBoxQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithLocalBox(self.localBox))
                self.axialQLabel.setPixmap(localBoxQPixmap)

    def mouseAxialRelease(self, event):
        ''' Reset mouse status
        '''
        self.bMousePress = False
        self.mouseStatus = None

        if self.bLocalHistogram and len(self.localBox) == 2:
            # Get local domain
            slices = (self.canvas.axialIndex,
                      slice(min(self.localBox[0][1], self.localBox[1][1]),
                            max(self.localBox[0][1], self.localBox[1][1])),
                      slice(min(self.localBox[0][0], self.localBox[1][0]),
                            max(self.localBox[0][0], self.localBox[1][0])),
                      0)
            domain = self.canvas.image[slices]

            # Set local histogram
            self.histogramView.drawLocalHistogram(
                                          self.canvas.image,
                                          domain,
                                          self.histCanvas,
                                          self.histAxes)

            # Set local histogram information
            self.lowerLine.setText("0")
            self.upperLine.setText("255")
            self.meanText.setText(
                str(np.mean(domain).astype(np.uint8)))
            self.stdText.setText(
                str(np.std(domain).astype(np.uint8)))
            self.gammaText.setText(str(self.canvas.gamma))
            self.percentText.setText("100 %")

    def wheelAxialScroll(self, event, zoomin=1.25, zoomout=.8):
        ''' Activate wheel action on axial view
        '''
        # Get wheel-rolling direction
        direction = np.sign(event.angleDelta().y())

        if event.modifiers() == Qt.ControlModifier:
            # Get scaling factor
            factor = zoomin if direction > 0 else zoomout

            # Update global scale factor
            self.scaleFactor *= factor

            # save previous status
            xpos, ypos = event.x(), event.y()
            xmax = self.axialScrollArea.horizontalScrollBar().maximum()
            ymax = self.axialScrollArea.verticalScrollBar().maximum()
            xstep = self.axialScrollArea.horizontalScrollBar().pageStep()
            ystep = self.axialScrollArea.verticalScrollBar().pageStep()

            # Resize label area            
            self.axialQLabel.resize(
                self.scaleFactor*self.axialQLabel.pixmap().size())

            # Locate to center
            self.axialScrollArea.setAlignment(Qt.AlignCenter)

            # Set move point 
            xmove = factor*xpos - xstep/2
            ymove = factor*ypos - ystep/2

            # Clip scroll ranges
            xmax = factor*xmax + (factor - 1)*xstep
            ymax = factor*ymax + (factor - 1)*ystep
            xmove = int(np.clip(xmove, 1, xmax-1))
            ymove = int(np.clip(ymove, 1, ymax-1))

            # Move scroll area
            self.axialScrollArea.horizontalScrollBar().setValue(xmove)
            self.axialScrollArea.verticalScrollBar().setValue(ymove)
        if event.modifiers() == Qt.ShiftModifier:
            if direction > 0:
                self.scrollBar.setValue(
                    max(0, self.scrollBar.value()-1))
            else:
                numSlice = len(self.canvas.image)
                self.scrollBar.setValue(
                    min(numSlice, self.scrollBar.value()+1))


class CoronalViewMouse(object):
    def __init__(self):
        super().__init__()

    def wheelCoronalScroll(self, event):
        ''' Activate wheel action on coronal view
        '''
        # Get wheel-rolling direction
        direction = np.sign(event.angleDelta().y())

        if event.modifiers() == Qt.ShiftModifier:
            if direction > 0:
                self.canvas.coronalIndex = (
                    max(0, self.canvas.coronalIndex-1))
            else:
                self.canvas.coronalIndex = (
                    min(self.canvas.image.shape[1],
                        self.canvas.coronalIndex+1))

            # Display coronal view
            coronalQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithCoronal())
            self.coronalQLabel.setPixmap(coronalQPixmap)


class SagittalViewMouse(object):
    def __init__(self):
        super().__init__()

    def wheelSagittalScroll(self, event):
        ''' Activate wheel action on sagittal view
        '''
        # Get wheel-rolling direction
        direction = np.sign(event.angleDelta().y())

        if event.modifiers() == Qt.ShiftModifier:
            if direction > 0:
                self.canvas.sagittalIndex = (
                    max(0, self.canvas.sagittalIndex-1))
            else:
                self.canvas.sagittalIndex = (
                    min(self.canvas.image.shape[2],
                        self.canvas.sagittalIndex+1))

            # Display sagittal view
            sagittalQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSagittal())
            self.sagittalQLabel.setPixmap(sagittalQPixmap)


class CPRViewMouse(object):
    def __init__(self):
        super().__init__()

    def wheelCPRViewScroll(self, event):
        ''' Activate wheel action on CPR view
        '''
        # Get wheel-rolling direction
        direction = np.sign(event.angleDelta().y())
                
        # Get shape of CPR view
        height, width, depth = self.canvas.CPRVolume.shape

        # Rotate CPR volumes
        if (event.modifiers() == Qt.ShiftModifier and
            not isinstance(self.canvas.CPRVolume, type(None))):
            if direction > 0:
                # Counter-clockwise
                self.canvas.medialAngle += 10
                matrix = cv2.getRotationMatrix2D((width/2, height/2), 
                                                 self.canvas.medialAngle, 
                                                 1)
            else:
                # Clockwise
                self.canvas.medialAngle -= 10
                matrix = cv2.getRotationMatrix2D((width/2, height/2), 
                                                 self.canvas.medialAngle,
                                                 1)
  
            # Get cut-plane at rotation frame
            self.canvas.RotVolume = cv2.warpAffine(
                                            self.canvas.CPRVolume,
                                            matrix,
                                            (width, height))        
            self.canvas.RotRegion = np.uint8(cv2.warpAffine(
                                                    self.canvas.CPRRegion,
                                                    matrix,
                                                    (width, height)) > .5)        

            # Display CPR view
            CPRViewQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithCPRView(bRotation=True))
            self.CPRViewQLabel.setPixmap(CPRViewQPixmap)
        # Slice cross-sections
        elif (event.modifiers() == Qt.ControlModifier and
              not isinstance(self.canvas.CPRVolume, type(None))):
            if direction > 0:
                # Counter-clockwise
                self.canvas.CPRIndex = min(self.canvas.CPRVolume.shape[2]-1,
                                           self.canvas.CPRIndex+1)
            else:
                # Clockwise
                self.canvas.CPRIndex = max(0, self.canvas.CPRIndex-1)

            # Check medial angle
            bRotation = False if self.canvas.medialAngle == 0 else True
 
            # Display CPR view
            CPRViewQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithCPRView(bRotation=bRotation))
            self.CPRViewQLabel.setPixmap(CPRViewQPixmap)

            # Display cross section
            crossSectionQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithCrossSection())
            self.crossSectionQLabel.setPixmap(crossSectionQPixmap)

    def wheelCrossSectionScroll(self, event):
        ''' Activate wheel action on cross-section view
        '''
        # Get wheel-rolling direction
        direction = np.sign(event.angleDelta().y())

        # Get shape of CPR view
        height, width, depth = self.canvas.CPRVolume.shape

        if (event.modifiers() == Qt.ControlModifier and
              not isinstance(self.canvas.CPRVolume, type(None))):

            if direction > 0:
                self.canvas.CPRIndex = min(self.canvas.CPRVolume.shape[2]-1,
                                           self.canvas.CPRIndex+1)
            else:
                self.canvas.CPRIndex = max(0, self.canvas.CPRIndex-1)

            # Check medial angle
            bRotation = False if self.canvas.medialAngle == 0 else True
 
            # Display CPR view
            CPRViewQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithCPRView(bRotation=bRotation))
            self.CPRViewQLabel.setPixmap(CPRViewQPixmap)

            # Display cross section
            crossSectionQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithCrossSection())
            self.crossSectionQLabel.setPixmap(crossSectionQPixmap)


class HistogramMouse(object):
    def __init__(self):
        super().__init__()

    def mouseHistPress(self, event):
        ''' Activate histogram mouse press
        '''
        # Change hitogram mouse status
        self.bMouseHistPress = True

        # Get cursor position
        xpos, ypos = event.xdata, event.ydata
        if xpos != None:
            if (xpos <= self.canvas.valmin+4 and
                xpos <= self.canvas.valmax):
                # Set lower limit
                lowerLimit = int(xpos)
                self.lowerLine.setText(str(int(lowerLimit)))

                # Set intensity limits
                self.canvas.valmin = lowerLimit

                # Set current operator 
                self.mouseHistStatus = "left"
            elif (xpos >= self.canvas.valmax-4 and
                  xpos >= self.canvas.valmin):
                # Set upper limit
                upperLimit = int(xpos)
                self.upperLine.setText(str(int(upperLimit)))

                # Set intensity limits
                self.canvas.valmax = upperLimit

                # Set current operator 
                self.mouseHistStatus = "right"

    def mouseHistMove(self, event):
        ''' Activate histogram mouse dragging
        '''
        # Get cursor position
        xpos, ypos = event.xdata, event.ydata
        if self.bMouseHistPress:
            # Update limits
            if (self.mouseHistStatus == "left" and
                xpos <= self.canvas.valmax):
                # Set lower limit
                lowerLimit = int(xpos)
                self.lowerLine.setText(str(int(lowerLimit)))

                # Set intensity limits
                self.canvas.valmin = lowerLimit
            elif (self.mouseHistStatus == "right" and
                  xpos >= self.canvas.valmin):
                # Set upper limit
                upperLimit = int(xpos)
                self.upperLine.setText(str(int(upperLimit)))

                # Set intensity limits
                self.canvas.valmax = upperLimit

            # Update histogram information
            volume = self.canvas.image[
                        (self.canvas.image >= self.canvas.valmin) &
                        (self.canvas.image <= self.canvas.valmax)]
            percentage = np.round(100.*volume.size
                              / self.canvas.image.size, 2)
            self.meanText.setText(
                str(np.mean(volume).astype(np.uint8)))
            self.stdText.setText(
                str(np.std(volume).astype(np.uint8)))
            self.percentText.setText("{0} %".format(percentage))

            # Shift range 
            self.histogramView.shiftRange(self.canvas.valmin,
                                          self.canvas.valmax,
                                          self.histCanvas,
                                          self.histAxes)

            # Display image
            axialQPixmap = self.canvas.getQPixmap(
                                self.canvas.getImageWithSeed())
            segmentQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSegment())
            coronalQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithCoronal())
            sagittalQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSagittal())

            self.axialQLabel.setPixmap(axialQPixmap)
            self.segmentQLabel.setPixmap(segmentQPixmap)
            self.coronalQLabel.setPixmap(coronalQPixmap)
            self.sagittalQLabel.setPixmap(sagittalQPixmap)

    def mouseHistRelease(self, event):
        ''' Reset histogram mouse status
        '''
        self.bMouseHistPress = False
        self.mouseHistStatus = None

    def wheelHistScroll(self, event, gammaup=1.25, gammadown=.8):
        ''' Do gamma correction
        '''
        # Update gamma
        if event.button == "up":
            self.canvas.gamma = min(gammaup**3.,
                                    self.canvas.gamma*gammaup)
        elif event.button == "down":
            self.canvas.gamma = max(gammadown**3.,
                                    self.canvas.gamma*gammadown)
        self.gammaText.setText(str(np.round(self.canvas.gamma, 2)))

        # Display image
        axialQPixmap = self.canvas.getQPixmap(
                            self.canvas.getImageWithSeed())
        segmentQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSegment())
        coronalQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithCoronal())
        sagittalQPixmap = self.canvas.getQPixmap(
                                self.canvas.getImageWithSagittal())

        self.axialQLabel.setPixmap(axialQPixmap)
        self.segmentQLabel.setPixmap(segmentQPixmap)
        self.coronalQLabel.setPixmap(coronalQPixmap)
        self.sagittalQLabel.setPixmap(sagittalQPixmap)


class ScreenshotMouse(object):
    def __init__(self):
        super().__init__()

    def mouseScreenPress(self, event):
        ''' Activate if mouse is pressed
        '''
        pass


    def mouseScreenMove(self, event):
        ''' Activate if mouse is moved
        '''
        pass

    def mouseScreenRelease(self, event):
        ''' Activate if mouse is released
        '''
        pass



class MouseAction(
            AxialViewMouse,
            CoronalViewMouse,
            SagittalViewMouse,
            CPRViewMouse,
            HistogramMouse,
            ScreenshotMouse):
    def __init__(self):
        super().__init__()
        # Boolean for mouse status 
        self.bMousePress = False
        self.mouseStatus = None

        # Boolean for histogram mouse status
        self.bMouseHistPress = False
        self.mouseHistStatus = None

        # Scale factor of image canvas
        self.scaleFactor = 1.

    def wheelSideScroll(self):
        ''' Change image slice
        '''
        # Change index
        self.canvas.axialIndex = self.scrollBar.value()

        # Change display
        axialQPixmap = self.canvas.getQPixmap(
                                self.canvas.getImageWithSeed())
        segmentQPixmap = self.canvas.getQPixmap(
                                self.canvas.getImageWithSegment())

        self.axialQLabel.setPixmap(axialQPixmap)
        self.segmentQLabel.setPixmap(segmentQPixmap)

