import  cv2
import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *

from    file            import  File


class FileIOAction(object):
    def __init__(self):
        super().__init__()
    
    def openDICOM(self):
        ''' Open input file
        '''
        # Read file path
        path = QFileDialog.getOpenFileName()

        # No path given
        if not str(path[0]): return 0

        # Get image layers
        layers, attribute = File.importDICOM(str(path[0]))
        self.canvas.image = layers[0]
        self.canvas.seedOverlay = layers[1]
        self.canvas.segmentOverlay = layers[2]
        self.canvas.seed = np.zeros_like(layers[1])[..., 0]
        self.canvas.spacing = layers[3]

        # Reset label
        self.canvas.label = np.zeros_like(self.canvas.seed)

        # Reset markers
        self.markers = {"dot": [],
                        "line": [],
                        "arc": [],
                        "box": [],
                        "curve": []}
        self.invisibleMarkerIndex = {"dot": [],
                                     "line": [],
                                     "arc": [],
                                     "box": [],
                                     "curve": []}
        self.markerHashMap = {}

        # Reset overview items
        overviewTreeItems = self.overviewTree.findItems(
                                                "Marker",
                                                Qt.MatchContains |
                                                Qt.MatchRecursive)
        for overviewTreeItem in overviewTreeItems:    
            self.annotationTreeItem.removeChild(overviewTreeItem)
        
        self.segmentTreeItem.takeChildren()
        self.segmentTreeItem.setFlags(Qt.ItemIsSelectable)
        
        self.CPRTreeItem.takeChildren()
        self.CPRTreeItem.setFlags(Qt.ItemIsSelectable)

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

        # Adjust scroll bar range
        self.scrollBar.setMaximum(len(self.canvas.image)-1)

        # Set histogram
        self.canvas.gamma = 1
        self.canvas.valmin, self.canvas.valmax = 0, 255
        self.histogramView.initialize(self.canvas.image,
                                      self.histCanvas,
                                      self.histAxes)

        # Set DICOM attribute table
        for k, (key, value) in enumerate(attribute.items()):
            tableItem = QTableWidgetItem(value)
            self.dicomTable.setItem(k, 0, tableItem)
        
        # Update python console variables
        variables = {"image": self.canvas.image}
        self.ipyConsole.pushVariables(variables)

    def openFile(self):
        ''' Open input file
        '''
        # Read file path
        path = QFileDialog.getOpenFileName()
   
        # No path given
        if not str(path[0]): return 0
    
        # Get image layers
        layers = File.importImage(str(path[0]))
        self.canvas.image = layers[0]
        self.canvas.seedOverlay = layers[1]
        self.canvas.segmentOverlay = layers[2]  
        self.canvas.seed = np.zeros_like(layers[1])[..., 0]      

        # Reset label
        self.canvas.label = np.uint8(self.canvas.image[..., 1] 
                                     - self.canvas.image[..., 2] == 255)
        #np.zeros_like(self.canvas.seed)

        # Reset markers
        self.markers = {"dot": [],
                        "line": [],
                        "arc": [],
                        "box": [],
                        "curve": []}
        self.invisibleMarkerIndex = {"dot": [],
                                     "line": [],
                                     "arc": [],
                                     "box": [],
                                     "curve": []}
        self.markerHashMap = {}

        # Reset overview items
        overviewTreeItems = self.overviewTree.findItems(
                                                "Marker",
                                                Qt.MatchContains |
                                                Qt.MatchRecursive)
        for overviewTreeItem in overviewTreeItems:
            self.annotationTreeItem.removeChild(overviewTreeItem)

        self.segmentTreeItem.takeChildren()
        self.segmentTreeItem.setFlags(Qt.ItemIsSelectable)

        self.CPRTreeItem.takeChildren()
        self.CPRTreeItem.setFlags(Qt.ItemIsSelectable)

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

        # Adjust scroll bar range
        self.scrollBar.setMaximum(len(self.canvas.image)-1)

        # Set histogram
        self.canvas.gamma = 1
        self.canvas.valmin, self.canvas.valmax = 0, 255
        self.histogramView.initialize(self.canvas.image,
                                      self.histCanvas,
                                      self.histAxes)
    
        # Update python console variables
        variables = {"image": self.canvas.image}
        self.ipyConsole.pushVariables(variables)

    def saveFile(self):
        ''' Save output file
        '''
        # Read output path
        path = QFileDialog.getExistingDirectory()
        
        # save output file    
        File.exportImage(path, 
                         self.canvas.image, 
                         self.canvas.label, 
                         self.canvas.colorSeed,
                         fformat='bmp')

    def saveSTL(self):
        ''' Save STL
        '''
        # Read output path
        path = QFileDialog.getExistingDirectory()
        
        # save output file    
        File.exportSTL(path, 
                       self.canvas.label)

    def closeWindow(self):
        ''' Close application window
        '''
        self.window.close()
