import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *


class MarkUpAction(object):
    def __init__(self):
        super().__init__()
        # Mark-up tag 
        self.markUpTag = None

        # Mark-up status
        self.bMarkUpUndo = False

        # Mark-up counts
        self.lineMarkUpCount = 0
        self.arcMarkUpCount = 0
        self.curveMarkUpCount = 0
        self.boxMarkUpCount = 0        

        # Mark-up markers
        self.markers = {"dot": [],
                        "line": [],
                        "arc": [],
                        "box": [],
                        "curve": []}

        # Invisible markers
        self.invisibleMarkerIndex = {"dot": [],
                                     "line": [],
                                     "arc": [],
                                     "box": [],
                                     "curve": []}

        # Marker hash map
        self.markerHashMap = {}
   
        # Mark-up history
        self.markUpHistory = []
        self.markUpUndos = []
        self.invisibleMarkerUndoIndex = {"dot": [],
                                         "line": [],
                                         "arc": [],
                                         "box": [],
                                         "curve": []}
        self.markerUndoHashMap = {} 
        self.markerUndoItems = []

    def turnOnMarkUpTag(self, value):
        ''' Turn on mark-upTag
        '''
        # Turn off local histogram
        if self.bLocalHistogram and value:
            self.localBox = []
            self.bLocalHistogram = False
            self.localHistButton.setStyleSheet("background-color: white")

        # Change markup button status
        numTurnOff = 0
        for marker, button in self.markUpButtons.items():
            # Read button status
            color = button.palette().button().color().name()
            
            # Turn on/off mark-up
            if marker == value:
                if (color == "#ffffff" and 
                    self.markUpTag == None):
                    # Change mark-up status
                    self.markUpTag = marker
                    
                    button.setStyleSheet("background-color: gray")
                elif (color == "#ffffff" and  
                      self.markUpTag != None):
                    # Turn off previous mark-up
                    self.markUpButtons[self.markUpTag].setStyleSheet(
                        "background-color: white")

                    # Change mark-up status
                    self.markUpTag = marker
                    button.setStyleSheet("background-color: gray")

                    # Reset local histogram
                    self.bLocalHistogram = False
                else:
                    # Change mark-up status
                    self.markUpTag = None
                    button.setStyleSheet("background-color: white")

                    # Count turn-on button
                    numTurnOff += 1
            else:
                button.setStyleSheet("background-color: white")

                # Count turn-on button
                numTurnOff += 1

        if numTurnOff == len(self.markers):
            # Reset status
            self.markUpTag = None            

            # Reset cache
            self.markUpHistory = []
            self.markUpUndos = []
            self.invisibleMarkerUndoIndex = {"dot": [],
                                             "line": [],
                                             "arc": [],
                                             "box": [],
                                             "curve": []}
            self.markerUndoHashMap = {}
            self.markerUndoItems = []

            # Update pixmap
            seedQPixmap = self.canvas.getQPixmap(
                                        self.canvas.getImageWithSeed())
            self.axialQLabel.setPixmap(seedQPixmap)
        else:
            # Update markers
            markUpQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithMark(self.markers,
                                             self.invisibleMarkerIndex))
            self.axialQLabel.setPixmap(markUpQPixmap)

    def turnOnMarkUpDo(self):
        if self.markUpUndos:
            # Read marking history
            undos = self.markUpUndos.pop(-1)
            markerCode, markerIndex, markerType, marker = undos
            self.markUpHistory.append((markerCode, 
                                       markerIndex, 
                                       markerType, 
                                       marker))
            
            # Update makers
            self.markers[markerType].append(marker)

            # Restore invisible index
            if markerIndex in self.invisibleMarkerUndoIndex[markerType]:
                self.invisibleMarkerIndex[markerType].append(markerIndex)
                self.invisibleMarkerUndoIndex[markerType].remove(markerIndex)            

            # Restore hash map component
            del self.markerUndoHashMap[(markerType, markerCode)]
            self.markerHashMap[(markerType, markerCode)] = markerIndex

            # Rearrange hash map
            for key, value in self.markerHashMap.items():
                keyMarkerType, keyMarkerCode = key
                if (keyMarkerType == markerType and
                    keyMarkerCode > markerCode):
                    self.markerHashMap[key] += 1

            # Restore overview tree item
            overviewTreeItem = self.markerUndoItems.pop(-1)
            self.annotationTreeItem.insertChild(0, overviewTreeItem)
            
            # Change mark-up status
            self.bMarkUpUndo = False

            # Update markers
            markUpQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithMark(self.markers,
                                             self.invisibleMarkerIndex))
            self.axialQLabel.setPixmap(markUpQPixmap)

    def turnOnMarkUpUndo(self):
        if self.markUpHistory:
            # Read marking history
            history = self.markUpHistory.pop(-1)
            markerCode, markerIndex, markerType, marker = history
            self.markUpUndos.append((markerCode, 
                                     markerIndex, 
                                     markerType, 
                                     marker))
         
            # Remove marker
            self.markers[markerType].remove(marker)
           
            # Remove invisible index
            if markerIndex in self.invisibleMarkerIndex[markerType]:
                self.invisibleMarkerIndex[markerType].remove(markerIndex)
                self.invisibleMarkerUndoIndex[markerType].append(markerIndex)
           
            # Remove hash map component
            del self.markerHashMap[(markerType, markerCode)]
            self.markerUndoHashMap[(markerType, markerCode)] = markerIndex

            # Rearrange hash map
            for key, value in self.markerHashMap.items():
                keyMarkerType, keyMarkerCode = key
                if (keyMarkerType == markerType and 
                    keyMarkerCode > markerCode):
                    self.markerHashMap[key] -= 1  
         
            # Remove overview tree item
            overviewTreeItem = self.overviewTree.findItems(
                                           "Marker%s"%markerCode,
                                            Qt.MatchContains |
                                            Qt.MatchRecursive)[0]
            self.annotationTreeItem.removeChild(overviewTreeItem)
            self.markerUndoItems.append(overviewTreeItem)
 
            # Change mark-up status
            self.bMarkUpUndo = True

            # Update markers
            markUpQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithMark(self.markers,
                                             self.invisibleMarkerIndex))
            self.axialQLabel.setPixmap(markUpQPixmap)
