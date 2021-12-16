from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *


class OverviewAction(object):
    def __init__(self):
        super().__init__()

    def turnOnVisibleMarker(self, a):
        ''' Set state of visible markers
        '''
        # Search annotation items
        numChild = self.annotationTreeItem.childCount()
        for index in range(numChild):
            markUpItem = self.annotationTreeItem.child(index)
          
            # Check 'Marker' item status
            if "Marker" in markUpItem.text(0): 
                # Get marker information
                markerType = markUpItem.text(1).lower()
                markerCode = int(markUpItem.text(0).split("Marker")[-1])
                markerHashMapKey = (markerType, markerCode)
                markerIndex = self.markerHashMap[markerHashMapKey]

                if (markUpItem.checkState(2) == 0 and
                    markerIndex not in 
                    self.invisibleMarkerIndex[markerType]):    # off state
                    # Add invisible marker
                    self.invisibleMarkerIndex[markerType].append(markerIndex)

                    # Update item status
                    markUpItem.setText(2, "Off")
                elif (markUpItem.checkState(2) == 2 and
                      markerIndex in
                      self.invisibleMarkerIndex[markerType]):    # on state
                    # Add visible marker
                    self.invisibleMarkerIndex[markerType].remove(markerIndex)

                    # Update item status
                    markUpItem.setText(2, "On")

                # Update texting
                if markerType == "dot":
                    # Read instance
                    axialIndex, (pivot, text) = self.markers["dot"][markerIndex]
                    
                    # Substitute instance
                    instance = (axialIndex, (pivot, markUpItem.text(3)))
                    self.markers["dot"][markerIndex] = instance

                    # Update history
                    i = self.markUpHistory.index((markerCode,
                                                  markerIndex,
                                                  markerType,
                                                  (axialIndex, (pivot, text))))
                    self.markUpHistory[i] = (markerCode,
                                             markerIndex,
                                             markerType,
                                             instance)

 
                # Update markers
                markUpQPixmap = self.canvas.getQPixmap(
                    self.canvas.getImageWithMark(self.markers, 
                                                 self.invisibleMarkerIndex))
                self.axialQLabel.setPixmap(markUpQPixmap)

    def turnOnRemoveMarker(self):
        ''' Remove marker
        '''
        # Get selected tree item
        markUpItems = self.overviewTree.selectedItems()

        # Check on selection
        if not markUpItems:
            return 0
        else:
            markUpItem = markUpItems[0]

        # Check 'Marker' item status
        if "Marker" in markUpItem.text(0):
            # Get marker information
            markerType = markUpItem.text(1).lower()
            markerCode = int(markUpItem.text(0).split("Marker")[-1])
            markerHashMapKey = (markerType, markerCode)
            markerIndex = self.markerHashMap[markerHashMapKey]
            
            # Remove mark-up item
            self.annotationTreeItem.removeChild(markUpItem)

            # Remove history
            self.markUpHistory.remove((markerCode, 
                                       markerIndex, 
                                       markerType, 
                                       self.markers[markerType][markerIndex]))
            # Rearrange history
            for k, history in enumerate(self.markUpHistory):
                if (history[1] > markerIndex and 
                    history[2] == markerType):
                    self.markUpHistory[k] = (history[0], 
                                             history[1]-1, 
                                             history[2],
                                             history[3])

            # Remove marker
            del self.markers[markerType][markerIndex]
            
            # Remove invisible index
            invisibles = self.invisibleMarkerIndex[markerType]
            if markerIndex in invisibles:
                for k, index in enumerate(invisibles):
                    if index == markerIndex:
                        self.invisibleMarkerIndex[markerType].remove(index)
                    elif index > markerIndex:
                        self.invisibleMarkerIndex[markerType][k] -= 1

            # Remove hash map component
            del self.markerHashMap[markerHashMapKey]

            # Rearrange hash map
            for key, value in self.markerHashMap.items():
                keyMarkerType, keyMarkerCode = key
                if (keyMarkerType == markerType and 
                    keyMarkerCode > markerHashMapKey[1]):
                    self.markerHashMap[key] -= 1                    

            # Update markers
            markUpQPixmap = self.canvas.getQPixmap(
                self.canvas.getImageWithMark(self.markers,
                                             self.invisibleMarkerIndex))
            self.axialQLabel.setPixmap(markUpQPixmap)
