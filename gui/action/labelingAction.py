import  cv2
import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *

from    utils.markovRandomField         import  markovRandomField


class SegmentationAction(object):
    def __init__(self):
        super().__init__()

    def turnOnSegmentation(self):
        ''' Turn on segmentation function
        '''
        # Get forground and background maps
        foreground = 1.*(self.canvas.seed == 1)
        background = 1.*(self.canvas.seed == 2)

        # Run segmentation
        result = markovRandomField(
            self.canvas.image[self.canvas.axialIndex, ..., 0], 
            self.canvas.seed[self.canvas.axialIndex]
        )
        self.canvas.label[self.canvas.axialIndex] = result[0]
        self.canvas.segmentOverlay[self.canvas.axialIndex] = result[1]

        # Display segmentation result
        segmentQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSegment())
        self.segmentQLabel.setPixmap(segmentQPixmap)

        # Surface Rendering
        x = np.repeat(np.uint8(self.canvas.label == 1), repeats=32, axis=0)
        self.isoSurface.run(x, self.renderer)
        self.renderer.ResetCamera()
        self.interactor.Initialize()
        self.interactor.Start()


class LabelingAction(SegmentationAction):
    def __init__(self):
        super().__init__()

    def turnOnSetBlusherSize(self):
        ''' Set blusher size
        '''
        # Get blusher size
        self.canvas.blusherSize = self.blusherSlider.value()

        # Print blusher size
        self.blusherLabel.setText("Blusher Size: "
                                  + str(self.canvas.blusherSize))

    def turnOnUpdateSpecies(self):
        ''' Update species information
        '''
        # Get selected tree item
        treeItems = self.labelTree.selectedItems()
      
        # Check on selection
        if not treeItems: 
            return 0
        else:
            treeItem = treeItems[0]

        # Update overview status
        overviewTreeItem = self.overviewTree.findItems(
                                           "Species%s"%treeItem.text(1),
                                            Qt.MatchContains |
                                            Qt.MatchRecursive)[0]
        overviewTreeItem.setText(1, treeItem.text(0))

        # Update seed information
        self.canvas.typeSeed = int(treeItem.text(1))
        self.canvas.colorSeed[self.canvas.typeSeed] = tuple(
            [int(c) for c in treeItem.text(2).split(", ")])

    def turnOnAddSpecies(self):
        ''' Add new labeling species
        '''
        # Find class of latest species 
        rowCount = self.labelTree.topLevelItemCount()
        bottomItem = self.labelTree.topLevelItem(rowCount-1)
        bottomCls = int(bottomItem.text(1))

        # Update tree item
        newTreeItem = QTreeWidgetItem(self.labelTree)
        newTreeItem.setText(0, "No Named")
        newTreeItem.setText(1, str(bottomCls+1))
        newTreeItem.setText(2, "0, 0, 0")
        newTreeItem.setFlags(Qt.ItemIsSelectable |
                             Qt.ItemIsEnabled | 
                             Qt.ItemIsEditable)
   
        # Update overview            
        overviewTreeItem = QTreeWidgetItem()
        overviewTreeItem.setText(0, "Species%s"%str(bottomCls+1))
        overviewTreeItem.setText(1, "No Named")
        overviewTreeItem.setText(2, "Class=%s"%str(bottomCls+1))
        self.annotationTreeItem.insertChild(0, overviewTreeItem)
 
    def turnOnRemoveSpecies(self):
        ''' Remove selected labeling species
        '''
        # Get index of selected row
        currentRow = self.labelTree.currentIndex().row()
        currentItem = self.labelTree.currentItem()        

        # Prune tree widget
        if currentRow > 1:    # Hold fore/background at least
            # Update overview status
            overviewTreeItem = self.overviewTree.findItems(
                                            "Class=%s"%currentItem.text(1), 
                                            Qt.MatchContains |
                                            Qt.MatchRecursive,
                                            2)[0]
            self.annotationTreeItem.removeChild(overviewTreeItem)

            # Remove tree item
            self.labelTree.takeTopLevelItem(currentRow)

    def turnOnCopySeed(self):
        ''' Set label with manual labeling
        '''
        # Copy seeds to label
        self.canvas.label = self.canvas.seed.copy()
        self.canvas.segmentOverlay = self.canvas.seedOverlay.copy()

        # Update overview status
        self.segmentTreeItem.setFlags(Qt.ItemIsSelectable |
                                      Qt.ItemIsEnabled)
        overviewTreeItem = QTreeWidgetItem()
        overviewTreeItem.setText(0, "Manual")
        overviewTreeItem.setText(1, "Aorta")
        self.segmentTreeItem.takeChildren()
        self.segmentTreeItem.insertChild(0, overviewTreeItem)

        # Display segment
        segmentQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSegment())
        self.segmentQLabel.setPixmap(segmentQPixmap)

    def turnOnClearSeed(self):
        ''' Turn on seed overlay reset
        '''
        # Clear seed overlay
        self.canvas.clearSeed()

        # Display clean seed overlay
        axialQPixmap = self.canvas.getQPixmap(
                                self.canvas.getImageWithSeed())
        self.axialQLabel.setPixmap(axialQPixmap)
