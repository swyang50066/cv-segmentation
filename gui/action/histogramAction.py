import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *


class HistogramAction(object):
    def __init__(self):
        super().__init__()
        # Boolean for local histogram
        self.bLocalHistogram = False
       
        self.prevMarkUpTag = None
        
        # Local histogram paramters
        self.localHistogramCount = 0
        self.localBox = []

    def turnOnThresholding(self):
        ''' Apply thresholding for non-zero images
        '''
        # Reset label and segment overlay
        self.canvas.label = np.zeros_like(self.canvas.label)
        self.canvas.segmentOverlay = np.zeros_like(
                                            self.canvas.segmentOverlay)

        # Update label
        imslice = self.canvas.image[..., 0].copy()
        self.canvas.label = np.uint8((imslice >= self.canvas.valmin) &
                                     (imslice <= self.canvas.valmax))

        # Update overview status
        self.segmentTreeItem.setFlags(Qt.ItemIsSelectable |
                                      Qt.ItemIsEnabled)
        overviewTreeItem = QTreeWidgetItem()
        overviewTreeItem.setText(0, "Thresholding")
        overviewTreeItem.setText(1, "Aorta")
        self.segmentTreeItem.takeChildren()
        self.segmentTreeItem.insertChild(0, overviewTreeItem)
        
        # Write segment overlay
        self.canvas.segmentOverlay[..., 1] = 255*self.canvas.label

        # Display image
        segmentQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSegment())
        self.segmentQLabel.setPixmap(segmentQPixmap)

        # Update python console variables
        variables = {"label": self.canvas.label}
        self.ipyConsole.pushVariables(variables)

    def turnOnResetHistogram(self):
        ''' Reset histogram parameter
        '''
        # Reset parameters
        self.canvas.valmin = 0
        self.canvas.valmax = 255
        self.canvas.gamma = 1.

        # Shift range
        self.histogramView.shiftRange(self.canvas.valmin,
                                      self.canvas.valmax,
                                      self.histCanvas,
                                      self.histAxes)

        # Reset texts
        self.lowerLine.setText("0")
        self.upperLine.setText("255")
        self.meanText.setText("0")
        self.stdText.setText("0")
        self.gammaText.setText(str(self.canvas.gamma))
        self.percentText.setText("100 %")

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

    def turnOnSetHistogram(self):
        ''' Set histogram with line edits
        '''
        if self.lowerLine.text() and self.upperLine.text():
            # Get upper & lower limits 
            lowerLimit, upperLimit = (int(self.lowerLine.text()),
                                      int(self.upperLine.text()))
            
            # Prevent overlap between limits
            if lowerLimit > upperLimit:
                lowerLimit = upperLimit
                self.lowerLine.setText(str(lowerLimit)) 
            
            # Set intensity limits
            self.canvas.valmin = lowerLimit
            self.canvas.valmax = upperLimit

            # Update histogram information
            volume = self.canvas.image[
                            (self.canvas.image >= lowerLimit) &
                            (self.canvas.image <= upperLimit)]
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

    def turnOnLocalHistogram(self):
        ''' Turn on local histogram
        '''
        # Change button status
        if self.bLocalHistogram:
            # Turn on mark-up
            if self.prevMarkUpTag:
                # Reset mark-up tag
                self.markUpTag = self.prevMarkUpTag
                self.prevMarkUpTag = None

                # Display mark-up
                self.turnOnMarkUpTag(self.markUpTag)
            else:
                # Display image
                axialQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSeed())
                self.axialQLabel.setPixmap(axialQPixmap)

            # Reset status
            self.localBox = []
            self.bLocalHistogram = False
            self.localHistButton.setStyleSheet("background-color: white")

            # Reset histogram
            self.histogramView.initialize(self.canvas.image,
                                          self.histCanvas,
                                          self.histAxes)

            # Update histogram information
            self.lowerLine.setText("0")
            self.upperLine.setText("255")
            self.meanText.setText(
                str(np.mean(self.canvas.image).astype(np.uint8)))
            self.stdText.setText(
                str(np.std(self.canvas.image).astype(np.uint8)))
            self.gammaText.setText(str(self.canvas.gamma))
            self.percentText.setText("100 %")
        else:
            # Turn off mark-up
            if self.markUpTag:
                # Reset mark-up tag
                self.prevMarkUpTag = self.markUpTag
                self.markUpTag = None

                # Display mark-up
                self.turnOnMarkUpTag("")

            # Set status
            self.bLocalHistogram = True
            self.localHistButton.setStyleSheet("background-color: gray")

