import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *


class RenderingAction(object):
    def __init__(self):
        super().__init__()
        
    def turnOnSTLRendering(self):
        ''' Turn on stl rendering
        '''
        # Switch button status
        self.STLRenderingButton.setStyleSheet("background-color: gray")

        # Load volume        
        mark = self.canvas.label

        # Update renderer
        self.renderer = self.isoSurface.run(mark, self.renderer)

        # Run interactor
        self.renderer.ResetCamera()
        self.interactor.Initialize()
        self.interactor.Start()
