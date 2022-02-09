import  cv2

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *

from    window                      import  Interface
from    action.labelingAction       import  LabelingAction
from    action.markUpAction         import  MarkUpAction
from    action.histogramAction      import  HistogramAction
from    action.renderingAction      import  RenderingAction
from    action.fileIOAction         import  FileIOAction
from    action.mouseAction          import  MouseAction
from    action.overviewAction       import  OverviewAction
from    action.textingAction        import  TextingAction


class Action(
        TextingAction,
        LabelingAction,
        MarkUpAction,
        OverviewAction, 
        HistogramAction,
        RenderingAction,
        FileIOAction,
        MouseAction,
        Interface):
    def __init__(self):
        super().__init__()
