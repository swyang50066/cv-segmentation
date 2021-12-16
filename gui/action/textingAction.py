import  numpy       as np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *


class TextingAction(object):
    def __init__(self):
        super().__init__()
 
    def turnOnEditTextMessage(self, item, index):
        ''' Edit text message
        '''
        if index == 3 and item.text(1) == "Dot":
            # Enable editing
            item.setFlags(Qt.ItemIsEnabled |
                          Qt.ItemIsEditable |
                          Qt.ItemIsSelectable |
                          Qt.ItemIsUserCheckable)
        elif index != 3 and item.text(1) == "Dot":
            # Disable editing
            item.setFlags(Qt.ItemIsEnabled |
                          Qt.ItemIsSelectable |
                          Qt.ItemIsUserCheckable)
