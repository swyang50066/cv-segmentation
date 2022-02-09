import  os
import  sys

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  *
from    PyQt5.QtWidgets         import  *

from    window          import  Interface, Structure
from    action.action   import  Action


class Watermelon(Action,
                 Interface, 
                 Structure):
    def __init__(self):
        super().__init__()

    def execute(self):
        ''' Execute WATERMELON
        '''
        # Apply style sheet
        # UI design is to be updated later.

        # Run application
        self.window.show()
        sys.exit(self.application.exec_())


if __name__ == "__main__":
    watermelon = Watermelon()
    watermelon.execute()
