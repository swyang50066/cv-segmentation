import  os
import  sys

import  numpy       as  np

from    PyQt5.QtCore            import  *
from    PyQt5.QtGui             import  * 
from    PyQt5.QtWidgets         import  *

from    interpreter.IPythonConsole              import  QIPythonWidget

from    matplotlib.figure                       import Figure
from    matplotlib.backends.backend_qt5agg      import FigureCanvas

import  vtk
from    vtk.qt.QVTKRenderWindowInteractor       import  QVTKRenderWindowInteractor

from    visualizer.canvas       import  Canvas
from    visualizer.graph        import  Variable, Histogram
from    visualizer.recon        import  IsoSurface, IsoLine


class Structure(object):
    def __init__(self):
        # Declare application 
        self.application = QApplication(sys.argv)
        
        # Declare main window
        self.window = QMainWindow()
        self.window.setWindowTitle("WATERMELON")
  
        # Declare window menu bar
        self.mainMenuBar = self.window.menuBar()

        # Declare main widget
        self.widget = QWidget()
      
        # Set up main menu
        self.setupMainMenu()
        
    def setupMainMenu(self):
        ''' Setup file menu in menu bar
        ''' 
        # Build file menu
        fileMenu = self.mainMenuBar.addMenu("&File")

        # Open button
        openButton = QAction(QIcon("exit.png"), 
                             "Open Image", 
                             self.window)
        openButton.setShortcut("Ctrl+O")
        openButton.setStatusTip("Open a image file for segmentation")
        openButton.triggered.connect(self.openFile)
        fileMenu.addAction(openButton)

        # Save button
        saveButton = QAction(QIcon("exit.png"), 
                             "Save Image", 
                             self.window)
        saveButton.setShortcut("Ctrl+S")
        saveButton.setStatusTip("Save result")
        saveButton.triggered.connect(self.saveFile)
        fileMenu.addAction(saveButton)
        
        # Save STL button
        saveSTLButton = QAction(QIcon("exit.png"), 
                                "Save STL", 
                                self.window)
        saveSTLButton.setShortcut("Ctrl+L")
        saveSTLButton.setStatusTip("Save STL")
        saveSTLButton.triggered.connect(self.saveSTL)
        fileMenu.addAction(saveSTLButton)

        # Close button
        closeButton = QAction(QIcon("exit.png"), 
                              "Exit", 
                              self.window)
        closeButton.setShortcut("Ctrl+Q")
        closeButton.setStatusTip("Exit application")
        closeButton.triggered.connect(self.closeWindow)
        fileMenu.addAction(closeButton)


class Interface(Structure):
    def __init__(self):
        super().__init__()
        # Declare visualizer classes
        self.canvas = Canvas()
        self.variableView = Variable()
        self.histogramView = Histogram()
        self.isoSurface = IsoSurface()
        self.isoLine = IsoLine()

        # Setup main box layout
        mainGridLayout = QGridLayout()
 
        # Build layouts
        topToolkitLayout = self.setupTopToolkitLayout()
        fileToolkitLayout = self.setupFileToolkitLayout()
        graphicWindowLayout = self.setupGraphicLayout()
        sideToolkitLayout = self.setupSideToolkitLayout()
        pythonConsoleLayout = self.setupPythonConsoleLayout() 

        # Build interface
        mainGridLayout.addLayout(topToolkitLayout, 0, 0, 1, 2)
        mainGridLayout.addLayout(fileToolkitLayout, 1, 0, 2, 1)
        mainGridLayout.addLayout(graphicWindowLayout, 1, 1)
        mainGridLayout.addLayout(sideToolkitLayout, 0, 2, 3, 1)
        mainGridLayout.addLayout(pythonConsoleLayout, 2, 1)

        # Add layouts to main frame 
        self.widget.setLayout(mainGridLayout)
        self.window.setCentralWidget(self.widget)

    def setupTopToolkitLayout(self):
        ''' Set up top toolkit layout
        '''
        # Set top-toolkit layout
        topToolkitLayout = QHBoxLayout()
        topToolkitFrame = QFrame()
        topToolkitFrameLayout = QHBoxLayout()
        topToolkitFrame.setFrameShape(QFrame.Panel)

        # Do-Undo
        doUndoGroupBox = QGroupBox("DO-UNDO")
        doUndoLayout = QHBoxLayout()
        
        markUpDoButton = QPushButton("DO")
        markUpDoButton.setStyleSheet("background-color: white")
        markUpDoButton.clicked.connect(self.turnOnMarkUpDo)         
 
        markUpUndoButton = QPushButton("UNDO")
        markUpUndoButton.setStyleSheet("background-color: white")
        markUpUndoButton.clicked.connect(self.turnOnMarkUpUndo)     

        # Add widgets to layout
        doUndoLayout.addWidget(markUpDoButton)
        doUndoLayout.addWidget(markUpUndoButton)
        doUndoGroupBox.setLayout(doUndoLayout)

        # Drawing toolkit
        drawingGroupBox = QGroupBox("Drawing Toolkit")
        drawingLayout = QHBoxLayout()
       
        # Mark-Up buttons
        dotMarkUpButton = QPushButton("Dot")
        dotMarkUpButton.setStyleSheet("background-color: white")
        dotMarkUpButton.clicked.connect(
                                lambda: self.turnOnMarkUpTag("dot"))

        lineMarkUpButton = QPushButton("Line")
        lineMarkUpButton.setStyleSheet("background-color: white")
        lineMarkUpButton.clicked.connect(
                                lambda: self.turnOnMarkUpTag("line"))

        arcMarkUpButton = QPushButton("Arc")
        arcMarkUpButton.setStyleSheet("background-color: white")
        arcMarkUpButton.clicked.connect(
                                lambda: self.turnOnMarkUpTag("arc"))

        boxMarkUpButton = QPushButton("Box")
        boxMarkUpButton.setStyleSheet("background-color: white")
        boxMarkUpButton.clicked.connect(
                                lambda: self.turnOnMarkUpTag("box"))

        curveMarkUpButton = QPushButton("Curve")
        curveMarkUpButton.setStyleSheet("background-color: white")
        curveMarkUpButton.clicked.connect(
                                lambda: self.turnOnMarkUpTag("curve"))
        
        # Contain mark-up buttons
        self.markUpButtons = {'dot': dotMarkUpButton,
                              'line': lineMarkUpButton,
                              'arc': arcMarkUpButton,
                              'box': boxMarkUpButton,
                              'curve': curveMarkUpButton}
        
        # Add widgets to layout
        drawingLayout.addWidget(dotMarkUpButton)
        drawingLayout.addWidget(lineMarkUpButton)
        drawingLayout.addWidget(arcMarkUpButton)
        drawingLayout.addWidget(boxMarkUpButton)
        drawingLayout.addWidget(curveMarkUpButton)
        drawingGroupBox.setLayout(drawingLayout)

        # Texting toolkit
        textingGroupBox = QGroupBox("Texting")
        textingLayout = QHBoxLayout()

        # Texting button
        boldFontButton = QPushButton("Bold")
        boldFontButton.setStyleSheet("background-color: white")

        tiltedFontButton = QPushButton("Tilted")
        tiltedFontButton.setStyleSheet("background-color: white")

        underlineButton = QPushButton("Underline")
        underlineButton.setStyleSheet("background-color: white")

        fontSizeSpinBox = QSpinBox()
        fontSizeSpinBox.setRange(8, 16)
        fontSizeSpinBox.setSingleStep(2)

        # Add widgets to layout
        textingLayout.addWidget(boldFontButton)
        textingLayout.addWidget(tiltedFontButton)
        textingLayout.addWidget(underlineButton)
        textingLayout.addWidget(fontSizeSpinBox)
        textingGroupBox.setLayout(textingLayout)

        # Set top toolkit layout
        topToolkitFrameLayout.addWidget(doUndoGroupBox)
        topToolkitFrameLayout.addWidget(drawingGroupBox)
        topToolkitFrameLayout.addWidget(textingGroupBox)
        topToolkitFrameLayout.addStretch(1)
        topToolkitFrame.setLayout(topToolkitFrameLayout)
        topToolkitLayout.addWidget(topToolkitFrame)

        return topToolkitLayout        

    def setupFileToolkitLayout(self):
        ''' Set up file toolkit layout
        '''
        # File toolkit layout
        fileToolkitLayout = QVBoxLayout()
        fileToolkitFrame = QFrame()
        fileToolkitFrameLayout = QVBoxLayout()
        fileToolkitFrame.setFrameShape(QFrame.Panel)        

        # Open file
        openDICOMButton = QPushButton("Import DICOM")
        openDICOMButton.setStyleSheet("background-color: white")
        openDICOMButton.clicked.connect(self.openDICOM)    
        
        # Open DICOM
        openBMPButton = QPushButton("Import .bmp")
        openBMPButton.setStyleSheet("background-color: white")
        openBMPButton.clicked.connect(self.openFile)    
        
        # Save file
        saveButton = QPushButton("Export .bmp")
        saveButton.setStyleSheet("background-color: white")
        saveButton.clicked.connect(self.saveFile)    
        
        # Save STL
        saveSTLButton = QPushButton("Export STL")
        saveSTLButton.setStyleSheet("background-color: white")
        saveSTLButton.clicked.connect(self.saveSTL)    
    
        # Help
        helpButton = QPushButton("Help (Q&&A)")
        helpButton.setStyleSheet("background-color: white")

        # Add widgets to group box
        fileToolkitFrameLayout.addWidget(openDICOMButton)
        fileToolkitFrameLayout.addWidget(openBMPButton)
        fileToolkitFrameLayout.addWidget(saveButton)
        fileToolkitFrameLayout.addWidget(saveSTLButton)
        fileToolkitFrameLayout.addStretch(1)
        fileToolkitFrameLayout.addWidget(helpButton)
        fileToolkitFrame.setLayout(fileToolkitFrameLayout)

        # Add widgets to layout
        fileToolkitLayout.addWidget(fileToolkitFrame)

        return fileToolkitLayout        

    def setupGraphicLayout(self):
        ''' Set up graphic windows layout
        '''
        # Setup Image Area 
        graphicWindowLayout = QGridLayout()

        # Get tab widget
        viewerTab = self.setupViewerLayout()
        reconTab = self.setupReconLayout()

        # Addtional scroll bar
        self.scrollBar = QScrollBar()
        self.scrollBar.setPageStep(1)
        self.scrollBar.setMaximum(len(self.canvas.image)-1)
        self.scrollBar.valueChanged.connect(self.wheelSideScroll)
 
        # Add widgets to layout
        graphicWindowLayout.addWidget(viewerTab, 0, 0)
        graphicWindowLayout.addWidget(reconTab, 0, 1)
        ####graphicWindowLayout.addWidget(self.scrollBar, 0, 2)

        return graphicWindowLayout

    def setupViewerLayout(self):
        ''' Setup viewer layout
        '''
        # Seed seed pixmap
        self.axialQLabel = QLabel()
        self.axialQLabel.setStyleSheet("background-color: black")
        self.axialQLabel.setScaledContents(True)
        
        axialQPixmap = self.canvas.getQPixmap(
                                self.canvas.getImageWithSeed())
        self.axialQLabel.setPixmap(axialQPixmap)
        
        # Add mouse events on seed pixel map
        self.axialQLabel.setMouseTracking(True)
        self.axialQLabel.mousePressEvent = self.mouseAxialPress
        self.axialQLabel.mouseReleaseEvent = self.mouseAxialRelease
        self.axialQLabel.mouseMoveEvent = self.mouseAxialMove
        self.axialQLabel.wheelEvent = self.wheelAxialScroll
 
        # Seed scroll area
        self.axialScrollArea = QScrollArea()
        self.axialScrollArea.setStyleSheet("background-color: black")
        self.axialScrollArea.setWidget(self.axialQLabel)
        self.axialScrollArea.setAlignment(Qt.AlignCenter)
        self.axialScrollArea.horizontalScrollBar().setEnabled(False)
        self.axialScrollArea.horizontalScrollBar().setStyleSheet(
            "height: 0px")
        self.axialScrollArea.verticalScrollBar().setEnabled(False)
        self.axialScrollArea.verticalScrollBar().setStyleSheet(
            "width: 0px")
        
        # Coronal view pixmap
        self.coronalQLabel = QLabel()
        self.coronalQLabel.setStyleSheet("background-color: black")
        self.coronalQLabel.setAlignment(Qt.AlignCenter)
        self.coronalQLabel.wheelEvent = self.wheelCoronalScroll
        
        coronalQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithCoronal())
        self.coronalQLabel.setPixmap(coronalQPixmap)

        # Sagittal view pixmap
        self.sagittalQLabel = QLabel()
        self.sagittalQLabel.setStyleSheet("background-color: black")
        self.sagittalQLabel.setAlignment(Qt.AlignCenter)
        self.sagittalQLabel.wheelEvent = self.wheelSagittalScroll
        
        sagittalQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSagittal())
        self.sagittalQLabel.setPixmap(sagittalQPixmap)

        # Tab menu
        viewerTab = QTabWidget()
        viewerTab.addTab(self.axialScrollArea, "AXIAL")
        viewerTab.addTab(self.coronalQLabel, "CORONAL")
        viewerTab.addTab(self.sagittalQLabel, "SAGITTAL")

        return viewerTab

    def setupReconLayout(self):
        ''' Set up reconstruction layout
        '''
        # Segmentation view pixmap
        self.segmentQLabel = QLabel()
        self.segmentQLabel.setStyleSheet("background-color: black")
        self.segmentQLabel.setAlignment(Qt.AlignCenter)
        
        segmentQPixmap = self.canvas.getQPixmap(
                                    self.canvas.getImageWithSegment())
        self.segmentQLabel.setPixmap(segmentQPixmap)

        # Initialize 3D Rendering view
        self.frame = QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.interactor.Initialize()
        self.interactor.Start()
        
        # Tab menu
        reconTab = QTabWidget()
        reconTab.addTab(self.segmentQLabel, "SEGMENTATION")
        reconTab.addTab(self.vtkWidget, "RENDERING")

        return reconTab

    def setupSideToolkitLayout(self):
        ''' Set up seed toolkit layout
        '''
        # Setup Mode Buttons 
        sideToolkitLayout = QGridLayout()
        
        # Get layouts
        overviewLayout = self.setupOverviewLayout()
        propertyLayout = self.setupPropertyLayout()
        
        # Set widgets
        overviewWidget = QWidget()
        overviewWidget.setLayout(overviewLayout)
        overviewWidget.setStyleSheet("background-color: white")

        propertyWidget = QWidget()
        propertyWidget.setLayout(propertyLayout)
        propertyWidget.setStyleSheet("background-color: white")
        
        # Tab menu
        sideToolkitTab = QTabWidget()
        sideToolkitTab.addTab(overviewWidget, "OVERVIEW")
        sideToolkitTab.addTab(propertyWidget, "PROPERTY")
        sideToolkitTab.setFixedWidth(384)

        # Add widget
        sideToolkitLayout.addWidget(sideToolkitTab, 0, 0)

        return sideToolkitLayout
    
    def setupOverviewLayout(self):
        ''' Set up overview layout
        '''        
        # Set overview layout
        overviewLayout = QVBoxLayout()
       
        # Tree widget
        self.overviewTree = QTreeWidget()
        self.overviewTree.setColumnCount(2)
        self.overviewTree.setHeaderLabels(["Step",
                                           "Attributes",
                                           "Status",
                                           "Value"])
        self.overviewTree.itemChanged.connect(
                                        self.turnOnVisibleMarker)
        self.overviewTree.itemDoubleClicked.connect(
                                        self.turnOnEditTextMessage)

        
        # Add tree menu actions
        removeMarker = QAction("Delete",
                               self.overviewTree)
        removeMarker.triggered.connect(self.turnOnRemoveMarker)
        
        self.overviewTree.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.overviewTree.addAction(removeMarker)

        # Set default tree items
        self.annotationTreeItem = QTreeWidgetItem(self.overviewTree)
        self.annotationTreeItem.setText(0, "Annotation")

        child1TreeItem = QTreeWidgetItem()
        child1TreeItem.setText(0, "Species1")
        child1TreeItem.setText(1, "Aorta")
        self.annotationTreeItem.insertChild(0, child1TreeItem)

        child2TreeItem = QTreeWidgetItem()
        child2TreeItem.setText(0, "Species2")
        child2TreeItem.setText(1, "Background")
        self.annotationTreeItem.insertChild(0, child2TreeItem)

        self.segmentTreeItem = QTreeWidgetItem(self.overviewTree)
        self.segmentTreeItem.setText(0, "Segmentation")
        self.segmentTreeItem.setFlags(Qt.ItemIsSelectable)

        # Grid layout
        gridWidget = QWidget()
        gridLayout = QGridLayout()

        # Species tree window
        self.labelTree = QTreeWidget()
        self.labelTree.setColumnCount(3)
        self.labelTree.setHeaderLabels(["Species",
                                        "Class",
                                        "BGR"])
        self.labelTree.itemChanged.connect(
                                    self.turnOnUpdateSpecies)
        self.labelTree.itemSelectionChanged.connect(
                                    self.turnOnUpdateSpecies)

        # Set default species
        foregroundTreeItem = QTreeWidgetItem(self.labelTree)
        foregroundTreeItem.setText(0, "Aorta")
        foregroundTreeItem.setText(1, "1")
        foregroundTreeItem.setText(2, "0, 255, 0")

        backgroundTreeItem = QTreeWidgetItem(self.labelTree)
        backgroundTreeItem.setText(0, "Background")
        backgroundTreeItem.setText(1, "2")
        backgroundTreeItem.setText(2, "0, 0, 255")

        # Add species
        addSpeciesButton = QPushButton("Add Species")
        addSpeciesButton.clicked.connect(self.turnOnAddSpecies)

        # Remove species
        removeSpeciesButton = QPushButton("Remove Species")
        removeSpeciesButton.clicked.connect(self.turnOnRemoveSpecies)

        # Set layers
        gridLayout.addWidget(self.labelTree, 0, 0, 1, 2)
        gridLayout.addWidget(addSpeciesButton, 1, 0, 1, 1)
        gridLayout.addWidget(removeSpeciesButton, 1, 1, 1, 1)
        gridWidget.setLayout(gridLayout)


        # Dicom attriute table
        self.dicomTable = QTableWidget()
        self.dicomTable.setRowCount(12)
        self.dicomTable.setColumnCount(1)
        self.dicomTable.horizontalHeader().setStretchLastSection(True)
        self.dicomTable.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Dicom table shape        
        self.dicomTable.resizeRowsToContents()
        self.dicomTable.resizeColumnsToContents()
        self.dicomTable.setMinimumHeight(256)
        self.dicomTable.verticalScrollBar().setStyleSheet(
            "width: 0px")

        self.dicomTable.setHorizontalHeaderLabels(["Attribute"])
        self.dicomTable.setVerticalHeaderLabels(["Name",
                                                 "ID",
                                                 "Age",
                                                 "Birth Date",
                                                 "Sex",
                                                 "Image Type",
                                                 "Modality",
                                                 "Manufacturer",
                                                 "KVP",
                                                 "Study Description",
                                                 "Series Description",
                                                 "Spacing (xy/z)"])
        
        # Add widgets to layout
        overviewLayout.addWidget(self.overviewTree)
        overviewLayout.addWidget(gridWidget)
        overviewLayout.addStretch(1)
        overviewLayout.addWidget(self.dicomTable)

        return overviewLayout

    def setupPropertyLayout(self):
        ''' Set up property layout
        '''        
        # Set property layout
        propertyLayout = QVBoxLayout()

        # Histogram
        self.histCanvas = FigureCanvas(Figure(figsize=(2, 2),
                                              tight_layout=True))
        self.histAxes = self.histCanvas.figure.subplots()

        self.histogramView.initialize(self.canvas.image,
                                      self.histCanvas,
                                      self.histAxes)
        self.histCanvas.mpl_connect('button_press_event', 
                                            self.mouseHistPress)
        self.histCanvas.mpl_connect('button_release_event', 
                                            self.mouseHistRelease)
        self.histCanvas.mpl_connect('motion_notify_event', 
                                            self.mouseHistMove)
        self.histCanvas.mpl_connect('scroll_event', 
                                            self.wheelHistScroll)

        # Set histogram widget
        histogramGroupBox = QGroupBox()
        histogramLayout = QGridLayout()
       
        # Histogram information labels 
        lineText1 = QLabel()
        lineText2 = QLabel()
        lineText3 = QLabel()
        lineText4 = QLabel()
        lineText5 = QLabel()
        lineText6 = QLabel()
        lineText7 = QLabel()
        lineText1.setText("LEVEL")
        lineText2.setText("Min")
        lineText2.setFont(QFont("Times", 10 , QFont.Bold))
        lineText3.setText("Max")
        lineText3.setFont(QFont("Times", 10 , QFont.Bold))
        lineText4.setText("Mean")
        lineText4.setFont(QFont("Times", 10 , QFont.Bold))
        lineText5.setText("Std")
        lineText5.setFont(QFont("Times", 10 , QFont.Bold))
        lineText6.setText("Gamma")
        lineText6.setFont(QFont("Times", 10 , QFont.Bold))
        lineText7.setText("PERCENTAGE")
        
        # Histogram information values
        self.lowerLine = QLineEdit()
        self.upperLine = QLineEdit()
        self.meanText = QLabel()
        self.stdText = QLabel()
        self.gammaText = QLabel()
        self.percentText = QLabel()
        
        self.lowerLine.setText("0")
        self.upperLine.setText("255")
        self.meanText.setText(
            str(np.mean(self.canvas.image).astype(np.uint8)))
        self.stdText.setText(
            str(np.std(self.canvas.image).astype(np.uint8)))
        self.gammaText.setText(str(self.canvas.gamma))
        self.percentText.setText("100 %")
        
        self.lowerLine.textEdited.connect(self.turnOnSetHistogram)
        self.upperLine.textEdited.connect(self.turnOnSetHistogram)

        # Build histogram toolkit
        histogramLayout.addWidget(lineText1, 0, 0)
        histogramLayout.addWidget(lineText2, 0, 1)
        histogramLayout.addWidget(lineText3, 0, 3)
        histogramLayout.addWidget(lineText4, 1, 1)
        histogramLayout.addWidget(lineText5, 2, 1)
        histogramLayout.addWidget(lineText6, 3, 1)
        histogramLayout.addWidget(lineText7, 3, 0)
        
        # Horizontal buttons
        gridWidget = QWidget()
        gridLayout = QGridLayout()

        # Set reset button
        resetHistButton = QPushButton("Reset Parameters")
        resetHistButton.setStyleSheet("background-color: white")
        resetHistButton.clicked.connect(self.turnOnResetHistogram)

        # Set thresholding
        thresholdingButton = QPushButton("Thresholding")
        thresholdingButton.setStyleSheet("background-color: white")
        thresholdingButton.clicked.connect(self.turnOnThresholding)
        
        # Set local histogram
        self.localHistButton = QPushButton("Local Histogram")
        self.localHistButton.setStyleSheet("background-color: white")
        self.localHistButton.clicked.connect(self.turnOnLocalHistogram)

        # Set layouts
        gridLayout.addWidget(resetHistButton, 0, 0, 1, 1)
        gridLayout.addWidget(thresholdingButton, 0, 1, 1, 1)
        gridLayout.addWidget(self.localHistButton, 1, 0, 1, 2)
        gridWidget.setLayout(gridLayout)
        
        histogramLayout.addWidget(self.lowerLine, 0, 2)
        histogramLayout.addWidget(self.upperLine, 0, 4)
        histogramLayout.addWidget(self.meanText, 1, 2, 1, 3)
        histogramLayout.addWidget(self.stdText, 2, 2, 1, 3)
        histogramLayout.addWidget(self.gammaText, 3, 2, 1, 3)
        histogramLayout.addWidget(self.percentText, 4, 1, 1, 4)
        histogramLayout.addWidget(gridWidget, 5, 0, 1, 5)
        histogramGroupBox.setLayout(histogramLayout)

        # Grid layout
        gridWidget = QWidget()
        gridLayout = QGridLayout()

        # Species tree window
        self.labelTree = QTreeWidget()
        self.labelTree.setColumnCount(3)
        self.labelTree.setHeaderLabels(["Species",
                                        "Class",
                                        "BGR"])
        self.labelTree.itemChanged.connect(
                                    self.turnOnUpdateSpecies)
        self.labelTree.itemSelectionChanged.connect(
                                    self.turnOnUpdateSpecies)

        # Set default species
        foregroundTreeItem = QTreeWidgetItem(self.labelTree)
        foregroundTreeItem.setText(0, "Aorta")
        foregroundTreeItem.setText(1, "1")
        foregroundTreeItem.setText(2, "0, 255, 0")

        backgroundTreeItem = QTreeWidgetItem(self.labelTree)
        backgroundTreeItem.setText(0, "Background")
        backgroundTreeItem.setText(1, "2")
        backgroundTreeItem.setText(2, "0, 0, 255")

        # Add species
        addSpeciesButton = QPushButton("Add Species")
        addSpeciesButton.clicked.connect(self.turnOnAddSpecies)

        # Remove species
        removeSpeciesButton = QPushButton("Remove Species")
        removeSpeciesButton.clicked.connect(self.turnOnRemoveSpecies)

        # Set layers
        gridLayout.addWidget(self.labelTree, 0, 0, 1, 2)
        gridLayout.addWidget(addSpeciesButton, 1, 0, 1, 1)
        gridLayout.addWidget(removeSpeciesButton, 1, 1, 1, 1)
        gridWidget.setLayout(gridLayout)

        # Labeling label
        labelingGroupBox = QGroupBox("Labeling")
        labelingLayout = QVBoxLayout()

        horizonWidget = QWidget()
        horizonLayout = QHBoxLayout()

        # Blusher size selecting
        self.blusherLabel = QLabel("Blusher Size: " + "1")

        self.blusherSlider = QSlider(Qt.Horizontal)
        self.blusherSlider.setRange(1, 50)
        self.blusherSlider.setSingleStep(1)
        self.blusherSlider.setTickPosition(QSlider.TicksBelow)
        self.blusherSlider.valueChanged.connect(self.turnOnSetBlusherSize)

        # Feed seeds
        feedButton = QPushButton("Set Manual Labeling")
        feedButton.clicked.connect(self.turnOnCopySeed)

        # Clear seeds
        clearButton = QPushButton("Clear All Seeds")
        clearButton.setShortcut("Ctrl+Z")
        clearButton.clicked.connect(self.turnOnClearSeed)

        # Add widgets to layout
        horizonLayout.addWidget(feedButton)
        horizonLayout.addWidget(clearButton)
        horizonWidget.setLayout(horizonLayout)

        # Build labeling group box
        labelingLayout.addWidget(self.blusherLabel)
        labelingLayout.addWidget(self.blusherSlider)
        labelingLayout.addWidget(horizonWidget)
        labelingGroupBox.setLayout(labelingLayout)

        # Add widgets to layout
        propertyLayout.addWidget(self.histCanvas)
        propertyLayout.addWidget(histogramGroupBox)
        propertyLayout.addWidget(gridWidget)
        propertyLayout.addWidget(labelingGroupBox)
        propertyLayout.addStretch(1)
        
        return propertyLayout

    def setupPythonConsoleLayout(self):
        ''' Setup python console layout
        '''
        # Set python console layout
        pythonConsoleLayout = QHBoxLayout()
        pythonConsoleFrame = QFrame()
        pythonConsoleFrameLayout = QVBoxLayout()
        pythonConsoleFrame.setFrameShape(QFrame.Panel)

        # IPython console
        self.ipyConsole = QIPythonWidget(
                        customBanner="Welcome to the WATERMELON")
        self.ipyConsole.setMinimumHeight(192)

        # Set initial import
        self.ipyConsole.executeCommand(
                            "from interpreter.interpreter import *")
        self.ipyConsole.executeCommand(
                            "%matplotlib inline")

        # Add widgets to layout
        pythonConsoleFrameLayout.addWidget(self.ipyConsole)
        pythonConsoleFrame.setLayout(pythonConsoleFrameLayout)
        pythonConsoleLayout.addWidget(pythonConsoleFrame)

        return pythonConsoleLayout

