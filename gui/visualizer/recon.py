import  numpy       as  np

import  vtk         as  vtk

import  matplotlib


def getColormap(valmin, valmax, colorScale="jet", colorLevel=1024):
    ''' Returns a vtk look-up table of colormap 
        based on the specified matplotlib colorscale
    '''
    # Build vtk look-up table
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(valmin, valmax)
    lut.SetNumberOfColors(colorLevel)
    lut.Build()

    # Set table value
    cmap = matplotlib.cm.get_cmap(colorScale)
    for i in range(colorLevel):
        color = cmap(float(i) / float(colorLevel))
        lut.SetTableValue(i, color[0], color[1], color[2], 1.)
   
    # Set periodic table
    lut.SetUseBelowRangeColor(True)
    lut.SetUseAboveRangeColor(True)
    
    return lut


def convertArrayToVTKArray(array, dtype=None):
    ''' Get vtkFloatArray/vtkDoubleArray from the input ndarray
    '''
    # Specify data type
    if dtype == None:
        dtype = array.dtype
    
    if dtype == np.float32:
        vtkArray = vtk.vtkFloatArray()
    elif dtpe == np.float64:
        vtkArray = vtk.vtkDoubleArray()
    elif dtype == np.uint8:
        float_array = vtk.vtkUnsignedCharArray()
    elif dtype == np.int8:
        float_array = vtk.vtkCharArray()
    else:
        raise ValueError("Wrong format of input array, must be a float32/float64/uint8/int8")
    
    # Set component dimension
    if len(array.shape) == 2:
        vtkArray.SetNumberOfComponents(array.shape[1])
    elif len(array.shape) == 1:
        vtkArray.SetNumberOfComponents(1)
    else:
        raise ValueError("Wrong shape of array must be 1D or 2D.")
    
    contiguous = np.ascontiguousarray(array, dtype)
    vtkArray.SetVoidArray(contiguous, contiguous.size, 1)
    
    # Hack to keep the array from being garbage collected
    vtkArray._contiguous_array = contiguous  
    
    return vtkArray


def convertArrayToVTKImage(array, dtype=None):
    ''' Get vtkImageData from the input ndarray
    '''
    # Get vtk array
    vtkArray = convertArrayToVTKArray(array.flatten(), dtype)
    
    # Convert vtk array to vtk image
    vtkImage = vtk.vtkImageData()
    vtkImage.SetDimensions(*(array.shape[::-1]))
    vtkImage.GetPointData().SetScalars(vtkArray)

    return vtkImage


class IsoSurface(object):
    """Generate and plot isosurfacs."""
    def __init__(self, spacing=(1., 1., 1.)):
        self.spacing = spacing

    def run(self, volume, renderer):
        self._renderer = renderer
        
        self._float_array = vtk.vtkFloatArray()
        self._image_data = vtk.vtkImageData()
        self._image_data.GetPointData().SetScalars(self._float_array)
        self._setup_data(np.float32(volume))
        self._image_data.SetSpacing(
            self.spacing[2], self.spacing[1], self.spacing[0]
        )
    
        self._surface_algorithm = vtk.vtkMarchingCubes()
        self._surface_algorithm.SetInputData(self._image_data)
        self._surface_algorithm.ComputeNormalsOn()

        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(
            self._surface_algorithm.GetOutputPort()
        )
        self._mapper.ScalarVisibilityOn() # new
        
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

        self.set_renderer(self._renderer)
        self.set_level(0, .5)

    def _setup_data(self, volume):
        """Setup the relation between the numpy volume, the vtkFloatArray and vtkImageData."""
        self._volume_array = np.zeros(
            volume.shape, dtype="float32", order="C"
        )
        self._volume_array[:] = volume
        
        self._float_array.SetNumberOfValues(np.product(volume.shape))
        self._float_array.SetNumberOfComponents(1)
        self._float_array.SetVoidArray(
            self._volume_array, np.product(volume.shape), 1
        )
        self._image_data.SetDimensions(*(self._volume_array.shape[::-1]))

    def set_volume(self, volume):
        if volume.shape != self._volume_array.shape:
            raise ValueError(f"volume must have same shape as previsous volume: {self._volume_array.shape}")
        self._volume_array[:] = volume
        self._float_array.SetVoidArray(
            self._volume_array, np.product(self._volume_array.shape), 1
        )
        
    def set_renderer(self, renderer):
        """Set the vtkRenderer to render the isosurfaces. Adding a new renderer will remove the last one."""
        if self._actor is None:
            raise RuntimeError("Actor does not exist.")
        if self._renderer is not None:
            self._renderer.RemoveActor(self._actor)
        self._renderer = renderer
        self._renderer.AddActor(self._actor)

    def get_levels(self):
        """Return a list of the current surface levels."""
        return [
            self._surface_algorithm.GetValue(index) 
            for index in range(self._surface_algorithm.GetNumberOfContours())
        ]

    def add_level(self, level):
        """Add a single surface level."""
        self._surface_algorithm.SetValue(
            self._surface_algorithm.GetNumberOfContours(), level
        )
        self._render()

    def remove_level(self, index):
        """Remove a singel surface level at the provided index."""
        for index in range(index, self._surface_algorithm.GetNumberOfContours()-1):
            self._surface_algorithm.SetValue(
                index, self._surface_algorithm.GetValue(index+1)
            )
        self._surface_algorithm.SetNumberOfContours(
            self._surface_algorithm.GetNumberOfContours()-1
        )
        self._render()

    def set_level(self, index, level):
        """Change the value of an existing surface level."""
        self._surface_algorithm.SetValue(index, level)
        self._render()

    def set_cmap(self, cmap):
        """Set the colormap. Supports all matplotlib colormaps."""
        self._mapper.ScalarVisibilityOn()
        self._mapper.SetLookupTable(getColormap(
            self._volume_array.min(), 
            self._volume_array.max(), 
            colorscale=cmap
        ))
        self._render()

    def set_color(self, color):
        """Plot all surfaces in the provided color. Accepts an rbg iterable."""
        self._mapper.ScalarVisibilityOff()
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._render()

    def set_opacity(self, opacity):
        """Set the opacity of all surfaces. (seting it individually for each surface is not supported)"""
        self._actor.GetProperty().SetOpacity(opacity)
        self._render()

    def _render(self):
        """Render if a renderer is set, otherwise do nothing."""
        if self._renderer is not None:
            self._renderer.GetRenderWindow().Render()

    def set_data(self, volume):
        """Change the data displayed. The new array must have the same shape as the current one."""
        if volume.shape != self._volume_array.shape:
            raise ValueError("New volume must be the same shape as the old one")
        self._volume_array[:] = volume
        self._float_array.Modified()
        self._render()


