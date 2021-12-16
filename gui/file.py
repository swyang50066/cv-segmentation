import  glob

import  cv2
import  numpy as np
        
from    pydicom         import  dcmread

from    stl                 import  mesh
from    skimage.measure     import  marching_cubes        

from    utils.GMM       import  gaussianMixtureModel


class File(object):
    @staticmethod
    def importDICOM(path):
        ''' import DICOMs
        '''
        # Catch image files
        sequence = glob.glob("/".join(path.split("/")[:-1]) + "/*")

        # Validate correct path is given 
        if not sequence:
            print("WRONG IMAGE PATH IS GIVEN")
            raise ValueError

        # Set attribute map
        attribute = {"PatientName": "None",
                     "PatientID": "None",
                     "PatientAge": "None",
                     "PatientBirthDate": "None",
                     "PatientSex": "None",
                     "ImageType": "None",
                     "Modality": "None",
                     "Manufacturer": "None",
                     "KVP": "None",
                     "StudyDescription": "None",
                     "SeriesDescription": "None",
                     "Spacing": "None"}

        # Convert image format to volume
        zpos = np.zeros(len(sequence))
        stack = np.zeros((len(sequence), 512, 512))
        for k, dicom in enumerate(sequence):
            # Read DICOM
            ds = dcmread(dicom)

            if k == 0:
                xyspacing = np.round(ds.PixelSpacing[0], 2) 
                if "PatientName" in ds:
                    attribute["PatientName"] = str(ds.PatientName)
                if "PatientID" in ds:
                    attribute["PatientID"] = str(ds.PatientID)
                if "PatientAge" in ds:
                    attribute["PatientAge"] = str(ds.PatientAge)
                if "PatientBirthDate" in ds:
                    attribute["PatientBirthDate"] = str(ds.PatientBirthDate)   
                if "PatientSex" in ds:
                    attribute["PatientSex"] = str(ds.PatientSex)
                if "ImageType" in ds:
                    attribute["ImageType"] = " ".join(ds.ImageType)
                if "Modality" in ds:
                    attribute["Modality"] = str(ds.Modality)
                if "Manufacturer" in ds:
                    attribute["Manufacturer"] = str(ds.Manufacturer)
                if "KVP" in ds:
                    attribute["KVP"] = str(ds.KVP)
                if "StudyDescription" in ds:
                    attribute["StudyDescription"] = str(ds.StudyDescription)
                if "SeriesDescription" in ds:
                    attribute["SeriesDescription"] = str(ds.SeriesDescription)

            # Get attributes
            zpos[k]   = float(ds.ImagePositionPatient[2])
            intercept = float(ds.RescaleIntercept)
            slope     = float(ds.RescaleSlope)

            # Rescalize HU
            stack[k] = slope*ds.pixel_array + intercept

        # Sort dicoms
        zth = np.argsort(zpos)[::-1]
        stack[np.arange(len(sequence))] = stack[zth]

        # Get spacing
        zspacing = np.round(zpos[zth[0]] - zpos[zth[1]], 2)
        spacing = np.array([xyspacing, xyspacing, zspacing])
        attribute["Spacing"] = str(xyspacing) + "/" + str(zspacing)

        # Apply adaptive windowing
        voxel = gaussianMixtureModel(stack)
        
        # Initialize image
        image = np.zeros(voxel.shape + (3,), dtype=np.uint8)
        image[...] = np.expand_dims(voxel, axis=-1)

        # Initialize overlays
        seedOverlay = np.zeros_like(image)
        segmentOverlay = np.zeros_like(image)

        # Stack layer
        layers = [image, seedOverlay, segmentOverlay, spacing]

        return layers, attribute

    @staticmethod
    def importImage(path):
        ''' import input file
        '''
        # Check existence of sequential images
        if "_" in path.split("/")[-1]:
            load = ("/".join(path.split("/")[:-1]) 
                    + '/' + path.split("/")[-1].split("_")[-2] 
                    + "*")
        else:
            load = ("/".join(path.split("/")[:-1]) 
                    + '/' + path.split("/")[-1] 
                    + "*")

        # Catch image files
        sequence = glob.glob(load) 
       
        # Validate correct path is given 
        if not sequence:
            print("WRONG IMAGE PATH IS GIVEN")
            raise ValueError
 
        # Sort sequenctial slices
        if len(sequence) > 1:
            sortKey = lambda x: int(x.split("_")[-1].split(".")[0])
            sequence = sorted(sequence, key=sortKey)

        # Initialize image stack
        image = np.array([cv2.imread(scene).astype(np.uint8)
                          for scene in sequence])

        # Initialize overlays
        seedOverlay = np.zeros_like(image)
        segmentOverlay = np.zeros_like(image)
                
        # Stack layer
        layers = [image, seedOverlay, segmentOverlay]

        return layers

    @staticmethod
    def exportImage(path, image, label, colors, fformat='bmp'):
        ''' export background-cropped image as 'fformat' file
        '''
        for k in range(len(image)):
            bgr = image[k].copy()
           
            if np.sum(label[k]):
                for cls, color in colors.items():
                    pos = np.argwhere(label[k] == cls)
                    for i, j in pos:
                        bgr[i, j, 1] = color[1]
                        bgr[i, j, 2] = color[2]

            cv2.imwrite(path + "/CT_{0:0>3}.{1}".format(str(k), fformat), bgr)

    @staticmethod
    def exportSTL(path, label):
        ''' Export image as STL format
        '''
        # Generate surface polygons
        verts, faces, _, _ = marching_cubes(
                                        label,
                                        allow_degenerate=True,
                                        step_size=2)

        # Create the mesh
        cube = mesh.Mesh(np.zeros(faces.shape[0], 
                                  dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = verts[face[j],:]

        # Write the mesh to file "cube.stl"
        cube.save(path + "/cube.stl")
