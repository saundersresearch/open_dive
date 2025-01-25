import nibabel as nib
import numpy as np
from dipy.viz import window, actor
from fury.actor import slicer
from fury.colormap import colormap_lookup_table
import vtk
import pdb 

def plot_nifti(
    nifti_path,
    data_slice="m",
    orientation="axial",
    size=(600, 400),
    value_range=None,
    radiological=True,
    save_path=None,
    interactive=True,
    scalar_colorbar=True,
    **kwargs,
):
    """Create a 2D rendering of a NIFTI slice.

    Parameters
    ----------
    nifti_path : str or Path
        Path to NIFTI to plot
    data_slice : str or int, optional
        Slice to plot or "m" for middle slice, by default "m"
    orientation : str, optional
        Can be "axial", "sagittal" or "coronal", by default "axial"
    size : tuple, optional
        Size of window, by default (600,400)
    radiological : bool, optional
        Whether to plot in radiological view (subject right is on image left), by default True
    save_path : str or Path, optional
        Optional path to save to, by default None
    interactive : bool, optional
        Whether to interactively show the scene, by default True
    colorbar : bool, optional
        Whether to show a scalar colorbar (for FA, T1, etc.), by default True
    **kwargs
        Additional keyword arguments to pass to fury.actor.slicer
    """

    # Load the data and convert to RAS
    nifti = nib.load(nifti_path)
    nifti = nib.as_closest_canonical(nifti)

    # Get the data and affine
    data = nifti.get_fdata()
    affine = nifti.affine

    # value range
    if value_range is None:
        value_range = [np.min(data), np.max(data)]
    else:
        value_range = [value_range[0], value_range[1]]


    if radiological and orientation == "axial":
        data = np.flip(data, axis=0)

    # Set up slicer and window
    slice_actor = slicer(data, affine=affine, **kwargs)
    scene = window.Scene()
    scene.add(slice_actor)

    # Get slice if not defined
    if orientation == "axial":
        data_slice = data.shape[2] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(
            0, data.shape[0], 0, data.shape[1], data_slice, data_slice
        )

        camera_pos = (0, 0, 1)
        camera_up = (0, 1, 0)
    elif orientation == "coronal":
        data_slice = data.shape[1] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(
            0, data.shape[0], data_slice, data_slice, 0, data.shape[2]
        )

        camera_pos = (0, 1, 0)
        camera_up = (0, 0, 1)
    elif orientation == "sagittal":
        data_slice = data.shape[0] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(
            data_slice, data_slice, 0, data.shape[1], 0, data.shape[2]
        )

        camera_pos = (1, 0, 0)
        camera_up = (0, 0, 1)
    camera_focal = (0, 0, 0)

    # Apply colorbar
    if scalar_colorbar:
        # Create a grayscale colormap (from black to white)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)  # Full grayscale (256 levels)
        lut.Build()  # Initialize the LUT
        
        
        for i in range(256):
            lut.SetTableValue(i, i / 255.0, i / 255.0, i / 255.0, 1)  # Grayscale colors
        '''
        We can further optimize this later to support orther colormaps; this just supports grayscale right now.
        '''
        # Set the full grayscale range (e.g., 0 to 255 for typical image data)
        lut.SetRange(value_range[0], value_range[1])  # This defines the grayscale range explicitly

        # Apply the lookup table to the slice actor
        slice_actor.GetProperty().SetLookupTable(lut)
        slice_actor.GetProperty().SetUseLookupTableScalarRange(True)  # Ensure LUT scalar range is used

        # Create the scalar bar (colorbar)
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)  # Attach the grayscale LUT
        scalar_bar.SetLabelFormat("%.0f")  # Integer labels (0, 1, 2, ... 255)
        scalar_bar.SetPosition(0.8, 0.1)  # Position of the colorbar
        scalar_bar.SetHeight(0.5)  # Adjust height (increase size)
        scalar_bar.SetWidth(0.1)   # Adjust width (increase size)

        # Add the scalar bar to the scene
        scene.add(scalar_bar)

        
    # Set up camera
    scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=camera_up)

    # Show the scene
    if save_path:
        window.record(scene=scene, out_path=save_path, size=size, reset_camera=True)

    if interactive:
        window.show(scene, size=size, reset_camera=True)




