import nibabel as nib
from dipy.viz import actor, window
from fury.actor import slicer
import numpy as np

def plot_nifti(nifti_path, data_slice="m", orientation="axial", size=(600,400), radiological=True, **kwargs):

    # Load the data and convert to RAS
    nifti = nib.load(nifti_path)
    nifti = nib.as_closest_canonical(nifti)

    # Get the data and affine
    data = nifti.get_fdata()
    affine = nifti.affine

    # Set up slicer and window
    slice_actor = slicer(data, affine=affine, **kwargs)
    scene = window.Scene()
    scene.add(slice_actor)

    # Get slice if not defined
    if orientation == "axial":
        data_slice = data.shape[2] // 2 if data_slice == "m" else data_slice
        if radiological:
            data = np.flip(data, axis=0)
        slice_actor.display_extent(0, data.shape[0], 0, data.shape[1], data_slice, data_slice)

        camera_pos = (0, 0, 1)
        camera_focal = (0, 0, 0)
        camera_up = (0, 1, 0)
    elif orientation == "coronal":
        data_slice = data.shape[1] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(0, data.shape[0], data_slice, data_slice, 0, data.shape[2])
    
        camera_pos = (0, 1, 0)
        camera_focal = (0, 0, 0)
        camera_up = (0, 0, 1)
    elif orientation == "sagittal":
        data_slice = data.shape[0] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(data_slice, data_slice, 0, data.shape[1], 0, data.shape[2])

        camera_pos = (1, 0, 0)
        camera_focal = (0, 0, 0)
        camera_up = (0, 0, 1)
    
    # Set up camera
    scene.reset_camera_tight()
    scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=camera_up)


    # Show the scene
    window.show(scene, size=size)