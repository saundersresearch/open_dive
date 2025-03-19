import nibabel as nib
import numpy as np
from dipy.viz import window, actor
import vtk
import matplotlib.pyplot as plt 
from fury.actor import slicer, tensor_slicer, odf_slicer, ellipsoid, _color_fa, _fa, contour_from_roi, surface
from fury.utils import apply_affine, get_bounds
from dipy.io.image import load_nifti
from dipy.data import get_sphere
from dipy.reconst.dti import from_lower_triangular, decompose_tensor
from dipy.reconst.shm import calculate_max_order, sh_to_sf_matrix
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.align.reslice import reslice
from scipy.ndimage import gaussian_filter, binary_dilation
import cmcrameri

def plot_nifti(
    nifti_path=None,
    data_slice="m",
    orientation="axial",
    size=(600, 400),
    azimuth=None,
    elevation=None,
    value_range=None,
    save_path=None,
    interactive=True,
    scalar_colorbar=True,
    tractography=None,
    tractography_opacity = 0.6,
    tractography_values=None,
    tractography_cmap=None,
    tractography_cmap_range=None,
    tractography_colorbar=False,
    volume_idx=None,
    tensor_image=None,
    odf_image=None,
    sh_basis="descoteaux07",
    scale=1,
    glass_brain_path=None,
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
    azimuth : float, optional
        Azimuth angle in degrees, by default None
    elevation : float, optional
        Elevation angle in degrees, by default None
    save_path : str or Path, optional
        Optional path to save to, by default None
    interactive : bool, optional
        Whether to interactively show the scene, by default True
    scalar_colorbar : bool, optional
        Whether to show a scalar colorbar (for FA, T1, etc.), by default True
    tractography : list of str or Path, optional
        Optional tractogram(s) to plot with slices. Can provide multiple files, by default None
    tractography_opacity : float, optional
        Optional opacity value for tractograms between (0, 1), by default 0.6
    tractography_values : list of float, optional
        Optional values to color the tractography with, by default None
    tractography_cmap : str, optional
        Optional colormap to use for the tractography, by default None
    tractography_cmap_range : list of float, optional
        Optional range to use for the colormap, by default None for 0 to 1
    tractography_colorbar : bool, optional
        Whether to show a colorbar for the tractography, by default False
    volume_idx : int, optional
        Index of the volume to display if the image is 4D, by default None
    tensor_image : str or Path, optional
        Path to a tensor image to visualize (format is Dxx, Dxy, Dyy, Dxz, Dyz, Dzz), by default None
    odf_image : str or Path, optional
        Path to an ODF image to visualize, by default None
    sh_basis : str, optional
        Basis type for the spherical harmonics, either "descoteaux07" or "tournier07", by default "descoteaux07"
    scale : float, optional
        Scale of the tensor glyphs or ODF glyphs, by default 1
    glass_brain_path : str or Path, optional
        Optional glass brain mask to overlay, by default None

    **kwargs
        Additional keyword arguments to pass to fury.actor.slicer
    """
    scene = window.Scene()

    # Variable to store size of things we are rendering
    scene_bound_data = None
    scene_bound_affine = None

    # Set slice to int if not "m"
    if data_slice != "m":
        data_slice = int(data_slice)

    if nifti_path is not None:
        # Load the data and convert to RAS
        nifti = nib.load(nifti_path)
        nifti = nib.as_closest_canonical(nifti)
        data = nifti.get_fdata()

        scene_bound_data = data
        scene_bound_affine = nifti.affine

        if len(data.shape) == 4:
            if volume_idx is None:
                raise ValueError(
                    "Input is a 4D image but no volume index specified. "
                    "Please provide a volume_idx parameter to select which 3D volume to display."
                )
            if not 0 <= volume_idx < data.shape[3]:
                raise ValueError(
                    f"volume_idx {volume_idx} is out of bounds for image with {data.shape[3]} volumes"
                )
            data = data[..., volume_idx]
        elif len(data.shape) != 3:
            raise ValueError(
                f"Expected 3D or 4D image, but got image with {len(data.shape)} dimensions"
            )

        # Get the data and affine
        affine = nifti.affine

        # value range
        if value_range is None:
            value_range = [np.min(data), np.max(data)]

        # Set up slicer and window
        slice_actor = slicer(data, value_range=value_range, affine=affine, **kwargs)
        scene.add(slice_actor)

        # Get slice if not defined
        if orientation == "axial":
            data_slice = data.shape[2] // 2 if data_slice == "m" else data_slice
            extent = (0, data.shape[0], 0, data.shape[1], data_slice, data_slice)
            offset = np.array([0, 0, scale])

        elif orientation == "coronal":
            data_slice = data.shape[1] // 2 if data_slice == "m" else data_slice
            extent = (0, data.shape[0], data_slice, data_slice, 0, data.shape[2])
            offset = np.array([0, scale, 0])

        elif orientation == "sagittal":
            data_slice = data.shape[0] // 2 if data_slice == "m" else data_slice
            extent = (data_slice, data_slice, 0, data.shape[1], 0, data.shape[2])
            offset = np.array([scale, 0, 0])
    
        if nifti_path is not None:
            slice_actor.display_extent(*extent)

        # Apply colorbar
        if scalar_colorbar:
            # Create a grayscale colormap (from black to white)
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)  # Full grayscale (256 levels)
            lut.Build()  # Initialize the LUT
            
            for i in range(256):
                lut.SetTableValue(i, i / 255.0, i / 255.0, i / 255.0, 1)  # Grayscale colors

            # Set the full grayscale range (e.g., 0 to 255 for typical image data)
            lut.SetRange(value_range[0], value_range[1])  # This defines the grayscale range explicitly

            # Create the scalar bar (colorbar)
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(lut)  # Attach the grayscale LUT
            scalar_bar.SetLabelFormat("%.0f")  # Integer labels (0, 1, 2, ... 255)
            scalar_bar.SetPosition(0.8, 0.1)  # Position of the colorbar
            scalar_bar.SetHeight(0.5)  # Adjust height (increase size)
            scalar_bar.SetWidth(0.1)   # Adjust width (increase size)

            # Add the scalar bar to the scene
            scene.add(scalar_bar)
        
    # Add tractography
    if tractography is not None:
        if tractography_cmap is None:
            tractography_cmap = "Set1" if tractography_values is None else "plasma"
        cmap = plt.get_cmap(tractography_cmap)

        if tractography_cmap_range is None:
            tractography_cmap_range = [0, 1]

        # Set to range
        if tractography_values is not None:
            norm = plt.Normalize(vmin=tractography_cmap_range[0], vmax=tractography_cmap_range[1])
            colors = [cmap(norm(val)) for val in tractography_values]
        else:
            colors = [cmap(i) for i in range(len(tractography))]

        # Apply colorbar
        if tractography_colorbar:
            # Create a grayscale colormap (from black to white)
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)  # Full grayscale (256 levels)
            lut.Build()  # Initialize the LUT
            
            cmap_range = np.linspace(tractography_cmap_range[0], tractography_cmap_range[1], 256)
            for i in range(256):
                # Sample the colormap
                color = cmap(norm(cmap_range[i]))
                lut.SetTableValue(i, color[0], color[1], color[2], 1)  # Grayscale colors

            if tractography_cmap_range is not None:
                lut.SetRange(tractography_cmap_range[0], tractography_cmap_range[1])
            else:
                lut.SetRange(0, 1)

            # Create the scalar bar (colorbar)
            tract_bar = vtk.vtkScalarBarActor()
            tract_bar.SetLookupTable(lut)  # Attach the grayscale LUT
            tract_bar.SetLabelFormat("%.2f")  # Labels
            tract_bar.SetPosition(0.1, 0.1)  # Position of the colorbar
            tract_bar.SetHeight(0.5)  # Adjust height (increase size)
            tract_bar.SetWidth(0.1)   # Adjust width (increase size)

            # Add the scalar bar to the scene
            scene.add(tract_bar)
                
        # Add each tractography with its corresponding color
        for tract_file, color in zip(tractography, colors):
            streamlines_nifti = nib.streamlines.load(tract_file)
            streamlines = streamlines_nifti.streamlines
            stream_actor = actor.line(streamlines, colors=color, linewidth=0.2, opacity=tractography_opacity)
            scene.add(stream_actor)
    
    # Add diffusion tensor visualization
    if tensor_image is not None:
        tensor_data, tensor_affine = load_nifti(tensor_image)
        sphere = get_sphere(name="repulsion724")  # Use a precomputed sphere
        tensor_matrix = from_lower_triangular(tensor_data)
        eigvals, eigvecs = decompose_tensor(tensor_matrix)
        mask = np.ones(tensor_data.shape[:3])
        centers = np.ascontiguousarray(np.array(np.nonzero(mask)).T)

        # Reshape to N x 3
        centers = centers.reshape(-1, 3)
        eigvals = eigvals.reshape(-1, 3)
        eigvecs = eigvecs.reshape(-1, 3, 3)

        # Only keep centers in extent
        indices = (centers[:, 0] >= extent[0]) & (centers[:, 0] <= extent[1]) & (centers[:, 1] >= extent[2]) & (centers[:, 1] <= extent[3]) & (centers[:, 2] >= extent[4]) & (centers[:, 2] <= extent[5])

        centers = centers[indices,:]
        eigvals = eigvals[indices,:]
        eigvecs = eigvecs[indices,:,:] 

        # Apply affine to centers
        centers = apply_affine(tensor_affine, centers)

        # Get colors
        colors = _color_fa(_fa(eigvals), eigvecs)

        tensor_actor = ellipsoid(centers=centers, axes=eigvecs, lengths=eigvals, scales=scale*2, colors=colors)
        scene.add(tensor_actor)

        position = tensor_actor.GetPosition()
        position += offset
        tensor_actor.SetPosition(position)

        if scene_bound_data is None:
            scene_bound_data = mask
            scene_bound_affine = tensor_affine

    # Add orientation distribution function visualization
    if odf_image:
        odf_data, odf_affine = load_nifti(odf_image)
        sphere = get_sphere(name="repulsion724")  # Use a precomputed sphere
        sh_order_max = calculate_max_order(odf_data.shape[-1])
        B, _ = sh_to_sf_matrix(sphere=sphere, sh_order_max=sh_order_max, basis_type=sh_basis)
        odf_actor = odf_slicer(odf_data, sphere=sphere, B_matrix=B, scale=scale, norm=False, affine=odf_affine)
        
        scene.add(odf_actor)
        odf_actor.display_extent(*extent)

        position = odf_actor.GetPosition()
        position += offset
        odf_actor.SetPosition(position)

        if scene_bound_data is None:
            scene_bound_data = odf_data
            scene_bound_affine = odf_affine

    if glass_brain_path:
        glass_brain_actor = create_glass_brain(glass_brain_path)
        scene.add(glass_brain_actor)
        
        if scene_bound_data is None:
            glass_brain = nib.load(glass_brain_path)
            scene_bound_data = np.ones_like(glass_brain.get_fdata())
            scene_bound_affine = glass_brain.affine

    if orientation == "axial":
        azimuth = 0 if azimuth is None else azimuth
        elevation = 0 if elevation is None else elevation
    elif orientation == "coronal":
        azimuth = 180 if azimuth is None else azimuth
        elevation = 90 if elevation is None else elevation
    elif orientation == "sagittal":
        azimuth = -90 if azimuth is None else azimuth
        elevation = 90 if elevation is None else elevation

    if (azimuth is not None or elevation is not None) and scene_bound_data is not None:
        camera_pos = np.array([0, 0, 1])
        camera_focal = np.array([0, 0, 0])
        camera_up = np.array([0, 1, 0])

        camera_pos_r, camera_pos_theta, camera_pos_phi = cart2sphere(*camera_pos)
        camera_up_r, camera_up_theta, camera_up_phi = cart2sphere(*camera_up)

        # Rotate by azimuth
        camera_pos_phi += np.deg2rad(azimuth)
        camera_up_phi += np.deg2rad(azimuth)

        # Rotate by elevation
        camera_pos_theta -= np.deg2rad(elevation)
        camera_pos_phi += np.deg2rad(90)
        camera_up_theta -= np.deg2rad(elevation)

        # Convert back to cartesian
        camera_pos = sphere2cart(camera_pos_r, camera_pos_theta, camera_pos_phi)
        camera_up = sphere2cart(camera_up_r, camera_up_theta, camera_up_phi)

        # Scale to 1.5*max dimension and shift to middle of array
        camera_pos = np.array(camera_pos)*1.5*max(scene_bound_data.shape) + np.array([scene_bound_data.shape[0] // 2, scene_bound_data.shape[1] // 2, scene_bound_data.shape[2] // 2])
        camera_focal = np.array(camera_focal) + np.array([scene_bound_data.shape[0] // 2, scene_bound_data.shape[1] // 2, scene_bound_data.shape[2] // 2])

        # Apply affine to translate into world coordinates
        camera_pos = apply_affine(scene_bound_affine, camera_pos)
        camera_focal = apply_affine(scene_bound_affine, camera_focal)

        # Set camera
        scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=camera_up)
    else: 
        scene.reset_camera()
        camera_pos, camera_focal, _ = scene.get_camera()
        view_up = (0, 1, 0)

        # Subtract focal point to get camera position
        camera_pos = np.array(camera_pos)
        camera_pos_r, camera_pos_theta, camera_pos_phi = cart2sphere(*camera_pos)
        view_up_r, view_up_theta, view_up_phi = cart2sphere(*view_up)

        # # Rotate by azimuth
        camera_pos_phi += np.deg2rad(azimuth) 
        view_up_phi += np.deg2rad(azimuth)

        # Rotate by elevation (theta)
        camera_pos_theta -= np.deg2rad(elevation)
        view_up_theta -= np.deg2rad(elevation)

        # Convert back to cartesian
        camera_pos = sphere2cart(camera_pos_r, camera_pos_theta, camera_pos_phi)
        camera_pos = np.array(camera_pos)
        view_up = sphere2cart(view_up_r, view_up_theta, view_up_phi)
        view_up = np.array(view_up) 

        scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=view_up)

    # Show the scene
    if save_path:
        window.record(scene=scene, out_path=save_path, size=size, reset_camera=False)

    if interactive:
        window.show(scene, size=size, reset_camera=False)

def create_glass_brain(mask_nifti, resample_factor=2, smooth_sigma=2, dilation_iters=2, opacity=0.33):
    """Create a "glass brain" visualization from a binary mask.

    Parameters
    ----------
    mask_nifti : str or Path
        Path to binary mask NIFTI image
    resample_factor : int, optional
        Factor to upsample the mask by, by default 3
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing, by default 2
    dilation_iters : int, optional
        Number of iterations for binary dilation, by default 2
    opacity : float, optional
        Opacity of the glass brain, by default 0.33

    Returns
    -------
    glass_brain : fury.actor.surface
        Glass brain surface actor
    """
    # Load the mask
    mask_nifti = nib.load(mask_nifti)
    mask = mask_nifti.get_fdata()
    affine = mask_nifti.affine
    zooms = mask_nifti.header.get_zooms()[:3]

    # Step 1: Upsample (regrid) the mask by a factor of 5
    new_zooms = tuple(z / resample_factor for z in zooms)
    mask_up, new_affine = reslice(mask, affine, zooms, new_zooms)

    # Step 2: Apply Gaussian smoothing with standard deviation 2
    mask_smooth = gaussian_filter(mask_up, sigma=smooth_sigma)

    # Step 3: Threshold the smoothed mask at 0.5
    mask_thres = (mask_smooth > 0.5).astype(np.uint8)

    # Step 4: Dilate the thresholded mask with 2 passes
    mask_dilated = binary_dilation(mask_thres, iterations=dilation_iters).astype(np.uint8)

    # Create a surface actor
    glass_brain_actor = contour_from_roi(mask_dilated, affine=new_affine, opacity=opacity, color=(0.5, 0.5, 0.5))
    return glass_brain_actor