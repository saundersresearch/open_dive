# OpenDIVE

![Axial slice of fractional anisotropy with two tractograms](https://github.com/user-attachments/assets/da300f5e-5148-4fef-befb-85920d03b6cd)

**OpenDIVE** (Open Diffusion Imaging Visualization for Everyone) is a command line interface tool for generating accessible, interpretable visualizations from diffusion MRI, initiated at [BrainHack Vandy 2025](https://brainhack-vandy.github.io/).

Despite the prolific availability of software tools to visualize diffusion MRI data, there is no standardized visualization to summarize longitudinal changes. Similarly, the current standard of visualizations are not accessible to people with common forms of colorblindness. We propose a software package to both standardize and improve accessibility to representations of diffusion data. 

## Installation

You can install the package using Python (3.10+):

```bash
pip install git+https://github.com/MASILab/open_dive
```

## Usage

After installing the package, you should be able to use the `open-dive` command to produce images.

```bash
# Save a T1-weighted image with no interactive visualization
open-dive -n t1.nii.gz -s 50 -o coronal --save_path slice_50_coronal.png --headless

# Custom colorbar
open-dive -n dwi.nii.gz --size 800 600 --value_range 0 1500 --scalar_colorbar --volume_idx 1

# Overlay values on tractography
open-dive -n fa.nii.gz --tractography_path my_tractogram1.trk my_tractogram2.trk --tractography_values -0.5 0.7 

# Overlay glass brain and tensor glyphs with custom viewing angles
open-dive -n dwi.nii.gz \
    --volume_idx 0 \
    --tensor_path tensor.nii.gz \
    --glass_brain_path mask.nii.gz \
    -o sagittal \
    --az 45 \
    --el 60 \
    -scale 2
```

Please see the [wiki](https://github.com/MASILab/open_dive/wiki) for documentation on using the command.

## Contributing

We welcome issues and pull requests! For details on contributing, please see [CONTRIBUTING.md](CONTRIBUTING.md).

### Aims 

- We aim to generate a standardized display for anatomical images to overlay the diffusion models based on user input, including support for multiple track files, color maps, and illustration of bundle summary metrics (p-value, volume, effect size, etc.) per bundle.
- We aim to propose a colorblind-friendly colormap for diffusion MRI images.

For FAQs related to diffusion MRI, see our [diffusion FAQ discussion](https://github.com/MASILab/open_dive/discussions/47).

## Contributors
- [Adam Saunders](https://github.com/saundersresearch)
- [Elyssa McMaster](https://github.com/ElyssaMcMaster)
- [Chris Rorden](https://github.com/neurolabusc/neurolabusc)
- [Johaan Kathilankal Jis](https://github.com/johaankjis)
- [Minyi Sun](https://github.com/Orekiwlg)
- [Adam Sadriddinov](https://github.com/mukhsadr)
- [Lukas VanTilburg](https://github.com/beeper-weepers)
- [Kurt Schilling](https://github.com/schillkg)

## License

[MIT](https://choosealicense.com/licenses/mit/)
