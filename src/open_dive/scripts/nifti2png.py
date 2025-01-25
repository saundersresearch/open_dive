import argparse
from pathlib import Path

from open_dive.viz import plot_nifti


def main():
    # Create args
    parser = argparse.ArgumentParser(description="Plot a slice of a NIFTI file")
    parser.add_argument("nifti_path", type=Path, help="Path to NIFTI to plot")
    parser.add_argument(
        "-s", "--slice", default="m", help='Slice to plot or "m" for middle slice'
    )
    parser.add_argument(
        "-o",
        "--orientation",
        default="axial",
        help='Can be "axial", "sagittal" or "coronal"',
    )
    parser.add_argument(
        "--size", default=(600, 400), help="Size of window, by default (600,400)"
    )
    parser.add_argument("--save_path", help="Optional path to save to")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to interactively show the scene",
    )
    parser.add_argument(
        "--value_range",
        type=int,
        nargs=2,
        help="Optional value range to pass to slicer. Default is min/max of image.",
    )
    parser.add_argument(
        "--volume_idx",
        type=int,
        help="Index of the volume to display if the image is 4D",
    )
    parser.add_argument(
        "--interpolation",
        default="nearest",
        help="Interpolation method to use (nearest or linear). Default is 'nearest'.",
    )
    parser.add_argument(
        "--not_radiological",
        action="store_true",
        help="Do not plot in radiological view (i.e., subject right is on image right)",
    )
    parser.add_argument(
        "--scalar_colorbar",
        action="store_true",
        help="Whether to show a colorbar, by default True",
    )
    parser.add_argument(
        ## plot tractogram with slices
        "--tractography",
        type=Path,
        help="Optional tractogram to plot with slices",
    )


    args = parser.parse_args()

    # Plot the NIFTI
    plot_nifti(
        args.nifti_path,
        data_slice=args.slice,
        orientation=args.orientation,
        size=args.size,
        volume_idx=args.volume_idx,
        radiological=not args.not_radiological,
        save_path=args.save_path,
        interactive=args.interactive,
        value_range=args.value_range,
        interpolation=args.interpolation,
        scalar_colorbar=args.scalar_colorbar,
        tractography=args.tractography,

    )

