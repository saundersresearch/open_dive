# nifti2png: Convert and Visualize NIfTI Files

### **Overview**

`nifti2png` is a command-line utility designed to simplify the visualization of medical imaging data stored in NIfTI (.nii or .nii.gz) format. With `nifti2png`, users can extract slices from 3D or 4D medical images and render them interactively or save them as PNG files for further use.

This tool is part of a larger project or library and offers flexible options for visualization, including slice orientation, output resolution, and intensity value range adjustments.

---

### **Setup and Installation**

To use `nifti2png`, ensure that you have the required dependencies installed. Follow these steps to set up the environment:

1. **Install Dependencies:**

   - Install Python (>= 3.7).
   - Use pip or conda to install the required libraries:
     ```bash
     pip install nibabel numpy dipy fury argparse
     ```

2. **Clone the Repository:** Clone the repository containing the `nifti2png` script:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

3. **Run the Script:**

   - Navigate to the folder containing the script and execute `nifti2png` directly via Python.

---

### **Usage Instructions**

The `nifti2png` utility provides an easy-to-use command-line interface. Below are the available options and examples.

#### **Command-Line Options**

```bash
python nifti2png.py [NIfTI file path] [OPTIONS]
```

| Option               | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| `nifti_path`         | (Required) Path to the NIfTI file to be visualized.                               |
| `-s, --slice`        | Slice to plot or `"m"` for the middle slice. Default is `"m"`.                    |
| `-o, --orientation`  | Slice orientation: `"axial"`, `"sagittal"`, or `"coronal"`. Default is `"axial"`. |
| `--size`             | Output resolution in pixels. Default: `(600, 400)`.                               |
| `--save_path`        | Path to save the resulting PNG file.                                              |
| `--interactive`      | Enable interactive display.                                                       |
| `--value_range`      | Set the intensity value range (e.g., `--value_range 0 1000`).                     |
| `--interpolation`    | Set interpolation: `"nearest"` or `"linear"`. Default is `"nearest"`.             |
| `--not_radiological` | Plot with the right-hand side of the image on the subject’s right.                |

#### **Example Commands**

1. **Visualize the Middle Axial Slice Interactively:**

   ```bash
   python nifti2png.py my_image.nii.gz --interactive
   ```

2. **Save a Coronal Slice to a File:**

   ```bash
   python nifti2png.py my_image.nii.gz -s 50 -o coronal --save_path slice_50_coronal.png
   ```

3. **Customize Resolution and Intensity Range:**

   ```bash
   python nifti2png.py my_image.nii.gz --size 800 600 --value_range 0 1500
   ```

4. **Generate Images in Neurological View:**

   ```bash
   python nifti2png.py my_image.nii.gz --not_radiological
   ```

---

### **Output Examples**

1. **Interactive View**:

   - When using the `--interactive` flag, the tool opens a visualization window to explore slices dynamically.

2. **Saved Images**:

   - PNG files containing axial, sagittal, or coronal slices can be generated with optional value range and resolution customizations. Example:



---

### **Troubleshooting**

#### **Common Errors and Fixes**

| Error                       | Cause                                               | Solution                                             |
| --------------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| `FileNotFoundError`         | Input NIfTI file path is incorrect or file missing. | Ensure the file exists and the path is correct.      |
| `ImportError`               | Required libraries are not installed.               | Install dependencies using pip.                      |
| Image is not displayed      | Missing display capability in the environment.      | Use a system with GUI support or save the output.    |
| `ValueError: Invalid range` | Incorrect `--value_range` arguments.                | Ensure range is within the NIfTI’s intensity values. |

#### **Getting Help**

If issues persist, refer to the project documentation or raise a ticket on the project’s GitHub page.

---

By following these guidelines, you can easily render and save 2D representations of NIfTI images with `nifti2png`.

