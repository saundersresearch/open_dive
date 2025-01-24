# Contributing to OpenDIVE
Thank you for your interest in contributing to OpenDIVE! 

## How to contribute
1. Fork the repository

Navigate to the main repository on GitHub. Click the Fork button in the upper-right corner of the repository page. This will create a copy of the repository under your GitHub account.

2. Clone the fork
Open a terminal and clone the repository to your machine:
```bash
git clone https://github.com/<your-username>/open_dive.git
```
Change into the cloned repository directory
```
cd open_dive
```
3. Set Up the Upstream Repository

Add the original repository as an upstream remote to keep your fork up-to-date:

```bash
git remote add upstream https://github.com/MASILab/open_dive.git
```

Verify the remotes:

```bash
git remote -v
```

4. Create a Feature Branch

Make sure you are on the default branch:

```bash
git checkout main
```

Pull the latest changes from the upstream repository:

```bash
git pull upstream main
```

```bash
git checkout -b <your-feature-branch-name>
```

5. Make Changes and Commit

Make your changes to the codebase. Add the changes to the staging area:
```bash
git add .
```

Commit your changes with a descriptive message:
```bash
git commit -m "added my changes"
```

6. Push Your Changes

Push your changes to your forked repository:
```bash
git push origin <your-feature-branch-name>
```

7. Open a Pull Request

Go to your forked repository on GitHub. You should see a notification that your recently pushed branch has changes. Click the Compare & pull request button.

Fill out the pull request form with a clear title and description of your changes. Select the target branch in the original repository (main). Submit the pull request by clicking the Create pull request button.

8. Respond to Feedback

Be ready to respond to any comments or requested changes from the maintainers. Make additional commits to your branch to address feedback, and push them to your fork. The pull request will update automatically.

## Installing OpenDIVE

It is highly recommended that you work within a virtual environment to isolate your packages from other Python installations on your system.

### Using `uv` (recommended)

We recommend installing Python via `uv`. See instructions [here](https://docs.astral.sh/uv/getting-started/installation/). This will allow you to create a virtual environment that ensures we are all working with the same package versions.

Once you have `uv` installed and have a local clone of the repository, you can create the virtual environment by running the following command:

```bash
uv sync 
uv pip install -e </path/to/open_dive>
```

Then, you can run code using the `uv` command:

```bash
uv run <file>
```

Alternatively, if you would rather use the `python` command, you can activate the virtual environment after syncing from within the `open_dive/` folder:

```bash
source .venv/bin/activate
python <file>
```

### Using `conda` or `mamba`

If you already are using `conda` or `mamba` on your machine, you can set up a virtual environment for `open-dive`:

```bash
conda create -n open-dive python=3.12
conda activate open-dive
pip install -e </path/to/open_dive>
```


## Coding guidelines
We ask that all code is well-commented. Please add `numpydoc`-style docstrings to document your functions and classes. For example, if you create code to add two numbers:

```python
def add(a, b):
    """
    Add together two numbers.

    Parameters
    ----------
    a : int
        First number to add.
    b : int
        Second number to add.
    
    Returns
    -------
    sum : int
        The sum of the two numbers.
    """
    sum = a + b
    return sum
```
