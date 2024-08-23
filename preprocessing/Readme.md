# Motion Password – Preprocessing

This repository contains scripts for preprocessing the Motion Password (MoP) datasets used for motion-based verification models.

## Dataset

You find the dataset in a [separate repository](). However, this is only required by the first scripts (`01-preprocess-*.py`), later scripts use the `intermediate/` file, which you can pull via [DVC](https://dvc.org) with `dvc pull`.

## Install Instructions

- **Python Version**: Python >= 3.7
- **Dependencies**: Install requirements from `requirements.txt` using the following command:
  ```sh
  pip install -r requirements.txt
  ```

## Preprocessing Steps

### `01-preprocess-*.py`

These scripts process the raw tracking data from the data collection studies ('main' and 'attack'), extracting individual Motion Password sequences and adding metadata such as the word being written, the hand used, the user, and the session.

- **Input**: Raw tracking data
- **Output**: Intermediate data blobs stored in the `intermediate/` folder

### `02-verify-*.py`

These scripts take the intermediate data blobs and produce 2D projections of each signature for visual inspection.

- **Input**: Intermediate data blobs from `01-preprocess-*.py`
- **Output**: Rendered visualizations stored in `rendered_<main|attack>_passwords/`

### `03-prepare-*.py`

These scripts prepare the Motion Password sequences as input for the verification models. Resampling and data encoding are performed using the [Motion Learning Toolbox](https://github.com/cschell/Motion-Learning-Toolbox).

- **Input**: Intermediate data blobs from `01-preprocess-*.py`
- **Output**: Preprocessed sequences stored in `out/<main|attack>_passwords_<encoding>-enc_<FPS>-fps.feather`

Additionally:
- **Password Rules**: We applied heuristics to trim and determine the validity of Motion Password signatures using `password_rules.py`.
- **Manual Corrections**: We applied manual corrections to include, exclude, or trim individual signatures using `manual_corrections.py`.

### `04-verify-preprocessed-*.py`

These scripts take the preprocessed sequences from `03-prepare-*.py` and produce 2D projections for visual inspection, placing them in `rendered_*SR_passwords/`.

- **Input**: Preprocessed sequences from `03-prepare-*.py`
- **Output**: Rendered visualizations for visual inspection stored in `rendered_*SR_passwords/`
