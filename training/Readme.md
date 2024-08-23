# Motion Password Training and Evaluation

This repository contains the machine learning models, training routines, and final evaluation scripts for our Motion Passwords study.

## Setup

We recommend using a virtual environment to manage dependencies. Follow the setup instructions below:

1. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   pip install virtualenv
   virtualenv .env
   source .env/bin/activate   # On Windows use `.env\Scripts\activate`
   ```

2. **Install required packages:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Pull data files:**
    Install [DVC](https://dvc.org), and then `dvc pull`.

## Data

### Data Sources

- **Who Is Alyx? Dataset**: Preprocessed using scripts in `data-preprocessing/training/`.
  - Original dataset: [Who Is Alyx?](https://github.com/cschell/who-is-alyx)
  - Processed data stored in: `data/wia/`

- **Ball-Throwing (BaT) Dataset**: Prepared using `data-preprocessing/testing/bat-dataset.py`.
  - Original dataset: [Miller et al.](https://github.com/Terascale-All-sensing-Research-Studio/VR-Biometric-Authentication)
  - Processed data stored in: `data/bat/`

- **Motion Passwords (MoP) Dataset**: Data preparation scripts are in a separate repository `../motion-password-preprocessing/`.
  - Prepared data copied to: `data/mop/`

### Data Alignment

All datasets have been aligned using scripts from [cschell/xr-motion-dataset-conversion-scripts](https://github.com/cschell/xr-motion-dataset-conversion-scripts) to ensure consistent formatting.

## Training the Similarity-Learning Model

To train the similarity-learning model:

1. **Configuration:** Update the Weights and Biases (wandb) settings in `train.py`:
    ```python
    wandb.init(project="your_wandb_project", entity="your_wandb_entity")
    ```
2. **Run the training script:** `python train.py`
3. **Hyperparameters:** The script reads hyperparameters from `config-defaults.yaml`. This file contains the best configuration found during our hyperparameter search.

### Hyperparameter Configuration

| Parameter        | Description                                                                     | Search Space  | Final Setting |
| ---------------- | ------------------------------------------------------------------------------- | ------------- | ------------- |
| Embedding Size   | Dimensionality of the vector space for embedding input items.                   | 64-512        | 320           |
|                  |                                                                                 |               |               |
| **GRU**          |                                                                                 |               |               |
| No. Layers       | No. of GRU layers.                                                              | 1-5           | 1             |
| Hidden Size      | Hidden layer size of the GRU.                                                   | 20-512        | 384           |
|                  |                                                                                 |               |               |
| **Transformer**  |                                                                                 |               |               |
| Dim. Model       | Size of Transformer's internal feature representations.                         | 24-2048       | 896           |
| Dim. Feedforward | Size of the network within the Transformer encoder.                             | 8-600         | 384           |
| No. Layers       | No. of stacked Transformer encoders                                             | 1-5           | 2             |
| No. Heads        | Affects how the Transformer processes input in parallel.                        | 2-64          | 16            |
|                  |                                                                                 |               |               |
| **Training**     |                                                                                 |               |               |
| Learning Rate    | Step size for the optimizer to adjust model weights during training.            | 4.6e-07-0.001 | 2.16e-05      |
| Dropout Frames   | Dropout applied immediately to the input sequences.                             | 0.0-0.5       | 0.15          |
| Dropout Global   | Overall dropout applied across the entire model to reduce overfitting.          | 0.0-0.4       | 0             |
| Dropout GRU      | Dropout specifically for GRU layers to prevent overfitting.                     | 0.0-0.6       | 0             |
| Loss Margin      | Determines min. angular margin, enhancing feature discriminability.             | 1-20          | 1.28          |
| Loss Scale       | Fine-tunes how bold or cautious the model is in distinguishing between classes. | 0-490         | 20            |

We provide the pretrained model in `models/abloe7xb.ckpt`.

## Evaluation

The results and figures presented in our paper were generated using the evaluation scripts in this repository, using the pretrained model from `models/abloe7xb.ckpt`.

- Run `evaluation/main.py` for final evaluation.
- This script is designed to be run in VSCode's interactive Jupyter mode or PyCharm's Scientific Mode.
