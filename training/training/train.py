# %% Imports and helper methods

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
import wandb

import rootutils

root_path = rootutils.find_root(search_from=__file__, indicator="requirements.txt")

os.environ["PYTHONPATH"] = str(root_path)
sys.path.insert(0, str(root_path))

from src.similarity_learning import SimilarityLearning


def parse_kv_args(args):
    kv_args = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key.startswith("-"):
                continue
            try:
                kv_args[key] = float(value) if "." in value else int(value)
            except ValueError:
                kv_args[key] = value
    return kv_args


# %%
DEBUG = os.environ.get("DEBUG", False)
L.seed_everything(42, workers=True)

# %% Initialize Weights and Biases

wandb.init(
    project="your_wandb_project",
    entity="your_wandb_entity",
    mode="disabled" if DEBUG else None,
    group=os.environ.get("WANDB_RUN_GROUP", "default"),
)

args = parse_kv_args(sys.argv[1:])

if args:
    wandb.config.update(args, allow_val_change=True)

CONFIG = wandb.config

wandb.define_metric("loss/train", summary="min")
wandb.define_metric("precision_at_1/val", summary="max")
wandb.define_metric("r_precision/val", summary="max")

# %% Load train and validation data

dataset_path = root_path / f"data/who-is-alyx/wia-16_{CONFIG['data_encoding']}-enc_{CONFIG['fps']}-fps_15-seq.npz"

wandb.summary["dataset_path"] = str(dataset_path)
wandb.summary["dataset_filename"] = dataset_path.name

wia_data = np.load(dataset_path)
train_X = wia_data["train_X"].astype("float64")
train_y = wia_data["train_y"]

val_X = wia_data["val_X"].astype("float64")
val_y = wia_data["val_y"]

num_train_samples, sequence_length, n_features = train_X.shape

std_train = np.std(train_X.reshape(-1, n_features), axis=0)
mean_train = np.mean(train_X.reshape(-1, n_features), axis=0)

train_X = ((train_X.reshape(-1, n_features) - mean_train) / std_train).reshape(*train_X.shape)
val_X = ((val_X.reshape(-1, n_features) - mean_train) / std_train).reshape(*val_X.shape)

train_y_int = LabelEncoder().fit_transform(train_y.tolist())
val_y_int = LabelEncoder().fit_transform(val_y.tolist())


# %% Setup datasets and dataloaders

n_train_classes = train_y_int.max() + 1

train_dset = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y_int).long())
val_dset = TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y_int).long())

train_loader = DataLoader(
    train_dset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=5,
    sampler=MPerClassSampler(
        train_y_int,
        CONFIG["batch_size"] // n_train_classes,
        length_before_new_iter=CONFIG["batch_size"] * 100,
    ),
)

val_loader = DataLoader(val_dset, batch_size=1000, shuffle=False, num_workers=5)

# %% Setup model

model = SimilarityLearning(
    **CONFIG,
    num_features=n_features,
    n_train_classes=n_train_classes,
    scaling_params={"mean": mean_train, "std": std_train},
)

# %% Setup lightning callbacks

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor="r_precision/val",
        min_delta=0.001,
        mode="max",
        patience=10,
        verbose=True,
        strict=False,
        stopping_threshold=1,
    ),
    ModelCheckpoint(
        monitor="precision_at_1/val",
        mode="max",
        filename="max_precision_at_1_val",
        auto_insert_metric_name=False,
    ),
    ModelCheckpoint(
        monitor="r_precision/val",
        mode="max",
        filename="max_r_precision_val",
        auto_insert_metric_name=False,
    ),
]

# %% Setup Trainer

from pytorch_lightning.loggers import WandbLogger

trainer = L.Trainer(
    max_epochs=1000,
    min_epochs=50,
    fast_dev_run=DEBUG,
    check_val_every_n_epoch=2,
    logger=[WandbLogger(log_model=True)],
    callbacks=callbacks,
    precision="16-mixed",
)

# %% Start training

trainer.fit(model, train_loader, val_loader)


# %% Upload model checkpoints to Weights and Biases

for cb in callbacks:
    if isinstance(cb, ModelCheckpoint):
        ckpt_path = Path(cb.dirpath) / f"{cb.filename}.ckpt"
        if ckpt_path.exists():
            wandb.log_model(ckpt_path, name=f"best_{cb.monitor.replace('/', '-')}")
        else:
            print(f"WARNING: couldn't find model checkpoint {ckpt_path}, did not upload to wandb")

wandb.finish()
