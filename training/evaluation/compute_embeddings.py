# %%

import hashlib
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

from evaluation.helpers import _hash_list_of_ndarrays
from evaluation.load_motion_passwords import load_motion_passwords
from src.similarity_learning import SimilarityLearning

import torch

import rootutils

root_path = rootutils.find_root(
    search_from=__file__, indicator="requirements.txt"
)


def get_from_wandb(run_id, metric):
    import wandb

    api = wandb.Api()
    run = api.run(f"<entity>/<project>/{run_id}")

    artifact = [
        a for a in run.logged_artifacts() if a.name.startswith(f"best_{metric}")
    ][0]
    checkpoint_path = artifact.download()
    return checkpoint_path


def compute_sl_embeddings(
    motion_sequences,
    sequence_metadata,
    preprocessing_info,
    model_path_or_id,
    device="cpu",
    metric="r_precision",
):
    if isinstance(model_path_or_id, Path):
        assert (
            model_path_or_id.exists()
        ), f"couldn't load checkpoint from {model_path_or_id}, file does not exist"
        checkpoint_path = model_path_or_id
        model_name = checkpoint_path.stem
    else:
        checkpoint_path = (
            get_from_wandb(run_id=model_path_or_id, metric=metric)
            + f"/max_{metric}_val.ckpt"
        )
        model_name = model_path_or_id

    cache_output_path = (
        root_path
        / f"data/cache/embedded_passwords_{model_name}_{_hash_list_of_ndarrays(motion_sequences)}.parquet"
    )

    if cache_output_path.exists():
        print(f"loading cached embeddings from {cache_output_path}")
        return pd.read_parquet(cache_output_path)

    model = SimilarityLearning.load_from_checkpoint(
        checkpoint_path, map_location=device
    ).eval()
    assert preprocessing_info["fps"] == model.hparams.fps
    assert preprocessing_info["enc"] == model.hparams.data_encoding

    def t(X):
        return torch.from_numpy(X)

    def scale(X):
        return (X - model.scaling_params["mean"]) / model.scaling_params["std"]

    scaled_tensor_sequences = [
        t(scale(pwd.to_numpy())) for pwd in motion_sequences
    ]
    X_test = (
        pad_sequence(scaled_tensor_sequences, batch_first=True)
        .float()
        .to(device)
    )
    if "sequence_length" in sequence_metadata:
        lengths = torch.tensor(sequence_metadata["sequence_length"])
    else:
        lengths = torch.zeros(len(X_test)) + X_test.shape[1]

    dl = DataLoader(
        TensorDataset(X_test, lengths),
        batch_size=100,
        shuffle=False,
        num_workers=5,
    )

    def run_inference(dl, model):
        with torch.inference_mode():
            for batch_X, batch_lengths in tqdm(dl, desc="running inference"):
                yield model(batch_X, batch_lengths)

    embedded_passwords_torch = torch.concat(list(run_inference(dl, model)))

    emb_dim = embedded_passwords_torch.shape[-1]
    embedded_passwords = pd.concat(
        [
            pd.DataFrame(
                embedded_passwords_torch.cpu().numpy(),
                columns=[f"dim_{d}" for d in range(emb_dim)],
            ),
            sequence_metadata,
        ],
        axis=1,
    )

    embedded_passwords.to_parquet(cache_output_path)
    return embedded_passwords
