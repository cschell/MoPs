# %%
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np
import rootutils
from evaluation.matplotlib_settings import (
    FULL_COLUMN_WIDTH,
    apply_default_matplotlib_settings,
)

apply_default_matplotlib_settings()
from evaluation.compute_embeddings import compute_sl_embeddings
from evaluation.load_motion_passwords import load_motion_passwords
from pytorch_metric_learning.distances import CosineSimilarity
import torch


root_path = rootutils.find_root(search_from="./", indicator="requirements.txt")

sl_model_path = Path(root_path / "models/abloe7xb.ckpt")


root_path = rootutils.find_root(search_from="./", indicator="requirements.txt")
prep_info = {"fps": 30, "enc": "BRA"}
motion_passwords, password_metadata = load_motion_passwords(
    root_path
    / f"data/mop/main_passwords_{prep_info['enc']}-enc_{prep_info['fps']}-fps.feather",
)
motion_passwords_sr, password_sr_metadata = load_motion_passwords(
    root_path
    / f"data/mop/main_passwords_SR-enc_{prep_info['fps']}-fps.feather",
)
sml_embeddings = compute_sl_embeddings(
    motion_passwords,
    password_metadata,
    prep_info,
    sl_model_path,
    device="cpu",
)

assert np.all(password_sr_metadata.user_id == password_metadata.user_id)
assert np.all(password_sr_metadata.word == password_metadata.word)
assert np.all(password_sr_metadata.session_id == password_metadata.session_id)
assert np.all(password_sr_metadata.iteration == password_metadata.iteration)
# %%
feature_columns = [
    col for col in sml_embeddings.columns if col.startswith("dim_")
]

embeddings = sml_embeddings.query("word == 'Motion' and iteration == 5")
embeddings = embeddings.set_index("user_id")

query_embeddings = embeddings.query("session_id == 1")
ref_embeddings = embeddings.query("session_id == 0")

filtered_sr_passwords = password_sr_metadata.query(
    "word == 'Motion' and iteration == 5"
)
query_passwords_sr = filtered_sr_passwords.query("session_id == 1").reset_index(
    drop=True
)
ref_passwords_sr = filtered_sr_passwords.query("session_id == 0").reset_index(
    drop=True
)


distances = CosineSimilarity()(
    torch.from_numpy(query_embeddings[feature_columns].to_numpy()),
    torch.from_numpy(ref_embeddings[feature_columns].to_numpy()),
)

n_refs = len(ref_embeddings)
n_queries = len(query_embeddings)

distances_flat = distances.flatten().numpy()
query_idxs, ref_idxs = np.mgrid[:n_queries, :n_refs]
query_idxs = query_idxs.flatten()
ref_idxs = ref_idxs.flatten()

ref_users, query_users = np.meshgrid(
    ref_embeddings.index, query_embeddings.index
)
query_users = query_users.flatten()
ref_users = ref_users.flatten()

gt = (query_users == ref_users).astype(int)

results = pd.DataFrame(
    [
        query_users,
        ref_users,
        query_idxs,
        ref_idxs,
        gt,
        distances_flat,
    ],
    index=[
        "query_users",
        "ref_users",
        "query_idx",
        "ref_idx",
        "gt",
        "score",
    ],
).T

n_examples = 10
worst_fns = results.query("gt == 1").sort_values("score", ascending=True)[
    :n_examples
]
worst_fps = results.query("gt == 0").sort_values("score", ascending=False)[
    :n_examples
]

# %%

pairs_to_analyze = []

for title, indices in [
    ("False Positives", worst_fps),
    ("False Negatives", worst_fns),
]:
    fig, axs = plt.subplots(
        nrows=n_examples, ncols=2, figsize=(FULL_COLUMN_WIDTH, 2 * n_examples)
    )
    for idx, (_, row) in enumerate(indices.iterrows()):
        ax_row = axs[idx]
        ref_data_index = int(row["ref_idx"])

        ref_user_name, ref_word, ref_iteration, ref_session_id = (
            ref_embeddings.iloc[ref_data_index][
                ["user_name", "word", "iteration", "session_id"]
            ]
        )

        ref_sr_idx = password_sr_metadata.query(
            "user_name == @ref_user_name and word == @ref_word and iteration == @ref_iteration and session_id == @ref_session_id"
        ).index[0]

        ref_motion_data = motion_passwords_sr[ref_sr_idx]

        ref_mean_head_rotation = Rotation.from_quat(
            quat=ref_motion_data[
                [
                    "head_rot_x",
                    "head_rot_y",
                    "head_rot_z",
                    "head_rot_w",
                ]
            ]
        ).mean()

        query_data_index = int(row["query_idx"])
        query_user_name, query_word, query_iteration, query_session_id = (
            query_embeddings.iloc[query_data_index][
                ["user_name", "word", "iteration", "session_id"]
            ]
        )

        query_sr_idx = password_sr_metadata.query(
            "user_name == @query_user_name and word == @query_word and iteration == @query_iteration and session_id == @query_session_id"
        ).index[0]

        query_motion_data = motion_passwords_sr[query_sr_idx]

        query_mean_head_rotation = Rotation.from_quat(
            quat=query_motion_data[
                [
                    "head_rot_x",
                    "head_rot_y",
                    "head_rot_z",
                    "head_rot_w",
                ]
            ]
        ).mean()

        def _rotate_hands_forward(df, mean_head_rotation):
            df = df.copy()
            df[
                ["right_hand_pos_x", "right_hand_pos_y", "right_hand_pos_z"]
            ] = mean_head_rotation.inv().apply(
                df[["right_hand_pos_x", "right_hand_pos_y", "right_hand_pos_z"]]
            )
            return df

        rotated_query_motion_data = query_motion_data.pipe(
            _rotate_hands_forward, query_mean_head_rotation
        )
        rotated_ref_motion_data = ref_motion_data.pipe(
            _rotate_hands_forward, ref_mean_head_rotation
        )

        rotated_ref_motion_data[
            ["right_hand_pos_x", "right_hand_pos_y"]
        ] -= rotated_ref_motion_data[
            ["right_hand_pos_x", "right_hand_pos_y"]
        ].iloc[
            0
        ]
        rotated_query_motion_data[
            ["right_hand_pos_x", "right_hand_pos_y"]
        ] -= rotated_query_motion_data[
            ["right_hand_pos_x", "right_hand_pos_y"]
        ].iloc[
            0
        ]

        ref_user_id = ref_user_name
        ref_iteration = ref_iteration
        query_user_id = query_user_name
        query_iteration = query_iteration

        for idx, (motion_data, user_id, iteration) in enumerate(
            [
                (rotated_ref_motion_data, ref_user_id, ref_iteration),
                (rotated_query_motion_data, query_user_id, query_iteration),
            ]
        ):
            ax = ax_row[idx]
            dx = np.diff(motion_data["right_hand_pos_x"])
            dy = np.diff(motion_data["right_hand_pos_y"])
            velocities = np.sqrt(dx**2 + dy**2)

            points = np.array(
                [
                    motion_data["right_hand_pos_x"],
                    motion_data["right_hand_pos_y"],
                ]
            ).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(
                segments,
                cmap="coolwarm",
                norm=plt.Normalize(0, 3),
                linewidth=3,
            )
            lc.set_array(velocities)

            ax.add_collection(lc)
            ax.autoscale()
            # ax.scatter(
            #     data=_upsample_df(motion_data),
            #     x="right_hand_pos_x",
            #     y="right_hand_pos_y",
            #     alpha=0.7,
            #     color="blue",
            #     s=1
            # )
            ax.set_yticks([])
            ax.set_xticks([])

            ax.set_title(f"User: {user_id}")
            if idx == 0:
                ax.set_ylabel(f"Similarity: {row['score']:.2f}")
        for ax in axs.flat:
            ax.set_aspect("equal")
        # fig.tight_layout()
        # break
    fp_fn = "FP" if ref_user_id != query_user_id else "FN"
    fig.suptitle(title + f" Comparison of {fp_fn}")
    fig.tight_layout()

    fig.subplots_adjust(top=0.95)
    fig.savefig(f"evaluation/results/{fp_fn}_examples.png")

print("Done")
