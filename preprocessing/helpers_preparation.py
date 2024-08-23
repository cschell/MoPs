import pandas as pd
from tqdm import tqdm
import motion_learning_toolbox as mlt
import numpy as np


encoding_functions = {
    "SR": lambda df: df,
    "BR": lambda df: df,
    "BRV": mlt.to_velocity,
    "BRA": mlt.to_acceleration,
}

RUF_XYZ = {
    "right": "x",
    "up": "y",
    "forward": "z",
}


def encode_passwords(passwords, target_fps: int, encoding: str, corrections, rules):
    processed_passwords = []

    for take_id, password in tqdm(list(passwords.groupby("take_id"))):
        user_name, iteration, hand, word, session = password.iloc[0][
            ["user_name", "iteration", "hand", "word", "session_id"]
        ]

        correction = corrections.get(user_name, {}).get(
            (word.lower(), iteration, session, hand), {}
        )

        if not rules.get("skip_selected_frames"):
            password = password.query("selected_frames").copy().reset_index(drop=True)

        duration = (
            password["timestamp"].iloc[-1] - password["timestamp"].iloc[0]
        ).total_seconds()
        actual_fps = len(password) / duration

        valid_after_rules = (actual_fps >= rules["min_fps"]) and (
            rules["min_word_duration"] <= duration <= rules["max_word_duration"]
        )

        if not correction.get("valid", valid_after_rules):
            print(
                f"excluding run #{take_id} ({user_name=}, {word=}, {iteration=}, {session=}, {hand=}))"
            )
            continue

        password["timestamp"] -= password["timestamp"].iloc[0]
        processed_password = (
            password.set_index("timestamp")
            .sort_index()
            .pipe(
                mlt.resample,
                target_fps,
                joint_names=["head", "left_hand", "right_hand"],
            )
            .pipe(
                mlt.to_body_relative if encoding != "SR" else lambda df, *_, **__: df,
                target_joints=["left_hand", "right_hand"],
                reference_joint="head",
                coordinate_system=RUF_XYZ,
            )
            .pipe(encoding_functions[encoding])
            .astype("float16")
            .sort_index(axis=1)
            .assign(
                frame_idx=lambda df: np.arange(len(df)),
                delta_seconds=lambda df: np.linspace(0, duration, len(df)),
                word=password["word"].iloc[0],
                hand=password["hand"].iloc[0],
                iteration=password["iteration"].iloc[0].astype("uint8"),
                user_name=password["user_name"].iloc[0],
                session_id=password["session_id"].iloc[0].astype("uint8"),
                take_id=password["take_id"].iloc[0].astype("uint16"),
            )
        )

        processed_passwords.append(processed_password)

    processed_passwords = (
        pd.concat(processed_passwords)
        .reset_index(drop=True)
        .sort_values(["take_id", "frame_idx"])
    )

    return processed_passwords
