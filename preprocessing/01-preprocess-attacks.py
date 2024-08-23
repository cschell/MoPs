# %%

from collections import defaultdict
from datetime import datetime
import pathlib

import pandas as pd
from manual_corrections import attack_corrections
from password_rules import attack_rules

from helpers_preprocessing import process_user


data_path = pathlib.Path("attack_data")


RUF_XYZ = {
    "right": "x",
    "up": "y",
    "forward": "z",
}

all_word_sequences = []
all_movement_files = list(data_path.rglob("*_movement_*.csv"))
all_state_files = list(data_path.rglob("*_state_*.csv"))
take_id = 0

users_to_load = ["V1", "V2", "A3", "A4", "A6", "A5", "A7"]

for user_id, user_name in enumerate(users_to_load):
    sort_by_timestamp = lambda file_path: datetime.strptime(
        file_path.stem.split("_")[2], "%Y-%m-%d"
    )
    motion_files = sorted(
        [
            f
            for f in all_movement_files
            if f.stem.split("_")[0].lower() == user_name.lower()
        ],
        key=sort_by_timestamp,
    )

    num_sessions = len(motion_files)

    if num_sessions > 2:
        motion_files = sorted(
            sorted(motion_files, key=lambda file_path: file_path.stat().st_size)[-2:],
            key=sort_by_timestamp,
        )

    word_counter = defaultdict(int)

    for word_motion_sequence in process_user(
        user_name,
        motion_files,
        corrections=attack_corrections,
        rules=attack_rules,
        include_mimic=True,
    ):
        word_motion_sequence = word_motion_sequence.assign(take_id=take_id)
        word, session = word_motion_sequence.iloc[0][["word", "session_id"]]
        if word.startswith("Ballthrowing"):
            continue
        word_counter[f"{word}_{session}"] += 1

        victims = ["V1", "V2"]
        n_iterations = 5 if user_name in victims else 3

        if not word.startswith("Ballthrowing"):
            if user_name in victims and session == 0:
                word = f"{word} [{user_name}]"
            elif word_counter[f"{word}_{session}"] <= n_iterations:
                word = f"{word} [V1]"
            else:
                word = f"{word} [V2]"

        word_motion_sequence["word"] = word

        take_id += 1

        from scipy.spatial.transform import Rotation as R

        rot_cols = [
            "right_hand_rot_x",
            "right_hand_rot_y",
            "right_hand_rot_z",
            "right_hand_rot_w",
        ]

        if user_name not in victims or session == 1:
            rotation_correction = R.from_euler("x", 60, degrees=True)

            for joint in ["left_hand", "right_hand"]:
                rot_cols = [f"{joint}_rot_{xyzw}" for xyzw in "xyzw"]

                word_motion_sequence[rot_cols] = (
                    R.from_quat(word_motion_sequence[rot_cols]) * rotation_correction
                ).as_quat()

        all_word_sequences.append(word_motion_sequence)

        take_id += 1
all_word_sequences_df = pd.concat(all_word_sequences)
all_word_sequences_df.to_feather("intermediate/attack_motion_passwords.feather")
