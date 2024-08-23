# %%
from datetime import datetime

import pandas as pd
import pathlib

from tqdm import tqdm

from password_rules import main_rules
from manual_corrections import main_corrections

from helpers_preprocessing import process_user

data_path = pathlib.Path("main_data")


RUF_XYZ = {
    "right": "x",
    "up": "y",
    "forward": "z",
}
take_id = 0

all_rotations = []
all_word_sequences = []

users = sorted({p.stem.split("_")[0] for p in data_path.glob("*movement*.csv")})

for user_name in tqdm(users):
    sort_by_timestamp = lambda file_path: datetime.strptime(
        file_path.stem.split("_")[2], "%Y-%m-%d"
    )
    motion_files = sorted(
        list(data_path.glob(f"{user_name}_movement_*.csv")), key=sort_by_timestamp
    )
    num_sessions = len(motion_files)

    if num_sessions > 2:
        motion_files = sorted(
            sorted(
                motion_files, key=lambda file_path: file_path.stat().st_size / 1024 ** 2
            )[-2:],
            key=sort_by_timestamp,
        )
        num_sessions = 2
    elif num_sessions < 2:
        continue

    for word_motion_sequence in process_user(
            user_name, motion_files, main_rules, corrections=main_corrections
    ):
        word_motion_sequence = word_motion_sequence.assign(take_id=take_id)
        all_word_sequences.append(word_motion_sequence)

        take_id += 1
        
all_word_sequences_df = pd.concat(all_word_sequences)
all_word_sequences_df.to_feather("intermediate/main_motion_passwords.feather")

