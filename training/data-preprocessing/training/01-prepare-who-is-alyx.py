# %%

from pathlib import Path
import numpy as np
import pandas as pd

import motion_learning_toolbox as mlt
from tqdm import tqdm
import rootutils

root_path = rootutils.find_root(search_from=__file__, indicator="requirements.txt")


JOINTS = ["head", "left_hand", "right_hand"]
FPS = 30
DATA_ENCODING = "BR"

column_mapping = {
    "delta_time_ms": "delta_time_ms",
    "hmd_pos_x": "head_pos_x",
    "hmd_pos_y": "head_pos_y",
    "hmd_pos_z": "head_pos_z",
    "hmd_rot_x": "head_rot_x",
    "hmd_rot_y": "head_rot_y",
    "hmd_rot_z": "head_rot_z",
    "hmd_rot_w": "head_rot_w",
    "left_controller_pos_x": "left_hand_pos_x",
    "left_controller_pos_y": "left_hand_pos_y",
    "left_controller_pos_z": "left_hand_pos_z",
    "left_controller_rot_x": "left_hand_rot_x",
    "left_controller_rot_y": "left_hand_rot_y",
    "left_controller_rot_z": "left_hand_rot_z",
    "left_controller_rot_w": "left_hand_rot_w",
    "right_controller_pos_x": "right_hand_pos_x",
    "right_controller_pos_y": "right_hand_pos_y",
    "right_controller_pos_z": "right_hand_pos_z",
    "right_controller_rot_x": "right_hand_rot_x",
    "right_controller_rot_y": "right_hand_rot_y",
    "right_controller_rot_z": "right_hand_rot_z",
    "right_controller_rot_w": "right_hand_rot_w",
}

coordinate_system = {
    "forward": "z",
    "right": "x",
    "up": "y",
}

encoding_functions = {
    "BR": lambda df: df,
    "BRV": mlt.to_velocity,
    "BRA": mlt.to_acceleration,
}

# %%

wia_path = Path("/storage/cs/who-is-alyx/")
storage_path = root_path / f"data/who-is-alyx/intermediate/{DATA_ENCODING}-enc_{FPS}-fps/"
storage_path.mkdir(exist_ok=True)

csv_files = list(wia_path.glob("**/vr-controllers*.csv"))
for file_name in tqdm(csv_files):
    user, session, _ = file_name.parts[-3:]

    raw_recording = (
        pd.read_csv(file_name)
        .rename(columns=column_mapping)
        .assign(delta_time_ms=lambda df: pd.to_timedelta(df["delta_time_ms"], unit="ms"))
        .set_index("delta_time_ms")[list(column_mapping.values())[1:]]
    )

    assert raw_recording.select_dtypes(include=[object]).shape[1] == 0, "DataFrame contains non-numeric columns"

    recording = (
        raw_recording.pipe(mlt.resample, FPS, JOINTS)
        .pipe(mlt.to_body_relative, ["left_hand", "right_hand"], coordinate_system)
        .pipe(encoding_functions[DATA_ENCODING])
    )[2:]

    recording.astype("float16").to_feather(storage_path / f"wia_{user}_{session}.feather")
