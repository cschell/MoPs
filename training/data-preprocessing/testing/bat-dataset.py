from pathlib import Path

import motion_learning_toolbox as mlt
import numpy as np
import pandas as pd
import rootutils
from tqdm import tqdm

root_path = rootutils.find_root(
    search_from=__file__, indicator="requirements.txt"
)
DEVICE = "cpu"
FPS = 30

DATA_ENCODING = "BRA"

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

# path to dataset after you processed it with script from https://github.com/cschell/xr-motion-dataset-conversion-scripts
dataset_path = Path("path/to/miller/dataset")


all_recordings = []
for idx, file in tqdm(list(enumerate(dataset_path.glob("*.parquet")))):
    device, user, session, iteration = file.stem.split("_")
    user, session, iteration = int(user), int(session), int(iteration)

    recording = (
        pd.read_parquet(file)
        .assign(
            delta_time_ms=lambda df: pd.to_timedelta(
                df["delta_time_ms"], unit="ms"
            )
        )
        .set_index("delta_time_ms")
        .pipe(
            mlt.resample, FPS, joint_names=["right_hand", "left_hand", "head"]
        )
        .pipe(
            mlt.to_body_relative, ["left_hand", "right_hand"], coordinate_system
        )
        .pipe(encoding_functions[DATA_ENCODING])
        .sort_index(axis=1)
        .dropna()
        .assign(
            device=device,
            user_id=user,
            session_id=session,
            iteration=iteration,
            take_id=idx,
            word="throw",
            frame_idx=lambda df: np.arange(len(df)),
        )
    )

    all_recordings.append(recording)

passwords = pd.concat(all_recordings, ignore_index=True)
passwords.to_parquet(
    root_path
    / "data/bat"
    / f"all-users_all-devices_{FPS}-fps_{DATA_ENCODING}-enc.parquet"
)
