import pandas as pd
from tqdm import tqdm


def load_bat_dset(dataset_path, device, query_addition="True"):
    ball_throws = pd.read_parquet(dataset_path).query(f"device == @device and ({query_addition})")

    n_ball_throws = len(ball_throws["take_id"].unique())
    user_names = []

    session = []
    words = []
    feature_columns = [c for c in ball_throws.columns if "_pos_" in c or "_rot_" in c]

    motion_ball_throws = []

    for _, passwd in tqdm(ball_throws.groupby("take_id"), total=n_ball_throws, desc=f"load ball throws from Miller dataset"):
        seq = passwd[feature_columns]
        motion_ball_throws.append(seq)
        user_names.append(passwd.iloc[0]["user_id"])
        words.append(passwd.iloc[0]["word"])
        session.append(passwd.iloc[0]["session_id"])

    metadata = pd.DataFrame(zip(user_names, words, session), columns=["user_name", "word", "session_id"])

    return motion_ball_throws, metadata
