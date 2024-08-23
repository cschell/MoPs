# %%
import random
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import rootutils

root_path = rootutils.find_root(search_from=__file__, indicator="requirements.txt")
# %%

PRECISION = 16
SECONDS = 15
FPS = 30
DATA_ENCODING = "BR"
processed_wia = root_path / f"data/who-is-alyx/intermediate/{DATA_ENCODING}-enc_{FPS}-fps"
assert processed_wia.exists()
# %%

users = list(sorted([path.stem.split("_")[1] for path in processed_wia.glob("*.feather")]))

# %%

count = Counter(users)

double_session_users = [item for item, c in count.items() if c > 1]

num_train_users = 60
num_val_users = 11

assert len(double_session_users) == num_train_users + num_val_users

random.seed(42)
val_users = random.sample(users, num_val_users)
train_users = sorted(list(set([u for u in users if u not in val_users])))

# %%

SEQUENCE_LENGTH = FPS * SECONDS

train_X = []
train_y = []
val_X = []
val_y = []


for path in tqdm(list(processed_wia.glob("*.feather"))):
    recording = pd.read_feather(path).sort_index(axis=1)

    _, user, date = path.stem.split("_")

    num_sequences = len(recording) // SEQUENCE_LENGTH

    sequences = np.array(np.vsplit(recording[: num_sequences * SEQUENCE_LENGTH].to_numpy(), num_sequences))
    ground_truth = np.array([user] * len(sequences))

    if user in train_users:
        train_X.append(sequences)
        train_y.append(ground_truth)
    elif user in val_users:
        val_X.append(sequences)
        val_y.append(ground_truth)
    else:
        raise Exception(f"Unknown user {user}")

# %%

np.savez(
    f"data/who-is-alyx/wia-{PRECISION}_{DATA_ENCODING}-enc_{FPS}-fps_{SECONDS}-seq",
    train_X=np.vstack(train_X),
    train_y=np.concatenate(train_y),
    val_X=np.vstack(val_X),
    val_y=np.concatenate(val_y),
)
