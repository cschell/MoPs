import hashlib
from typing import List
import numpy as np
import pandas as pd


def _hash_ndarray(np_array: np.ndarray):
    if isinstance(np_array, pd.DataFrame):
        np_array = np_array.to_numpy()
    tensor_bytes = np_array.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def _hash_list_of_ndarrays(list_of_ndarrays: List[np.ndarray]):
    hashes_list = [_hash_ndarray(pwd) for pwd in list_of_ndarrays]
    combined_hashes = "".join(hashes_list).encode()
    return hashlib.sha256(combined_hashes).hexdigest()[:10]
