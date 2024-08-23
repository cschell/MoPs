# %%

import numpy as np
from tqdm import tqdm
from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler

import rootutils

from evaluation.helpers import _hash_list_of_ndarrays


root_path = rootutils.find_root(
    search_from=__file__, indicator="requirements.txt"
)


def compute_fd_distances(
    motion_sequences,
    sequence_metadata,
    ref_word=None,
    query_word=None,
    ref_session=None,
    query_session=None,
):
    cache_output_path = (
        root_path
        / f"data/cache"
        / f"dtw_distances_{_hash_list_of_ndarrays(motion_sequences)}_{ref_word=}_{query_word=}_{ref_session=}_{query_session=}.npz"
    )

    if cache_output_path.exists():
        print(f"loading cached dtw distances from {cache_output_path}")
        loaded = np.load(cache_output_path)
        distances = loaded["distances"]
        labels = loaded["labels"]
        return distances, labels

    distances = []
    labels = []
    users = sequence_metadata.query("session_id == @ref_session")[
        "user_name"
    ].unique()
    verification_users = sequence_metadata.query("session_id == @query_session")[
        "user_name"
    ].unique()

    motion_sequences = [
        seq[[c for c in seq.columns if "_pos_" in c]].to_numpy()
        for seq in motion_sequences
    ]
    scaler = StandardScaler()
    scaler.fit(np.concatenate(motion_sequences))

    for claimed_user in tqdm(users, desc="computing DTW distances"):
        ref_query = (
            "(session_id == @ref_session) and (user_name == @claimed_user)"
        )
        if ref_word:
            ref_query += "  and (word == @ref_word)"
        reference_idxs = sequence_metadata.query(ref_query).index.to_numpy()
        references = [motion_sequences[idx] for idx in reference_idxs]

        for verification_user in tqdm(verification_users, leave=False):
            query_query = "(session_id == @query_session) and (user_name == @verification_user)"
            if query_word:
                query_query += "  and (word == @query_word)"
            query_idxs = sequence_metadata.query(query_query).index.to_numpy()
            queries = [motion_sequences[idx] for idx in query_idxs]

            for query in queries:
                query_distances = []
                for ref in references:
                    distance = dtw(
                        scaler.transform(ref), scaler.transform(query)
                    )
                    query_distances.append(-distance)

                if query_distances: 
                    distances.append(np.mean(query_distances))
                    labels.append(int(claimed_user == verification_user))
                assert len(distances) == len(labels)
    assert len(distances) == len(labels)
    distances, labels = np.array(distances), np.array(labels)
    assert len(distances) == len(labels)
    assert not np.isnan(distances).any()
    np.savez(cache_output_path, distances=distances, labels=labels)

    return distances, labels

