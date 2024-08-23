# %%

import numpy as np
from evaluation.compute_embeddings import compute_sl_embeddings
from pytorch_metric_learning.distances import CosineSimilarity

import torch
import rootutils

root_path = rootutils.find_root(
    search_from=__file__, indicator="requirements.txt"
)


def compute_sl_distances(
    embedded_passwords,
    ref_word=None,
    query_word=None,
    ref_session=0,
    query_session=1,
    return_query_users=False,
):
    ref_query = "(session_id == @ref_session)"
    if ref_word:
        ref_query += " and (word == @ref_word)"
    ref_emebddings = (
        embedded_passwords.query(ref_query)
        .filter(like="dim_")
        .groupby(embedded_passwords["user_name"])
        .mean()
    )

    query_query = "(session_id == @query_session)"
    if query_word:
        query_query += " and (word == @query_word)"

    query_embeddings = (
        embedded_passwords.query(query_query)
        .set_index("user_name")
        .filter(like="dim_")
    )

    n_refs = len(ref_emebddings)
    n_queries = len(query_embeddings)

    distances = CosineSimilarity()(
        torch.from_numpy(query_embeddings.to_numpy()),
        torch.from_numpy(ref_emebddings.to_numpy()),
    )

    flattened_distances = distances.flatten()

    flattened_binary_labels = (
        np.tile(ref_emebddings.index.to_numpy(), n_queries).reshape(
            n_queries, n_refs
        )
        == query_embeddings.index.to_numpy()[:, None]
    ).flatten()

    if return_query_users:
        query_users = (
            query_embeddings.index.to_numpy()
            .repeat(len(np.unique(ref_emebddings.index.to_numpy())))
            .flatten()
        )

        return (
            flattened_distances.numpy(),
            flattened_binary_labels,
            query_users,
        )
    else:
        return flattened_distances.numpy(), flattened_binary_labels


if __name__ == "__main__":
    embedded_passwords = compute_sl_embeddings(
        model_path_or_id="btmrrf63", data_encoding="BRA"
    )
    distances, labels = compute_sl_distances(
        embedded_passwords, "Motion", "Motion"
    )
