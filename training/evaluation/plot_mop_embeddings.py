# This script uses PCA, Umap and TSNE to visualize the embeddings in 2D
# You need to run `pip install polars umap-learn` for the required packages

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl  # install with `pip install polars`
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP # install with `pip install umap-learn`
import random
import numpy as np
import rootutils

from evaluation.compute_embeddings import compute_sl_embeddings
from evaluation.load_motion_passwords import load_motion_passwords

root_path = rootutils.find_root(search_from="./", indicator="requirements.txt")
prep_info = {"fps": 30, "enc": "BRA"}
motion_passwords, password_metadata = load_motion_passwords(
    root_path
    / f"data/mop/main_passwords_{prep_info['enc']}-enc_{prep_info['fps']}-fps.feather",
)
sml_embeddings = compute_sl_embeddings(
    motion_passwords,
    password_metadata,
    prep_info,
    root_path / "models/abloe7xb.ckpt",
    device="cpu",
)
print(len(sml_embeddings))
# %%

feature_columns = [
    col for col in sml_embeddings.columns if col.startswith("dim_")
]
print(len(feature_columns))
filter_before_dim_reduction = True

np.random.seed(42)
n_random_user = 5
color_palette = sns.color_palette(n_colors=n_random_user)

random_users_to_colorize = random.sample(
    sml_embeddings.user_name.unique().tolist(), n_random_user
)
user_colors = dict(zip(random_users_to_colorize, color_palette))
word_filter = sml_embeddings["word"] == "Motion"


# %%

metadata = sml_embeddings.drop(feature_columns, axis=1)
embeddings = sml_embeddings[feature_columns]
if filter_before_dim_reduction:
    metadata = metadata[word_filter]
    embeddings = embeddings[word_filter]

pca_components = 5
pca = PCA(n_components=pca_components)
reduced_embeddings = pca.fit_transform(embeddings)
print(pca.explained_variance_ratio_)

pca_data = pd.DataFrame(
    reduced_embeddings,
    columns=[f"PCA{index}" for index in range(1, pca_components + 1)],
)
pca_data["session"] = metadata["session_id"]
pca_data["user"] = metadata["user_name"]
pca_data["marker"] = pca_data["session"].apply(lambda x: "x" if x == 1 else "o")
# pca_data = pca_data[pca_data.user.isin(random_users_to_colorize)]

pca_data["color"] = pca_data["user"].apply(
    lambda x: user_colors[x] if x in user_colors else "gray"
)
pca_data["alpha"] = pca_data["user"].apply(
    lambda x: 0.6 if x in user_colors else 0.3
)

fig, ax = plt.subplots(figsize=(8, 8))
if not filter_before_dim_reduction:
    pca_data = pca_data[word_filter]
session_1_data = pca_data[pca_data["session"] == 0]
session_2_data = pca_data[pca_data["session"] == 1]
ax.scatter(
    session_1_data["PCA1"],
    session_1_data["PCA2"],
    c=session_1_data["color"],
    marker="x",
    alpha=session_1_data["alpha"],
    label="Session 1",
)
ax.scatter(
    session_2_data["PCA1"],
    session_2_data["PCA2"],
    c=session_2_data["color"],
    marker="o",
    alpha=session_2_data["alpha"],
    label="Session 2",
)
plt.legend(loc="lower right")
legend = ax.get_legend()
for handle in legend.legend_handles:
    handle.set_color("black")

before_or_after = "before" if filter_before_dim_reduction else "after"
plt.title(f"PCA for word 'Motion' filtered {before_or_after} dim reduction")
plt.show()


# %%
metadata = sml_embeddings.drop(feature_columns, axis=1)
embeddings = sml_embeddings[feature_columns]
if filter_before_dim_reduction:
    metadata = metadata[word_filter]
    embeddings = embeddings[word_filter]

metadata["encoded_user_ids"] = LabelEncoder().fit_transform(
    metadata["user_name"]
)


umap = UMAP(n_neighbors=15, n_components=2)
transformed_embeddings = umap.fit_transform(embeddings, metadata["encoded_user_ids"])

umap_data = pd.DataFrame(transformed_embeddings, columns=["UMAP1", "UMAP2"])
umap_data["session"] = metadata["session_id"]
umap_data["user"] = metadata["user_name"]
umap_data["marker"] = umap_data["session"].apply(
    lambda x: "x" if x == 1 else "o"
)
# umap_data = umap_data[umap_data.user.isin(random_users_to_colorize)]
umap_data["color"] = umap_data["user"].apply(
    lambda x: user_colors[x] if x in user_colors else "gray"
)
umap_data["alpha"] = umap_data["user"].apply(
    lambda x: 0.6 if x in user_colors else 0.3
)
fig, ax = plt.subplots(figsize=(8, 8))
if not filter_before_dim_reduction:
    umap_data = umap_data[word_filter]
session_1_data = umap_data[umap_data["session"] == 0]
session_2_data = umap_data[umap_data["session"] == 1]
ax.scatter(
    session_1_data["UMAP1"],
    session_1_data["UMAP2"],
    c=session_1_data["color"],
    marker="x",
    alpha=session_1_data["alpha"],
    label="Session 1",
)
ax.scatter(
    session_2_data["UMAP1"],
    session_2_data["UMAP2"],
    c=session_2_data["color"],
    marker="o",
    alpha=session_2_data["alpha"],
    label="Session 2",
)
plt.legend(loc="lower right")
legend = ax.get_legend()
for handle in legend.legend_handles:
    handle.set_color("black")
# ax.scatter(df[df.session == 1], x="UMAP1", y="UMAP2", hue="user", marker="o")
before_or_after = "before" if filter_before_dim_reduction else "after"
plt.title(f"UMAP for word 'Motion' filtered {before_or_after} dim reduction")
plt.show()
# %%

metadata = sml_embeddings.drop(feature_columns, axis=1)
embeddings = sml_embeddings[feature_columns]
if filter_before_dim_reduction:
    metadata = metadata[word_filter]
    embeddings = embeddings[word_filter]

tsne = TSNE(n_components=2)
tsne_transformed_embeddings = tsne.fit_transform(
    embeddings, metadata["user_name"]
)

tsne_data = pd.DataFrame(
    tsne_transformed_embeddings, columns=["TSNE1", "TSNE2"]
)
tsne_data["session"] = metadata["session_id"]
tsne_data["user"] = metadata["user_name"]
tsne_data["marker"] = tsne_data["session"].apply(
    lambda x: "x" if x == 1 else "o"
)

tsne_data["color"] = tsne_data["user"].apply(
    lambda x: user_colors[x] if x in user_colors else "gray"
)
tsne_data["alpha"] = tsne_data["user"].apply(
    lambda x: 0.6 if x in user_colors else 0.3
)
fig, ax = plt.subplots(figsize=(8, 8))
if not filter_before_dim_reduction:
    tsne_data = tsne_data[word_filter]
session_1_data = tsne_data[tsne_data["session"] == 0]
session_2_data = tsne_data[tsne_data["session"] == 1]
ax.scatter(
    session_1_data["TSNE1"],
    session_1_data["TSNE2"],
    c=session_1_data["color"],
    marker="x",
    alpha=session_1_data["alpha"],
    label="Session 1",
)
ax.scatter(
    session_2_data["TSNE1"],
    session_2_data["TSNE2"],
    c=session_2_data["color"],
    marker="o",
    alpha=session_2_data["alpha"],
    label="Session 2",
)
plt.legend(loc="lower right")
legend = ax.get_legend()
for handle in legend.legend_handles:
    handle.set_color("black")
# ax.scatter(df[df.session == 1], x="TSNE1", y="TSNE2", hue="user", marker="o")
before_or_after = "before" if filter_before_dim_reduction else "after"
plt.title(f"TSNE for word 'Motion' filtered {before_or_after} dim reduction")
plt.show()
