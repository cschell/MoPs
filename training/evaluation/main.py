# NOTE: This file is meant to be run with PyCharm's "Scientific Mode" or VSCode's interactive mode

# %% Boilerplate

from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import RocCurveDisplay, auc, roc_auc_score, roc_curve

from evaluation.compute_embeddings import compute_sl_embeddings
from evaluation.compute_fd_distances import compute_fd_distances
from evaluation.compute_sl_distances import compute_sl_distances
from evaluation.load_bat_dset import load_bat_dset
from evaluation.load_motion_passwords import load_motion_passwords
from evaluation.matplotlib_settings import (FULL_COLUMN_WIDTH,
                                            SINGLE_COLUMN_WIDTH,
                                            apply_default_matplotlib_settings,
                                            colors)

apply_default_matplotlib_settings()

root_path = rootutils.find_root(search_from="./", indicator="requirements.txt")

# %% Settings

prep_info = {"fps": 30, "enc": "BRA"}
sl_model_path = Path(root_path / "models/abloe7xb.ckpt")

assert sl_model_path.exists()

# %% Load datasets

mop_path = (
    root_path
    / "data/mop"
    / f"main_passwords_{prep_info['enc']}-enc_{prep_info['fps']}-fps.feather"
)

bat_path = (
    root_path
    / "data/bat"
    / f"all-users_all-devices_{prep_info['fps']}-fps_{prep_info['enc']}-enc.parquet"
)

bat_br_path = (
    root_path
    / "data/bat"
    / f"all-users_all-devices_{prep_info['fps']}-fps_BR-enc.parquet"
)

attack_path = (
    root_path
    / f"data/mop/attack_passwords_{prep_info['enc']}-enc_{prep_info['fps']}-fps.feather"
)

mop_signatures, mop_metadata = load_motion_passwords(mop_path, hand="right")
left_motion_passwords, left_password_metadata = load_motion_passwords(
    mop_path, hand="left"
)
bat_signatures, bat_metadata = load_bat_dset(bat_path, device="Quest")
bat_signatures_br, bat_metadata_br = load_bat_dset(bat_br_path, device="Quest")

# %% Set Thresholds (Section 5.3)

thresh_names = ["strict", "moderate", "lenient", "permissive"]
target_fprs = [0.001, 0.01, 0.1, 0.25]

# %% Compute Similarity-Learning model emebddings for MoP and BaT

mop_embeddings = compute_sl_embeddings(
    motion_sequences=mop_signatures,
    sequence_metadata=mop_metadata,
    preprocessing_info=prep_info,
    model_path_or_id=sl_model_path,
    device="cpu",
)

bat_embeddings = compute_sl_embeddings(
    motion_sequences=bat_signatures,
    sequence_metadata=bat_metadata,
    preprocessing_info=prep_info,
    model_path_or_id=sl_model_path,
    device="cpu",
)

# %% Compute distances for BaT (i.e., SL+BaT and FD+BaT)

sl_bat_distances, sl_bat_labels = compute_sl_distances(
    bat_embeddings,
    ref_session=1,
    query_session=2,
)

fd_bat_distances, fd_bat_labels = compute_fd_distances(
    bat_signatures_br,
    bat_metadata_br,
    ref_session=1,
    query_session=2,
)

# %% Compute distances for MoP (i.e., SL+MoP and FD+MoP)


sl_mop_results = []
fd_mop_results = []

ref_words = ["Motion", "Password", "word_0", "word_1"]
query_words = ["Motion", "Secure", "word_0", "word_1"]

for ref_word in ref_words:
    for query_word in query_words:

        sl_mop_dists, sl_mop_labls = compute_sl_distances(
            mop_embeddings,
            ref_word,
            query_word,
            ref_session=0,
            query_session=1,
        )

        sl_mop_results.append(
            pd.DataFrame(
                np.c_[sl_mop_dists, sl_mop_labls], columns=["distance", "label"]
            ).assign(ref_word=ref_word, query_word=query_word)
        )

        fd_mop_dists, fd_mop_labls = compute_fd_distances(
            mop_signatures,
            mop_metadata,
            ref_word=ref_word,
            query_word=query_word,
            ref_session=0,
            query_session=1,
        )

        fd_mop_results.append(
            pd.DataFrame(
                np.c_[fd_mop_dists, fd_mop_labls], columns=["distance", "label"]
            ).assign(ref_word=ref_word, query_word=query_word)
        )


sl_mop_results = pd.concat(sl_mop_results, ignore_index=True)
fd_mop_results = pd.concat(fd_mop_results, ignore_index=True)

# %% Results for 6.1 'Genuine Attempts & Uninformed Attacks'


def generate_roc(conditions, legend_title, ax=None):

    if ax == None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    for label, dist, labels, plot_kwargs in conditions:
        fpr, tpr, thresholds = roc_curve(labels, dist)
        roc_auc = roc_auc_score(labels, dist)

        # print(label, roc_auc)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=label
        )
        display.plot(ax=ax, **plot_kwargs)
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Acceptance Rate")
    ax.legend(loc="best", title=legend_title)
    return fig, ax


def bootstrapped_zoomed_roc(
    conditions, target_fprs, n_bootstraps=1000, ax=None
):
    rng_seed = 42  # Control the randomness
    base_fpr = np.linspace(0, 1, 1000 + 1)

    all_bootstrapped_results = []

    target_thresholds = defaultdict(list)
    scored_tars = defaultdict(list)
    scored_eers = dict()
    for label, dist, labels, plot_kwargs in conditions:

        orig_fpr, orig_tpr, orig_thresholds = roc_curve(labels, dist)

        resampled_tpr = np.interp(base_fpr, orig_fpr, orig_tpr)
        resampled_threshs = np.interp(base_fpr, orig_fpr, orig_thresholds)

        min_index = np.argmin(np.abs(resampled_tpr - (1 - base_fpr)))
        eer = 1 - resampled_tpr[min_index]
        scored_eers[label] = eer
        print(label, eer)

        bootstrapped_roc_auc = []

        # To store the TPRs for the CI calculation
        boostrapped_tprs = []
        # tpr_upper = []

        rng = np.random.RandomState(rng_seed)

        for i in range(n_bootstraps):
            # Bootstrap by sampling with replacement on the indices of `dist`
            indices = rng.randint(0, len(dist), len(dist))
            if len(np.unique(labels[indices])) < 2:
                continue  # Skip this loop if not both classes present

            fpr, tpr, _ = roc_curve(labels[indices], dist[indices])
            roc_auc = auc(fpr, tpr)
            bootstrapped_roc_auc.append(roc_auc)

            tpr_interp = np.interp(base_fpr, fpr, tpr)
            boostrapped_tprs.append(tpr_interp)
            # tpr_upper.append(tpr_interp)

        # Convert lists to arrays for easier manipulation
        boostrapped_tprs = np.array(boostrapped_tprs)

        # Calculate mean and 95% CI

        alpha = 0.95
        p_lower = ((1.0 - alpha) / 2.0) * 100
        p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
        tpr_lower = np.clip(
            np.percentile(boostrapped_tprs, p_lower, axis=0),
            a_min=0.0,
            a_max=1.0,
        )
        tpr_upper = np.clip(
            np.percentile(boostrapped_tprs, p_upper, axis=0),
            a_min=0.0,
            a_max=1.0,
        )

        all_bootstrapped_results.append(
            (
                label,
                resampled_tpr,
                tpr_lower,
                tpr_upper,
                boostrapped_tprs,
                plot_kwargs,
            )
        )

        # Original ROC Curve
        fpr, tpr, _ = roc_curve(labels, dist)
        roc_auc = auc(fpr, tpr)

        for fpr in target_fprs:
            idx = np.argmax(fpr == base_fpr)
            target_thresholds[label].append(resampled_threshs[idx])
            scored_tars[label].append(resampled_tpr[idx])

    if ax == None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    n_conds = len(all_bootstrapped_results)
    for x_idx, target_fpr in enumerate(target_fprs):
        idx = np.argmax(base_fpr == target_fpr)

        assert idx
        for cond_idx, (
            label,
            resampled_tpr,
            tpr_lower,
            tpr_upper,
            boostrapped_tprs,
            plot_kwargs,
        ) in enumerate(all_bootstrapped_results):
            og = resampled_tpr[idx]
            shift = ((((cond_idx + 1) / n_conds) - 0.5) - 0.25 / 2) * 0.8

            lower = max(og - tpr_lower[idx], 0)
            upper = min(tpr_upper[idx] - og, 1)

            if x_idx != 0:
                label = None

            x_pos = x_idx + shift

            ax.errorbar(
                x_pos,
                og,
                [(lower,), (upper,)],
                fmt="o",
                c=plot_kwargs["c"],
                ecolor="black",
                markersize=4,
                linewidth=0.6,
                markeredgewidth=0.3,
                markeredgecolor="#000000",
                capsize=2,
                label=label,
            )

            samples = boostrapped_tprs[::20, idx]
            n_samples = len(samples)
            jittered_x_pos = ((np.random.rand(n_samples) - 0.5)) * 0.12 + x_pos
            ax.scatter(
                jittered_x_pos, samples, c=plot_kwargs["c"], alpha=0.1, s=0.8
            )

    xticks = np.arange(len(target_fprs))
    ax.set_xticks(xticks)
    ax.set_xticklabels(target_fprs)
    ax.grid(True)

    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Acceptance Rate")
    ax.set_title("ROC 95% Boostrapped")
    ax.set_ylim(-0.02, 1.02)

    return fig, ax, target_thresholds, scored_tars, scored_eers


sl_mop_color = colors["vermillion"]  # "#785ef0"
sl_bat_color = colors["reddish_purple"]  # "#ffb000"
fd_mop_color = colors["blue"]  # "#648fff"
fd_bat_color = colors["sky_blue"]  # "#fe6000"

same_genuine_different_attack_query = "((label == 1) and (ref_word == query_word)) or ((label == 0) and (ref_word != query_word))"

happy_path_sl_mop_distances, happy_path_sl_mop_labels = (
    sl_mop_results.query(same_genuine_different_attack_query)[
        ["distance", "label"]
    ]
    .to_numpy()
    .T
)

happy_path_fd_mop_distances, happy_path_fd_mop_labels = (
    fd_mop_results.query(same_genuine_different_attack_query)[
        ["distance", "label"]
    ]
    .to_numpy()
    .T
)

happy_path_conditions = [
    (
        "SL+BaT",
        sl_bat_distances,
        sl_bat_labels,
        dict(lw=1, linestyle="-", c=sl_bat_color),
    ),
    (
        "SL+MoP",
        happy_path_sl_mop_distances,
        happy_path_sl_mop_labels,
        dict(lw=1, linestyle="-", c=sl_mop_color),
    ),
    (
        "FD+BaT",
        fd_bat_distances,
        fd_bat_labels,
        dict(lw=1, linestyle="--", c=fd_bat_color),
    ),
    (
        "FD+MoP",
        happy_path_fd_mop_distances,
        happy_path_fd_mop_labels,
        dict(lw=1, linestyle="--", c=fd_mop_color),
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN_WIDTH, 2.5), sharey=True)

ax = axes[0]
generate_roc(happy_path_conditions, "method + signature", ax=axes[0])
ax.set_title("ROC Curve")
ax.set_aspect("auto")


ax = axes[1]
_, _, thresholds, scored_tars, eers = bootstrapped_zoomed_roc(
    happy_path_conditions, target_fprs, ax=ax
)
ax.set_title("Detailed View")
ax.set_ylabel("")


legend_handles = axes[0].legend().legend_handles
axes[0].legend().remove()
fig.suptitle("Comparison of Verification Accuracy")
fig.tight_layout()
fig.subplots_adjust(bottom=0.3)
fig.legend(
    handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2
)
fig.savefig("evaluation/results/happy_path.pgf")

# %% Results for 6.2
same_word_query = "ref_word == query_word"

same_word_sl_mop_distances, same_word_sl_mop_labels = (
    sl_mop_results.query(same_word_query)[["distance", "label"]].to_numpy().T
)

same_word_fd_mop_distances, same_word_fd_mop_labels = (
    fd_mop_results.query(same_word_query)[["distance", "label"]].to_numpy().T
)

different_word_query = "ref_word != query_word"

different_word_sl_mop_distances, different_word_sl_mop_labels = (
    sl_mop_results.query(different_word_query)[["distance", "label"]]
    .to_numpy()
    .T
)

different_word_fd_mop_distances, different_word_fd_mop_labels = (
    fd_mop_results.query(different_word_query)[["distance", "label"]]
    .to_numpy()
    .T
)

different_word_conditions = [
    (
        "ref. != query",
        different_word_sl_mop_distances,
        different_word_sl_mop_labels,
        dict(lw=2, linestyle="-", c=sl_mop_color),
    ),
    (
        "ref. == query",
        same_word_sl_mop_distances,
        same_word_sl_mop_labels,
        dict(lw=2, linestyle="-", c=sl_mop_color),
    ),
]


# %%

conditions = different_word_conditions
ths = thresholds["SL+MoP"]
threshold_names = thresh_names
n_bootstraps = 1000

rng_seed = 42

import seaborn as sns

fig, axes = plt.subplots(1, 2, sharey=True)


for idx, tar_or_far in enumerate(["tar", "far"]):
    all_bootstrapped_results = []
    ax = axes[idx]

    for label, dist, labels, plot_kwargs in conditions:
        if tar_or_far.lower() == "tar":
            dist = dist[labels.astype(bool)]
        elif tar_or_far.lower() == "far":
            dist = dist[~labels.astype(bool)]
        else:
            raise Exception("Unknown tar_or_far")

        def get_rates(distances):
            rates = []
            for th in ths:
                rate = (distances >= th).mean()
                rates.append(rate)

            return np.array(rates)

        rng = np.random.RandomState(rng_seed)

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(dist), len(dist))
            bded_rates = get_rates(dist[indices])

            for th_n, bded_rate in zip(threshold_names, bded_rates):
                all_bootstrapped_results.append((label, th_n, bded_rate))

    bootstrapped_results = pd.DataFrame(
        all_bootstrapped_results,
        columns=[
            "label",
            "threshold_name",
            "bootstrapped_rate",
        ],
    )

    sns.pointplot(
        bootstrapped_results,
        x="threshold_name",
        y="bootstrapped_rate",
        hue="label",
        dodge=True,
        palette=[colors["sky_blue"], colors["vermillion"]],
        linewidth=1,
        marker="d",
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.3,
        alpha=0.9,
        zorder=2,
        ax=ax,
        legend=tar_or_far == "far",
    )
    sns.stripplot(
        bootstrapped_results.groupby(["label", "threshold_name"]).nth(
            range(200)
        ),
        x="threshold_name",
        y="bootstrapped_rate",
        hue="label",
        size=1,
        palette="dark:#666666",
        ax=ax,
        alpha=0.5,
        legend=False,
        zorder=1,
    )

    ax.set_xticks(range(4))
    ax.set_xticklabels(threshold_names, rotation=20, ha="right")
    ax.grid(True)

    ax.set_xlabel("Thresholds")
    if tar_or_far == "tar":
        ax.set_title("Genuine Attempts")
        ax.set_ylabel("Acceptance Rate")
    else:
        ax.set_title("(Partially) Uninformed Attacks")
        ax.legend().set_title("")

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    means = bootstrapped_results.groupby(["label", "threshold_name"])[
        "bootstrapped_rate"
    ].mean()

    diffs = 1 - means["ref. != query"] / means["ref. == query"]

    print(tar_or_far)
    print(diffs[thresh_names].round(4))

legend_handles = axes[1].legend().legend_handles
axes[1].legend().remove()
fig.suptitle("SL+MoP: Correct vs. Wrong Words")
fig.tight_layout()
fig.legend(
    handles=legend_handles,
    loc="center",
    bbox_to_anchor=(0.6, 0.5),
    ncol=1,
    framealpha=0.9,
)
fig.savefig("evaluation/results/same-vs-wrong-words.pgf")

# %%


df = (
    pd.DataFrame(
        zip(
            thresh_names,
            np.array(target_fprs) * 100,
            np.array(scored_tars["SL+MoP"]) * 100,
            thresholds["SL+MoP"],
        ),
        columns=[
            "Threshold",
            "FAR",
            "TAR",
            "Similarity",
        ],
    )
    .style.format(
        {"FAR": "{:.2g}\%", "TAR": "{:.2g}\%"},
        precision=3,
    )
    .hide(axis=0)
)

print(df)

latex = df.to_latex(
    label="tab:threshs",
    caption="The four thresholds we selected as benchmarks for the SL+MoP model during our analyses, based on the results of \Cref{...}; `Similarity' lists the required minimum cosine similarity between two password embeddings for a verification success.",
    position="rr|rr",
    hrules=True,
)

print(latex)


# %%

victims = ["V1", "V2"]

attack_passwords, attack_metadata = load_motion_passwords(
    attack_path,
    hand="right",
    encode_words=False,
    query_addition="not word.str.startswith('Ball')",
)

attack_metadata.loc[
    attack_metadata.eval("user_name not in @victims"), "session_id"
] = 1


# %%

attack_embeddings = compute_sl_embeddings(
    metric="precision_at_1",
    motion_sequences=attack_passwords,
    sequence_metadata=attack_metadata,
    preprocessing_info=prep_info,
    model_path_or_id=sl_model_path,
    device="cpu",
)

attack_distances, attack_labels = compute_sl_distances(
    attack_embeddings,
    ref_session=0,
    query_session=1,
)

# %%
import numpy as np
import rootutils

attack_words = [
    "Motion [V1]",
    "Motion [V2]",
    "Motion Password [V1]",
    "Motion Password [V2]",
    "Secure [V1]",
    "Secure [V2]",
]

attack_results = []

for word in attack_words:
    fd_mop_dists, fd_mop_labls, query_users = compute_sl_distances(
        embedded_passwords=attack_embeddings,
        ref_word=word,
        query_word=word,
        ref_session=0,
        query_session=1,
        return_query_users=True,
    )

    attack_results.append(
        pd.DataFrame(
            np.c_[fd_mop_dists, fd_mop_labls], columns=["distance", "label"]
        ).assign(
            ref_word=word,
            query_word=word,
            attacker=query_users,
        )
    )

attack_results = pd.concat(attack_results, ignore_index=True)

attack_distances, attack_labels = (
    attack_results.query("query_word.str.contains('Motion \[')")[
        ["distance", "label"]
    ]
    .to_numpy()
    .T
)

attack_conditions = [
    (
        "Attacks",
        attack_distances,
        attack_labels,
        dict(lw=2, linestyle="-", c=sl_bat_color),
    ),
    (
        "SL+MoP",
        happy_path_sl_mop_distances,
        happy_path_sl_mop_labels,
        dict(lw=2, linestyle="-", c=sl_mop_color),
    ),
]

generate_roc(attack_conditions, "attack")

# %%

attack_results["victim"] = attack_results["ref_word"].str.extract("\[(.+)\]")
attack_results["word"] = attack_results["ref_word"].str.extract("(.+) \[.+\]")

mean_attacks = attack_results.groupby(["label", "victim", "word", "attacker"])[
    "distance"
].mean()
min_attacks = attack_results.groupby(["label", "victim", "word", "attacker"])[
    "distance"
].min()
max_attacks = attack_results.groupby(["label", "victim", "word", "attacker"])[
    "distance"
].max()

breaches = []
threshold_breaches = {name: [] for name in thresh_names}


for v in victims:
    for w in attack_results["word"].unique():
        genuine_score = mean_attacks.loc[(1, v, w, v)]

        attacks = max_attacks.loc[(0, v, w)]
        current_breaches = attacks > genuine_score

        for a, breach in attacks[current_breaches].items():
            breaches.append((v, w, a, breach))

        for tn, th in zip(thresh_names, thresholds["SL+MoP"]):
            th_breaches = attacks > th

            for a, breach in attacks[th_breaches].items():
                threshold_breaches[tn].append((v, w, a, breach))


for thresh_name, breaches in threshold_breaches.items():
    print("Breaches with threshold", thresh_name)
    [print(b) for b in threshold_breaches[thresh_name]]
    print("\n\n")

# %% Fully informed Attacks (Section 6.3)

attack_results["victim"] = attack_results["ref_word"].str.extract("\[(.+)\]")
attack_results["word"] = attack_results["ref_word"].str.extract("(.+) \[.+\]")

fig, axes = plt.subplots(1, 3, figsize=(FULL_COLUMN_WIDTH, 2), sharey=True)

styles = {
    "V1": "v",
    "V2": "^",
}

victim_colors = {
    "V1": "black",
    "V2": "#666666",
}

th = thresholds["SL+MoP"]


# Example base color (red in this case)
base_color = "#FF0000"

# Create a monochrome colormap
monochrome_cmap = LinearSegmentedColormap.from_list(
    "monochrome",
    ["white", colors["blue"]],
).reversed()

attack_study_users = ["V1", "V2", "A1", "A2", "A3", "A4", "A6", "A5", "A7"]

for word_idx, (word, word_df) in enumerate(attack_results.groupby("word")):
    ax = axes[word_idx]

    th_bgs = []
    for idx, (a, b) in enumerate(pairwise([1] + th)):
        th_bg = ax.axhspan(
            a, b, color=monochrome_cmap(idx * 70), alpha=0.9, ec=None
        )
        th_bgs.append(th_bg)

    for idx, (v_name, df) in enumerate(
        word_df.query("victim == attacker").groupby("attacker"), start=-2
    ):
        x = [attack_study_users.index(v_name)] * len(df)

        ax.scatter(
            x=x,
            y=df["distance"],
            s=20,
            lw=0.5,
            edgecolors="white",
            c=victim_colors[v_name],
            marker=styles[v_name],
            alpha=0.8,
        )
        ax.set_ylim(0.2, 1.05)

    for a_name, attacker_df in word_df.query("victim != attacker").groupby(
        "attacker"
    ):
        for v_name, df in attacker_df.groupby("victim"):
            x = [attack_study_users.index(a_name.replace("V", "A"))] * len(df)
            ax.scatter(
                x=x,
                y=df["distance"],
                s=20,
                alpha=0.7,
                marker=styles[v_name],
                c=victim_colors[v_name],
                edgecolors="white",
                lw=0.5,
            )

    ax.axvline(
        1.5, ymin=-0.1, linestyle="--", lw=1, color="black", clip_on=False
    )
    ax.set_title(f"Word '{word}'")

    ax.set_xticks(range(len(attack_study_users)))
    ax.set_xticklabels(attack_study_users)

axes[0].set_ylabel("similarity between ref. and query")

custom_handles = [
    Line2D(
        [0],
        [0],
        marker="v",
        markersize=3.5,
        color=victim_colors["V1"],
        lw=0,
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        markersize=3.5,
        color=victim_colors["V2"],
        lw=0,
    ),
]
labels = [
    "victim 1",
    "victim 2",
]

fig.suptitle("Fully Informed Attacks")
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

fig.legend(
    handles=custom_handles,
    labels=labels,
    loc=(0.1, -0.0),
    ncol=2,
    frameon=False,
    alignment="left",
    title="Victims",
)

leg = Legend(
    fig,
    handles=th_bgs,
    labels=[
        "strict",
        "moderate",
        "lenient",
        "permissive",
    ],
    title="Thresholds",
    ncol=4,
    loc=(0.4, -0.0),
    # bbox_to_anchor=(0.6, -0.1),
    frameon=False,
    alignment="left",
)

fig.add_artist(leg)
# fig.tight_layout()
fig.savefig("evaluation/results/fully_informed_attack.png")
fig.savefig("evaluation/results/fully_informed_attack.pgf")

# %% Attempts with wrong hand (Section 6.2)

left_mp_embeddings = compute_sl_embeddings(
    motion_sequences=left_motion_passwords,
    sequence_metadata=left_password_metadata,
    preprocessing_info=prep_info,
    model_path_or_id=sl_model_path,
    device="cpu",
)

left_right_test = pd.concat(
    [
        mop_embeddings.query("session_id == 1"),
        left_mp_embeddings.query("session_id == 0"),
    ],
    axis=0,
    ignore_index=True,
)


left_right_dists, left_right_labls = compute_sl_distances(
    embedded_passwords=left_right_test,
    ref_word="Password",
    query_word="Secure",
    ref_session=0,
    query_session=1,
)

left_right_conditions = [
    (
        "right-right",
        happy_path_sl_mop_distances,
        happy_path_sl_mop_labels,
        dict(lw=1, linestyle="-", c=sl_mop_color),
    ),
    (
        "left-right",
        left_right_dists,
        left_right_labls,
        dict(lw=1, linestyle="-", c=sl_bat_color),
    ),
]

generate_roc(left_right_conditions, "left/right")
