from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def _rotate_hands_forward(df):
    df = df.copy()
    mean_head_rotation = Rotation.from_quat(
        df[
            [
                "head_rot_x",
                "head_rot_y",
                "head_rot_z",
                "head_rot_w",
            ]
        ]
    ).mean()

    for hand in ["left_hand", "right_hand"]:
        cols = [f"{hand}_pos_x", f"{hand}_pos_y", f"{hand}_pos_z"]
        df[cols] = mean_head_rotation.inv().apply(df[cols])

    return df


def visualize_passwords(passwords, rules: dict, corrections: dict):
    timestamps = []

    for user_name, user_sequences in tqdm(
        passwords.groupby(by="user_name"), total=len(passwords["user_name"].unique())
    ):
        word_mapping = (
            user_sequences[["word", "session_id", "hand"]]
            .drop_duplicates()
            .sort_values(["session_id", "word", "hand"])
            .reset_index(drop=True)
        )

        num_iterations = len(user_sequences["iteration"].unique())

        nrows = len(word_mapping)
        ncols = num_iterations

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4, nrows * 3),
        )

        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05
        )

        for take_id, sequence in user_sequences.groupby("take_id"):
            sequence = sequence.pipe(_rotate_hands_forward)

            _user_name, word, hand, iteration, session = sequence.iloc[0][
                ["user_name", "word", "hand", "iteration", "session_id"]
            ]

            correction = corrections.get(user_name, {}).get(
                (word.lower(), iteration, session, hand), {}
            )

            assert _user_name == user_name

            col_idx = int(iteration) - 1
            row_idx = (
                word_mapping.query(
                    f"word == @word and session_id == @session and hand == @hand"
                )
                .iloc[0]
                .name
            )

            ax = axes[row_idx, col_idx]

            total_length = sequence["delta_seconds"].max()

            if rules:
                void_drawing_frames = sequence.eval(
                    f"(delta_seconds <= {rules['begin_cut_off']}) or (delta_seconds >= {total_length - rules['end_cut_off']})"
                )
            else:
                void_drawing_frames = np.zeros(len(sequence)).astype(bool)

            if "is_drawing" not in sequence:
                is_drawing_column_exists = False
                sequence["is_drawing"] = False
            else:
                is_drawing_column_exists = True

            if "selected_frames" not in sequence:
                sequence["selected_frames"] = True

            ax.scatter(
                x=f"{hand}_hand_pos_x",
                y=f"{hand}_hand_pos_y",
                data=sequence[~void_drawing_frames].query("not is_drawing"),
                alpha=0.1,
                c="gray",
                s=5,
            )

            ax.scatter(
                x=f"{hand}_hand_pos_x",
                y=f"{hand}_hand_pos_y",
                data=sequence[void_drawing_frames].query("not is_drawing"),
                alpha=0.2,
                c="blue",
                s=5,
            )

            ax.plot(
                f"{hand}_hand_pos_x",
                f"{hand}_hand_pos_y",
                data=sequence[~void_drawing_frames].query("selected_frames"),
                alpha=0.4,
                c="green",
                # s=5,
            )

            if is_drawing_column_exists:
                ax.scatter(
                    x=f"{hand}_hand_pos_x",
                    y=f"{hand}_hand_pos_y",
                    data=sequence[void_drawing_frames].query("is_drawing"),
                    alpha=0.4,
                    c="violet",
                    s=5,
                )
                ax.scatter(
                    x=f"{hand}_hand_pos_x",
                    y=f"{hand}_hand_pos_y",
                    data=sequence[~void_drawing_frames].query("is_drawing"),
                    alpha=0.4,
                    c="orange",
                    s=8,
                )

            ax.set_yticks([])
            ax.set_xticks([])
            overall_duration = sequence.delta_seconds.values[-1]

            writing_sequence = sequence.query("selected_frames")
            if len(writing_sequence) == 0:
                writing_sequence = sequence
            (_, start), (_, stop) = writing_sequence.iloc[[0, -1]].iterrows()
            word_duration = stop.delta_seconds - start.delta_seconds

            if word_duration:
                actual_fps = len(writing_sequence) / word_duration
            else:
                actual_fps = 0.0

            if rules:
                valid = (
                    (actual_fps >= rules["min_fps"])
                    and word_duration >= rules["min_word_duration"]
                    and word_duration <= rules["max_word_duration"]
                )
            else:
                valid = True

            override_valid = correction.get("valid")
            for spine in ax.spines.values():
                if valid and (override_valid or override_valid == None):
                    spine.set_edgecolor("darkgreen")
                elif not valid and override_valid:
                    spine.set_edgecolor("lightgreen")
                elif valid and override_valid == False:
                    spine.set_edgecolor("orange")
                elif not valid and override_valid == False:
                    spine.set_edgecolor("darkred")
                else:
                    spine.set_edgecolor("red")
                spine.set_linewidth(5)

            drawing_start = start.delta_seconds
            drawing_end = stop.delta_seconds
            ax.text(
                0.01,
                0.01,
                f"""{start.delta_seconds:.1f}s-{stop.delta_seconds:.2f}s ({word_duration:0.1f}s)
    {actual_fps=:0.0f}
    """,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=ax.transAxes,
                bbox=dict(facecolor="#ffffff", alpha=0.5),
            )

            if col_idx == 0:
                ax.set_ylabel(f"{session=} | {hand=}\n{word=}")

            ax.set_xlim(
                sequence[f"{hand}_hand_pos_x"].min() - 5,
                sequence[f"{hand}_hand_pos_x"].max() + 5,
            )
            # ax.set_aspect("equal")

            timestamps.append(
                {
                    "user": user_name,
                    "word": word,
                    "hand": hand,
                    "iteration": iteration,
                    "session": session,
                    "valid": valid,
                    "override": override_valid,
                    "word_duration": word_duration,
                    "overall_duration": overall_duration,
                    "drawing_start": round(drawing_start, 2),
                    "drawing_end": round(drawing_end, 2),
                    "drawing_start_override": None,
                    "drawing_end_override": None,
                }
            )

        fig.tight_layout()
        yield fig, user_name  # , timestamps
