import numpy as np
import pandas as pd
from tqdm import tqdm

column_mapping = {
    "RealtimeSinceStartup": "timestamp",
    "DevicePositionX": "head_pos_x",
    "DevicePositionY": "head_pos_y",
    "DevicePositionZ": "head_pos_z",
    "DeviceRotationX": "head_rot_x",
    "DeviceRotationY": "head_rot_y",
    "DeviceRotationZ": "head_rot_z",
    "DeviceRotationW": "head_rot_w",
    "RightControllerGripButton": "drawing_right",
    "RightControllerDevicePositionX": "right_hand_pos_x",
    "RightControllerDevicePositionY": "right_hand_pos_y",
    "RightControllerDevicePositionZ": "right_hand_pos_z",
    "RightControllerDeviceRotationX": "right_hand_rot_x",
    "RightControllerDeviceRotationY": "right_hand_rot_y",
    "RightControllerDeviceRotationZ": "right_hand_rot_z",
    "RightControllerDeviceRotationW": "right_hand_rot_w",
    "LeftControllerGripButton": "drawing_left",
    "LeftControllerDevicePositionX": "left_hand_pos_x",
    "LeftControllerDevicePositionY": "left_hand_pos_y",
    "LeftControllerDevicePositionZ": "left_hand_pos_z",
    "LeftControllerDeviceRotationX": "left_hand_rot_x",
    "LeftControllerDeviceRotationY": "left_hand_rot_y",
    "LeftControllerDeviceRotationZ": "left_hand_rot_z",
    "LeftControllerDeviceRotationW": "left_hand_rot_w",
}


def _convert_m_to_cm(df):
    df = df.copy()

    for c in df.columns:
        if "_pos_" in c:
            df[c] *= 100

    return df


def _convert_coord_system_from_RUB_to_RUF(df):
    df = df.copy()

    for c in df.columns:
        if c.endswith("_z") or c.endswith("_w"):
            df[c] *= -1

    return df


def _extract_hand_from_state(df):
    df = df.copy()
    df["hand"] = ""

    df.loc[df["sub_state"].str.contains("LeftHandTurnState", na=False), "hand"] = "left"
    df.loc[df["sub_state"].str.contains("RightHandTurnState", na=False), "hand"] = (
        "right"
    )

    return df


def _writing_frames_mask(df, drawing_column_name, rules, correction):
    df = df.copy().reset_index(drop=True)
    if not len(df):
        return
    total_length = df["delta_seconds"].max()
    void_frames_mask = df.eval(
        f"(delta_seconds <= {rules['begin_cut_off']}) or (delta_seconds >= {total_length - rules['end_cut_off']})"
    )

    df.loc[void_frames_mask, drawing_column_name] = False

    try:
        drawing_column_diff = (
            df[drawing_column_name].astype(int).diff().reset_index(drop=True).abs()
        )

        start = df.index >= drawing_column_diff.idxmax()
        stop = df.index < drawing_column_diff[::-1].idxmax()

        # in case the button has not been released before clip end
        if drawing_column_diff.sum() < 2:
            stop = start

        if start_ts := correction.get("start"):
            print("applying start correction")
            start = (df.delta_seconds >= start_ts).to_numpy()
        if stop_ts := correction.get("end"):
            print("applying end correction")
            stop = (df.delta_seconds < stop_ts).to_numpy()

        assert np.any(start & stop)
        assert not np.isnan(start & stop).any()
        return start & stop
    except ValueError as e:
        return np.zeros(len(df))


def process_user(user_name, motion_files, rules, corrections=None, include_mimic=False):
    if corrections is None:
        corrections = {}
    for session_id, motion_file_path in enumerate(motion_files):
        states_file_path = motion_file_path.with_name(
            motion_file_path.name.replace("_movement", "_states")
        )
        motion_data = (
            pd.read_csv(motion_file_path)
            .rename(columns=column_mapping)
            .assign(timestamp=lambda df: pd.to_timedelta(df["timestamp"], unit="s"))
        )[list(column_mapping.values())]
        states = (
            pd.read_csv(states_file_path)
            .pipe(_extract_hand_from_state)
            .assign(
                timestamp=lambda df: pd.to_timedelta(
                    df["realtime_since_startup"], unit="s"
                ),
                step_duration=lambda df: df["timestamp"].diff(-1).abs(),
                start=lambda df: df["timestamp"],
                end=lambda df: df["timestamp"] + df["step_duration"],
                next_action=lambda df: df["action"].shift(-1),
            )
        )

        include_mimic_query = (
            "| (state.str.contains('MimicPassword'))" if include_mimic else ""
        )

        words = (
            states.query(
                "(next_action == 'NextButtonPressed')"
                "& (current_password == current_password)"
                "& (current_password != 'Test')"
                "& ("
                "(state.str.contains('SpecificPassword'))"
                "| (state.str.contains('RandomPassword'))"
                f"{include_mimic_query}"
                ")"
            )
            .rename(
                columns={
                    "step_duration": "duration",
                    "current_password": "word",
                    "sub_state_iteration": "iteration",
                }
            )
            .reset_index(drop=True)
        )[["start", "end", "duration", "word", "iteration", "hand"]]

        for _idx, word in tqdm(list(words.iterrows()), leave=False):
            correction = corrections.get(user_name, {}).get(
                (word.word.lower(), word.iteration, session_id, word.hand), {}
            )

            drawing_column_name = f"drawing_{word.hand}"

            word_motion_sequence = (
                motion_data.query(
                    "(timestamp >= @word.start) & (timestamp <= @word.end)",
                    engine="python",
                )
                .copy()
                .pipe(_convert_coord_system_from_RUB_to_RUF)
                .pipe(_convert_m_to_cm)
                .assign(
                    is_drawing=lambda df: df[drawing_column_name],
                    word=word.word,
                    hand=word.hand,
                    iteration=word.iteration,
                    user_name=user_name,
                    session_id=session_id,
                    delta_seconds=lambda df: (
                        (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
                        if len(df)
                        else None
                    ),
                    selected_frames=lambda df: _writing_frames_mask(
                        df, drawing_column_name, rules, correction
                    ),
                )
                .reset_index(drop=True)
                .drop(["drawing_left", "drawing_right"], axis="columns")
            )

            if len(word_motion_sequence) < 50:
                print(
                    f"motion sequence has only {len(word_motion_sequence)} frames! "
                    f"\nFile: {motion_file_path}"
                    f"\nWord: {word.word}\n"
                )
                continue
            yield word_motion_sequence
