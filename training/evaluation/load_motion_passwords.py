import pandas as pd
from tqdm import tqdm


def load_motion_passwords(
    dataset_path, hand="right", query_addition="True", encode_words=True
):
    def encode_individual_words(df):
        df = df.copy()

        if encode_words:
            for user_name, pwd in df.groupby("user_name"):
                common_words = [
                    "Motion",
                    "Password",
                    "Secure",
                    "Motion Password",
                ]
                v1_common_words = [f"{w} [V1]" for w in common_words]
                v2_common_words = [f"{w} [V2]" for w in common_words]
                individual_words = [
                    w
                    for w in pwd["word"].unique()
                    if w not in v1_common_words
                    and w not in v2_common_words
                    and w not in common_words
                    and not w.lower().startswith("ball")
                ]
                for w_idx, word in enumerate(individual_words):
                    df.loc[
                        df.eval("user_name == @user_name and word == @word"),
                        "word",
                    ] = f"word_{w_idx}"
                    assert w_idx < 4
        return df

    passwords = (
        pd.read_feather(dataset_path)
        .query(f"(hand == @hand) and ({query_addition})")
        .dropna()
        .pipe(encode_individual_words)
    )

    SEQUENCE_LENGTH = passwords["frame_idx"].max()

    n_passwords = len(passwords["take_id"].unique())

    motion_passwords = []
    user_names = []

    session = []
    words = []
    password_lengths = []
    iterations = []

    feature_columns = [
        c for c in passwords.columns if "_rot_" in c or "_pos_" in c
    ]

    for _, passwd in tqdm(
        passwords.groupby("take_id"),
        total=n_passwords,
        desc=f"loading Motion Passwords ({hand} hand)",
    ):
        X = passwd[:SEQUENCE_LENGTH][feature_columns]

        motion_passwords.append(X)
        password_lengths.append(len(X))

        user_names.append(passwd.iloc[0]["user_name"])
        words.append(passwd.iloc[0]["word"])
        session.append(passwd.iloc[0]["session_id"])
        iterations.append(passwd.iloc[0]["iteration"])

    password_metadata = pd.DataFrame(
        zip(user_names, words, session, iterations, password_lengths),
        columns=[
            "user_name",
            "word",
            "session_id",
            "iteration",
            "sequence_length",
        ],
    )

    return motion_passwords, password_metadata
