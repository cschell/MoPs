# %%

import pandas as pd
from helpers_preparation import encode_passwords
from manual_corrections import attack_corrections
from password_rules import attack_rules

FPS = 30

passwords = pd.read_feather("intermediate/attack_motion_passwords.feather")

for data_encoding in ["BRA", "SR"]:
    processed_passwords = encode_passwords(
        passwords,
        target_fps=FPS,
        encoding=data_encoding,
        corrections=attack_corrections,
        rules=attack_rules,
    )
    processed_passwords.to_feather(f"out/attack_passwords_{data_encoding}-enc_{FPS}-fps.feather")
