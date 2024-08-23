# %%

import pathlib
from matplotlib import pyplot as plt
import pandas as pd
from helpers_verification import visualize_passwords
from manual_corrections import attack_corrections
from password_rules import attack_rules

passwords = pd.read_feather("intermediate/attack_motion_passwords.feather")

for fig, user_name in visualize_passwords(passwords, attack_rules, attack_corrections):
    filepath = pathlib.Path(f"rendered_attack_passwords/user_{user_name}.jpg")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(filepath)
    plt.close()
