# %%

import pathlib
from matplotlib import pyplot as plt
import pandas as pd
from manual_corrections import main_corrections
from password_rules import main_rules
from helpers_verification import visualize_passwords


passwords = pd.read_feather("intermediate/main_motion_passwords.feather")

for fig, user_name in visualize_passwords(passwords.query("hand == 'right'"), main_rules, main_corrections):
    filepath = pathlib.Path(f"rendered_main_passwords/user_{user_name}.jpg")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(filepath)
    plt.close()
