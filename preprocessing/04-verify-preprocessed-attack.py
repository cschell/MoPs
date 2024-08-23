# %%

import pathlib
from matplotlib import pyplot as plt
import pandas as pd
from manual_corrections import attack_corrections
from helpers_verification import visualize_passwords


passwords = pd.read_feather("out/attack_passwords_SR-enc_30-fps.feather")

for fig, user_name in visualize_passwords(passwords, rules=None, corrections=attack_corrections):
    filepath = pathlib.Path(f"rendered_attack_SR_passwords/user_{user_name}.jpg")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(filepath)
    plt.close()
