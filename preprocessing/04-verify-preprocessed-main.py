# %%

import pathlib
from matplotlib import pyplot as plt
import pandas as pd
from manual_corrections import main_corrections
from helpers_verification import visualize_passwords


passwords = pd.read_feather("out/main_passwords_SR-enc_30-fps.feather")

for fig, user_name in visualize_passwords(passwords.query("hand == 'right'"), rules=None, corrections=main_corrections):
    filepath = pathlib.Path(f"rendered_main_SR_passwords/user_{user_name}.jpg")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(filepath)
    plt.close()
