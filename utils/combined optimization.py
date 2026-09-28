import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    # model, steps, fid,       time,      resolution
    ["A", 30, 152.03,   1.09376,    512],
    ["A", 30, 254.33,   0.84998,    256],
    ["B", 30, 165.3772, 1.4811,     256],
    ["B", 30, 199.3467, 0.7345,     128],
]

df = pd.DataFrame(data, columns=["model", "steps", "fid", "time", "resolution"])
df["mu"] = 1.0 / df["time"]

Q = 5.0         
V = 1.0          
lambda_vals = np.linspace(0.1, 1.5, 200)

def select_config(df_ctrl, arrival_rate):
    feasible = df_ctrl[df_ctrl["mu"] > arrival_rate]
    if feasible.empty:
        return None
    cost = Q / (feasible["mu"] - arrival_rate) + V * feasible["fid"]
    idx = cost.idxmin()
    return feasible.loc[idx]

selected_res, selected_model = [], []

for lam in lambda_vals:
    cfg = select_config(df, lam)
    if cfg is None:
        selected_res.append(np.nan)
        selected_model.append(None)
    else:
        selected_res.append(cfg["resolution"])
        selected_model.append(cfg["model"])

selected_res = np.array(selected_res)
selected_model = np.array(selected_model)
plt.figure(figsize=(6, 4))

mask_A = selected_model == "A"
plt.plot(lambda_vals[mask_A], selected_res[mask_A],
         color="tab:blue", linewidth=2, label="Method A")

mask_B = selected_model == "B"
plt.plot(lambda_vals[mask_B], selected_res[mask_B],
         color="tab:orange", linewidth=2, label="Method B")

plt.xlabel("Arrival rate λ (requests/s)")
plt.ylabel("Selected resolution (pixels)")
plt.title("Online adaptation: model switch under increasing load\n(steps = 30)")
plt.grid(True)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("model_switch_colored.png", dpi=300)
plt.show()