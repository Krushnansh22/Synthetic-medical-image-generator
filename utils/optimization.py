import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    # steps,  FID,      time,      resolution
    [10,     161.3278, 0.537,    512],
    [20,     159.8,    0.80655,  512],
    [30,     152.03,   1.09376,  512],
    [40,     159.7,    1.39018,  512],
    [50,     150.727,  1.7004,   512],

    [10,     406.71,   0.356779, 256],
    [20,     282.52,   0.5979617,256],
    [30,     254.33,   0.84998,  256],
    [40,     234.16,   1.1902,   256],
    [50,     239.56,   1.44889,  256],
]

df = pd.DataFrame(
    data,
    columns=["steps", "fid", "time", "resolution"]
)

df["mu"] = 1.0 / df["time"]

Q = 5.0          # queue backlog
V = 1.0          # quality weight
lambda_vals = np.linspace(0.1, 1.5, 200)

def select_config(df_ctrl, arrival_rate):
    feasible = df_ctrl[df_ctrl["mu"] > arrival_rate]
    if feasible.empty:
        return None

    cost = Q / (feasible["mu"] - arrival_rate) + V * feasible["fid"]
    idx = cost.idxmin()
    return feasible.loc[idx]

df_steps = df[df["resolution"] == 512]

step_decisions = []

for lam in lambda_vals:
    cfg = select_config(df_steps, lam)
    if cfg is None:
        step_decisions.append((lam, np.nan))
    else:
        step_decisions.append((lam, cfg["steps"]))

step_df = pd.DataFrame(
    step_decisions,
    columns=["lambda", "steps"]
)

df_res = df[df["steps"] == 30]

res_decisions = []

for lam in lambda_vals:
    cfg = select_config(df_res, lam)
    if cfg is None:
        res_decisions.append((lam, np.nan))
    else:
        res_decisions.append((lam, cfg["resolution"]))

res_df = pd.DataFrame(
    res_decisions,
    columns=["lambda", "resolution"]
)

fig, axes = plt.subplots(
    2, 1, figsize=(6, 6), sharex=True, constrained_layout=True
)

axes[0].plot(step_df["lambda"], step_df["steps"])
axes[0].set_ylabel("Selected diffusion steps")
axes[0].set_title("Online adaptation: diffusion step control")
axes[0].grid(True)

axes[1].plot(res_df["lambda"], res_df["resolution"])
axes[1].set_ylabel("Selected resolution")
axes[1].set_xlabel("Arrival rate λ (requests/s)")
axes[1].set_title("Online adaptation: resolution control")
axes[1].grid(True)
plt.savefig('optimization.png')

plt.show()