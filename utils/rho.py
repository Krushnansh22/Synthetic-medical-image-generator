import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

true_lambda = 0.003  # 10.8 requests/hour
possible_images = np.array([50, 100, 250, 500, 1000])

scenario_probs = {
    "Scenario 1 (Heavy)": np.array([0.05, 0.2, 0.35, 0.2, 0.2]),
    "Scenario 2 (Normal)": np.array([0.25, 0.25, 0.35, 0.1, 0.05]),
}

modelA = pd.DataFrame([
    [512, 10, 0.537, 161.33],
    [512, 20, 0.80655, 159.8],
    [512, 30, 1.09376, 152.03],
    [512, 40, 1.39018, 159.7],
    [512, 50, 1.7004, 150.73],
    [256, 10, 0.356779, 406.71],
    [256, 20, 0.597962, 282.52],
    [256, 30, 0.84998, 254.33],
    [256, 40, 1.1902, 234.16],
    [256, 50, 1.44889, 239.56],
], columns=["Resolution", "Steps", "PerImageTime", "FID"])
modelA["Model"] = "Model A"

modelB = pd.DataFrame([
    [128, 10, 0.24116, 203.95],
    [128, 20, 0.47550, 200.22],
    [128, 30, 0.73450, 199.35],
    [128, 40, 0.95380, 193.30],
    [128, 50, 1.18830, 188.17],
    [256, 10, 0.95850, 170.61],
    [256, 20, 1.24750, 166.50],
    [256, 30, 1.48110, 165.38],
    [256, 40, 1.67410, 161.12],
    [256, 50, 1.81230, 158.35],
], columns=["Resolution", "Steps", "PerImageTime", "FID"])
modelB["Model"] = "Model B"

all_models = pd.concat([modelA, modelB], ignore_index=True)

results = []

for scenario_name, probs in scenario_probs.items():
    E_N = np.sum(possible_images * probs) 
    
    for _, row in all_models.iterrows():
        res = row["Resolution"]
        steps = row["Steps"]
        t_img = row["PerImageTime"]
        fid = row["FID"]
        model = row["Model"]

        E_S = E_N * t_img              # avg service time per request
        mu = 1 / E_S                   # service rate
        rho = true_lambda / mu         # utilization
        
        results.append({
            "Model": model,
            "Scenario": scenario_name,
            "Resolution": res,
            "Steps": steps,
            "PerImageTime": t_img,
            "FID": fid,
            "E_N": E_N,
            "E_S": E_S,
            "Mu": mu,
            "Rho": rho
        })

df = pd.DataFrame(results)

df.to_csv("rho_analysis.csv", index=False)
print(df.head(10))

for scenario_name in scenario_probs.keys():
    plt.figure(figsize=(10, 5))
    subset = df[df["Scenario"] == scenario_name]
    
    for (model, res), group in subset.groupby(["Model", "Resolution"]):
        plt.plot(group["Steps"], group["Rho"], marker="o", label=f"{model} ({res}x{res})")
    
    plt.axhline(1.0, color="red", linestyle="--", label="ρ = 1 (Overload threshold)")
    plt.xlabel("Inference Steps")
    plt.ylabel("Utilization ρ (λ/μ)")
    plt.title(f"System Utilization vs Inference Steps — {scenario_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"rho_vs_steps_{scenario_name.replace(' ', '_')}.png", dpi=200)
    plt.show()