import csv
import os

import numpy as np

from isee import aggregate, data, normalise, weights, figures, robustness

RES, FIG = "results", "figures"
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def save_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  results/{name}")


print("== Part 1: four case-study systems")
full = data.load()
cases = full.subset(full.flagship)
short = ["Chile\n(copper)", "Australia\n(lithium)", "Indonesia\n(nickel)",
         "DR Congo\n(cobalt)"]
names = ["Chile", "Australia", "Indonesia", "DR Congo"]

Xn = normalise.minmax(cases.X)
S = normalise.dimension_scores(Xn)
B = aggregate.benefit_oriented(S)

w_eq = weights.equal()
isee_eq = aggregate.additive(S, w_eq)
w_ent, w_pca, w_cri = weights.entropy(B), weights.pca(B), weights.critic(B)

specs = [
    ("Equal weights, additive (baseline)", aggregate.additive(S, w_eq)),
    ("Entropy weights, additive", aggregate.additive(S, w_ent)),
    ("PCA weights, additive", aggregate.additive(S, w_pca)),
    ("CRITIC weights, additive", aggregate.additive(S, w_cri)),
    ("Equal weights, geometric", aggregate.geometric(S, w_eq)),
]

fv_rank = aggregate.ranks(S[:, 0])
save_csv("table2_dimension_scores.csv",
         ["system"] + data.DIMENSIONS + ["ISEE", "ISEE_rank", "FV_only_rank"],
         [[names[j]] + [f"{S[j, i]:.3f}" for i in range(5)]
          + [f"{isee_eq[j]:+.3f}", aggregate.ranks(isee_eq)[j], fv_rank[j]]
          for j in range(4)])

spec_weights = [w_eq, w_ent, w_pca, w_cri, w_eq]
save_csv("table3_specifications.csv",
         ["specification"] + names + ["weights (FV, MF, SV, EX, LL)"],
         [[label] + [f"{sc[j]:+.3f} ({aggregate.ranks(sc)[j]})"
                     for j in range(4)]
          + [", ".join(f"{v:.3f}" for v in spec_weights[i])]
          for i, (label, sc) in enumerate(specs)])
print("  weights: entropy", np.round(w_ent, 3), " pca", np.round(w_pca, 3),
      " critic", np.round(w_cri, 3))

mc_scores, mc_freq = robustness.weight_mc(S, n=10_000)
save_csv("table4_mc_rank_frequencies.csv",
         ["rank"] + names,
         [[r + 1] + [f"{mc_freq[r, j]:.3f}" for j in range(4)]
          for r in range(4)])

dmc_scores, dmc_freq = robustness.data_mc(cases.X, w_eq, n=5_000, noise=0.10)
save_csv("table5_data_noise_rank_frequencies.csv",
         ["rank"] + names,
         [[r + 1] + [f"{dmc_freq[r, j]:.3f}" for j in range(4)]
          for r in range(4)])

figures.radar(B, names, os.path.join(FIG, "fig1_profiles.png"))
figures.contributions(S, w_eq, short, os.path.join(FIG, "fig2_contributions.png"))
figures.mc_violin(mc_scores, mc_freq[0], short,
                  os.path.join(FIG, "fig3_montecarlo.png"))

for label, sc in specs:
    rk = aggregate.ranks(sc)
    print(f"  {label:36s}: " + "  ".join(
        f"{names[j]} {sc[j]:+.3f} (#{rk[j]})" for j in range(4)))
print("  MC rank-1 frequencies:",
      {names[j]: f"{mc_freq[0, j]:.1%}" for j in range(4)})
print("  data-noise rank-1 frequencies:",
      {names[j]: f"{dmc_freq[0, j]:.1%}" for j in range(4)})
