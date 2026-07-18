import csv
import os

import numpy as np

from isee import aggregate, data, figures, normalise, robustness, typology, weights

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

print("== Part 2: typology of 30 country-mineral systems")
Xn_all = normalise.minmax(full.X)
S_all = normalise.dimension_scores(Xn_all)
B_all = aggregate.benefit_oriented(S_all)
isee_all = aggregate.additive(S_all, w_eq)

sil = typology.select_k(B_all)
k_best = max(sil, key=sil.get)
print("  silhouette by k:", {k: round(v, 3) for k, v in sil.items()},
      "-> k =", k_best)

labels_km, centers = typology.kmeans(B_all, k_best)
labels_w, Z, coph = typology.ward(B_all, k_best)
ari = typology.agreement(labels_km, labels_w)
print(f"  cophenetic corr = {coph:.3f}; ARI(k-means, Ward) = {ari:.3f}")

coords, var, _ = typology.pca_project(B_all)

save_csv("table6_full_ranking.csv",
         ["system", "cluster"] + data.DIMENSIONS + ["ISEE", "rank"],
         [[full.ids[j], labels_km[j] + 1]
          + [f"{S_all[j, i]:.3f}" for i in range(5)]
          + [f"{isee_all[j]:+.3f}", aggregate.ranks(isee_all)[j]]
          for j in np.argsort(-isee_all)])

save_csv("table7_cluster_profiles.csv",
         ["cluster", "n", "members"] + [f"mean_{d}" for d in data.DIMENSIONS]
         + ["mean_ISEE"],
         [[c + 1, int((labels_km == c).sum()),
           "; ".join(np.array(full.ids)[labels_km == c])]
          + [f"{S_all[labels_km == c, i].mean():.3f}" for i in range(5)]
          + [f"{isee_all[labels_km == c].mean():+.3f}"]
          for c in range(k_best)])

figures.typology_scatter(coords, var, labels_km, full,
                         os.path.join(FIG, "fig4_typology.png"))
figures.typology_dendrogram(Z, full, labels_km,
                            os.path.join(FIG, "fig5_dendrogram.png"))

for c in range(k_best):
    ids = np.array(full.ids)[labels_km == c]
    print(f"  cluster {c + 1} (n={len(ids)}, mean ISEE "
          f"{isee_all[labels_km == c].mean():+.3f}): {', '.join(ids)}")

top = np.argsort(-isee_all)
print("  top-5:", [(full.ids[j], f"{isee_all[j]:+.3f}") for j in top[:5]])
print("  bottom-5:", [(full.ids[j], f"{isee_all[j]:+.3f}") for j in top[-5:]])
print("done.")
