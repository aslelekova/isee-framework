import csv
import os

import numpy as np

from isee import (aggregate, counterfactual, data, figures, normalise,
                  robustness, smaa, typology, weights)

RES, FIG = "results", "figures"
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def save_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  results/{name}")


full = data.load()
cases = full.subset(full.flagship)
short = ["Chile\n(copper)", "Australia\n(lithium)", "Indonesia\n(nickel)",
         "DR Congo\n(cobalt)"]
names = ["Chile", "Australia", "Indonesia", "DR Congo"]

Xn_all = normalise.minmax(full.X)
S_all = normalise.dimension_scores(Xn_all)
B_all = aggregate.benefit_oriented(S_all)

print("== Reference-sample weights (30 systems)")
w_eq = weights.equal()
w_ent = weights.entropy(B_all)
w_cri = weights.critic(B_all)
print("  entropy:", np.round(w_ent, 3), " critic:", np.round(w_cri, 3))

corr = robustness.spearman_matrix(full.X)
save_csv("indicator_spearman_correlations.csv",
         ["indicator"] + [n.split(":")[0] for n in data.INDICATOR_NAMES],
         [[data.INDICATOR_NAMES[i].split(":")[0]]
          + [f"{corr[i, j]:.3f}" for j in range(10)] for i in range(10)])

print("== Part 1: four case-study systems")
Xn = normalise.minmax(cases.X)
S = normalise.dimension_scores(Xn)
B = aggregate.benefit_oriented(S)

isee_eq = aggregate.additive(S, w_eq)

specs = [
    ("Equal weights, additive (baseline)", aggregate.additive(S, w_eq), w_eq),
    ("Entropy weights (reference sample), additive",
     aggregate.additive(S, w_ent), w_ent),
    ("CRITIC weights (reference sample), additive",
     aggregate.additive(S, w_cri), w_cri),
    ("Equal weights, geometric", aggregate.geometric(S, w_eq), w_eq),
]

fv_rank = aggregate.ranks(S[:, 0])
save_csv("table2_dimension_scores.csv",
         ["system"] + data.DIMENSIONS + ["ISEE", "ISEE_rank", "FV_only_rank"],
         [[names[j]] + [f"{S[j, i]:.3f}" for i in range(5)]
          + [f"{isee_eq[j]:+.3f}", aggregate.ranks(isee_eq)[j], fv_rank[j]]
          for j in range(4)])

save_csv("table3_specifications.csv",
         ["specification"] + names + ["weights (FV, MF, SV, EX, LL)"],
         [[label] + [f"{sc[j]:+.3f} ({aggregate.ranks(sc)[j]})"
                     for j in range(4)]
          + [", ".join(f"{v:.3f}" for v in w)]
          for label, sc, w in specs])

mc_scores, mc_freq = robustness.weight_mc(S, n=10_000)
save_csv("table4_mc_rank_frequencies.csv",
         ["rank"] + names,
         [[r + 1] + [f"{mc_freq[r, j]:.3f}" for j in range(4)]
          for r in range(4)])

dmc_scores, dmc_freq = robustness.data_mc(cases, w_eq, n=5_000, noise=0.10)
save_csv("table5_data_noise_rank_frequencies.csv",
         ["rank"] + names,
         [[r + 1] + [f"{dmc_freq[r, j]:.3f}" for j in range(4)]
          for r in range(4)])

loio = robustness.leave_one_indicator_out(cases.X, w_eq)
save_csv("loio_four_cases.csv",
         ["dropped_indicator"] + names + ["max_rank_shift"],
         [[data.INDICATOR_NAMES[e["dropped"]]]
          + [f"{e['scores'][j]:+.3f} ({e['ranks'][j]})" for j in range(4)]
          + [e["max_rank_shift"]] for e in loio])
print("  LOIO max rank shift across indicators:",
      max(e["max_rank_shift"] for e in loio))

figures.radar(B, names, os.path.join(FIG, "fig1_profiles.png"))
figures.contributions(S, w_eq, short, os.path.join(FIG, "fig2_contributions.png"))
figures.mc_violin(mc_scores, mc_freq[0], short,
                  os.path.join(FIG, "fig3_montecarlo.png"))

for label, sc, _ in specs:
    rk = aggregate.ranks(sc)
    print(f"  {label:46s}: " + "  ".join(
        f"{names[j]} {sc[j]:+.3f} (#{rk[j]})" for j in range(4)))
print("  MC rank-1 frequencies:",
      {names[j]: f"{mc_freq[0, j]:.1%}" for j in range(4)})
print("  data-noise rank-1 frequencies:",
      {names[j]: f"{dmc_freq[0, j]:.1%}" for j in range(4)})

print("== Part 2: compensability and SMAA rank acceptability")
rho_grid, pm_scores, pm_ranks = smaa.compensability_sweep(S, w_eq)
save_csv("compensability_sweep.csv",
         ["rho"] + [f"score_{n}" for n in names] + [f"rank_{n}" for n in names],
         [[f"{rho_grid[i]:.2f}"] + [f"{pm_scores[i, j]:.4f}" for j in range(4)]
          + [pm_ranks[i, j] for j in range(4)] for i in range(len(rho_grid))])

floors = [0.01, 0.05, 0.10, 0.20]
crossovers = {f: smaa.crossover(S, w_eq, i=1, j=2, floor=f) for f in floors}
save_csv("compensability_floor_sensitivity.csv",
         ["floor", "crossover_rho_AUS_IDN"],
         [[f, f"{r:.3f}" if r is not None else "none"]
          for f, r in crossovers.items()])
print("  AUS-IDN crossover rho by floor:",
      {f: round(r, 3) if r else None for f, r in crossovers.items()})

res = smaa.sample(cases, n=20_000, noise=0.10)
b = smaa.rank_acceptability(res["ranks"])
P = smaa.pairwise_probability(res["scores"])
prof = smaa.central_profile(res, rank=1)
erank = smaa.expected_rank(b)
entro = smaa.rank_entropy(b)

save_csv("smaa_acceptability.csv",
         ["rank"] + names,
         [[r + 1] + [f"{b[r, j]:.3f}" for j in range(4)] for r in range(4)])
save_csv("smaa_pairwise.csv",
         ["P(row > col)"] + names,
         [[names[i]] + [f"{P[i, j]:.3f}" if i != j else "—" for j in range(4)]
          for i in range(4)])
save_csv("smaa_central_profiles.csv",
         ["system", "n_rank1"] + [f"w_{d}_mean" for d in data.DIMENSIONS]
         + [f"w_{d}_10_90" for d in data.DIMENSIONS]
         + ["rho_mean", "rho_median", "rho_10_90", "expected_rank",
            "rank_entropy"],
         [[names[j], prof[j]["n"]]
          + [f"{prof[j]['w_mean'][i]:.3f}" for i in range(5)]
          + [f"[{prof[j]['w_lo'][i]:.2f}, {prof[j]['w_hi'][i]:.2f}]"
             for i in range(5)]
          + [f"{prof[j]['rho_mean']:.2f}", f"{prof[j]['rho_median']:.2f}",
             f"[{prof[j]['rho_lo']:.2f}, {prof[j]['rho_hi']:.2f}]",
             f"{erank[j]:.2f}", f"{entro[j]:.3f}"]
          for j in range(4) if prof[j] is not None])
print("  acceptability rank1:", {names[j]: f"{b[0, j]:.1%}" for j in range(4)})
print("  expected ranks:", {names[j]: round(erank[j], 2) for j in range(4)})
print("  P(Chile>Australia) =", f"{P[0, 1]:.3f}",
      " P(Australia>Indonesia) =", f"{P[1, 2]:.3f}",
      " P(Indonesia>DRC) =", f"{P[2, 3]:.3f}",
      " P(Chile>DRC) =", f"{P[0, 3]:.3f}")
for j in range(4):
    if prof[j]:
        print(f"    {names[j]}: w_mean={np.round(prof[j]['w_mean'], 3)} "
              f"rho_mean={prof[j]['rho_mean']:.2f} "
              f"rho_10_90=[{prof[j]['rho_lo']:.2f}, {prof[j]['rho_hi']:.2f}]")

print("  prior sensitivity:")
prior_rows = []
for label, kw in [
    ("alpha=0.5, noise 10%", dict(alpha=0.5, noise=0.10)),
    ("alpha=1, noise 10% (baseline)", dict(alpha=1.0, noise=0.10)),
    ("alpha=2, noise 10%", dict(alpha=2.0, noise=0.10)),
    ("alpha=5, noise 10%", dict(alpha=5.0, noise=0.10)),
    ("alpha=1, noise 5%", dict(alpha=1.0, noise=0.05)),
    ("alpha=1, noise 20%", dict(alpha=1.0, noise=0.20)),
    ("alpha=1, rho U(-1,1)", dict(alpha=1.0, noise=0.10,
                                  rho_range=(-1.0, 1.0))),
    ("alpha=1, rho U(0,2)", dict(alpha=1.0, noise=0.10,
                                 rho_range=(0.0, 2.0))),
]:
    r = smaa.sample(cases, n=10_000, **kw)
    bb = smaa.rank_acceptability(r["ranks"])
    prior_rows.append([label, f"{bb[0, 0]:.3f}",
                       f"{bb[0, 0] + bb[1, 0]:.3f}", f"{bb[3, 3]:.3f}"])
    print(f"    {label}: Chile rank1 {bb[0, 0]:.1%}, Chile top2 "
          f"{bb[0, 0] + bb[1, 0]:.1%}, DRC rank4 {bb[3, 3]:.1%}")
save_csv("smaa_prior_sensitivity.csv",
         ["specification", "Chile_rank1", "Chile_top2", "DRC_rank4"],
         prior_rows)

figures.compensability(rho_grid, pm_scores, names,
                       os.path.join(FIG, "fig4_compensability.png"))
figures.acceptability(b, [s.replace(chr(10), " ") for s in short],
                      os.path.join(FIG, "fig5_acceptability.png"))

print("  counterfactual scenarios (IDN-Ni, P(rank<=2)>=0.60):")
cf_rows = []
for scen, costs in [("equal cost", {"grid_co2": 1.0, "rule_of_law_gap": 1.0}),
                    ("carbon-expensive", {"grid_co2": 2.0,
                                          "rule_of_law_gap": 1.0}),
                    ("governance-expensive", {"grid_co2": 1.0,
                                              "rule_of_law_gap": 2.0})]:
    cf = counterfactual.scenario_search(
        cases, system_idx=2, levers=["grid_co2", "rule_of_law_gap"],
        target_rank=2, probability=0.60, costs=costs)
    ch = cf["changes"]
    cf_rows.append([scen,
                    f"-{ch['grid_co2']:.0%}" if ch else "n/a",
                    f"-{ch['rule_of_law_gap']:.0%}" if ch else "n/a",
                    f"{cf['baseline_probability']:.3f}",
                    f"{cf['probability']:.3f}" if cf["probability"] else "n/a"])
    print(f"    {scen}: {cf_rows[-1]}")
save_csv("counterfactual_scenarios_idn.csv",
         ["scenario", "grid_co2_change", "rule_of_law_gap_change",
          "baseline_P", "achieved_P"],
         cf_rows)

print("== Part 3: typology of 30 country-mineral systems")
isee_all = aggregate.additive(S_all, w_eq)

metrics = typology.select_k(B_all, k_range=range(2, 11))
sil_max = max(m["silhouette"] for m in metrics.values())
candidates = [k for k, m in metrics.items()
              if m["silhouette"] >= sil_max - 0.02]
stability = {k: typology.bootstrap_stability(B_all, k) for k in candidates}
k_best = max(candidates, key=lambda k: (round(stability[k][0], 3), -k))

save_csv("cluster_selection_metrics.csv",
         ["k", "silhouette", "calinski_harabasz", "bootstrap_ARI_mean",
          "bootstrap_ARI_sd"],
         [[k, f"{m['silhouette']:.3f}", f"{m['calinski_harabasz']:.1f}",
           f"{stability[k][0]:.3f}" if k in stability else "",
           f"{stability[k][1]:.3f}" if k in stability else ""]
          for k, m in metrics.items()])
print("  silhouette by k:",
      {k: round(m["silhouette"], 3) for k, m in metrics.items()})
print("  candidates (within 0.02 of max):", candidates)
print("  bootstrap ARI:", {k: (round(v[0], 3), round(v[1], 3))
                           for k, v in stability.items()})
print("  selected k =", k_best)

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
                         os.path.join(FIG, "fig6_typology.png"))
figures.typology_dendrogram(Z, full, labels_km,
                            os.path.join(FIG, "fig7_dendrogram.png"))

for c in range(k_best):
    ids = np.array(full.ids)[labels_km == c]
    print(f"  cluster {c + 1} (n={len(ids)}, mean ISEE "
          f"{isee_all[labels_km == c].mean():+.3f}): {', '.join(ids)}")

top = np.argsort(-isee_all)
print("  top-5:", [(full.ids[j], f"{isee_all[j]:+.3f}") for j in top[:5]])
print("  bottom-5:", [(full.ids[j], f"{isee_all[j]:+.3f}") for j in top[-5:]])
print("done.")
