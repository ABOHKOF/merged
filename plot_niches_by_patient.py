"""
Représentation des niches spatiales par patient
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

out = Path("plots/niches_by_patient")
out.mkdir(parents=True, exist_ok=True)

print("Chargement...")
adata = ad.read_h5ad("adata_spatial_final.h5ad")

# Palette niches
niches = adata.obs["niches_annotation_based"].cat.categories.tolist()
palette = sns.color_palette("tab20", n_colors=len(niches))
niche_colors = dict(zip(niches, palette))

# Ajouter condition et trier patients par condition
meta = adata.obs[["orig_ident", "niches_annotation_based", "Condition"]].copy()
meta["orig_ident"] = meta["orig_ident"].astype(str)

# Condition simplifiée pour le tri
ckd_conds = {"CKD", "DKD", "DKD+FSGS", "FSGS", "DM", "DM/HTN",
             "C3GN", "IgA", "MN", "AA amyloid", "TMA", "HTN"}
meta["broad_cond"] = meta["Condition"].apply(
    lambda x: "Control" if x == "Control" else "CKD"
)

# ── 1. Stacked bar : proportion de chaque niche par patient ──────────────────
print("Figure 1 : stacked bar proportions...")

counts = (meta.groupby(["orig_ident", "niches_annotation_based"], observed=True)
              .size().unstack(fill_value=0))
props  = counts.div(counts.sum(axis=1), axis=0)

# Trier : Control d'abord, puis CKD, par condition détaillée
patient_cond = (meta.groupby("orig_ident")["Condition"]
                    .first().reindex(props.index))
patient_broad = (meta.groupby("orig_ident")["broad_cond"]
                     .first().reindex(props.index))
order = props.index[
    pd.Series(patient_broad.values, index=props.index)
      .sort_values().index.map(lambda x: props.index.get_loc(x))
    if False else  # fallback: trier par broad puis condition
    np.argsort(
        [f"{patient_broad[p]}_{patient_cond[p]}_{p}" for p in props.index]
    )
]
props = props.loc[order]
patient_cond  = patient_cond.loc[order]
patient_broad = patient_broad.loc[order]

n_patients = len(props)
fig, ax = plt.subplots(figsize=(max(16, n_patients * 0.35), 7))

bottom = np.zeros(n_patients)
x = np.arange(n_patients)

for niche in props.columns:
    vals = props[niche].values
    ax.bar(x, vals, bottom=bottom,
           color=niche_colors[niche], label=niche, width=0.85, edgecolor="none")
    bottom += vals

# Séparateur Control / CKD
ctrl_mask = patient_broad == "Control"
n_ctrl = ctrl_mask.sum()
if n_ctrl > 0 and n_ctrl < n_patients:
    ax.axvline(n_ctrl - 0.5, color="black", lw=2, ls="--")
    ax.text(n_ctrl / 2, 1.02, "Control", ha="center", fontsize=11,
            fontweight="bold", color="#4393c3", transform=ax.get_xaxis_transform())
    ax.text(n_ctrl + (n_patients - n_ctrl) / 2, 1.02, "CKD",
            ha="center", fontsize=11, fontweight="bold", color="#d6604d",
            transform=ax.get_xaxis_transform())

# Colorier les xticks par condition
ax.set_xticks(x)
ax.set_xticklabels(props.index, rotation=70, ha="right", fontsize=7)
for tick, pat in zip(ax.get_xticklabels(), props.index):
    tick.set_color("#4393c3" if patient_broad[pat] == "Control" else "#d6604d")

# Annoter la condition détaillée sous le nom du patient
for xi, pat in enumerate(props.index):
    ax.text(xi, -0.06, patient_cond[pat], rotation=70, ha="right",
            fontsize=5.5, color="grey", transform=ax.get_xaxis_transform())

ax.set_ylabel("Proportion de cellules", fontsize=12)
ax.set_title("Distribution des niches spatiales par patient (cellules immunes)", fontsize=13)
ax.set_ylim(0, 1)
ax.legend(handles=[mpatches.Patch(color=niche_colors[n], label=n) for n in props.columns],
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title="Niche")
fig.tight_layout()
fig.savefig(out / "01_niches_stacked_bar_by_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 01_niches_stacked_bar_by_patient.png")

# ── 2. Heatmap : proportion niche × patient (clustermap) ────────────────────
print("Figure 2 : heatmap clustermap...")

row_colors = patient_broad.map({"Control": "#4393c3", "CKD": "#d6604d"})

g = sns.clustermap(
    props, cmap="YlOrRd", figsize=(14, max(10, n_patients * 0.22)),
    row_colors=row_colors.values, col_cluster=True, row_cluster=True,
    xticklabels=True, yticklabels=True,
    cbar_kws={"label": "Proportion"},
    dendrogram_ratio=(0.12, 0.15),
)
g.ax_heatmap.set_xlabel("Niche spatiale", fontsize=11)
g.ax_heatmap.set_ylabel("Patient", fontsize=11)
g.ax_heatmap.tick_params(axis="x", labelsize=9, rotation=30)
g.ax_heatmap.tick_params(axis="y", labelsize=7)
g.fig.suptitle("Proportion des niches par patient (cellules immunes)", y=1.01, fontsize=13)

legend_els = [mpatches.Patch(color="#4393c3", label="Control"),
              mpatches.Patch(color="#d6604d", label="CKD")]
g.ax_row_dendrogram.legend(handles=legend_els, loc="center", fontsize=9, title="Condition")
g.fig.savefig(out / "02_niches_heatmap_clustermap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 02_niches_heatmap_clustermap.png")

# ── 3. Boxplot : proportion de chaque niche — Control vs CKD ────────────────
print("Figure 3 : boxplot Control vs CKD par niche...")

props_reset = props.copy()
props_reset["broad_cond"] = patient_broad
props_long = props_reset.melt(id_vars="broad_cond",
                               var_name="niche", value_name="proportion")

from scipy import stats
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, niche in enumerate(props.columns):
    ax = axes[i]
    sub = props_long[props_long["niche"] == niche]
    ctrl = sub[sub["broad_cond"] == "Control"]["proportion"]
    ckd  = sub[sub["broad_cond"] == "CKD"]["proportion"]

    ax.boxplot([ctrl, ckd], labels=["Control", "CKD"],
               patch_artist=True,
               boxprops=dict(facecolor="none"),
               medianprops=dict(color="black", lw=2))

    # Points individuels
    jitter = np.random.uniform(-0.08, 0.08, len(ctrl))
    ax.scatter(1 + jitter, ctrl, color="#4393c3", s=20, alpha=0.7, zorder=3)
    jitter = np.random.uniform(-0.08, 0.08, len(ckd))
    ax.scatter(2 + jitter, ckd,  color="#d6604d", s=20, alpha=0.7, zorder=3)

    # Test Mann-Whitney
    if len(ctrl) > 1 and len(ckd) > 1:
        _, p = stats.mannwhitneyu(ctrl, ckd, alternative="two-sided")
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.set_title(f"{niche}\n{stars} (p={p:.3f})", fontsize=8)
    else:
        ax.set_title(niche, fontsize=8)

    ax.set_ylabel("Proportion", fontsize=7)
    ax.tick_params(labelsize=8)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Proportion de chaque niche par patient — Control vs CKD (Mann-Whitney)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(out / "03_niches_boxplot_CKD_vs_Control.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 03_niches_boxplot_CKD_vs_Control.png")

# ── 4. Stacked bar par condition détaillée (moyenne) ────────────────────────
print("Figure 4 : stacked bar moyen par condition...")

meta2 = meta.copy()
props_by_cond = (meta2.groupby(["Condition", "niches_annotation_based"], observed=True)
                      .size().unstack(fill_value=0))
props_by_cond = props_by_cond.div(props_by_cond.sum(axis=1), axis=0)

cond_order = ["Control", "DKD", "DKD+FSGS", "FSGS", "DM", "DM/HTN",
              "CKD", "C3GN", "IgA", "MN", "AA amyloid", "TMA", "HTN"]
cond_order = [c for c in cond_order if c in props_by_cond.index]
props_by_cond = props_by_cond.loc[cond_order]

fig, ax = plt.subplots(figsize=(14, 6))
bottom = np.zeros(len(props_by_cond))
x = np.arange(len(props_by_cond))

for niche in props_by_cond.columns:
    ax.bar(x, props_by_cond[niche].values, bottom=bottom,
           color=niche_colors.get(niche, "grey"), label=niche, width=0.75)
    bottom += props_by_cond[niche].values

ax.set_xticks(x)
ax.set_xticklabels(cond_order, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Proportion moyenne", fontsize=12)
ax.set_title("Composition en niches spatiales par condition (cellules immunes)", fontsize=13)
ax.legend(handles=[mpatches.Patch(color=niche_colors[n], label=n) for n in props_by_cond.columns],
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title="Niche")
fig.tight_layout()
fig.savefig(out / "04_niches_stacked_by_condition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 04_niches_stacked_by_condition.png")

# ── 5. Résumé statistique ────────────────────────────────────────────────────
summary = props.copy()
summary["Condition"] = patient_cond
summary["Group"] = patient_broad
summary.to_csv(out / "niche_proportions_by_patient.csv")
print("  saved niche_proportions_by_patient.csv")

print(f"\nTerminé. Figures dans {out}/")
