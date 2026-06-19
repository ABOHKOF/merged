"""
Représentation des Immune Microenvironments (immune_ME) par patient ordonnés par GFR croissant
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
from scipy import stats
from pathlib import Path

out = Path("plots/immuneME_by_GFR")
out.mkdir(parents=True, exist_ok=True)

# ── Charger ─────────────────────────────────────────────────────────────────
print("Chargement...")
adata = ad.read_h5ad("adata_spatial_final.h5ad")

diag = pd.read_csv("Diagnosis.csv", sep=";")
diag["GFR"] = diag["GFR"].astype(str).str.strip()

def base_id(s):
    s = str(s).strip()
    for suf in ["_Xenium", "_CosMx"]:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s

diag["orig_ident"] = diag["Sample ID"].apply(base_id)
diag_unique = diag.drop_duplicates("orig_ident")[["orig_ident","GFR","Condition"]].copy()
diag_unique = diag_unique[diag_unique["GFR"].notna() & (diag_unique["GFR"] != "nan")]

# Proportions immune_ME par patient
meta = adata.obs[["orig_ident","immune_ME"]].copy()
meta["orig_ident"] = meta["orig_ident"].astype(str)

counts = (meta.groupby(["orig_ident","immune_ME"], observed=True)
              .size().unstack(fill_value=0))
props  = counts.div(counts.sum(axis=1), axis=0)
props.columns = props.columns.astype(str)
props.index   = props.index.astype(str)

diag_idx = diag_unique.set_index("orig_ident")[["GFR","Condition"]]
diag_idx.index = diag_idx.index.astype(str)
props_gfr = props.join(diag_idx, how="inner")

print(f"Patients avec GFR + immune_ME: {len(props_gfr)}")

gfr_order  = ["<30", "30-60", ">60"]
gfr_colors = {"<30": "#d6604d", "30-60": "#f4a582", ">60": "#4393c3"}
gfr_labels = {"<30": "GFR < 30\n(sévère)", "30-60": "GFR 30–60\n(modéré)", ">60": "GFR > 60\n(préservé)"}

props_gfr["GFR_cat"] = pd.Categorical(props_gfr["GFR"], categories=gfr_order, ordered=True)
props_gfr = props_gfr.sort_values(["GFR_cat","Condition"])

me_cols = [c for c in props_gfr.columns if c not in ["GFR","Condition","GFR_cat"]]

# Palette immune ME
palette = {
    "Unknown":                   "#cccccc",
    "Immune Immune 1":           "#e41a1c",
    "Residential Immune ME":     "#377eb8",
    "Inj. Tubular Immune ME":    "#ff7f00",
    "Fibro Immune ME":           "#984ea3",
    "Vascular Immune ME":        "#a65628",
    "Glomerular Immune ME":      "#4daf4a",
    "B predom. Immune ME":       "#f781bf",
}
me_colors = {me: palette.get(me, "#888888") for me in me_cols}

# ── 1. Stacked bar par GFR ──────────────────────────────────────────────────
print("Figure 1: stacked bar...")
n_pat = len(props_gfr)
fig, ax = plt.subplots(figsize=(max(14, n_pat * 0.38), 7))
x = np.arange(n_pat)
bottom = np.zeros(n_pat)

for me in me_cols:
    vals = props_gfr[me].values
    ax.bar(x, vals, bottom=bottom, color=me_colors[me],
           label=me, width=0.85, edgecolor="none")
    bottom += vals

prev = 0
for gfr in gfr_order:
    n = (props_gfr["GFR_cat"] == gfr).sum()
    if n == 0:
        prev += n; continue
    if prev > 0:
        ax.axvline(prev - 0.5, color="black", lw=1.5, ls="--")
    ax.text(prev + n / 2, 1.035, gfr_labels[gfr],
            ha="center", fontsize=10, fontweight="bold",
            color=gfr_colors[gfr], transform=ax.get_xaxis_transform())
    ax.axvspan(prev - 0.5, prev + n - 0.5, ymin=0, ymax=0.02,
               color=gfr_colors[gfr], alpha=0.5, zorder=5)
    prev += n

ax.set_xticks(x)
ax.set_xticklabels(props_gfr.index, rotation=65, ha="right", fontsize=7)
for tick, pat in zip(ax.get_xticklabels(), props_gfr.index):
    tick.set_color(gfr_colors.get(props_gfr.loc[pat,"GFR"], "black"))
for xi, pat in enumerate(props_gfr.index):
    ax.text(xi, -0.06, props_gfr.loc[pat,"Condition"], rotation=65, ha="right",
            fontsize=5.5, color="grey", transform=ax.get_xaxis_transform())

ax.set_ylabel("Proportion de cellules", fontsize=12)
ax.set_ylim(0, 1)
ax.set_title("Immune Microenvironments par patient ordonnés par GFR croissant", fontsize=13)
ax.legend(handles=[mpatches.Patch(color=me_colors[m], label=m) for m in me_cols],
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title="Immune ME")
fig.tight_layout()
fig.savefig(out / "01_immuneME_by_GFR_stacked.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 01_immuneME_by_GFR_stacked.png")

# ── 2. Heatmap proportion moyenne ME × GFR ──────────────────────────────────
print("Figure 2: heatmap...")
mean_by_gfr = props_gfr.groupby("GFR_cat", observed=True)[me_cols].mean()
mean_by_gfr.index = [gfr_labels[g] for g in mean_by_gfr.index]

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(mean_by_gfr, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f",
            annot_kws={"size": 9}, linewidths=0.5,
            cbar_kws={"label": "Proportion moyenne"})
ax.set_xlabel("Immune Microenvironment", fontsize=11)
ax.set_ylabel("Catégorie GFR", fontsize=11)
ax.set_title("Proportion moyenne des Immune ME par catégorie de GFR", fontsize=13)
ax.tick_params(axis="x", rotation=30, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
fig.tight_layout()
fig.savefig(out / "02_immuneME_heatmap_mean_by_GFR.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 02_immuneME_heatmap_mean_by_GFR.png")

# ── 3. Boxplot par catégorie GFR ────────────────────────────────────────────
print("Figure 3: boxplot...")
n_me = len(me_cols)
ncols = 4
nrows = int(np.ceil(n_me / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.2))
axes = axes.flatten()

for i, me in enumerate(me_cols):
    ax = axes[i]
    groups = [props_gfr.loc[props_gfr["GFR_cat"] == g, me].dropna()
              for g in gfr_order if (props_gfr["GFR_cat"] == g).sum() > 0]
    labels_g = [gfr_labels[g].replace("\n"," ")
                for g in gfr_order if (props_gfr["GFR_cat"] == g).sum() > 0]
    colors_g = [gfr_colors[g]
                for g in gfr_order if (props_gfr["GFR_cat"] == g).sum() > 0]

    bp = ax.boxplot(groups, tick_labels=labels_g, patch_artist=True,
                    medianprops=dict(color="black", lw=2), widths=0.5)
    for patch, col in zip(bp["boxes"], colors_g):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    for j, (grp, col) in enumerate(zip(groups, colors_g)):
        jitter = np.random.uniform(-0.1, 0.1, len(grp))
        ax.scatter(j + 1 + jitter, grp, color=col, s=20, alpha=0.8, zorder=3)

    if len(groups) == 3:
        _, p = stats.kruskal(*groups); test = "KW"
    elif len(groups) == 2:
        _, p = stats.mannwhitneyu(*groups, alternative="two-sided"); test = "MW"
    else:
        p = np.nan; test = ""
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.set_title(f"{me}\n{stars} p={p:.3f} ({test})" if not np.isnan(p) else me, fontsize=8)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.tick_params(labelsize=6.5)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Proportion des Immune ME par catégorie de GFR", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(out / "03_immuneME_boxplot_by_GFR.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 03_immuneME_boxplot_by_GFR.png")

# ── 4. Corrélations Spearman ME ~ GFR numérique ─────────────────────────────
print("Figure 4: corrélations Spearman...")
gfr_num = {"<30": 15, "30-60": 45, ">60": 75}
props_gfr["GFR_num"] = props_gfr["GFR"].map(gfr_num)

corr_results = []
for me in me_cols:
    sub = props_gfr[[me,"GFR_num"]].dropna()
    if len(sub) > 3:
        r, p = stats.spearmanr(sub["GFR_num"], sub[me])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        corr_results.append({"immune_ME": me, "spearman_r": r, "pval": p, "sig": stars})

corr_df = pd.DataFrame(corr_results).sort_values("spearman_r")
corr_df.to_csv(out / "spearman_immuneME_GFR.csv", index=False)

print("\n  Corrélations significatives (p<0.05):")
print(corr_df[corr_df["pval"] < 0.05].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = [gfr_colors["<30"] if r < 0 else gfr_colors[">60"]
              for r in corr_df["spearman_r"]]
bars = ax.barh(corr_df["immune_ME"], corr_df["spearman_r"],
               color=colors_bar, alpha=0.85)
for bar, row in zip(bars, corr_df.itertuples()):
    if row.pval < 0.05:
        offset = 0.01 if row.spearman_r >= 0 else -0.01
        ax.text(row.spearman_r + offset, bar.get_y() + bar.get_height() / 2,
                row.sig, va="center",
                ha="left" if row.spearman_r >= 0 else "right",
                fontsize=10, fontweight="bold")
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Spearman r (Immune ME ~ GFR)", fontsize=11)
ax.set_title("Corrélation Immune ME ~ GFR\n(bleu = augmente avec GFR, rouge = diminue avec GFR)",
             fontsize=12)
ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(out / "04_spearman_immuneME_GFR.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 04_spearman_immuneME_GFR.png")

# ── 5. Scatter proportion ~ GFR pour chaque ME ──────────────────────────────
print("Figure 5: scatter...")
ncols = 4
nrows = int(np.ceil(n_me / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
axes = axes.flatten()

for i, me in enumerate(me_cols):
    ax = axes[i]
    sub = props_gfr[[me,"GFR_num","GFR"]].dropna()
    for gfr_cat, col in gfr_colors.items():
        mask = sub["GFR"] == gfr_cat
        ax.scatter(sub.loc[mask,"GFR_num"] + np.random.uniform(-2,2,mask.sum()),
                   sub.loc[mask, me], color=col, s=35, alpha=0.85,
                   label=gfr_cat, zorder=3)
    if len(sub) > 3:
        z = np.polyfit(sub["GFR_num"], sub[me], 1)
        xr = np.linspace(10, 80, 100)
        ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1.2, alpha=0.7)
        r, p = stats.spearmanr(sub["GFR_num"], sub[me])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.set_title(f"{me}\nr={r:.2f}, {stars}", fontsize=8)
    else:
        ax.set_title(me, fontsize=8)
    ax.set_xticks([15, 45, 75])
    ax.set_xticklabels(["<30","30-60",">60"], fontsize=7)
    ax.set_xlabel("GFR", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.tick_params(labelsize=7)

axes[0].legend(fontsize=7, title="GFR", loc="upper right")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Proportion des Immune ME ~ GFR (Spearman)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(out / "05_immuneME_scatter_GFR.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 05_immuneME_scatter_GFR.png")

print(f"\nTermine. Figures dans {out}/")
