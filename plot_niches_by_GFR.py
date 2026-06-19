"""
Représentation des niches spatiales par patient ordonnés par GFR croissant
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

out = Path("plots/niches_by_GFR")
out.mkdir(parents=True, exist_ok=True)

# ── Charger adata et Diagnosis ───────────────────────────────────────────────
print("Chargement...")
adata = ad.read_h5ad("adata_spatial_final.h5ad")

diag = pd.read_csv("Diagnosis.csv", sep=";")
diag["GFR"] = diag["GFR"].astype(str).str.strip()

# Extraire l'ID de base depuis Sample ID (retirer _Xenium / _CosMx)
def base_id(s):
    s = str(s).strip()
    for suf in ["_Xenium", "_CosMx"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

diag["orig_ident"] = diag["Sample ID"].apply(base_id)

# Garder une ligne par patient (dédupliquer — même GFR pour Xenium+CosMx)
diag_unique = diag.drop_duplicates("orig_ident")[["orig_ident", "GFR", "Condition"]].copy()
diag_unique = diag_unique[diag_unique["GFR"].notna() & (diag_unique["GFR"] != "nan")]

print("Patients avec GFR:", len(diag_unique))
print("Valeurs GFR:", diag_unique["GFR"].value_counts().to_dict())

# Ordre GFR croissant
gfr_order  = ["<30", "30-60", ">60"]
gfr_colors = {"<30": "#d6604d", "30-60": "#f4a582", ">60": "#4393c3"}
gfr_labels = {"<30": "GFR < 30\n(sévère)", "30-60": "GFR 30–60\n(modéré)", ">60": "GFR > 60\n(préservé)"}

# Proportions de niches par patient (depuis adata)
meta = adata.obs[["orig_ident", "niches_annotation_based"]].copy()
meta["orig_ident"] = meta["orig_ident"].astype(str)

counts = (meta.groupby(["orig_ident", "niches_annotation_based"], observed=True)
              .size().unstack(fill_value=0))
props  = counts.div(counts.sum(axis=1), axis=0)
# S'assurer que les colonnes sont des strings simples (pas Categorical)
props.columns = props.columns.astype(str)
props.index   = props.index.astype(str)

# Joindre avec GFR
diag_idx = diag_unique.set_index("orig_ident")[["GFR","Condition"]]
diag_idx.index = diag_idx.index.astype(str)
props_gfr = props.join(diag_idx, how="inner")
print(f"\nPatients avec GFR + niches: {len(props_gfr)}")

# Trier par GFR croissant puis par Condition
props_gfr["GFR_cat"] = pd.Categorical(props_gfr["GFR"], categories=gfr_order, ordered=True)
props_gfr = props_gfr.sort_values(["GFR_cat", "Condition"])

niche_cols = [c for c in props_gfr.columns if c not in ["GFR","Condition","GFR_cat"]]

# Palette niches
palette = sns.color_palette("tab20", n_colors=len(niche_cols))
niche_colors = dict(zip(niche_cols, palette))

# ── 1. Stacked bar ordonné par GFR ──────────────────────────────────────────
print("\nFigure 1: stacked bar par GFR...")
n_pat = len(props_gfr)
fig, ax = plt.subplots(figsize=(max(14, n_pat * 0.38), 7))
x = np.arange(n_pat)
bottom = np.zeros(n_pat)

for niche in niche_cols:
    vals = props_gfr[niche].values
    ax.bar(x, vals, bottom=bottom, color=niche_colors[niche],
           label=niche, width=0.85, edgecolor="none")
    bottom += vals

# Séparateurs GFR et annotations
prev = 0
for gfr in gfr_order:
    n = (props_gfr["GFR_cat"] == gfr).sum()
    if n == 0:
        prev += n
        continue
    if prev > 0:
        ax.axvline(prev - 0.5, color="black", lw=1.5, ls="--")
    ax.text(prev + n / 2, 1.035, gfr_labels[gfr],
            ha="center", fontsize=10, fontweight="bold",
            color=gfr_colors[gfr], transform=ax.get_xaxis_transform())
    # rectangle coloré sous les barres
    ax.axvspan(prev - 0.5, prev + n - 0.5, ymin=0, ymax=0.02,
               color=gfr_colors[gfr], alpha=0.5, zorder=5)
    prev += n

# xticks
ax.set_xticks(x)
ax.set_xticklabels(props_gfr.index, rotation=65, ha="right", fontsize=7)
for tick, pat in zip(ax.get_xticklabels(), props_gfr.index):
    gfr_val = props_gfr.loc[pat, "GFR"]
    tick.set_color(gfr_colors.get(gfr_val, "black"))

# Condition sous le nom du patient
for xi, pat in enumerate(props_gfr.index):
    cond = props_gfr.loc[pat, "Condition"]
    ax.text(xi, -0.06, cond, rotation=65, ha="right", fontsize=5.5,
            color="grey", transform=ax.get_xaxis_transform())

ax.set_ylabel("Proportion de cellules", fontsize=12)
ax.set_ylim(0, 1)
ax.set_title("Niches spatiales par patient ordonnés par GFR croissant (cellules immunes)", fontsize=13)
ax.legend(handles=[mpatches.Patch(color=niche_colors[n], label=n) for n in niche_cols],
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5, title="Niche")
fig.tight_layout()
fig.savefig(out / "01_niches_by_GFR_stacked.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 01_niches_by_GFR_stacked.png")

# ── 2. Boxplot : proportion de chaque niche par catégorie GFR ───────────────
print("Figure 2: boxplot par catégorie GFR...")
niche_data = props_gfr[niche_cols + ["GFR_cat"]].melt(
    id_vars="GFR_cat", var_name="niche", value_name="proportion"
)

n_niches = len(niche_cols)
ncols = 4
nrows = int(np.ceil(n_niches / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.2))
axes = axes.flatten()

for i, niche in enumerate(niche_cols):
    ax = axes[i]
    groups = [props_gfr.loc[props_gfr["GFR_cat"] == g, niche].dropna()
              for g in gfr_order if (props_gfr["GFR_cat"] == g).sum() > 0]
    labels_g = [gfr_labels[g].replace("\n", " ")
                for g in gfr_order if (props_gfr["GFR_cat"] == g).sum() > 0]
    colors_g = [gfr_colors[g]
                for g in gfr_order if (props_gfr["GFR_cat"] == g).sum() > 0]

    bp = ax.boxplot(groups, tick_labels=labels_g, patch_artist=True,
                    medianprops=dict(color="black", lw=2), widths=0.5)
    for patch, col in zip(bp["boxes"], colors_g):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    # Points individuels
    for j, (grp, col) in enumerate(zip(groups, colors_g)):
        jitter = np.random.uniform(-0.1, 0.1, len(grp))
        ax.scatter(j + 1 + jitter, grp, color=col, s=18, alpha=0.8, zorder=3)

    # Kruskal-Wallis si 3 groupes, sinon Mann-Whitney
    if len(groups) == 3:
        _, p = stats.kruskal(*groups)
        test = "KW"
    elif len(groups) == 2:
        _, p = stats.mannwhitneyu(*groups, alternative="two-sided")
        test = "MW"
    else:
        p = np.nan
        test = ""
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    title_str = f"{niche}\n{stars} p={p:.3f} ({test})" if not np.isnan(p) else niche
    ax.set_title(title_str, fontsize=8)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Proportion de chaque niche par catégorie de GFR (Kruskal-Wallis / Mann-Whitney)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(out / "02_niches_boxplot_by_GFR.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 02_niches_boxplot_by_GFR.png")

# ── 3. Scatter : corrélation proportion niche ~ GFR (valeur numérique) ───────
print("Figure 3: scatter corrélation niche ~ GFR...")
gfr_num = {"<30": 15, "30-60": 45, ">60": 75}
props_gfr["GFR_num"] = props_gfr["GFR"].map(gfr_num)

n_niches = len(niche_cols)
ncols = 4
nrows = int(np.ceil(n_niches / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
axes = axes.flatten()

corr_results = []
for i, niche in enumerate(niche_cols):
    ax = axes[i]
    sub = props_gfr[[niche, "GFR_num", "GFR"]].dropna()
    x_vals = sub["GFR_num"].values
    y_vals = sub[niche].values

    # Points colorés par GFR
    for gfr_cat, col in gfr_colors.items():
        mask = sub["GFR"] == gfr_cat
        ax.scatter(sub.loc[mask, "GFR_num"] + np.random.uniform(-2, 2, mask.sum()),
                   sub.loc[mask, niche], color=col, s=30, alpha=0.8, label=gfr_cat, zorder=3)

    # Régression linéaire
    if len(x_vals) > 3:
        r, p = stats.spearmanr(x_vals, y_vals)
        z = np.polyfit(x_vals, y_vals, 1)
        poly = np.poly1d(z)
        xr = np.linspace(10, 80, 100)
        ax.plot(xr, poly(xr), "k--", lw=1.2, alpha=0.7)
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.set_title(f"{niche}\nSpearman r={r:.2f}, {stars}", fontsize=8)
        corr_results.append({"niche": niche, "spearman_r": r, "pval": p, "sig": stars})
    else:
        ax.set_title(niche, fontsize=8)

    ax.set_xticks([15, 45, 75])
    ax.set_xticklabels(["<30", "30-60", ">60"], fontsize=7)
    ax.set_xlabel("GFR", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.tick_params(labelsize=7)

axes[0].legend(fontsize=7, title="GFR", loc="upper right")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Corrélation niche spatiale ~ GFR (Spearman)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(out / "03_niches_scatter_GFR_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 03_niches_scatter_GFR_correlation.png")

# ── 4. Heatmap résumé : proportion moyenne par niche × GFR ──────────────────
print("Figure 4: heatmap moyenne niche × GFR...")
mean_by_gfr = props_gfr.groupby("GFR_cat", observed=True)[niche_cols].mean()
mean_by_gfr.index = [gfr_labels[g] for g in mean_by_gfr.index]

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(mean_by_gfr, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.5, cbar_kws={"label": "Proportion moyenne"})
ax.set_xlabel("Niche spatiale", fontsize=11)
ax.set_ylabel("Catégorie GFR", fontsize=11)
ax.set_title("Proportion moyenne des niches par catégorie de GFR", fontsize=13)
ax.tick_params(axis="x", rotation=35, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
fig.tight_layout()
fig.savefig(out / "04_niches_heatmap_mean_by_GFR.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 04_niches_heatmap_mean_by_GFR.png")

# ── 5. Résumé corrélations ───────────────────────────────────────────────────
if corr_results:
    corr_df = pd.DataFrame(corr_results).sort_values("spearman_r")
    corr_df.to_csv(out / "spearman_niche_GFR.csv", index=False)
    print("\n  Corrélations significatives (p<0.05):")
    print(corr_df[corr_df["pval"] < 0.05].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#d6604d" if r < 0 else "#4393c3" for r in corr_df["spearman_r"]]
    bars = ax.barh(corr_df["niche"], corr_df["spearman_r"], color=colors_bar, alpha=0.8)
    for bar, row in zip(bars, corr_df.itertuples()):
        if row.pval < 0.05:
            ax.text(row.spearman_r + (0.01 if row.spearman_r >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    row.sig, va="center", ha="left" if row.spearman_r >= 0 else "right",
                    fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Spearman r (niche ~ GFR)", fontsize=11)
    ax.set_title("Corrélation proportion niche ~ GFR\n(bleu = augmente avec GFR, rouge = diminue avec GFR)",
                 fontsize=12)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out / "05_spearman_barplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 05_spearman_barplot.png")

print(f"\nTermine. Figures dans {out}/")
