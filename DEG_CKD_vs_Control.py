"""
DEG analysis: CKD (merged CKD + DKD) vs Control
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sc.settings.verbosity = 1
out = Path("plots/DEG_CKD_vs_Control")
out.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = str(out)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading adata...")
adata = sc.read_h5ad("adata_spatial_final.h5ad")
print(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

# ── Merge CKD + DKD → 'CKD', keep Control ────────────────────────────────────
ckd_conditions = {"CKD", "DKD", "DKD+FSGS"}
adata.obs["group"] = adata.obs["Condition"].apply(
    lambda x: "CKD" if x in ckd_conditions else ("Control" if x == "Control" else None)
)

mask = adata.obs["group"].notna()
adata_sub = adata[mask].copy()
adata_sub.obs["group"] = adata_sub.obs["group"].astype("category")

n_ckd  = (adata_sub.obs["group"] == "CKD").sum()
n_ctrl = (adata_sub.obs["group"] == "Control").sum()
print(f"  CKD (CKD+DKD+DKD+FSGS): {n_ckd:,} cells")
print(f"  Control:                 {n_ctrl:,} cells")

# ── Normalise ─────────────────────────────────────────────────────────────────
print("\nNormalising...")
adata_sub.X = adata_sub.layers["counts"].copy()
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)
sc.pp.filter_genes(adata_sub, min_cells=100)
print(f"  Genes after filter (min 100 cells): {adata_sub.shape[1]}")

# ── DEG: Wilcoxon CKD vs Control ─────────────────────────────────────────────
print("\nRunning Wilcoxon rank-sum test (CKD vs Control)...")
sc.tl.rank_genes_groups(
    adata_sub, groupby="group", groups=["CKD"],
    reference="Control", method="wilcoxon",
    key_added="deg_CKD_vs_Control",
)

all_deg = sc.get.rank_genes_groups_df(
    adata_sub, group="CKD", key="deg_CKD_vs_Control"
)
all_deg["-log10_padj"] = -np.log10(all_deg["pvals_adj"].clip(1e-300))
all_deg["regulation"] = "ns"
all_deg.loc[(all_deg["pvals_adj"] < 0.05) & (all_deg["logfoldchanges"] >  0.5), "regulation"] = "Up in CKD"
all_deg.loc[(all_deg["pvals_adj"] < 0.05) & (all_deg["logfoldchanges"] < -0.5), "regulation"] = "Down in CKD"

sig_deg = all_deg[all_deg["regulation"] != "ns"]
all_deg.to_csv(out / "all_DEG_CKD_vs_Control.csv", index=False)
sig_deg.to_csv(out / "sig_DEG_CKD_vs_Control.csv", index=False)

up   = sig_deg[sig_deg["regulation"] == "Up in CKD"]
down = sig_deg[sig_deg["regulation"] == "Down in CKD"]
print(f"\n  Total genes tested:   {len(all_deg)}")
print(f"  Significant (p<0.05, |LFC|>0.5):")
print(f"    Up   in CKD: {len(up)}")
print(f"    Down in CKD: {len(down)}")
print(f"\n  Top 10 upregulated:")
print(up.nlargest(10, "logfoldchanges")[["names","logfoldchanges","pvals_adj"]].to_string())
print(f"\n  Top 10 downregulated:")
print(down.nsmallest(10, "logfoldchanges")[["names","logfoldchanges","pvals_adj"]].to_string())

# ── Volcano plot ──────────────────────────────────────────────────────────────
print("\nPlotting volcano...")
fig, ax = plt.subplots(figsize=(10, 8))

colors = {"ns": "#cccccc", "Up in CKD": "#d6604d", "Down in CKD": "#4393c3"}
for label, grp in all_deg.groupby("regulation"):
    s = 3 if label == "ns" else 6
    a = 0.25 if label == "ns" else 0.75
    ax.scatter(grp["logfoldchanges"], grp["-log10_padj"],
               s=s, alpha=a, color=colors[label], label=label, linewidths=0, rasterized=True)

# Label top genes
top_label = pd.concat([
    up.nlargest(20, "-log10_padj"),
    down.nlargest(20, "-log10_padj"),
])
for _, row in top_label.iterrows():
    ax.text(row["logfoldchanges"], row["-log10_padj"], row["names"],
            fontsize=6.5, ha="center", va="bottom", fontweight="bold")

ax.axvline( 0.5, ls="--", lw=0.8, color="black")
ax.axvline(-0.5, ls="--", lw=0.8, color="black")
ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="black")
ax.set_xlabel("Log2 fold change  (CKD vs Control)", fontsize=12)
ax.set_ylabel("-log10(adjusted p-value)", fontsize=12)
ax.set_title(
    f"Volcano: CKD (CKD+DKD+DKD+FSGS, n={n_ckd:,}) vs Control (n={n_ctrl:,})\n"
    f"Up: {len(up)}  |  Down: {len(down)}  |  |LFC|>0.5, padj<0.05",
    fontsize=11,
)
ax.legend(markerscale=2, fontsize=10)
fig.tight_layout()
fig.savefig(out / "01_volcano_CKD_vs_Control.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 01_volcano_CKD_vs_Control.png")

# ── Dot plot: top 20 up + top 20 down ────────────────────────────────────────
top_genes = (
    up.nlargest(20, "logfoldchanges")["names"].tolist() +
    down.nsmallest(20, "logfoldchanges")["names"].tolist()
)
top_genes = [g for g in top_genes if g in adata_sub.var_names]

sc.pl.dotplot(
    adata_sub, var_names=top_genes, groupby="group",
    categories_order=["Control", "CKD"],
    save="_top20_CKD_vs_Control.png", show=False,
    title="Top 20 up/down DEGs: CKD vs Control  (|LFC|>0.5, padj<0.05)",
)
print("  saved dotplot__top20_CKD_vs_Control.png")

# ── Violin plot: top 6 up + top 6 down ───────────────────────────────────────
vln_genes = (
    up.nlargest(6, "logfoldchanges")["names"].tolist() +
    down.nsmallest(6, "logfoldchanges")["names"].tolist()
)
vln_genes = [g for g in vln_genes if g in adata_sub.var_names]

sc.pl.violin(
    adata_sub, keys=vln_genes, groupby="group",
    order=["Control", "CKD"],
    rotation=30, save="_top6_CKD_vs_Control.png", show=False,
)
print("  saved violin__top6_CKD_vs_Control.png")

# ── Heatmap: top 50 DEGs ─────────────────────────────────────────────────────
hm_genes = (
    up.nlargest(25, "-log10_padj")["names"].tolist() +
    down.nlargest(25, "-log10_padj")["names"].tolist()
)
hm_genes = [g for g in hm_genes if g in adata_sub.var_names]

sc.pl.matrixplot(
    adata_sub, var_names=hm_genes, groupby="group",
    categories_order=["Control", "CKD"],
    standard_scale="var", cmap="RdBu_r",
    save="_heatmap_top50_CKD_vs_Control.png", show=False,
    title="Top 50 DEGs (most significant): CKD vs Control",
)
print("  saved matrixplot__heatmap_top50_CKD_vs_Control.png")

# ── DEG by Immune ME (within CKD vs Control) ─────────────────────────────────
print("\nDEG by Immune ME (CKD vs Control within each ME)...")
me_results = []
for me in adata_sub.obs["immune_ME"].unique():
    sub = adata_sub[adata_sub.obs["immune_ME"] == me].copy()
    n_ckd_me  = (sub.obs["group"] == "CKD").sum()
    n_ctrl_me = (sub.obs["group"] == "Control").sum()
    if n_ckd_me < 30 or n_ctrl_me < 30:
        print(f"  Skipping {me}: too few cells (CKD={n_ckd_me}, Ctrl={n_ctrl_me})")
        continue
    try:
        sc.tl.rank_genes_groups(sub, groupby="group", groups=["CKD"],
                                reference="Control", method="wilcoxon",
                                key_added="deg_me")
        df = sc.get.rank_genes_groups_df(sub, group="CKD", key="deg_me",
                                          pval_cutoff=0.05)
        df["immune_ME"] = me
        df["n_CKD"]  = n_ckd_me
        df["n_Ctrl"] = n_ctrl_me
        me_results.append(df)
        print(f"  {me}: {len(df)} sig. DEGs  (CKD={n_ckd_me:,}, Ctrl={n_ctrl_me:,})")
    except Exception as e:
        print(f"  {me}: failed — {e}")

if me_results:
    me_df = pd.concat(me_results, ignore_index=True)
    me_df.to_csv(out / "DEG_by_immuneME_CKD_vs_Control.csv", index=False)
    print(f"\n  saved DEG_by_immuneME_CKD_vs_Control.csv  ({len(me_df):,} sig. markers)")

    # Bubble plot: top 5 up per ME
    top5 = (
        me_df[me_df["logfoldchanges"] > 0]
        .groupby("immune_ME", group_keys=False)
        .apply(lambda x: x.nlargest(5, "scores"))
    )
    pivot_lfc  = top5.pivot_table(index="names", columns="immune_ME", values="logfoldchanges", fill_value=0)
    pivot_padj = top5.pivot_table(index="names", columns="immune_ME", values="pvals_adj", fill_value=1)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_lfc.columns)*1.5), max(6, len(pivot_lfc)*0.35)))
    for j, me_col in enumerate(pivot_lfc.columns):
        for i, gene in enumerate(pivot_lfc.index):
            lfc  = pivot_lfc.loc[gene, me_col]
            padj = pivot_padj.loc[gene, me_col]
            if lfc == 0:
                continue
            size = max(20, -np.log10(padj + 1e-300) * 15)
            ax.scatter(j, i, s=size, c=lfc, cmap="Reds", vmin=0, vmax=2,
                       edgecolors="grey", linewidths=0.3)
    ax.set_xticks(range(len(pivot_lfc.columns)))
    ax.set_xticklabels(pivot_lfc.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_lfc.index)))
    ax.set_yticklabels(pivot_lfc.index, fontsize=8)
    ax.set_title("Top 5 upregulated DEGs per Immune ME (CKD vs Control)\nSize = -log10(padj), Color = LFC", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 2))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Log2 FC")
    fig.tight_layout()
    fig.savefig(out / "02_bubble_DEG_by_ME.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 02_bubble_DEG_by_ME.png")

# ── Cell-type abundance: CKD vs Control ──────────────────────────────────────
print("\nCell-type abundance comparison...")
ct_cols = [
    "B_Cells", "B_Memory", "B_Naive", "Basophile",
    "CD14_Mono", "CD16_Mono", "CD4_Activated", "CD4_Trm", "CD4_signaling",
    "CD8_MAIT", "CD8_central_memory", "CD8_cytotoxic/effector_memory",
    "FOLR2+_resident", "FOLR2_CKD", "NK/T_cells", "NK_cytotoxic",
    "Neutro_FPR2+", "Plasma_cells", "TREM2+_macro", "cDC", "pDC",
]

abund = adata_sub.obs[ct_cols + ["group", "orig_ident"]].copy()
sample_abund = abund.groupby(["orig_ident", "group"])[ct_cols].mean().reset_index()

from scipy import stats
pvals = {}
for ct in ct_cols:
    ctrl_vals = sample_abund.loc[sample_abund["group"] == "Control", ct].dropna()
    ckd_vals  = sample_abund.loc[sample_abund["group"] == "CKD",     ct].dropna()
    if len(ctrl_vals) > 1 and len(ckd_vals) > 1:
        _, p = stats.mannwhitneyu(ckd_vals, ctrl_vals, alternative="two-sided")
    else:
        p = np.nan
    pvals[ct] = p

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(ct_cols))
w = 0.35
ctrl_means = sample_abund[sample_abund["group"] == "Control"][ct_cols].mean()
ckd_means  = sample_abund[sample_abund["group"] == "CKD"][ct_cols].mean()
ctrl_sems  = sample_abund[sample_abund["group"] == "Control"][ct_cols].sem()
ckd_sems   = sample_abund[sample_abund["group"] == "CKD"][ct_cols].sem()

ax.bar(x - w/2, ctrl_means, w, yerr=ctrl_sems, label="Control", color="#4393c3", alpha=0.85, capsize=3)
ax.bar(x + w/2, ckd_means,  w, yerr=ckd_sems,  label="CKD",     color="#d6604d", alpha=0.85, capsize=3)

for i, ct in enumerate(ct_cols):
    p = pvals.get(ct, np.nan)
    if not np.isnan(p) and p < 0.05:
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        ymax = max(ctrl_means[ct] + ctrl_sems[ct], ckd_means[ct] + ckd_sems[ct])
        ax.text(i, ymax + 0.01, stars, ha="center", fontsize=8, color="black")

ax.set_xticks(x)
ax.set_xticklabels(ct_cols, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean cell2location abundance (per sample)")
ax.set_title("Cell-type abundance: CKD (CKD+DKD+DKD+FSGS) vs Control\n(* p<0.05, ** p<0.01, *** p<0.001, Mann-Whitney U)")
ax.legend()
fig.tight_layout()
fig.savefig(out / "03_celltype_abundance_CKD_vs_Control.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 03_celltype_abundance_CKD_vs_Control.png")

print(f"\nDone. All outputs in {out}/")
