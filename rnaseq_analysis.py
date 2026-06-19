"""
RNA-seq / spatial transcriptomics analysis of adata_spatial_final.h5ad
Immune cells from kidney CKD study (CosMx + Xenium)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

sc.settings.verbosity = 1
sc.settings.figdir = "plots/rnaseq_analysis"
Path("plots/rnaseq_analysis").mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading adata...")
adata = sc.read_h5ad("adata_spatial_final.h5ad")
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Cell-type abundance columns from cell2location
ct_cols = [
    "B_Cells", "B_Memory", "B_Naive", "Basophile",
    "CD14_Mono", "CD16_Mono", "CD4_Activated", "CD4_Trm", "CD4_signaling",
    "CD8_MAIT", "CD8_central_memory", "CD8_cytotoxic/effector_memory",
    "FOLR2+_resident", "FOLR2_CKD", "NK/T_cells", "NK_cytotoxic",
    "Neutro_FPR2+", "Plasma_cells", "TREM2+_macro", "cDC", "pDC",
]

# Group conditions: broad CKD vs Control
adata.obs["broad_condition"] = adata.obs["Condition"].map(
    lambda x: "Control" if x == "Control" else "CKD"
).astype("category")

print("\n── 1. QC metrics ────────────────────────────────────────────────────")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ax, col, label in zip(
    axes[0],
    ["nCount_RNA", "nFeature_RNA", "total_counts"],
    ["UMI counts", "Genes detected", "Total counts (scanpy)"],
):
    for cond, color in [("Control", "#4393c3"), ("CKD", "#d6604d")]:
        vals = adata.obs.loc[adata.obs["broad_condition"] == cond, col]
        ax.hist(np.log1p(vals), bins=60, alpha=0.55, label=cond, color=color, density=True)
    ax.set_xlabel(f"log1p({label})")
    ax.set_ylabel("Density")
    ax.set_title(label)
    ax.legend()

for ax, col, label in zip(
    axes[1],
    ["nCount_RNA", "nFeature_RNA", "total_counts"],
    ["UMI counts", "Genes detected", "Total counts (scanpy)"],
):
    for tech, color in [("CosMx", "#1b7837"), ("Xenium", "#762a83")]:
        vals = adata.obs.loc[adata.obs["tech"] == tech, col]
        ax.hist(np.log1p(vals), bins=60, alpha=0.55, label=tech, color=color, density=True)
    ax.set_xlabel(f"log1p({label})")
    ax.set_ylabel("Density")
    ax.set_title(f"{label} by platform")
    ax.legend()

fig.suptitle("QC metrics — CKD vs Control & platform", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("plots/rnaseq_analysis/01_qc_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 01_qc_metrics.png")

print("\n── 2. UMAP coloured by key variables ────────────────────────────────")
if "X_umap" in adata.obsm:
    vars_to_plot = [
        ("immune_ME",            "Immune microenvironment"),
        ("broad_condition",      "Condition (broad)"),
        ("Condition",            "Condition (detailed)"),
        ("tech",                 "Platform"),
        ("niches_annotation_based", "Spatial niche"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, vars_to_plot):
        cats = adata.obs[col].astype(str)
        uniq = list(cats.unique())
        palette = sns.color_palette("tab20", n_colors=len(uniq))
        color_map = dict(zip(uniq, palette))
        colors = np.array([color_map[v] for v in cats])
        umap = adata.obsm["X_umap"]
        idx = np.random.choice(len(umap), size=min(50000, len(umap)), replace=False)
        ax.scatter(umap[idx, 0], umap[idx, 1],
                   c=colors[idx],
                   s=0.3, alpha=0.4, linewidths=0, rasterized=True)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=color_map[c], markersize=6, label=c)
                   for c in uniq]
        ax.legend(handles=handles, title=title, fontsize=6, title_fontsize=7,
                  loc="upper right", ncol=1,
                  bbox_to_anchor=(1, 1), framealpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig("plots/rnaseq_analysis/02_umap_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 02_umap_overview.png")

print("\n── 3. Cell-type abundance by condition ──────────────────────────────")
abund = adata.obs[ct_cols + ["broad_condition", "Condition", "orig_ident"]].copy()

# Mean abundance per sample
sample_abund = abund.groupby(["orig_ident", "broad_condition"])[ct_cols].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
ctrl = sample_abund[sample_abund["broad_condition"] == "Control"]
ckd  = sample_abund[sample_abund["broad_condition"] == "CKD"]

x = np.arange(len(ct_cols))
w = 0.35
ax.bar(x - w/2, ctrl[ct_cols].mean(), w, yerr=ctrl[ct_cols].sem(),
       label="Control", color="#4393c3", alpha=0.8, capsize=3)
ax.bar(x + w/2, ckd[ct_cols].mean(), w, yerr=ckd[ct_cols].sem(),
       label="CKD", color="#d6604d", alpha=0.8, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(ct_cols, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean cell2location abundance")
ax.set_title("Cell-type abundance: Control vs CKD (mean ± SEM across samples)")
ax.legend()
fig.tight_layout()
fig.savefig("plots/rnaseq_analysis/03_celltype_abundance_condition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 03_celltype_abundance_condition.png")

# Heatmap: samples × cell types
fig, ax = plt.subplots(figsize=(12, max(6, len(sample_abund) * 0.25)))
pivot = sample_abund.set_index("orig_ident")[ct_cols].fillna(0)
cond_order = sample_abund.set_index("orig_ident")["broad_condition"]
row_colors = cond_order.map({"Control": "#4393c3", "CKD": "#d6604d"})
g = sns.clustermap(pivot, row_colors=row_colors, cmap="viridis",
                   standard_scale=1, figsize=(14, max(8, len(pivot) * 0.3)),
                   dendrogram_ratio=(0.1, 0.15), xticklabels=True, yticklabels=True)
g.ax_heatmap.set_xlabel("Cell type")
g.ax_heatmap.set_ylabel("Sample")
g.fig.suptitle("Cell-type abundance per sample (row-normalised)", y=1.01)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#4393c3", label="Control"),
              Patch(facecolor="#d6604d", label="CKD")]
g.ax_row_dendrogram.legend(handles=legend_els, loc="center", fontsize=9)
g.fig.savefig("plots/rnaseq_analysis/04_celltype_heatmap_samples.png",
              dpi=150, bbox_inches="tight")
plt.close()
print("  saved 04_celltype_heatmap_samples.png")

print("\n── 4. Cell-type abundance by Immune ME ──────────────────────────────")
me_abund = adata.obs[ct_cols + ["immune_ME"]].groupby("immune_ME")[ct_cols].mean()
fig, ax = plt.subplots(figsize=(14, 6))
me_abund_norm = me_abund.div(me_abund.sum(axis=1), axis=0)
me_abund_norm.plot(kind="bar", stacked=True, colormap="tab20", ax=ax, width=0.8)
ax.set_title("Cell-type composition by Immune Microenvironment")
ax.set_ylabel("Proportion")
ax.set_xlabel("Immune ME")
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=7, ncol=1)
fig.tight_layout()
fig.savefig("plots/rnaseq_analysis/05_celltype_by_immuneME.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 05_celltype_by_immuneME.png")

print("\n── 5. Differential expression: CKD vs Control ───────────────────────")
# Use raw counts layer
adata_deg = adata.copy()
adata_deg.X = adata_deg.layers["counts"]

# Filter to Control vs DKD (largest CKD group) for power
mask = adata.obs["Condition"].isin(["Control", "DKD"])
adata_sub = adata_deg[mask].copy()
print(f"  DEG subset: {adata_sub.shape[0]:,} cells")

# Normalise
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# Filter to expressed genes
sc.pp.filter_genes(adata_sub, min_cells=50)
print(f"  Genes after filter: {adata_sub.shape[1]}")

# Rank genes
sc.tl.rank_genes_groups(adata_sub, groupby="Condition", groups=["DKD"],
                         reference="Control", method="wilcoxon",
                         key_added="deg_DKD_vs_Control")

deg_df = sc.get.rank_genes_groups_df(adata_sub, group="DKD",
                                      key="deg_DKD_vs_Control",
                                      pval_cutoff=0.05)
deg_df.to_csv("plots/rnaseq_analysis/DEG_DKD_vs_Control.csv", index=False)
print(f"  Significant DEGs (p<0.05): {len(deg_df):,}")
print(f"  Top upregulated:\n{deg_df.nlargest(10, 'scores')[['names','scores','logfoldchanges','pvals_adj']].to_string()}")
print(f"\n  Top downregulated:\n{deg_df.nsmallest(10, 'scores')[['names','scores','logfoldchanges','pvals_adj']].to_string()}")

# Volcano plot
all_deg = sc.get.rank_genes_groups_df(adata_sub, group="DKD",
                                       key="deg_DKD_vs_Control")
all_deg["-log10_pval"] = -np.log10(all_deg["pvals_adj"].clip(1e-300))
all_deg["sig"] = (all_deg["pvals_adj"] < 0.05) & (all_deg["logfoldchanges"].abs() > 0.5)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(all_deg.loc[~all_deg["sig"], "logfoldchanges"],
           all_deg.loc[~all_deg["sig"], "-log10_pval"],
           s=2, alpha=0.3, color="grey", label="ns")
up = all_deg[all_deg["sig"] & (all_deg["logfoldchanges"] > 0)]
dn = all_deg[all_deg["sig"] & (all_deg["logfoldchanges"] < 0)]
ax.scatter(up["logfoldchanges"], up["-log10_pval"], s=4, alpha=0.7,
           color="#d6604d", label=f"Up in DKD (n={len(up)})")
ax.scatter(dn["logfoldchanges"], dn["-log10_pval"], s=4, alpha=0.7,
           color="#4393c3", label=f"Down in DKD (n={len(dn)})")
for _, row in all_deg[all_deg["sig"]].nlargest(20, "-log10_pval").iterrows():
    ax.text(row["logfoldchanges"], row["-log10_pval"], row["names"],
            fontsize=6, ha="center")
ax.axvline(0.5, ls="--", lw=0.8, color="black")
ax.axvline(-0.5, ls="--", lw=0.8, color="black")
ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="black")
ax.set_xlabel("Log2 fold change (DKD vs Control)")
ax.set_ylabel("-log10(adj. p-value)")
ax.set_title("Volcano: DKD vs Control (immune cells)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("plots/rnaseq_analysis/06_volcano_DKD_vs_Control.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved 06_volcano_DKD_vs_Control.png")

# Top DEG dot plot
top_genes = (
    deg_df.nlargest(15, "scores")["names"].tolist() +
    deg_df.nsmallest(15, "scores")["names"].tolist()
)
top_genes = [g for g in top_genes if g in adata_sub.var_names]

sc.pl.dotplot(adata_sub, var_names=top_genes, groupby="Condition",
              save="_DEG_DKD_vs_Control.png", show=False,
              title="Top 15 up/down DEGs: DKD vs Control")
print("  saved dotplot_DEG_DKD_vs_Control.png")

print("\n── 6. DEG by Immune ME ───────────────────────────────────────────────")
sc.tl.rank_genes_groups(adata_sub, groupby="immune_ME", method="wilcoxon",
                         key_added="deg_by_ME")

# Save top markers per ME
me_markers = {}
for me in adata_sub.obs["immune_ME"].cat.categories:
    try:
        df = sc.get.rank_genes_groups_df(adata_sub, group=me, key="deg_by_ME", pval_cutoff=0.05)
        me_markers[me] = df
    except Exception:
        pass

all_me_df = pd.concat(me_markers, names=["immune_ME"]).reset_index(level=0)
all_me_df.to_csv("plots/rnaseq_analysis/DEG_by_immuneME.csv", index=False)
print(f"  saved DEG_by_immuneME.csv ({len(all_me_df):,} sig. markers)")

# Dot plot top 5 per ME
top_per_me = []
for me, df in me_markers.items():
    top_per_me.extend(df.nlargest(5, "scores")["names"].tolist())
top_per_me = list(dict.fromkeys(top_per_me))  # deduplicate, preserve order
top_per_me = [g for g in top_per_me if g in adata_sub.var_names][:60]

sc.pl.dotplot(adata_sub, var_names=top_per_me, groupby="immune_ME",
              save="_top5_per_ME.png", show=False,
              title="Top 5 markers per Immune ME")
print("  saved dotplot_top5_per_ME.png")

print("\n── 7. Summary statistics ────────────────────────────────────────────")
summary = pd.DataFrame({
    "n_cells":  adata.obs.groupby("Condition").size(),
    "median_UMI": adata.obs.groupby("Condition")["nCount_RNA"].median(),
    "median_genes": adata.obs.groupby("Condition")["nFeature_RNA"].median(),
})
summary.to_csv("plots/rnaseq_analysis/summary_stats_by_condition.csv")
print(summary.to_string())

print("\nDone. All outputs in plots/rnaseq_analysis/")
