"""
Cell-Cell Communication Analysis by Cell2Location Subgroups
============================================================
Uses squidpy ligrec (permutation test) on cell type subgroups
defined by cell2location immune microenvironments (immune_ME)
and spatial niches (niches_annotation_based).

Analyses:
  1. Communication globale (toutes cellules)
  2. Par immune_ME (8 microenvironements)
  3. CKD vs Control par microenvironement
  4. Interactions dominantes par sous-groupe cell2location
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.sparse import issparse

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output"
H5AD = os.path.join(BASE, "adata_spatial_final.h5ad")
OUT  = os.path.join(BASE, "cell_communication")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("[1/7] Loading adata_spatial_final.h5ad ...")
adata = sc.read_h5ad(H5AD)
print(f"  Shape: {adata.shape}")

# Use log-normalized X (values 0-178 already normalized)
# Store raw counts separately
adata.layers["lognorm"] = adata.X.copy()

# Cell2location cell type columns (immune deconvolution)
C2L_TYPES = [
    "B_Cells", "B_Memory", "B_Naive", "Basophile",
    "CD14_Mono", "CD16_Mono", "CD4_Activated", "CD4_Trm",
    "CD4_signaling", "CD8_MAIT", "CD8_central_memory",
    "CD8_cytotoxic/effector_memory", "FOLR2+_resident", "FOLR2_CKD",
    "NK/T_cells", "NK_cytotoxic", "Neutro_FPR2+", "Plasma_cells",
    "TREM2+_macro", "cDC", "pDC"
]

BROAD_TYPES = "immune_cell_annotation_combined"
NICHE_COL   = "immune_ME"
SPATIAL_NICHE = "niches_annotation_based"
CONDITION   = "Condition"

# Dominant cell2location type per cell (argmax of abundances)
c2l_matrix = adata.obs[C2L_TYPES].values
adata.obs["dominant_c2l_type"] = pd.Categorical(
    [C2L_TYPES[i] for i in np.argmax(c2l_matrix, axis=1)]
)
print("  dominant_c2l_type computed")

# ─────────────────────────────────────────────
# Helper: run ligrec safely
# ─────────────────────────────────────────────
def run_ligrec(adata_sub, cluster_key, n_perms=500, seed=42, label=""):
    """Runs sq.gr.ligrec and returns the result (uns key = 'ligrec')."""
    # Need raw counts for squidpy ligrec
    adata_sub = adata_sub.copy()
    if "counts" in adata_sub.layers:
        adata_sub.X = adata_sub.layers["counts"].copy()

    # Remove cell types with < 10 cells
    ct_counts = adata_sub.obs[cluster_key].value_counts()
    keep_ct = ct_counts[ct_counts >= 10].index.tolist()
    adata_sub = adata_sub[adata_sub.obs[cluster_key].isin(keep_ct)].copy()
    adata_sub.obs[cluster_key] = pd.Categorical(adata_sub.obs[cluster_key])

    if len(keep_ct) < 2:
        print(f"  [SKIP] {label}: fewer than 2 cell types with >=10 cells")
        return None

    print(f"  Running ligrec: {label} | {adata_sub.shape[0]} cells | {len(keep_ct)} types")
    try:
        sq.gr.ligrec(
            adata_sub,
            n_perms=n_perms,
            cluster_key=cluster_key,
            use_raw=False,
            copy=False,
            seed=seed,
            show_progress_bar=False,
        )
        return adata_sub
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return None


def save_ligrec_csv(adata_res, cluster_key, label, outdir):
    """Saves means and pvalues to CSV."""
    key = f"{cluster_key}_ligrec"
    if key not in adata_res.uns:
        return
    means  = adata_res.uns[key]["means"]
    pvals  = adata_res.uns[key]["pvalues"]
    means.to_csv(os.path.join(outdir, f"{label}_means.csv"))
    pvals.to_csv(os.path.join(outdir, f"{label}_pvalues.csv"))


def get_significant_interactions(adata_res, cluster_key, pval_thresh=0.05, mean_thresh=0.5):
    """Returns df of significant interactions with mean expression."""
    key = f"{cluster_key}_ligrec"
    if key not in adata_res.uns:
        return pd.DataFrame()
    means  = adata_res.uns[key]["means"]
    pvals  = adata_res.uns[key]["pvalues"]

    records = []
    for (source, target) in pvals.columns:
        sig_mask = (pvals[(source, target)] < pval_thresh) & (means[(source, target)] > mean_thresh)
        for lr_pair in pvals.index[sig_mask]:
            records.append({
                "ligand":  lr_pair[0],
                "receptor": lr_pair[1],
                "source":  source,
                "target":  target,
                "mean":    means.loc[lr_pair, (source, target)],
                "pvalue":  pvals.loc[lr_pair, (source, target)],
            })
    return pd.DataFrame(records).sort_values("mean", ascending=False)


# ─────────────────────────────────────────────
# 1. Global communication — all immune cells
# ─────────────────────────────────────────────
print("\n[2/7] Global ligand-receptor analysis (immune_cell_annotation_combined) ...")
out_global = os.path.join(OUT, "global")
os.makedirs(out_global, exist_ok=True)

adata_global = run_ligrec(adata, BROAD_TYPES, n_perms=500, label="Global")
if adata_global is not None:
    save_ligrec_csv(adata_global, BROAD_TYPES, "global", out_global)
    sig_global = get_significant_interactions(adata_global, BROAD_TYPES)
    sig_global.to_csv(os.path.join(out_global, "global_significant_interactions.csv"), index=False)
    print(f"  Significant interactions: {len(sig_global)}")

    # Dotplot global
    try:
        key = f"{BROAD_TYPES}_ligrec"
        fig = sq.pl.ligrec(
            adata_global, cluster_key=BROAD_TYPES,
            source_groups=None, target_groups=None,
            pvalue_threshold=0.05, means_range=(0.5, np.inf),
            show=False, return_ax=True,
        )
        plt.savefig(os.path.join(out_global, "global_ligrec_dotplot.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [WARN] dotplot global: {e}")

# ─────────────────────────────────────────────
# 2. Per immune_ME microenvironment
# ─────────────────────────────────────────────
print("\n[3/7] Per immune_ME microenvironment ...")
out_me = os.path.join(OUT, "per_immune_ME")
os.makedirs(out_me, exist_ok=True)

immune_mes = [m for m in adata.obs[NICHE_COL].unique() if m != "Unknown"]
me_sig_dfs = {}

for me in immune_mes:
    adata_me = adata[adata.obs[NICHE_COL] == me].copy()
    safe_name = me.replace(" ", "_").replace("/", "-").replace(".", "")
    res = run_ligrec(adata_me, BROAD_TYPES, n_perms=300, label=f"ME={me}")
    if res is not None:
        me_dir = os.path.join(out_me, safe_name)
        os.makedirs(me_dir, exist_ok=True)
        save_ligrec_csv(res, BROAD_TYPES, safe_name, me_dir)
        sig = get_significant_interactions(res, BROAD_TYPES)
        sig["immune_ME"] = me
        sig.to_csv(os.path.join(me_dir, f"{safe_name}_significant.csv"), index=False)
        me_sig_dfs[me] = sig

        try:
            sq.pl.ligrec(
                res, cluster_key=BROAD_TYPES,
                pvalue_threshold=0.05, means_range=(0.5, np.inf),
                show=False, return_ax=True,
                title=me,
            )
            plt.savefig(os.path.join(me_dir, f"{safe_name}_dotplot.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  [WARN] dotplot {me}: {e}")

# Concatenate all ME results
if me_sig_dfs:
    all_me = pd.concat(me_sig_dfs.values(), ignore_index=True)
    all_me.to_csv(os.path.join(out_me, "all_ME_significant_interactions.csv"), index=False)
    print(f"  Total significant interactions across MEs: {len(all_me)}")

# ─────────────────────────────────────────────
# 3. CKD vs Control — per immune_ME
# ─────────────────────────────────────────────
print("\n[4/7] CKD vs Control analysis per immune_ME ...")
out_cond = os.path.join(OUT, "CKD_vs_Control")
os.makedirs(out_cond, exist_ok=True)

# Group CKD conditions
ckd_conditions  = ["DKD", "DM/HTN", "C3GN", "DM", "AA amyloid", "IgA", "MN", "FSGS", "CKD",
                   "DKD+FSGS", "TMA", "HTN"]
ctrl_conditions = ["Control"]

adata.obs["Disease_group"] = pd.Categorical(
    np.where(adata.obs[CONDITION].isin(ctrl_conditions), "Control", "CKD")
)

cond_me_records = []
for me in immune_mes:
    for disease in ["CKD", "Control"]:
        mask = (adata.obs[NICHE_COL] == me) & (adata.obs["Disease_group"] == disease)
        adata_sub = adata[mask].copy()
        safe_name = f"{me.replace(' ','_').replace('/','').replace('.','')}__{disease}"
        res = run_ligrec(adata_sub, BROAD_TYPES, n_perms=200, label=f"{me} | {disease}")
        if res is not None:
            sig = get_significant_interactions(res, BROAD_TYPES)
            sig["immune_ME"] = me
            sig["condition"] = disease
            cond_me_records.append(sig)

if cond_me_records:
    cond_df = pd.concat(cond_me_records, ignore_index=True)
    cond_df.to_csv(os.path.join(out_cond, "CKD_vs_Control_all_ME_interactions.csv"), index=False)

    # Differential: interactions enriched in CKD vs Control
    ckd_pairs  = set(zip(cond_df[cond_df.condition=="CKD"]["ligand"],
                         cond_df[cond_df.condition=="CKD"]["receptor"],
                         cond_df[cond_df.condition=="CKD"]["source"],
                         cond_df[cond_df.condition=="CKD"]["target"],
                         cond_df[cond_df.condition=="CKD"]["immune_ME"]))
    ctrl_pairs = set(zip(cond_df[cond_df.condition=="Control"]["ligand"],
                         cond_df[cond_df.condition=="Control"]["receptor"],
                         cond_df[cond_df.condition=="Control"]["source"],
                         cond_df[cond_df.condition=="Control"]["target"],
                         cond_df[cond_df.condition=="Control"]["immune_ME"]))
    ckd_specific  = ckd_pairs - ctrl_pairs
    ctrl_specific = ctrl_pairs - ckd_pairs
    print(f"  CKD-specific interactions: {len(ckd_specific)}")
    print(f"  Control-specific interactions: {len(ctrl_specific)}")

    pd.DataFrame(ckd_specific, columns=["ligand","receptor","source","target","immune_ME"]
                 ).to_csv(os.path.join(out_cond, "CKD_specific_interactions.csv"), index=False)
    pd.DataFrame(ctrl_specific, columns=["ligand","receptor","source","target","immune_ME"]
                 ).to_csv(os.path.join(out_cond, "Control_specific_interactions.csv"), index=False)

# ─────────────────────────────────────────────
# 4. Dominant cell2location type — communication
# ─────────────────────────────────────────────
print("\n[5/7] Communication by dominant cell2location type ...")
out_c2l = os.path.join(OUT, "dominant_c2l_type")
os.makedirs(out_c2l, exist_ok=True)

res_c2l = run_ligrec(adata, "dominant_c2l_type", n_perms=300, label="dominant_c2l")
if res_c2l is not None:
    save_ligrec_csv(res_c2l, "dominant_c2l_type", "dominant_c2l", out_c2l)
    sig_c2l = get_significant_interactions(res_c2l, "dominant_c2l_type")
    sig_c2l.to_csv(os.path.join(out_c2l, "dominant_c2l_significant.csv"), index=False)
    print(f"  Significant interactions: {len(sig_c2l)}")

# ─────────────────────────────────────────────
# 5. Summary heatmap: n° interactions per ME pair
# ─────────────────────────────────────────────
print("\n[6/7] Summary figures ...")

if me_sig_dfs and len(all_me) > 0:
    # Number of significant interactions per immune_ME
    n_int = all_me.groupby("immune_ME").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    n_int.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Immune Microenvironment", fontsize=11)
    ax.set_ylabel("Number of significant LR pairs", fontsize=11)
    ax.set_title("Significant Ligand-Receptor Interactions per Immune ME", fontsize=12, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "summary_n_interactions_per_ME.png"), dpi=150)
    plt.close()

    # Top 20 LR pairs across all MEs
    top_lr = (all_me.groupby(["ligand","receptor"])["mean"]
              .max().sort_values(ascending=False).head(20))
    top_lr_df = top_lr.reset_index()
    top_lr_df["lr_pair"] = top_lr_df["ligand"] + " → " + top_lr_df["receptor"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_lr_df["lr_pair"][::-1], top_lr_df["mean"][::-1], color="coral", edgecolor="white")
    ax.set_xlabel("Max mean expression", fontsize=11)
    ax.set_title("Top 20 Ligand-Receptor Pairs Across All Immune MEs", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "summary_top20_LR_pairs.png"), dpi=150)
    plt.close()

    # Heatmap: source → target interaction count per ME
    all_me["pair"] = all_me["source"] + " → " + all_me["target"]
    heat_data = (all_me.groupby(["immune_ME", "pair"]).size()
                 .unstack(fill_value=0))
    if heat_data.shape[0] > 1:
        fig, ax = plt.subplots(figsize=(max(12, heat_data.shape[1]*0.4),
                                         max(5, heat_data.shape[0]*0.5)))
        sns.heatmap(heat_data, cmap="YlOrRd", ax=ax, linewidths=0.3,
                    cbar_kws={"label": "# LR interactions"})
        ax.set_title("LR Interaction Count: Cell-Type Pair × Immune ME", fontsize=12, fontweight="bold")
        ax.set_xlabel("Source → Target", fontsize=10)
        ax.set_ylabel("Immune Microenvironment", fontsize=10)
        plt.xticks(rotation=60, ha="right", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "summary_heatmap_pair_x_ME.png"), dpi=150, bbox_inches="tight")
        plt.close()

# CKD vs Control bar
if cond_me_records and len(cond_df) > 0:
    n_cond = cond_df.groupby(["immune_ME","condition"]).size().reset_index(name="n")
    pivot   = n_cond.pivot(index="immune_ME", columns="condition", values="n").fillna(0)
    pivot.plot(kind="bar", figsize=(10, 5), color=["#E55050","#5090E5"], edgecolor="white")
    plt.title("LR Interactions per Immune ME: CKD vs Control", fontsize=12, fontweight="bold")
    plt.xlabel("Immune Microenvironment"); plt.ylabel("# Significant LR pairs")
    plt.xticks(rotation=35, ha="right"); plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "summary_CKD_vs_Control_per_ME.png"), dpi=150)
    plt.close()

# ─────────────────────────────────────────────
# 6. Cell2location abundance correlation with interactions
# ─────────────────────────────────────────────
print("\n[7/7] Cell2location abundance per immune_ME heatmap ...")

c2l_by_me = adata.obs.groupby(NICHE_COL)[C2L_TYPES].mean()
c2l_by_me = c2l_by_me.loc[c2l_by_me.index != "Unknown"]

# Normalize by row (relative abundance within ME)
c2l_norm = c2l_by_me.div(c2l_by_me.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(c2l_norm, cmap="Blues", ax=ax, linewidths=0.3,
            cbar_kws={"label": "Relative abundance"})
ax.set_title("Cell2Location Cell Type Abundance per Immune Microenvironment", fontsize=12, fontweight="bold")
ax.set_xlabel("Cell2Location Cell Type", fontsize=10)
ax.set_ylabel("Immune ME", fontsize=10)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "c2l_abundance_per_immuneME.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Analysis complete. Results saved to:")
print(f"  {OUT}")
print(f"{'='*55}")
print("\nOutput structure:")
print("  global/                   — Communication globale")
print("  per_immune_ME/            — Par microenvironnement (8 ME)")
print("  CKD_vs_Control/           — CKD vs Control par ME")
print("  dominant_c2l_type/        — Par type cell2location dominant")
print("  summary_*.png             — Figures récapitulatives")
print("  c2l_abundance_per_immuneME.png")
