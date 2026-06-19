"""
Trajectoire pseudotemps — FOLR2 macrophages
De l'état résident homéostatique vers l'état pathologique CKD
Identification des gènes conducteurs de la pathologie
"""
import anndata as ad, pandas as pd, numpy as np, scanpy as sc
from scipy.sparse import issparse
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

PATH = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
DE   = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\DE_FOLR2CKD_CKD_vs_Control.csv"
OUT  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication"

# ── 1. Load & subset macrophages
print("[1/6] Loading macrophage subset ...")
adata = ad.read_h5ad(PATH)
adata.obs["annot_final2"]    = adata.obs["annot_final2"].astype(str)
adata.obs["condition_group"] = adata.obs["diseasetype"].map(
    lambda d: "CKD" if d in ("DKD","HKD","CKD") else ("Control" if d=="Control" else "Other")
).astype(str)

macro_types = ["FOLR2+_resident", "FOLR2_CKD", "TREM2+_macro", "CD14_Mono", "CD16_Mono"]
sub = adata[adata.obs["annot_final2"].isin(macro_types) &
            adata.obs["condition_group"].isin(["CKD","Control"])].copy()

sub.X = sub.layers["scvi_normalized"].copy()

# Labels combinés pour la trajectoire
sub.obs["cell_state"] = (sub.obs["annot_final2"].astype(str) + " | " +
                          sub.obs["condition_group"].astype(str))

# Palette
palette_state = {
    "FOLR2+_resident | Control": "#2ECC71",
    "FOLR2+_resident | CKD":    "#27AE60",
    "FOLR2_CKD | Control":      "#F39C12",
    "FOLR2_CKD | CKD":          "#E74C3C",
    "TREM2+_macro | Control":   "#9B59B6",
    "TREM2+_macro | CKD":       "#6C3483",
    "CD14_Mono | Control":      "#85C1E9",
    "CD14_Mono | CKD":          "#1A5276",
    "CD16_Mono | Control":      "#A9CCE3",
    "CD16_Mono | CKD":          "#21618C",
}
palette_type = {
    "FOLR2+_resident": "#27AE60",
    "FOLR2_CKD":       "#E74C3C",
    "TREM2+_macro":    "#8E44AD",
    "CD14_Mono":       "#2980B9",
    "CD16_Mono":       "#5DADE2",
}
palette_cond = {"Control": "#2980B9", "CKD": "#C0392B"}

# ── 2. Recompute PCA + neighbors on scVI embedding
print("[2/6] Computing neighbors + UMAP + DPT ...")
sc.pp.neighbors(sub, use_rep="X_scVI", n_neighbors=20, n_pcs=10)
sc.tl.umap(sub, min_dist=0.4, spread=1.2)

# PAGA for trajectory
sc.tl.paga(sub, groups="annot_final2")
sc.pl.paga(sub, show=False)   # compute positions

# Diffusion map for pseudotime
sc.tl.diffmap(sub, n_comps=15)

# Root: FOLR2+_resident Control cells (most homeostatic)
root_mask = ((sub.obs["annot_final2"] == "FOLR2+_resident") &
             (sub.obs["condition_group"] == "Control"))
root_idx  = np.where(root_mask)[0][0]
sub.uns["iroot"] = root_idx
sc.tl.dpt(sub, n_dcs=10)

print(f"  Pseudotime range: {sub.obs['dpt_pseudotime'].min():.3f} – {sub.obs['dpt_pseudotime'].max():.3f}")

# ── 3. Genes correlated with pseudotime
print("[3/6] Computing gene-pseudotime correlations ...")
de = pd.read_csv(DE)
# Focus on significant DE genes
sig_genes = de[de["status"].isin(["UP_CKD","DOWN_CKD"])]["names"].tolist()
sig_genes = [g for g in sig_genes if g in sub.var.index]

pt = sub.obs["dpt_pseudotime"].values

# Sample cells for speed (keep all FOLR2_CKD)
np.random.seed(42)

corr_records = []
for g in sig_genes:
    idx = list(sub.var.index).index(g)
    expr = sub.X[:, idx]
    if issparse(expr): expr = expr.toarray().flatten()
    expr = expr.astype(float)
    if expr.std() < 0.01: continue
    r, p = spearmanr(pt, expr)
    corr_records.append({"gene": g, "spearman_r": r, "pvalue": p,
                          "status": de.set_index("names")["status"].get(g, "NS"),
                          "logFC":  de.set_index("names")["logFC"].get(g, 0)})

corr_df = pd.DataFrame(corr_records).sort_values("spearman_r", ascending=False)
corr_df.to_csv(os.path.join(OUT, "pseudotime_gene_correlations.csv"), index=False)

# Top drivers
top_positive = corr_df[corr_df.spearman_r > 0].head(15)   # increase with pathology
top_negative = corr_df[corr_df.spearman_r < 0].tail(15)   # decrease with pathology
driver_genes  = list(top_positive.gene[:8]) + list(top_negative.gene[-8:])

print("  Top drivers (+):", list(top_positive.gene[:8]))
print("  Top drivers (-):", list(top_negative.gene[-8:]))

# Key genes to always show
key_show = ["CCL8","SPP1","APOE","MRC1","FOLR2","LYVE1","STAB1","MAF",
            "C1QA","S100A2","SAA1","FABP4","APOC1","CHI3L1","CCL2","TREM2"]
key_show = [g for g in key_show if g in sub.var.index]

# ── 4. Main figure
print("[4/6] Figure 1 — Trajectory overview ...")

fig = plt.figure(figsize=(22, 18))
gs_main = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

umap1 = sub.obsm["X_umap"][:, 0]
umap2 = sub.obsm["X_umap"][:, 1]
pt_vals = sub.obs["dpt_pseudotime"].values

# Panel A — UMAP par type cellulaire
ax_a = fig.add_subplot(gs_main[0, 0])
for ct, col in palette_type.items():
    mask = sub.obs["annot_final2"] == ct
    ax_a.scatter(umap1[mask], umap2[mask], s=4, color=col, alpha=0.5,
                 label=ct, rasterized=True)
ax_a.set_title("Types cellulaires", fontsize=10, fontweight="bold")
ax_a.legend(fontsize=6.5, markerscale=2, loc="best", framealpha=0.7)
ax_a.set_xlabel("UMAP1", fontsize=8); ax_a.set_ylabel("UMAP2", fontsize=8)
ax_a.tick_params(labelsize=7)
ax_a.spines[["top","right"]].set_visible(False)

# Panel B — UMAP par condition
ax_b = fig.add_subplot(gs_main[0, 1])
for cond, col in palette_cond.items():
    mask = sub.obs["condition_group"] == cond
    ax_b.scatter(umap1[mask], umap2[mask], s=4, color=col, alpha=0.45,
                 label=cond, rasterized=True)
ax_b.set_title("Condition (CKD vs Control)", fontsize=10, fontweight="bold")
ax_b.legend(fontsize=8, markerscale=2, framealpha=0.8)
ax_b.set_xlabel("UMAP1", fontsize=8); ax_b.set_ylabel("UMAP2", fontsize=8)
ax_b.tick_params(labelsize=7)
ax_b.spines[["top","right"]].set_visible(False)

# Panel C — UMAP pseudotemps
ax_c = fig.add_subplot(gs_main[0, 2])
sc_c = ax_c.scatter(umap1, umap2, c=pt_vals, cmap="viridis", s=4,
                     alpha=0.6, rasterized=True)
plt.colorbar(sc_c, ax=ax_c, label="Pseudotime", shrink=0.8)
ax_c.set_title("Pseudotemps (DPT)\nRacine: FOLR2+_resident Control", fontsize=10, fontweight="bold")
ax_c.set_xlabel("UMAP1", fontsize=8); ax_c.set_ylabel("UMAP2", fontsize=8)
ax_c.tick_params(labelsize=7)
ax_c.spines[["top","right"]].set_visible(False)

# Panel D — PAGA connectivity
ax_d = fig.add_subplot(gs_main[0, 3])
sc.pl.paga(sub, ax=ax_d, show=False, frameon=False,
           node_size_scale=2, edge_width_scale=1.5,
           title="Connectivité PAGA\n(trajectoire entre types)", fontsize=8,
           color="annot_final2")
ax_d.set_title("Connectivité PAGA", fontsize=10, fontweight="bold")

# Panel E — Pseudotime distribution par état
ax_e = fig.add_subplot(gs_main[1, 0:2])
states_order = ["FOLR2+_resident | Control","FOLR2+_resident | CKD",
                "FOLR2_CKD | Control","FOLR2_CKD | CKD",
                "TREM2+_macro | Control","TREM2+_macro | CKD",
                "CD14_Mono | Control","CD14_Mono | CKD"]
states_order = [s for s in states_order if s in sub.obs["cell_state"].unique()]

pt_by_state = [sub.obs[sub.obs["cell_state"]==s]["dpt_pseudotime"].values for s in states_order]
vp = ax_e.violinplot(pt_by_state, positions=range(len(states_order)),
                      showmedians=True, showextrema=False, widths=0.65)
for pc, st in zip(vp["bodies"], states_order):
    pc.set_facecolor(palette_state.get(st, "#AAAAAA"))
    pc.set_alpha(0.75)
    pc.set_edgecolor("white")
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
ax_e.set_xticks(range(len(states_order)))
ax_e.set_xticklabels([s.replace(" | ","\n") for s in states_order],
                      fontsize=7.5, rotation=30, ha="right")
ax_e.set_ylabel("Pseudotime (DPT)", fontsize=9)
ax_e.set_title("Distribution du pseudotemps par état cellulaire\n"
               "→ progression de l'état homéostatique vers l'état pathologique", fontsize=10, fontweight="bold")
ax_e.spines[["top","right"]].set_visible(False)

# Flèche de progression
ax_e.annotate("", xy=(len(states_order)-0.5, ax_e.get_ylim()[1]*0.92),
               xytext=(0.5, ax_e.get_ylim()[1]*0.92),
               arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2))
ax_e.text(len(states_order)/2, ax_e.get_ylim()[1]*0.95,
          "Progression pathologique →", ha="center", fontsize=9, color="#E74C3C", fontweight="bold")

# Panel F — Top genes corrélés au pseudotemps
ax_f = fig.add_subplot(gs_main[1, 2:4])
top15_pos = corr_df.head(15)
top15_neg = corr_df.tail(15).iloc[::-1]
combined  = pd.concat([top15_pos, top15_neg])
colors_r  = ["#C0392B" if r > 0 else "#1A5276" for r in combined.spearman_r]
y_pos     = np.arange(len(combined))
ax_f.barh(y_pos, combined.spearman_r, color=colors_r, edgecolor="white", height=0.7, alpha=0.8)
ax_f.axvline(0, color="black", lw=0.8)
ax_f.axvline( 0.1, color="#AAAAAA", ls="--", lw=0.6)
ax_f.axvline(-0.1, color="#AAAAAA", ls="--", lw=0.6)
ax_f.set_yticks(y_pos)
ax_f.set_yticklabels(combined.gene, fontsize=8)
for i, (_, row) in enumerate(combined.iterrows()):
    if row.gene in key_show:
        ax_f.get_yticklabels()[i].set_fontweight("bold")
        ax_f.get_yticklabels()[i].set_color("#C0392B" if row.spearman_r > 0 else "#1A5276")
ax_f.set_xlabel("Corrélation de Spearman (gène ~ pseudotemps)", fontsize=9)
ax_f.set_title("Gènes conducteurs de la trajectoire pathologique\n"
               "(corrélation avec pseudotemps DPT)", fontsize=10, fontweight="bold")
ax_f.invert_yaxis()
ax_f.spines[["top","right"]].set_visible(False)
ax_f.text(0.98, 0.98, "▲ Augmente avec la pathologie", transform=ax_f.transAxes,
          ha="right", va="top", fontsize=8, color="#C0392B", fontweight="bold")
ax_f.text(0.02, 0.02, "▼ Diminue avec la pathologie", transform=ax_f.transAxes,
          ha="left", va="bottom", fontsize=8, color="#1A5276", fontweight="bold")

# ── 5. Gene expression along pseudotime (bottom row)
print("[5/6] Figure — Gene profiles along pseudotime ...")

def smooth_along_pt(pt_arr, expr_arr, n_bins=40):
    bins = np.linspace(pt_arr.min(), pt_arr.max(), n_bins + 1)
    bin_idx = np.digitize(pt_arr, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
    means   = [expr_arr[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan for i in range(n_bins)]
    return np.array(centers), np.array(means)

# Focus: FOLR2_CKD + FOLR2+_resident only for pseudotime profiles
f2_mask = sub.obs["annot_final2"].isin(["FOLR2_CKD","FOLR2+_resident"])
sub_f2  = sub[f2_mask].copy()
pt_f2   = sub_f2.obs["dpt_pseudotime"].values

# Genes to plot: top drivers + key biological genes
genes_profile = []
for g in (list(top_positive.gene[:4]) + list(top_negative.gene[-4:]) + key_show):
    if g in sub_f2.var.index and g not in genes_profile:
        genes_profile.append(g)
genes_profile = genes_profile[:16]

n_cols = 4
n_rows_p = (len(genes_profile) + n_cols - 1) // n_cols
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    n_rows_p, n_cols, subplot_spec=gs_main[2, :], hspace=0.6, wspace=0.4)

corr_idx = corr_df.set_index("gene")

for k, gene in enumerate(genes_profile):
    ax_g = fig.add_subplot(gs_bottom[k // n_cols, k % n_cols])
    gene_idx = list(sub_f2.var.index).index(gene)
    expr_g   = sub_f2.X[:, gene_idx]
    if issparse(expr_g): expr_g = expr_g.toarray().flatten()
    expr_g = expr_g.astype(float)

    # Scatter by cell type
    for ct, col in [("FOLR2+_resident","#27AE60"),("FOLR2_CKD","#E74C3C")]:
        mask_ct = sub_f2.obs["annot_final2"] == ct
        ax_g.scatter(pt_f2[mask_ct], expr_g[mask_ct], s=3, color=col, alpha=0.25,
                     rasterized=True)

    # Smoothed line
    cx, cy = smooth_along_pt(pt_f2, expr_g, n_bins=35)
    valid = ~np.isnan(cy)
    ax_g.plot(cx[valid], cy[valid], color="black", lw=2, zorder=5)

    # Correlation info
    if gene in corr_idx.index:
        r_val = corr_idx.loc[gene, "spearman_r"]
        direction = "▲" if r_val > 0 else "▼"
        col_title = "#C0392B" if r_val > 0 else "#1A5276"
    else:
        r_val, direction, col_title = 0, "—", "gray"

    ax_g.set_title(f"{gene}  {direction}\nr={r_val:.2f}", fontsize=8.5,
                   fontweight="bold", color=col_title, pad=2)
    ax_g.set_xlabel("Pseudotime", fontsize=7)
    ax_g.set_ylabel("Expression", fontsize=7)
    ax_g.tick_params(labelsize=6.5)
    ax_g.spines[["top","right"]].set_visible(False)

# Legend for gene profiles
leg_handles = [
    mpatches.Patch(color="#27AE60", label="FOLR2+_resident", alpha=0.7),
    mpatches.Patch(color="#E74C3C", label="FOLR2_CKD",       alpha=0.7),
    plt.Line2D([0],[0], color="black", lw=2, label="Tendance lissée"),
]
fig.legend(handles=leg_handles, loc="lower right", bbox_to_anchor=(0.99, 0.01),
           fontsize=8.5, framealpha=0.9, title="Type cellulaire", ncol=3)

fig.suptitle(
    "Trajectoire pseudotemps FOLR2 macrophages — Du phénotype résident vers le phénotype pathologique CKD\n"
    "Racine: FOLR2+_resident (Control)  →  Terminal: FOLR2_CKD (CKD)  |  Diffusion Pseudotime (DPT)",
    fontsize=13, fontweight="bold", y=1.005
)

out1 = os.path.join(OUT, "trajectory_FOLR2CKD_pseudotime.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Fig 1 saved: {out1}")

# ── 6. Summary: driver gene ranking
print("[6/6] Driver gene summary ...")

# Merge with DE results
driver_summary = corr_df.merge(
    de[["names","logFC","pvals_adj","status"]],
    left_on="gene", right_on="names", how="left"
).drop(columns="names")

driver_summary.to_csv(os.path.join(OUT, "pseudotime_driver_genes.csv"), index=False)

print(f"\n{'='*60}")
print("TOP 15 gènes CONDUCTEURS de la pathologie (corrélés au pseudotemps)")
print(f"{'='*60}")
cols_show = [c for c in ["gene","spearman_r","logFC","logfoldchanges","pvals_adj","status"] if c in driver_summary.columns]
print("\nAugmentent avec la progression pathologique (-> CKD):")
print(driver_summary[driver_summary.spearman_r > 0].head(15)[cols_show].to_string(index=False))
print("\nDiminuent avec la progression pathologique (perte homeostasie):")
print(driver_summary[driver_summary.spearman_r < 0].tail(15).iloc[::-1][cols_show].to_string(index=False))

print("\nOutputs:")
print("  trajectory_FOLR2CKD_pseudotime.png")
print("  pseudotime_driver_genes.csv")
print("  pseudotime_gene_correlations.csv")
