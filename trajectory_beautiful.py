"""
Beautiful trajectory plot — FOLR2 macrophages
Pseudotemps + gènes clés homeostasie → pathologie CKD
"""
import anndata as ad, pandas as pd, numpy as np, scanpy as sc
from scipy.sparse import issparse
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

PATH = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
OUT  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication"

# ══════════════════════════════════════════════
# 1. Data preparation
# ══════════════════════════════════════════════
print("[1/4] Preparing data ...")
adata = ad.read_h5ad(PATH)
adata.obs["annot_final2"]    = adata.obs["annot_final2"].astype(str)
adata.obs["condition_group"] = adata.obs["diseasetype"].map(
    lambda d: "CKD" if d in ("DKD","HKD","CKD") else ("Control" if d=="Control" else "Other")
).astype(str)

macro_types = ["FOLR2+_resident", "FOLR2_CKD", "TREM2+_macro", "CD14_Mono", "CD16_Mono"]
sub = adata[adata.obs["annot_final2"].isin(macro_types) &
            adata.obs["condition_group"].isin(["CKD","Control"])].copy()
sub.X = sub.layers["scvi_normalized"].copy()

# Trajectory computation
sc.pp.neighbors(sub, use_rep="X_scVI", n_neighbors=20, n_pcs=10)
sc.tl.umap(sub, min_dist=0.35, spread=1.1, random_state=42)
sc.tl.paga(sub, groups="annot_final2")
sc.tl.diffmap(sub, n_comps=15)
root_idx = np.where((sub.obs["annot_final2"]=="FOLR2+_resident") &
                    (sub.obs["condition_group"]=="Control"))[0][0]
sub.uns["iroot"] = root_idx
sc.tl.dpt(sub, n_dcs=10)

umap1 = sub.obsm["X_umap"][:, 0]
umap2 = sub.obsm["X_umap"][:, 1]
pt    = sub.obs["dpt_pseudotime"].values

# Focus subset: FOLR2 only for profile curves
f2_mask = sub.obs["annot_final2"].isin(["FOLR2+_resident","FOLR2_CKD"])
sub_f2  = sub[f2_mask]
pt_f2   = sub_f2.obs["dpt_pseudotime"].values

# ── Palettes
pal_type = {
    "FOLR2+_resident": "#2ECC71",
    "FOLR2_CKD":       "#E74C3C",
    "TREM2+_macro":    "#8E44AD",
    "CD14_Mono":       "#2980B9",
    "CD16_Mono":       "#85C1E9",
}
pal_cond = {"Control": "#3498DB", "CKD": "#E74C3C"}

# ── Key genes
genes_up   = ["SPP1","APOE","SERPINH1","CHI3L1","S100A2","SAA1","MZB1","TPM2"]
genes_down = ["MRC1","STAB1","C1QA","MAF","LYVE1","FOLR2","CCL8","CCR1"]
key_genes  = genes_up + genes_down
key_genes  = [g for g in key_genes if g in sub.var.index]

# ── Get expression helper
def get_expr(adata_sub, gene):
    idx = list(adata_sub.var.index).index(gene)
    x = adata_sub.X[:, idx]
    if issparse(x): x = x.toarray().flatten()
    return x.astype(float)

# ── Smooth profile helper
def smooth_profile(pt_arr, expr_arr, n_bins=50, sigma=2.5):
    order = np.argsort(pt_arr)
    pt_s, ex_s = pt_arr[order], expr_arr[order]
    bins = np.linspace(pt_s.min(), pt_s.max(), n_bins+1)
    cx, cy, cn = [], [], []
    for i in range(n_bins):
        mask = (pt_s >= bins[i]) & (pt_s < bins[i+1])
        if mask.sum() > 0:
            cx.append((bins[i]+bins[i+1])/2)
            cy.append(ex_s[mask].mean())
            cn.append(mask.sum())
    cx, cy = np.array(cx), np.array(cy)
    cy_sm = gaussian_filter1d(cy, sigma=sigma)
    return cx, cy_sm

# ── Pseudotime colormap (green → yellow → red)
pt_cmap = LinearSegmentedColormap.from_list(
    "pseudotime", ["#2ECC71","#F1C40F","#E74C3C"], N=256)

# ══════════════════════════════════════════════
# 2. Figure layout
# ══════════════════════════════════════════════
print("[2/4] Building figure ...")

fig = plt.figure(figsize=(24, 22), facecolor="white")
fig.patch.set_facecolor("white")

# GridSpec: 4 rows
gs = gridspec.GridSpec(4, 5, figure=fig,
                       hspace=0.52, wspace=0.38,
                       top=0.93, bottom=0.04, left=0.05, right=0.97)

# ── Row 0 : UMAP panels (3 + legend strip)
ax_type = fig.add_subplot(gs[0, 0:2])  # UMAP cell type
ax_pt   = fig.add_subplot(gs[0, 2:4])  # UMAP pseudotime
ax_cond = fig.add_subplot(gs[0, 4])    # UMAP condition

def style_umap(ax):
    ax.set_xlabel("UMAP 1", fontsize=9, color="#555")
    ax.set_ylabel("UMAP 2", fontsize=9, color="#555")
    ax.tick_params(labelsize=7, colors="#888")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

# A — cell type UMAP
for ct in macro_types[::-1]:
    m = sub.obs["annot_final2"] == ct
    ax_type.scatter(umap1[m], umap2[m], s=5, color=pal_type[ct],
                    alpha=0.55, label=ct, rasterized=True, zorder=macro_types.index(ct))
style_umap(ax_type)
ax_type.set_title("Types cellulaires macrophages", fontsize=11, fontweight="bold", pad=6)
leg_handles = [mpatches.Patch(color=pal_type[ct], label=ct, alpha=0.85) for ct in macro_types]
ax_type.legend(handles=leg_handles, fontsize=7.5, loc="upper left",
               framealpha=0.85, edgecolor="lightgray", title="Cell type", title_fontsize=8)

# B — pseudotime UMAP
sc_pt = ax_pt.scatter(umap1, umap2, c=pt, cmap=pt_cmap, s=5,
                       alpha=0.65, rasterized=True, vmin=0, vmax=1)
cb = plt.colorbar(sc_pt, ax=ax_pt, shrink=0.75, pad=0.02,
                  label="Pseudotemps (DPT)")
cb.ax.tick_params(labelsize=8)
cb.set_label("Pseudotemps (DPT)", fontsize=9)

# Arrow showing trajectory direction
# Compute mean UMAP per cell type for arrow
ct_centers = {ct: (umap1[sub.obs["annot_final2"]==ct].mean(),
                   umap2[sub.obs["annot_final2"]==ct].mean())
              for ct in macro_types}

style_umap(ax_pt)
ax_pt.set_title("Pseudotemps — racine: FOLR2+_resident (Control)", fontsize=11, fontweight="bold", pad=6)

# Annotate start and end
start_x, start_y = ct_centers["FOLR2+_resident"]
end_x,   end_y   = ct_centers["FOLR2_CKD"]
ax_pt.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
               arrowprops=dict(arrowstyle="-|>", color="black", lw=2,
                               mutation_scale=18, connectionstyle="arc3,rad=0.2"))
ax_pt.text(start_x-0.3, start_y+0.2, "Debut\n(resident)", fontsize=8,
           color="#27AE60", fontweight="bold", ha="center")
ax_pt.text(end_x+0.3,   end_y-0.2,   "Terminal\n(CKD)", fontsize=8,
           color="#C0392B", fontweight="bold", ha="center")

# C — condition UMAP
for cond in ["Control","CKD"]:
    m = sub.obs["condition_group"] == cond
    ax_cond.scatter(umap1[m], umap2[m], s=4, color=pal_cond[cond],
                    alpha=0.45, label=cond, rasterized=True)
style_umap(ax_cond)
ax_cond.set_title("Condition", fontsize=11, fontweight="bold", pad=6)
ax_cond.legend(fontsize=8, loc="best", framealpha=0.85,
               handles=[mpatches.Patch(color=pal_cond[c], label=c) for c in ["Control","CKD"]])

# ── Row 1 & 2 : Feature plots — gènes clés sur UMAP (8 gènes)
feat_genes = ["SPP1","APOE","SERPINH1","S100A2",
              "MRC1","STAB1","C1QA","MAF"]
feat_genes = [g for g in feat_genes if g in sub.var.index]

expr_cmap_up   = LinearSegmentedColormap.from_list("up",   ["#F5F5F5","#FADBD8","#E74C3C","#922B21"])
expr_cmap_down = LinearSegmentedColormap.from_list("down", ["#F5F5F5","#D6EAF8","#2980B9","#1A5276"])

n_feat = len(feat_genes)
for k, gene in enumerate(feat_genes):
    row_g = 1 + k // 5
    col_g = k % 5
    ax_g  = fig.add_subplot(gs[row_g, col_g])

    expr = get_expr(sub, gene)
    vmax = np.percentile(expr[expr > 0], 95) if (expr>0).any() else 1
    is_up = gene in genes_up
    cmap_g = expr_cmap_up if is_up else expr_cmap_down

    # Grey background for zero
    ax_g.scatter(umap1[expr==0], umap2[expr==0], s=3, color="#EBEBEB",
                 alpha=0.3, rasterized=True, zorder=1)
    sc_g = ax_g.scatter(umap1[expr>0], umap2[expr>0], c=expr[expr>0],
                        cmap=cmap_g, s=8, alpha=0.75, vmin=0, vmax=vmax,
                        rasterized=True, zorder=2)

    cb_g = plt.colorbar(sc_g, ax=ax_g, shrink=0.75, pad=0.02)
    cb_g.ax.tick_params(labelsize=6)

    direction = "▲ UP" if is_up else "▼ DOWN"
    col_title = "#C0392B" if is_up else "#1A5276"
    ax_g.set_title(f"{gene}  {direction}", fontsize=10, fontweight="bold",
                   color=col_title, pad=4)
    ax_g.set_xticks([]); ax_g.set_yticks([])
    ax_g.spines[["top","right","left","bottom"]].set_visible(False)

    # Spearman r with pseudotime
    r, _ = spearmanr(pt, expr)
    ax_g.text(0.98, 0.02, f"r={r:+.2f}", transform=ax_g.transAxes,
              ha="right", va="bottom", fontsize=8, color=col_title,
              fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=col_title, lw=0.8))

# ── Row 3 : Expression profiles along pseudotime
print("[3/4] Pseudotime profiles ...")

profile_genes = ["SPP1","APOE","SERPINH1","S100A2","MRC1","STAB1","C1QA","MAF","CCL8"]
profile_genes = [g for g in profile_genes if g in sub_f2.var.index]

ax_profiles = fig.add_subplot(gs[3, :])
ax_profiles.set_visible(False)

# Sub-gridspec for profile row
gs_prof = gridspec.GridSpecFromSubplotSpec(
    1, len(profile_genes), subplot_spec=gs[3, :], wspace=0.35)

for k, gene in enumerate(profile_genes):
    ax_p = fig.add_subplot(gs_prof[0, k])
    expr_g = get_expr(sub_f2, gene)
    is_up  = gene in genes_up
    col_g  = "#C0392B" if is_up else "#1A5276"
    col_fill = "#FADBD8" if is_up else "#D6EAF8"

    # Scatter by cell type (FOLR2 only)
    for ct, col_ct in [("FOLR2+_resident","#2ECC71"),("FOLR2_CKD","#E74C3C")]:
        m = sub_f2.obs["annot_final2"] == ct
        ax_p.scatter(pt_f2[m], expr_g[m], s=2.5, color=col_ct,
                     alpha=0.2, rasterized=True, zorder=1)

    # Smooth line
    cx, cy = smooth_profile(pt_f2, expr_g, n_bins=50, sigma=3)
    ax_p.plot(cx, cy, color=col_g, lw=2.5, zorder=5)
    ax_p.fill_between(cx, 0, cy, color=col_fill, alpha=0.35, zorder=3)

    # Condition boundary (mean pseudotime of CKD vs Control in FOLR2_CKD)
    pt_ctrl_f2 = pt_f2[sub_f2.obs["condition_group"]=="Control"]
    pt_ckd_f2  = pt_f2[sub_f2.obs["condition_group"]=="CKD"]
    boundary = (pt_ctrl_f2.mean() + pt_ckd_f2.mean()) / 2
    ax_p.axvline(boundary, color="#888", lw=1, ls="--", alpha=0.7)

    r_val, _ = spearmanr(pt_f2, expr_g)
    direction = "▲" if r_val > 0 else "▼"
    ax_p.set_title(f"{gene}  {direction}", fontsize=9.5, fontweight="bold",
                   color=col_g, pad=3)
    ax_p.set_xlabel("Pseudotemps →", fontsize=7.5, color="#666")
    ax_p.set_ylabel("Expression", fontsize=7.5) if k == 0 else None
    ax_p.tick_params(labelsize=6.5)
    ax_p.spines[["top","right"]].set_visible(False)
    ax_p.text(0.97, 0.93, f"r={r_val:+.2f}", transform=ax_p.transAxes,
              ha="right", va="top", fontsize=8, color=col_g, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85,
                        ec=col_g, lw=0.8))

    # Annotate boundary
    if k == 0:
        ax_p.text(boundary, ax_p.get_ylim()[1] if ax_p.get_ylim()[1]>0 else 1,
                  "Ctrl|CKD", fontsize=6.5, color="#888",
                  ha="center", va="bottom", rotation=90)

# ── Global legend for profile row
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ECC71",
           markersize=7, label="FOLR2+ resident"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#E74C3C",
           markersize=7, label="FOLR2 CKD"),
    Line2D([0],[0], color="black", lw=2, label="Tendance lissée"),
    Line2D([0],[0], color="#888", lw=1.5, ls="--", label="Frontière Ctrl|CKD"),
    mpatches.Patch(color="#FADBD8", label="UP in CKD"),
    mpatches.Patch(color="#D6EAF8", label="DOWN in CKD"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=6,
           fontsize=9, framealpha=0.9, edgecolor="lightgray",
           bbox_to_anchor=(0.5, 0.005), title="Légende profils pseudotemps",
           title_fontsize=9.5)

# ── Row labels
for row, label, ypos in [
    (0, "A  |  UMAP — Types, Pseudotemps, Condition", 0.956),
    (1, "B  |  Expression des gènes pathologiques (UP) et homéostatiques (DOWN) sur UMAP", 0.675),
    (3, "C  |  Profils d'expression le long du pseudotemps (FOLR2+ resident → FOLR2_CKD)", 0.225),
]:
    fig.text(0.01, ypos, label, fontsize=10.5, fontweight="bold",
             color="#2C3E50", va="bottom",
             bbox=dict(boxstyle="round,pad=0.25", fc="#ECF0F1", alpha=0.6, ec="none"))

# ── Suptitle
fig.suptitle(
    "Trajectoire pseudotemps — FOLR2 macrophages rénaux\n"
    "De l'homéostasie (FOLR2+_resident) vers le phénotype pathologique CKD (FOLR2_CKD)\n"
    "Gènes conducteurs : SPP1, APOE, SERPINH1 (fibrose)  ·  MRC1, STAB1, C1QA (perte résidente)",
    fontsize=13, fontweight="bold", color="#1A252F", y=0.985,
    bbox=dict(boxstyle="round,pad=0.4", fc="#FDFEFE", alpha=0.7, ec="lightgray")
)

print("[4/4] Saving ...")
outfile = f"{OUT}/trajectory_FOLR2CKD_beautiful.png"
fig.savefig(outfile, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {outfile}")
