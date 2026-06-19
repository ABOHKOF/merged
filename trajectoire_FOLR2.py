"""
Analyse de trajectoire FOLR2+_resident → FOLR2_CKD
PAGA + Diffusion Pseudotime (DPT) sur le sous-ensemble macrophagique FOLR2
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

sc.settings.verbosity = 1

H5AD = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
OUT  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\figures_rapport"

import pathlib
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

# ── 1. Charger et filtrer ──────────────────────────────────────────────────
print("Chargement ...")
adata_full = sc.read_h5ad(H5AD)

# Exclure AKI, garder macrophages FOLR2 + TREM2
mask = (
    adata_full.obs["diseasetype"].isin(["CKD","DKD","HKD","Control"]) &
    adata_full.obs["annot_final2"].isin(["FOLR2+_resident","FOLR2_CKD","TREM2+_macro"])
)
adata = adata_full[mask].copy()
print(f"  Sous-ensemble : {adata.n_obs:,} cellules")
print(adata.obs["annot_final2"].value_counts())

# Condition poolée
adata.obs["condition"] = adata.obs["diseasetype"].map(
    lambda x: "CKD poolé" if x in ("CKD","DKD","HKD") else "Contrôle"
)

# ── 2. Recalculer voisins sur l'espace scVI ────────────────────────────────
print("\nRecalcul des voisins (scVI) ...")
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15, n_pcs=10)

# ── 3. UMAP du sous-ensemble ───────────────────────────────────────────────
print("UMAP ...")
sc.tl.umap(adata, min_dist=0.4)

# ── 4. PAGA — connectivité entre populations ──────────────────────────────
print("PAGA ...")
sc.tl.paga(adata, groups="annot_final2")

palette = {
    "FOLR2+_resident": "#27AE60",
    "FOLR2_CKD":       "#C0392B",
    "TREM2+_macro":    "#E67E22",
}
adata.uns["annot_final2_colors"] = [palette[c]
    for c in adata.obs["annot_final2"].cat.categories]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A : PAGA graph
sc.pl.paga(adata, color="annot_final2", ax=axes[0],
           show=False, frameon=False,
           title="Connectivité PAGA entre populations FOLR2",
           node_size_scale=3, edge_width_scale=2,
           fontsize=11)

# Panel B : UMAP coloré par population
for pop, col in palette.items():
    m = adata.obs["annot_final2"] == pop
    axes[1].scatter(adata.obsm["X_umap"][m,0], adata.obsm["X_umap"][m,1],
                    c=col, s=6, alpha=0.7, rasterized=True,
                    label=f"{pop} (n={m.sum():,})", zorder=3)
axes[1].set_title("UMAP sous-ensemble macrophages FOLR2", fontsize=12)
axes[1].axis("off")
axes[1].legend(fontsize=9, markerscale=3, loc="best", framealpha=0.7)

fig.suptitle("Analyse de trajectoire — Macrophages FOLR2 rénaux", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig7_PAGA_FOLR2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig7_PAGA_FOLR2.png")

# ── 5. Diffusion Pseudotime ────────────────────────────────────────────────
print("\nDiffusion map + pseudotemps ...")
sc.tl.diffmap(adata, n_comps=10)

# Racine = cellules FOLR2+_resident en Contrôle (état le plus homéostatique)
root_mask = (adata.obs["annot_final2"] == "FOLR2+_resident") & \
            (adata.obs["condition"] == "Contrôle")
print(f"  Cellules racine : {root_mask.sum()}")

# Trouver la cellule la plus centrale dans cet état (min diffusion component 1)
root_candidates = np.where(root_mask)[0]
dc1 = adata.obsm["X_diffmap"][root_candidates, 1]
root_idx = root_candidates[np.argmin(dc1)]
adata.uns["iroot"] = int(root_idx)

sc.tl.dpt(adata, n_dcs=10)
print(f"  Pseudotemps calculé : min={adata.obs['dpt_pseudotime'].min():.3f}, "
      f"max={adata.obs['dpt_pseudotime'].max():.3f}")

# ── 6. Figures pseudotemps ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# A : UMAP coloré par pseudotemps
sc_map = axes[0].scatter(
    adata.obsm["X_umap"][:,0], adata.obsm["X_umap"][:,1],
    c=adata.obs["dpt_pseudotime"], cmap="viridis",
    s=5, alpha=0.8, rasterized=True
)
plt.colorbar(sc_map, ax=axes[0], label="Pseudotemps (DPT)", shrink=0.8)
# Marquer la racine
axes[0].scatter(adata.obsm["X_umap"][root_idx,0],
                adata.obsm["X_umap"][root_idx,1],
                c="white", s=80, marker="*", zorder=10, edgecolors="black")
axes[0].set_title("Pseudotemps de diffusion\n(★ = racine FOLR2+_resident Contrôle)",
                  fontsize=11, fontweight="bold")
axes[0].axis("off")

# B : UMAP split par population + pseudotemps en transparence
for pop, col in palette.items():
    m = adata.obs["annot_final2"] == pop
    axes[1].scatter(adata.obsm["X_umap"][m,0], adata.obsm["X_umap"][m,1],
                    c=col, s=5, alpha=0.6, rasterized=True, label=pop)
axes[1].set_title("Populations macrophagiques", fontsize=11, fontweight="bold")
axes[1].axis("off")
handles = [mpatches.Patch(color=c, label=p) for p, c in palette.items()]
axes[1].legend(handles=handles, fontsize=9, loc="best", framealpha=0.7)

# C : UMAP coloré par condition
cond_colors = {"Contrôle":"#2980B9","CKD poolé":"#C0392B"}
for cond, col in cond_colors.items():
    m = adata.obs["condition"] == cond
    axes[2].scatter(adata.obsm["X_umap"][m,0], adata.obsm["X_umap"][m,1],
                    c=col, s=5, alpha=0.5, rasterized=True, label=cond)
axes[2].set_title("Condition clinique", fontsize=11, fontweight="bold")
axes[2].axis("off")
handles2 = [mpatches.Patch(color=c, label=p) for p, c in cond_colors.items()]
axes[2].legend(handles=handles2, fontsize=9, loc="best", framealpha=0.7)

fig.suptitle("Pseudotemps de diffusion — Transition FOLR2+_resident → FOLR2_CKD",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig8_Pseudotime_UMAP.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig8_Pseudotime_UMAP.png")

# ── 7. Distribution du pseudotemps par population et condition ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Violin par population
pops = ["FOLR2+_resident","TREM2+_macro","FOLR2_CKD"]
pop_colors = [palette[p] for p in pops]
data_viol = [adata.obs.loc[adata.obs["annot_final2"]==p, "dpt_pseudotime"].values
             for p in pops]
parts = axes[0].violinplot(data_viol, showmedians=True, showextrema=False)
for pc, col in zip(parts["bodies"], pop_colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.8)
parts["cmedians"].set_colors("black")
parts["cmedians"].set_linewidth(2.5)
axes[0].set_xticks([1,2,3])
axes[0].set_xticklabels(pops, fontsize=10)
for i, (p, d) in enumerate(zip(pops, data_viol)):
    axes[0].text(i+1, np.median(d)+0.01, f"méd={np.median(d):.2f}",
                 ha="center", fontsize=8.5, fontweight="bold")
axes[0].set_ylabel("Pseudotemps (DPT)", fontsize=11)
axes[0].set_title("Distribution du pseudotemps par population", fontsize=12, fontweight="bold")
axes[0].spines[["top","right"]].set_visible(False)

# Violin FOLR2_CKD par condition
sub_ckd = adata[adata.obs["annot_final2"]=="FOLR2_CKD"]
data_cond = [
    sub_ckd.obs.loc[sub_ckd.obs["condition"]=="Contrôle", "dpt_pseudotime"].values,
    sub_ckd.obs.loc[sub_ckd.obs["condition"]=="CKD poolé", "dpt_pseudotime"].values,
]
parts2 = axes[1].violinplot(data_cond, showmedians=True, showextrema=False)
for pc, col in zip(parts2["bodies"], ["#F1948A","#C0392B"]):
    pc.set_facecolor(col)
    pc.set_alpha(0.8)
parts2["cmedians"].set_colors("black")
parts2["cmedians"].set_linewidth(2.5)
axes[1].set_xticks([1,2])
axes[1].set_xticklabels(["FOLR2_CKD\n(Contrôle)","FOLR2_CKD\n(CKD poolé)"], fontsize=10)
for i, d in enumerate(data_cond):
    axes[1].text(i+1, np.median(d)+0.01, f"méd={np.median(d):.2f}\nn={len(d):,}",
                 ha="center", fontsize=8.5, fontweight="bold")
axes[1].set_ylabel("Pseudotemps (DPT)", fontsize=11)
axes[1].set_title("Pseudotemps FOLR2_CKD selon la condition", fontsize=12, fontweight="bold")
axes[1].spines[["top","right"]].set_visible(False)

fig.suptitle("Position dans la trajectoire de différenciation macrophagique",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig9_Pseudotime_violins.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig9_Pseudotime_violins.png")

# ── 8. Expression des gènes-clés le long du pseudotemps ───────────────────
print("\nGènes le long du pseudotemps ...")

genes_traj = {
    "Homéostasie (↓)": ["LILRB1","C3AR1","FGL2","FCGR1A","HIF1A"],
    "Transition (→)":  ["TREM2","SPP1","VEGFA","CCL2","IL1B"],
    "Fibrose/CKD (↑)": ["S100A2","SAA1","COL1A1","CD248","LOXL2"],
}
genes_all = [g for grp in genes_traj.values() for g in grp
             if g in adata.var_names]

from scipy.sparse import issparse
from scipy.ndimage import uniform_filter1d

# Trier par pseudotemps
pt_order = np.argsort(adata.obs["dpt_pseudotime"].values)
pt_vals   = adata.obs["dpt_pseudotime"].values[pt_order]

# Expression lissée par bin
n_bins = 60
bins   = np.array_split(pt_order, n_bins)
pt_bin = np.array([adata.obs["dpt_pseudotime"].values[b].mean() for b in bins])

if "scvi_normalized" in adata.layers:
    X_layer = adata.layers["scvi_normalized"]
else:
    X_layer = adata.X

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

colors_grp = {
    "Homéostasie (↓)": ["#1ABC9C","#27AE60","#2ECC71","#58D68D","#A9DFBF"],
    "Transition (→)":  ["#F39C12","#E67E22","#D35400","#F1C40F","#F9E79F"],
    "Fibrose/CKD (↑)": ["#E74C3C","#C0392B","#922B21","#F1948A","#FADBD8"],
}

for ax, (grp_name, gene_list) in zip(axes, genes_traj.items()):
    color_list = colors_grp[grp_name]
    for gi, gene in enumerate(gene_list):
        if gene not in adata.var_names:
            continue
        gidx = list(adata.var_names).index(gene)
        if issparse(X_layer):
            expr = np.array(X_layer[:, gidx].todense()).flatten()
        else:
            expr = X_layer[:, gidx]
        # Normaliser 0-1
        expr_norm = (expr - expr.min()) / (expr.max() - expr.min() + 1e-9)
        # Moyenne par bin
        expr_bin = np.array([expr_norm[b].mean() for b in bins])
        # Lisser
        expr_smooth = uniform_filter1d(expr_bin, size=5)
        col = color_list[gi % len(color_list)]
        ax.plot(pt_bin, expr_smooth, lw=2.2, color=col, label=gene)
        ax.fill_between(pt_bin, expr_smooth, alpha=0.1, color=col)

    ax.set_ylabel("Expression\nnormalisée", fontsize=10)
    ax.set_title(grp_name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right" if "↓" in grp_name else "upper left",
              framealpha=0.6, ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(pt_bin.min(), pt_bin.max())

    # Annoter les zones
    ax.axvspan(pt_bin.min(), pt_bin[n_bins//3], alpha=0.04, color="#27AE60")
    ax.axvspan(pt_bin[n_bins//3], pt_bin[2*n_bins//3], alpha=0.04, color="#E67E22")
    ax.axvspan(pt_bin[2*n_bins//3], pt_bin.max(), alpha=0.04, color="#C0392B")

axes[-1].set_xlabel("Pseudotemps (DPT) →  FOLR2+_resident ... FOLR2_CKD", fontsize=11)

# Ajouter annotation des zones
for ax in axes:
    ymax = ax.get_ylim()[1]
    ax.text(pt_bin[n_bins//6],    ymax*0.95, "FOLR2+\nrés.", ha="center",
            fontsize=8, color="#27AE60", alpha=0.8)
    ax.text(pt_bin[n_bins//2],    ymax*0.95, "Transition", ha="center",
            fontsize=8, color="#E67E22", alpha=0.8)
    ax.text(pt_bin[5*n_bins//6],  ymax*0.95, "FOLR2\nCKD", ha="center",
            fontsize=8, color="#C0392B", alpha=0.8)

fig.suptitle("Dynamique d'expression le long de la trajectoire FOLR2+_resident → FOLR2_CKD",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig10_Genes_pseudotime.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig10_Genes_pseudotime.png")

# ── 9. Résumé : proportion de cellules par quartile de pseudotemps ─────────
print("\nFigure composition par quartile de pseudotemps ...")

adata.obs["pt_quartile"] = pd.qcut(
    adata.obs["dpt_pseudotime"], q=4,
    labels=["Q1\n(précoce)","Q2","Q3","Q4\n(tardif)"]
)

q_counts = adata.obs.groupby(["pt_quartile","annot_final2"]).size().unstack(fill_value=0)
q_pct    = q_counts.div(q_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(q_pct))
for pop in ["FOLR2+_resident","TREM2+_macro","FOLR2_CKD"]:
    if pop not in q_pct.columns:
        continue
    vals = q_pct[pop].values
    ax.bar(range(len(q_pct)), vals, bottom=bottom,
           color=palette[pop], label=pop, edgecolor="white", width=0.6)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax.text(i, b + v/2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    bottom += vals

ax.set_xticks(range(len(q_pct)))
ax.set_xticklabels(q_pct.index, fontsize=11)
ax.set_ylabel("% de cellules", fontsize=11)
ax.set_ylim(0, 105)
ax.set_title("Composition macrophagique par quartile de pseudotemps\n"
             "(Q1 = état le plus homéostatique, Q4 = état le plus pathologique)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, bbox_to_anchor=(1,1))
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/Fig11_Composition_quartiles_pseudotime.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig11_Composition_quartiles_pseudotime.png")

print(f"\n=== Analyse de trajectoire terminée. Figures dans :\n{OUT} ===")
