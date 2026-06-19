"""
UMAP de toutes les cellules — integrated.h5ad
3 panels : annotation large, annotation fine, condition
"""
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

PATH = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\integrated.h5ad"
OUT  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\umap_integrated_all.png"

print("[1/3] Loading data ...")
adata = ad.read_h5ad(PATH)
umap1 = adata.obsm["X_umap"][:, 0]
umap2 = adata.obsm["X_umap"][:, 1]

# ── Palettes
pal_broad = {
    "Immune":           "#E74C3C",
    "Proximal_Tubule":  "#2ECC71",
    "Distal_Tubule":    "#27AE60",
    "Loop_of_Henle":    "#1ABC9C",
    "Endothelial":      "#3498DB",
    "Epithelial":       "#85C1E9",
    "Stroma":           "#E67E22",
    "Mesenchymal":      "#F39C12",
    "Specialized":      "#9B59B6",
    "Neuronal":         "#D35400",
    "Other":            "#BDC3C7",
}

pal_fine = {
    # Immune
    "CD8T":                  "#C0392B",
    "CD4T":                  "#E74C3C",
    "NK":                    "#F1948A",
    "Natural Killer Cells":  "#F1948A",
    "B Cells":               "#922B21",
    "B_Naive":               "#A93226",
    "B_memory":              "#CB4335",
    "Baso/Mast":             "#7B241C",
    "Basophils":             "#7B241C",
    "Neutrophil":            "#F5B7B1",
    "Plasma Cells":          "#D98880",
    "Plasma_Cells":          "#D98880",
    "CD14_Mono":             "#E59866",
    "CD14+_mono":            "#E59866",
    "CD16_Mono":             "#FAD7A0",
    "CD16+_mono":            "#FAD7A0",
    "CD16+_RUNX3-_mono":     "#FAD7A0",
    "FOLR2+_resident":       "#2ECC71",
    "FOLR2+_CKD":            "#E74C3C",
    "TREM2+_CD1E+_macro":    "#8E44AD",
    "Mac":                   "#AF7AC5",
    "cDC":                   "#A569BD",
    "pDC":                   "#6C3483",
    "Dendritic Cells":       "#6C3483",
    "IFNab_peri_like":       "#D2B4DE",
    "T Cells":               "#F1948A",
    # Proximal Tubule
    "PT_S1":                 "#1E8449",
    "PT_S2":                 "#27AE60",
    "PT_S3":                 "#52BE80",
    "iPT":                   "#A9DFBF",
    "Injured Tubule":        "#FDEBD0",
    "Proximal Tubule":       "#52BE80",
    # Distal Tubule
    "DCT1":                  "#0E6655",
    "DCT2":                  "#148F77",
    "CNT":                   "#1ABC9C",
    "Connecting Tubule":     "#1ABC9C",
    "Distal Convoluted Tubule": "#76D7C4",
    "IC_A":                  "#117A65",
    "IC_B":                  "#0A3D2E",
    "Intercalated Cells":    "#117A65",
    "Collecting Duct Principal Cells": "#D5F5E3",
    "PC":                    "#D5F5E3",
    # Loop of Henle
    "ATL":                   "#A3E4D7",
    "DTL":                   "#76D7C4",
    "C_TAL":                 "#45B39D",
    "M_TAL":                 "#1ABC9C",
    "Thick Ascending Limb":  "#45B39D",
    "Descending Thin Limb":  "#76D7C4",
    "Macula_Densa":          "#148F77",
    # Endothelial
    "Endo_GC":               "#1A5276",
    "Glomerular Capillaries":"#2E86C1",
    "Endo_Peritubular":      "#2980B9",
    "Vasa Recta":            "#5DADE2",
    "Enod_Lym":              "#7FB3D3",
    "Arterioral Endothelium":"#154360",
    "Venular Endothelium":   "#1F618D",
    "Injured Endothelium":   "#85C1E9",
    "Lymph Endothelium":     "#AED6F1",
    # Stroma
    "Fib":                   "#E67E22",
    "CXCL_iFibro":           "#CA6F1E",
    "detox_iFibro":          "#9A7D0A",
    "tgfb_myoFibro":         "#F0B27A",
    "wound_myoFibro":        "#E59866",
    "MyoFib":                "#F5CBA7",
    "VSMC/Pericyte":         "#784212",
    "contractile_peri_like": "#A04000",
    "GS_Stromal":            "#D4AC0D",
    # Specialized
    "Podo":                  "#8E44AD",
    "Podocytes":             "#8E44AD",
    "PEC":                   "#BB8FCE",
    "Mes":                   "#6C3483",
    # Other
    "Neural_Cells":          "#D35400",
    "Schwann Cells":         "#DC7633",
    "Uroethlial Cells":      "#F0E68C",
    "Other":                 "#BDC3C7",
}

pal_cond = {
    "Control":     "#2ECC71",
    "LivingDonor": "#27AE60",
    "HKD":         "#3498DB",
    "CKD":         "#E67E22",
    "DKD":         "#E74C3C",
    "AKI":         "#8E44AD",
}

# ── Figure layout
print("[2/3] Plotting ...")
fig = plt.figure(figsize=(26, 9), facecolor="white")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08,
                        left=0.02, right=0.98, top=0.92, bottom=0.06)

def style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=9, color="#555")
    ax.set_ylabel("UMAP 2", fontsize=9, color="#555")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","left","bottom"]].set_visible(False)

# ── Panel 1 : Annotation large (annot_atlas_low)
ax1 = fig.add_subplot(gs[0])
broad_types = list(pal_broad.keys())
obs_broad   = adata.obs["annot_atlas_low"].astype(str).fillna("Other")

for ct in broad_types[::-1]:
    m = obs_broad == ct
    if m.sum() == 0: continue
    ax1.scatter(umap1[m], umap2[m], s=0.6, color=pal_broad.get(ct, "#BDC3C7"),
                alpha=0.45, rasterized=True, label=ct)

style_ax(ax1, "Annotation large")
handles1 = [mpatches.Patch(color=pal_broad.get(ct,"#BDC3C7"), label=f"{ct}\n({(obs_broad==ct).sum():,})")
            for ct in broad_types if (obs_broad==ct).sum() > 0]
ax1.legend(handles=handles1, fontsize=7.5, loc="lower left", framealpha=0.88,
           edgecolor="lightgray", title="Type cellulaire", title_fontsize=8.5,
           markerscale=3, ncol=1, handlelength=1.2)

# ── Panel 2 : Annotation fine (merged_annot)
ax2 = fig.add_subplot(gs[1])
obs_fine = adata.obs["merged_annot"].astype(str).fillna("Other")
fine_types_present = sorted(obs_fine.unique())

for ct in fine_types_present:
    m = obs_fine == ct
    col = pal_fine.get(ct, "#BDC3C7")
    ax2.scatter(umap1[m], umap2[m], s=0.5, color=col,
                alpha=0.4, rasterized=True, label=ct)

style_ax(ax2, "Annotation fine")
# Legend: group by broad type for readability
legend_groups = {
    "Immune":        ["CD8T","CD4T","NK","Natural Killer Cells","B Cells","B_Naive","B_memory",
                      "Neutrophil","Plasma Cells","Plasma_Cells","CD14_Mono","CD14+_mono",
                      "CD16_Mono","CD16+_mono","FOLR2+_resident","FOLR2+_CKD","TREM2+_CD1E+_macro",
                      "Mac","cDC","pDC","Dendritic Cells","Baso/Mast","Basophils","IFNab_peri_like"],
    "Tubules":       ["PT_S1","PT_S2","PT_S3","iPT","Proximal Tubule","DCT1","DCT2","CNT",
                      "Connecting Tubule","IC_A","IC_B","Intercalated Cells","PC",
                      "Collecting Duct Principal Cells","ATL","DTL","C_TAL","M_TAL",
                      "Thick Ascending Limb","Descending Thin Limb","Macula_Densa","Injured Tubule"],
    "Endothelial":   ["Endo_GC","Glomerular Capillaries","Endo_Peritubular","Vasa Recta",
                      "Arterioral Endothelium","Venular Endothelium","Injured Endothelium",
                      "Enod_Lym","Lymph Endothelium"],
    "Stroma/Spec.":  ["Fib","CXCL_iFibro","detox_iFibro","tgfb_myoFibro","wound_myoFibro",
                      "MyoFib","VSMC/Pericyte","contractile_peri_like","GS_Stromal",
                      "Podo","Podocytes","PEC","Mes","Neural_Cells","Schwann Cells"],
}
handles2 = []
for grp, genes in legend_groups.items():
    for ct in genes:
        if ct in fine_types_present:
            n = (obs_fine==ct).sum()
            if n > 0:
                handles2.append(mpatches.Patch(color=pal_fine.get(ct,"#BDC3C7"),
                                               label=f"{ct} ({n:,})"))
ax2.legend(handles=handles2, fontsize=5.8, loc="lower left", framealpha=0.85,
           edgecolor="lightgray", title="Type fin", title_fontsize=7,
           markerscale=2.5, ncol=2, handlelength=1.0, columnspacing=0.8,
           labelspacing=0.25)

# ── Panel 3 : Condition (diseasetype)
ax3 = fig.add_subplot(gs[2])
obs_cond  = adata.obs["diseasetype"].astype(str).fillna("Other")
cond_order = ["LivingDonor","Control","HKD","CKD","DKD","AKI"]

for cond in cond_order[::-1]:
    m = obs_cond == cond
    if m.sum() == 0: continue
    ax3.scatter(umap1[m], umap2[m], s=0.5, color=pal_cond.get(cond,"#BDC3C7"),
                alpha=0.45, rasterized=True, label=cond)

style_ax(ax3, "Condition / Pathologie")
handles3 = [mpatches.Patch(color=pal_cond[c], label=f"{c} (n={( obs_cond==c).sum():,})")
            for c in cond_order if (obs_cond==c).sum() > 0]
ax3.legend(handles=handles3, fontsize=9, loc="lower left", framealpha=0.88,
           edgecolor="lightgray", title="Condition", title_fontsize=10,
           markerscale=3, handlelength=1.2)

# ── Suptitle
n_total = adata.shape[0]
n_genes = adata.shape[1]
fig.suptitle(
    f"UMAP intégré — Rein humain scRNA-seq / snRNA-seq\n"
    f"{n_total:,} cellules  ·  {n_genes:,} gènes  ·  intégration scVI  ·  multi-cohorte (KPMP + HK + autres)",
    fontsize=13, fontweight="bold", y=0.98, color="#1A252F"
)

print("[3/3] Saving ...")
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {OUT}")

# ── Save h5ad with global annotations
OUT_H5AD = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\integrated_annot_global.h5ad"
print("[+] Saving h5ad (global annotations) ...")
cols_keep = [c for c in ["annot_atlas_low", "merged_annot", "diseasetype"] if c in adata.obs.columns]
adata_out = adata[:, []].copy()   # keep obs + obsm, drop gene matrix
adata_out.obs = adata.obs[cols_keep].copy()
adata_out.obsm["X_umap"] = adata.obsm["X_umap"]
adata_out.write_h5ad(OUT_H5AD)
print(f"  Saved: {OUT_H5AD}  ({adata_out.n_obs:,} cells, colonnes: {cols_keep})")
