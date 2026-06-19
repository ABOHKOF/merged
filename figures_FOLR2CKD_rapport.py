"""
Figures Section II.3 — Caractérisation transcriptomique de la population FOLR2_CKD
Rapport de stage

Fig 1 : UMAP global — tous les types immunitaires
Fig 2 : UMAP split CKD-poolé vs Contrôle — FOLR2_CKD en avant-plan
Fig 3 : Proportion de FOLR2_CKD par condition (enrichissement CKD)
Fig 4 : Composition immunitaire CKD-poolé vs Contrôle (stacked bar)
Fig 5 : Dotplot — marqueurs-clés sur FOLR2_CKD (CKD vs Ctrl) et FOLR2+_resident
Fig 6 : Violin — score transcriptomique FOLR2_CKD selon les conditions
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

sc.settings.verbosity = 0

H5AD   = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
OUT    = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\figures_rapport"

import pathlib
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

print("Chargement adata_immune_annot_final.h5ad ...")
adata = sc.read_h5ad(H5AD)
print(f"  {adata.n_obs:,} cellules, {adata.n_vars} gènes")

# ── Exclure les patients AKI ─────────────────────────────────────────────────
mask_no_aki = adata.obs["diseasetype"] != "AKI"
adata = adata[mask_no_aki].copy()
print(f"  Après exclusion AKI : {adata.n_obs:,} cellules")

# ── Condition poolée (CKD + DKD + HKD = CKD poolé) ──────────────────────────
adata.obs["condition"] = adata.obs["diseasetype"].map(
    lambda x: "CKD poolé" if x in ("CKD","DKD","HKD") else "Contrôle"
)

# ── Palette types cellulaires ─────────────────────────────────────────────────
cell_types = adata.obs["annot_final2"].cat.categories.tolist() \
             if hasattr(adata.obs["annot_final2"], "cat") \
             else sorted(adata.obs["annot_final2"].unique())

base_colors = [
    "#AAAAAA","#BBBBBB","#CCCCCC","#DDDDDD","#EEEEEE","#CDCDCD",
    "#B0C4DE","#87CEEB","#4682B4","#1E90FF",
    "#90EE90","#3CB371","#006400",
    "#FFD700","#FFA500","#FF8C00",
    "#DDA0DD","#9370DB","#6A0DAD",
    "#FFC0CB","#FF69B4",
]
palette = {}
for i, ct in enumerate(cell_types):
    if ct == "FOLR2_CKD":
        palette[ct] = "#C0392B"
    elif ct == "FOLR2+_resident":
        palette[ct] = "#27AE60"
    elif ct == "TREM2+_macro":
        palette[ct] = "#E67E22"
    else:
        palette[ct] = base_colors[i % len(base_colors)]

umap = adata.obsm["X_umap"]
adata.obs["UMAP1"] = umap[:, 0]
adata.obs["UMAP2"] = umap[:, 1]

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — UMAP global tous types immunitaires
# ════════════════════════════════════════════════════════════════════════════
print("Figure 1 : UMAP global ...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax_idx, (ax, color_by, title) in enumerate(zip(
    axes,
    ["annot_final2", "condition"],
    ["Types cellulaires immunitaires", "Condition"],
)):
    if color_by == "annot_final2":
        cpal = palette
        cats = cell_types
    else:
        cpal = {"CKD poolé":"#C0392B","Contrôle":"#2980B9"}
        cats = ["Contrôle","CKD poolé"]

    # background gris
    ax.scatter(adata.obs["UMAP1"], adata.obs["UMAP2"],
               c="#EEEEEE", s=0.3, rasterized=True, zorder=1)

    for cat in cats:
        mask = adata.obs[color_by] == cat
        if mask.sum() == 0:
            continue
        size = 3 if cat in ("FOLR2_CKD","FOLR2+_resident") else 1.5
        zorder = 5 if cat in ("FOLR2_CKD","FOLR2+_resident") else 2
        ax.scatter(adata.obs.loc[mask, "UMAP1"],
                   adata.obs.loc[mask, "UMAP2"],
                   c=cpal.get(cat,"#999999"), s=size,
                   rasterized=True, zorder=zorder, label=cat)

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    handles = [mpatches.Patch(color=cpal.get(c,"#999999"), label=c)
               for c in cats if (adata.obs[color_by] == c).sum() > 0]
    ncol = 2 if color_by == "annot_final2" else 1
    ax.legend(handles=handles, fontsize=7, markerscale=2,
              loc="lower left", ncol=ncol,
              framealpha=0.7, handlelength=1)

    # label centroïde FOLR2
    for pop in ["FOLR2_CKD","FOLR2+_resident"]:
        if color_by == "annot_final2":
            mask = adata.obs["annot_final2"] == pop
            cx = adata.obs.loc[mask,"UMAP1"].median()
            cy = adata.obs.loc[mask,"UMAP2"].median()
            ax.annotate(pop, (cx, cy), fontsize=8, fontweight="bold",
                        color=palette[pop],
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=palette[pop]),
                        xytext=(cx+1, cy+1))

fig.suptitle("Paysage immunitaire rénal — Atlas intégré (91 680 cellules)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/Fig1_UMAP_global.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig1_UMAP_global.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — UMAP FOLR2 populations, split CKD-poolé vs Contrôle
# ════════════════════════════════════════════════════════════════════════════
print("Figure 2 : UMAP split conditions ...")

conditions = ["Contrôle","CKD poolé"]
cond_colors = {"FOLR2_CKD":"#C0392B","FOLR2+_resident":"#27AE60","other":"#EEEEEE"}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, cond in zip(axes, conditions):
    mask_cond = adata.obs["condition"] == cond
    sub = adata[mask_cond]

    # background gris
    ax.scatter(sub.obs["UMAP1"], sub.obs["UMAP2"],
               c="#DDDDDD", s=0.4, rasterized=True, zorder=1, alpha=0.5)

    for pop, col, z in [("FOLR2+_resident","#27AE60",3),
                         ("FOLR2_CKD","#C0392B",4)]:
        m = sub.obs["annot_final2"] == pop
        if m.sum() == 0:
            continue
        ax.scatter(sub.obs.loc[m,"UMAP1"], sub.obs.loc[m,"UMAP2"],
                   c=col, s=6, zorder=z, rasterized=True,
                   label=f"{pop} (n={m.sum():,})", alpha=0.85)

    n_ckd = (sub.obs["annot_final2"]=="FOLR2_CKD").sum()
    n_res = (sub.obs["annot_final2"]=="FOLR2+_resident").sum()
    n_tot = len(sub)
    pct_ckd = 100*n_ckd/n_tot
    pct_res = 100*n_res/n_tot

    ax.set_title(f"{cond}\n(n={n_tot:,} cellules immunitaires)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.legend(fontsize=8, markerscale=2, loc="lower left", framealpha=0.7)

    ax.text(0.98, 0.02,
            f"FOLR2_CKD : {pct_ckd:.1f}%\nFOLR2+_res : {pct_res:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(fc="white", alpha=0.7, ec="none"))

fig.suptitle("Distribution des macrophages FOLR2 dans le contexte CKD vs Contrôle",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig2_UMAP_FOLR2_split.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig2_UMAP_FOLR2_split.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Proportion de FOLR2_CKD par condition
# ════════════════════════════════════════════════════════════════════════════
print("Figure 3 : Proportion FOLR2_CKD par condition ...")

cond_order = ["Contrôle","CKD","DKD","HKD"]
prop_data = []
for c in cond_order:
    if c in ("CKD","DKD","HKD"):
        mask = adata.obs["diseasetype"] == c
    else:
        mask = adata.obs["diseasetype"] == "Control"
    n_total = mask.sum()
    n_ckd   = ((adata.obs["annot_final2"]=="FOLR2_CKD") & mask).sum()
    n_res   = ((adata.obs["annot_final2"]=="FOLR2+_resident") & mask).sum()
    prop_data.append({
        "Condition": c,
        "n_total": n_total,
        "pct_FOLR2_CKD": 100*n_ckd/n_total if n_total>0 else 0,
        "pct_FOLR2_res": 100*n_res/n_total if n_total>0 else 0,
    })
df_prop = pd.DataFrame(prop_data)

colors_cond = {"Contrôle":"#2980B9","CKD":"#C0392B","DKD":"#E74C3C","HKD":"#F1948A"}
bar_colors = [colors_cond[c] for c in df_prop["Condition"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title, ylabel in zip(
    axes,
    ["pct_FOLR2_CKD","pct_FOLR2_res"],
    ["FOLR2_CKD (population pathologique)",
     "FOLR2+_resident (population homéostatique)"],
    ["% des cellules immunitaires","% des cellules immunitaires"],
):
    bars = ax.bar(df_prop["Condition"], df_prop[col],
                  color=bar_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, df_prop[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, df_prop[col].max()*1.3 + 0.05)
    ax.spines[["top","right"]].set_visible(False)
    # ligne médiane contrôle
    ctrl_val = df_prop.loc[df_prop["Condition"]=="Contrôle", col].values[0]
    ax.axhline(ctrl_val, color="#2980B9", ls="--", lw=1, alpha=0.7,
               label=f"Niveau Contrôle ({ctrl_val:.2f}%)")
    ax.legend(fontsize=9)

fig.suptitle("Proportion des macrophages FOLR2 par condition clinique",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig3_Proportion_FOLR2_conditions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig3_Proportion_FOLR2_conditions.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Composition immunitaire CKD-poolé vs Contrôle (stacked bar)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 4 : Composition immunitaire ...")

comp_conds = {"Contrôle": adata.obs["diseasetype"]=="Control",
              "CKD poolé": adata.obs["diseasetype"].isin(["CKD","DKD","HKD"])}

comp_data = {}
for cond, mask in comp_conds.items():
    n = mask.sum()
    comp_data[cond] = {ct: ((adata.obs["annot_final2"]==ct) & mask).sum()/n*100
                       for ct in cell_types}

df_comp = pd.DataFrame(comp_data)

# Trier par différence CKD-Ctrl
df_comp["diff"] = df_comp["CKD poolé"] - df_comp["Contrôle"]
df_comp = df_comp.sort_values("diff", ascending=True)

fig, ax = plt.subplots(figsize=(13, 9))
x = np.arange(len(df_comp))
width = 0.35

bars1 = ax.barh(x - width/2, df_comp["Contrôle"], width,
                color="#2980B9", alpha=0.85, label="Contrôle", edgecolor="white")
bars2 = ax.barh(x + width/2, df_comp["CKD poolé"], width,
                color="#C0392B", alpha=0.85, label="CKD poolé", edgecolor="white")

ax.set_yticks(x)
ax.set_yticklabels(df_comp.index, fontsize=9)
ax.set_xlabel("% des cellules immunitaires", fontsize=11)
ax.set_title("Composition immunitaire rénale\nCKD poolé vs Contrôle",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.axvline(0, color="black", lw=0.5)

# Annoter la différence
for i, (ct, row) in enumerate(df_comp.iterrows()):
    diff = row["diff"]
    color = "#C0392B" if diff > 0 else "#2980B9"
    ax.text(max(row["Contrôle"], row["CKD poolé"]) + 0.05,
            i, f"{diff:+.2f}%", va="center", fontsize=7.5,
            color=color, fontweight="bold" if ct in ("FOLR2_CKD","FOLR2+_resident") else "normal")

fig.tight_layout()
fig.savefig(f"{OUT}/Fig4_Composition_immunitaire.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved Fig4_Composition_immunitaire.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Dotplot marqueurs-clés FOLR2_CKD vs FOLR2+_resident
# ════════════════════════════════════════════════════════════════════════════
print("Figure 5 : Dotplot marqueurs-clés ...")

genes_up  = ["S100A2","SAA1","AGR2","CD248","SLPI","KRT7","SOD3","COL1A1","COL3A1","LOXL2"]
genes_down = ["LILRB1","C3AR1","FGL2","FCGR1A","PTPRE","REL","HIF1A","C3","CXCL10","HLA-DRA"]

# vérifier présence dans var_names
all_genes = [g for g in (genes_up + genes_down) if g in adata.var_names]
genes_up   = [g for g in genes_up   if g in adata.var_names]
genes_down = [g for g in genes_down if g in adata.var_names]
print(f"  Gènes UP trouvés : {genes_up}")
print(f"  Gènes DOWN trouvés : {genes_down}")

pops_of_interest = ["FOLR2_CKD","FOLR2+_resident","TREM2+_macro","CD14_Mono"]
mask_pops = adata.obs["annot_final2"].isin(pops_of_interest)
sub = adata[mask_pops].copy()

# Ajouter group CKD/Ctrl pour FOLR2_CKD
sub.obs["group_detail"] = sub.obs["annot_final2"].astype(str)
mask_ckd_pop  = (sub.obs["annot_final2"]=="FOLR2_CKD") & (sub.obs["condition"]=="CKD poolé")
mask_ctrl_pop = (sub.obs["annot_final2"]=="FOLR2_CKD") & (sub.obs["condition"]=="Contrôle")
mask_res_pop  = sub.obs["annot_final2"]=="FOLR2+_resident"
sub.obs.loc[mask_ckd_pop,  "group_detail"] = "FOLR2_CKD (CKD)"
sub.obs.loc[mask_ctrl_pop, "group_detail"] = "FOLR2_CKD (Ctrl)"
sub.obs.loc[mask_res_pop,  "group_detail"] = "FOLR2+_resident"

group_order = ["Contrôle\n(ref)","FOLR2_CKD (Ctrl)","FOLR2_CKD (CKD)",
               "FOLR2+_resident","TREM2+_macro","CD14_Mono"]
group_order = [g for g in group_order if g in sub.obs["group_detail"].values]

if all_genes and len(sub) > 0:
    sub.obs["group_detail"] = pd.Categorical(
        sub.obs["group_detail"],
        categories=[g for g in ["FOLR2_CKD (Ctrl)","FOLR2_CKD (CKD)",
                                 "FOLR2+_resident","TREM2+_macro","CD14_Mono"]
                    if g in sub.obs["group_detail"].values]
    )
    sc.tl.rank_genes_groups(sub, groupby="group_detail", method="wilcoxon",
                             use_raw=False, layer="scvi_normalized")

    # Calcul manuel mean expression + pct expressed
    from scipy.sparse import issparse
    groups = sub.obs["group_detail"].cat.categories.tolist()
    gene_list = genes_up + genes_down

    mean_expr = {}
    pct_expr  = {}
    for g in groups:
        m = sub.obs["group_detail"] == g
        if "scvi_normalized" in sub.layers:
            X = sub[m].layers["scvi_normalized"]
        else:
            X = sub[m].X
        if issparse(X):
            X = X.toarray()
        gene_idx = [list(sub.var_names).index(gn) for gn in gene_list if gn in sub.var_names]
        gene_list_filt = [gn for gn in gene_list if gn in sub.var_names]
        Xg = X[:, gene_idx]
        mean_expr[g] = np.mean(Xg, axis=0)
        pct_expr[g]  = np.mean(Xg > 0, axis=0)

    df_mean = pd.DataFrame(mean_expr, index=gene_list_filt).T
    df_pct  = pd.DataFrame(pct_expr,  index=gene_list_filt).T

    # normaliser par gène (0-1)
    df_mean_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(16, 5))
    genes_plot = gene_list_filt
    groups_plot = groups
    x_pos = {g: i for i, g in enumerate(genes_plot)}
    y_pos = {g: i for i, g in enumerate(groups_plot)}

    for gn in genes_plot:
        for gr in groups_plot:
            x = x_pos[gn]
            y = y_pos[gr]
            val = df_mean_norm.loc[gr, gn]
            pct = df_pct.loc[gr, gn]
            size = pct * 500 + 5
            color_val = val
            ax.scatter(x, y, s=size, c=[[color_val, 0, 0]] if gn in genes_up
                       else [[0, 0, color_val]],
                       alpha=0.85, zorder=3)

    ax.set_xticks(range(len(genes_plot)))
    ax.set_xticklabels(genes_plot, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(groups_plot)))
    ax.set_yticklabels(groups_plot, fontsize=9)
    ax.set_xlim(-0.7, len(genes_plot)-0.3)
    ax.set_ylim(-0.7, len(groups_plot)-0.3)
    ax.grid(alpha=0.3)
    ax.axvline(len(genes_up)-0.5, color="gray", ls="--", lw=1)
    ax.text(len(genes_up)/2-0.5, -0.6, "↑ UP_CKD (fibrose / stress)",
            ha="center", fontsize=9, color="#C0392B", fontweight="bold")
    ax.text(len(genes_up) + len(genes_down)/2-0.5, -0.6, "↓ DOWN_CKD (homéostasie perdue)",
            ha="center", fontsize=9, color="#2980B9", fontweight="bold")
    ax.set_title("Expression des marqueurs discriminants FOLR2_CKD\n(taille = % cellules exprimant, couleur = expression normalisée)",
                 fontsize=12, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#C0392B",
               markersize=8, label="Gènes UP CKD (fibrose/EMT)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#2980B9",
               markersize=8, label="Gènes DOWN CKD (homéostasie)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, label="Taille ∝ % cellules positives"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig5_Dotplot_marqueurs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved Fig5_Dotplot_marqueurs.png")
else:
    print("  Gènes non trouvés ou sous-ensemble vide — Fig5 ignorée")

# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Violin Zscore_FOLR2_CKD par condition (dans la population FOLR2_CKD)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 6 : Violin Zscore FOLR2_CKD ...")

if "Zscore_FOLR2_CKD" in adata.obs.columns:
    mask_folr2 = adata.obs["annot_final2"].isin(["FOLR2_CKD","FOLR2+_resident"])
    sub_f = adata[mask_folr2].copy()
    sub_f.obs["pop_cond"] = (sub_f.obs["annot_final2"].astype(str) + "\n" +
                             sub_f.obs["condition"].astype(str))

    order = [c for c in [
        "FOLR2+_resident\nContrôle","FOLR2+_resident\nCKD poolé",
        "FOLR2_CKD\nContrôle","FOLR2_CKD\nCKD poolé",
    ] if c in sub_f.obs["pop_cond"].values]

    palette_viol = {
        "FOLR2+_resident\nContrôle": "#27AE60",
        "FOLR2+_resident\nCKD poolé": "#A9DFBF",
        "FOLR2_CKD\nContrôle": "#F1948A",
        "FOLR2_CKD\nCKD poolé": "#C0392B",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    data_viol = [sub_f.obs.loc[sub_f.obs["pop_cond"]==g, "Zscore_FOLR2_CKD"].values
                 for g in order]
    parts = ax.violinplot(data_viol, positions=range(len(order)),
                          showmedians=True, showextrema=False)
    for pc, g in zip(parts["bodies"], order):
        pc.set_facecolor(palette_viol.get(g,"#AAAAAA"))
        pc.set_alpha(0.8)
    parts["cmedians"].set_colors(["black"]*len(order))
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("Score transcriptomique FOLR2_CKD (Z-score)", fontsize=11)
    ax.set_title("Score pathologique FOLR2_CKD dans les populations de macrophages FOLR2\n"
                 "selon la condition clinique", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)

    for i, (g, d) in enumerate(zip(order, data_viol)):
        ax.text(i, np.median(d) + 0.05, f"n={len(d):,}\nméd={np.median(d):.2f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig6_Violin_Zscore_FOLR2CKD.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved Fig6_Violin_Zscore_FOLR2CKD.png")
else:
    print("  Colonne Zscore_FOLR2_CKD absente — Fig6 ignorée")

print(f"\n=== Toutes les figures enregistrées dans :\n{OUT} ===")
