import anndata as ad, pandas as pd, numpy as np, scanpy as sc
from scipy.sparse import issparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings
warnings.filterwarnings("ignore")

path = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
out  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication"

print("[1/4] Loading data...")
adata = ad.read_h5ad(path)
adata.obs["annot_final2"]    = adata.obs["annot_final2"].astype(str)
adata.obs["condition_group"] = adata.obs["diseasetype"].map(
    lambda d: "CKD" if d in ("DKD","HKD","CKD") else ("Control" if d=="Control" else "Other")
).astype(str)

f2 = adata[(adata.obs["annot_final2"]=="FOLR2_CKD") &
           (adata.obs["condition_group"].isin(["CKD","Control"]))].copy()
f2.X = f2.layers["scvi_normalized"].copy()
print(f"  FOLR2_CKD subset: {f2.shape}  CKD={( f2.obs.condition_group=='CKD').sum()}  Control={( f2.obs.condition_group=='Control').sum()}")

print("[2/4] Running Wilcoxon DE (CKD vs Control)...")
sc.tl.rank_genes_groups(f2, groupby="condition_group", groups=["CKD"], reference="Control",
                         method="wilcoxon", n_genes=f2.shape[1], pts=True)
res = sc.get.rank_genes_groups_df(f2, group="CKD")

pts_ckd  = pd.DataFrame(list(f2.uns["rank_genes_groups"]["pts"]["CKD"].items()),  columns=["names","pct_ckd"])
pts_ctrl = pd.DataFrame(list(f2.uns["rank_genes_groups"]["pts"]["Control"].items()), columns=["names","pct_ctrl"])
res = res.merge(pts_ckd, on="names", how="left").merge(pts_ctrl, on="names", how="left")
res["-log10_padj"] = -np.log10(res["pvals_adj"].clip(1e-300))
res["logFC"] = res["logfoldchanges"]

lfc_thr = 0.5; padj_thr = 0.05
res["status"] = "NS"
res.loc[(res.logFC >=  lfc_thr) & (res.pvals_adj < padj_thr), "status"] = "UP_CKD"
res.loc[(res.logFC <= -lfc_thr) & (res.pvals_adj < padj_thr), "status"] = "DOWN_CKD"
print(f"  {res['status'].value_counts().to_dict()}")

res.to_csv(os.path.join(out, "DE_FOLR2CKD_CKD_vs_Control.csv"), index=False)
print("  DE table saved.")

# ── Gene categories for phenotype
genes_highlight = {
    "Inflammation":   ["CCL8","CCL2","CCL7","CCL20","CXCL10","CXCL9","IL1B","TNF","IL6","IL18",
                       "S100A8","S100A9","S100A4","S100A2","SAA1","HMOX1","NF1"],
    "Fibrosis":       ["SPP1","FN1","LGALS3","MMP9","MMP12","MMP19","VCAN","GPNMB","TREM2",
                       "APOE","FABP5","CTSD","LPL","CHI3L1","CHIL3","PDGFB","VEGFA"],
    "Lipid_Ox":       ["APOC1","FABP4","OLR1","CD36","LDLR","NPC1","LIPA","PLIN2","ABCA1"],
    "CCL8_axis":      ["CCL8","CCR2","CCR1","CCL7","CCL13"],
    "Resident_lost":  ["FOLR2","LYVE1","MRC1","TIMD4","CX3CR1","CD163","MAF","NR4A1",
                       "STAB1","F13A1","RNASE1","C1QA","C1QB","C1QC","VSIG4"],
}
all_hl = {g: cat for cat, gs in genes_highlight.items() for g in gs}

cat_colors = {
    "Inflammation":  "#E74C3C",
    "Fibrosis":      "#E67E22",
    "Lipid_Ox":      "#8E44AD",
    "CCL8_axis":     "#27AE60",
    "Resident_lost": "#2980B9",
}
key_genes = {"CCL8","SPP1","TREM2","FOLR2","LYVE1","MRC1","IL1B","MMP9","APOE","GPNMB",
             "LGALS3","CXCL10","SAA1","S100A9","VCAN","CD163","C1QA","FABP5","FABP4","CHI3L1"}

print("[3/4] Generating volcano plot...")

# ── Layout : volcano (gauche) | table UP (milieu) | table DOWN (droite)
fig = plt.figure(figsize=(20, 11))
gs  = fig.add_gridspec(1, 3, width_ratios=[2.8, 0.95, 0.95], wspace=0.08)
ax     = fig.add_subplot(gs[0])   # volcano
ax_up  = fig.add_subplot(gs[1])   # table TOP UP
ax_dn  = fig.add_subplot(gs[2])   # table TOP DOWN

for a in [ax_up, ax_dn]:
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.axis("off")

# ── Volcano : NS
ns_d = res[res.status == "NS"]
ax.scatter(ns_d.logFC, ns_d["-log10_padj"], s=3, color="#D5D8DC", alpha=0.35, rasterized=True, zorder=1)

# UP / DOWN non-annotés
up_o = res[(res.status=="UP_CKD")  & (~res.names.isin(all_hl))]
dn_o = res[(res.status=="DOWN_CKD") & (~res.names.isin(all_hl))]
ax.scatter(up_o.logFC, up_o["-log10_padj"], s=8,  color="#FADBD8", alpha=0.65, rasterized=True, zorder=2)
ax.scatter(dn_o.logFC, dn_o["-log10_padj"], s=8,  color="#D6EAF8", alpha=0.65, rasterized=True, zorder=2)

# Gènes annotés
texts = []
for cat, genes in genes_highlight.items():
    col = cat_colors[cat]
    for g in genes:
        row = res[res.names == g]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        size   = 90 if g in key_genes else 42
        zorder = 6 if g in key_genes else 5
        ax.scatter(r.logFC, r["-log10_padj"], s=size, color=col, zorder=zorder,
                   edgecolors="white", linewidths=0.5)
        if abs(r.logFC) > 0.25 or r["-log10_padj"] > 5:
            fw = "bold" if g in key_genes else "normal"
            texts.append(ax.text(r.logFC, r["-log10_padj"], g, fontsize=8.5,
                                 fontweight=fw, color=col, zorder=7))

try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand_points=(1.3, 1.5), force_points=0.3)
except ImportError:
    pass

# Lignes de seuil
ax.axvline(-lfc_thr, color="#95A5A6", ls="--", lw=0.9)
ax.axvline( lfc_thr, color="#95A5A6", ls="--", lw=0.9)
ax.axhline(-np.log10(padj_thr), color="#95A5A6", ls="--", lw=0.9)

n_up = (res.status=="UP_CKD").sum()
n_dn = (res.status=="DOWN_CKD").sum()
ax.text(0.98, 0.97, f"UP in CKD: {n_up}", transform=ax.transAxes,
        ha="right", va="top", fontsize=11, color="#C0392B", fontweight="bold")
ax.text(0.02, 0.97, f"DOWN in CKD: {n_dn}", transform=ax.transAxes,
        ha="left",  va="top", fontsize=11, color="#1A5276", fontweight="bold")
ax.text(0.5, 0.01, f"|LFC| > {lfc_thr}  |  padj < {padj_thr}", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=8.5, color="gray")

ax.set_xlabel("Log2 Fold Change  (CKD / Control)", fontsize=12)
ax.set_ylabel("-log10(adj. p-value)", fontsize=12)
ax.set_title("FOLR2_CKD macrophages : CKD vs Control\n"
             "DKD + HKD + CKD poolés → CKD  |  Wilcoxon test", fontsize=12, fontweight="bold")

# Légende catégories — en dehors du plot, sous le titre
handles = [mpatches.Patch(color=cat_colors[c], label=c.replace("_"," ")) for c in cat_colors]
handles += [mpatches.Patch(color="#FADBD8", label="Other UP (sig.)"),
            mpatches.Patch(color="#D6EAF8", label="Other DOWN (sig.)"),
            mpatches.Patch(color="#D5D8DC", label="NS")]
ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9,
          title="Catégorie", title_fontsize=9, ncol=1,
          bbox_to_anchor=(0.01, 0.01))

# ── Fonction table propre
def draw_table(ax, genes_df, title, title_col, sym, n_rows=18):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Fond titre
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=title_col, alpha=0.12,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.975, title, ha="center", va="center", fontsize=10,
            fontweight="bold", color=title_col, transform=ax.transAxes)
    # Header
    y = 0.935
    ax.text(0.04, y, "Gene",  fontsize=8,   color="#555", transform=ax.transAxes, fontweight="bold")
    ax.text(0.56, y, "LFC",   fontsize=8,   color="#555", transform=ax.transAxes, fontweight="bold")
    ax.text(0.78, y, "padj",  fontsize=8,   color="#555", transform=ax.transAxes, fontweight="bold")
    y -= 0.018
    ax.plot([0.02, 0.98], [y, y], color="#AAAAAA", lw=0.8, transform=ax.transAxes)
    y -= 0.022
    row_h = (y - 0.01) / n_rows
    for i, (_, row) in enumerate(genes_df.head(n_rows).iterrows()):
        cat = all_hl.get(row["names"], "")
        col = cat_colors.get(cat, title_col)
        fw  = "bold" if row["names"] in key_genes else "normal"
        # Fond alterné
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.01, y - row_h*0.15), 0.98, row_h,
                                        color="#F8F9FA", transform=ax.transAxes, zorder=0))
        ax.text(0.04, y, f"{sym} {row['names']}", fontsize=8.5, color=col,
                fontweight=fw, transform=ax.transAxes, va="center")
        ax.text(0.56, y, f"{row.logFC:+.2f}", fontsize=8, color="#555",
                transform=ax.transAxes, va="center")
        ax.text(0.78, y, f"{row.pvals_adj:.0e}", fontsize=7.5, color="#555",
                transform=ax.transAxes, va="center")
        y -= row_h

top_up = res[res.status=="UP_CKD"].sort_values("scores",  ascending=False).head(18)
top_dn = res[res.status=="DOWN_CKD"].sort_values("scores").head(18)

draw_table(ax_up, top_up, "▲  TOP UP in CKD",   "#C0392B", "▲", n_rows=18)
draw_table(ax_dn, top_dn, "▼  TOP DOWN in CKD", "#1A5276", "▼", n_rows=18)

# Séparateur vertical entre tables
fig.add_artist(plt.Line2D([gs[1].get_position(fig).x0 - 0.005,
                            gs[1].get_position(fig).x0 - 0.005],
                           [0.05, 0.95], transform=fig.transFigure,
                           color="#CCCCCC", lw=0.8))

plt.suptitle("Volcano plot — Phénotype pathologique FOLR2_CKD en CKD\n"
             "n_CKD = 1 978 cellules / 33 patients   |   n_Control = 963 cellules / 38 patients",
             fontsize=13, fontweight="bold", y=1.01)

outfile = os.path.join(out, "volcano_FOLR2CKD_CKD_vs_Control.png")
plt.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {outfile}")

print("[4/4] Summary")
print(f"\nUP in CKD : {n_up} genes  |  DOWN in CKD : {n_dn} genes")
print()

for cat, genes in genes_highlight.items():
    sub = res[res.names.isin(genes)].sort_values("logFC", ascending=False)
    print(f"[{cat}]")
    print(sub[["names","logFC","pvals_adj","status"]].to_string(index=False))
    print()
