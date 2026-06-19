"""
Violin plots — Top gènes UP et DOWN par catégorie
FOLR2_CKD : CKD vs Control
"""
import anndata as ad, pandas as pd, numpy as np, scanpy as sc
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")

# ── Paths
PATH = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
DE   = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\DE_FOLR2CKD_CKD_vs_Control.csv"
OUT  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication"

# ── Load
print("[1/3] Loading data ...")
adata = ad.read_h5ad(PATH)
adata.obs["annot_final2"]    = adata.obs["annot_final2"].astype(str)
adata.obs["condition_group"] = adata.obs["diseasetype"].map(
    lambda d: "CKD" if d in ("DKD","HKD","CKD") else ("Control" if d=="Control" else "Other")
).astype(str)

f2 = adata[(adata.obs["annot_final2"]=="FOLR2_CKD") &
           (adata.obs["condition_group"].isin(["CKD","Control"]))].copy()
f2.X = f2.layers["scvi_normalized"].copy()
genes_all = list(f2.var.index)

de = pd.read_csv(DE)

# ── Gene categories (top sig. per category)
cat_colors = {
    "Inflammation":  "#E74C3C",
    "Fibrosis":      "#E67E22",
    "Lipid_Ox":      "#8E44AD",
    "CCL8_axis":     "#27AE60",
    "Resident_lost": "#2980B9",
}

genes_highlight = {
    "Inflammation":  ["S100A2","SAA1","S100A8","S100A9","S100A4","CCL20",
                      "CXCL9","CXCL10","IL1B","CCL2","CCL8"],
    "Fibrosis":      ["APOE","SPP1","CHI3L1","LGALS3","LPL","FABP5",
                      "MMP19","VEGFA","TREM2","GPNMB"],
    "Lipid_Ox":      ["APOC1","FABP4","OLR1","CD36","LDLR","PLIN2","ABCA1"],
    "CCL8_axis":     ["CCL13","CCL8","CCR1","CCR2","CCL7"],
    "Resident_lost": ["MRC1","MAF","C1QA","C1QB","F13A1","NR4A1",
                      "STAB1","FOLR2","LYVE1","CD163","TIMD4"],
}

# Select top 6 UP + top 4 DOWN per category (present in data + DE)
def pick_genes(cat_genes, de_df, n_up=6, n_dn=4):
    present = [g for g in cat_genes if g in genes_all]
    sub = de_df[de_df.names.isin(present)].copy()
    up  = sub[sub.status=="UP_CKD"].sort_values("scores",  ascending=False).head(n_up)
    dn  = sub[sub.status=="DOWN_CKD"].sort_values("scores").head(n_dn)
    ns  = sub[sub.status=="NS"].sort_values("logFC", ascending=False).head(2)
    return pd.concat([up, ns, dn]).drop_duplicates("names")

palette = {"CKD": "#C0392B", "Control": "#2980B9"}
order   = ["Control", "CKD"]

# ── Helper: get expression vector
def get_expr(adata_sub, gene):
    idx = list(adata_sub.var.index).index(gene)
    x = adata_sub.X[:, idx]
    if issparse(x): x = x.toarray().flatten()
    return x.astype(float)

# ── Helper: significance stars
def stars(p):
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"

# ── Helper: draw one violin
def draw_vln(ax, expr_ctrl, expr_ckd, gene, de_row, col_cat, is_up):
    data = {"Control": expr_ctrl, "CKD": expr_ckd}
    positions = [0, 1]
    parts = ax.violinplot([data[g] for g in order], positions=positions,
                          showmedians=True, showextrema=False, widths=0.65)
    for pc, grp in zip(parts["bodies"], order):
        pc.set_facecolor(palette[grp])
        pc.set_alpha(0.75)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.4)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)

    # Jitter
    for i, grp in enumerate(order):
        vals = data[grp]
        np.random.seed(42)
        jit  = np.random.normal(i, 0.06, size=min(150, len(vals)))
        samp = np.random.choice(vals, size=min(150, len(vals)), replace=False)
        ax.scatter(jit, samp, s=2.5, color=palette[grp], alpha=0.35, zorder=3)

    # Stat bar
    _, p = mannwhitneyu(expr_ctrl, expr_ckd, alternative="two-sided")
    y_max = max(np.percentile(expr_ctrl, 97), np.percentile(expr_ckd, 97))
    y_bar = y_max * 1.12
    ax.plot([0, 1], [y_bar, y_bar], "k-", lw=0.9)
    ax.text(0.5, y_bar * 1.04, stars(p), ha="center", va="bottom", fontsize=9,
            color="black" if stars(p) != "ns" else "gray")

    # Gene label
    lfc = de_row["logFC"].values[0] if len(de_row) else 0
    direction = "▲" if lfc > 0 else ("▼" if lfc < 0 else "—")
    lfc_str = f"{direction} LFC={lfc:+.2f}" if len(de_row) else ""
    fw = "bold" if abs(lfc) >= 0.5 else "normal"
    ax.set_title(f"{gene}\n{lfc_str}", fontsize=8.5, fontweight=fw, color=col_cat, pad=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(order, fontsize=7.5, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(bottom=-0.05)

# ── Main figure: one row per category
print("[2/3] Drawing violin plots ...")

n_cats = len(genes_highlight)
max_genes_per_cat = 10

fig = plt.figure(figsize=(22, n_cats * 3.2))
outer = gridspec.GridSpec(n_cats, 1, figure=fig, hspace=0.55)

for row_idx, (cat, cat_genes) in enumerate(genes_highlight.items()):
    col = cat_colors[cat]
    sel = pick_genes(cat_genes, de, n_up=6, n_dn=4)
    sel_genes = list(sel["names"])
    n_g = len(sel_genes)

    # Inner grid for this category
    inner = gridspec.GridSpecFromSubplotSpec(
        1, max_genes_per_cat, subplot_spec=outer[row_idx], wspace=0.35
    )

    # Category label on left
    ax_label = fig.add_subplot(inner[0, 0])
    ax_label.axis("off")
    ax_label.text(0.5, 0.5, cat.replace("_", "\n"), ha="center", va="center",
                  fontsize=11, fontweight="bold", color=col,
                  transform=ax_label.transAxes,
                  bbox=dict(boxstyle="round,pad=0.4", fc=col, alpha=0.12, ec=col, lw=1.2))

    # Category separator line
    fig.add_artist(plt.Line2D(
        [outer[row_idx].get_position(fig).x0,
         outer[row_idx].get_position(fig).x1],
        [outer[row_idx].get_position(fig).y1 + 0.003] * 2,
        transform=fig.transFigure, color=col, lw=1.5, alpha=0.6
    ))

    for col_idx, gene in enumerate(sel_genes[:max_genes_per_cat - 1], start=1):
        if gene not in genes_all:
            continue
        ax = fig.add_subplot(inner[0, col_idx])
        expr_ctrl = get_expr(f2[f2.obs["condition_group"]=="Control"], gene)
        expr_ckd  = get_expr(f2[f2.obs["condition_group"]=="CKD"],     gene)
        de_row    = de[de.names == gene]
        is_up     = (de_row["status"].values[0] == "UP_CKD") if len(de_row) else None
        draw_vln(ax, expr_ctrl, expr_ckd, gene, de_row, col, is_up)

        # Shade UP/DOWN background lightly
        if len(de_row) and de_row["status"].values[0] == "UP_CKD":
            ax.set_facecolor("#FEF9F9")
        elif len(de_row) and de_row["status"].values[0] == "DOWN_CKD":
            ax.set_facecolor("#F5F8FE")

    # Fill empty slots
    for empty in range(len(sel_genes), max_genes_per_cat):
        fig.add_subplot(inner[0, empty]).axis("off")

# ── Global legend
legend_handles = [
    mpatches.Patch(color=palette["Control"], label="Control (n=963)", alpha=0.8),
    mpatches.Patch(color=palette["CKD"],     label="CKD poolé (n=1978)", alpha=0.8),
    mpatches.Patch(color="#FEF9F9", label="UP in CKD", ec="#E74C3C", lw=1),
    mpatches.Patch(color="#F5F8FE", label="DOWN in CKD", ec="#2980B9", lw=1),
]
fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.995),
           fontsize=9.5, framealpha=0.9, title="Condition / Direction",
           title_fontsize=10, ncol=2)

fig.suptitle(
    "Expression des gènes différentiels par catégorie — FOLR2_CKD macrophages\n"
    "CKD (DKD+HKD+CKD poolés) vs Control  |  fond rose = UP in CKD  |  fond bleu = DOWN in CKD",
    fontsize=13, fontweight="bold", y=1.005
)

outfile = os.path.join(OUT, "vlnplot_FOLR2CKD_categories.png")
plt.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {outfile}")
print("[3/3] Done.")
