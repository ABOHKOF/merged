"""
DEG FOLR2_CKD CKD (malade) vs FOLR2_CKD Control (contrôle)
Heatmap + Dotplot + Violin des top marqueurs de chaque état
"""
import anndata as ad, pandas as pd, numpy as np, scanpy as sc
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

PATH = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\adata_immune_annot_final.h5ad"
DE   = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\DE_FOLR2CKD_CKD_vs_Control.csv"
OUT  = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication"

# ── 1. Load & subset
print("[1/5] Loading ...")
adata = ad.read_h5ad(PATH)
adata.obs["annot_final2"]    = adata.obs["annot_final2"].astype(str)
adata.obs["condition_group"] = adata.obs["diseasetype"].map(
    lambda d: "CKD" if d in ("DKD","HKD","CKD") else ("Control" if d=="Control" else "Other")
).astype(str)

f2 = adata[(adata.obs["annot_final2"]=="FOLR2_CKD") &
           (adata.obs["condition_group"].isin(["CKD","Control"]))].copy()
f2.X = f2.layers["scvi_normalized"].copy()
f2.obs["condition_group"] = pd.Categorical(f2.obs["condition_group"], categories=["Control","CKD"])
print(f"  FOLR2_CKD: CKD={( f2.obs.condition_group=='CKD').sum()}  Control={( f2.obs.condition_group=='Control').sum()}")

de = pd.read_csv(DE)

# ── 2. Top gènes par état
print("[2/5] Selecting top genes ...")
N_TOP = 25

# Top UP in CKD = signature pathologique
top_ckd  = de[de.status == "UP_CKD"].sort_values("scores", ascending=False).head(N_TOP)
# Top UP in Control = signature homéostatique (= DOWN in CKD sorted by most negative score)
top_ctrl = de[de.status == "DOWN_CKD"].sort_values("scores").head(N_TOP)

genes_ckd  = list(top_ckd["names"])
genes_ctrl = list(top_ctrl["names"])
all_genes  = genes_ckd + genes_ctrl  # ordered: CKD markers first, then Control markers

# Filter to genes present in var
all_genes = [g for g in all_genes if g in f2.var.index]
genes_ckd  = [g for g in genes_ckd  if g in f2.var.index]
genes_ctrl = [g for g in genes_ctrl if g in f2.var.index]

print(f"  Genes CKD signature: {len(genes_ckd)}")
print(f"  Genes Control signature: {len(genes_ctrl)}")

# ── 3. Expression matrix per cell (for heatmap)
def get_expr_matrix(adata_sub, genes):
    idx = [list(adata_sub.var.index).index(g) for g in genes]
    X = adata_sub.X[:, idx]
    if issparse(X): X = X.toarray()
    return pd.DataFrame(X.astype(float), columns=genes, index=adata_sub.obs.index)

# ── 4. Mean expression per condition
def mean_expr(adata_sub, genes):
    idx = [list(adata_sub.var.index).index(g) for g in genes]
    X = adata_sub.X[:, idx]
    if issparse(X): X = X.toarray()
    return pd.DataFrame(X.astype(float), columns=genes).mean(axis=0)

mean_ckd  = mean_expr(f2[f2.obs.condition_group=="CKD"],     all_genes)
mean_ctrl = mean_expr(f2[f2.obs.condition_group=="Control"], all_genes)
pct_ckd   = de.set_index("names")["pct_ckd"].reindex(all_genes)
pct_ctrl  = de.set_index("names")["pct_ctrl"].reindex(all_genes)
lfc       = de.set_index("names")["logFC"].reindex(all_genes)
padj      = de.set_index("names")["pvals_adj"].reindex(all_genes)

# Stats
def stars(p):
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"

# ── 5. Build figures
print("[3/5] Figure 1 — Heatmap mean expression ...")

# ── Fig 1: Heatmap double (mean expression scaled)
heat_df = pd.DataFrame({"Control": mean_ctrl, "CKD": mean_ckd})
# Z-score par gène
heat_z = heat_df.apply(lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1)

fig1, axes = plt.subplots(1, 2, figsize=(14, max(10, len(all_genes)*0.32)),
                           gridspec_kw={"width_ratios": [1, 3], "wspace": 0.05})

# Left: heatmap z-scored
ax_heat = axes[0]
sns.heatmap(heat_z, ax=ax_heat, cmap="RdBu_r", center=0,
            linewidths=0.3, linecolor="#EEEEEE",
            cbar_kws={"label": "Z-score", "shrink": 0.6},
            yticklabels=True, xticklabels=True)
ax_heat.set_title("Expression\n(Z-score par gène)", fontsize=10, fontweight="bold")
ax_heat.set_xlabel("")
ax_heat.tick_params(axis="y", labelsize=8)

# Color ytick labels by gene group
n_ckd  = len(genes_ckd)
n_ctrl = len(genes_ctrl)
for i, tick in enumerate(ax_heat.get_yticklabels()):
    tick.set_color("#C0392B" if i < n_ckd else "#1A5276")
    if i < n_ckd:
        tick.set_fontweight("bold")

# Separator line between CKD and Control marker blocks
ax_heat.axhline(n_ckd, color="black", lw=1.5, ls="--")

# Right: LFC bar + pct dots
ax_bar = axes[1]
y_pos  = np.arange(len(all_genes))
colors_bar = ["#C0392B" if lfc.iloc[i] > 0 else "#1A5276" for i in range(len(all_genes))]
bars = ax_bar.barh(y_pos, lfc.values, color=colors_bar, edgecolor="white",
                   height=0.6, alpha=0.75)
ax_bar.axvline(0, color="black", lw=0.8)
ax_bar.axvline( 0.5, color="#AAAAAA", lw=0.6, ls="--")
ax_bar.axvline(-0.5, color="#AAAAAA", lw=0.6, ls="--")
ax_bar.axhline(n_ckd - 0.5, color="black", lw=1.5, ls="--")

# pct expressed as scatter overlay
ax_pct = ax_bar.twiny()
ax_pct.scatter(pct_ckd.values * 100,  y_pos + 0.15, s=30, color="#C0392B",
               alpha=0.7, zorder=5, label="% CKD")
ax_pct.scatter(pct_ctrl.values * 100, y_pos - 0.15, s=30, color="#1A5276",
               alpha=0.7, zorder=5, label="% Control", marker="D")
ax_pct.set_xlabel("% cellules exprimant le gène", fontsize=8.5, color="gray")
ax_pct.tick_params(axis="x", labelsize=7.5, colors="gray")
ax_pct.legend(loc="lower right", fontsize=7.5, framealpha=0.8)

# Stars
for i, g in enumerate(all_genes):
    p = padj.loc[g] if g in padj.index else 1
    s = stars(p)
    if s != "ns":
        x = lfc.loc[g] if g in lfc.index else 0
        ax_bar.text(x + (0.1 if x >= 0 else -0.1), i, s, va="center",
                    ha="left" if x >= 0 else "right", fontsize=7, color="black")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(all_genes, fontsize=8)
for i, tick in enumerate(ax_bar.get_yticklabels()):
    tick.set_color("#C0392B" if i < n_ckd else "#1A5276")
ax_bar.set_xlabel("Log2 Fold Change (CKD / Control)", fontsize=9)
ax_bar.set_title("LFC  +  % exprimé (● CKD  ◆ Control)", fontsize=10, fontweight="bold")
ax_bar.invert_yaxis()

# Group annotations
ax_bar.text(ax_bar.get_xlim()[1]*0.98, n_ckd/2, "SIGNATURE\nCKD", va="center", ha="right",
            fontsize=9, color="#C0392B", fontweight="bold", rotation=90,
            bbox=dict(boxstyle="round", fc="#FDECEA", alpha=0.7))
ax_bar.text(ax_bar.get_xlim()[1]*0.98, n_ckd + n_ctrl/2, "SIGNATURE\nCONTROL", va="center",
            ha="right", fontsize=9, color="#1A5276", fontweight="bold", rotation=90,
            bbox=dict(boxstyle="round", fc="#EBF5FB", alpha=0.7))

fig1.suptitle(
    "DEG — FOLR2_CKD : Signature pathologique (CKD) vs Signature homéostatique (Control)\n"
    f"Top {N_TOP} UP in CKD  |  Top {N_TOP} UP in Control  |  Wilcoxon test",
    fontsize=12, fontweight="bold", y=1.01
)
fig1.savefig(os.path.join(OUT, "DEG_FOLR2CKD_heatmap_LFC.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  Fig 1 saved.")

# ── Fig 2 : Scanpy dotplot
print("[4/5] Figure 2 — Dotplot scanpy ...")

# Subset for dotplot (top 15 per group for readability)
genes_dot = [g for g in genes_ckd[:15] if g in f2.var.index] + \
            [g for g in genes_ctrl[:15] if g in f2.var.index]

f2_dot = f2.copy()
f2_dot.obs["condition_group"] = pd.Categorical(
    f2_dot.obs["condition_group"], categories=["Control","CKD"])

sc.settings.figdir = OUT
sc.set_figure_params(dpi=150, figsize=(max(12, len(genes_dot)*0.55), 3.5))

dp = sc.pl.dotplot(
    f2_dot, var_names=genes_dot, groupby="condition_group",
    use_raw=False, standard_scale="var",
    color_map="RdBu_r", show=False, return_fig=True,
    title="Dotplot — FOLR2_CKD CKD vs Control\n(taille = % cellules, couleur = expression normalisée)",
    var_group_positions=[(0, len(genes_ckd[:15])-1), (len(genes_ckd[:15]), len(genes_dot)-1)],
    var_group_labels=["Signature CKD", "Signature Control"],
)
dp.savefig(os.path.join(OUT, "DEG_FOLR2CKD_dotplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Fig 2 saved.")

# ── Fig 3 : Matrixplot (mean expression)
print("[5/5] Figure 3 — Matrixplot scanpy ...")

mp = sc.pl.matrixplot(
    f2_dot, var_names=genes_dot, groupby="condition_group",
    use_raw=False, standard_scale="var",
    cmap="RdBu_r", show=False, return_fig=True,
    title="Matrixplot — expression moyenne par condition\n(FOLR2_CKD CKD vs Control)",
    var_group_positions=[(0, len(genes_ckd[:15])-1), (len(genes_ckd[:15]), len(genes_dot)-1)],
    var_group_labels=["Signature CKD", "Signature Control"],
)
mp.savefig(os.path.join(OUT, "DEG_FOLR2CKD_matrixplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Fig 3 saved.")

# ── Save gene lists
pd.DataFrame({
    "Signature_CKD_top25":     genes_ckd  + [""] * (N_TOP - len(genes_ckd)),
    "Signature_Control_top25": genes_ctrl + [""] * (N_TOP - len(genes_ctrl)),
}).to_csv(os.path.join(OUT, "DEG_FOLR2CKD_signatures.csv"), index=False)

print(f"\n{'='*55}")
print("Outputs:")
print("  DEG_FOLR2CKD_heatmap_LFC.png   — heatmap Z-score + LFC + pct")
print("  DEG_FOLR2CKD_dotplot.png        — dotplot scanpy")
print("  DEG_FOLR2CKD_matrixplot.png     — matrixplot scanpy")
print("  DEG_FOLR2CKD_signatures.csv     — listes de gènes")
print()
print(f"Signature CKD (top 10):     {genes_ckd[:10]}")
print(f"Signature Control (top 10): {genes_ctrl[:10]}")
