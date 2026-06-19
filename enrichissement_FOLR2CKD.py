"""
Enrichissement fonctionnel — FOLR2_CKD : CKD vs Control
GO Biological Process, KEGG, Reactome via gseapy (Enrichr)
Séparé : UP in CKD (pathologique) / DOWN in CKD (homéostatique perdu)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gseapy as gp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DE_PATH = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\DE_FOLR2CKD_CKD_vs_Control.csv"
OUT     = Path(r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\xemuin\cell2location_output\cell_communication\enrichissement")
OUT.mkdir(exist_ok=True)

# ── 1. Charger DEG ───────────────────────────────────────────────────────────
print("Chargement DEG...")
de = pd.read_csv(DE_PATH)
print(f"  Total gènes: {len(de)} | UP_CKD: {(de.status=='UP_CKD').sum()} | DOWN_CKD: {(de.status=='DOWN_CKD').sum()}")

genes_up   = de[de.status == "UP_CKD"]["names"].tolist()
genes_down = de[de.status == "DOWN_CKD"]["names"].tolist()
genes_all  = de["names"].tolist()   # background

# ── Bases de données à interroger ────────────────────────────────────────────
DATABASES = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

# ── 2. Fonction enrichissement + figures ─────────────────────────────────────
def run_enrichr(gene_list, label, color, n_top=20):
    print(f"\n── Enrichissement {label} ({len(gene_list)} gènes)...")
    results = {}
    for db in DATABASES:
        try:
            enr = gp.enrichr(
                gene_list   = gene_list,
                gene_sets   = db,
                background  = genes_all,
                organism    = "human",
                outdir      = None,
                verbose     = False,
            )
            df = enr.results
            df = df[df["Adjusted P-value"] < 0.05].copy()
            df = df.sort_values("Adjusted P-value")
            if len(df) > 0:
                results[db] = df
                print(f"  {db}: {len(df)} termes sig.")
            else:
                print(f"  {db}: aucun terme sig.")
        except Exception as e:
            print(f"  {db}: erreur — {e}")

    if not results:
        print(f"  Aucun enrichissement significatif pour {label}")
        return results

    # ── Sauvegarder tables ────────────────────────────────────────────────────
    for db, df in results.items():
        fname = OUT / f"enrichr_{label}_{db.replace(' ','_')}.csv"
        df.to_csv(fname, index=False)

    # ── Figure 1 : dotplot multi-bases ───────────────────────────────────────
    combined = []
    for db, df in results.items():
        top = df.head(8).copy()
        top["Database"] = db.split("_")[0]
        top["Term_short"] = top["Term"].str[:60]
        top["-log10_padj"] = -np.log10(top["Adjusted P-value"].clip(1e-300))
        if "Overlap" in top.columns:
            top["GeneRatio"] = top["Overlap"].apply(
                lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) if "/" in str(x) else np.nan
            )
        else:
            top["GeneRatio"] = np.nan
        combined.append(top)

    if combined:
        comb_df = pd.concat(combined, ignore_index=True)
        comb_df = comb_df.sort_values(["-log10_padj"], ascending=False).head(n_top * len(results))

        fig, ax = plt.subplots(figsize=(14, max(8, len(comb_df) * 0.38)))
        db_palette = {
            "GO": "#2980B9", "KEGG": "#E67E22",
            "Reactome": "#27AE60", "MSigDB": "#8E44AD",
        }
        scatter = ax.scatter(
            comb_df["-log10_padj"],
            range(len(comb_df)),
            s=comb_df["GeneRatio"] * 800,
            c=[db_palette.get(r["Database"], "#95A5A6") for _, r in comb_df.iterrows()],
            alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3
        )
        ax.set_yticks(range(len(comb_df)))
        ax.set_yticklabels(comb_df["Term_short"], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(padj)", fontsize=11)
        ax.set_title(f"Enrichissement fonctionnel — {label}\n(taille = GeneRatio, couleur = base de données)",
                     fontsize=12, fontweight="bold")
        ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
        ax.grid(axis="x", alpha=0.3)

        handles = [mpatches.Patch(color=c, label=db)
                   for db, c in db_palette.items()]
        ax.legend(handles=handles, loc="lower right", fontsize=9, title="Base")

        fig.tight_layout()
        fig.savefig(OUT / f"01_dotplot_enrichr_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved 01_dotplot_enrichr_{label}.png")

    # ── Figure 2 : barplot GO BP top 20 ──────────────────────────────────────
    if "GO_Biological_Process_2023" in results:
        go_df = results["GO_Biological_Process_2023"].head(n_top).copy()
        go_df["Term_short"] = go_df["Term"].str[:55]
        go_df["-log10_padj"] = -np.log10(go_df["Adjusted P-value"].clip(1e-300))
        if "Overlap" in go_df.columns:
            go_df["GeneRatio"] = go_df["Overlap"].apply(
                lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) if "/" in str(x) else np.nan
            )
        else:
            go_df["GeneRatio"] = np.nan

        fig, ax = plt.subplots(figsize=(12, max(6, len(go_df) * 0.42)))
        bars = ax.barh(range(len(go_df)), go_df["-log10_padj"],
                       color=color, alpha=0.75, edgecolor="white")
        ax.set_yticks(range(len(go_df)))
        ax.set_yticklabels(go_df["Term_short"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(padj)", fontsize=11)
        ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
        ax.set_title(f"GO Biological Process — Top {n_top} | {label}",
                     fontsize=12, fontweight="bold")

        sc = ax.twinx()
        sc.scatter(go_df["-log10_padj"], range(len(go_df)),
                   s=go_df["GeneRatio"] * 600, color="black", alpha=0.5, zorder=5)
        sc.set_ylim(ax.get_ylim())
        sc.set_yticks([])
        sc.set_ylabel("GeneRatio (●)", fontsize=9, color="gray")

        fig.tight_layout()
        fig.savefig(OUT / f"02_GOBP_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved 02_GOBP_{label}.png")

    # ── Figure 3 : barplot KEGG top 15 ───────────────────────────────────────
    if "KEGG_2021_Human" in results:
        kegg_df = results["KEGG_2021_Human"].head(15).copy()
        kegg_df["Term_short"] = kegg_df["Term"].str[:55]
        kegg_df["-log10_padj"] = -np.log10(kegg_df["Adjusted P-value"].clip(1e-300))

        fig, ax = plt.subplots(figsize=(11, max(5, len(kegg_df) * 0.42)))
        ax.barh(range(len(kegg_df)), kegg_df["-log10_padj"],
                color="#E67E22", alpha=0.75, edgecolor="white")
        ax.set_yticks(range(len(kegg_df)))
        ax.set_yticklabels(kegg_df["Term_short"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(padj)", fontsize=11)
        ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
        ax.set_title(f"KEGG Pathways — Top 15 | {label}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"03_KEGG_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved 03_KEGG_{label}.png")

    # ── Figure 4 : Reactome top 15 ───────────────────────────────────────────
    if "Reactome_2022" in results:
        r_df = results["Reactome_2022"].head(15).copy()
        r_df["Term_short"] = r_df["Term"].str[:60]
        r_df["-log10_padj"] = -np.log10(r_df["Adjusted P-value"].clip(1e-300))

        fig, ax = plt.subplots(figsize=(12, max(5, len(r_df) * 0.42)))
        ax.barh(range(len(r_df)), r_df["-log10_padj"],
                color="#27AE60", alpha=0.75, edgecolor="white")
        ax.set_yticks(range(len(r_df)))
        ax.set_yticklabels(r_df["Term_short"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(padj)", fontsize=11)
        ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
        ax.set_title(f"Reactome Pathways — Top 15 | {label}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"04_Reactome_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved 04_Reactome_{label}.png")

    # ── Figure 5 : Hallmark top 15 ───────────────────────────────────────────
    if "MSigDB_Hallmark_2020" in results:
        h_df = results["MSigDB_Hallmark_2020"].head(15).copy()
        h_df["Term_short"] = h_df["Term"].str.replace("HALLMARK_","").str[:55]
        h_df["-log10_padj"] = -np.log10(h_df["Adjusted P-value"].clip(1e-300))

        fig, ax = plt.subplots(figsize=(11, max(5, len(h_df) * 0.42)))
        ax.barh(range(len(h_df)), h_df["-log10_padj"],
                color="#8E44AD", alpha=0.75, edgecolor="white")
        ax.set_yticks(range(len(h_df)))
        ax.set_yticklabels(h_df["Term_short"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(padj)", fontsize=11)
        ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
        ax.set_title(f"MSigDB Hallmarks — Top 15 | {label}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"05_Hallmark_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved 05_Hallmark_{label}.png")

    return results

# ── 3. Lancer UP et DOWN ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("ENRICHISSEMENT UP_CKD (signature pathologique)")
print("="*60)
res_up = run_enrichr(genes_up, "UP_CKD", "#C0392B")

print("\n" + "="*60)
print("ENRICHISSEMENT DOWN_CKD (signature homéostatique perdue)")
print("="*60)
res_down = run_enrichr(genes_down, "DOWN_CKD", "#2980B9")

# ── 4. Figure comparative : GO BP UP vs DOWN ─────────────────────────────────
print("\nFigure comparative UP vs DOWN...")
if "GO_Biological_Process_2023" in res_up and "GO_Biological_Process_2023" in res_down:
    up_top  = res_up["GO_Biological_Process_2023"].head(15).copy()
    dn_top  = res_down["GO_Biological_Process_2023"].head(15).copy()

    up_top["lfc_dir"] = -np.log10(up_top["Adjusted P-value"].clip(1e-300))
    dn_top["lfc_dir"] = np.log10(dn_top["Adjusted P-value"].clip(1e-300))   # négatif = DOWN

    up_top["Term_short"] = up_top["Term"].str[:50]
    dn_top["Term_short"] = dn_top["Term"].str[:50]

    combined = pd.concat([
        up_top[["Term_short","lfc_dir"]].assign(side="UP CKD"),
        dn_top[["Term_short","lfc_dir"]].assign(side="DOWN CKD"),
    ])
    combined = combined.sort_values("lfc_dir")

    fig, ax = plt.subplots(figsize=(13, max(8, len(combined) * 0.35)))
    colors = ["#2980B9" if v < 0 else "#C0392B" for v in combined["lfc_dir"]]
    ax.barh(range(len(combined)), combined["lfc_dir"], color=colors, alpha=0.75, edgecolor="white")
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined["Term_short"], fontsize=8.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline( np.log10(0.05),  color="gray", ls="--", lw=0.7)
    ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.7)
    ax.set_xlabel("← -log10(padj) DOWN CKD   |   +log10(padj) UP CKD →", fontsize=10)
    ax.set_title("GO Biological Process — Enrichissement différentiel\nUP CKD (rouge) vs DOWN CKD / homéostatique (bleu)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "06_GOBP_UP_vs_DOWN_comparative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 06_GOBP_UP_vs_DOWN_comparative.png")

print(f"\n=== Enrichissement terminé. Fichiers dans {OUT} ===")
