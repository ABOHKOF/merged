"""
Export integrated.h5ad (data1, DiseaseID.x) → Matrix Market pour CellChat comparatif
Deux dossiers : cellchat_input_CKD/  et  cellchat_input_Ctrl/
Annotation : annot_atlas (35 types), raw counts depuis layers['counts']
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import anndata as ad
from pathlib import Path

H5AD = r"C:\Users\RICHMOND\Documents\Richmond\singlecell_rein\script_exemple\data2\integrated.h5ad"

print("Chargement integrated.h5ad ...")
adata = ad.read_h5ad(H5AD)
print(f"  {adata.n_obs:,} cellules  x  {adata.n_vars} gènes")

# ── Sous-ensemble data1 (CDm/CDp, DiseaseID.x disponible) ──────────────────
mask_data1 = adata.obs["dataset"] == "data1"
adata_d1 = adata[mask_data1].copy()
print(f"\ndata1 subset : {adata_d1.n_obs:,} cellules")
print("Conditions :")
print(adata_d1.obs["DiseaseID.x"].value_counts().to_string())

# ── Vérification de la couche counts ────────────────────────────────────────
if "counts" not in adata_d1.layers:
    raise KeyError("Layer 'counts' absent — vérifier integrated.h5ad")

# ── Filtrage : garder types avec ≥ 10 cellules dans CHAQUE condition ────────
ct_col = "annot_atlas"
cond_col = "DiseaseID.x"

ct_counts = adata_d1.obs.groupby([ct_col, cond_col]).size().unstack(fill_value=0)
valid_types = ct_counts[(ct_counts.get("CKD", 0) >= 10) &
                         (ct_counts.get("control", 0) >= 10)].index.tolist()

n_before = adata_d1.obs[ct_col].nunique()
print(f"\nTypes cellulaires présents : {n_before}")
print(f"Types gardés (≥10 cellules dans CKD ET contrôle) : {len(valid_types)}")
excluded = sorted(set(adata_d1.obs[ct_col].unique()) - set(valid_types))
if excluded:
    print(f"Exclus : {', '.join(excluded)}")

adata_d1 = adata_d1[adata_d1.obs[ct_col].isin(valid_types)].copy()
print(f"Cellules après filtrage : {adata_d1.n_obs:,}")

# ── Export par condition ─────────────────────────────────────────────────────
for cond, label in [("CKD", "CKD"), ("control", "Ctrl")]:
    out_dir = Path(f"cellchat_input_{label}")
    out_dir.mkdir(exist_ok=True)

    sub = adata_d1[adata_d1.obs[cond_col] == cond].copy()
    print(f"\n── {label}: {sub.n_obs:,} cellules")
    print(sub.obs[ct_col].value_counts().to_string())

    X = sub.layers["counts"]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_t = X.T.tocsc()   # genes × cells

    print(f"  Matrix (genes × cells) : {X_t.shape}")

    scipy.io.mmwrite(str(out_dir / "matrix.mtx"), X_t)

    pd.Series(sub.obs_names).to_csv(out_dir / "barcodes.tsv", index=False, header=False)
    pd.Series(sub.var_names).to_csv(out_dir / "features.tsv",  index=False, header=False)

    meta_cols = [ct_col, cond_col, "patientID"]
    avail = [c for c in meta_cols if c in sub.obs.columns]
    sub.obs[avail].to_csv(out_dir / "metadata.csv")

    print(f"  Sauvegardé dans {out_dir}/")

print("\nExport terminé.")
print("Prochaine étape : lancer cellchat_CKDvsCtrl.R")
