"""
Export adata_spatial_final.h5ad → Matrix Market format pour CellChat (R)
- Type cellulaire = dominant par cell2location
- Condition CKD (CKD + DKD + DKD+FSGS) uniquement
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import anndata as ad
from pathlib import Path

out = Path("cellchat_input")
out.mkdir(exist_ok=True)

print("Loading adata...")
adata = ad.read_h5ad("adata_spatial_final.h5ad")
print(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

# -- Assign dominant cell type from cell2location --------------------------
ct_cols = [
    "B_Cells", "B_Memory", "B_Naive", "Basophile",
    "CD14_Mono", "CD16_Mono", "CD4_Activated", "CD4_Trm", "CD4_signaling",
    "CD8_MAIT", "CD8_central_memory", "CD8_cytotoxic/effector_memory",
    "FOLR2+_resident", "FOLR2_CKD", "NK/T_cells", "NK_cytotoxic",
    "Neutro_FPR2+", "Plasma_cells", "TREM2+_macro", "cDC", "pDC",
]

abund = adata.obs[ct_cols].values.astype(float)
dominant_idx = np.argmax(abund, axis=1)
adata.obs["celltype_c2l"] = np.array(ct_cols)[dominant_idx]

# -- Filter: CKD condition only --------------------------------------------
ckd_conds = {"CKD", "DKD", "DKD+FSGS"}
mask = adata.obs["Condition"].isin(ckd_conds)
adata_ckd = adata[mask].copy()
print(f"\nCKD subset: {adata_ckd.shape[0]:,} cells")
print("Cell type distribution:")
print(adata_ckd.obs["celltype_c2l"].value_counts().to_string())

# -- Use raw counts --------------------------------------------------------
X = adata_ckd.layers["counts"]
if not sp.issparse(X):
    X = sp.csr_matrix(X)
X = X.T  # genes x cells (CellChat format)

print(f"\nMatrix shape (genes x cells): {X.shape}")

# -- Save Matrix Market ----------------------------------------------------
print("Writing matrix.mtx ...")
scipy.io.mmwrite(str(out / "matrix.mtx"), X)

print("Writing barcodes.tsv ...")
pd.Series(adata_ckd.obs_names).to_csv(out / "barcodes.tsv", index=False, header=False)

print("Writing features.tsv ...")
pd.Series(adata_ckd.var_names).to_csv(out / "features.tsv", index=False, header=False)

print("Writing metadata.csv ...")
meta = adata_ckd.obs[["celltype_c2l", "Condition", "orig_ident",
                        "immune_ME", "niches_annotation_based",
                        "tech", "Sex", "Age"]].copy()
meta.to_csv(out / "metadata.csv")

print(f"\nDone. Files in {out}/")
print("  matrix.mtx")
print("  barcodes.tsv")
print("  features.tsv")
print("  metadata.csv")
