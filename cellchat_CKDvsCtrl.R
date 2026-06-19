# CellChat comparatif : CKD vs Contrôle
# Source : integrated.h5ad > data1 (CDm/CDp), annot_atlas, DiseaseID.x
# Requiert : export_cellchat_CKDvsCtrl.py exécuté au préalable

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(ComplexHeatmap)
})

set.seed(42)
options(future.globals.maxSize = 8000 * 1024^2)

out_dir <- "plots/cellchat_CKDvsCtrl"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Fonction : charger un dossier Matrix Market → CellChat ──────────────────
load_cellchat <- function(dir_path, condition_name) {
  cat(sprintf("\n── Chargement %s depuis %s\n", condition_name, dir_path))

  counts <- readMM(file.path(dir_path, "matrix.mtx"))
  genes  <- read.table(file.path(dir_path, "features.tsv"), header = FALSE)$V1
  cells  <- read.table(file.path(dir_path, "barcodes.tsv"), header = FALSE)$V1
  meta   <- read.csv(file.path(dir_path, "metadata.csv"), row.names = 1)

  rownames(counts) <- genes
  colnames(counts) <- cells
  counts <- as(counts, "CsparseMatrix")
  cat(sprintf("  %d gènes x %d cellules\n", nrow(counts), ncol(counts)))
  cat("  Types cellulaires:\n")
  print(sort(table(meta$annot_atlas), decreasing = TRUE))

  cc <- createCellChat(object = counts, meta = meta, group.by = "annot_atlas")
  cc@DB <- CellChatDB.human
  cc <- subsetData(cc)
  future::plan("multisession", workers = 4)
  cc <- identifyOverExpressedGenes(cc)
  cc <- identifyOverExpressedInteractions(cc)
  cat(sprintf("  Paires L-R surexprimées: %d\n", nrow(cc@LR$LRsig)))

  cc <- computeCommunProb(cc, type = "triMean", population.size = TRUE)
  cc <- filterCommunication(cc, min.cells = 10)
  cc <- computeCommunProbPathway(cc)
  cc <- aggregateNet(cc)
  cc <- netAnalysis_computeCentrality(cc, slot.name = "netP")

  saveRDS(cc, file.path(out_dir, paste0("cellchat_", condition_name, ".rds")))
  cat(sprintf("  Sauvegardé: cellchat_%s.rds\n", condition_name))
  return(cc)
}

# ── 1. Créer les deux objets ─────────────────────────────────────────────────
cc_ckd  <- load_cellchat("cellchat_input_CKD",  "CKD")
cc_ctrl <- load_cellchat("cellchat_input_Ctrl", "Ctrl")

# ── 2. Fusionner pour comparaison ────────────────────────────────────────────
cat("\nFusion des objets CellChat...\n")
object.list <- list(Ctrl = cc_ctrl, CKD = cc_ckd)
cc_merged   <- mergeCellChat(object.list, add.names = names(object.list))
saveRDS(cc_merged, file.path(out_dir, "cellchat_merged.rds"))
cat("  Sauvegardé: cellchat_merged.rds\n")

# ── 3. Comparaison globale : nombre et force d'interactions ──────────────────
cat("\nFigures comparatives globales...\n")

# 3a. Bar plots nb/force interactions
tryCatch({
  p1 <- compareInteractions(cc_merged, show.legend = FALSE, group = c(1, 2)) +
    ggtitle("Nombre d'interactions")
  p2 <- compareInteractions(cc_merged, show.legend = FALSE, group = c(1, 2),
                             measure = "weight") +
    ggtitle("Force des interactions")
  ggsave(file.path(out_dir, "01_compare_interactions.png"),
         p1 + p2, width = 10, height = 5, dpi = 150)
  cat("  saved 01_compare_interactions.png\n")
}, error = function(e) cat("  01:", conditionMessage(e), "\n"))

# 3b. Cercle comparatif (différence nb interactions)
tryCatch({
  png(file.path(out_dir, "02_diffInteractions_circle.png"),
      width = 1400, height = 700, res = 130)
  par(mfrow = c(1, 2), mar = c(1, 1, 2, 1))
  netVisual_diffInteraction(cc_merged, weight.scale = TRUE,
                            title.name = "Différence nb (CKD - Ctrl)")
  netVisual_diffInteraction(cc_merged, weight.scale = TRUE, measure = "weight",
                            title.name = "Différence force (CKD - Ctrl)")
  dev.off()
  cat("  saved 02_diffInteractions_circle.png\n")
}, error = function(e) {
  tryCatch(dev.off(), error = function(x) NULL)
  cat("  02:", conditionMessage(e), "\n")
})

# 3c. Heatmap différentielle
tryCatch({
  p1 <- netVisual_heatmap(cc_merged, color.heatmap = "RdBu") +
    ggtitle("Différence nb interactions (CKD - Ctrl)")
  p2 <- netVisual_heatmap(cc_merged, measure = "weight", color.heatmap = "RdBu") +
    ggtitle("Différence force (CKD - Ctrl)")
  ggsave(file.path(out_dir, "03_heatmap_diff.png"),
         p1 + p2, width = 22, height = 9, dpi = 150)
  cat("  saved 03_heatmap_diff.png\n")
}, error = function(e) cat("  03:", conditionMessage(e), "\n"))

# ── 4. Analyse différentielle des voies de signalisation ─────────────────────
cat("\nAnalyse des voies...\n")

# 4a. Dot plot : voies partagées / spécifiques
tryCatch({
  rankNet_ckd  <- rankNet(cc_ckd,  mode = "comparison", stacked = TRUE,
                          do.stat = TRUE)
  rankNet_ctrl <- rankNet(cc_ctrl, mode = "comparison", stacked = TRUE,
                          do.stat = TRUE)
}, error = function(e) cat("  rankNet:", conditionMessage(e), "\n"))

tryCatch({
  p <- rankNet(cc_merged, mode = "comparison", stacked = TRUE, do.stat = TRUE) +
    scale_fill_manual(values = c("Ctrl" = "#4393c3", "CKD" = "#d6604d")) +
    ggtitle("Force des voies de signalisation (Ctrl vs CKD)")
  ggsave(file.path(out_dir, "04_rankNet_pathways.png"),
         p, width = 12, height = 14, dpi = 150)
  cat("  saved 04_rankNet_pathways.png\n")
}, error = function(e) cat("  04:", conditionMessage(e), "\n"))

# 4b. Scatter outgoing vs incoming par type cellulaire
tryCatch({
  num_link <- sapply(object.list, function(x) rowSums(x@net$count) + colSums(x@net$count) - diag(x@net$count))
  weight_link <- sapply(object.list, function(x) rowSums(x@net$weight) + colSums(x@net$weight) - diag(x@net$weight))

  p <- netAnalysis_signalingRole_scatter(cc_merged) +
    ggtitle("Rôle sender/receiver : Ctrl (bleu) vs CKD (rouge)")
  ggsave(file.path(out_dir, "05_signalingRole_scatter.png"),
         p, width = 12, height = 8, dpi = 150)
  cat("  saved 05_signalingRole_scatter.png\n")
}, error = function(e) cat("  05:", conditionMessage(e), "\n"))

# 4c. Heatmaps outgoing / incoming par voie
tryCatch({
  p1 <- netAnalysis_signalingRole_heatmap(cc_ckd,  pattern = "outgoing",
                                           title = "CKD — Outgoing") +
    theme(plot.title = element_text(size = 11))
  p2 <- netAnalysis_signalingRole_heatmap(cc_ctrl, pattern = "outgoing",
                                           title = "Ctrl — Outgoing") +
    theme(plot.title = element_text(size = 11))
  ggsave(file.path(out_dir, "06_heatmap_outgoing_comparison.png"),
         p1 + p2, width = 22, height = 10, dpi = 150)
  cat("  saved 06_heatmap_outgoing_comparison.png\n")
}, error = function(e) cat("  06:", conditionMessage(e), "\n"))

tryCatch({
  p1 <- netAnalysis_signalingRole_heatmap(cc_ckd,  pattern = "incoming",
                                           title = "CKD — Incoming")
  p2 <- netAnalysis_signalingRole_heatmap(cc_ctrl, pattern = "incoming",
                                           title = "Ctrl — Incoming")
  ggsave(file.path(out_dir, "07_heatmap_incoming_comparison.png"),
         p1 + p2, width = 22, height = 10, dpi = 150)
  cat("  saved 07_heatmap_incoming_comparison.png\n")
}, error = function(e) cat("  07:", conditionMessage(e), "\n"))

# ── 5. Bubble plot : interactions différentielles ────────────────────────────
cat("\nBubble plots différentiels...\n")

tryCatch({
  p <- netVisual_bubble(cc_merged,
                        sources.use = NULL, targets.use = NULL,
                        comparison = c(1, 2),
                        angle.x = 45,
                        remove.isolate = TRUE) +
    ggtitle("Interactions différentielles : Ctrl (bleu) vs CKD (rouge)")
  ggsave(file.path(out_dir, "08_bubble_diff_all.png"),
         p, width = 16, height = 12, dpi = 150)
  cat("  saved 08_bubble_diff_all.png\n")
}, error = function(e) cat("  08:", conditionMessage(e), "\n"))

# Bubble centré sur les macrophages FOLR2
for (ct in c("FOLR2+_CKD", "FOLR2+_resident", "Injured Endothelium", "Injured Tubule")) {
  tryCatch({
    p <- netVisual_bubble(cc_merged,
                          sources.use = ct, targets.use = NULL,
                          comparison = c(1, 2),
                          angle.x = 45,
                          remove.isolate = TRUE) +
      ggtitle(paste("Ligands émis par", ct, "| Ctrl vs CKD"))
    fname <- paste0("09_bubble_", gsub("[^A-Za-z0-9]", "_", ct), ".png")
    ggsave(file.path(out_dir, fname), p, width = 14, height = 9, dpi = 150)
    cat(sprintf("  saved %s\n", fname))
  }, error = function(e) cat(sprintf("  bubble %s: %s\n", ct, conditionMessage(e))))
}

# ── 6. Chord différentiel (CKD enrichi) ─────────────────────────────────────
cat("\nChord plots différentiels...\n")

tryCatch({
  png(file.path(out_dir, "10_chord_diff_CKD_enriched.png"),
      width = 1000, height = 1000, res = 130)
  netVisual_chord_cell(cc_merged, comparison = c(1, 2), net = "CKD",
                       title.name = "Interactions enrichies en CKD")
  dev.off()
  cat("  saved 10_chord_diff_CKD_enriched.png\n")
}, error = function(e) {
  tryCatch(dev.off(), error = function(x) NULL)
  cat("  10:", conditionMessage(e), "\n")
})

tryCatch({
  png(file.path(out_dir, "11_chord_diff_Ctrl_enriched.png"),
      width = 1000, height = 1000, res = 130)
  netVisual_chord_cell(cc_merged, comparison = c(1, 2), net = "Ctrl",
                       title.name = "Interactions enrichies en Contrôle")
  dev.off()
  cat("  saved 11_chord_diff_Ctrl_enriched.png\n")
}, error = function(e) {
  tryCatch(dev.off(), error = function(x) NULL)
  cat("  11:", conditionMessage(e), "\n")
})

# ── 7. Export des tables d'interactions ─────────────────────────────────────
cat("\nExport des tables...\n")

for (nm in names(object.list)) {
  df <- tryCatch(subsetCommunication(object.list[[nm]], thresh = 0.05),
                 error = function(e) subsetCommunication(object.list[[nm]], thresh = 1))
  write.csv(df, file.path(out_dir, paste0("interactions_", nm, ".csv")),
            row.names = FALSE)
  cat(sprintf("  %s : %d interactions (p<0.05)\n", nm, nrow(df)))
}

# Table différentielle (voies : CKD vs Ctrl)
tryCatch({
  # Voies présentes dans CKD mais pas Ctrl
  pw_ckd  <- cc_ckd@netP$pathways
  pw_ctrl <- cc_ctrl@netP$pathways
  only_ckd  <- setdiff(pw_ckd,  pw_ctrl)
  only_ctrl <- setdiff(pw_ctrl, pw_ckd)
  shared    <- intersect(pw_ckd, pw_ctrl)

  cat(sprintf("\n  Voies CKD uniquement (%d) : %s\n",
              length(only_ckd), paste(only_ckd, collapse = ", ")))
  cat(sprintf("  Voies Ctrl uniquement (%d) : %s\n",
              length(only_ctrl), paste(only_ctrl, collapse = ", ")))
  cat(sprintf("  Voies partagées : %d\n", length(shared)))

  df_pw <- data.frame(
    pathway = union(pw_ckd, pw_ctrl),
    in_CKD  = union(pw_ckd, pw_ctrl) %in% pw_ckd,
    in_Ctrl = union(pw_ckd, pw_ctrl) %in% pw_ctrl
  )
  write.csv(df_pw, file.path(out_dir, "pathways_CKD_vs_Ctrl.csv"), row.names = FALSE)
  cat("  saved pathways_CKD_vs_Ctrl.csv\n")
}, error = function(e) cat("  pathways:", conditionMessage(e), "\n"))

cat("\n=== Analyse terminée. Fichiers dans", out_dir, "===\n")
