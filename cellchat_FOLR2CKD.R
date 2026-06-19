# CellChat: communication des FOLR2_CKD avec les autres types cellulaires
# Condition: CKD (CKD + DKD + DKD+FSGS)

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
})

set.seed(42)
options(future.globals.maxSize = 4000 * 1024^2)  # 4 GB

out_dir <- "plots/cellchat_FOLR2CKD"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Charger les données ──────────────────────────────────────────────────
cat("Chargement des donnees...\n")
counts <- readMM("cellchat_input/matrix.mtx")
genes  <- read.table("cellchat_input/features.tsv",  header = FALSE)$V1
cells  <- read.table("cellchat_input/barcodes.tsv",  header = FALSE)$V1
meta   <- read.csv("cellchat_input/metadata.csv",    row.names = 1)

rownames(counts) <- genes
colnames(counts) <- cells
counts <- as(counts, "CsparseMatrix")

cat(sprintf("  Matrix: %d genes x %d cells\n", nrow(counts), ncol(counts)))
cat("  Types cellulaires:\n")
print(table(meta$celltype_c2l))

# ── 2. Créer l'objet CellChat ───────────────────────────────────────────────
cat("\nCreation de l objet CellChat...\n")
cellchat <- createCellChat(
  object   = counts,
  meta     = meta,
  group.by = "celltype_c2l"
)

# Base de données humaine
CellChatDB         <- CellChatDB.human
cellchat@DB        <- CellChatDB
cat(sprintf("  DB: %d interactions\n", nrow(CellChatDB$interaction)))

# ── 3. Preprocessing ────────────────────────────────────────────────────────
cat("\nPreprocessing...\n")
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)

# ── 4. Calcul des probabilités de communication ──────────────────────────────
cat("\nCalcul des probabilites de communication...\n")
cellchat <- computeCommunProb(
  cellchat,
  type        = "triMean",
  population.size = TRUE
)
# Seuil min.cells bas car panel spatial (peu de gènes)
cellchat <- filterCommunication(cellchat, min.cells = 5)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)

# Sauvegarder l'objet
saveRDS(cellchat, file.path(out_dir, "cellchat_FOLR2CKD.rds"))
cat("  Objet sauvegarde: cellchat_FOLR2CKD.rds\n")

# ── 5. Résumé des interactions ──────────────────────────────────────────────
cat("\nNombre d interactions significatives:\n")

# Diagnostique: combien d'interactions brutes avant seuil p-value?
net_raw <- cellchat@net
cat(sprintf("  Interactions brutes (prob > 0): %d\n",
            sum(net_raw$prob > 0, na.rm = TRUE)))

# Essayer avec seuil p-value relâché (thresh=0.05 par défaut)
df_net <- tryCatch(
  subsetCommunication(cellchat, thresh = 0.05),
  error = function(e) {
    cat("  thresh=0.05 vide, on essaie thresh=1 (toutes)...\n")
    subsetCommunication(cellchat, thresh = 1)
  }
)
cat(sprintf("  Total: %d interactions LR\n", nrow(df_net)))

# Interactions impliquant FOLR2_CKD
folr2_source <- df_net[df_net$source == "FOLR2_CKD", ]
folr2_target <- df_net[df_net$target == "FOLR2_CKD", ]
cat(sprintf("  FOLR2_CKD comme source: %d\n", nrow(folr2_source)))
cat(sprintf("  FOLR2_CKD comme cible:  %d\n", nrow(folr2_target)))

write.csv(folr2_source, file.path(out_dir, "FOLR2CKD_as_sender.csv"),   row.names = FALSE)
write.csv(folr2_target, file.path(out_dir, "FOLR2CKD_as_receiver.csv"), row.names = FALSE)
write.csv(df_net,       file.path(out_dir, "all_interactions.csv"),     row.names = FALSE)

# Top partenaires
cat("\nTop partenaires quand FOLR2_CKD est SOURCE:\n")
top_targets <- folr2_source %>%
  group_by(target) %>%
  summarise(n_LR = n(), mean_prob = mean(prob)) %>%
  arrange(desc(n_LR))
print(top_targets)

cat("\nTop partenaires quand FOLR2_CKD est CIBLE:\n")
top_sources <- folr2_target %>%
  group_by(source) %>%
  summarise(n_LR = n(), mean_prob = mean(prob)) %>%
  arrange(desc(n_LR))
print(top_sources)

# Top voies de signalisation
cat("\nVoies de signalisation impliquant FOLR2_CKD:\n")
pathways_source <- unique(folr2_source$pathway_name)
pathways_target <- unique(folr2_target$pathway_name)
cat("  Source:", paste(pathways_source, collapse = ", "), "\n")
cat("  Cible: ", paste(pathways_target, collapse = ", "), "\n")

# ── 6. Visualisations ───────────────────────────────────────────────────────
cat("\nGeneration des figures...\n")
group_size <- as.numeric(table(cellchat@idents))

# 6a. Réseau global: nombre d'interactions
png(file.path(out_dir, "01_network_count.png"), width = 1200, height = 600, res = 120)
par(mfrow = c(1, 2), mar = c(1, 1, 2, 1))
netVisual_circle(
  cellchat@net$count,
  vertex.weight = group_size,
  weight.scale  = TRUE,
  label.edge    = FALSE,
  title.name    = "Nombre d interactions"
)
netVisual_circle(
  cellchat@net$weight,
  vertex.weight = group_size,
  weight.scale  = TRUE,
  label.edge    = FALSE,
  title.name    = "Force des interactions"
)
dev.off()
cat("  saved 01_network_count.png\n")

# 6b. Spotlight FOLR2_CKD – interactions sortantes
png(file.path(out_dir, "02_FOLR2CKD_outgoing.png"), width = 800, height = 800, res = 120)
netVisual_individual(
  cellchat,
  signaling        = NULL,
  pairLR.use       = NULL,
  sources.use      = "FOLR2_CKD",
  targets.use      = NULL,
  vertex.receiver  = seq_len(length(unique(cellchat@idents))),
  remove.isolate   = TRUE,
  top              = 1
)
dev.off()
cat("  saved 02_FOLR2CKD_outgoing.png\n")

# 6c. Chord diagram – FOLR2_CKD comme source
png(file.path(out_dir, "03_chord_FOLR2CKD_source.png"), width = 900, height = 900, res = 120)
netVisual_chord_gene(
  cellchat,
  sources.use = "FOLR2_CKD",
  targets.use = NULL,
  lab.cex     = 0.6,
  title.name  = "Ligands emis par FOLR2_CKD"
)
dev.off()
cat("  saved 03_chord_FOLR2CKD_source.png\n")

# 6d. Chord diagram – FOLR2_CKD comme cible
png(file.path(out_dir, "04_chord_FOLR2CKD_target.png"), width = 900, height = 900, res = 120)
netVisual_chord_gene(
  cellchat,
  sources.use = NULL,
  targets.use = "FOLR2_CKD",
  lab.cex     = 0.6,
  title.name  = "Ligands recus par FOLR2_CKD"
)
dev.off()
cat("  saved 04_chord_FOLR2CKD_target.png\n")

# 6e. Bubble plot: toutes les interactions avec FOLR2_CKD
tryCatch({
  p <- netVisual_bubble(
    cellchat,
    sources.use = "FOLR2_CKD",
    targets.use = NULL,
    remove.isolate = TRUE,
    angle.x        = 45
  ) + ggtitle("FOLR2_CKD → autres types (signaux emis)")
  ggsave(file.path(out_dir, "05_bubble_FOLR2CKD_source.png"),
         p, width = 14, height = 8, dpi = 150)
  cat("  saved 05_bubble_FOLR2CKD_source.png\n")
}, error = function(e) cat("  bubble source: ", conditionMessage(e), "\n"))

tryCatch({
  p <- netVisual_bubble(
    cellchat,
    sources.use = NULL,
    targets.use = "FOLR2_CKD",
    remove.isolate = TRUE,
    angle.x        = 45
  ) + ggtitle("Autres types → FOLR2_CKD (signaux recus)")
  ggsave(file.path(out_dir, "06_bubble_FOLR2CKD_target.png"),
         p, width = 14, height = 8, dpi = 150)
  cat("  saved 06_bubble_FOLR2CKD_target.png\n")
}, error = function(e) cat("  bubble target: ", conditionMessage(e), "\n"))

# 6f. Heatmap des interactions par voie (FOLR2_CKD source)
tryCatch({
  p <- netAnalysis_signalingRole_heatmap(
    cellchat,
    pattern   = "outgoing",
    signaling = pathways_source[seq_len(min(20, length(pathways_source)))]
  )
  png(file.path(out_dir, "07_heatmap_outgoing_pathways.png"),
      width = 1000, height = 800, res = 120)
  print(p)
  dev.off()
  cat("  saved 07_heatmap_outgoing_pathways.png\n")
}, error = function(e) cat("  heatmap outgoing: ", conditionMessage(e), "\n"))

tryCatch({
  p <- netAnalysis_signalingRole_heatmap(
    cellchat,
    pattern   = "incoming",
    signaling = pathways_target[seq_len(min(20, length(pathways_target)))]
  )
  png(file.path(out_dir, "08_heatmap_incoming_pathways.png"),
      width = 1000, height = 800, res = 120)
  print(p)
  dev.off()
  cat("  saved 08_heatmap_incoming_pathways.png\n")
}, error = function(e) cat("  heatmap incoming: ", conditionMessage(e), "\n"))

# 6g. Top voies de signalisation – violin/scatter des L-R
for (pathway in head(pathways_source, 5)) {
  tryCatch({
    fname <- file.path(out_dir, paste0("09_pathway_", gsub("[^A-Za-z0-9]", "_", pathway), ".png"))
    png(fname, width = 900, height = 700, res = 120)
    netVisual_aggregate(
      cellchat,
      signaling       = pathway,
      layout          = "chord",
      sources.use     = "FOLR2_CKD"
    )
    title(main = paste("Voie:", pathway, "(FOLR2_CKD source)"), cex.main = 1)
    dev.off()
    cat(sprintf("  saved pathway chord: %s\n", pathway))
  }, error = function(e) cat(sprintf("  %s: %s\n", pathway, conditionMessage(e))))
}

# 6h. Dot plot: rôle de signalisation (dominance)
tryCatch({
  cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP")
  p <- netAnalysis_signalingRole_scatter(cellchat) +
    ggtitle("Role de signalisation de chaque type cellulaire")
  ggsave(file.path(out_dir, "10_signaling_role_scatter.png"),
         p, width = 10, height = 8, dpi = 150)
  cat("  saved 10_signaling_role_scatter.png\n")
}, error = function(e) cat("  centrality: ", conditionMessage(e), "\n"))

cat("\n=== Terminé. Fichiers dans", out_dir, "===\n")
