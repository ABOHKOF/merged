# CellChat: communication des FOLR2_CKD — données scRNA-seq (2580 gènes)

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
})

set.seed(42)
options(future.globals.maxSize = 4000 * 1024^2)
out_dir <- "plots/cellchat_FOLR2CKD"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Charger les données scRNA ─────────────────────────────────────────────
cat("Chargement des donnees scRNA...\n")
counts <- readMM("cellchat_input_scrna/matrix.mtx")
genes  <- read.table("cellchat_input_scrna/features.tsv",  header=FALSE)$V1
cells  <- read.table("cellchat_input_scrna/barcodes.tsv",  header=FALSE)$V1
meta   <- read.csv("cellchat_input_scrna/metadata.csv",    row.names=1)

rownames(counts) <- genes
colnames(counts) <- cells
counts <- as(counts, "CsparseMatrix")
cat(sprintf("  Matrix: %d genes x %d cells\n", nrow(counts), ncol(counts)))
cat("  Types cellulaires:\n")
print(sort(table(meta$annot_final2), decreasing=TRUE))

# ── 2. Créer objet CellChat ──────────────────────────────────────────────────
cat("\nCreation de l objet CellChat...\n")
cellchat <- createCellChat(
  object   = counts,
  meta     = meta,
  group.by = "annot_final2"
)
cellchat@DB <- CellChatDB.human
cat(sprintf("  DB: %d interactions\n", nrow(CellChatDB.human$interaction)))

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
cat("\nPreprocessing...\n")
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)

n_overexp <- nrow(cellchat@LR$LRsig)
cat(sprintf("  Paires L-R surexprimees: %d\n", n_overexp))

# ── 4. Calcul des probabilités ───────────────────────────────────────────────
cat("\nCalcul des probabilites de communication...\n")
cellchat <- computeCommunProb(cellchat, type="triMean", population.size=TRUE)
cellchat <- filterCommunication(cellchat, min.cells=10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)
cellchat <- netAnalysis_computeCentrality(cellchat, slot.name="netP")

saveRDS(cellchat, file.path(out_dir, "cellchat_scrna.rds"))
cat("  Objet sauvegarde: cellchat_scrna.rds\n")

# ── 5. Extraire interactions ─────────────────────────────────────────────────
cat("\nExtraction des interactions...\n")
prob_pos <- sum(cellchat@net$prob > 0, na.rm=TRUE)
cat(sprintf("  prob > 0: %d\n", prob_pos))

df_net <- tryCatch(
  subsetCommunication(cellchat, thresh=0.05),
  error = function(e) {
    cat("  p<0.05 vide, essai thresh=0.1...\n")
    tryCatch(subsetCommunication(cellchat, thresh=0.1),
             error = function(e2) subsetCommunication(cellchat, thresh=1))
  }
)
cat(sprintf("  Total interactions: %d\n", nrow(df_net)))

folr2_src <- df_net[df_net$source == "FOLR2_CKD", ]
folr2_tgt <- df_net[df_net$target == "FOLR2_CKD", ]
cat(sprintf("  FOLR2_CKD SOURCE: %d\n", nrow(folr2_src)))
cat(sprintf("  FOLR2_CKD CIBLE:  %d\n", nrow(folr2_tgt)))

write.csv(df_net,      file.path(out_dir, "all_interactions_scrna.csv"),     row.names=FALSE)
write.csv(folr2_src,   file.path(out_dir, "FOLR2CKD_as_sender_scrna.csv"),   row.names=FALSE)
write.csv(folr2_tgt,   file.path(out_dir, "FOLR2CKD_as_receiver_scrna.csv"), row.names=FALSE)

if (nrow(folr2_src) > 0) {
  cat("\n  Top partenaires (FOLR2_CKD emetteur):\n")
  print(folr2_src %>% group_by(target) %>%
    summarise(n_LR=n(), mean_prob=mean(prob)) %>% arrange(desc(n_LR)), n=25)
  cat("  Voies emises:", paste(unique(folr2_src$pathway_name), collapse=", "), "\n")
}
if (nrow(folr2_tgt) > 0) {
  cat("\n  Top partenaires (FOLR2_CKD receveur):\n")
  print(folr2_tgt %>% group_by(source) %>%
    summarise(n_LR=n(), mean_prob=mean(prob)) %>% arrange(desc(n_LR)), n=25)
  cat("  Voies recues:", paste(unique(folr2_tgt$pathway_name), collapse=", "), "\n")
}

# ── 6. Figures ───────────────────────────────────────────────────────────────
cat("\nGeneration des figures...\n")
group_size <- as.numeric(table(cellchat@idents))

# 6a. Réseau global (nb et poids)
tryCatch({
  png(file.path(out_dir, "01_network_global.png"), width=1400, height=700, res=130)
  par(mfrow=c(1,2), mar=c(1,1,2,1))
  netVisual_circle(cellchat@net$count, vertex.weight=group_size,
                   weight.scale=TRUE, label.edge=FALSE, title.name="Nb interactions")
  netVisual_circle(cellchat@net$weight, vertex.weight=group_size,
                   weight.scale=TRUE, label.edge=FALSE, title.name="Force")
  dev.off(); cat("  saved 01_network_global.png\n")
}, error=function(e){tryCatch(dev.off(),error=function(x)NULL); cat("  01:",conditionMessage(e),"\n")})

# 6b. Heatmap nb interactions
tryCatch({
  p1 <- netVisual_heatmap(cellchat, measure="count", color.heatmap="Blues") + ggtitle("Nb interactions")
  p2 <- netVisual_heatmap(cellchat, measure="weight", color.heatmap="Reds")  + ggtitle("Force")
  ggsave(file.path(out_dir,"02_heatmap_interactions.png"), p1+p2, width=20, height=9, dpi=150)
  cat("  saved 02_heatmap_interactions.png\n")
}, error=function(e) cat("  02:",conditionMessage(e),"\n"))

# 6c. Bubble plots FOLR2_CKD
tryCatch({
  p <- netVisual_bubble(cellchat, sources.use="FOLR2_CKD", targets.use=NULL,
                        remove.isolate=TRUE, angle.x=45) +
    ggtitle("Ligands emis par FOLR2_CKD → tous types")
  ggsave(file.path(out_dir,"03_bubble_FOLR2CKD_source.png"), p, width=14, height=9, dpi=150)
  cat("  saved 03_bubble_FOLR2CKD_source.png\n")
}, error=function(e) cat("  03:",conditionMessage(e),"\n"))

tryCatch({
  p <- netVisual_bubble(cellchat, sources.use=NULL, targets.use="FOLR2_CKD",
                        remove.isolate=TRUE, angle.x=45) +
    ggtitle("Ligands recus par FOLR2_CKD ← tous types")
  ggsave(file.path(out_dir,"04_bubble_FOLR2CKD_target.png"), p, width=14, height=9, dpi=150)
  cat("  saved 04_bubble_FOLR2CKD_target.png\n")
}, error=function(e) cat("  04:",conditionMessage(e),"\n"))

# 6d. Chord diagrams
tryCatch({
  png(file.path(out_dir,"05_chord_FOLR2CKD_source.png"), width=1000, height=1000, res=130)
  netVisual_chord_gene(cellchat, sources.use="FOLR2_CKD", targets.use=NULL,
                       lab.cex=0.7, title.name="Ligands emis par FOLR2_CKD")
  dev.off(); cat("  saved 05_chord_FOLR2CKD_source.png\n")
}, error=function(e){tryCatch(dev.off(),error=function(x)NULL); cat("  05:",conditionMessage(e),"\n")})

tryCatch({
  png(file.path(out_dir,"06_chord_FOLR2CKD_target.png"), width=1000, height=1000, res=130)
  netVisual_chord_gene(cellchat, sources.use=NULL, targets.use="FOLR2_CKD",
                       lab.cex=0.7, title.name="Ligands recus par FOLR2_CKD")
  dev.off(); cat("  saved 06_chord_FOLR2CKD_target.png\n")
}, error=function(e){tryCatch(dev.off(),error=function(x)NULL); cat("  06:",conditionMessage(e),"\n")})

# 6e. Scatter rôle sender/receiver
tryCatch({
  p <- netAnalysis_signalingRole_scatter(cellchat) +
    ggtitle("Role sender vs receiver (toutes voies)")
  ggsave(file.path(out_dir,"07_signaling_role_scatter.png"), p, width=10, height=8, dpi=150)
  cat("  saved 07_signaling_role_scatter.png\n")
}, error=function(e) cat("  07:",conditionMessage(e),"\n"))

# 6f. Bar plot partenaires FOLR2_CKD
if (nrow(folr2_src) > 0 || nrow(folr2_tgt) > 0) {
  combined <- rbind(
    if(nrow(folr2_src)>0) data.frame(partner=folr2_src$target, dir="Emis par FOLR2_CKD",  prob=folr2_src$prob) else NULL,
    if(nrow(folr2_tgt)>0) data.frame(partner=folr2_tgt$source, dir="Recu par FOLR2_CKD", prob=folr2_tgt$prob) else NULL
  )
  summ <- combined %>% group_by(partner, dir) %>%
    summarise(n_LR=n(), mean_prob=mean(prob, na.rm=TRUE), .groups="drop")

  p <- ggplot(summ, aes(x=reorder(partner,n_LR), y=n_LR, fill=dir)) +
    geom_col(position="dodge") + coord_flip() +
    scale_fill_manual(values=c("Emis par FOLR2_CKD"="#d6604d","Recu par FOLR2_CKD"="#4393c3")) +
    labs(x="Partenaire cellulaire", y="Nb paires L-R",
         title="Interactions de FOLR2_CKD avec chaque type cellulaire", fill="") +
    theme_bw(base_size=12) + theme(legend.position="top")
  ggsave(file.path(out_dir,"08_barplot_partners.png"), p, width=11, height=8, dpi=150)
  cat("  saved 08_barplot_partners.png\n")
}

# 6g. Chord cellulaire (niveau type)
tryCatch({
  png(file.path(out_dir,"09_chord_celltype_FOLR2CKD.png"), width=1000, height=1000, res=130)
  netVisual_chord_cell(cellchat, sources.use="FOLR2_CKD",
                       title.name="FOLR2_CKD: interactions sortantes")
  dev.off(); cat("  saved 09_chord_celltype_FOLR2CKD.png\n")
}, error=function(e){tryCatch(dev.off(),error=function(x)NULL); cat("  09:",conditionMessage(e),"\n")})

# 6h. Voies de signalisation FOLR2_CKD (top 6)
pathways_src <- unique(folr2_src$pathway_name)
pathways_src <- pathways_src[!is.na(pathways_src)]
for (pw in head(pathways_src, 6)) {
  tryCatch({
    fname <- file.path(out_dir, paste0("10_chord_", gsub("[^A-Za-z0-9]","_",pw), ".png"))
    png(fname, width=900, height=900, res=120)
    netVisual_aggregate(cellchat, signaling=pw, layout="chord",
                        sources.use="FOLR2_CKD")
    title(main=paste("Voie:", pw, "| FOLR2_CKD source"), cex.main=1.1)
    dev.off()
    cat(sprintf("  saved 10_chord_%s.png\n", pw))
  }, error=function(e){tryCatch(dev.off(),error=function(x)NULL); cat(sprintf("  %s: %s\n",pw,conditionMessage(e)))})
}

# 6i. Heatmap rôles par voie
tryCatch({
  p <- netAnalysis_signalingRole_heatmap(cellchat, pattern="outgoing") +
    ggtitle("Signaux emis par type cellulaire (outgoing)")
  ggsave(file.path(out_dir,"11_heatmap_outgoing.png"), p, width=12, height=8, dpi=150)
  cat("  saved 11_heatmap_outgoing.png\n")
}, error=function(e) cat("  11:",conditionMessage(e),"\n"))

tryCatch({
  p <- netAnalysis_signalingRole_heatmap(cellchat, pattern="incoming") +
    ggtitle("Signaux recus par type cellulaire (incoming)")
  ggsave(file.path(out_dir,"12_heatmap_incoming.png"), p, width=12, height=8, dpi=150)
  cat("  saved 12_heatmap_incoming.png\n")
}, error=function(e) cat("  12:",conditionMessage(e),"\n"))

cat("\n=== Termine. Fichiers dans", out_dir, "===\n")
