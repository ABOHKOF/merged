# Chargement de l'objet CellChat sauvegardé + visualisations FOLR2_CKD

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
})

set.seed(42)
out_dir <- "plots/cellchat_FOLR2CKD"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Charger l'objet ──────────────────────────────────────────────────────
cat("Chargement de cellchat_FOLR2CKD.rds...\n")
cellchat <- readRDS(file.path(out_dir, "cellchat_FOLR2CKD.rds"))

# ── 2. Diagnostique ─────────────────────────────────────────────────────────
cat("\nDiagnostique de l objet CellChat:\n")
prob_mat <- cellchat@net$prob   # array: cell_type x cell_type x LR_pairs
pval_mat <- cellchat@net$pval

cat(sprintf("  Dimensions prob: %s\n", paste(dim(prob_mat), collapse=" x ")))
cat(sprintf("  Interactions prob > 0:     %d\n", sum(prob_mat > 0,    na.rm=TRUE)))
cat(sprintf("  Interactions pval < 0.05:  %d\n", sum(pval_mat < 0.05, na.rm=TRUE)))
cat(sprintf("  Interactions pval < 0.10:  %d\n", sum(pval_mat < 0.10, na.rm=TRUE)))
cat(sprintf("  Interactions pval < 0.50:  %d\n", sum(pval_mat < 0.50, na.rm=TRUE)))

# ── 3. Extraire les interactions avec seuil adapté ──────────────────────────
# Choisir le seuil le plus permissif qui donne des résultats
thresh_to_try <- c(0.05, 0.10, 0.25, 0.50, 1.00)
df_net <- NULL
used_thresh <- NA

for (th in thresh_to_try) {
  tmp <- tryCatch(
    subsetCommunication(cellchat, thresh = th),
    error = function(e) NULL
  )
  if (!is.null(tmp) && nrow(tmp) > 0) {
    df_net      <- tmp
    used_thresh <- th
    cat(sprintf("\n  Interactions trouvees avec thresh=%.2f: %d\n", th, nrow(df_net)))
    break
  }
}

if (is.null(df_net)) {
  # Extraction manuelle depuis le tableau prob/pval
  cat("\n  Extraction manuelle depuis prob/pval...\n")
  ct_names <- dimnames(prob_mat)[[1]]
  lr_names <- dimnames(prob_mat)[[3]]

  rows <- list()
  for (i in seq_along(ct_names)) {
    for (j in seq_along(ct_names)) {
      for (k in seq_along(lr_names)) {
        p   <- prob_mat[i, j, k]
        pv  <- pval_mat[i, j, k]
        if (p > 0) {
          rows[[length(rows)+1]] <- data.frame(
            source   = ct_names[i],
            target   = ct_names[j],
            interaction_name = lr_names[k],
            prob     = p,
            pval     = pv,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  df_net <- do.call(rbind, rows)
  used_thresh <- "manual (prob>0)"
  cat(sprintf("  Interactions brutes (prob>0): %d\n", nrow(df_net)))
}

# Ajouter infos pathway si disponible
if ("pathway_name" %in% colnames(df_net)) {
  cat("  Colonnes: pathway_name disponible\n")
} else {
  # Joindre depuis la DB
  db <- CellChatDB.human$interaction
  lr_col <- intersect(c("interaction_name", "interaction_name_2"), colnames(df_net))[1]
  if (!is.na(lr_col)) {
    df_net <- merge(df_net, db[, c("interaction_name","pathway_name","ligand","receptor")],
                    by.x = lr_col, by.y = "interaction_name", all.x = TRUE)
  }
}

write.csv(df_net, file.path(out_dir, "all_interactions_raw.csv"), row.names = FALSE)
cat(sprintf("  Sauvegarde: all_interactions_raw.csv (%d lignes)\n", nrow(df_net)))

# ── 4. Focus FOLR2_CKD ──────────────────────────────────────────────────────
folr2_src <- df_net[df_net$source == "FOLR2_CKD", ]
folr2_tgt <- df_net[df_net$target == "FOLR2_CKD", ]

cat(sprintf("\n  FOLR2_CKD comme SOURCE: %d interactions\n", nrow(folr2_src)))
cat(sprintf("  FOLR2_CKD comme CIBLE:  %d interactions\n", nrow(folr2_tgt)))

write.csv(folr2_src, file.path(out_dir, "FOLR2CKD_as_sender.csv"),   row.names = FALSE)
write.csv(folr2_tgt, file.path(out_dir, "FOLR2CKD_as_receiver.csv"), row.names = FALSE)

# Résumé partenaires
if (nrow(folr2_src) > 0) {
  cat("\n  Top partenaires (FOLR2_CKD → ?):\n")
  top_tgt <- folr2_src %>% group_by(target) %>%
    summarise(n=n(), mean_prob=mean(prob, na.rm=TRUE)) %>% arrange(desc(n))
  print(top_tgt, n=20)
}
if (nrow(folr2_tgt) > 0) {
  cat("\n  Top partenaires (? → FOLR2_CKD):\n")
  top_src <- folr2_tgt %>% group_by(source) %>%
    summarise(n=n(), mean_prob=mean(prob, na.rm=TRUE)) %>% arrange(desc(n))
  print(top_src, n=20)
}

# ── 5. Post-processing CellChat (voies, centralité) ─────────────────────────
cat("\nPost-processing CellChat...\n")
tryCatch({
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)
  cat("  computeCommunProbPathway OK\n")
}, error = function(e) cat("  computeCommunProbPathway:", conditionMessage(e), "\n"))

tryCatch({
  cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP")
  cat("  computeCentrality OK\n")
}, error = function(e) cat("  computeCentrality:", conditionMessage(e), "\n"))

# ── 6. Figures ───────────────────────────────────────────────────────────────
cat("\nGeneration des figures...\n")
group_size <- as.numeric(table(cellchat@idents))

# 6a. Réseau global
tryCatch({
  png(file.path(out_dir, "01_network_count.png"), width=1400, height=700, res=130)
  par(mfrow=c(1,2), mar=c(1,1,2,1))
  netVisual_circle(cellchat@net$count, vertex.weight=group_size,
                   weight.scale=TRUE, label.edge=FALSE,
                   title.name="Nombre d interactions")
  netVisual_circle(cellchat@net$weight, vertex.weight=group_size,
                   weight.scale=TRUE, label.edge=FALSE,
                   title.name="Force des interactions")
  dev.off()
  cat("  saved 01_network_count.png\n")
}, error = function(e) { dev.off(); cat("  01:", conditionMessage(e), "\n") })

# 6b. Heatmap interactions
tryCatch({
  p1 <- netVisual_heatmap(cellchat, measure="count", color.heatmap="Blues") +
    ggtitle("Nombre d interactions")
  p2 <- netVisual_heatmap(cellchat, measure="weight", color.heatmap="Reds") +
    ggtitle("Force des interactions")
  ggsave(file.path(out_dir, "02_heatmap_interactions.png"),
         p1 + p2, width=18, height=8, dpi=150)
  cat("  saved 02_heatmap_interactions.png\n")
}, error = function(e) cat("  02:", conditionMessage(e), "\n"))

# 6c. Bubble plot FOLR2_CKD source
tryCatch({
  p <- netVisual_bubble(cellchat, sources.use="FOLR2_CKD", targets.use=NULL,
                        remove.isolate=TRUE, angle.x=45, thresh=used_thresh) +
    ggtitle("FOLR2_CKD → (signaux emis)")
  ggsave(file.path(out_dir, "03_bubble_FOLR2CKD_source.png"),
         p, width=14, height=8, dpi=150)
  cat("  saved 03_bubble_FOLR2CKD_source.png\n")
}, error = function(e) cat("  03:", conditionMessage(e), "\n"))

# 6d. Bubble plot FOLR2_CKD cible
tryCatch({
  p <- netVisual_bubble(cellchat, sources.use=NULL, targets.use="FOLR2_CKD",
                        remove.isolate=TRUE, angle.x=45, thresh=used_thresh) +
    ggtitle("→ FOLR2_CKD (signaux recus)")
  ggsave(file.path(out_dir, "04_bubble_FOLR2CKD_target.png"),
         p, width=14, height=8, dpi=150)
  cat("  saved 04_bubble_FOLR2CKD_target.png\n")
}, error = function(e) cat("  04:", conditionMessage(e), "\n"))

# 6e. Chord diagram source
tryCatch({
  png(file.path(out_dir, "05_chord_FOLR2CKD_source.png"), width=900, height=900, res=130)
  netVisual_chord_gene(cellchat, sources.use="FOLR2_CKD", targets.use=NULL,
                       lab.cex=0.6, title.name="Ligands emis par FOLR2_CKD",
                       thresh=used_thresh)
  dev.off()
  cat("  saved 05_chord_FOLR2CKD_source.png\n")
}, error = function(e) { dev.off(); cat("  05:", conditionMessage(e), "\n") })

# 6f. Chord diagram cible
tryCatch({
  png(file.path(out_dir, "06_chord_FOLR2CKD_target.png"), width=900, height=900, res=130)
  netVisual_chord_gene(cellchat, sources.use=NULL, targets.use="FOLR2_CKD",
                       lab.cex=0.6, title.name="Ligands recus par FOLR2_CKD",
                       thresh=used_thresh)
  dev.off()
  cat("  saved 06_chord_FOLR2CKD_target.png\n")
}, error = function(e) { dev.off(); cat("  06:", conditionMessage(e), "\n") })

# 6g. Scatter rôle (sender vs receiver)
tryCatch({
  p <- netAnalysis_signalingRole_scatter(cellchat) +
    ggtitle("Role de signalisation (sender vs receiver)")
  ggsave(file.path(out_dir, "07_signaling_role_scatter.png"),
         p, width=10, height=8, dpi=150)
  cat("  saved 07_signaling_role_scatter.png\n")
}, error = function(e) cat("  07:", conditionMessage(e), "\n"))

# 6h. Bar plot: interactions FOLR2_CKD par partenaire (depuis CSV brut)
if (nrow(folr2_src) > 0 || nrow(folr2_tgt) > 0) {
  combined <- rbind(
    if (nrow(folr2_src) > 0) data.frame(partner=folr2_src$target, direction="Emis",   prob=folr2_src$prob) else NULL,
    if (nrow(folr2_tgt) > 0) data.frame(partner=folr2_tgt$source, direction="Recus",  prob=folr2_tgt$prob) else NULL
  )
  summ <- combined %>%
    group_by(partner, direction) %>%
    summarise(n_LR=n(), mean_prob=mean(prob, na.rm=TRUE), .groups="drop")

  p <- ggplot(summ, aes(x=reorder(partner, n_LR), y=n_LR, fill=direction)) +
    geom_col(position="dodge") +
    coord_flip() +
    scale_fill_manual(values=c("Emis"="#d6604d","Recus"="#4393c3")) +
    labs(x="Type cellulaire partenaire", y="Nombre de paires L-R",
         title="Interactions de FOLR2_CKD avec chaque type cellulaire",
         fill="Direction") +
    theme_bw(base_size=12)
  ggsave(file.path(out_dir, "08_barplot_FOLR2CKD_partners.png"),
         p, width=10, height=7, dpi=150)
  cat("  saved 08_barplot_FOLR2CKD_partners.png\n")
}

# 6i. Voies de signalisation FOLR2_CKD
if ("pathway_name" %in% colnames(folr2_src) && nrow(folr2_src) > 0) {
  pathways_src <- unique(na.omit(folr2_src$pathway_name))
  cat(sprintf("\n  Voies emises par FOLR2_CKD (%d): %s\n",
              length(pathways_src), paste(pathways_src, collapse=", ")))

  for (pw in head(pathways_src, 6)) {
    tryCatch({
      fname <- file.path(out_dir, paste0("09_chord_", gsub("[^A-Za-z0-9]","_",pw), ".png"))
      png(fname, width=800, height=800, res=120)
      netVisual_aggregate(cellchat, signaling=pw, layout="chord",
                          sources.use="FOLR2_CKD")
      title(main=paste("Voie:", pw, "| FOLR2_CKD source"), cex.main=1)
      dev.off()
      cat(sprintf("  saved chord_%s.png\n", pw))
    }, error=function(e) { tryCatch(dev.off(), error=function(x) NULL); cat(sprintf("  %s: %s\n", pw, conditionMessage(e))) })
  }
}

cat("\n=== Termine. Fichiers dans", out_dir, "===\n")
