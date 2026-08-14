#!/usr/bin/env Rscript

# Generate the case-study arc comparison with the pinned R4RNA source.
#
# Usage from the repository root:
#   module load R/4.4.0
#   Rscript analysis/plot_case_study_arcs.R

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else
  "results/structure_prediction/case_study.csv"
output_path <- if (length(args) >= 2) args[[2]] else
  "figures/case_study_arcs.pdf"

r4rna_root <- "software/R4RNA"
required_paths <- c(
  input_path,
  file.path(r4rna_root, "R", "misc.R"),
  file.path(r4rna_root, "R", "io.R"),
  file.path(r4rna_root, "R", "plot.R")
)
missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths) > 0) {
  stop("Missing required path(s): ", paste(missing_paths, collapse = ", "))
}

# These three files provide the dot-bracket conversion and arc-plotting
# functions used below. Sourcing the pinned submodule avoids modifying the
# user's R library.
source(file.path(r4rna_root, "R", "misc.R"))
source(file.path(r4rna_root, "R", "io.R"))
source(file.path(r4rna_root, "R", "plot.R"))

case <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required_methods <- c(
  "vienna_mfe",
  "vienna_centroid",
  "mountain_centroid_relaxed",
  "mountain_centroid_sequence_constrained"
)
if (!all(required_methods %in% case$method)) {
  stop("Case-study CSV does not contain all required methods")
}
case <- case[match(required_methods, case$method), ]

if (length(unique(case$id)) != 1 || length(unique(case$sequence)) != 1 ||
    length(unique(case$reference_structure)) != 1) {
  stop("Case-study rows must describe one sequence and one reference")
}

mfe <- case[case$method == "vienna_mfe", ]
centroid <- case[case$method == "vienna_centroid", ]
if (mfe$predicted_structure != centroid$predicted_structure ||
    !isTRUE(all.equal(mfe$base_pair_f1, centroid$base_pair_f1)) ||
    !isTRUE(all.equal(
      mfe$mean_squared_mountain_distance,
      centroid$mean_squared_mountain_distance
    ))) {
  stop("ViennaRNA MFE and centroid differ; do not combine their panels")
}

plot_methods <- c(
  "vienna_mfe",
  "mountain_centroid_sequence_constrained",
  "mountain_centroid_relaxed"
)
case <- case[match(plot_methods, case$method), ]

sequence <- strsplit(case$sequence[[1]], "", fixed = TRUE)[[1]]
reference_structure <- case$reference_structure[[1]]
sequence_length <- nchar(case$sequence[[1]])
if (nchar(reference_structure) != sequence_length) {
  stop("Reference and sequence lengths differ")
}
positions <- seq_len(sequence_length - 1)
nmsmd_denominator <- sum(pmin(positions, sequence_length - positions)^2)

reference <- viennaToHelix(reference_structure)

method_labels <- c(
  vienna_mfe = "MFE / centroid",
  mountain_centroid_relaxed = "Mountain-path relaxation",
  mountain_centroid_sequence_constrained = "Mountain Centroid"
)

colours <- c(
  method_result = "#0072B2",
  reference = "#009E73"
)
plot_family <- "Times New Roman"

mountainProfile <- function(structure) {
  characters <- strsplit(structure, "", fixed = TRUE)[[1]]
  depth <- 0
  profile <- numeric(length(characters) - 1)
  for (position in seq_along(characters)) {
    if (characters[[position]] == "(") {
      depth <- depth + 1
    } else if (characters[[position]] == ")") {
      depth <- depth - 1
    }
    if (position < length(characters)) {
      profile[[position]] <- depth
    }
  }
  profile
}

reference_profile <- mountainProfile(reference_structure)
prediction_profiles <- lapply(case$predicted_structure, mountainProfile)
profile_maximum <- max(reference_profile, unlist(prediction_profiles))
profile_positions <- seq_len(sequence_length - 1)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
cairo_pdf(
  output_path,
  width = 10.5,
  height = 6.5,
  family = plot_family,
  onefile = TRUE
)
on.exit(dev.off(), add = TRUE)
layout(
  matrix(seq_len(6), nrow = 2, byrow = TRUE),
  heights = c(3.0, 1.6)
)
par(
  oma = c(2.8, 0.5, 1.5, 0.5),
  family = plot_family
)

for (row_index in seq_len(nrow(case))) {
  par(mar = c(0.2, 0.3, 2.8, 0.3))
  prediction <- viennaToHelix(case$predicted_structure[[row_index]])
  prediction$col <- colours[["method_result"]]
  prediction$lwd <- 1.35
  prediction$lty <- 1

  panel_reference <- reference
  panel_reference$col <- colours[["reference"]]
  panel_reference$lwd <- 1.35
  panel_reference$lty <- 1

  top_height <- maxHeight(prediction)
  bottom_height <- maxHeight(panel_reference)
  blankPlot(
    sequence_length,
    top_height,
    -bottom_height,
    pad = c(5, 4, 12, 4),
    scale = FALSE,
    no.par = TRUE
  )
  plotHelix(prediction, add = TRUE)
  plotHelix(panel_reference, flip = TRUE, add = TRUE)
  lines(c(0.5, sequence_length + 0.5), c(0, 0), col = "#303030", lwd = 0.8)

  mtext(
    method_labels[[case$method[[row_index]]]],
    side = 3,
    line = 1.35,
    adj = 0,
    font = 2,
    cex = 1.18
  )
  nmsmd <- case$mean_squared_mountain_distance[[row_index]] *
    (sequence_length - 1) / nmsmd_denominator
  metric_label <- sprintf(
    "BP F1 %.3f; NMSMD %.5f",
    case$base_pair_f1[[row_index]],
    nmsmd
  )
  mtext(
    metric_label,
    side = 3,
    line = 0.05,
    adj = 0,
    cex = 1.0
  )
}

profile_ticks <- pretty(c(0, profile_maximum), n = 4)
profile_plot_maximum <- max(profile_maximum, profile_ticks)
position_ticks <- seq(20, sequence_length - 1, by = 20)
for (row_index in seq_len(nrow(case))) {
  par(mar = c(3.0, if (row_index == 1) 3.5 else 2.4, 0.5, 0.6))
  plot(
    profile_positions,
    reference_profile,
    type = "n",
    xlim = c(1, sequence_length - 1),
    ylim = c(0, profile_plot_maximum),
    xlab = "",
    ylab = "",
    axes = FALSE,
    xaxs = "i",
    yaxs = "i"
  )
  abline(h = profile_ticks, col = "#E5E5E5", lwd = 0.7)
  abline(v = position_ticks, col = "#EEEEEE", lwd = 0.7)
  lines(
    profile_positions,
    reference_profile,
    col = colours[["reference"]],
    lwd = 2.1,
    lty = 1
  )
  lines(
    profile_positions,
    prediction_profiles[[row_index]],
    col = colours[["method_result"]],
    lwd = 1.7,
    lty = 2
  )
  axis(1, at = position_ticks, cex.axis = 0.9, tck = -0.025)
  axis(
    2,
    at = profile_ticks,
    labels = if (row_index == 1) profile_ticks else FALSE,
    cex.axis = 0.9,
    las = 1,
    tck = -0.025
  )
  box(col = "#303030", lwd = 0.8)
  mtext("Position", side = 1, line = 1.8, cex = 0.9)
  if (row_index == 1) {
    mtext("Mountain height", side = 2, line = 2.3, cex = 0.9)
  }
}

mtext(
  sprintf("Case study: %s (%d nt)", case$id[[1]], sequence_length),
  side = 3,
  outer = TRUE,
  line = 0.25,
  font = 2,
  cex = 1.15
)
grid::grid.text(
  "Blue: method result",
  x = grid::unit(0.485, "npc"),
  y = grid::unit(0.022, "npc"),
  just = "right",
  gp = grid::gpar(
    col = colours[["method_result"]],
    fontsize = 13,
    fontfamily = plot_family,
    fontface = "bold"
  )
)
grid::grid.text(
  "Green: reference",
  x = grid::unit(0.515, "npc"),
  y = grid::unit(0.022, "npc"),
  just = "left",
  gp = grid::gpar(
    col = colours[["reference"]],
    fontsize = 13,
    fontfamily = plot_family,
    fontface = "bold"
  )
)

message("Wrote ", output_path)
