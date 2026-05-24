#!/usr/bin/env Rscript

# Test whether toRvik returns current available transfer records.
# Run from the repo root:
#   Rscript scripts/test_torvik_transfer_portal.R 2026

args <- commandArgs(trailingOnly = TRUE)
year <- if (length(args) >= 1) as.integer(args[[1]]) else 2026L

if (!requireNamespace("toRvik", quietly = TRUE)) {
  stop(
    "Package 'toRvik' is not installed. Install it with:\n",
    "install.packages('toRvik')\n",
    call. = FALSE
  )
}

transfers <- toRvik::transfer_portal(year = year)

available_values <- c("", "na", "nan", "none", "uncommitted", "undecided", "tbd")
to_text <- tolower(trimws(as.character(transfers$to)))
to_text[is.na(to_text)] <- ""
transfers$available <- to_text %in% available_values
transfers$status <- ifelse(transfers$available, "Available transfer", "Portal committed")

cat("Year requested:", year, "\n")
cat("Rows returned:", nrow(transfers), "\n")
if ("year" %in% names(transfers)) {
  cat("Year range:", min(transfers$year, na.rm = TRUE), "-", max(transfers$year, na.rm = TRUE), "\n")
}
cat("Available:", sum(transfers$available, na.rm = TRUE), "\n")
cat("Committed:", sum(!transfers$available, na.rm = TRUE), "\n")

out <- sprintf("transfer_portal_%s_torvik_test.csv", year)
write.csv(transfers, out, row.names = FALSE)
cat("Wrote:", out, "\n")

available <- transfers[transfers$available, , drop = FALSE]
if (nrow(available) > 0) {
  cols <- intersect(c("id", "player", "from", "to", "exp", "year", "status"), names(available))
  print(utils::head(available[, cols, drop = FALSE], 25))
}
