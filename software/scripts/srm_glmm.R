# Social relations analysis on codebook v3.5.1 consensus codes, with the right model for each outcome
# (A2's review: yes/no outcomes need GLMMs, not linear mixed models).
# Input: analysis/srm_glmm/long.csv (scripts/srm_glmm_prep.py). Rememberer and partner IDs are global, so they are crossed.
# Binary outcomes:  glmer(y ~ 1 + (1|batch) + (1|rem) + (1|par), binomial(logit)); variance shares on the latent scale,
#                   residual = pi^2/3 (Nakagawa & Schielzeth 2010). Also the linear-probability LMM for comparison.
# Numeric outcomes: lmer with the same random effects; shares of total variance.
# Rare outcomes can separate (variances explode); those (any variance > 10) are refit with blme::bglmer and a weak
#   gamma prior on the standard deviations (Chung et al. 2013), and flagged "regularised".
# Each random effect is tested by a likelihood-ratio test against the model without it (p halved: variance on the boundary).
# Writes analysis/srm_glmm/srm_glmm.json.
suppressPackageStartupMessages({library(lme4); library(jsonlite); library(blme)})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
root <- normalizePath(file.path(here, ".."))
d <- read.csv(file.path(root, "analysis", "srm_glmm", "long.csv"), stringsAsFactors = FALSE)
info <- fromJSON(file.path(root, "analysis", "srm_glmm", "outcomes.json"))
ctrl_g <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
ctrl_l <- lmerControl(optimizer = "bobyqa")
vc <- function(m) { v <- as.data.frame(VarCorr(m)); setNames(v$vcov, v$grp) }
lrt <- function(full, reduced) { s <- 2 * (as.numeric(logLik(full)) - as.numeric(logLik(reduced))); max(0, s) |> (\(x) 0.5 * pchisq(x, 1, lower.tail = FALSE))() }
out <- list()
for (y in names(info)) {
  kind <- info[[y]]$kind
  if (grepl("__NO$", y) && sub("__NO$", "__YES", y) %in% names(info) && !grepl("^b__new_info", y) && !grepl("new_info", y)) next   # the YES column carries the same information for two-answer questions; new_info has three answers (Oct 10, A1 OK), so its NO is fit too
  dd <- d[!is.na(d[[y]]), c("batch", "rem", "par", y)]; names(dd)[4] <- "y"
  res <- list(kind = kind, n = nrow(dd), n_rememberers = length(unique(dd$rem)), n_partners = length(unique(dd$par)))
  f_full <- y ~ 1 + (1 | batch) + (1 | rem) + (1 | par)
  if (kind == "binary") {
    m <- tryCatch(glmer(f_full, dd, family = binomial, control = ctrl_g), error = function(e) NULL)
    if (is.null(m)) { res$error <- "glmer failed"; out[[y]] <- res; next }
    v <- vc(m); reg <- FALSE
    if (max(v) > 10) {   # separation: refit with a weak prior on the SDs
      m <- bglmer(f_full, dd, family = binomial, control = ctrl_g, cov.prior = gamma(shape = 2, rate = 0.5)); v <- vc(m); reg <- TRUE
    }
    tot <- sum(v[c("rem", "par", "batch")]) + pi^2 / 3
    res$glmm <- list(rememberer = v[["rem"]] / tot, partner = v[["par"]] / tot, batch = v[["batch"]] / tot,
                     var_rem = v[["rem"]], var_par = v[["par"]], var_batch = v[["batch"]],
                     singular = isSingular(m), regularised = reg,
                     converged = !any(grepl("failed to converge|degenerate|Hessian", m@optinfo$conv$lme4$messages)),
                     p_rem = lrt(m, glmer(y ~ 1 + (1 | batch) + (1 | par), dd, family = binomial, control = ctrl_g)),
                     p_par = lrt(m, glmer(y ~ 1 + (1 | batch) + (1 | rem), dd, family = binomial, control = ctrl_g)))
    ml <- lmer(f_full, dd, REML = TRUE, control = ctrl_l); vl <- vc(ml); tl <- sum(vl)
    res$lpm <- list(rememberer = vl[["rem"]] / tl, partner = vl[["par"]] / tl, batch = vl[["batch"]] / tl)
  } else {
    m <- lmer(f_full, dd, REML = FALSE, control = ctrl_l); v <- vc(m); tot <- sum(v)
    res$lmm <- list(rememberer = v[["rem"]] / tot, partner = v[["par"]] / tot, batch = v[["batch"]] / tot, residual = v[["Residual"]] / tot,
                    singular = isSingular(m),
                    p_rem = lrt(m, lmer(y ~ 1 + (1 | batch) + (1 | par), dd, REML = FALSE, control = ctrl_l)),
                    p_par = lrt(m, lmer(y ~ 1 + (1 | batch) + (1 | rem), dd, REML = FALSE, control = ctrl_l)))
  }
  out[[y]] <- res
  cat(sprintf("%-55s %s\n", y, if (kind == "binary") sprintf("rem %.2f par %.2f (LPM rem %.2f par %.2f) p_rem %.3g p_par %.3g",
      res$glmm$rememberer, res$glmm$partner, res$lpm$rememberer, res$lpm$partner, res$glmm$p_rem, res$glmm$p_par)
      else sprintf("rem %.2f par %.2f resid %.2f p_rem %.3g p_par %.3g", res$lmm$rememberer, res$lmm$partner, res$lmm$residual, res$lmm$p_rem, res$lmm$p_par)))
}
write_json(out, file.path(root, "analysis", "srm_glmm", "srm_glmm.json"), auto_unbox = TRUE, digits = 4, pretty = TRUE)
cat("DONE\n")
