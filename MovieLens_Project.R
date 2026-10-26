# ============================================================================
# MOVIELENS 10M PROJECT — HARVARDX DATA SCIENCE
# Complete, commented, reproducible script
# IMPORTANT: final_holdout_test is used only once, at the end.
# Run from the directory containing MovieLens_Report.Rmd:
# source("MovieLens_Project.R", encoding = "UTF-8")
# ============================================================================

# 0. DEPENDENCIES AND CONFIGURATION --------------------------------------------
packages <- c("dplyr", "ggplot2", "readr", "caret", "stringr", "tibble")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(readr)
  library(caret); library(stringr); library(tibble)
})
# The working directory must contain both submission source files.
if (!file.exists("MovieLens_Report.Rmd")) {
  stop("MovieLens_Report.Rmd was not found. Place both files in the same folder and run source() from that folder.")
}
cat("Working directory:", getwd(), "\n")
outdir <- "resultats_movielens"
# Safe resume: if the final metrics were saved, only rebuild the report.
# Never reopen the final holdout or retrain after a completed evaluation.
render_saved_report <- function() {
  if (!requireNamespace("rmarkdown", quietly = TRUE))
    install.packages("rmarkdown", repos = "https://cloud.r-project.org")
  rmarkdown::render("MovieLens_Report.Rmd",
                    output_format = "pdf_document",
                    output_file = "MovieLens_Report.pdf",
                    knit_root_dir = normalizePath(getwd()),
                    envir = new.env(parent = globalenv()))
}
run_project <- function() {
if (file.exists(file.path(outdir, "indicateurs.rds"))) {
  message("Saved final metrics found. Rendering PDF only; no model retraining or holdout access.")
  render_saved_report()
  return(invisible(NULL))
}
if (file.exists(file.path(outdir, "FINAL_HOLDOUT_EVALUATION_STARTED.txt")) ||
    file.exists(file.path(outdir, "resume_final.txt"))) {
  stop("Final evaluation was started previously, but saved metrics are missing. ",
       "Do not rerun the holdout. Recover the original result files first.")
}
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
out <- function(filename) file.path(outdir, filename)
RMSE <- function(actual, predicted) {
  stopifnot(length(actual) == length(predicted), all(is.finite(predicted)))
  sqrt(mean((actual - predicted)^2))
}

# 1. DATA PREPARATION --------------------------------------------------
# The split follows the original HarvardX course procedure.
# The holdout construction ensures that its users and movies appear in edx.
prepare_data <- function() {
  existing <- file.exists(c("edx.rds", "final_holdout_test.rds"))
  if (all(existing)) {
    message("Loading the existing edx data. The final holdout remains unopened.")
    return(readRDS("edx.rds"))
  }
  if (any(existing)) {
    stop("Only one dataset RDS file exists. Restore the matching pair created by the official split; do not mix datasets from different runs.")
  }
  message("RDS data files not found: preparing MovieLens 10M...")
  zipfile <- "ml-10M100K.zip"
  download_url <- "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
  # Official MovieLens 10M archive size. A truncated download must not be reused.
  expected_zip_bytes <- 65566137
  required_members <- c("ml-10M100K/ratings.dat", "ml-10M100K/movies.dat")
  archive_ok <- function(path) {
    if (!file.exists(path) || is.na(file.info(path)$size) ||
        file.info(path)$size != expected_zip_bytes) return(FALSE)
    members <- tryCatch(utils::unzip(path, list = TRUE)$Name,
                        error = function(e) character())
    if (!all(required_members %in% members)) return(FALSE)
    # Test compressed-data CRCs when the system unzip utility is available.
    unzip_bin <- Sys.which("unzip")
    if (nzchar(unzip_bin)) {
      status <- suppressWarnings(system2(unzip_bin, c("-tqq", shQuote(path)),
                                         stdout = FALSE, stderr = FALSE))
      if (!identical(as.integer(status), 0L)) return(FALSE)
    }
    TRUE
  }
  if (file.exists(zipfile) && !archive_ok(zipfile)) {
    message("Removing an incomplete or invalid MovieLens ZIP archive.")
    if (!file.remove(zipfile)) stop("Cannot remove invalid archive: ", zipfile)
  }
  if (!file.exists(zipfile)) {
    options(timeout = max(1800, getOption("timeout", 60)))
    # Retry with a clean temporary file. No partial ZIP is kept as the final archive.
    for (attempt in seq_len(3)) {
      temp_zip <- paste0(zipfile, ".part")
      if (file.exists(temp_zip)) file.remove(temp_zip)
      message(sprintf("Downloading MovieLens 10M (attempt %d of 3)...", attempt))
      tryCatch(
        suppressWarnings(utils::download.file(
          download_url, destfile = temp_zip, method = "libcurl",
          mode = "wb", quiet = FALSE)),
        error = function(e) message("Download interrupted: ", conditionMessage(e))
      )
      if (archive_ok(temp_zip)) {
        if (!file.rename(temp_zip, zipfile)) {
          if (!file.copy(temp_zip, zipfile, overwrite = TRUE))
            stop("Unable to save the downloaded archive.")
          file.remove(temp_zip)
        }
        break
      }
      if (file.exists(temp_zip)) file.remove(temp_zip)
      if (attempt < 3) message("Incomplete download; trying again.")
    }
    if (!archive_ok(zipfile)) {
      stop("MovieLens 10M download failed after three attempts. Download the official ",
           "ml-10m.zip using a web browser, save it as ml-10M100K.zip in the ",
           "project folder, then rerun the script. No model evaluation has occurred.")
    }
  }
  # Do not overwrite already extracted, complete files.
  for (f in required_members) {
    if (!file.exists(f)) utils::unzip(zipfile, files = f)
    if (!file.exists(f) || file.info(f)$size == 0)
      stop("File missing or empty after extraction: ", f)
  }
  ratings_raw <- readr::read_lines("ml-10M100K/ratings.dat", progress = FALSE)
  ratings <- as.data.frame(
    stringr::str_split(ratings_raw, stringr::fixed("::"), simplify = TRUE),
    stringsAsFactors = FALSE
  )
  rm(ratings_raw)
  names(ratings) <- c("userId", "movieId", "rating", "timestamp")
  ratings <- ratings %>% mutate(
    userId = as.integer(userId), movieId = as.integer(movieId),
    rating = as.numeric(rating), timestamp = as.integer(timestamp)
  )
  movies_raw <- readr::read_lines("ml-10M100K/movies.dat", progress = FALSE)
  movies <- as.data.frame(
    stringr::str_split(movies_raw, stringr::fixed("::"), simplify = TRUE),
    stringsAsFactors = FALSE
  )
  rm(movies_raw)
  names(movies) <- c("movieId", "title", "genres")
  movies <- movies %>% mutate(movieId = as.integer(movieId))
  movielens <- left_join(ratings, movies, by = "movieId")
  rm(ratings, movies)
  set.seed(1, sample.kind = "Rounding")
  test_index <- caret::createDataPartition(movielens$rating, times = 1,
                                            p = 0.1, list = FALSE)
  edx <- movielens[-test_index, ]
  temp <- movielens[test_index, ]
  holdout <- temp %>% semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  # The following two lines reproduce the supplied course procedure.
  removed <- anti_join(temp, holdout)
  edx <- rbind(edx, removed)
  saveRDS(edx, "edx.rds")
  saveRDS(holdout, "final_holdout_test.rds")
  edx
}
edx <- prepare_data()
# The final_holdout_test.rds file is read only after model selection.

# Structural, range, and integrity checks.
required_cols <- c("userId", "movieId", "rating", "title", "genres")
stopifnot(all(required_cols %in% names(edx)),
          nrow(edx) > 8000000,
          !anyNA(edx$rating),
          all(edx$rating >= 0.5 & edx$rating <= 5))
cat("edx:", nrow(edx), "rows.\n")

# 2. EXPLORATORY ANALYSIS: edx ONLY -----------------------------------
distribution <- edx %>% count(rating, name = "nombre") %>% arrange(rating)
write_csv(distribution, out("distribution_notes.csv"))
films <- edx %>% count(movieId, title, sort = TRUE, name = "nombre")
write_csv(head(films, 20), out("films_plus_notes.csv"))
# Individual genres: a multi-genre movie is counted in each matching genre.
genre_counts <- tibble(genre = c("Drama", "Comedy", "Thriller", "Romance")) %>%
  mutate(evaluations = vapply(genre, function(g) {
    sum(str_detect(edx$genres, fixed(g)), na.rm = TRUE)
  }, numeric(1)))
write_csv(genre_counts, out("genres.csv"))

p_notes <- ggplot(distribution, aes(x = factor(rating), y = nombre)) +
  geom_col(fill = "#32658a") +
  labs(title = "Distribution of ratings — edx", x = "Rating", y = "Count") +
  theme_minimal()
ggsave(out("distribution_notes.png"), p_notes, width = 9, height = 5, dpi = 150)

user_activity <- edx %>% count(userId, name = "nombre_evaluations")
movie_activity <- edx %>% count(movieId, name = "nombre_evaluations")
write_csv(user_activity, out("activite_utilisateurs.csv"))
write_csv(movie_activity, out("activite_films.csv"))
p_users <- ggplot(user_activity, aes(x = nombre_evaluations)) +
  geom_histogram(bins = 50, fill = "#32658a") +
  scale_x_log10() +
  labs(title = "User activity", x = "Ratings per user (log10)",
       y = "Number of users") + theme_minimal()
ggsave(out("activite_utilisateurs.png"), p_users,
       width = 9, height = 5, dpi = 150)
p_movies <- ggplot(movie_activity, aes(x = nombre_evaluations)) +
  geom_histogram(bins = 50, fill = "#32658a") +
  scale_x_log10() +
  labs(title = "Movie popularity", x = "Ratings per movie (log10)",
       y = "Number of movies") + theme_minimal()
ggsave(out("activite_films.png"), p_movies,
       width = 9, height = 5, dpi = 150)

movie_stats <- edx %>% group_by(movieId, title) %>%
  summarise(nombre = n(), note_moyenne = mean(rating), .groups = "drop")
write_csv(movie_stats, out("statistiques_films.csv"))
p_pop <- ggplot(movie_stats, aes(x = nombre, y = note_moyenne)) +
  geom_point(alpha = 0.15, size = 0.7) + scale_x_log10() +
  labs(title = "Movie popularity and average rating",
       x = "Number of ratings (log10)", y = "Average rating") +
  theme_minimal()
ggsave(out("popularite_note_moyenne.png"), p_pop,
       width = 9, height = 5, dpi = 150)

# 3. INTERNAL SPLIT: edx -> train_set / validation_set --------------------
# The final holdout remains entirely excluded from model selection.
set.seed(1, sample.kind = "Rounding")
idx <- createDataPartition(edx$rating, p = 0.2, list = FALSE)
train_set <- edx[-idx, ]
temp <- edx[idx, ]
validation_set <- temp %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
removed <- anti_join(temp, validation_set)
train_set <- bind_rows(train_set, removed)
rm(idx, temp, removed)
stopifnot(nrow(train_set) + nrow(validation_set) == nrow(edx),
          all(validation_set$userId %in% train_set$userId),
          all(validation_set$movieId %in% train_set$movieId))
cat("train_set :", nrow(train_set), "; validation_set :",
    nrow(validation_set), "\n")

# 4. MODELING FUNCTIONS -----------------------------------------------
# Biases are estimated SEQUENTIALLY, not jointly.
# Genres are EXACT COMBINATIONS (e.g., Action|Adventure).
# lambda=0: unregularized biases; lambda>0: shrinkage for sparsely observed groups.
fit_movie <- function(train, lambda = 0) {
  mu <- mean(train$rating)
  bi <- train %>% group_by(movieId) %>%
    summarise(bi = sum(rating - mu) / (n() + lambda), .groups = "drop")
  list(mu = mu, bi = bi, bu = NULL, bg = NULL)
}
fit_biases <- function(train, lambda = 0, use_genres = FALSE) {
  model <- fit_movie(train, lambda)
  tr <- train %>% left_join(model$bi, by = "movieId")
  model$bu <- tr %>% group_by(userId) %>%
    summarise(bu = sum(rating - model$mu - bi) / (n() + lambda),
              .groups = "drop")
  if (use_genres) {
    tr <- tr %>% left_join(model$bu, by = "userId")
    model$bg <- tr %>% group_by(genres) %>%
      summarise(bg = sum(rating - model$mu - bi - bu) / (n() + lambda),
                .groups = "drop")
  }
  model
}
predict_biases <- function(model, newdata,
                           mode = c("global", "movie", "movie_user",
                                    "movie_user_genres")) {
  mode <- match.arg(mode)
  pred <- rep(model$mu, nrow(newdata))
  if (mode == "global") return(pred)
  b <- model$bi$bi[match(newdata$movieId, model$bi$movieId)]
  pred <- pred + replace(b, is.na(b), 0)
  if (mode == "movie") return(pred)
  b <- model$bu$bu[match(newdata$userId, model$bu$userId)]
  pred <- pred + replace(b, is.na(b), 0)
  if (mode == "movie_user_genres") {
    if (is.null(model$bg)) stop("The model has no genre effects.")
    b <- model$bg$bg[match(newdata$genres, model$bg$genres)]
    pred <- pred + replace(b, is.na(b), 0)
  }
  pred
}

# 5. COMPARISON OF THREE BASELINE MODELS ------------------------------------
base <- fit_biases(train_set, lambda = 0, use_genres = FALSE)
results <- tibble(
  modele = c("Global mean", "Movie effect", "Movie + user effects"),
  mode = c("global", "movie", "movie_user"),
  lambda = c(NA_real_, 0, 0),
  RMSE_validation = c(
    RMSE(validation_set$rating,
         predict_biases(base, validation_set, "global")),
    RMSE(validation_set$rating,
         predict_biases(base, validation_set, "movie")),
    RMSE(validation_set$rating,
         predict_biases(base, validation_set, "movie_user"))
  )
)

# 6. REGULARIZATION AND VALIDATION-ONLY SELECTION -------------------
# Use the prespecified grid before final evaluation: 0 to 10 in increments of 0.5.
lambdas <- seq(0, 10, by = 0.5)
tuning <- bind_rows(lapply(lambdas, function(lam) {
  message("Validating lambda = ", lam)
  model <- fit_biases(train_set, lambda = lam, use_genres = TRUE)
  tibble(
    lambda = lam,
    RMSE_regularise = RMSE(validation_set$rating,
                           predict_biases(model, validation_set, "movie_user")),
    RMSE_genres = RMSE(validation_set$rating,
                       predict_biases(model, validation_set,
                                      "movie_user_genres"))
  )
}))
write_csv(tuning, out("recherche_lambda.csv"))
best_regular <- tuning[which.min(tuning$RMSE_regularise), ]
best_genres <- tuning[which.min(tuning$RMSE_genres), ]
results <- bind_rows(
  results,
  tibble(
    modele = c("Regularized", "Regularized + exact genre combinations"),
    mode = c("movie_user", "movie_user_genres"),
    lambda = c(best_regular$lambda, best_genres$lambda),
    RMSE_validation = c(best_regular$RMSE_regularise,
                        best_genres$RMSE_genres)
  )
) %>% arrange(RMSE_validation)
write_csv(results, out("comparaison_validation.csv"))

p_compare <- ggplot(results,
                    aes(x = reorder(modele, RMSE_validation), y = RMSE_validation)) +
  geom_col(fill = "#32658a") + coord_flip() +
  labs(title = "Model comparison on validation set",
       x = NULL, y = "RMSE") + theme_minimal()
ggsave(out("comparaison_rmse.png"), p_compare,
       width = 9, height = 5, dpi = 150)

tuning_long <- bind_rows(
  tuning %>% transmute(lambda, modele = "Movie + user",
                       RMSE = RMSE_regularise),
  tuning %>% transmute(lambda, modele = "Movie + user + genres",
                       RMSE = RMSE_genres)
)
p_lambda <- ggplot(tuning_long, aes(x = lambda, y = RMSE, linetype = modele)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1) +
  labs(title = "Regularization tuning on validation set",
       x = "Lambda", y = "RMSE", linetype = "Model") + theme_minimal()
ggsave(out("optimisation_lambda.png"), p_lambda,
       width = 9, height = 5, dpi = 150)

# 7. REFITTING AND ONE-TIME FINAL EVALUATION ------------------------------
# Selection is complete: refit only the winning model on all of edx.
winner <- results[1, ]
lam <- if (is.na(winner$lambda)) 0 else winner$lambda
if (winner$mode == "global") {
  full_model <- list(mu = mean(edx$rating), bi = NULL, bu = NULL, bg = NULL)
} else if (winner$mode == "movie") {
  full_model <- fit_movie(edx, lambda = lam)
} else {
  full_model <- fit_biases(edx, lambda = lam,
                           use_genres = (winner$mode == "movie_user_genres"))
}
# Do NOT test other models or hyperparameters on final_holdout_test.
# Create a durable one-time evaluation marker BEFORE opening the holdout.
# If the process is interrupted, do not automatically repeat the final test.
# The marker is deliberately NOT removed by this script.
marker <- out("FINAL_HOLDOUT_EVALUATION_STARTED.txt")
if (file.exists(marker)) {
  stop("The final holdout evaluation was already started. Do not run it again for model selection. If a run failed, document and investigate the failure without tuning against the holdout.")
}
writeLines(c(
  paste("Final evaluation started:", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste("Selected model:", as.character(winner$modele)),
  paste("Selected lambda:", lam),
  "This file guards against accidental repeated final testing."
), marker)
# Load final-test labels only after model selection.
final_holdout_test <- readRDS("final_holdout_test.rds")
stopifnot(all(required_cols %in% names(final_holdout_test)),
          nrow(edx) + nrow(final_holdout_test) == 10000054,
          all(final_holdout_test$userId %in% edx$userId),
          all(final_holdout_test$movieId %in% edx$movieId),
          !anyNA(final_holdout_test$rating))
final_predictions <- predict_biases(full_model, final_holdout_test, winner$mode)
stopifnot(length(final_predictions) == nrow(final_holdout_test),
          all(is.finite(final_predictions)))
# No clipping: validation and final-test RMSE use the same prediction rule.
final_rmse <- RMSE(final_holdout_test$rating, final_predictions)
stopifnot(is.finite(final_rmse))

# 8. OUTPUTS AND REPRODUCIBILITY ---------------------------------------------
summary_lines <- c(
  "MOVIELENS 10M PROJECT — HARVARDX DATA SCIENCE",
  paste("Run date:", Sys.Date()),
  paste("edx rows:", nrow(edx)),
  paste("final_holdout_test rows:", nrow(final_holdout_test)),
  paste("train_set rows:", nrow(train_set)),
  paste("validation_set rows:", nrow(validation_set)),
  paste("Selected model:", winner$modele),
  paste("Lambda :", lam),
  sprintf("Validation RMSE: %.6f", winner$RMSE_validation),
  sprintf("Final holdout RMSE: %.6f", final_rmse),
  "Selection performed exclusively on the validation subset of edx.",
  "Genre effect: exact genre combinations; biases estimated sequentially.",
  paste("Mean edx rating:", sprintf("%.4f", mean(edx$rating))),
  paste("Distinct edx users:", n_distinct(edx$userId)),
  paste("Distinct edx movies:", n_distinct(edx$movieId))
)
# Save structured results first; a PDF can be rebuilt without rerunning models.
# The RDS file is a technical intermediate artifact, not a submission deliverable.
saveRDS(list(comparison = results, tuning = tuning, winner = winner,
             rmse_final = final_rmse, rmse_validation = winner$RMSE_validation,
             n_edx = nrow(edx), n_holdout = nrow(final_holdout_test),
             n_train = nrow(train_set), n_validation = nrow(validation_set),
             mean_rating = mean(edx$rating),
             n_users = n_distinct(edx$userId),
             n_movies = n_distinct(edx$movieId)), out("indicateurs.rds"))
writeLines(summary_lines, out("resume_final.txt"))
writeLines(capture.output(sessionInfo()), out("session_info.txt"))
print(results)
cat(paste(summary_lines, collapse = "\n"), "\n")
# End of modeling script.

# 9. AUTOMATIC REPORT COMPILATION --------------------------------------
# The Rmd reads previously generated CSV and PNG files; it does not refit models.
# It does not consult holdout data to select hyperparameters.
render_saved_report()
if (!file.exists("MovieLens_Report.pdf")) stop("The PDF report was not created.")
cat("\nPDF created:", normalizePath("MovieLens_Report.pdf"), "\n")
} # End run_project()
run_project()
