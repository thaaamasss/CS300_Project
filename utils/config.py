# -----------------------------------------------------
# DATASET SETTINGS
# -----------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = 2

# -----------------------------------------------------
# TRAINING SETTINGS
# -----------------------------------------------------
EPOCHS = 10
SGD_LEARNING_RATE = 0.01
ADAM_LEARNING_RATE = 0.001
RMSPROP_LEARNING_RATE = 0.001
RANDOM_SEED = 42

# -----------------------------------------------------
# SISA TRAINING SETTINGS
# -----------------------------------------------------
NUM_SHARDS = 5

# Phase 1 fix: NUM_SLICES is now wired into sisa_training.py
# Set to 1 for original behaviour (no slices), 3-5 for true SISA with slices.
NUM_SLICES = 3

# -----------------------------------------------------
# DELETION SETTINGS
# -----------------------------------------------------
DELETE_SAMPLES = 500
DELETE_PERCENTAGE = 10

# -----------------------------------------------------
# MODEL SAVING SETTINGS
# -----------------------------------------------------
MODEL_SAVE_DIR = "saved_models"

# -----------------------------------------------------
# RESULTS DIRECTORY SETTINGS
# -----------------------------------------------------
RESULTS_DIR = "results"
CSV_RESULTS_DIR = "results/csv_results"
PLOTS_DIR = "results/plots"

# -----------------------------------------------------
# DEVICE SETTINGS
# -----------------------------------------------------
DEVICE = "cuda"

# -----------------------------------------------------
# LOGGING SETTINGS
# -----------------------------------------------------
VERBOSE = True

# -----------------------------------------------------
# INFLUENCE UNLEARNING SETTINGS (Phase 1)
# -----------------------------------------------------
# Conjugate gradient iterations — higher = better H^-1 approximation, slower.
# 20 is a good default. Try 10 for speed, 30-50 for more accuracy.
INFLUENCE_CG_ITERATIONS = 20

# Damping for (H + lambda*I) — prevents CG divergence on near-singular H.
# 1e-2 is conservative. Lower (1e-3) if model is well-conditioned.
INFLUENCE_CG_DAMPING = 1e-2

# Number of remaining samples used for HVP estimation.
# 1000 gives stable estimates. Reduce to 500 if memory is tight.
INFLUENCE_CG_SAMPLES = 1000

# Learning rate for applying the CG influence update.
# Much smaller than training lr — the CG vector already encodes curvature.
INFLUENCE_UPDATE_LR = 1e-4

# -----------------------------------------------------
# PER-LAYER FREEZING SETTINGS (Phase 1)
# -----------------------------------------------------
# When True: Conv1+Conv2 frozen during FineTuning and SISA fallback unlearning.
# Selective forgetting — only FC1+FC2 (class-specific layers) are updated.
# Set False to restore original behaviour (all layers updated).
UNLEARNING_FREEZE_BACKBONE = True