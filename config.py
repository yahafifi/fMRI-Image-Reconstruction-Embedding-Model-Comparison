# config.py
import torch
import os

# --- Paths ---
# Adjust these paths based on your Kaggle environment or local setup
KAGGLE_BASE_PATH = '/kaggle/working' # Output directory
INPUT_BASE_PATH = '/kaggle/input'    # Input dataset directory base

DATA_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, 'data/fMRI')
MODELS_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, 'models')
OUTPUT_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, 'output')

# GOD Dataset Paths
GOD_FMRI_PATH = os.path.join(DATA_BASE_PATH, 'GOD')
# Ensure this path points to the directory containing 'training' and 'test' subdirs with GOD stimuli images
GOD_IMAGENET_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/images')
GOD_FEATURES_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/features') # Precomputed features if needed

# --- ImageNet-256 Path (Used for Retrieval Database - Optional, not needed for direct generation) ---
# Adjust dataset name if needed based on exact Kaggle path
IMAGENET256_DATASET_NAME = 'imagenet-256'
# Make sure this path is correct if you ever uncomment retrieval steps
IMAGENET256_PATH = '/kaggle/input/imagenet-256'
IMAGENET256_FEATURES_PATH = os.path.join(OUTPUT_BASE_PATH, 'imagenet256_features')

# --- Other File Paths (Ensure they exist if used) ---
# CLASS_TO_WORDNET_JSON = os.path.join(KAGGLE_BASE_PATH, "class_to_wordnet.json") # Not used in current pipeline

# --- Output Directories ---
SAVED_KNN_MODELS_PATH = os.path.join(OUTPUT_BASE_PATH, 'knn_models') # For optional retrieval models
GENERATED_IMAGES_PATH = os.path.join(OUTPUT_BASE_PATH, 'generated_images')
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_BASE_PATH, 'evaluation_results')

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Handling ---
SUBJECT_ID = "3" # Subject for GOD dataset
ROI = "ROI_VC" # Region of Interest (~4500 voxels for VC)
TEST_SPLIT_SIZE = 0.1 # Use 10% of training data for validation
RANDOM_STATE = 42

# --- Feature Extraction (Ensure 'clip' uses the 768D model) ---
EMBEDDING_MODELS = {
    "resnet50": {
        "repo_id": "torchvision",
        "preprocessor": None,
        "embedding_dim": 2048
    },
    "vit": {
        "repo_id": "google/vit-base-patch16-224-in21k",
        "preprocessor": "google/vit-base-patch16-224-in21k",
        "embedding_dim": 768
    },
    "clip": {
        "repo_id": "openai/clip-vit-large-patch14", # 768D model needed for SD v1.5
        "preprocessor": "openai/clip-vit-large-patch14",
        "embedding_dim": 768 # *** MUST BE 768 ***
    }
}
TARGET_IMAGE_SIZE = 224 # Input size for embedding models
# BATCH_SIZE used for feature extraction and evaluation dataloaders
BATCH_SIZE = 64
NUM_WORKERS = 2 # For dataloaders

# --- Mapping Model ---
# === Ridge Regression (Optional Baseline) ===
RIDGE_ALPHA = 1000.0
RIDGE_MAX_ITER = 5000

# === Advanced MLP Config ===
# --- Architecture ---
MLP_HIDDEN_DIM = 4096       # Main hidden dimension for layers
MLP_NUM_RESIDUAL_BLOCKS = 6 # Number of residual blocks (adjust for depth)
MLP_SE_REDUCTION_RATIO = 16 # Reduction ratio for Squeeze-and-Excitation blocks
MLP_DROPOUT_RATE = 0.5      # Dropout probability

# --- Training ---
MLP_LEARNING_RATE = 1e-4      # Starting learning rate
MLP_WEIGHT_DECAY = 1e-5       # Weight decay for AdamW optimizer
MLP_EPOCHS = 150              # Maximum training epochs
MLP_BATCH_SIZE = 64           # Training batch size (adjust based on GPU memory)
MLP_GRADIENT_CLIP_NORM = 1.0  # Max norm for gradient clipping (0 to disable)
MLP_SCHEDULER_TYPE = 'cosine' # 'cosine' or 'reduce_on_plateau'
MLP_COSINE_T_MAX = MLP_EPOCHS # For CosineAnnealingLR, T_max controls the cycle length
MLP_PLATEAU_FACTOR = 0.2      # For ReduceLROnPlateau factor if used
MLP_PLATEAU_PATIENCE = 5      # For ReduceLROnPlateau patience if used
MLP_EARLY_STOPPING_PATIENCE = 15 # Patience for early stopping based on validation loss

# --- Retrieval (k-NN - Optional, Not Used in Direct Generation) ---
KNN_N_NEIGHBORS = 5
KNN_METRIC = 'cosine'
KNN_ALGORITHM = 'brute'

# --- Generation (Stable Diffusion) ---
# Ensure this model's text encoder matches the embedding dim (768 for SD v1.5)
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_GUIDANCE_SCALE = 7.5 # CFG scale
NUM_GENERATION_STEPS = 30 # Number of diffusion steps

# --- Evaluation ---
EVAL_METRICS = ['ssim', 'lpips', 'clip_sim'] # Reconstruction metrics
CLIP_SCORE_MODEL_NAME = "ViT-L-14" # Use matching CLIP model for scoring if possible
CLIP_SCORE_PRETRAINED = "openai"   # Source for the scoring CLIP model

# --- Utility: Ensure output directories exist ---
# Use try-except for robustness, esp. in read-only environments for input paths
try:
    os.makedirs(IMAGENET256_FEATURES_PATH, exist_ok=True)
    os.makedirs(SAVED_KNN_MODELS_PATH, exist_ok=True)
    os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
    os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)
    os.makedirs(MODELS_BASE_PATH, exist_ok=True)
    # Data paths are usually inputs, but create if base path allows writing
    os.makedirs(GOD_FMRI_PATH, exist_ok=True)
    os.makedirs(GOD_IMAGENET_PATH, exist_ok=True)
    os.makedirs(GOD_FEATURES_PATH, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create directory {e.filename}. Error: {e.strerror}")


print("-" * 50)
print(f"--- Configuration ---")
print(f"Device: {DEVICE}")
print(f"Output Base Path: {OUTPUT_BASE_PATH}")
print(f"Models Base Path: {MODELS_BASE_PATH}")
print(f"Using Embedding Model for Mapping: clip (Dim: {EMBEDDING_MODELS['clip']['embedding_dim']})")
print(f"Using Stable Diffusion Model: {STABLE_DIFFUSION_MODEL_ID}")
print("-" * 50)