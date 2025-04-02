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
GOD_IMAGENET_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/images') # Contains train/test stimuli for GOD
GOD_FEATURES_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/features') # Precomputed features if needed

# --- ImageNet-256 Path (Used for Retrieval Database) ---
# Adjust dataset name if needed based on exact Kaggle path
IMAGENET256_DATASET_NAME = 'imagenet-256'
IMAGENET256_PATH = '/kaggle/input/imagenet-256' # Path to the directory containing class folders
IMAGENET256_FEATURES_PATH = os.path.join(OUTPUT_BASE_PATH, 'imagenet256_features')

# Other Files
CLASS_TO_WORDNET_JSON = os.path.join(KAGGLE_BASE_PATH, "class_to_wordnet.json") # From GOD dataset context (keep if GOD uses WordNet IDs)
# TINY_IMAGENET_WORDS_TXT = os.path.join(TINY_IMAGENET_PATH, "words.txt") # REMOVED
# TINY_IMAGENET_WNIDS_TXT = os.path.join(TINY_IMAGENET_PATH, "wnids.txt") # REMOVED
SAVED_KNN_MODELS_PATH = os.path.join(OUTPUT_BASE_PATH, 'knn_models')
GENERATED_IMAGES_PATH = os.path.join(OUTPUT_BASE_PATH, 'generated_images')
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_BASE_PATH, 'evaluation_results')


# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Handling ---
SUBJECT_ID = "3" # Subject for GOD dataset
ROI = "ROI_VC" # Region of Interest
TEST_SPLIT_SIZE = 0.1
RANDOM_STATE = 42

# --- Feature Extraction ---
EMBEDDING_MODELS = {
    "resnet50": {
        "repo_id": "torchvision", # Indicator for torchvision
        "preprocessor": None, # Will use standard transforms
        "embedding_dim": 2048 # Output dim of avgpool layer
    },
    "vit": {
        "repo_id": "google/vit-base-patch16-224-in21k",
        "preprocessor": "google/vit-base-patch16-224-in21k",
        "embedding_dim": 768 # Output dim of pooler_output
    },
    "clip": {
        # --- MODIFIED ---
        "repo_id": "openai/clip-vit-large-patch14", # Changed to Large model
        "preprocessor": "openai/clip-vit-large-patch14", # Changed preprocessor to match
        "embedding_dim": 768 # <<<--- UPDATED DIMENSION ---<<<
        # --- END MODIFIED ---
    }
}
TARGET_IMAGE_SIZE = 224
BATCH_SIZE = 64  # Keep moderate for feature extraction RAM
NUM_WORKERS = 2

# --- Mapping Model (Ridge Regression) ---
RIDGE_ALPHA = 1000.0
RIDGE_MAX_ITER = 5000
# --- Suggestion: Add MLP config here later ---
# MLP_HIDDEN_DIMS = [2048, 2048]
# MLP_LEARNING_RATE = 1e-4
# MLP_EPOCHS = 50
# MLP_WEIGHT_DECAY = 1e-3
# MLP_DROPOUT = 0.5


# --- Retrieval (k-NN) ---
KNN_N_NEIGHBORS = 5
KNN_METRIC = 'cosine' # Cosine is good for embeddings
KNN_ALGORITHM = 'brute' # Start with brute for correctness, monitor RAM. Change to 'auto' if needed later.

# --- Generation (Stable Diffusion) ---
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_GUIDANCE_SCALE = 7.5
NUM_GENERATION_SAMPLES = 50 # Generate for all 50 test samples
# --- Suggestion: Add config for direct embedding guidance later ---
# SD_GUIDANCE_TYPE = 'text' # or 'embedding'

# --- Evaluation ---
EVAL_METRICS = ['ssim', 'clip_sim', 'lpips'] # Add more if needed
CLIP_SCORE_MODEL_NAME = "ViT-B-32" # For CLIP-based similarity scoring
CLIP_SCORE_PRETRAINED = "openai"
# --- Suggestion: Add semantic classification metric ---
# EVAL_CLASSIFIER_MODEL = "resnet50" # Model to classify generated images

# --- Utility ---
# Ensure output directories exist
os.makedirs(IMAGENET256_FEATURES_PATH, exist_ok=True) # UPDATED
os.makedirs(SAVED_KNN_MODELS_PATH, exist_ok=True)
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_BASE_PATH, exist_ok=True) # For saving Ridge/MLP models
os.makedirs(GOD_FMRI_PATH, exist_ok=True)
os.makedirs(GOD_IMAGENET_PATH, exist_ok=True)
os.makedirs(GOD_FEATURES_PATH, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Output base path: {OUTPUT_BASE_PATH}")
print(f"Using ImageNet256 path: {IMAGENET256_PATH}") # Added print
