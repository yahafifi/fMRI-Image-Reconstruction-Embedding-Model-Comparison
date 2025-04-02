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
SAVED_KNN_MODELS_PATH = os.path.join(OUTPUT_BASE_PATH, 'knn_models')
GENERATED_IMAGES_PATH = os.path.join(OUTPUT_BASE_PATH, 'generated_images')
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_BASE_PATH, 'evaluation_results')


# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Handling ---
SUBJECT_ID = "3" # Subject for GOD dataset
ROI = "ROI_VC" # Region of Interest
TEST_SPLIT_SIZE = 0.1 # Proportion of training data used for validation
RANDOM_STATE = 42
MAPPING_BATCH_SIZE = 128 # Batch size specifically for training the mapping model
MAPPING_NUM_WORKERS = 2

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
        "repo_id": "openai/clip-vit-base-patch16",
        "preprocessor": "openai/clip-vit-base-patch16",
        "embedding_dim": 512 # Output dim of image encoder
    }
}
TARGET_IMAGE_SIZE = 224
FEATURE_EXTRACTION_BATCH_SIZE = 64  # Batch size for extracting image features
FEATURE_EXTRACTION_NUM_WORKERS = 2

# --- Mapping Model Selection ---
# Choose 'ridge' or 'mlp'
MAPPING_MODEL_TYPE = 'mlp' # <--- SELECT MAPPING MODEL HERE

# --- Ridge Regression Config (if MAPPING_MODEL_TYPE='ridge') ---
RIDGE_ALPHA = 10000.0 # Might need tuning if R2 is poor
RIDGE_MAX_ITER = 10000 # Increase iterations

# --- Deep Residual SE MLP Config (if MAPPING_MODEL_TYPE='mlp') ---
MLP_HIDDEN_DIMS = [4096, 4096, 4096, 4096] # Example: Four deep layers, adjust based on VRAM/performance
MLP_ACTIVATION = torch.nn.GELU # GELU is often good in transformer-like contexts
MLP_USE_LAYER_NORM = True
MLP_DROPOUT_RATE = 0.25 # Regularization
MLP_USE_SE_BLOCK = True
MLP_SE_REDUCTION = 16 # Squeeze-and-Excitation reduction ratio (common value)
MLP_LEARNING_RATE = 1e-4 # Starting learning rate
MLP_WEIGHT_DECAY = 1e-5 # AdamW weight decay
MLP_EPOCHS = 150 # Number of training epochs (adjust based on convergence)
MLP_OPTIMIZER = 'AdamW' # AdamW is generally preferred
MLP_SCHEDULER = 'CosineAnnealingLR' # Learning rate scheduler
MLP_SCHEDULER_T_MAX = MLP_EPOCHS # T_max for CosineAnnealingLR often set to total epochs
MLP_EARLY_STOPPING_PATIENCE = 15 # Stop if validation R2 doesn't improve for N epochs
MLP_GRAD_CLIP_VALUE = 1.0 # Gradient clipping to prevent exploding gradients

# --- Retrieval (k-NN) ---
KNN_N_NEIGHBORS = 5
KNN_METRIC = 'cosine' # Cosine is good for embeddings
KNN_ALGORITHM = 'brute' # Start with brute, change to 'auto' or 'hnsw' (requires faiss) if slow/RAM issue

# --- Generation (Stable Diffusion) ---
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_GUIDANCE_SCALE = 7.5
NUM_GENERATION_SAMPLES = 50 # Generate for all 50 test samples
NUM_INFERENCE_STEPS = 25 # Diffusion inference steps

# --- Evaluation ---
EVAL_METRICS = ['ssim', 'clip_sim', 'lpips'] # Add more if needed
CLIP_SCORE_MODEL_NAME = "ViT-B-32" # For CLIP-based similarity scoring
CLIP_SCORE_PRETRAINED = "openai"

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

print(f"--- Configuration ---")
print(f"Using device: {DEVICE}")
print(f"Output base path: {OUTPUT_BASE_PATH}")
print(f"Using ImageNet256 path: {IMAGENET256_PATH}")
print(f"Selected Mapping Model: {MAPPING_MODEL_TYPE.upper()}")
print(f"--------------------")