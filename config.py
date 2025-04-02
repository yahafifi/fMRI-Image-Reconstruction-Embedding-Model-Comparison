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
GOD_IMAGENET_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/images') # Should contain train/test folders & CSVs after download/org
GOD_FEATURES_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/features') # Precomputed features if needed

# --- ImageNet-256 Path (Used for Retrieval Database) ---
# Ensure this matches the Kaggle dataset name or path after download
IMAGENET256_DATASET_NAME = 'imagenet-object-localization-challenge/imagenet_object_localization_data' # Example - VERIFY/ADJUST THIS SLUG!
# IMAGENET256_DATASET_NAME = 'ifigedy/imagenetmini-1000' # Example Mini - VERIFY/ADJUST!
IMAGENET256_PATH = '/kaggle/input/imagenet-256' # Path where the class folders reside - VERIFY/ADJUST!
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
MAPPING_BATCH_SIZE = 64 # Batch size for training the mapping model (Reduce if OOM)
MAPPING_NUM_WORKERS = 2

# --- Feature Extraction ---
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
        "repo_id": "openai/clip-vit-base-patch16",
        "preprocessor": "openai/clip-vit-base-patch16",
        "embedding_dim": 512
    }
}
TARGET_IMAGE_SIZE = 224
FEATURE_EXTRACTION_BATCH_SIZE = 64  # Batch size for extracting image features (GOD & ImageNet256)
FEATURE_EXTRACTION_NUM_WORKERS = 2

# --- Mapping Model Selection ---
MAPPING_MODEL_TYPE = 'mlp' # Keep MLP for now, but with adjusted params

# --- Ridge Regression Config (if MAPPING_MODEL_TYPE='ridge') ---
RIDGE_ALPHA = 10000.0
RIDGE_MAX_ITER = 10000

# --- Deep Residual SE MLP Config (if MAPPING_MODEL_TYPE='mlp') ---
# *** REDUCED MODEL SIZE & INCREASED REGULARIZATION ***
MLP_HIDDEN_DIMS = [2048, 1024] # Significantly smaller
MLP_ACTIVATION = torch.nn.GELU
MLP_USE_LAYER_NORM = True
MLP_DROPOUT_RATE = 0.45 # Increased dropout
MLP_USE_SE_BLOCK = True # Keep SE for now, can test disabling later
MLP_SE_REDUCTION = 16
MLP_LEARNING_RATE = 5e-5 # Slightly reduced LR
MLP_WEIGHT_DECAY = 1e-4 # Increased weight decay
MLP_EPOCHS = 150 # Keep epochs, rely on early stopping
MLP_OPTIMIZER = 'AdamW'
MLP_SCHEDULER = 'CosineAnnealingLR'
MLP_SCHEDULER_T_MAX = MLP_EPOCHS
MLP_EARLY_STOPPING_PATIENCE = 20 # Slightly increased patience
MLP_GRAD_CLIP_VALUE = 1.0

# --- Retrieval (k-NN) ---
KNN_N_NEIGHBORS = 5
KNN_METRIC = 'cosine'
KNN_ALGORITHM = 'brute'

# --- Generation (Stable Diffusion) ---
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
STABLE_DIFFUSION_GUIDANCE_SCALE = 7.5
# NUM_GENERATION_SAMPLES = 50 # Determined by test set size automatically
NUM_INFERENCE_STEPS = 25

# --- Evaluation ---
EVAL_METRICS = ['ssim', 'clip_sim', 'lpips']
CLIP_SCORE_MODEL_NAME = "ViT-B-32"
CLIP_SCORE_PRETRAINED = "openai"

# --- Utility ---
# Ensure output directories exist (done in download_data.py now)
# os.makedirs(IMAGENET256_FEATURES_PATH, exist_ok=True)
# ... other makedirs ...

print(f"--- Configuration ---")
print(f"Using device: {DEVICE}")
print(f"Output base path: {OUTPUT_BASE_PATH}")
print(f"GOD fMRI Path: {GOD_FMRI_PATH}")
print(f"GOD Imagenet Path: {GOD_IMAGENET_PATH}")
print(f"ImageNet256 Path: {IMAGENET256_PATH} (Dataset Slug: {IMAGENET256_DATASET_NAME})")
print(f"Selected Mapping Model: {MAPPING_MODEL_TYPE.upper()}")
print(f"MLP Hidden Dims: {MLP_HIDDEN_DIMS if MAPPING_MODEL_TYPE == 'mlp' else 'N/A'}")
print(f"MLP Dropout: {MLP_DROPOUT_RATE if MAPPING_MODEL_TYPE == 'mlp' else 'N/A'}")
print(f"MLP Weight Decay: {MLP_WEIGHT_DECAY if MAPPING_MODEL_TYPE == 'mlp' else 'N/A'}")
print(f"MLP Learning Rate: {MLP_LEARNING_RATE if MAPPING_MODEL_TYPE == 'mlp' else 'N/A'}")
print(f"--------------------")