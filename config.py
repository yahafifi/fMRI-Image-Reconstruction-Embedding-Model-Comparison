# config.py
 import torch
 import os
 
 # --- Paths ---
 # Adjust these paths based on your Kaggle environment or local setup
 KAGGLE_BASE_PATH = '/kaggle/working'
 DATA_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, 'data/fMRI')
 MODELS_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, 'models')
 OUTPUT_BASE_PATH = os.path.join(KAGGLE_BASE_PATH, 'output')
 
 # GOD Dataset Paths
 GOD_FMRI_PATH = os.path.join(DATA_BASE_PATH, 'GOD')
 GOD_IMAGENET_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/images') # Contains train/test stimuli for GOD
 GOD_FEATURES_PATH = os.path.join(DATA_BASE_PATH, 'imagenet/features') # Precomputed features if needed
 
 # Tiny ImageNet Path (Used for Retrieval Database)
 TINY_IMAGENET_PATH = "/kaggle/input/tiny-imagenet/tiny-imagenet-200"
 TINY_IMAGENET_FEATURES_PATH = os.path.join(OUTPUT_BASE_PATH, 'tiny_imagenet_features')
 
 # Other Files
 CLASS_TO_WORDNET_JSON = os.path.join(KAGGLE_BASE_PATH, "class_to_wordnet.json") # From GOD dataset context
 TINY_IMAGENET_WORDS_TXT = os.path.join(TINY_IMAGENET_PATH, "words.txt")
 TINY_IMAGENET_WNIDS_TXT = os.path.join(TINY_IMAGENET_PATH, "wnids.txt")
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
         "repo_id": "openai/clip-vit-base-patch16",
         "preprocessor": "openai/clip-vit-base-patch16",
         "embedding_dim": 512 # Output dim of image encoder
     }
 }
 TARGET_IMAGE_SIZE = 224
 BATCH_SIZE = 64
 NUM_WORKERS = 2
 
 # --- Mapping Model (Ridge Regression) ---
 RIDGE_ALPHA = 1000.0
 RIDGE_MAX_ITER = 5000
 
 # --- Retrieval (k-NN) ---
 KNN_N_NEIGHBORS = 5
 
 # --- Generation (Stable Diffusion) ---
 # Using runwayml/stable-diffusion-v1-5 as it's widely available
 # Changed from CompVis/stable-diffusion-v1-4 in original code
 # User can change this if needed
 STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
 STABLE_DIFFUSION_GUIDANCE_SCALE = 7.5
 NUM_GENERATION_SAMPLES = 50 # Generate for all 50 test samples
 
 # --- Evaluation ---
 EVAL_METRICS = ['ssim', 'clip_sim', 'lpips'] # Add more if needed
 CLIP_SCORE_MODEL_NAME = "ViT-B-32" # For CLIP-based similarity scoring
 CLIP_SCORE_PRETRAINED = "openai"
 
 # --- Utility ---
 # Ensure output directories exist
 os.makedirs(TINY_IMAGENET_FEATURES_PATH, exist_ok=True)
 os.makedirs(SAVED_KNN_MODELS_PATH, exist_ok=True)
 os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
 os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)
 os.makedirs(MODELS_BASE_PATH, exist_ok=True) # For saving Ridge models
 os.makedirs(GOD_FMRI_PATH, exist_ok=True)
 os.makedirs(GOD_IMAGENET_PATH, exist_ok=True)
 os.makedirs(GOD_FEATURES_PATH, exist_ok=True)
 
 print(f"Using device: {DEVICE}")
 print(f"Output base path: {OUTPUT_BASE_PATH}")
