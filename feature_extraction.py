# feature_extraction.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import ViTModel, ViTImageProcessor, CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import tqdm
import glob
import time # Added for timing

import config
from data_loading import get_default_image_transform # Reuse standard transform

# --- Model Loading ---
def load_embedding_model(model_name):
    """Loads the specified pre-trained model and its preprocessor."""
    if model_name not in config.EMBEDDING_MODELS:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from {list(config.EMBEDDING_MODELS.keys())}")

    model_info = config.EMBEDDING_MODELS[model_name]
    repo_id = model_info["repo_id"]
    device = config.DEVICE

    model = None

    print(f"Loading model: {model_name}...")
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer, use features from avgpool
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = model.to(device).eval()
        # Processor will be the standard transform applied in Dataset
    elif model_name == "vit":
        model = ViTModel.from_pretrained(repo_id).to(device).eval()
        # Processor will be handled later or assumes tensor input
    elif model_name == "clip":
        model = CLIPModel.from_pretrained(repo_id).to(device).eval()
        # Processor will be handled later or assumes tensor input

    print(f"Model {model_name} loaded.")
    # Note: We don't return the processor here anymore, as it's handled inside extract_features
    # or assumed to be part of the dataset's transform.
    return model, None # Return None for processor for now


# --- Feature Extraction Function ---
@torch.no_grad() # Disable gradient calculations for efficiency
def extract_features(model, dataloader, model_name, device):
    """Extracts features/embeddings from a model for all data in a dataloader."""
    model.eval() # Ensure model is in evaluation mode
    all_features = []
    all_fmri_data = [] # Store corresponding fMRI if available in dataloader

    has_fmri = hasattr(dataloader.dataset, 'fmri_data') # Check if it's our FmriImageDataset

    # --- Pre-fetch processors if needed (though we assume tensors for now) ---
    # This section is mostly for reference if you switch dataset to return PIL for ViT/CLIP
    # hf_processor_vit = None
    # hf_processor_clip = None
    # if model_name == "vit":
    #     hf_processor_vit = ViTImageProcessor.from_pretrained(config.EMBEDDING_MODELS["vit"]["preprocessor"])
    # elif model_name == "clip":
    #     hf_processor_clip = CLIPProcessor.from_pretrained(config.EMBEDDING_MODELS["clip"]["preprocessor"])
    # --------------------------------------------------------------------------

    print(f"Extracting {model_name} features...")
    for batch in tqdm.tqdm(dataloader, desc=f"Extracting {model_name}"):
        if has_fmri:
            fmri_batch, data_batch = batch # data_batch should be image tensors
            all_fmri_data.append(fmri_batch.cpu().numpy()) # Store fMRI data
        else: # Assuming dataloader yields (image_tensor, label)
            data_batch, _ = batch # We only need the image tensor
            
        # --- Assume data_batch is already the correct tensor format ---
        # This relies on the Dataset applying the appropriate transform (get_default_image_transform)
        images = data_batch.to(device)

        # --- Alternative if Dataset returned PIL images (more complex batching needed) ---
        # if model_name == "resnet50":
        #     images = data_batch.to(device) # Assuming tensors came from dataset
        # elif model_name == "vit":
        #     # Requires collating PIL images and processing list
        #     processed_batch = torch.stack([hf_processor_vit(images=pil_img, return_tensors="pt").pixel_values.squeeze(0) for pil_img in data_batch]).to(device)
        #     images = processed_batch
        # elif model_name == "clip":
        #      processed_batch = torch.stack([hf_processor_clip(images=pil_img, return_tensors="pt").pixel_values.squeeze(0) for pil_img in data_batch]).to(device)
        #      images = processed_batch
        # ----------------------------------------------------------------------------------

        # Get embeddings based on model type
        if model_name == "resnet50":
            features = model(images) # Output of Sequential model (after avgpool)
            features = torch.flatten(features, 1) # Flatten the output
        elif model_name == "vit":
            outputs = model(pixel_values=images)
            features = outputs.pooler_output # Use the pooled output [batch_size, embedding_dim]
        elif model_name == "clip":
            # CLIP image encoder part
            outputs = model.get_image_features(pixel_values=images)
            features = outputs # Output is already [batch_size, embedding_dim]

        all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    if has_fmri:
         all_fmri_data = np.concatenate(all_fmri_data, axis=0)
         print(f"Finished extraction. Features shape: {all_features.shape}, fMRI shape: {all_fmri_data.shape}")
         return all_fmri_data, all_features
    else:
        print(f"Finished extraction. Features shape: {all_features.shape}")
        return all_features # Return only features if no fMRI data


# --- ImageNet-256 Dataset ---
class ImageNet256Dataset(Dataset):
    """ Loads images from the ImageNet-256 dataset structure """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = [] # Numerical labels
        self.class_names = [] # Human-readable names (folder names)
        self.class_to_idx = {}
        self.idx_to_class = {}

        print(f"Loading ImageNet256 file list from: {root_dir}")
        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"ImageNet-256 directory not found at {root_dir}")

        # Get class names (folder names) and sort them for consistent indexing
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if not self.class_names:
             raise FileNotFoundError(f"No class subdirectories found in {root_dir}")

        print(f"Found {len(self.class_names)} classes.")

        # Build mappings and file lists
        for idx, class_name in enumerate(tqdm.tqdm(self.class_names, desc="Scanning classes")):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            class_dir = os.path.join(root_dir, class_name)
            # Find image files (assuming .jpg, adjust if needed)
            # Consider adding more extensions if necessary (like .JPEG)
            image_files = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                          glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                          glob.glob(os.path.join(class_dir, '*.png'))

            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(idx) # Assign the numerical index

        print(f"Found {len(self.image_paths)} images in total.")
        if len(self.image_paths) == 0:
             print(f"Warning: No images found. Check dataset path and file extensions.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Load as PIL image initially
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning dummy image.")
            # Return a dummy PIL image matching target size for transform consistency
            try:
                # Use TARGET_IMAGE_SIZE from config
                size = (config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE)
                image = Image.new('RGB', size, color = 'grey')
            except NameError: # Fallback if config not fully loaded yet during testing
                image = Image.new('RGB', (224, 224), color = 'grey')
            # label remains the same

        # Apply transforms (which should include ToTensor and normalization)
        # We apply the standard transform here for all models when processing ImageNet256
        # This ensures consistency for the k-NN database features.
        if self.transform:
            try:
                image = self.transform(image) # Should output a Tensor
            except Exception as e:
                 print(f"Error applying transform to {img_path}: {e}. Returning dummy tensor.")
                 # Create dummy tensor matching expected output shape
                 image = torch.zeros((3, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))

        return image, label # Return Tensor and numeric label

    def get_readable_label(self, idx):
         """ Get human-readable label for a given sample index """
         numeric_label = self.labels[idx]
         return self.idx_to_class.get(numeric_label, f"UnknownLabel_{numeric_label}")


# --- ImageNet-256 Feature Precomputation ---
def precompute_imagenet256_features(model_name):
    """Loads ImageNet-256, extracts features using the specified model, and saves them."""
    feature_file = os.path.join(config.IMAGENET256_FEATURES_PATH, f"imagenet256_features_{model_name}.npy")
    labels_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_labels.npy") # Labels are same for all models
    class_map_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_idx_to_class.npy") # idx -> readable name

    # Check if features specific to this model exist
    if os.path.exists(feature_file):
        print(f"ImageNet-256 {model_name} features already exist: {feature_file}")
        features = np.load(feature_file)
        # Load labels and map if they exist, otherwise generate them below
        if os.path.exists(labels_file) and os.path.exists(class_map_file):
             labels = np.load(labels_file)
             class_map = np.load(class_map_file, allow_pickle=True).item() # Load the dictionary
             return features, labels, class_map
        else:
             print("Labels or class map file missing, will regenerate...")

    print(f"Precomputing ImageNet-256 features for {model_name}...")
    start_time = time.time()
    # Pass model_name to load_embedding_model
    model, _ = load_embedding_model(model_name)
    if model is None:
        print(f"Failed to load model {model_name}. Cannot extract features.")
        return None, None, None

    # Use the *standard* transform for consistency across models for the database
    # Ensure the dataset applies this transform
    transform = get_default_image_transform(config.TARGET_IMAGE_SIZE)

    try:
        dataset = ImageNet256Dataset(config.IMAGENET256_PATH, transform=transform)
        if len(dataset) == 0:
             print("Dataset is empty. Cannot extract features.")
             return None, None, None
    except FileNotFoundError as e:
         print(f"Error initializing ImageNet256Dataset: {e}")
         print(f"Please ensure config.IMAGENET256_PATH ('{config.IMAGENET256_PATH}') points to the correct directory containing class folders.")
         return None, None, None


    dataloader = DataLoader(dataset,
                            batch_size=config.BATCH_SIZE, # Adjust batch size based on GPU memory
                            shuffle=False,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    # Extract features (returns only features array)
    features = extract_features(model, dataloader, model_name, config.DEVICE)

    # Get labels and class map directly from the dataset instance
    labels = np.array(dataset.labels) # Get all numerical labels
    class_map = dataset.idx_to_class # Get the {index: class_name} dictionary

    # Save features, labels, and class map
    try:
        # Ensure the directory exists before saving
        os.makedirs(config.IMAGENET256_FEATURES_PATH, exist_ok=True)

        np.save(feature_file, features)
        # Save labels and class map only if they don't exist already (or overwrite if regeneration was needed)
        if not os.path.exists(labels_file) or not os.path.exists(class_map_file):
            np.save(labels_file, labels)
            np.save(class_map_file, class_map) # Save the dictionary {idx: class_name}

        end_time = time.time()
        print(f"Saved ImageNet-256 {model_name} features ({features.shape}) to {feature_file}")
        print(f"Saved ImageNet-256 labels ({labels.shape}) to {labels_file}")
        print(f"Saved ImageNet-256 class map ({len(class_map)} entries) to {class_map_file}")
        print(f"Feature extraction took {(end_time - start_time)/60:.2f} minutes.")

    except Exception as e:
        print(f"Error saving features/labels/map: {e}")
        # Avoid returning potentially inconsistent data if saving failed
        return None, None, None


    return features, labels, class_map


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Feature Extraction ---")

    # --- Test GOD Data Feature Extraction ---
    # This part requires data_loading.py and GOD data to be present
    # try:
    #     print("\n--- Testing GOD Data Feature Extraction ---")
    #     handler = data_loading.GodFmriDataHandler(config.SUBJECT_ID, config.ROI, config.GOD_FMRI_PATH, config.GOD_IMAGENET_PATH)
    #     data_splits = handler.get_data_splits(normalize_runs=True, test_split_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
    #     image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)
    #     dataloaders = data_loading.get_dataloaders(data_splits, config.BATCH_SIZE, config.NUM_WORKERS, image_transform)

    #     test_god_model = "resnet50"
    #     god_model, _ = load_embedding_model(test_god_model)

    #     if dataloaders['train']:
    #         X_train, Z_train = extract_features(god_model, dataloaders['train'], test_god_model, config.DEVICE)
    #         print(f"Extracted GOD Train {test_god_model} features: X={X_train.shape}, Z={Z_train.shape}")
    #     if dataloaders['test_avg']:
    #         X_test_avg, Z_test_avg = extract_features(god_model, dataloaders['test_avg'], test_god_model, config.DEVICE)
    #         print(f"Extracted GOD Test (Avg) {test_god_model} features: X={X_test_avg.shape}, Z={Z_test_avg.shape}")

    # except Exception as e:
    #     print(f"\nError during GOD feature extraction test (might be normal if GOD data not loaded): {e}")


    # --- Test ImageNet-256 Feature Precomputation ---
    print("\n--- Testing ImageNet-256 Feature Precomputation ---")
    try:
        # Test for one model to save time during direct execution
        test_imagenet_model = "resnet50" # Or "vit", "clip"
        print(f"\n--- Processing ImageNet-256 for: {test_imagenet_model} ---")
        features, labels, class_map = precompute_imagenet256_features(test_imagenet_model)

        # Loop through all models (comment out the single model test above if using this)
        # for model_name in config.EMBEDDING_MODELS.keys():
        #      print(f"\n--- Processing ImageNet-256 for: {model_name} ---")
        #      features, labels, class_map = precompute_imagenet256_features(model_name)

        if features is not None:
            print(f"Successfully precomputed/loaded features for {test_imagenet_model}: {features.shape}")
            print(f"Labels shape: {labels.shape if labels is not None else 'N/A'}")
            print(f"Example class map entry - Label 0: {class_map.get(0, 'N/A') if class_map else 'N/A'}")
        else:
            print(f"Feature computation failed for {test_imagenet_model}")
    except Exception as e:
        print(f"\nError during ImageNet-256 feature precomputation test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Feature Extraction Test Complete ---")
