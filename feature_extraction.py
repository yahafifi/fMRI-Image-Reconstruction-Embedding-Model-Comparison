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
# Ensure data_loading is importable if needed for transforms, but better to define transform here
# from data_loading import get_default_image_transform

# --- Define Default Transform Here ---
# Avoids circular dependency, makes feature_extraction more self-contained
def get_default_image_transform(target_size=224):
     """ Standard image transforms for ImageNet-trained models """
     # Check if target_size is valid
     if not isinstance(target_size, int) or target_size <= 0:
          target_size = 224 # Default fallback
          print(f"Warning: Invalid target_size. Using default {target_size}.")

     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
     return transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(target_size),
         transforms.ToTensor(),
         normalize
     ])

# --- Model Loading ---
def load_embedding_model(model_name):
    """Loads the specified pre-trained model and its preprocessor."""
    if model_name not in config.EMBEDDING_MODELS:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from {list(config.EMBEDDING_MODELS.keys())}")

    model_info = config.EMBEDDING_MODELS[model_name]
    repo_id = model_info["repo_id"]
    device = config.DEVICE

    model = None
    processor = None # Store processor/transform needed specifically by the model if any

    print(f"Loading model: {model_name}...")
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*(list(model.children())[:-1])) # Use features before final FC
        model = model.to(device).eval()
        # For ResNet, standard transform is usually applied in Dataset
        processor = get_default_image_transform(config.TARGET_IMAGE_SIZE)
    elif model_name == "vit":
        model = ViTModel.from_pretrained(repo_id).to(device).eval()
        # ViT models require specific preprocessing
        processor = ViTImageProcessor.from_pretrained(model_info["preprocessor"])
    elif model_name == "clip":
        model = CLIPModel.from_pretrained(repo_id).to(device).eval()
        # CLIP models require specific preprocessing
        processor = CLIPProcessor.from_pretrained(model_info["preprocessor"])

    if model is None:
         raise ValueError(f"Model loading failed for {model_name}")

    print(f"Model {model_name} loaded successfully.")
    # Return both model and its specific processor (if any)
    # The dataset should return PIL images if model needs HF processor
    return model, processor


# --- Feature Extraction Function ---
@torch.no_grad()
def extract_features(model, dataloader, model_name, device, processor=None):
    """
    Extracts features/embeddings from a model for all data in a dataloader.
    Handles different model input requirements (tensors vs PIL images).
    Assumes dataloader yields (fmri_batch, image_batch).
    image_batch can be Tensors (if pre-transformed in Dataset) or PIL images.
    """
    model.eval()
    all_features = []
    all_fmri_data = [] # Store corresponding fMRI if available in dataloader

    # Determine input type needed by the model
    needs_pil_input = model_name in ["vit", "clip"] and processor is not None

    print(f"Extracting {model_name} features...")
    for batch in tqdm.tqdm(dataloader, desc=f"Extracting {model_name}"):
        # Dataloader should yield fmri, image data
        fmri_batch, data_batch = batch

        # Store fMRI data if it's not empty/dummy
        if fmri_batch.numel() > 0:
             all_fmri_data.append(fmri_batch.cpu().numpy())

        # --- Prepare image data based on model requirements ---
        if needs_pil_input:
            # We assume data_batch contains PIL Images here
            # Processor expects list of PIL images, returns batch of tensors
            try:
                # Handle potential errors if data_batch isn't PIL
                if not isinstance(data_batch, list) and not isinstance(data_batch[0], Image.Image):
                     # This might happen if Dataset applies transform incorrectly
                     print("Warning: Expected PIL images for ViT/CLIP processor, but received tensors. Attempting tensor processing.")
                     # If data_batch is already a tensor batch from Compose transform:
                     if isinstance(data_batch, torch.Tensor):
                          images_input = data_batch.to(device)
                     else:
                          raise TypeError("Unsupported batch type for ViT/CLIP")
                else:
                    # Normal case: process list of PIL images
                    inputs = processor(images=data_batch, return_tensors="pt", padding=True) # Added padding just in case
                    images_input = inputs['pixel_values'].to(device)

            except Exception as e:
                 print(f"Error processing batch with {model_name} processor: {e}. Skipping batch.")
                 continue # Skip this batch if processing fails
        else:
             # Assume data_batch contains image tensors (e.g., for ResNet)
             # Move tensor to the correct device
             if isinstance(data_batch, torch.Tensor):
                  images_input = data_batch.to(device)
             else:
                  print(f"Warning: Expected image tensors for {model_name}, but received {type(data_batch)}. Skipping batch.")
                  continue


        # --- Get Embeddings ---
        try:
            if model_name == "resnet50":
                features = model(images_input) # Output of Sequential model (after avgpool)
                features = torch.flatten(features, 1) # Flatten the output
            elif model_name == "vit":
                outputs = model(pixel_values=images_input)
                features = outputs.pooler_output # Use the pooled output [batch_size, embedding_dim]
            elif model_name == "clip":
                outputs = model.get_image_features(pixel_values=images_input)
                features = outputs # Output is already [batch_size, embedding_dim]
            else:
                 # Should not happen due to initial check, but good practice
                 raise ValueError(f"Embedding extraction logic not defined for {model_name}")

            all_features.append(features.cpu().numpy())

        except Exception as e:
            print(f"Error during model forward pass or feature extraction for batch: {e}")
            # Decide whether to skip batch or append NaNs of correct shape?
            # Appending NaNs might be better to keep alignment if possible
            try:
                batch_size = images_input.shape[0]
                emb_dim = config.EMBEDDING_MODELS[model_name]['embedding_dim']
                nan_features = np.full((batch_size, emb_dim), np.nan, dtype=np.float32)
                all_features.append(nan_features)
                print(f"Appended NaNs for failed batch.")
            except: # Fallback if even getting shape fails
                 print("Could not append NaNs for failed batch.")


    # --- Concatenate results ---
    if not all_features:
         print("Error: No features were extracted.")
         return None, None # Or return empty arrays?

    try:
        all_features = np.concatenate(all_features, axis=0)
    except ValueError:
         print("Error concatenating features, likely due to inconsistent shapes from errors.")
         # Attempt to filter out potential NaN arrays if they cause issues
         valid_features = [f for f in all_features if isinstance(f, np.ndarray) and f.ndim == 2]
         if valid_features:
              print("Attempting concatenation with valid features only...")
              all_features = np.concatenate(valid_features, axis=0)
         else:
              print("No valid features found after error handling.")
              return None, None


    # Concatenate fMRI data if collected
    if all_fmri_data:
         try:
            all_fmri_data = np.concatenate(all_fmri_data, axis=0)
            print(f"Finished extraction. Features shape: {all_features.shape}, fMRI shape: {all_fmri_data.shape}")
            # Ensure shapes match if fMRI was present
            if all_features.shape[0] != all_fmri_data.shape[0]:
                 print(f"Warning: Final feature count ({all_features.shape[0]}) != fMRI count ({all_fmri_data.shape[0]}) due to errors.")
                 # How to handle? Return None? Try to align? For now return as is.
            return all_fmri_data, all_features
         except ValueError:
            print("Error concatenating fMRI data.")
            # Decide return strategy, maybe return features only?
            print(f"Finished extraction. Features shape: {all_features.shape}, fMRI shape: Inconsistent/Error")
            return None, all_features # Indicate fMRI issue
    else:
        # This path is taken if dataloader didn't yield valid fMRI or for ImageNet256
        print(f"Finished extraction. Features shape: {all_features.shape} (No corresponding fMRI data returned)")
        return None, all_features # Return None for fMRI, just the image features


# --- ImageNet-256 Dataset ---
# This dataset should yield PIL images if extraction model needs them (ViT, CLIP)
class ImageNet256Dataset(Dataset):
    """ Loads images from the ImageNet-256 dataset structure """
    def __init__(self, root_dir, transform=None, return_pil=False):
        self.root_dir = root_dir
        self.transform = transform # Standard Torchvision transform (applied if return_pil=False)
        self.return_pil = return_pil # Flag to return PIL image instead of tensor
        self.image_paths = []
        self.labels = [] # Numerical labels
        self.class_names = [] # Human-readable names (folder names)
        self.class_to_idx = {}
        self.idx_to_class = {}

        print(f"Loading ImageNet256 file list from: {root_dir}")
        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"ImageNet-256 directory not found at {root_dir}")

        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.class_names:
             raise FileNotFoundError(f"No class subdirectories found in {root_dir}")
        print(f"Found {len(self.class_names)} classes.")

        for idx, class_name in enumerate(tqdm.tqdm(self.class_names, desc="Scanning classes")):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            class_dir = os.path.join(root_dir, class_name)
            # Find common image types
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
            for ext in extensions:
                 image_files = glob.glob(os.path.join(class_dir, ext))
                 for img_path in image_files:
                     self.image_paths.append(img_path)
                     self.labels.append(idx)

        print(f"Found {len(self.image_paths)} images in total.")
        if len(self.image_paths) == 0:
             print(f"Warning: No images found. Check dataset path '{root_dir}' and file extensions.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB") # Load as PIL
            valid_image = True
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning dummy PIL image.")
            try:
                size = (config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE)
                image = Image.new('RGB', size, color = 'grey')
            except NameError: # Fallback
                image = Image.new('RGB', (224, 224), color = 'grey')
            valid_image = False


        # Decide what to return based on return_pil flag
        if self.return_pil:
            # Return the PIL image (potentially dummy) and label
            # The HuggingFace processor will handle transformations in the collate/batch step
            # Note: If returning PIL, the 'transform' arg of this dataset is NOT used.
             dummy_fmri = torch.tensor([]) # Return dummy fmri for consistency
             return dummy_fmri, image # Return PIL image
        else:
            # Apply the standard torchvision transform (if provided)
            if self.transform and valid_image:
                try:
                    image_tensor = self.transform(image) # Should output a Tensor
                except Exception as e:
                     print(f"Error applying transform to {img_path}: {e}. Returning dummy tensor.")
                     image_tensor = torch.zeros((3, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
            elif self.transform and not valid_image:
                 # Apply transform to dummy PIL if needed (some transforms might work)
                 try: image_tensor = self.transform(image)
                 except: image_tensor = torch.zeros((3, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
            else:
                 # No transform or invalid image - return dummy tensor
                 image_tensor = torch.zeros((3, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))

            dummy_fmri = torch.tensor([]) # Return dummy fmri for consistency
            return dummy_fmri, image_tensor # Return Tensor

    def get_readable_label(self, idx):
         numeric_label = self.labels[idx]
         return self.idx_to_class.get(numeric_label, f"UnknownLabel_{numeric_label}")


# --- ImageNet-256 Feature Precomputation ---
def precompute_imagenet256_features(model_name):
    """Loads ImageNet-256, extracts features using the specified model, and saves them."""
    feature_file = os.path.join(config.IMAGENET256_FEATURES_PATH, f"imagenet256_features_{model_name}.npy")
    labels_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_labels.npy")
    class_map_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_idx_to_class.npy")

    if os.path.exists(feature_file):
        print(f"ImageNet-256 {model_name} features already exist: {feature_file}")
        try:
            features = np.load(feature_file)
            if os.path.exists(labels_file) and os.path.exists(class_map_file):
                 labels = np.load(labels_file)
                 class_map = np.load(class_map_file, allow_pickle=True).item()
                 return features, labels, class_map
            else:
                 print("Labels or class map file missing, will regenerate...")
                 # Need to reload dataset to get labels/map
                 dataset_for_meta = ImageNet256Dataset(config.IMAGENET256_PATH) # No transform needed
                 labels = np.array(dataset_for_meta.labels)
                 class_map = dataset_for_meta.idx_to_class
                 np.save(labels_file, labels)
                 np.save(class_map_file, class_map)
                 print("Regenerated and saved labels/class map.")
                 return features, labels, class_map
        except Exception as e:
             print(f"Error loading existing features/labels: {e}. Recomputing...")


    print(f"Precomputing ImageNet-256 features for {model_name}...")
    start_time = time.time()

    try:
        model, processor = load_embedding_model(model_name)
        if model is None: raise ValueError("Failed to load model.")

        # Decide if dataset should return PIL based on model type
        return_pil_for_dataset = model_name in ["vit", "clip"]

        # Define transform only if not returning PIL
        transform_for_dataset = None
        if not return_pil_for_dataset:
             transform_for_dataset = get_default_image_transform(config.TARGET_IMAGE_SIZE)

        # Create dataset
        dataset = ImageNet256Dataset(
             config.IMAGENET256_PATH,
             transform=transform_for_dataset, # Pass transform only if needed
             return_pil=return_pil_for_dataset
        )
        if len(dataset) == 0:
             print("Dataset is empty. Cannot extract features.")
             return None, None, None

        # *** USE CORRECT BATCH SIZE CONFIG ***
        dataloader = DataLoader(dataset,
                                batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, # Use specific batch size
                                shuffle=False,
                                num_workers=config.FEATURE_EXTRACTION_NUM_WORKERS,
                                pin_memory=True)

        # Extract features (returns fmri_data, features - fmri_data will be None here)
        _, features = extract_features(model, dataloader, model_name, config.DEVICE, processor)

        if features is None:
             print("Feature extraction returned None. Aborting.")
             return None, None, None

        # Get labels and class map from the dataset instance
        labels = np.array(dataset.labels)
        class_map = dataset.idx_to_class

        # Save features, labels, and class map
        os.makedirs(config.IMAGENET256_FEATURES_PATH, exist_ok=True)
        np.save(feature_file, features)
        if not os.path.exists(labels_file): np.save(labels_file, labels)
        if not os.path.exists(class_map_file): np.save(class_map_file, class_map)

        end_time = time.time()
        print(f"Saved ImageNet-256 {model_name} features ({features.shape}) to {feature_file}")
        print(f"Saved ImageNet-256 labels ({labels.shape}) to {labels_file}")
        print(f"Saved ImageNet-256 class map ({len(class_map)} entries) to {class_map_file}")
        print(f"Feature extraction took {(end_time - start_time)/60:.2f} minutes.")

    except FileNotFoundError as e:
         print(f"Error initializing ImageNet256Dataset: {e}")
         print(f"Please ensure config.IMAGENET256_PATH ('{config.IMAGENET256_PATH}') points to the correct directory containing class folders.")
         return None, None, None
    except Exception as e:
        print(f"Error during ImageNet-256 feature precomputation for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    return features, labels, class_map


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Feature Extraction ---")
    # Test ImageNet-256 Precomputation for a model
    test_model = "clip" # Try CLIP which needs PIL images
    print(f"\n--- Testing ImageNet-256 Precomputation for: {test_model} ---")
    try:
        # Ensure the ImageNet path exists for the test
        if not os.path.exists(config.IMAGENET256_PATH):
             print(f"SKIPPING test: ImageNet path not found at {config.IMAGENET256_PATH}")
        else:
             features, labels, class_map = precompute_imagenet256_features(test_model)
             if features is not None:
                 print(f"Successfully precomputed/loaded features for {test_model}: {features.shape}")
                 print(f"Labels shape: {labels.shape if labels is not None else 'N/A'}")
                 print(f"Example class map entry - Label 0: {class_map.get(0, 'N/A') if class_map else 'N/A'}")
             else:
                 print(f"Feature computation failed for {test_model}")
    except Exception as e:
        print(f"\nError during ImageNet-256 feature precomputation test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Feature Extraction Test Complete ---")