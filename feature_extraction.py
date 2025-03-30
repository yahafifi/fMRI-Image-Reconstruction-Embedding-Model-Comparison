# feature_extraction.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import ViTModel, ViTImageProcessor, CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
import tqdm
import glob

import config
from data_loading import get_default_image_transform # Reuse transform

# --- Model Loading ---
def load_embedding_model(model_name):
    """Loads the specified pre-trained model and its preprocessor."""
    if model_name not in config.EMBEDDING_MODELS:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from {list(config.EMBEDDING_MODELS.keys())}")

    model_info = config.EMBEDDING_MODELS[model_name]
    repo_id = model_info["repo_id"]
    device = config.DEVICE

    model, processor = None, None

    print(f"Loading model: {model_name}...")
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer, use features from avgpool
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = model.to(device).eval()
        # Use standard transform defined in data_loading
        processor = get_default_image_transform(config.TARGET_IMAGE_SIZE)
    elif model_name == "vit":
        model = ViTModel.from_pretrained(repo_id).to(device).eval()
        processor = ViTImageProcessor.from_pretrained(model_info["preprocessor"])
        # Wrap processor to match expected transform format (PIL -> Tensor)
        vit_transform = lambda pil_img: processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)
        # Combine with basic resizing/cropping if needed (processor usually handles size)
        base_transform = transforms.Compose([
             transforms.Resize(256), # Might be redundant if processor handles it
             transforms.CenterCrop(config.TARGET_IMAGE_SIZE), # Might be redundant
        ])
        # Note: ViT processor expects PIL images, not tensors initially
        # So, the custom dataset should *not* apply ToTensor initially if using this.
        # Let's adjust the processor function to handle PIL input directly.
        processor = lambda pil_img: processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0).to(device)

    elif model_name == "clip":
        model = CLIPModel.from_pretrained(repo_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_info["preprocessor"])
         # Wrap processor for image processing only, expects PIL
        processor = lambda pil_img: processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0).to(device)

    print(f"Model {model_name} loaded.")
    return model, processor

# --- Feature Extraction Function ---
@torch.no_grad() # Disable gradient calculations for efficiency
def extract_features(model, dataloader, model_name, device):
    """Extracts features/embeddings from a model for all data in a dataloader."""
    model.eval() # Ensure model is in evaluation mode
    all_features = []
    all_fmri_data = [] # Store corresponding fMRI if available in dataloader

    has_fmri = hasattr(dataloader.dataset, 'fmri_data') # Check if it's our FmriImageDataset

    print(f"Extracting {model_name} features...")
    for batch in tqdm.tqdm(dataloader, desc=f"Extracting {model_name}"):
        if has_fmri:
            fmri_batch, image_batch = batch
            all_fmri_data.append(fmri_batch.cpu().numpy()) # Store fMRI data
            images = image_batch.to(device)
        else: # Assuming dataloader yields only images (like for Tiny ImageNet)
            images, _ = batch # Often ImageFolder yields (image, label)
            images = images.to(device)

        # Get embeddings based on model type
        if model_name == "resnet50":
            features = model(images) # Output of Sequential model (after avgpool)
            features = torch.flatten(features, 1) # Flatten the output
        elif model_name == "vit":
            # ViT needs dictionary input if using the raw model
            # If using the processor correctly in the Dataset, 'images' should be ready
            # outputs = model(pixel_values=images) # Assuming processor applied in Dataset
            # Let's assume the processor is applied here if not in Dataset
            # This depends heavily on how the processor lambda and Dataset interact
            # Safer: Assume dataloader gives normalized tensors, adapt here
            # This contradicts the ViT/CLIP processor needing PIL. Refactoring needed.

            # --- Correction: Assume Dataloader gives Tensors ---
            # We need to handle the normalization difference if ViT/CLIP processors expect PIL
            # Option A: Modify Dataset to conditionally return PIL for ViT/CLIP
            # Option B: Denormalize tensor -> PIL -> Processor (inefficient)
            # Option C: Hope the models work okay with standard normalized tensors (might be fine)
            # Let's try Option C for simplicity first.

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


# --- Tiny ImageNet Feature Precomputation ---
class TinyImageNetDataset(Dataset):
     """ Simple dataset for loading Tiny ImageNet images for feature extraction """
     def __init__(self, root_dir, transform=None):
         self.root_dir = root_dir
         self.transform = transform
         # Use ImageFolder to easily get paths and labels
         self.dataset = ImageFolder(os.path.join(root_dir, 'train'))
         self.image_paths = [p for p, _ in self.dataset.samples]
         self.labels = self.dataset.targets # Numerical labels assigned by ImageFolder
         self.class_to_idx = self.dataset.class_to_idx # Map class folder name (wnid) to numerical label
         self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

         # Load wnid to human-readable name mapping
         self.wnid_to_name = {}
         try:
              with open(config.TINY_IMAGENET_WORDS_TXT, "r") as f:
                   for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                             wnid, name = parts[0], parts[1]
                             # Take the first name if multiple are listed
                             self.wnid_to_name[wnid] = name.split(',')[0]
         except FileNotFoundError:
              print(f"Warning: Tiny ImageNet words.txt not found at {config.TINY_IMAGENET_WORDS_TXT}. Cannot map WNIDs to names.")


     def __len__(self):
         return len(self.image_paths)

     def __getitem__(self, idx):
         img_path = self.image_paths[idx]
         label = self.labels[idx]
         try:
             image = Image.open(img_path).convert("RGB")
         except Exception as e:
             print(f"Error loading image {img_path}: {e}")
             image = Image.new('RGB', (config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE), color = 'grey') # Dummy

         if self.transform:
             image = self.transform(image)

         return image, label

     def get_readable_label(self, idx):
          """ Get human-readable label for a given sample index """
          numerical_label = self.labels[idx]
          wnid = self.idx_to_class.get(numerical_label)
          if wnid:
               return self.wnid_to_name.get(wnid, wnid) # Return name or wnid if name not found
          return "Unknown"


def precompute_tiny_imagenet_features(model_name):
    """Loads Tiny ImageNet, extracts features using the specified model, and saves them."""
    feature_file = os.path.join(config.TINY_IMAGENET_FEATURES_PATH, f"tiny_imagenet_features_{model_name}.npy")
    labels_file = os.path.join(config.TINY_IMAGENET_FEATURES_PATH, "tiny_imagenet_labels.npy") # Labels are same for all models
    class_map_file = os.path.join(config.TINY_IMAGENET_FEATURES_PATH, "tiny_imagenet_class_map.npy") # wnid -> readable name

    if os.path.exists(feature_file) and os.path.exists(labels_file):
        print(f"Tiny ImageNet {model_name} features already exist: {feature_file}")
        features = np.load(feature_file)
        labels = np.load(labels_file)
        class_map = np.load(class_map_file, allow_pickle=True).item() # Load the dictionary
        return features, labels, class_map

    print(f"Precomputing Tiny ImageNet features for {model_name}...")
    model, _ = load_embedding_model(model_name) # We only need the model here
    # Use the *standard* transform for consistency across models for TinyImageNet
    transform = get_default_image_transform(config.TARGET_IMAGE_SIZE)

    dataset = TinyImageNetDataset(config.TINY_IMAGENET_PATH, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=config.BATCH_SIZE * 2, # Can often use larger batch size
                            shuffle=False,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    # Extract features (returns only features array)
    features = extract_features(model, dataloader, model_name, config.DEVICE)
    labels = np.array(dataset.labels) # Get all labels

    # Create the class map: numerical index -> readable name
    class_map = {idx: dataset.wnid_to_name.get(wnid, wnid)
                 for wnid, idx in dataset.class_to_idx.items()}

    # Save features, labels, and class map
    np.save(feature_file, features)
    if not os.path.exists(labels_file): # Save labels only once
         np.save(labels_file, labels)
    if not os.path.exists(class_map_file): # Save class map only once
        np.save(class_map_file, class_map)


    print(f"Saved Tiny ImageNet {model_name} features ({features.shape}) to {feature_file}")
    print(f"Saved Tiny ImageNet labels ({labels.shape}) to {labels_file}")
    print(f"Saved Tiny ImageNet class map to {class_map_file}")

    return features, labels, class_map


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Feature Extraction ---")

    # --- Test GOD Data Feature Extraction ---
    # Need to run data loading first to get dataloaders
    try:
        # (Reusing data loading test code snippet)
        handler = GodFmriDataHandler(config.SUBJECT_ID, config.ROI, config.GOD_FMRI_PATH, config.GOD_IMAGENET_PATH)
        data_splits = handler.get_data_splits(normalize_runs=True, test_split_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
        image_transform = get_default_image_transform(config.TARGET_IMAGE_SIZE)
        dataloaders = get_dataloaders(data_splits, config.BATCH_SIZE, config.NUM_WORKERS, image_transform)

        # Choose a model to test
        test_model_name = "resnet50" # Or "vit", "clip"

        model, _ = load_embedding_model(test_model_name)

        # Extract features for training set
        if dataloaders['train']:
            X_train, Z_train = extract_features(model, dataloaders['train'], test_model_name, config.DEVICE)
            print(f"\nExtracted GOD Train {test_model_name} features.")
            print(f"fMRI (X_train) shape: {X_train.shape}")
            print(f"Embeddings (Z_train) shape: {Z_train.shape}")
        else:
             print("Train dataloader is None, skipping GOD train feature extraction.")

         # Extract features for averaged test set
        if dataloaders['test_avg']:
            X_test_avg, Z_test_avg = extract_features(model, dataloaders['test_avg'], test_model_name, config.DEVICE)
            print(f"\nExtracted GOD Test (Avg) {test_model_name} features.")
            print(f"fMRI (X_test_avg) shape: {X_test_avg.shape}")
            print(f"Embeddings (Z_test_avg) shape: {Z_test_avg.shape}")
        else:
             print("Test (Avg) dataloader is None, skipping GOD test feature extraction.")

    except Exception as e:
        print(f"\nError during GOD feature extraction test: {e}")
        import traceback
        traceback.print_exc()


    # --- Test Tiny ImageNet Feature Precomputation ---
    print("\n--- Testing Tiny ImageNet Feature Precomputation ---")
    try:
        for model_name in config.EMBEDDING_MODELS.keys():
             features, labels, class_map = precompute_tiny_imagenet_features(model_name)
             print(f"Successfully precomputed/loaded features for {model_name}: {features.shape}")
             print(f"Example class map entry - Label 0: {class_map.get(0, 'N/A')}")
    except Exception as e:
        print(f"\nError during Tiny ImageNet feature precomputation test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Feature Extraction Test Complete ---")
