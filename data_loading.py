# data_loading.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import bdpy # Brain Decoding Toolbox (assuming it's installed: pip install bdpyd)
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

import config

# --- BDPy Data Handler (Simplified & adapted from original) ---
class GodFmriDataHandler:
    def __init__(self, subject_id, roi, data_dir, image_dir, test_img_csv_name="image_test_id.csv", train_img_csv_name="image_training_id.csv"):
        self.subject_id = subject_id
        self.roi = roi
        self.h5_file = os.path.join(data_dir, f"Subject{subject_id}.h5")
        self.image_dir = image_dir # Path to the GOD stimuli images (train/test folders)
        self.test_img_csv = os.path.join(image_dir, test_img_csv_name)
        self.train_img_csv = os.path.join(image_dir, train_img_csv_name)

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"fMRI data file not found: {self.h5_file}")
        if not os.path.exists(self.test_img_csv) or not os.path.exists(self.train_img_csv):
             raise FileNotFoundError(f"Image ID CSV files not found in {image_dir}")

        print(f"Loading fMRI data for Subject {subject_id}, ROI: {roi}")
        self.dat = bdpy.BData(self.h5_file)

        # Load image ID mappings
        self.test_img_df = pd.read_csv(self.test_img_csv, header=None, names=['id', 'filename'])
        self.train_img_df = pd.read_csv(self.train_img_csv, header=None, names=['id', 'filename'])

        # Extract basic info immediately
        self._extract_metadata_and_roi_data()
        self._map_filenames()

    def _extract_metadata_and_roi_data(self):
        """Extracts ROI data and relevant metadata."""
        self.voxel_data = self.dat.select(self.roi) # Shape: (n_samples, n_voxels)
        self.metadata = {
            'DataType': self.dat.select('DataType').squeeze(), # 1: train, 2: test, 3: test_imagine
            'Run': self.dat.select('Run').astype('int').squeeze(),
            'stimulus_id': self.dat.select('stimulus_id').squeeze()
        }
        self.n_samples, self.n_voxels = self.voxel_data.shape
        print(f"Loaded {self.n_samples} samples, {self.n_voxels} voxels.")

    def _map_filenames(self):
        """Creates lists of image filenames corresponding to fMRI samples."""
        self.image_filenames = []
        self.sample_types = [] # Store type for each sample

        train_id_to_fname = self.train_img_df.set_index('id')['filename'].to_dict()
        test_id_to_fname = self.test_img_df.set_index('id')['filename'].to_dict()

        for i in range(self.n_samples):
            sample_type = self.metadata['DataType'][i]
            stim_id = self.metadata['stimulus_id'][i]

            fname = None
            folder = None
            if sample_type == 1: # Training
                fname = train_id_to_fname.get(stim_id)
                folder = "training"
            elif sample_type == 2 or sample_type == 3: # Test or Test Imagine
                fname = test_id_to_fname.get(stim_id)
                folder = "test" # Both use test set images

            if fname and folder:
                full_path = os.path.join(self.image_dir, folder, fname)
                if os.path.exists(full_path):
                     self.image_filenames.append(full_path)
                     self.sample_types.append(sample_type)
                else:
                    # Handle missing files gracefully - append None or raise error?
                    print(f"Warning: Image file not found: {full_path}")
                    self.image_filenames.append(None) # Placeholder
                    self.sample_types.append(sample_type) # Keep type consistent
            else:
                print(f"Warning: Could not map stimulus ID {stim_id} (Type {sample_type}) to filename.")
                self.image_filenames.append(None)
                self.sample_types.append(sample_type)

        # Filter out samples where filename mapping failed
        valid_indices = [i for i, fname in enumerate(self.image_filenames) if fname is not None]
        if len(valid_indices) < self.n_samples:
            print(f"Warning: Dropping {self.n_samples - len(valid_indices)} samples due to missing image files.")
            self.voxel_data = self.voxel_data[valid_indices]
            self.image_filenames = [self.image_filenames[i] for i in valid_indices]
            self.sample_types = [self.sample_types[i] for i in valid_indices]
            # Also filter metadata if needed, though maybe not necessary for train/test split logic
            self.metadata['DataType'] = self.metadata['DataType'][valid_indices]
            self.metadata['Run'] = self.metadata['Run'][valid_indices]
            self.metadata['stimulus_id'] = self.metadata['stimulus_id'][valid_indices]
            self.n_samples = len(valid_indices)


    def get_data_splits(self, normalize_runs=True, test_split_size=0.1, random_state=42):
        """
        Splits data into training, validation, and testing sets based on DataType.
        Applies optional run-wise normalization.
        Returns fMRI data (numpy arrays) and corresponding image paths (lists).
        """
        data = self.voxel_data.copy()

        # --- Normalization (Run-wise) ---
        if normalize_runs:
            print("Applying run-wise normalization (scaling)...")
            runs = self.metadata['Run']
            unique_runs = np.unique(runs)
            for r in unique_runs:
                run_indices = (runs == r)
                if np.any(run_indices): # Ensure the run exists in the (potentially filtered) data
                    scaler = sklearn.preprocessing.StandardScaler()
                    data[run_indices] = scaler.fit_transform(data[run_indices])
            print("Normalization complete.")

        # --- Identify Train/Val/Test Samples based on DataType ---
        # DataType 1 = Training/Validation Pool
        # DataType 2 = Testing Pool
        # DataType 3 = Test Imagine (Typically excluded from basic reconstruction training/testing)

        train_val_indices = [i for i, t in enumerate(self.sample_types) if t == 1]
        test_indices = [i for i, t in enumerate(self.sample_types) if t == 2]
        # imagine_indices = [i for i, t in enumerate(self.sample_types) if t == 3] # Keep track if needed later

        if not train_val_indices:
             raise ValueError("No training samples (DataType 1) found.")
        if not test_indices:
             raise ValueError("No testing samples (DataType 2) found.")


        # --- Split Training Pool into Train and Validation ---
        fmri_train_val = data[train_val_indices]
        images_train_val = [self.image_filenames[i] for i in train_val_indices]

        if test_split_size > 0:
             fmri_train, fmri_val, images_train, images_val = train_test_split(
                 fmri_train_val, images_train_val,
                 test_size=test_split_size,
                 random_state=random_state
             )
        else: # No validation set
             fmri_train = fmri_train_val
             images_train = images_train_val
             fmri_val, images_val = np.array([]), [] # Empty val set


        # --- Prepare Test Set ---
        # Important: The original code averaged test data per stimulus ID.
        # We will return the *unaveraged* test data first, as that's often needed
        # for direct prediction before potential averaging later.
        fmri_test = data[test_indices]
        images_test = [self.image_filenames[i] for i in test_indices]
        test_stimulus_ids = self.metadata['stimulus_id'][test_indices]

        # --- Create Averaged Test Set (Optional but common) ---
        # Map stimulus IDs back to the unique test filenames for consistent ordering
        unique_test_stim_ids = sorted(list(set(test_stimulus_ids))) # Get unique IDs
        fmri_test_avg = []
        images_test_avg = [] # Will contain one path per unique test stimulus

        test_id_to_fname = self.test_img_df.set_index('id')['filename'].to_dict()

        print(f"Averaging test data across {len(unique_test_stim_ids)} unique stimuli...")
        for stim_id in unique_test_stim_ids:
             id_indices = [i for i, s_id in enumerate(test_stimulus_ids) if s_id == stim_id]
             if id_indices:
                  avg_fmri = np.mean(fmri_test[id_indices], axis=0)
                  fmri_test_avg.append(avg_fmri)
                  # Get the corresponding filename (should be the same for all instances of stim_id)
                  fname = test_id_to_fname.get(stim_id)
                  if fname:
                       img_path = os.path.join(self.image_dir, "test", fname)
                       if os.path.exists(img_path):
                           images_test_avg.append(img_path)
                       else:
                           print(f"Warning: Averaged test image path missing: {img_path}")
                           images_test_avg.append(None) # Placeholder
                  else:
                       print(f"Warning: Could not find filename for averaged test stim ID {stim_id}")
                       images_test_avg.append(None)

        fmri_test_avg = np.array(fmri_test_avg)
        # Filter out None paths if any occurred
        valid_avg_indices = [i for i, p in enumerate(images_test_avg) if p is not None]
        fmri_test_avg = fmri_test_avg[valid_avg_indices]
        images_test_avg = [images_test_avg[i] for i in valid_avg_indices]


        print(f"Data split sizes: Train={len(fmri_train)}, Val={len(fmri_val)}, Test (Avg)={len(fmri_test_avg)}")

        return {
            "train": (fmri_train, images_train),
            "val": (fmri_val, images_val),
            "test_avg": (fmri_test_avg, images_test_avg),
             # Optionally return the non-averaged test set too
             # "test": (fmri_test, images_test),
        }


# --- PyTorch Custom Dataset ---
class FmriImageDataset(Dataset):
    def __init__(self, fmri_data, image_paths, transform=None):
        """
        Args:
            fmri_data (numpy.ndarray): fMRI data samples.
            image_paths (list): List of paths to the corresponding images.
            transform (callable, optional): Optional transform to be applied on a sample's image.
        """
        if len(fmri_data) != len(image_paths):
            raise ValueError(f"Mismatch between number of fMRI samples ({len(fmri_data)}) and image paths ({len(image_paths)})")

        # Convert fMRI data to float32 tensor, suitable for PyTorch models
        self.fmri_data = torch.tensor(fmri_data, dtype=torch.float32)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fmri_sample = self.fmri_data[idx]
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image or raise error? For now, dummy tensor.
            # Be careful with this in production.
            image = torch.zeros((3, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE)) # Adjust size if needed
            # If using transforms, apply a dummy version or skip
            # return fmri_sample, image # Skip transform if dummy

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 print(f"Error applying transform to image {image_path}: {e}")
                 # Handle transform error, maybe return dummy
                 image = torch.zeros((3, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))


        return fmri_sample, image


# --- Function to get DataLoaders ---
def get_dataloaders(god_data_splits, batch_size, num_workers, image_transform):
    """Creates PyTorch DataLoaders for train, validation, and test sets."""

    dataloaders = {}

    # Training DataLoader
    if len(god_data_splits['train'][0]) > 0:
        train_dataset = FmriImageDataset(
            fmri_data=god_data_splits['train'][0],
            image_paths=god_data_splits['train'][1],
            transform=image_transform
        )
        dataloaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        print(f"Train DataLoader created with {len(train_dataset)} samples.")
    else:
        dataloaders['train'] = None
        print("No training data, Train DataLoader not created.")


    # Validation DataLoader
    if len(god_data_splits['val'][0]) > 0:
        val_dataset = FmriImageDataset(
            fmri_data=god_data_splits['val'][0],
            image_paths=god_data_splits['val'][1],
            transform=image_transform
        )
        dataloaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        print(f"Validation DataLoader created with {len(val_dataset)} samples.")
    else:
        dataloaders['val'] = None
        print("No validation data, Validation DataLoader not created.")

    # Test (Averaged) DataLoader
    if len(god_data_splits['test_avg'][0]) > 0:
        test_dataset_avg = FmriImageDataset(
            fmri_data=god_data_splits['test_avg'][0],
            image_paths=god_data_splits['test_avg'][1],
            transform=image_transform
        )
        # No shuffle for test set
        dataloaders['test_avg'] = DataLoader(
            test_dataset_avg, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        print(f"Test (Averaged) DataLoader created with {len(test_dataset_avg)} samples.")
    else:
        dataloaders['test_avg'] = None
        print("No averaged test data, Test (Averaged) DataLoader not created.")


    return dataloaders


# --- Image Transformation ---
# Define standard transforms matching typical pre-trained model inputs
def get_default_image_transform(target_size=224):
     # Normalization values common for models trained on ImageNet
     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
     return transforms.Compose([
         transforms.Resize(256),            # Resize slightly larger than crop
         transforms.CenterCrop(target_size), # Crop to target size
         transforms.ToTensor(),             # Convert PIL Image to tensor [0, 1]
         normalize                          # Normalize tensor
     ])

# --- Example Usage (for testing this module) ---
if __name__ == "__main__":
    print("--- Testing Data Loading ---")
    # Ensure data is downloaded first by running download_data.py
    try:
        handler = GodFmriDataHandler(
            subject_id=config.SUBJECT_ID,
            roi=config.ROI,
            data_dir=config.GOD_FMRI_PATH,
            image_dir=config.GOD_IMAGENET_PATH
        )

        data_splits = handler.get_data_splits(
             normalize_runs=True,
             test_split_size=config.TEST_SPLIT_SIZE,
             random_state=config.RANDOM_STATE
        )

        image_transform = get_default_image_transform(config.TARGET_IMAGE_SIZE)

        dataloaders = get_dataloaders(
            god_data_splits=data_splits,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            image_transform=image_transform
        )

        # Test iterating through a batch
        if dataloaders['train']:
            print("\nTesting Train DataLoader batch:")
            fmri_batch, image_batch = next(iter(dataloaders['train']))
            print("fMRI batch shape:", fmri_batch.shape) # Should be [batch_size, n_voxels]
            print("Image batch shape:", image_batch.shape) # Should be [batch_size, 3, height, width]
        else:
             print("Train dataloader is None.")

        if dataloaders['test_avg']:
             print("\nTesting Test (Avg) DataLoader batch:")
             fmri_batch_test, image_batch_test = next(iter(dataloaders['test_avg']))
             print("fMRI batch shape (test_avg):", fmri_batch_test.shape)
             print("Image batch shape (test_avg):", image_batch_test.shape)

             # Store the test ground truth image paths for later evaluation
             test_avg_image_paths = data_splits['test_avg'][1]
             print(f"\nFirst 5 Test (Avg) Image Paths: {test_avg_image_paths[:5]}")

        print("\n--- Data Loading Test Complete ---")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure you have run 'download_data.py' successfully first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during data loading test: {e}")
        import traceback
        traceback.print_exc()
