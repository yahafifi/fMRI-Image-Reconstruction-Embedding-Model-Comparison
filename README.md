# fMRI-Image-Reconstruction-Embedding-Model-Comparison

## subjects
* https://figshare.com/ndownloader/files/24080120
* https://figshare.com/ndownloader/files/24080123
* https://figshare.com/ndownloader/files/24080126
* https://figshare.com/ndownloader/files/24080129
* https://figshare.com/ndownloader/files/24080132

# Comparative Evaluation of Visual Embedding Models for fMRI-Based Image Reconstruction (Ridge Regression Version)

## Overview

This project implements and evaluates a pipeline for reconstructing visual stimuli perceived by humans directly from their functional Magnetic Resonance Imaging (fMRI) data. It leverages pre-trained visual embedding models (ResNet50, ViT, CLIP) and generative AI (Stable Diffusion) to achieve this.

This specific version of the pipeline focuses on:

1.  **Mapping:** Using **Ridge Regression** (a linear model) to map fMRI voxel activity to the latent space of different visual embedding models.
2.  **Generation Conditioning:** Using **text prompts** derived from retrieved semantic class labels to condition the Stable Diffusion model for image generation.
3.  **Comparison:** Systematically comparing the reconstruction quality achieved using three different visual embedding models:
    *   ResNet50 (CNN-based)
    *   Vision Transformer (ViT) (Transformer-based)
    *   CLIP (Contrastive Language-Image Pretraining, using its vision encoder)

The primary objective is to understand how the choice of visual embedding influences the fidelity and semantic accuracy of the reconstructed images when using this specific linear mapping and text-based generation approach.

## Pipeline Workflow

The core pipeline implemented in this codebase follows these steps:

1.  **Data Loading (`data_loading.py`):**
    *   Loads fMRI data for a specific subject (e.g., Subject 3) from the Generic Object Decoding (GOD) dataset (`.h5` file).
    *   Loads the corresponding ground truth visual stimuli (images) presented during the fMRI scans.
    *   Handles metadata (run information, stimulus IDs, data types - train/test).
    *   Applies optional run-wise normalization to fMRI data.
    *   Splits data into training, validation, and averaged test sets.
    *   Creates PyTorch DataLoaders for efficient batch processing.

2.  **Feature Extraction (GOD Stimuli) (`feature_extraction.py`):**
    *   Loads the specified pre-trained visual embedding model (ResNet50, ViT, or CLIP).
    *   Extracts deep visual embeddings (features) for the training, validation, and test set ground truth images using the chosen model. These serve as the target variables for the mapping model.

3.  **Mapping Model (fMRI -> Embedding) (`mapping_models.py`):**
    *   **Trains or loads a Ridge Regression model.**
    *   The model learns a linear mapping from the fMRI voxel patterns (`X_train`) to the corresponding target visual embeddings (`Z_train`).
    *   *(Note: This version focuses solely on Ridge Regression. MLP mapping was explored as a later enhancement).*

4.  **Embedding Prediction (`mapping_models.py`):**
    *   Uses the trained Ridge model to predict visual embeddings (`Z_test_avg_pred`) directly from the test set fMRI data (`X_test_avg`).
    *   Applies a standardization step to the predicted embeddings based on training set statistics.

5.  **Feature Extraction (Retrieval DB) (`feature_extraction.py`):**
    *   **Precomputes or loads visual embeddings** for a large external image dataset (Tiny ImageNet) using the *same* visual embedding model (ResNet50, ViT, or CLIP). This creates the feature database for retrieval.
    *   Saves the features, corresponding labels (numeric class IDs), and a class map (ID to readable name).

6.  **Retrieval Model (k-NN) (`retrieval.py`):**
    *   **Trains or loads a k-Nearest Neighbors (k-NN)** model fitted on the Tiny ImageNet feature database.
    *   Uses 'cosine' distance as the metric, suitable for comparing high-dimensional embeddings.

7.  **Semantic Label Retrieval (`retrieval.py`):**
    *   Uses the k-NN model to find the nearest neighbors in the Tiny ImageNet feature space for each *predicted* visual embedding (from step 4).
    *   Retrieves the class labels of these nearest neighbors.
    *   Selects the **Top-1** retrieved class label for each test sample.

8.  **Image Generation (`generation.py`):**
    *   Loads a pre-trained Stable Diffusion pipeline (e.g., v1.5).
    *   Uses the **Top-1 retrieved class label** (from step 7) as a **text prompt** to generate an image.

9.  **Evaluation (`evaluation.py`):**
    *   Compares the generated images (from step 8) to the original ground truth test images (from step 1).
    *   Calculates quantitative metrics:
        *   Structural Similarity Index (SSIM)
        *   Learned Perceptual Image Patch Similarity (LPIPS)
        *   CLIP Score (Image-Image Similarity using CLIP)
    *   Saves the results to a CSV file.
    *   Optionally generates a visualization comparing ground truth and generated images.

10. **Orchestration (`run_experiment.py`):**
    *   Manages the overall workflow, calling functions from other modules in sequence.
    *   Handles command-line arguments for selecting the visual model and other options.

## Prerequisites

*   **Python:** 3.8+
*   **pip:** Package installer for Python.
*   **Git:** For cloning the repository.
*   **CUDA-enabled GPU:** Required for efficient deep learning model inference (feature extraction, Stable Diffusion) and potentially MLP training (if added later). Significant GPU RAM (e.g., 16GB+) is recommended for Stable Diffusion.
*   **Disk Space:** Sufficient space for datasets (GOD, Tiny ImageNet), downloaded models, extracted features, and generated images (potentially > 50-100 GB).

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(See `requirements.txt` for the full list of packages).*

## Data Setup (CRITICAL STEP)

This pipeline requires several datasets. Due to the size and potential licensing of the GOD dataset, **manual setup is highly recommended, especially when using Kaggle.**

1.  **Generic Object Decoding (GOD) Dataset:**
    *   **fMRI Data:** You need the `.h5` file for the subject (e.g., `Subject3.h5`).
    *   **Stimulus Images:** You need the corresponding image stimuli presented during the scans. These are typically organized into `training` and `test` folders, along with `image_training_id.csv` and `image_test_id.csv` mapping files.
    *   **Acquisition:** Obtain this data from the original source (e.g., figshare, authors' websites). Links provided in the original `download_data.py` script (especially Google Drive links) may be unreliable or expired.
    *   **Recommended Setup (Kaggle):**
        1.  Upload `Subject3.h5` as a **private Kaggle Dataset** (e.g., named `god-fmri-data`).
        2.  Upload the entire stimulus image directory structure (containing `training/`, `test/`, and the two `.csv` files) as another **private Kaggle Dataset** (e.g., named `god-stimuli-images`).
        3.  **Edit `config.py`:** Update the `GOD_FMRI_FILE_PATH` and `GOD_IMAGENET_PATH` variables in `config.py` to point to the correct paths within `/kaggle/input/` based on your dataset names (e.g., `/kaggle/input/god-fmri-data/Subject3.h5`, `/kaggle/input/god-stimuli-images`).

2.  **Tiny ImageNet Dataset:**
    *   **Purpose:** Used as the retrieval database for finding semantic labels.
    *   **Acquisition:** Download from the official source or use a standard Kaggle Dataset (e.g., search for "Tiny ImageNet").
    *   **Recommended Setup (Kaggle):**
        1.  Add the Tiny ImageNet dataset as an input source to your Kaggle Notebook/environment.
        2.  **Identify the correct path:** Use `!ls /kaggle/input/<tiny-imagenet-dataset-name>/` in a notebook cell to find the directory that *directly contains* the `train/`, `val/`, `words.txt`, `wnids.txt` files/folders. The structure can sometimes be nested (e.g., `/kaggle/input/tiny-imagenet/tiny-imagenet-200/tiny-imagenet-200`).
        3.  **Edit `config.py`:** Update the `TINY_IMAGENET_PATH` variable to this correct path.

3.  **Metadata Files (Optional Download):**
    *   The `download_data.py` script attempts to download `class_to_wordnet.json` from GitHub. If this fails, you may need to acquire it manually if needed by specific analysis steps (not strictly required for the core Ridge + Text Prompt pipeline).

**Note:** Using the `--download` flag with `run_experiment.py` will attempt downloads based on URLs in `download_data.py`, but this is **not reliable** for the large GOD dataset files. Manual setup is strongly advised.

## Configuration (`config.py`)

This file centralizes all important paths, model identifiers, and hyperparameters.

*   **CRITICAL:** Ensure `GOD_FMRI_FILE_PATH`, `GOD_IMAGENET_PATH`, and `TINY_IMAGENET_PATH` are correctly set according to your Data Setup.
*   Adjust `BATCH_SIZE` based on available GPU memory.
*   Modify hyperparameters like `RIDGE_ALPHA`, `KNN_N_NEIGHBORS`, Stable Diffusion settings if needed for experimentation.

## Running the Experiment (`run_experiment.py`)

The main script orchestrates the pipeline. Execute it from the command line within the repository's root directory.

**Command:**

```bash
python run_experiment.py --model_name <visual_model> [OPTIONS]
```

**Arguments:**

*   `--model_name <visual_model>`: **(Required)** Specifies the visual embedding model to use.
    *   Choices: `resnet50`, `vit`, `clip`
*   `--download`: (Optional) Attempts to download data using URLs in `download_data.py` (Use with caution, see Data Setup).
*   `--force_retrain`: (Optional) Forces retraining of the Ridge mapping model and the k-NN retrieval model, ignoring any saved files. Use this after code changes or for the first run.
*   `--visualize`: (Optional) Generates and saves a plot comparing ground truth and generated images for the first few test samples.

**Examples:**

```bash
# Run with ResNet50 embeddings (using Ridge mapping)
python run_experiment.py --model_name resnet50 --visualize

# Run with ViT embeddings (using Ridge mapping), force retraining
python run_experiment.py --model_name vit --force_retrain --visualize

# Run with CLIP embeddings (using Ridge mapping)
python run_experiment.py --model_name clip --visualize
```

*(Note: The `--mapping_model` argument is not used in this version, as it defaults to Ridge).*

## Output

The script generates several outputs saved in the directory specified by `OUTPUT_BASE_PATH` in `config.py` (defaults to `/kaggle/working/output` or `./output`):

*   **`output/tiny_imagenet_features/`**: Saved NumPy arrays (`.npy`) containing precomputed Tiny ImageNet features for each visual model, plus label (`.npy`) and path (`.pkl`, if implemented) files.
*   **`output/models/`**: Saved mapping models. For this version, contains Ridge model files (`ridge_mapping_<model_name>_... .sav`).
*   **`output/knn_models/`**: Saved k-NN retrieval models (`knn_<model_name>_... .sav`).
*   **`output/generated_images/<model_name>/`**: Contains the generated images (`.png`) alongside copies of their corresponding ground truth images (`.png`).
*   **`output/evaluation_results/`**:
    *   `evaluation_results_<model_name>_ridge.csv`: CSV file containing SSIM, LPIPS, CLIP-Sim scores for each generated image.
    *   `prediction_metrics_<model_name>_ridge.csv`: CSV file containing intermediate evaluation metrics (RMSE, R2, Cosine Sim) for the fMRI-to-embedding prediction step.
    *   `visualization_<model_name>_ridge.png`: Comparison plot image (if `--visualize` flag is used).

## Code Structure

*   `config.py`: Configuration settings and hyperparameters.
*   `download_data.py`: Utilities for downloading and organizing data (use with caution).
*   `data_loading.py`: Classes and functions for loading GOD fMRI and stimuli data.
*   `feature_extraction.py`: Handles loading embedding models and extracting features.
*   `mapping_models.py`: Implements Ridge Regression mapping (and potentially others in different versions).
*   `retrieval.py`: Implements k-NN fitting and nearest neighbor retrieval.
*   `generation.py`: Implements image generation using Stable Diffusion (Text2Img in this version).
*   `evaluation.py`: Implements quantitative evaluation metrics.
*   `run_experiment.py`: Main script to run the full pipeline.
*   `requirements.txt`: Lists Python package dependencies.

## References

