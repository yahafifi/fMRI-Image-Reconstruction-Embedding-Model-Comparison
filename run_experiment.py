# run_experiment.py
import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import traceback # Import traceback for detailed error printing
import torch # Added torch import

# Import project modules
import config
import download_data # Keep for --download option
import data_loading
import feature_extraction
import mapping_models
# import retrieval # Retrieval module no longer needed for core path
import generation # Uses the updated generation.py
import evaluation

def main(args):
    """Runs the fMRI decoding experiment using direct embedding conditioning."""
    start_time = time.time()
    model_name = args.model_name # Refers to the *embedding* model (resnet50, vit, clip)

    print(f"--- Starting Experiment for Embedding Model: {model_name.upper()} ---")
    print(f"--- Generating via DIRECT EMBEDDING CONDITIONING ---")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 1. Data Download (Optional) ---
    if args.download:
        print("\n--- Attempting Data Download ---")
        # Assume download_data handles all necessary downloads (GOD, stimuli)
        if not download_data.download_all_data(): # Make sure this function exists and works
            print("Data download/setup failed. Please check URLs and paths. Exiting.")
            return
        else:
            print("Data download/setup step completed.")
    else:
        print("\n--- Skipping Data Download ---")
        # Basic checks (paths from config)
        god_fmri_file = os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5")
        god_train_dir = os.path.join(config.GOD_IMAGENET_PATH, "training")
        if not os.path.exists(god_fmri_file): print(f"Warning: GOD fMRI file not found: {god_fmri_file}")
        if not os.path.exists(god_train_dir): print(f"Warning: GOD stimuli 'training' directory not found: {god_train_dir}")
        # Retrieval DB check removed as it's not needed for generation now


    # --- 2. Load fMRI Data and Prepare Dataloaders ---
    print("\n--- Loading GOD fMRI Data ---")
    try:
        handler = data_loading.GodFmriDataHandler(
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
        image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)
        dataloaders = data_loading.get_dataloaders(
            god_data_splits=data_splits,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            image_transform=image_transform
        )
        # Store ground truth paths for test set (averaged) for later evaluation
        test_avg_gt_paths = data_splits['test_avg'][1]
        if not test_avg_gt_paths:
             print("Error: No averaged test set ground truth image paths found after data loading.")
             return

    except Exception as e:
        print(f"Error during data loading: {e}")
        traceback.print_exc()
        return

    # --- 3. Extract GOD Image Embeddings (for mapping target) ---
    print(f"\n--- Extracting GOD Image Embeddings ({model_name}) ---")
    try:
        embedding_model, _ = feature_extraction.load_embedding_model(model_name)
        if embedding_model is None: raise ValueError("Failed to load embedding model.")

        # Extract for Training set
        if dataloaders.get('train'):
            X_train, Z_train = feature_extraction.extract_features(
                embedding_model, dataloaders['train'], model_name, config.DEVICE
            )
        else:
             print("Error: Train dataloader is missing.")
             return

        # Extract for Validation set (optional, for evaluating mapping)
        Z_val = None # Initialize
        if dataloaders.get('val'):
             X_val, Z_val = feature_extraction.extract_features(
                 embedding_model, dataloaders['val'], model_name, config.DEVICE
             )
             print(f"Extracted Validation features: X={X_val.shape}, Z={Z_val.shape}")
        # else: X_val, Z_val remain None or empty

        # Extract for Averaged Test set (ground truth embeddings for eval)
        if dataloaders.get('test_avg'):
             X_test_avg, Z_test_avg_true = feature_extraction.extract_features(
                 embedding_model, dataloaders['test_avg'], model_name, config.DEVICE
             )
             print(f"Extracted Averaged Test features: X={X_test_avg.shape}, Z_true={Z_test_avg_true.shape}")
        else:
             print("Error: Test (Averaged) dataloader is missing.")
             return

    except Exception as e:
        print(f"Error during GOD feature extraction: {e}")
        traceback.print_exc()
        return

    # --- 4. Train/Load Mapping Model (fMRI -> Embedding) ---
    # Using Ridge as requested by user
    print(f"\n--- Training/Loading Ridge Mapping Model ({model_name}) ---")
    ridge_model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{config.RIDGE_ALPHA}.sav")
    ridge_model = None
    if args.force_retrain or not os.path.exists(ridge_model_filename):
        print("Training new Ridge model...")
        try:
            if X_train.shape[0] != Z_train.shape[0]: raise ValueError("Training data mismatch")
            ridge_model, saved_path = mapping_models.train_ridge_mapping(
                X_train, Z_train, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name
            )
            if ridge_model is None: raise ValueError("Ridge training failed.")
            ridge_model_filename = saved_path
        except Exception as e:
            print(f"Error training Ridge model: {e}")
            traceback.print_exc(); return
    else:
        print(f"Loading existing Ridge model from: {ridge_model_filename}")
        try:
            ridge_model = mapping_models.load_ridge_model(ridge_model_filename)
            if ridge_model is None: raise FileNotFoundError("Failed to load ridge model.")
        except Exception as e:
            print(f"Error loading Ridge model: {e}")
            traceback.print_exc(); return

    # --- 5. Predict Embeddings from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings from fMRI ({model_name}) ---")
    prediction_metrics = {}
    Z_test_avg_pred_adj = None # Initialize
    try:
        Z_test_avg_pred = mapping_models.predict_embeddings(ridge_model, X_test_avg)
        if Z_test_avg_pred is None: raise ValueError("Prediction failed.")

        print("Evaluating RAW embedding prediction performance (RMSE, R2):")
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true, Z_test_avg_pred)
        prediction_metrics['rmse_raw'] = pred_rmse; prediction_metrics['r2_raw'] = pred_r2

        # Apply standardization adjustment
        print("Applying standardization adjustment to predicted embeddings...")
        epsilon = 1e-9
        train_mean = np.mean(Z_train, axis=0); train_std = np.std(Z_train, axis=0)
        pred_mean = np.mean(Z_test_avg_pred, axis=0); pred_std = np.std(Z_test_avg_pred, axis=0)
        Z_test_avg_pred_adj = ((Z_test_avg_pred - pred_mean) / (pred_std + epsilon)) * train_std + train_mean
        print("Standardization complete.")

        print("Evaluating ADJUSTED embedding prediction performance (RMSE, R2):")
        adj_pred_rmse, adj_pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true, Z_test_avg_pred_adj)
        prediction_metrics['rmse_adj'] = adj_pred_rmse; prediction_metrics['r2_adj'] = adj_pred_r2

        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"embedding_prediction_metrics_{model_name}.csv")
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

        # --- Use adjusted embeddings for generation ---
        query_embeddings = Z_test_avg_pred_adj
        print("Using *adjusted* predicted embeddings for generation.")

    except Exception as e:
        print(f"Error during embedding prediction or evaluation: {e}")
        traceback.print_exc(); return

    # --- Steps 6 & 7 (Retrieval) are REMOVED ---
    # No need for ImageNet-256 features (unless used for separate eval later)
    # No need for k-NN training or retrieval

    # --- 8. Generate Images using Stable Diffusion (Directly from Embeddings) ---
    print(f"\n--- Generating Images using Stable Diffusion ({model_name}) ---")
    generated_images_pil = [] # Initialize
    eval_results_df = pd.DataFrame() # Initialize empty dataframe

    if query_embeddings is None or len(query_embeddings) == 0:
        print("No predicted embeddings available for generation. Skipping generation and evaluation.")
    else:
        # Check number of embeddings matches number of GT images expected
        num_expected_gt = len(test_avg_gt_paths)
        if len(query_embeddings) != num_expected_gt:
            print(f"Warning: Number of predicted embeddings ({len(query_embeddings)}) differs from expected averaged test samples ({num_expected_gt}). Generation/Evaluation might be misaligned.")
            # Adjust lists if needed, e.g., truncate gt_paths? Or generate only for available embeddings?
            # Safest: Only evaluate pairs where both exist.

        try:
            # *** Call the NEW generation function ***
            generated_images_pil = generation.generate_images_from_embeddings(
                query_embeddings,
                guidance_scale=config.STABLE_DIFFUSION_GUIDANCE_SCALE,
                num_inference_steps=30 # Or config value
            )

            # Align generated images with ground truth paths
            evaluation_pairs = []
            valid_indices_generated = []
            for i, gen_img in enumerate(generated_images_pil):
                 if gen_img is not None and i < len(test_avg_gt_paths):
                      evaluation_pairs.append((test_avg_gt_paths[i], gen_img))
                      valid_indices_generated.append(i)

            if not evaluation_pairs:
                 print("Image generation failed for all samples or alignment failed.")
            else:
                valid_gt_paths = [pair[0] for pair in evaluation_pairs]
                valid_generated_images = [pair[1] for pair in evaluation_pairs]
                print(f"Successfully generated and aligned {len(valid_generated_images)} images with ground truths.")

                # --- 9. Save Generated Images ---
                generation.save_generated_images(valid_generated_images, valid_gt_paths, f"{model_name}_direct_embed") # Add identifier

                # --- 10. Evaluate Reconstructions ---
                print(f"\n--- Evaluating Reconstructions ({model_name}) ---")
                eval_results_df = evaluation.evaluate_reconstructions(
                    valid_gt_paths, valid_generated_images, config.EVAL_METRICS
                )
                # Add original index back
                if eval_results_df is not None and not eval_results_df.empty:
                    eval_results_df['original_test_index'] = valid_indices_generated
                    print("Added 'original_test_index' to evaluation results.")

        except Exception as e:
            print(f"Error during image generation, saving, or evaluation: {e}")
            traceback.print_exc()
            # eval_results_df remains empty if error occurs here

    # --- 11. Save Evaluation Results ---
    if eval_results_df is not None and not eval_results_df.empty:
         evaluation.save_evaluation_results(eval_results_df, f"{model_name}_direct_embed") # Add identifier
    else:
         print("Evaluation resulted in empty DataFrame or generation failed. No final results saved.")

    # --- 12. Basic Visualization (Optional) ---
    if args.visualize and eval_results_df is not None and not eval_results_df.empty:
        print("\n--- Visualizing Sample Results ---")
        if 'valid_gt_paths' in locals() and 'valid_generated_images' in locals():
            num_to_show = min(10, len(valid_gt_paths)) # Show more samples
            if num_to_show > 0:
                 try:
                      fig, axes = plt.subplots(num_to_show, 2, figsize=(8, num_to_show * 4))
                      if num_to_show == 1: axes = np.array([axes])
                      fig.suptitle(f'Reconstructions (Direct Embedding) - {model_name.upper()}', fontsize=16)

                      for i in range(num_to_show):
                          gt_path_viz = valid_gt_paths[i]
                          gen_img_viz = valid_generated_images[i]
                          original_index = valid_indices_generated[i]

                          try:
                              gt_img_pil = Image.open(gt_path_viz).convert("RGB")
                              axes[i, 0].imshow(gt_img_pil)
                              axes[i, 0].set_title(f"Ground Truth {original_index}")
                              axes[i, 0].axis("off")

                              axes[i, 1].imshow(gen_img_viz)
                              # *** Remove prompt from title ***
                              axes[i, 1].set_title(f"Generated Image {original_index}")
                              axes[i, 1].axis("off")
                          except Exception as plot_e:
                              print(f"Error plotting sample {i} (Original Index: {original_index}): {plot_e}")
                              # Handle plot errors for individual images

                      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                      vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{model_name}_direct_embed.png")
                      plt.savefig(vis_filename)
                      print(f"Saved visualization to {vis_filename}")
                      plt.close(fig)
                 except Exception as viz_e:
                      print(f"Error during visualization creation: {viz_e}")
                      traceback.print_exc()
            else: print("No valid generated images available for visualization.")
        else: print("Could not find valid generated images/paths for visualization.")


    end_time = time.time()
    print(f"\n--- Experiment for {model_name.upper()} (Direct Embedding) Finished ---")
    print(f"Total Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment (Direct Embedding Conditioning)")
    parser.add_argument(
        "--model_name", type=str, required=True, choices=list(config.EMBEDDING_MODELS.keys()),
        help="Name of the visual embedding model used for mapping (e.g., clip)."
    )
    parser.add_argument(
        "--download", action="store_true", help="Run the data download step first."
    )
    parser.add_argument(
        "--force_retrain", action="store_true", help="Force retraining of mapping (Ridge) model."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate and save a visualization of sample reconstructions."
    )

    args = parser.parse_args()

    # --- Ensure Device is Set Early ---
    # config.py should handle this, but double check
    if torch.cuda.is_available():
        print(f"CUDA Device Found: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA device not found, using CPU (will be very slow).")

    main(args)

    # Example usage from command line:
    # python run_experiment.py --model_name clip --visualize
    # python run_experiment.py --model_name clip --force_retrain --download