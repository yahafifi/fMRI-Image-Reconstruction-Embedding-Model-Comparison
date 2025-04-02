# run_experiment.py
import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import traceback # Import traceback for detailed error printing
import torch

# Import project modules
import config # Import config object directly
# *** Ensure you have a download_data.py file with a download_all_data function ***
import download_data
import data_loading
import feature_extraction
import mapping_models # Uses the updated mapping_models.py
import generation # Uses the generation.py from previous steps
import evaluation

def main(args):
    """Runs the fMRI decoding experiment using the Advanced MLP for mapping."""
    start_time = time.time()
    embedding_model_name = args.model_name
    mapping_method = args.mapping_method

    print(f"--- Starting Experiment ---")
    print(f"Target Embedding Model: {embedding_model_name.upper()}")
    print(f"Mapping Method: {mapping_method.upper()}")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 1. Data Download (Optional) ---
    # <<< --- REINSTATED DOWNLOAD LOGIC --- >>>
    if args.download:
        print("\n--- Attempting Data Download ---")
        # Ensure download_data.py exists and has download_all_data() function
        try:
            if not download_data.download_all_data():
                print("Data download/setup failed based on download_data script return value. Exiting.")
                return
            else:
                print("Data download/setup step completed.")
        except AttributeError:
            print("Error: download_data.py does not have a 'download_all_data' function.")
            print("Please implement the download logic in download_data.py. Exiting.")
            return
        except Exception as download_e:
            print(f"An error occurred during data download: {download_e}")
            traceback.print_exc()
            print("Exiting due to download error.")
            return
    else:
        print("\n--- Skipping Data Download ---")
    # <<< --- END REINSTATED DOWNLOAD LOGIC --- >>>

    # Basic checks remain useful even after download attempt
    god_fmri_file = os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5")
    god_train_dir = os.path.join(config.GOD_IMAGENET_PATH, "training")
    if not os.path.exists(god_fmri_file): print(f"Warning: GOD fMRI file not found: {god_fmri_file}")
    if not os.path.exists(god_train_dir): print(f"Warning: GOD stimuli 'training' directory not found: {god_train_dir}")


    # --- 2. Load fMRI Data and Prepare Dataloaders ---
    print("\n--- Loading GOD fMRI Data ---")
    # ... (rest of the main function remains the same as the previous version) ...
    # ... (Data Loading, Feature Extraction, Mapping Model Train/Load, Prediction, Generation, Evaluation, Visualization) ...
    try:
        handler = data_loading.GodFmriDataHandler(
            subject_id=config.SUBJECT_ID,
            roi=config.ROI,
            data_dir=config.GOD_FMRI_PATH,
            image_dir=config.GOD_IMAGENET_PATH
        )
        # Make sure get_data_splits returns numpy arrays for X and lists for image paths
        data_splits = handler.get_data_splits(
             normalize_runs=True, # Keep normalization based on previous setup
             test_split_size=config.TEST_SPLIT_SIZE,
             random_state=config.RANDOM_STATE
        )
        image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)
        dataloaders = data_loading.get_dataloaders(
            god_data_splits=data_splits,
            # Use config batch size for feature extraction dataloaders
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            image_transform=image_transform
        )
        test_avg_gt_paths = data_splits['test_avg'][1]
        if not test_avg_gt_paths:
             print("Error: No averaged test set ground truth image paths found.")
             return

    except Exception as e:
        print(f"Error during data loading: {e}")
        traceback.print_exc(); return

    # --- 3. Extract GOD Image Embeddings (Target Embeddings) ---
    # Using the model specified by args.model_name (e.g., 'clip')
    print(f"\n--- Extracting Target GOD Image Embeddings ({embedding_model_name}) ---")
    X_train, Z_train, X_val, Z_val, X_test_avg, Z_test_avg_true = [None] * 6 # Initialize
    try:
        # Load the specified embedding model (e.g., CLIP ViT-L/14 for 768D)
        target_embedding_model, _ = feature_extraction.load_embedding_model(embedding_model_name)
        if target_embedding_model is None: raise ValueError("Failed to load target embedding model.")

        if dataloaders.get('train'):
            # Returns fMRI (X) and Embeddings (Z)
            X_train, Z_train = feature_extraction.extract_features(
                target_embedding_model, dataloaders['train'], embedding_model_name, config.DEVICE
            )
        else: raise ValueError("Train dataloader is missing.")

        if dataloaders.get('val'):
             X_val, Z_val = feature_extraction.extract_features(
                 target_embedding_model, dataloaders['val'], embedding_model_name, config.DEVICE
             )
             print(f"Extracted Validation features: X={X_val.shape if X_val is not None else 'None'}, Z={Z_val.shape if Z_val is not None else 'None'}") # Added safe print
        else: print("No validation set found or loaded.")

        if dataloaders.get('test_avg'):
             X_test_avg, Z_test_avg_true = feature_extraction.extract_features(
                 target_embedding_model, dataloaders['test_avg'], embedding_model_name, config.DEVICE
             )
             print(f"Extracted Averaged Test features: X={X_test_avg.shape}, Z_true={Z_test_avg_true.shape}")
        else: raise ValueError("Test (Averaged) dataloader is missing.")

        # Simple validation
        if Z_train.shape[1] != config.EMBEDDING_MODELS[embedding_model_name]['embedding_dim']:
             raise ValueError(f"Extracted training embedding dim {Z_train.shape[1]} != config dim {config.EMBEDDING_MODELS[embedding_model_name]['embedding_dim']}")

    except Exception as e:
        print(f"Error during target GOD feature extraction: {e}")
        traceback.print_exc(); return

    # --- 4. Train/Load Mapping Model (fMRI -> Embedding) ---
    print(f"\n--- Training/Loading {mapping_method.upper()} Mapping Model ({embedding_model_name}) ---")
    mapping_model = None
    mapping_model_filename = "" # Initialize filename

    try:
        # --- MLP BRANCH ---
        if mapping_method == 'mlp':
            mapping_model_filename = os.path.join(config.MODELS_BASE_PATH, f"adv_mlp_mapping_{embedding_model_name}.pth")
            if args.force_retrain or not os.path.exists(mapping_model_filename):
                print("Training new Advanced MLP model...")
                # Ensure validation data are numpy arrays or None
                X_val_np = X_val if isinstance(X_val, np.ndarray) and len(X_val)>0 else None
                Z_val_np = Z_val if isinstance(Z_val, np.ndarray) and len(Z_val)>0 else None

                mapping_model, saved_path = mapping_models.train_mlp_mapping(
                    X_train, Z_train, X_val_np, Z_val_np, embedding_model_name, config.DEVICE, config # Pass config object
                )
                if mapping_model is None: raise ValueError("MLP training failed.")
                mapping_model_filename = saved_path
            else:
                print(f"Loading existing Advanced MLP model from: {mapping_model_filename}")
                fmri_dim = X_train.shape[1]
                embedding_dim = Z_train.shape[1]
                mapping_model = mapping_models.load_mlp_model(
                    mapping_model_filename, fmri_dim, embedding_dim, config.DEVICE, config # Pass config
                )
                if mapping_model is None: raise FileNotFoundError("Failed to load MLP model.")

        # --- RIDGE BRANCH ---
        elif mapping_method == 'ridge':
            mapping_model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{embedding_model_name}_alpha{config.RIDGE_ALPHA}.sav")
            if args.force_retrain or not os.path.exists(mapping_model_filename):
                 print("Training new Ridge model...")
                 if X_train.shape[0] != Z_train.shape[0]: raise ValueError("Training data mismatch")
                 mapping_model, saved_path = mapping_models.train_ridge_mapping(
                      X_train, Z_train, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, embedding_model_name
                 )
                 if mapping_model is None: raise ValueError("Ridge training failed.")
                 mapping_model_filename = saved_path # Use returned path
            else:
                 print(f"Loading existing Ridge model from: {mapping_model_filename}")
                 mapping_model = mapping_models.load_ridge_model(mapping_model_filename)
                 if mapping_model is None: raise FileNotFoundError("Failed to load Ridge model.")
        else:
            raise ValueError(f"Unsupported mapping_method: {mapping_method}")

    except Exception as e:
        print(f"Error during mapping model training/loading: {e}")
        traceback.print_exc(); return


    # --- 5. Predict Embeddings from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings from fMRI ({mapping_method.upper()}) ---")
    prediction_metrics = {'mapping_method': mapping_method, 'embedding_model': embedding_model_name}
    query_embeddings = None # This will hold the final embeddings used for generation

    try:
        Z_test_avg_pred_raw = None
        # Predict using the chosen method
        if mapping_method == 'mlp':
            Z_test_avg_pred_raw = mapping_models.predict_embeddings_mlp(
                 mapping_model, X_test_avg, config.DEVICE, batch_size=config.MLP_BATCH_SIZE * 2
            )
        elif mapping_method == 'ridge':
             Z_test_avg_pred_raw = mapping_models.predict_embeddings_ridge(mapping_model, X_test_avg) # Use renamed ridge function

        if Z_test_avg_pred_raw is None: raise ValueError(f"{mapping_method.upper()} Prediction failed.")

        # Evaluate the RAW predictions (RMSE, R2, Cosine Similarity)
        print("Evaluating RAW embedding prediction performance:")
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true, Z_test_avg_pred_raw)
        # Calculate Cosine Similarity safely
        cos_sim = np.nan
        if Z_test_avg_true is not None and Z_test_avg_pred_raw is not None and len(Z_test_avg_true) == len(Z_test_avg_pred_raw):
             if np.all(np.isfinite(Z_test_avg_true)) and np.all(np.isfinite(Z_test_avg_pred_raw)):
                  norms_true = np.linalg.norm(Z_test_avg_true, axis=1)
                  norms_pred = np.linalg.norm(Z_test_avg_pred_raw, axis=1)
                  valid_indices = (norms_true > 1e-9) & (norms_pred > 1e-9) # Avoid division by zero
                  if np.any(valid_indices):
                       dot_products = np.sum(Z_test_avg_true[valid_indices] * Z_test_avg_pred_raw[valid_indices], axis=1)
                       cos_sim_valid = dot_products / (norms_true[valid_indices] * norms_pred[valid_indices])
                       cos_sim = np.mean(cos_sim_valid)
                       print(f"Evaluation - Cosine Similarity (avg over valid): {cos_sim:.4f}")
                  else:
                       print("Warning: Could not compute cosine similarity (zero vectors or all invalid).")
             else:
                 print("Warning: NaN/Inf detected in prediction/truth, skipping cosine similarity calculation.")
        else:
            print("Warning: Mismatch in truth/prediction data for cosine similarity.")


        prediction_metrics['rmse_raw'] = pred_rmse
        prediction_metrics['r2_raw'] = pred_r2
        prediction_metrics['cosine_sim_raw'] = cos_sim

        # --- Standardization Adjustment (Decision: OFF by default) ---
        APPLY_STANDARDIZATION = False
        if APPLY_STANDARDIZATION:
             print("Applying standardization adjustment to predicted embeddings...")
             # ... (standardization code remains commented out unless enabled) ...
             # query_embeddings = Z_test_avg_pred_adj
             # print("Using *adjusted* predicted embeddings for generation.")
             pass # Placeholder if uncommented
        else:
             print("Skipping standardization adjustment.")
             query_embeddings = Z_test_avg_pred_raw # USE RAW PREDICTIONS
             print("Using RAW predicted embeddings for generation.")


        # Save prediction metrics
        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"{mapping_method}_embedding_prediction_metrics_{embedding_model_name}.csv")
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

    except Exception as e:
        print(f"Error during embedding prediction or evaluation: {e}")
        traceback.print_exc(); return

    # --- 8. Generate Images using Stable Diffusion (Directly from Embeddings) ---
    print(f"\n--- Generating Images via Direct Embedding ({mapping_method.upper()}) ---")
    generated_images_pil = []
    eval_results_df = pd.DataFrame() # Initialize empty dataframe

    if query_embeddings is None or len(query_embeddings) == 0:
        print("No predicted embeddings available for generation. Skipping.")
    else:
        num_expected_gt = len(test_avg_gt_paths)
        if len(query_embeddings) != num_expected_gt:
            print(f"Warning: Embeddings count ({len(query_embeddings)}) != Expected GT count ({num_expected_gt}). Might misalign.")

        try:
            generated_images_pil = generation.generate_images_from_embeddings(
                query_embeddings,
                guidance_scale=config.STABLE_DIFFUSION_GUIDANCE_SCALE,
                num_inference_steps=config.NUM_GENERATION_STEPS
            )

            evaluation_pairs = []
            valid_indices_generated = []
            for i, gen_img in enumerate(generated_images_pil):
                 if gen_img is not None and i < len(test_avg_gt_paths) and test_avg_gt_paths[i] is not None:
                      evaluation_pairs.append((test_avg_gt_paths[i], gen_img))
                      valid_indices_generated.append(i)

            if not evaluation_pairs:
                 print("Image generation failed or no valid pairs found.")
            else:
                valid_gt_paths = [pair[0] for pair in evaluation_pairs]
                valid_generated_images = [pair[1] for pair in evaluation_pairs]
                print(f"Successfully generated and aligned {len(valid_generated_images)} images.")

                # --- 9. Save Generated Images ---
                output_tag = f"{embedding_model_name}_{mapping_method}_direct"
                generation.save_generated_images(valid_generated_images, valid_gt_paths, output_tag) # Use function from generation.py

                # --- 10. Evaluate Reconstructions ---
                print(f"\n--- Evaluating Reconstructions ({output_tag}) ---")
                eval_results_df = evaluation.evaluate_reconstructions(
                    valid_gt_paths, valid_generated_images, config.EVAL_METRICS
                )
                if eval_results_df is not None and not eval_results_df.empty:
                    eval_results_df['original_test_index'] = valid_indices_generated
                    print("Added 'original_test_index' to evaluation results.")

        except Exception as e:
            print(f"Error during image generation, saving, or evaluation: {e}")
            traceback.print_exc()


    # --- 11. Save Evaluation Results ---
    if eval_results_df is not None and not eval_results_df.empty:
         output_tag = f"{embedding_model_name}_{mapping_method}_direct"
         evaluation.save_evaluation_results(eval_results_df, output_tag) # Use function from evaluation.py
    else:
         print("Evaluation resulted in empty DataFrame or generation failed. No final results saved.")


    # --- 12. Basic Visualization (Optional) ---
    if args.visualize and eval_results_df is not None and not eval_results_df.empty:
        print("\n--- Visualizing Sample Results ---")
        if 'valid_gt_paths' in locals() and 'valid_generated_images' in locals():
            num_to_show = min(10, len(valid_gt_paths))
            if num_to_show > 0:
                 try:
                      fig, axes = plt.subplots(num_to_show, 2, figsize=(8, num_to_show * 4))
                      if num_to_show == 1: axes = np.array([axes]) # Ensure iterable
                      output_tag = f"{embedding_model_name}_{mapping_method}_direct"
                      fig.suptitle(f'Reconstructions ({output_tag})', fontsize=16)

                      for i in range(num_to_show):
                          gt_path_viz = valid_gt_paths[i]
                          gen_img_viz = valid_generated_images[i]
                          # Get original index from the aligned list
                          original_index = valid_indices_generated[i]

                          try:
                              gt_img_pil = Image.open(gt_path_viz).convert("RGB")
                              axes[i, 0].imshow(gt_img_pil)
                              axes[i, 0].set_title(f"Ground Truth {original_index}")
                              axes[i, 0].axis("off")

                              axes[i, 1].imshow(gen_img_viz)
                              axes[i, 1].set_title(f"Generated Image {original_index}")
                              axes[i, 1].axis("off")
                          except Exception as plot_e:
                              print(f"Error plotting sample {i} (Original Index: {original_index}): {plot_e}")
                              # Safe plotting error handling
                              ax_row = axes if axes.ndim == 1 else axes[i]
                              if len(ax_row) > 0: ax_row[0].set_title("Error Plotting GT"); ax_row[0].axis("off")
                              if len(ax_row) > 1: ax_row[1].set_title("Error Plotting Gen"); ax_row[1].axis("off")


                      plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                      vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{output_tag}.png")
                      plt.savefig(vis_filename)
                      print(f"Saved visualization to {vis_filename}")
                      plt.close(fig) # Close figure
                 except Exception as viz_e:
                      print(f"Error during visualization creation: {viz_e}")
                      traceback.print_exc()
            else: print("No valid generated images available for visualization.")
        else: print("Could not find valid generated images/paths for visualization.")


    end_time = time.time()
    print(f"\n--- Experiment Finished ---")
    print(f"Target Embedding: {embedding_model_name.upper()}, Mapping: {mapping_method.upper()}")
    print(f"Total Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment")
    parser.add_argument(
        "--model_name", type=str, default="clip", choices=list(config.EMBEDDING_MODELS.keys()),
        help="Name of the target visual embedding model (e.g., 'clip' for CLIP ViT-L/14)."
    )
    parser.add_argument(
        "--mapping_method", type=str, default="mlp", choices=['mlp', 'ridge'],
        help="Method used to map fMRI to embeddings ('mlp' or 'ridge')."
    )
    # <<< --- REINSTATED DOWNLOAD ARGUMENT --- >>>
    parser.add_argument(
        "--download", action="store_true",
        help="Run the data download step first (requires download_data.py)."
    )
    # <<< --- END REINSTATED ARGUMENT --- >>>
    parser.add_argument(
        "--force_retrain", action="store_true",
        help="Force retraining of the specified mapping model, even if saved files exist."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate and save a visualization of sample reconstructions."
    )

    args = parser.parse_args()

    # --- Basic Checks ---
    if args.model_name == 'clip' and config.EMBEDDING_MODELS['clip']['embedding_dim'] != 768:
         print("CRITICAL WARNING: Config 'clip' embedding dimension is not 768, but SD v1.5 expects 768. Generation will likely fail. Please correct config.py.")
         # exit() # Optionally exit if critical

    if not torch.cuda.is_available():
        print("Warning: CUDA device not found, using CPU. This will be extremely slow.")
    else:
        print(f"Using CUDA Device: {torch.cuda.get_device_name(0)}")

    main(args)