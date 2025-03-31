# run_experiment.py
import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import traceback # Import traceback for detailed error printing
import torch # Need torch

# Import project modules
import config
import download_data
import data_loading
import feature_extraction
import mapping_models # This now contains MLP logic
import retrieval
import generation
import evaluation

def main(args):
    """Runs the full fMRI decoding experiment for a given embedding model."""
    start_time = time.time()
    visual_embedding_model_name = args.model_name # Use a more descriptive name
    fmri_source_name = f"Subj{config.SUBJECT_ID}_{config.ROI}" # Identifier for fMRI data source

    print(f"--- Starting Experiment ---")
    print(f"Visual Embedding: {visual_embedding_model_name.upper()}")
    print(f"fMRI Source:      {fmri_source_name}")
    print(f"Mapping Model:    {config.MAPPING_MODEL_TYPE.upper()}")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 1. Data Download (Optional) ---
    if args.download:
        print("\n--- Attempting Data Download ---")
        if not download_data.download_all_data():
            print("Data download/setup failed. Please check URLs and paths. Exiting.")
            return
        else:
            print("Data download/setup step completed.")
    else:
        # Basic check if essential data seems present
        print("\n--- Skipping Data Download ---")
        god_fmri_file = os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5")
        god_train_dir = os.path.join(config.GOD_IMAGENET_PATH, "training")
        imagenet256_dir = config.IMAGENET256_PATH # Check the base retrieval dir

        # Add checks for existence and exit if critical data is missing
        if not os.path.exists(god_fmri_file):
             print(f"CRITICAL ERROR: GOD fMRI file not found at {god_fmri_file}. Check path or run with --download.")
             return
        if not os.path.exists(god_train_dir):
             print(f"CRITICAL ERROR: GOD stimuli 'training' directory not found at {god_train_dir}. Check path or run with --download.")
             return
        if not os.path.exists(imagenet256_dir):
             print(f"CRITICAL ERROR: ImageNet-256 directory not found at {imagenet256_dir}. Check path/dataset name in config.")
             return
        print("Basic data checks passed.")


    # --- 2. Load fMRI Data and Prepare Dataloaders ---
    print("\n--- Loading GOD fMRI Data ---")
    try:
        handler = data_loading.GodFmriDataHandler(
            subject_id=config.SUBJECT_ID,
            roi=config.ROI,
            data_dir=config.GOD_FMRI_PATH,
            image_dir=config.GOD_IMAGENET_PATH
        )
        # Get the raw numpy arrays from the handler for MLP training
        data_splits_numpy = handler.get_data_splits(
             normalize_runs=True, # Keep run-wise normalization
             test_split_size=config.TEST_SPLIT_SIZE, # Use config for validation split size
             random_state=config.RANDOM_STATE
        )
        # Extract numpy arrays needed for mapping model training/evaluation
        X_train_np, _ = data_splits_numpy['train']
        X_val_np, _ = data_splits_numpy['val']
        X_test_avg_np, test_avg_gt_paths = data_splits_numpy['test_avg'] # Keep paths for later

        # Add checks for empty arrays after splitting
        if X_train_np.size == 0: raise ValueError("Training fMRI data (X_train_np) is empty after splitting.")
        if X_test_avg_np.size == 0: raise ValueError("Test (Avg) fMRI data (X_test_avg_np) is empty after splitting.")
        if not test_avg_gt_paths: raise ValueError("No test (Avg) ground truth image paths found after splitting.")
        # Handle case where validation split might be 0
        if X_val_np.size == 0:
            print("Note: Validation set size is 0 based on test_split_size. MLP will train without validation-based early stopping/best model selection.")

        # Create dataloaders (primarily for feature extraction)
        image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)
        dataloaders = data_loading.get_dataloaders(
            god_data_splits=data_splits_numpy, # Use the numpy splits to initialize datasets
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            image_transform=image_transform
        )

    except Exception as e:
        print(f"Error during data loading: {e}")
        traceback.print_exc()
        return


    # --- 3. Extract GOD Image Embeddings (for mapping targets) ---
    # We need Z_train, Z_val, Z_test_avg_true as numpy arrays
    print(f"\n--- Extracting GOD Image Embeddings ({visual_embedding_model_name}) ---")
    Z_train_np, Z_val_np, Z_test_avg_true_np = None, None, None # Initialize
    try:
        embedding_model, _ = feature_extraction.load_embedding_model(visual_embedding_model_name)
        if embedding_model is None: raise ValueError("Failed to load embedding model.")

        # Extract for Training set targets
        if dataloaders.get('train'):
            _, Z_train_np = feature_extraction.extract_features(
                embedding_model, dataloaders['train'], visual_embedding_model_name, config.DEVICE
            )
            if Z_train_np.shape[0] != X_train_np.shape[0]:
                 raise ValueError(f"Shape mismatch after train feature extraction: X={X_train_np.shape}, Z={Z_train_np.shape}")
            print(f"Extracted Training target embeddings: Z_train={Z_train_np.shape}")
        else:
             print("Error: Train dataloader missing, cannot extract training embeddings.")
             return

        # Extract for Validation set targets
        if dataloaders.get('val') and X_val_np.size > 0 : # Check if val data exists
             _, Z_val_np = feature_extraction.extract_features(
                 embedding_model, dataloaders['val'], visual_embedding_model_name, config.DEVICE
             )
             if Z_val_np.shape[0] != X_val_np.shape[0]:
                  raise ValueError(f"Shape mismatch after val feature extraction: X={X_val_np.shape}, Z={Z_val_np.shape}")
             print(f"Extracted Validation target embeddings: Z_val={Z_val_np.shape}")
        else:
             Z_val_np = np.array([]) # Ensure it's an empty array if no validation data
             print("No validation set target embeddings extracted (or validation set is empty).")


        # Extract for Averaged Test set (ground truth targets)
        if dataloaders.get('test_avg'):
             _, Z_test_avg_true_np = feature_extraction.extract_features(
                 embedding_model, dataloaders['test_avg'], visual_embedding_model_name, config.DEVICE
             )
             if Z_test_avg_true_np.shape[0] != X_test_avg_np.shape[0]:
                  raise ValueError(f"Shape mismatch after test feature extraction: X={X_test_avg_np.shape}, Z_true={Z_test_avg_true_np.shape}")
             print(f"Extracted Averaged Test target embeddings: Z_true={Z_test_avg_true_np.shape}")
        else:
             print("Error: Test (Averaged) dataloader missing, cannot extract test embeddings.")
             return

    except Exception as e:
        print(f"Error during GOD target feature extraction: {e}")
        traceback.print_exc()
        return


    # --- 4. Train/Load Mapping Model (fMRI -> Embedding) ---
    print(f"\n--- Training/Loading {config.MAPPING_MODEL_TYPE.upper()} Mapping Model ({visual_embedding_model_name}) ---")

    mapping_model = None
    model_filename = None

    try:
        if config.MAPPING_MODEL_TYPE == "mlp":
            model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{fmri_source_name}_{visual_embedding_model_name}.pt")
            if args.force_retrain or not os.path.exists(model_filename):
                # Pass necessary numpy arrays to training function
                mapping_model, saved_path = mapping_models.train_mlp_mapping(
                    X_train_np, Z_train_np, X_val_np, Z_val_np,
                    visual_embedding_model_name, fmri_source_name
                )
                if mapping_model is None: raise ValueError("MLP training function returned None.")
                # Update filename only if saving was successful and returned a path
                model_filename = saved_path if saved_path else model_filename
            else:
                print(f"Loading existing MLP model from: {model_filename}")
                mapping_model = mapping_models.load_mlp_model(model_filename)
                if mapping_model is None: raise FileNotFoundError(f"Failed to load MLP model from {model_filename}.")

        elif config.MAPPING_MODEL_TYPE == "ridge":
            model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{fmri_source_name}_{visual_embedding_model_name}_alpha{config.RIDGE_ALPHA}.sav")
            if args.force_retrain or not os.path.exists(model_filename):
                mapping_model, saved_path = mapping_models.train_ridge_mapping(
                    X_train_np, Z_train_np, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER,
                    visual_embedding_model_name, fmri_source_name
                )
                if mapping_model is None: raise ValueError("Ridge training function returned None.")
                model_filename = saved_path if saved_path else model_filename
            else:
                print(f"Loading existing Ridge model from: {model_filename}")
                mapping_model = mapping_models.load_ridge_model(model_filename)
                if mapping_model is None: raise FileNotFoundError(f"Failed to load Ridge model from {model_filename}.")
        else:
            raise ValueError(f"Unsupported MAPPING_MODEL_TYPE in config: {config.MAPPING_MODEL_TYPE}")

    except Exception as e:
        print(f"Error during mapping model training/loading: {e}")
        traceback.print_exc()
        return # Exit if model cannot be obtained

    if mapping_model is None:
        print("Failed to obtain a mapping model. Exiting.")
        return

    # --- 5. Predict Embeddings from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings from fMRI ({visual_embedding_model_name} using {config.MAPPING_MODEL_TYPE.upper()}) ---")
    prediction_metrics = {} # Store prediction eval results
    Z_test_avg_pred_np = None # Initialize prediction variable
    query_embeddings = None   # Initialize query embeddings

    try:
        # Use the appropriate prediction function based on model type
        if config.MAPPING_MODEL_TYPE == "mlp":
            Z_test_avg_pred_np = mapping_models.predict_embeddings_mlp(mapping_model, X_test_avg_np)
        elif config.MAPPING_MODEL_TYPE == "ridge":
            Z_test_avg_pred_np = mapping_models.predict_embeddings_ridge(mapping_model, X_test_avg_np)

        if Z_test_avg_pred_np is None: raise ValueError("Prediction function returned None.")

        # --- IMPORTANT: Evaluate Embedding Prediction Quality FIRST ---
        print("\n--- Evaluating EMBEDDING Prediction Performance (using RAW predictions) ---")
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_np, Z_test_avg_pred_np, dataset_name="Test (Raw)")
        prediction_metrics['rmse_raw'] = pred_rmse
        prediction_metrics['r2_raw'] = pred_r2
        print("--------------------------------------------------------------------------")

        # Check R2 score and warn if poor
        if pred_r2 is None or np.isnan(pred_r2) or pred_r2 < 0.01: # Adjusted threshold slightly
             print(f"WARNING: Initial embedding prediction R2 score ({pred_r2:.4f}) is very low or NaN.")
             print("         The mapping model may not be capturing the fMRI-embedding relationship well.")
             print("         Downstream results (retrieval, generation) will likely be poor.")
             # Consider adding an option to exit here:
             # if args.exit_on_poor_mapping:
             #    print("Exiting due to poor mapping performance.")
             #    return

        # --- Apply standardization adjustment (Optional - Disabled by default) ---
        apply_adjustment = False # <<< DISABLE ADJUSTMENT BY DEFAULT >>>
        if apply_adjustment:
            print("Applying standardization adjustment to predicted embeddings...")
            epsilon = 1e-9
            # Ensure Z_train_np exists and is not empty
            if Z_train_np is not None and Z_train_np.size > 0:
                train_mean = np.mean(Z_train_np, axis=0)
                train_std = np.std(Z_train_np, axis=0)
                pred_mean = np.mean(Z_test_avg_pred_np, axis=0)
                pred_std = np.std(Z_test_avg_pred_np, axis=0)

                # Add epsilon to denominator std dev to prevent division by zero or NaNs
                Z_test_avg_pred_adj = ((Z_test_avg_pred_np - pred_mean) / (pred_std + epsilon)) * (train_std + epsilon) + train_mean
                print("Standardization complete.")

                # Evaluate adjusted predictions too
                print("Evaluating ADJUSTED embedding prediction performance:")
                adj_pred_rmse, adj_pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_np, Z_test_avg_pred_adj, dataset_name="Test (Adjusted)")
                prediction_metrics['rmse_adj'] = adj_pred_rmse
                prediction_metrics['r2_adj'] = adj_pred_r2

                query_embeddings = Z_test_avg_pred_adj # Use adjusted
                print("Using *adjusted* predicted embeddings for retrieval.")
            else:
                print("Warning: Cannot apply adjustment because Z_train_np is missing or empty. Using raw predictions.")
                query_embeddings = Z_test_avg_pred_np # Fallback to raw
                print("Using *raw* predicted embeddings for retrieval.")
        else:
            print("Skipping standardization adjustment.")
            query_embeddings = Z_test_avg_pred_np # Use raw predictions
            print("Using *raw* predicted embeddings for retrieval.")


        # Save prediction metrics
        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"embedding_prediction_metrics_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}.csv")
        os.makedirs(config.EVALUATION_RESULTS_PATH, exist_ok=True) # Ensure dir exists
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

    except Exception as e:
        print(f"Error during embedding prediction or evaluation: {e}")
        traceback.print_exc()
        return # Exit if prediction fails

    # Check if query_embeddings were successfully assigned
    if query_embeddings is None:
        print("Error: Query embeddings were not generated. Exiting.")
        return


    # --- 6. Precompute/Load ImageNet-256 Features & Train/Load k-NN ---
    # ... (No changes needed in this section) ...
    print(f"\n--- Preparing ImageNet-256 Retrieval Database ({visual_embedding_model_name}) ---")
    knn_model = None
    db_features, db_labels, db_class_map = None, None, None # Initialize
    try:
        db_features, db_labels, db_class_map = feature_extraction.precompute_imagenet256_features(visual_embedding_model_name)
        if db_features is None or db_labels is None or db_class_map is None:
             raise ValueError("Failed to load or compute ImageNet-256 features/labels/map.")

        knn_model_filename = os.path.join(config.SAVED_KNN_MODELS_PATH, f"knn_{visual_embedding_model_name}_k{config.KNN_N_NEIGHBORS}.sav")
        if args.force_retrain or not os.path.exists(knn_model_filename):
            print("Training new k-NN model...")
            knn_model, _ = retrieval.train_knn_retrieval(db_features, config.KNN_N_NEIGHBORS, visual_embedding_model_name)
        else:
            print(f"Loading existing k-NN model from {knn_model_filename}...")
            knn_model = retrieval.load_knn_model(knn_model_filename)

        if knn_model is None: raise ValueError("Failed to train or load k-NN model.")

    except Exception as e:
        print(f"Error preparing retrieval database or k-NN model: {e}")
        traceback.print_exc()
        return

    # --- 7. Retrieve Neighbor Labels from ImageNet-256 ---
    # ... (No changes needed in this section) ...
    print(f"\n--- Retrieving Semantic Labels using k-NN ({visual_embedding_model_name}) ---")
    retrieved_readable_labels = None
    top1_prompts = []
    try:
        indices, distances, retrieved_readable_labels = retrieval.retrieve_nearest_neighbors(
            knn_model, query_embeddings, db_labels, db_class_map
        )
        if retrieved_readable_labels is None:
             print("Label retrieval failed or returned None. Proceeding without prompts.")
        else:
            # Ensure sublists are not empty before accessing index 0
            top1_prompts = [labels[0] for labels in retrieved_readable_labels if isinstance(labels, list) and len(labels) > 0]
            if len(top1_prompts) != len(query_embeddings):
                 print(f"Warning: Number of prompts generated ({len(top1_prompts)}) doesn't match number of queries ({len(query_embeddings)}). Some queries might not have valid labels retrieved.")
            print(f"Generated {len(top1_prompts)} top-1 prompts. Example: {top1_prompts[:5] if top1_prompts else 'None'}")

            # Save retrieval details...
            try:
                retrieval_info = {
                    'query_index': list(range(len(query_embeddings))),
                    'retrieved_indices': indices.tolist() if indices is not None else [None]*len(query_embeddings),
                    'retrieved_distances': distances.tolist() if distances is not None else [None]*len(query_embeddings),
                    'retrieved_labels': retrieved_readable_labels if retrieved_readable_labels is not None else [None]*len(query_embeddings),
                    'top1_prompt': top1_prompts + ([None]*(len(query_embeddings)-len(top1_prompts)) if len(top1_prompts) < len(query_embeddings) else []) # Pad if needed more safely
                }
                retrieval_df = pd.DataFrame(retrieval_info)
                retrieval_output_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"retrieval_details_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}.csv")
                retrieval_df.to_csv(retrieval_output_file, index=False)
                print(f"Saved retrieval details to {retrieval_output_file}")
            except Exception as save_e:
                print(f"Warning: Could not save retrieval details: {save_e}")

    except Exception as e:
        print(f"Error during label retrieval: {e}")
        traceback.print_exc()
        # Continue with empty prompts if retrieval fails


    # --- 8. Generate Images using Stable Diffusion ---
    # ... (No changes needed) ...
    print(f"\n--- Generating Images using Stable Diffusion ({visual_embedding_model_name}) ---")
    eval_results_df = None # Initialize
    valid_generated_images = [] # Initialize
    valid_gt_paths = [] # Initialize
    valid_indices_generated = [] # Initialize

    if not top1_prompts:
         print("No prompts available for generation. Skipping generation and evaluation.")
         eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS) # Create empty frame
    else:
        try:
            generated_images_pil = generation.generate_images_from_prompts(top1_prompts)
            evaluation_pairs = []
            for i, gen_img in enumerate(generated_images_pil):
                 # Check index bounds for test_avg_gt_paths
                 if i < len(test_avg_gt_paths):
                     if gen_img is not None: # Check if generation was successful for this prompt
                          evaluation_pairs.append((test_avg_gt_paths[i], gen_img))
                          valid_indices_generated.append(i) # Keep track of original index of successful pairs
                     else:
                          print(f"Note: Generation failed for prompt index {i} ('{top1_prompts[i]}'). Skipping.")
                 else:
                      print(f"Warning: Generated image index {i} out of bounds for ground truth paths (length {len(test_avg_gt_paths)}). Skipping.")


            if not evaluation_pairs:
                 print("Image generation failed for all prompts or alignment failed.")
                 eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)
            else:
                valid_gt_paths = [pair[0] for pair in evaluation_pairs]
                valid_generated_images = [pair[1] for pair in evaluation_pairs]
                print(f"Successfully generated and aligned {len(valid_generated_images)} images with ground truths.")

                # --- 9. Save Generated Images ---
                generation.save_generated_images(valid_generated_images, valid_gt_paths, visual_embedding_model_name)

                # --- 10. Evaluate Reconstructions ---
                print(f"\n--- Evaluating Reconstructions ({visual_embedding_model_name}) ---")
                eval_results_df = evaluation.evaluate_reconstructions(
                    valid_gt_paths, valid_generated_images, config.EVAL_METRICS
                )
                # Add original index back for clarity if needed
                if eval_results_df is not None and not eval_results_df.empty:
                    # Ensure the indices match the length of the dataframe rows
                    if len(valid_indices_generated) == len(eval_results_df):
                         eval_results_df['original_test_index'] = valid_indices_generated
                         print("Added 'original_test_index' to evaluation results.")
                    else:
                         print("Warning: Length mismatch between valid indices and evaluation results. Cannot add 'original_test_index'.")


        except Exception as e:
            print(f"Error during image generation, saving, or evaluation: {e}")
            traceback.print_exc()
            # Create empty eval results if generation failed badly
            eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)

    # --- 11. Save Evaluation Results ---
    if eval_results_df is not None:
         eval_results_filename = f"evaluation_results_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}.csv"
         evaluation.save_evaluation_results(eval_results_df, visual_embedding_model_name, filename=eval_results_filename)
    else:
         print("No evaluation results DataFrame generated or available to save.")


    # --- 12. Basic Visualization (Optional) ---
    if args.visualize and eval_results_df is not None and not eval_results_df.empty:
        print("\n--- Visualizing Sample Results ---")
        if valid_generated_images and valid_gt_paths: # Check if lists are populated
            num_to_show = min(5, len(valid_gt_paths))
            if num_to_show > 0:
                 try:
                      fig, axes = plt.subplots(num_to_show, 2, figsize=(8, num_to_show * 4))
                      if num_to_show == 1: axes = np.array([axes]) # Ensure axes is iterable even for 1 sample
                      fig.suptitle(f'Sample Reconstructions - {visual_embedding_model_name.upper()} ({config.MAPPING_MODEL_TYPE.upper()})', fontsize=16)

                      for i in range(num_to_show):
                          gt_path_viz = valid_gt_paths[i]
                          gen_img_viz = valid_generated_images[i]
                          original_index = valid_indices_generated[i] # Get index used for prompt list
                          prompt_viz = top1_prompts[original_index] if original_index < len(top1_prompts) else "N/A"

                          try:
                              gt_img_pil = Image.open(gt_path_viz).convert("RGB")
                              axes[i, 0].imshow(gt_img_pil)
                              axes[i, 0].set_title(f"Ground Truth {original_index}")
                              axes[i, 0].axis("off")
                              axes[i, 1].imshow(gen_img_viz)
                              axes[i, 1].set_title(f"Generated (Prompt: {prompt_viz})")
                              axes[i, 1].axis("off")
                          except Exception as plot_e:
                              print(f"Error plotting sample {i} (Original Index: {original_index}): {plot_e}")
                              if i < len(axes):
                                   axes[i, 0].set_title("Error Loading GT")
                                   axes[i, 0].axis("off")
                                   axes[i, 1].set_title("Error Loading Gen")
                                   axes[i, 1].axis("off")

                      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                      vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}.png")
                      plt.savefig(vis_filename)
                      print(f"Saved visualization to {vis_filename}")
                      plt.close(fig)
                 except Exception as viz_e:
                      print(f"Error during visualization creation: {viz_e}")
                      traceback.print_exc()
            else:
                 print("No valid generated/aligned images available for visualization.")
        else:
            print("Could not find valid generated images/paths populated for visualization.")

    # --- Finish ---
    end_time = time.time()
    print(f"\n--- Experiment Finished ---")
    print(f"Visual Embedding: {visual_embedding_model_name.upper()}")
    print(f"fMRI Source:      {fmri_source_name}")
    print(f"Mapping Model:    {config.MAPPING_MODEL_TYPE.upper()}")
    print(f"Total Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment")
    parser.add_argument(
        "--model_name", type=str, required=True, choices=list(config.EMBEDDING_MODELS.keys()),
        help="Name of the VISUAL EMBEDDING model to use (e.g., resnet50, vit, clip)."
    )
    # Add arguments for MLP config override if needed later
    # parser.add_argument("--mapping_model", type=str, default=config.MAPPING_MODEL_TYPE, choices=['mlp', 'ridge'], help="Type of mapping model.")
    parser.add_argument("--download", action="store_true", help="Run data download first.")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining mapping and k-NN models.")
    parser.add_argument("--visualize", action="store_true", help="Generate sample visualization.")
    # parser.add_argument("--exit_on_poor_mapping", action="store_true", help="Exit early if embedding R2 score is too low.") # Example optional arg

    args = parser.parse_args()

    # # Optional: Override config based on args if mapping_model arg is added
    # if args.mapping_model:
    #      config.MAPPING_MODEL_TYPE = args.mapping_model

    main(args)
