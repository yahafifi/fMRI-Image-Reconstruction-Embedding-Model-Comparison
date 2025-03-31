# run_experiment.py
import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import traceback # Import traceback for detailed error printing
import torch # Need torch for tensor conversion

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
    # ... (no changes needed) ...
    if args.download:
        print("\n--- Attempting Data Download ---")
        if not download_data.download_all_data():
            print("Data download/setup failed. Please check URLs and paths. Exiting.")
            return
        else:
            print("Data download/setup step completed.")
    else:
        print("\n--- Skipping Data Download ---")
        # Basic checks... (no changes needed)

    # --- 2. Load fMRI Data and Prepare Dataloaders ---
    # ... (no changes needed, dataloaders are primarily for feature extraction here) ...
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
             normalize_runs=True,
             test_split_size=config.TEST_SPLIT_SIZE,
             random_state=config.RANDOM_STATE
        )
        # Extract numpy arrays needed for mapping model training/evaluation
        X_train_np, _ = data_splits_numpy['train']
        X_val_np, _ = data_splits_numpy['val']
        X_test_avg_np, test_avg_gt_paths = data_splits_numpy['test_avg'] # Keep paths for later

        if X_train_np.size == 0: raise ValueError("Training fMRI data is empty.")
        if X_test_avg_np.size == 0: raise ValueError("Test (Avg) fMRI data is empty.")
        if not test_avg_gt_paths: raise ValueError("No test (Avg) ground truth image paths found.")

        # Create dataloaders (primarily for feature extraction)
        image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)
        dataloaders = data_loading.get_dataloaders(
            god_data_splits=data_splits_numpy, # Can still use numpy splits here
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
    try:
        embedding_model, _ = feature_extraction.load_embedding_model(visual_embedding_model_name)
        if embedding_model is None: raise ValueError("Failed to load embedding model.")

        # Extract for Training set targets
        if dataloaders.get('train'):
            # We only need the Z_train part from the feature extractor now
            _, Z_train_np = feature_extraction.extract_features(
                embedding_model, dataloaders['train'], visual_embedding_model_name, config.DEVICE
            )
            if Z_train_np.shape[0] != X_train_np.shape[0]:
                 raise ValueError(f"Shape mismatch after train feature extraction: X={X_train_np.shape}, Z={Z_train_np.shape}")
        else:
             print("Error: Train dataloader is missing.")
             return

        # Extract for Validation set targets
        if dataloaders.get('val') and X_val_np.size > 0:
             _, Z_val_np = feature_extraction.extract_features(
                 embedding_model, dataloaders['val'], visual_embedding_model_name, config.DEVICE
             )
             if Z_val_np.shape[0] != X_val_np.shape[0]:
                  raise ValueError(f"Shape mismatch after val feature extraction: X={X_val_np.shape}, Z={Z_val_np.shape}")
             print(f"Extracted Validation target embeddings: Z_val={Z_val_np.shape}")
        else:
             Z_val_np = np.array([]) # Empty array if no validation data
             print("No validation set target embeddings extracted.")


        # Extract for Averaged Test set (ground truth targets)
        if dataloaders.get('test_avg'):
             _, Z_test_avg_true_np = feature_extraction.extract_features(
                 embedding_model, dataloaders['test_avg'], visual_embedding_model_name, config.DEVICE
             )
             if Z_test_avg_true_np.shape[0] != X_test_avg_np.shape[0]:
                  raise ValueError(f"Shape mismatch after test feature extraction: X={X_test_avg_np.shape}, Z_true={Z_test_avg_true_np.shape}")
             print(f"Extracted Averaged Test target embeddings: Z_true={Z_test_avg_true_np.shape}")
        else:
             print("Error: Test (Averaged) dataloader is missing.")
             return

    except Exception as e:
        print(f"Error during GOD feature extraction: {e}")
        traceback.print_exc()
        return


    # --- 4. Train/Load Mapping Model (fMRI -> Embedding) ---
    # <<< MODIFIED SECTION >>>
    print(f"\n--- Training/Loading {config.MAPPING_MODEL_TYPE.upper()} Mapping Model ({visual_embedding_model_name}) ---")

    mapping_model = None
    model_filename = None

    if config.MAPPING_MODEL_TYPE == "mlp":
        model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{fmri_source_name}_{visual_embedding_model_name}.pt")
        if args.force_retrain or not os.path.exists(model_filename):
            print("Training new MLP model...")
            try:
                mapping_model, saved_path = mapping_models.train_mlp_mapping(
                    X_train_np, Z_train_np, X_val_np, Z_val_np,
                    visual_embedding_model_name, fmri_source_name
                )
                if mapping_model is None: raise ValueError("MLP training function failed.")
                model_filename = saved_path if saved_path else model_filename # Use returned path if available
            except Exception as e:
                print(f"Error training MLP model: {e}")
                traceback.print_exc()
                return
        else:
            print(f"Loading existing MLP model from: {model_filename}")
            try:
                mapping_model = mapping_models.load_mlp_model(model_filename)
                if mapping_model is None: raise FileNotFoundError("Failed to load MLP model.")
            except Exception as e:
                print(f"Error loading MLP model: {e}")
                traceback.print_exc()
                return # Exit if loading fails

    elif config.MAPPING_MODEL_TYPE == "ridge":
        # Keep Ridge logic as a fallback if needed
        model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{fmri_source_name}_{visual_embedding_model_name}_alpha{config.RIDGE_ALPHA}.sav")
        if args.force_retrain or not os.path.exists(model_filename):
             print("Training new Ridge model...")
             try:
                  mapping_model, saved_path = mapping_models.train_ridge_mapping(
                       X_train_np, Z_train_np, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER,
                       visual_embedding_model_name, fmri_source_name
                  )
                  if mapping_model is None: raise ValueError("Ridge training failed.")
                  model_filename = saved_path if saved_path else model_filename
             except Exception as e:
                  print(f"Error training Ridge model: {e}")
                  traceback.print_exc()
                  return
        else:
             print(f"Loading existing Ridge model from: {model_filename}")
             try:
                  mapping_model = mapping_models.load_ridge_model(model_filename)
                  if mapping_model is None: raise FileNotFoundError("Failed to load Ridge model.")
             except Exception as e:
                  print(f"Error loading Ridge model: {e}")
                  traceback.print_exc()
                  return
    else:
        print(f"Error: Unsupported MAPPING_MODEL_TYPE in config: {config.MAPPING_MODEL_TYPE}")
        return

    if mapping_model is None:
        print("Failed to obtain a mapping model. Exiting.")
        return

    # --- 5. Predict Embeddings from Test fMRI ---
    # <<< MODIFIED SECTION >>>
    print(f"\n--- Predicting Test Embeddings from fMRI ({visual_embedding_model_name} using {config.MAPPING_MODEL_TYPE.upper()}) ---")
    prediction_metrics = {} # Store prediction eval results
    Z_test_avg_pred_np = None # Initialize prediction variable
    try:
        # Use the appropriate prediction function based on model type
        if config.MAPPING_MODEL_TYPE == "mlp":
            Z_test_avg_pred_np = mapping_models.predict_embeddings_mlp(mapping_model, X_test_avg_np)
        elif config.MAPPING_MODEL_TYPE == "ridge":
            Z_test_avg_pred_np = mapping_models.predict_embeddings_ridge(mapping_model, X_test_avg_np)

        if Z_test_avg_pred_np is None: raise ValueError("Prediction failed.")

        # --- IMPORTANT: Evaluate Embedding Prediction Quality FIRST ---
        print("\n--- Evaluating EMBEDDING Prediction Performance ---")
        # Use the *raw* predictions before adjustment for primary evaluation
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_np, Z_test_avg_pred_np)
        prediction_metrics['rmse_raw'] = pred_rmse
        prediction_metrics['r2_raw'] = pred_r2
        print("----------------------------------------------------")

        # If the raw prediction metrics (esp. R2) are very poor (e.g., negative or near zero),
        # the subsequent steps (adjustment, retrieval, generation) are unlikely to work well.
        if pred_r2 is None or pred_r2 < 0.01: # Threshold can be adjusted
             print(f"WARNING: Initial embedding prediction R2 score ({pred_r2:.4f}) is very low.")
             print("         The mapping model may not be capturing the fMRI-embedding relationship well.")
             print("         Downstream results (retrieval, generation) will likely be poor.")
             # Consider adding an option to exit here if results are expected to be meaningless
             # if args.exit_on_poor_mapping: return


        # --- Apply standardization adjustment (Optional - Re-evaluate if needed for MLP) ---
        # This step might be less necessary or even counterproductive with a well-trained MLP
        # compared to Ridge. Consider making it conditional or removing it later.
        apply_adjustment = True # Set to False to disable easily
        if apply_adjustment:
            print("Applying standardization adjustment to predicted embeddings...")
            epsilon = 1e-9
            train_mean = np.mean(Z_train_np, axis=0)
            train_std = np.std(Z_train_np, axis=0)
            pred_mean = np.mean(Z_test_avg_pred_np, axis=0)
            pred_std = np.std(Z_test_avg_pred_np, axis=0)

            Z_test_avg_pred_adj = ((Z_test_avg_pred_np - pred_mean) / (pred_std + epsilon)) * train_std + train_mean
            print("Standardization complete.")

            # Evaluate adjusted predictions too
            print("Evaluating ADJUSTED embedding prediction performance (RMSE, R2):")
            adj_pred_rmse, adj_pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_np, Z_test_avg_pred_adj)
            prediction_metrics['rmse_adj'] = adj_pred_rmse
            prediction_metrics['r2_adj'] = adj_pred_r2

            # Use adjusted embeddings for retrieval if adjustment was applied
            query_embeddings = Z_test_avg_pred_adj
            print("Using *adjusted* predicted embeddings for retrieval.")
        else:
            print("Skipping standardization adjustment.")
            query_embeddings = Z_test_avg_pred_np # Use raw predictions
            print("Using *raw* predicted embeddings for retrieval.")


        # Save prediction metrics
        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"embedding_prediction_metrics_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}.csv")
        os.makedirs(config.EVALUATION_RESULTS_PATH, exist_ok=True)
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

    except Exception as e:
        print(f"Error during embedding prediction or evaluation: {e}")
        traceback.print_exc()
        return

    # --- 6. Precompute/Load ImageNet-256 Features & Train/Load k-NN ---
    # ... (No changes needed in this section, uses visual_embedding_model_name) ...
    print(f"\n--- Preparing ImageNet-256 Retrieval Database ({visual_embedding_model_name}) ---")
    knn_model = None
    try:
        db_features, db_labels, db_class_map = feature_extraction.precompute_imagenet256_features(visual_embedding_model_name)
        if db_features is None: raise ValueError("Failed to load ImageNet-256 features.")

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
    # ... (No changes needed in this section, uses query_embeddings) ...
    print(f"\n--- Retrieving Semantic Labels using k-NN ({visual_embedding_model_name}) ---")
    retrieved_readable_labels = None
    top1_prompts = []
    try:
        indices, distances, retrieved_readable_labels = retrieval.retrieve_nearest_neighbors(
            knn_model, query_embeddings, db_labels, db_class_map
        )
        if retrieved_readable_labels is None:
             print("Label retrieval failed. Proceeding without prompts.")
        else:
            top1_prompts = [labels[0] for labels in retrieved_readable_labels if labels]
            if len(top1_prompts) != len(query_embeddings):
                 print(f"Warning: Number of prompts ({len(top1_prompts)}) != queries ({len(query_embeddings)}).")
            print(f"Generated {len(top1_prompts)} top-1 prompts. Example: {top1_prompts[:5]}")
            # Save retrieval details... (no changes needed)
    except Exception as e:
        print(f"Error during label retrieval: {e}")
        traceback.print_exc()
        # Continue with empty prompts if retrieval fails but prediction worked

    # --- 8. Generate Images using Stable Diffusion ---
    # ... (No changes needed, uses top1_prompts) ...
    print(f"\n--- Generating Images using Stable Diffusion ({visual_embedding_model_name}) ---")
    generated_images_pil = []
    if not top1_prompts:
         print("No prompts available for generation.")
         eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)
    else:
        # Generation, Alignment, Saving, Evaluation... (no changes needed)
        try:
            generated_images_pil = generation.generate_images_from_prompts(top1_prompts)
            evaluation_pairs = []
            valid_indices_generated = []
            for i, gen_img in enumerate(generated_images_pil):
                 if gen_img is not None and i < len(test_avg_gt_paths):
                      evaluation_pairs.append((test_avg_gt_paths[i], gen_img))
                      valid_indices_generated.append(i)

            if not evaluation_pairs:
                 print("Image generation failed/alignment failed.")
                 eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)
            else:
                valid_gt_paths = [pair[0] for pair in evaluation_pairs]
                valid_generated_images = [pair[1] for pair in evaluation_pairs]
                print(f"Generated/aligned {len(valid_generated_images)} images.")
                generation.save_generated_images(valid_generated_images, valid_gt_paths, visual_embedding_model_name)
                print(f"\n--- Evaluating Reconstructions ({visual_embedding_model_name}) ---")
                eval_results_df = evaluation.evaluate_reconstructions(
                    valid_gt_paths, valid_generated_images, config.EVAL_METRICS
                )
                if eval_results_df is not None and 'sample_index' in eval_results_df.columns:
                    eval_results_df['original_test_index'] = valid_indices_generated

        except Exception as e:
            print(f"Error during image generation/saving/evaluation: {e}")
            traceback.print_exc()
            eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)

    # --- 11. Save Evaluation Results ---
    # ... (No changes needed, uses eval_results_df) ...
    if eval_results_df is not None:
         eval_results_filename = f"evaluation_results_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}.csv"
         evaluation.save_evaluation_results(eval_results_df, visual_embedding_model_name, filename=eval_results_filename) # Pass specific filename
    else:
         print("No evaluation results to save.")


    # --- 12. Basic Visualization (Optional) ---
    # ... (No changes needed, uses variables from step 8/10) ...
    if args.visualize and eval_results_df is not None and not eval_results_df.empty and 'ground_truth_path' in eval_results_df.columns:
        # Visualization logic... (no changes needed)
        pass # Keep existing visualization code here

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

    args = parser.parse_args()

    # # Optional: Override config based on args if mapping_model arg is added
    # if args.mapping_model:
    #      config.MAPPING_MODEL_TYPE = args.mapping_model

    main(args)
