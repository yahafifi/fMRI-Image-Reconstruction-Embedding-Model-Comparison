# run_experiment.py
import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image

# Import project modules
import config
import download_data
import data_loading
import feature_extraction
import mapping_models
import retrieval
import generation
import evaluation

def main(args):
    """Runs the full fMRI decoding experiment for a given embedding model."""
    start_time = time.time()
    model_name = args.model_name

    print(f"--- Starting Experiment for Embedding Model: {model_name.upper()} ---")

    # --- 1. Data Download (Optional) ---
    if args.download:
        if not download_data.download_all_data():
            print("Data download/setup failed. Exiting.")
            return
    else:
        # Basic check if essential data seems present
        if not os.path.exists(os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5")) or \
           not os.path.exists(config.GOD_IMAGENET_PATH):
             print("Essential fMRI or ImageNet stimuli data not found. Run with --download first.")
             # return # Or attempt to continue if user is sure

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
             normalize_runs=True, # Use normalization as per original code
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
             print("Error: No averaged test set ground truth image paths found.")
             return

    except Exception as e:
        print(f"Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. Extract GOD Image Embeddings (for mapping) ---
    # We need embeddings for train, val (optional), and test_avg ground truth images
    print(f"\n--- Extracting GOD Image Embeddings ({model_name}) ---")
    try:
        embedding_model, _ = feature_extraction.load_embedding_model(model_name)

        # Extract for Training set
        if dataloaders['train']:
            X_train, Z_train = feature_extraction.extract_features(
                embedding_model, dataloaders['train'], model_name, config.DEVICE
            )
        else:
             print("Error: Train dataloader is missing.")
             return

        # Extract for Validation set (optional, for evaluating mapping)
        Z_val = None
        if dataloaders['val']:
             X_val, Z_val = feature_extraction.extract_features(
                 embedding_model, dataloaders['val'], model_name, config.DEVICE
             )
        else:
             X_val = np.array([]) # Keep consistent type


        # Extract for Averaged Test set (ground truth embeddings)
        if dataloaders['test_avg']:
             X_test_avg, Z_test_avg_true = feature_extraction.extract_features(
                 embedding_model, dataloaders['test_avg'], model_name, config.DEVICE
             )
        else:
             print("Error: Test (Averaged) dataloader is missing.")
             return

    except Exception as e:
        print(f"Error during GOD feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Train/Load Mapping Model (fMRI -> Embedding) ---
    print(f"\n--- Training/Loading {MAPPING_MODEL_TYPE.upper()} Mapping Model ({model_name}) ---")
    
    mapping_model = None
    fmri_scaler = None # Only used by MLP
    
    if MAPPING_MODEL_TYPE == "ridge":
        ridge_model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{config.RIDGE_ALPHA}.sav")
        if args.force_retrain or not os.path.exists(ridge_model_filename):
            if X_train is None or Z_train is None: raise ValueError("Training data missing for Ridge.")
            try:
                # Assuming train_ridge_mapping now saves {'model': ..., 'scaler': None}
                # And load_ridge_model returns only the model object
                mapping_model, _ = mapping_models.train_ridge_mapping(
                    X_train, Z_train, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name
                )
            except Exception as e: print(f"Error training Ridge model: {e}"); raise # Stop if training fails
        else:
            try: mapping_model = mapping_models.load_ridge_model(ridge_model_filename)
            except Exception as e: print(f"Error loading Ridge model: {e}"); raise
    
    elif MAPPING_MODEL_TYPE == "mlp":
        mlp_model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{model_name}_best.pt")
        mlp_scaler_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_scaler_{model_name}.sav")
        if args.force_retrain or not os.path.exists(mlp_model_filename) or not os.path.exists(mlp_scaler_filename):
             if X_train is None or Z_train is None or X_val is None or Z_val is None:
                 raise ValueError("Training/Validation data missing for MLP.")
             try:
                 # Need embedding_dim which depends on the model_name
                 embedding_dim = config.EMBEDDING_MODELS[model_name]['embedding_dim']
                 mapping_model, fmri_scaler, _ = mapping_models.train_mlp_mapping(
                     X_train, Z_train, X_val, Z_val, model_name, embedding_dim
                 )
             except Exception as e: print(f"Error training MLP model: {e}"); raise
        else:
            try:
                # Need input_dim (n_voxels) and embedding_dim to load architecture
                n_voxels = X_train.shape[1] # Get from loaded data
                embedding_dim = config.EMBEDDING_MODELS[model_name]['embedding_dim']
                mapping_model, fmri_scaler = mapping_models.load_mlp_model(model_name, embedding_dim, n_voxels)
            except Exception as e: print(f"Error loading MLP model: {e}"); raise
    
    if mapping_model is None:
        print("ERROR: Failed to train or load the mapping model. Exiting.")
        return # Exit if no model is available


    # --- 5. Predict Embeddings from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings using {MAPPING_MODEL_TYPE.upper()} ({model_name}) ---")
    Z_test_avg_pred = None
    query_embeddings = None # This will hold the final embeddings used for retrieval
    prediction_metrics = {} # Store metrics like RMSE, R2, CosSim
    
    if X_test_avg is None:
         print("ERROR: Test fMRI data (X_test_avg) is missing. Cannot predict.")
    elif mapping_model is None:
         print("ERROR: Mapping model not loaded/trained. Cannot predict.")
    else:
        try:
            if MAPPING_MODEL_TYPE == "ridge":
                Z_test_avg_pred = mapping_models.predict_embeddings_ridge(mapping_model, X_test_avg)
            elif MAPPING_MODEL_TYPE == "mlp":
                # Ensure scaler was loaded/trained
                if fmri_scaler is None: raise ValueError("MLP scaler not available for prediction.")
                Z_test_avg_pred = mapping_models.predict_embeddings_mlp(mapping_model, fmri_scaler, X_test_avg)
    
            # Evaluate the raw prediction quality (Important!)
            print("\nEvaluating RAW embedding prediction performance:")
            # Use the updated evaluate_embedding_prediction function
            pred_rmse, pred_r2, pred_cos_sim = mapping_models.evaluate_embedding_prediction(Z_test_avg_true, Z_test_avg_pred)
            prediction_metrics['raw_rmse'] = pred_rmse
            prediction_metrics['raw_r2'] = pred_r2
            prediction_metrics['raw_cos_sim'] = pred_cos_sim
    
            # --- Standardization Step (Optional but Recommended) ---
            # Let's keep this adjustment step for now as it might help both models
            print("\nApplying standardization adjustment to predicted embeddings...")
            epsilon = 1e-10
            # Use Z_train stats for adjustment (Make sure Z_train exists)
            if Z_train is not None and Z_train.size > 0:
                train_mean = np.mean(Z_train, axis=0)
                train_std = np.std(Z_train, axis=0)
                pred_mean = np.mean(Z_test_avg_pred, axis=0)
                pred_std = np.std(Z_test_avg_pred, axis=0)
    
                Z_test_avg_pred_adj = ((Z_test_avg_pred - pred_mean) / (pred_std + epsilon)) * train_std + train_mean
                print("Standardization complete.")
    
                # Evaluate adjusted predictions too
                print("\nEvaluating ADJUSTED embedding prediction performance:")
                adj_pred_rmse, adj_pred_r2, adj_pred_cos_sim = mapping_models.evaluate_embedding_prediction(Z_test_avg_true, Z_test_avg_pred_adj)
                prediction_metrics['adj_rmse'] = adj_pred_rmse
                prediction_metrics['adj_r2'] = adj_pred_r2
                prediction_metrics['adj_cos_sim'] = adj_pred_cos_sim
    
                # --- Choose which predicted embeddings to use for retrieval ---
                query_embeddings = Z_test_avg_pred_adj # Use adjusted ones
                print("\nUsing *adjusted* predicted embeddings for retrieval.")
            else:
                print("WARNING: Z_train not available for standardization. Using RAW predictions for retrieval.")
                query_embeddings = Z_test_avg_pred # Fallback to raw predictions
    
        except Exception as e:
            print(f"Error during embedding prediction or evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Decide whether to stop or continue without query_embeddings
    
    # Ensure query_embeddings is defined before proceeding
    if query_embeddings is None:
        print("ERROR: Query embeddings could not be generated. Stopping.")
        return

    # --- 6. Precompute/Load Tiny ImageNet Features & Train/Load k-NN ---
    print(f"\n--- Preparing Tiny ImageNet Retrieval Database ({model_name}) ---")
    try:
        db_features, db_labels, db_class_map = feature_extraction.precompute_tiny_imagenet_features(model_name)

        knn_model_filename = os.path.join(config.SAVED_KNN_MODELS_PATH, f"knn_{model_name}_k{config.KNN_N_NEIGHBORS}.sav")
        if args.force_retrain or not os.path.exists(knn_model_filename):
            knn_model, _ = retrieval.train_knn_retrieval(db_features, config.KNN_N_NEIGHBORS, model_name)
        else:
            knn_model = retrieval.load_knn_model(knn_model_filename)

    except Exception as e:
        print(f"Error preparing retrieval database or k-NN model: {e}")
        return

    # --- 7. Retrieve Neighbor Labels from Tiny ImageNet ---
    print(f"\n--- Retrieving Semantic Labels using k-NN ({model_name}) ---")
    try:
        _, _, retrieved_readable_labels = retrieval.retrieve_nearest_neighbors(
            knn_model, query_embeddings, db_labels, db_class_map
        )
        # Example: [['cat', 'dog', ...], ['car', 'truck', ...], ...]

        # --- Select Top-1 Prompt ---
        # Using the most frequent strategy from original code
        top1_prompts = [labels[0] for labels in retrieved_readable_labels if labels] # Get first label if list is not empty
        if len(top1_prompts) != len(query_embeddings):
             print(f"Warning: Number of prompts ({len(top1_prompts)}) doesn't match queries ({len(query_embeddings)}). Check retrieval.")
             # Pad with placeholders? For now, proceed with available prompts.

        print(f"Generated {len(top1_prompts)} top-1 prompts. Example: {top1_prompts[:5]}")

    except Exception as e:
        print(f"Error during label retrieval: {e}")
        return

    # --- 8. Generate Images using Stable Diffusion ---
    print(f"\n--- Generating Images using Stable Diffusion ({model_name}) ---")
    if not top1_prompts:
         print("No prompts available for generation. Exiting.")
         return

    # Ensure number of prompts matches number of GT images expected
    num_expected = len(test_avg_gt_paths)
    if len(top1_prompts) != num_expected:
        print(f"Warning: Number of prompts ({len(top1_prompts)}) differs from expected test samples ({num_expected}). Will generate based on prompts available.")
        # Adjust GT paths list if necessary? Or pad prompts?
        # Let's generate for the prompts we have, evaluation will handle mismatch later if needed.

    try:
        generated_images_pil = generation.generate_images_from_prompts(top1_prompts) # Returns list of PIL images or None

        # Filter out None results if generation failed for some prompts
        valid_generated_images = [img for img in generated_images_pil if img is not None]
        valid_indices = [i for i, img in enumerate(generated_images_pil) if img is not None]

        if not valid_generated_images:
            print("Image generation failed for all prompts.")
            # Still proceed to save empty evaluation results? Or exit?
            # Let's create an empty eval dataframe.
            eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)

        else:
             print(f"Successfully generated {len(valid_generated_images)} images.")

             # --- 9. Save Generated Images ---
             # Important: Need to align saved images with the *original* ground truth paths
             # Use the valid_indices to select the corresponding GT paths.
             corresponding_gt_paths = [test_avg_gt_paths[i] for i in valid_indices if i < len(test_avg_gt_paths)]

             if len(valid_generated_images) != len(corresponding_gt_paths):
                  print("Warning: Mismatch between successfully generated images and corresponding GT paths after filtering.")
                  # This shouldn't happen if indices are handled correctly, but check logic.
             else:
                  generation.save_generated_images(valid_generated_images, corresponding_gt_paths, model_name)


             # --- 10. Evaluate Reconstructions ---
             print(f"\n--- Evaluating Reconstructions ({model_name}) ---")
             eval_results_df = evaluation.evaluate_reconstructions(
                  corresponding_gt_paths, valid_generated_images, config.EVAL_METRICS
             )

    except Exception as e:
        print(f"Error during image generation or saving: {e}")
        import traceback
        traceback.print_exc()
        # Create empty eval results if generation failed badly
        eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)


    # --- 11. Save Evaluation Results ---
    if eval_results_df is not None:
         evaluation.save_evaluation_results(eval_results_df, model_name)
    else:
         print("Evaluation resulted in None DataFrame. No results saved.")


    # --- 12. Basic Visualization (Optional) ---
    if args.visualize and eval_results_df is not None and not eval_results_df.empty:
        print("\n--- Visualizing Sample Results ---")
        num_to_show = min(5, len(corresponding_gt_paths)) # Show first 5 valid samples
        if num_to_show > 0:
             fig, axes = plt.subplots(num_to_show, 2, figsize=(8, num_to_show * 4))
             fig.suptitle(f'Sample Reconstructions - {model_name.upper()}', fontsize=16)

             for i in range(num_to_show):
                 gt_path_viz = corresponding_gt_paths[i]
                 gen_img_viz = valid_generated_images[i]

                 try:
                     gt_img_pil = Image.open(gt_path_viz).convert("RGB")

                     # Plot Ground Truth
                     ax_gt = axes[i, 0] if num_to_show > 1 else axes[0]
                     ax_gt.imshow(gt_img_pil)
                     ax_gt.set_title(f"Ground Truth {i}")
                     ax_gt.axis("off")

                     # Plot Generated Image
                     ax_gen = axes[i, 1] if num_to_show > 1 else axes[1]
                     ax_gen.imshow(gen_img_viz)
                     ax_gen.set_title(f"Generated (Prompt: {top1_prompts[valid_indices[i]]})") # Show prompt used
                     ax_gen.axis("off")

                 except Exception as e:
                     print(f"Error visualizing sample {i}: {e}")

             plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
             vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{model_name}.png")
             plt.savefig(vis_filename)
             print(f"Saved visualization to {vis_filename}")
             # plt.show() # Uncomment if running interactively and want to display plot

    end_time = time.time()
    print(f"\n--- Experiment for {model_name.upper()} Finished ---")
    print(f"Total Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ridge", # Default to ridge for backward compatibility
        choices=["ridge", "mlp"],
        help="Type of mapping model to use (ridge or mlp)."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run the data download and setup step first."
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining of mapping (Ridge) and retrieval (k-NN) models, even if saved files exist."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save a visualization of sample reconstructions."
    )

    args = parser.parse_args()
    MAPPING_MODEL_TYPE = args.mapping_model # Use this variable later
    main(args)

    # Example usage from command line:
    # python run_experiment.py --model_name resnet50 --download --visualize
    # python run_experiment.py --model_name vit
    # python run_experiment.py --model_name clip --force_retrain
