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
from sklearn.decomposition import PCA # <<< Import PCA
import joblib # For saving PCA model

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
    print(f"Use PCA Target:   {config.USE_PCA_TARGET} (Components: {config.PCA_N_COMPONENTS if config.USE_PCA_TARGET else 'N/A'})")
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
    X_train_np, X_val_np, X_test_avg_np = None, None, None
    test_avg_gt_paths = None
    dataloaders = None
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
        if X_train_np is None or X_train_np.size == 0: raise ValueError("Training fMRI data (X_train_np) is empty after splitting.")
        if X_test_avg_np is None or X_test_avg_np.size == 0: raise ValueError("Test (Avg) fMRI data (X_test_avg_np) is empty after splitting.")
        if not test_avg_gt_paths: raise ValueError("No test (Avg) ground truth image paths found after splitting.")
        # Handle case where validation split might be 0
        if X_val_np is None or X_val_np.size == 0:
            print("Note: Validation set size is 0 based on test_split_size or data loading. MLP will train without validation-based early stopping/best model selection.")
            X_val_np = np.array([]) # Ensure it's an empty array for consistency

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


    # --- 3. Extract GOD Image Embeddings (Original Full Dimension) ---
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
            if Z_train_np is None or Z_train_np.shape[0] != X_train_np.shape[0]:
                 raise ValueError(f"Shape mismatch or error after train feature extraction: X={X_train_np.shape}, Z={Z_train_np.shape if Z_train_np is not None else 'None'}")
            print(f"Extracted Training target embeddings: Z_train={Z_train_np.shape}")
        else:
             print("Error: Train dataloader missing, cannot extract training embeddings.")
             return

        # Extract for Validation set targets
        if dataloaders.get('val') and X_val_np.size > 0 : # Check if val data exists
             _, Z_val_np = feature_extraction.extract_features(
                 embedding_model, dataloaders['val'], visual_embedding_model_name, config.DEVICE
             )
             if Z_val_np is None or Z_val_np.shape[0] != X_val_np.shape[0]:
                  raise ValueError(f"Shape mismatch or error after val feature extraction: X={X_val_np.shape}, Z={Z_val_np.shape if Z_val_np is not None else 'None'}")
             print(f"Extracted Validation target embeddings: Z_val={Z_val_np.shape}")
        else:
             Z_val_np = np.array([]) # Ensure it's an empty array if no validation data
             print("No validation set target embeddings extracted (or validation set is empty).")


        # Extract for Averaged Test set (ground truth targets)
        if dataloaders.get('test_avg'):
             _, Z_test_avg_true_np = feature_extraction.extract_features(
                 embedding_model, dataloaders['test_avg'], visual_embedding_model_name, config.DEVICE
             )
             if Z_test_avg_true_np is None or Z_test_avg_true_np.shape[0] != X_test_avg_np.shape[0]:
                  raise ValueError(f"Shape mismatch or error after test feature extraction: X={X_test_avg_np.shape}, Z_true={Z_test_avg_true_np.shape if Z_test_avg_true_np is not None else 'None'}")
             print(f"Extracted Averaged Test target embeddings: Z_true={Z_test_avg_true_np.shape}")
        else:
             print("Error: Test (Averaged) dataloader missing, cannot extract test embeddings.")
             return

    except Exception as e:
        print(f"Error during GOD target feature extraction: {e}")
        traceback.print_exc()
        return


    # --- 3.5 Apply PCA Transformation (if enabled in config) --- <<< NEW STEP >>>
    pca_model = None
    pca_output_dim = Z_train_np.shape[1] # Default to original dimension
    Z_train_target = Z_train_np         # Default target
    Z_val_target = Z_val_np             # Default target
    Z_test_avg_true_target = Z_test_avg_true_np # Default target

    if config.USE_PCA_TARGET:
        print(f"\n--- Applying PCA Transformation (n_components={config.PCA_N_COMPONENTS}) ---")
        try:
            # Ensure n_components is valid
            if config.PCA_N_COMPONENTS <= 0 or config.PCA_N_COMPONENTS > Z_train_np.shape[1]:
                raise ValueError(f"Invalid PCA_N_COMPONENTS ({config.PCA_N_COMPONENTS}). Must be > 0 and <= {Z_train_np.shape[1]}")

            print(f"Fitting PCA on training embeddings (Z_train_np shape: {Z_train_np.shape})...")
            pca_model = PCA(n_components=config.PCA_N_COMPONENTS, random_state=config.RANDOM_STATE, svd_solver='full') # Use 'full' for reliability
            pca_model.fit(Z_train_np) # Fit only on training data

            # Transform training, validation, and test target embeddings
            Z_train_transformed = pca_model.transform(Z_train_np)
            print(f"Transformed Z_train shape: {Z_train_transformed.shape}")
            explained_variance = np.sum(pca_model.explained_variance_ratio_)
            print(f"Explained variance ratio by {config.PCA_N_COMPONENTS} components: {explained_variance:.4f}")

            Z_val_transformed = None
            if Z_val_np is not None and Z_val_np.size > 0:
                Z_val_transformed = pca_model.transform(Z_val_np)
                print(f"Transformed Z_val shape: {Z_val_transformed.shape}")

            Z_test_avg_true_transformed = pca_model.transform(Z_test_avg_true_np)
            print(f"Transformed Z_test_avg_true shape: {Z_test_avg_true_transformed.shape}")

            # --- Update target variables for mapping model ---
            Z_train_target = Z_train_transformed
            Z_val_target = Z_val_transformed if Z_val_transformed is not None else np.array([]) # Use transformed or empty
            Z_test_avg_true_target = Z_test_avg_true_transformed
            pca_output_dim = config.PCA_N_COMPONENTS # Update output dim for MLP

            # Save the fitted PCA model
            pca_model_filename = os.path.join(config.MODELS_BASE_PATH, f"pca_model_{visual_embedding_model_name}_{config.PCA_N_COMPONENTS}c.joblib")
            os.makedirs(config.MODELS_BASE_PATH, exist_ok=True)
            joblib.dump(pca_model, pca_model_filename)
            print(f"Saved PCA model to {pca_model_filename}")

        except Exception as e:
            print(f"Error during PCA transformation: {e}")
            traceback.print_exc()
            return # Stop if PCA fails
    else:
        print("\n--- Skipping PCA Transformation ---")
        # Targets remain the original full-dimensional embeddings
        pca_output_dim = Z_train_np.shape[1] # Output dim is original


    # --- 4. Train/Load Mapping Model (fMRI -> Embedding/PCA) ---
    print(f"\n--- Training/Loading {config.MAPPING_MODEL_TYPE.upper()} Mapping Model ({visual_embedding_model_name}) ---")
    print(f"    Target dimension: {pca_output_dim}") # Show target dimension

    mapping_model = None
    model_filename = None # Use specific filename based on PCA/NoPCA

    try:
        # Determine model filename suffix based on PCA usage
        pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""

        if config.MAPPING_MODEL_TYPE == "mlp":
            # Adjust filename based on whether PCA is used
            model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{fmri_source_name}_{visual_embedding_model_name}{pca_suffix}.pt")

            if args.force_retrain or not os.path.exists(model_filename):
                print(f"Training new MLP model (Target Dim: {pca_output_dim})...")
                # Pass the potentially PCA-transformed targets
                mapping_model, saved_path = mapping_models.train_mlp_mapping(
                    X_train_np, Z_train_target, X_val_np, Z_val_target,
                    visual_embedding_model_name, fmri_source_name # Names are for logging/saving only
                )
                if mapping_model is None: raise ValueError("MLP training function failed.")
                # Update filename ONLY if saving worked and returned a path
                model_filename = saved_path if saved_path else model_filename
            else:
                print(f"Loading existing MLP model from: {model_filename}")
                mapping_model = mapping_models.load_mlp_model(model_filename)
                # --- Sanity check loaded model output dim ---
                if mapping_model:
                     # Safely access the last layer, assuming it's Linear
                     last_layer = list(mapping_model.network.children())[-1]
                     if isinstance(last_layer, nn.Linear):
                          loaded_output_dim = last_layer.out_features
                          if loaded_output_dim != pca_output_dim:
                               print(f"ERROR: Loaded MLP output dimension ({loaded_output_dim}) does not match expected target dimension ({pca_output_dim}) based on config.USE_PCA_TARGET!")
                               print("       Delete the saved model file or ensure config matches the saved model.")
                               return
                          print(f"   Loaded model output dimension ({loaded_output_dim}) matches expected target dimension.")
                     else:
                           print("Warning: Could not verify output dimension of loaded model's last layer.")
                else:
                     raise FileNotFoundError(f"Failed to load MLP model from {model_filename}.")

        elif config.MAPPING_MODEL_TYPE == "ridge":
             model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{fmri_source_name}_{visual_embedding_model_name}{pca_suffix}_alpha{config.RIDGE_ALPHA}.sav")
             if args.force_retrain or not os.path.exists(model_filename):
                 print(f"Training new Ridge model (Target Dim: {pca_output_dim})...")
                 mapping_model, saved_path = mapping_models.train_ridge_mapping(
                      X_train_np, Z_train_target, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER,
                      visual_embedding_model_name, fmri_source_name
                 )
                 if mapping_model is None: raise ValueError("Ridge training function returned None.")
                 model_filename = saved_path if saved_path else model_filename
             else:
                 print(f"Loading existing Ridge model from: {model_filename}")
                 mapping_model = mapping_models.load_ridge_model(model_filename)
                 if mapping_model is None: raise FileNotFoundError(f"Failed to load Ridge model from {model_filename}.")
             # Note: Cannot easily verify Ridge output dim like MLP

        else:
            raise ValueError(f"Unsupported MAPPING_MODEL_TYPE: {config.MAPPING_MODEL_TYPE}")

    except Exception as e:
        print(f"Error during mapping model training/loading: {e}")
        traceback.print_exc()
        return

    if mapping_model is None:
        print("Failed to obtain a mapping model. Exiting.")
        return


    # --- 5. Predict Embeddings/PCA from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings/PCA from fMRI ({visual_embedding_model_name} using {config.MAPPING_MODEL_TYPE.upper()}) ---")
    prediction_metrics = {}
    Z_test_avg_pred_target = None # This will hold predicted embeddings OR PCA components
    query_embeddings = None # This will hold the embeddings needed for retrieval (might need inverse PCA)

    try:
        # Predict using the trained model (outputs embeddings or PCA components)
        if config.MAPPING_MODEL_TYPE == "mlp":
            Z_test_avg_pred_target = mapping_models.predict_embeddings_mlp(mapping_model, X_test_avg_np)
        elif config.MAPPING_MODEL_TYPE == "ridge":
            Z_test_avg_pred_target = mapping_models.predict_embeddings_ridge(mapping_model, X_test_avg_np)

        if Z_test_avg_pred_target is None: raise ValueError("Prediction function returned None.")
        # Handle case where prediction might return empty array
        if Z_test_avg_pred_target.size == 0 and X_test_avg_np.size != 0:
            raise ValueError("Prediction resulted in an empty array for non-empty input.")
        elif Z_test_avg_pred_target.size == 0 and X_test_avg_np.size == 0:
             print("Note: Test input was empty, prediction is also empty.")
             # Allow proceeding, but subsequent steps likely won't do much

        print(f"Shape of predicted targets: {Z_test_avg_pred_target.shape}")

        # --- Evaluate Prediction Quality (Compare predicted PCA/Embeddings vs true PCA/Embeddings) ---
        target_type_str = "PCA Components" if config.USE_PCA_TARGET else "Embeddings"
        print(f"\n--- Evaluating {target_type_str} Prediction Performance (RAW) ---")
        # Compare the direct output of the model (Z_test_avg_pred_target) with the corresponding ground truth (Z_test_avg_true_target)
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_target, Z_test_avg_pred_target, dataset_name="Test (Raw Target)")
        prediction_metrics['rmse_raw_target'] = pred_rmse
        prediction_metrics['r2_raw_target'] = pred_r2
        print("--------------------------------------------------------------------------")

        # Check R2 score for the direct prediction
        if pred_r2 is None or np.isnan(pred_r2) or pred_r2 < 0.01:
             print(f"WARNING: R2 score for predicting {target_type_str} ({pred_r2:.4f}) is very low or NaN.")
             print("         The mapping model may not be effectively learning the relationship.")
             # Consider exiting if the direct prediction is poor
             # return

        # --- Prepare Query Embeddings for Retrieval ---
        if config.USE_PCA_TARGET:
            # If we predicted PCA components, we need to inverse transform them
            # back to the original embedding space for retrieval/generation guidance.
            print("Applying inverse PCA transform to get query embeddings...")
            if pca_model is None:
                 # Try loading PCA model if not available from training step
                 pca_model_filename = os.path.join(config.MODELS_BASE_PATH, f"pca_model_{visual_embedding_model_name}_{config.PCA_N_COMPONENTS}c.joblib")
                 if os.path.exists(pca_model_filename):
                      print(f"Loading PCA model from {pca_model_filename}")
                      pca_model = joblib.load(pca_model_filename)
                 else:
                      # Try to refit PCA model if file not found? Risky as it might differ. Better to error.
                      raise ValueError("PCA was used but the fitted PCA model is not available for inverse transform. Delete saved MLP/Ridge and PCA models and rerun.")

            # Apply inverse transform only if predictions are not empty
            if Z_test_avg_pred_target.size > 0:
                 query_embeddings = pca_model.inverse_transform(Z_test_avg_pred_target)
                 print(f"Shape of inverse-transformed query embeddings: {query_embeddings.shape}")
                 # OPTIONAL: Evaluate the quality of the inverse-transformed embeddings vs original true embeddings
                 print("Evaluating INVERSE-TRANSFORMED embedding prediction performance:")
                 inv_rmse, inv_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_np, query_embeddings, dataset_name="Test (Inverse PCA)")
                 prediction_metrics['rmse_inverse_pca'] = inv_rmse
                 prediction_metrics['r2_inverse_pca'] = inv_r2
            else:
                 print("Predicted PCA components are empty, cannot inverse transform.")
                 query_embeddings = np.array([]) # Empty query embeddings


        else:
            # If we predicted embeddings directly, no inverse transform needed
            query_embeddings = Z_test_avg_pred_target # Use raw predictions directly
            print("Using raw predicted embeddings for retrieval (PCA not used).")


        # --- Standardization Adjustment (Now applied to query_embeddings) ---
        # Keep disabled for now, re-evaluate usefulness later
        apply_adjustment = False # <<< KEEP DISABLED >>>
        if apply_adjustment and query_embeddings is not None and query_embeddings.size > 0:
             # ... (Adjustment code would go here, operating on 'query_embeddings') ...
            pass
        else:
            print("Skipping standardization adjustment on query embeddings.")
            # query_embeddings remain as they are (raw predicted embeddings or inverse PCA)


        # Save prediction metrics
        pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""
        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"embedding_prediction_metrics_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}{pca_suffix}.csv")
        os.makedirs(config.EVALUATION_RESULTS_PATH, exist_ok=True) # Ensure dir exists
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

    except Exception as e:
        print(f"Error during embedding prediction/evaluation/PCA steps: {e}")
        traceback.print_exc()
        return # Exit if prediction fails

    # Check if query_embeddings were successfully generated (and are not empty if input wasn't empty)
    if query_embeddings is None or (X_test_avg_np.size > 0 and query_embeddings.size == 0):
        print("Error: Query embeddings for retrieval were not generated successfully. Exiting.")
        return


    # --- Step 6 onwards (Retrieval, Generation, Evaluation) ---
    # These steps now operate on the final `query_embeddings`

    # --- 6. Precompute/Load ImageNet-256 Features & Train/Load k-NN ---
    print(f"\n--- Preparing ImageNet-256 Retrieval Database ({visual_embedding_model_name}) ---")
    knn_model = None
    db_features, db_labels, db_class_map = None, None, None # Initialize
    try:
        db_features, db_labels, db_class_map = feature_extraction.precompute_imagenet256_features(visual_embedding_model_name)
        if db_features is None or db_labels is None or db_class_map is None:
             raise ValueError("Failed to load or compute ImageNet-256 features/labels/map.")

        # Determine k-NN filename based on PCA setting of the *query* embeddings used for retrieval
        # (Filename doesn't strictly depend on PCA, but maybe log helps?)
        # Keep standard filename for k-NN model itself
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
    print(f"\n--- Retrieving Semantic Labels using k-NN ({visual_embedding_model_name}) ---")
    retrieved_readable_labels = None
    top1_prompts = []
    indices, distances = None, None # Initialize
    
    # Proceed only if query embeddings exist
    if query_embeddings.size == 0:
         print("Query embeddings are empty, skipping retrieval.")
    else:
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
                    # Ensure indices/distances have the correct expected shape/type before converting
                    indices_list = indices.tolist() if isinstance(indices, np.ndarray) else ([None]*len(query_embeddings) if indices is None else indices)
                    distances_list = distances.tolist() if isinstance(distances, np.ndarray) else ([None]*len(query_embeddings) if distances is None else distances)
                    labels_list = retrieved_readable_labels if retrieved_readable_labels is not None else [None]*len(query_embeddings)
                    prompts_list = top1_prompts + ([None]*(len(query_embeddings)-len(top1_prompts)) if len(top1_prompts) < len(query_embeddings) else []) # Pad if needed more safely

                    # Ensure all lists have the same length matching query_embeddings
                    num_queries = len(query_embeddings)
                    retrieval_info = {
                        'query_index': list(range(num_queries)),
                        'retrieved_indices': indices_list + ([None]*(num_queries-len(indices_list))) if len(indices_list) < num_queries else indices_list[:num_queries],
                        'retrieved_distances': distances_list + ([None]*(num_queries-len(distances_list))) if len(distances_list) < num_queries else distances_list[:num_queries],
                        'retrieved_labels': labels_list + ([None]*(num_queries-len(labels_list))) if len(labels_list) < num_queries else labels_list[:num_queries],
                        'top1_prompt': prompts_list + ([None]*(num_queries-len(prompts_list))) if len(prompts_list) < num_queries else prompts_list[:num_queries]
                    }

                    retrieval_df = pd.DataFrame(retrieval_info)
                    pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""
                    retrieval_output_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"retrieval_details_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}{pca_suffix}.csv")
                    retrieval_df.to_csv(retrieval_output_file, index=False)
                    print(f"Saved retrieval details to {retrieval_output_file}")
                except Exception as save_e:
                    print(f"Warning: Could not save retrieval details: {save_e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Error during label retrieval: {e}")
            traceback.print_exc()
            # Continue with empty prompts if retrieval fails


    # --- 8. Generate Images using Stable Diffusion ---
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
                          # Try to get the prompt for the message, handle index error if prompts list was shorter
                          prompt_msg = top1_prompts[i] if i < len(top1_prompts) else "N/A"
                          print(f"Note: Generation failed for prompt index {i} ('{prompt_msg}'). Skipping.")
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
                         print(f"Warning: Length mismatch between valid indices ({len(valid_indices_generated)}) and evaluation results ({len(eval_results_df)}). Cannot add 'original_test_index'.")


        except Exception as e:
            print(f"Error during image generation, saving, or evaluation: {e}")
            traceback.print_exc()
            # Create empty eval results if generation failed badly
            eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)

    # --- 11. Save Evaluation Results ---
    if eval_results_df is not None:
         pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""
         eval_results_filename = f"evaluation_results_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}{pca_suffix}.csv"
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
                      fig.suptitle(f'Sample Reconstructions - {visual_embedding_model_name.upper()} ({config.MAPPING_MODEL_TYPE.upper()}{" PCA" if config.USE_PCA_TARGET else ""})', fontsize=16)

                      for i in range(num_to_show):
                          gt_path_viz = valid_gt_paths[i]
                          gen_img_viz = valid_generated_images[i]
                          # Safely get original index and prompt
                          original_index = valid_indices_generated[i] if i < len(valid_indices_generated) else -1
                          prompt_viz = top1_prompts[original_index] if original_index != -1 and original_index < len(top1_prompts) else "N/A"

                          try:
                              gt_img_pil = Image.open(gt_path_viz).convert("RGB")
                              axes[i, 0].imshow(gt_img_pil)
                              axes[i, 0].set_title(f"Ground Truth {original_index if original_index != -1 else 'Idx?'}")
                              axes[i, 0].axis("off")
                              axes[i, 1].imshow(gen_img_viz)
                              axes[i, 1].set_title(f"Generated (Prompt: {prompt_viz})")
                              axes[i, 1].axis("off")
                          except Exception as plot_e:
                              print(f"Error plotting sample {i} (Original Index: {original_index}): {plot_e}")
                              if i < len(axes): # Check if axes exist for this index
                                   axes[i, 0].set_title("Error Loading GT")
                                   axes[i, 0].axis("off")
                                   axes[i, 1].set_title("Error Loading Gen")
                                   axes[i, 1].axis("off")

                      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                      pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""
                      vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{fmri_source_name}_{visual_embedding_model_name}_{config.MAPPING_MODEL_TYPE}{pca_suffix}.png")
                      plt.savefig(vis_filename)
                      print(f"Saved visualization to {vis_filename}")
                      plt.close(fig) # Close the figure to free memory
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
    print(f"PCA Target used:  {config.USE_PCA_TARGET} (Components: {config.PCA_N_COMPONENTS if config.USE_PCA_TARGET else 'N/A'})")
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
