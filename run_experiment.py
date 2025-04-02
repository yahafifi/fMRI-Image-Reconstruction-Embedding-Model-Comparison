# run_experiment.py
import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import traceback # Import traceback for detailed error printing

# Import project modules
import config
import download_data # <<<=== ENSURE THIS IS IMPORTED
import data_loading
import feature_extraction
import mapping_models
import retrieval
import generation
import evaluation

# --- Function to check data download status ---
def check_data_exists():
    """Checks if essential data files/dirs seem to exist after download/organization."""
    print("Verifying essential data paths...")
    # Check fMRI file
    god_fmri_file = os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5")
    fmri_ok = os.path.exists(god_fmri_file)
    if not fmri_ok: print(f"- MISSING: GOD fMRI file ({god_fmri_file})")

    # Check GOD stimuli base, train/test folders, and CSVs (now expected directly in GOD_IMAGENET_PATH)
    god_stim_base_ok = os.path.isdir(config.GOD_IMAGENET_PATH)
    god_train_dir = os.path.join(config.GOD_IMAGENET_PATH, "training")
    god_test_dir = os.path.join(config.GOD_IMAGENET_PATH, "test")
    god_train_csv = os.path.join(config.GOD_IMAGENET_PATH, "image_training_id.csv")
    god_test_csv = os.path.join(config.GOD_IMAGENET_PATH, "image_test_id.csv")

    stim_train_ok = os.path.isdir(god_train_dir) and len(os.listdir(god_train_dir)) > 0
    stim_test_ok = os.path.isdir(god_test_dir) and len(os.listdir(god_test_dir)) > 0
    stim_csv_ok = os.path.exists(god_train_csv) and os.path.exists(god_test_csv)

    if not god_stim_base_ok: print(f"- MISSING: GOD Stimuli base directory ({config.GOD_IMAGENET_PATH})")
    elif not stim_train_ok: print(f"- MISSING/EMPTY: GOD Stimuli training folder ({god_train_dir})")
    elif not stim_test_ok: print(f"- MISSING/EMPTY: GOD Stimuli test folder ({god_test_dir})")
    elif not stim_csv_ok: print(f"- MISSING: GOD Stimuli CSV file(s) in ({config.GOD_IMAGENET_PATH})")

    # Check ImageNet-256 retrieval database directory
    imagenet256_dir = config.IMAGENET256_PATH
    retrieval_ok = os.path.isdir(imagenet256_dir) and len(os.listdir(imagenet256_dir)) > 0
    if not retrieval_ok: print(f"- MISSING/EMPTY: ImageNet-256 directory ({imagenet256_dir})")

    # Check other potentially downloaded files (optional, based on download script)
    # e.g., wordnet mapping
    wordnet_ok = os.path.exists(config.CLASS_TO_WORDNET_JSON)
    if not wordnet_ok: print(f"- MISSING: WordNet mapping file ({config.CLASS_TO_WORDNET_JSON})")


    all_essential_ok = fmri_ok and god_stim_base_ok and stim_train_ok and stim_test_ok and stim_csv_ok and retrieval_ok
    if all_essential_ok:
         print("All essential data paths verified.")
    else:
         print("ERROR: One or more essential data paths are missing or invalid.")
    return all_essential_ok


def main(args):
    """Runs the full fMRI decoding experiment for a given embedding model."""
    start_time = time.time()
    model_name = args.model_name
    mapping_model_type = args.mapping_model # Get mapping type from args

    print(f"--- Starting Experiment ---")
    print(f"Embedding Model: {model_name.upper()}")
    print(f"Mapping Model:   {mapping_model_type.upper()}")
    print(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"--------------------------")

    # --- 1. Data Download (Conditional) ---
    if args.download:
        print("\n--- Running Data Download and Setup ---")
        # Use the specific download_all_data function from the user's script
        if not download_data.download_all_data():
            print("\nERROR during data download/setup. Please check 'download_data.py', logs, URLs, and paths.")
            print("Exiting.")
            return # Stop execution if download fails
        else:
            print("\nData download/setup step completed (check logs for details).")
    else:
        print("\n--- Skipping Data Download Step ---")
        print("Assuming data is already present and correctly organized.")


    # --- 2. Data Check ---
    print("\n--- Checking Data Presence ---")
    if not check_data_exists():
         print("\nEssential data is missing. Please ensure datasets are correctly placed or run with --download.")
         return
    else:
         print("Proceeding with pipeline.")


    # --- 3. Load fMRI Data and Prepare Dataloaders ---
    print("\n--- Loading GOD fMRI Data ---")
    try:
        # Use the GodFmriDataHandler from the provided data_loading.py
        handler = data_loading.GodFmriDataHandler(
            subject_id=config.SUBJECT_ID,
            roi=config.ROI,
            data_dir=config.GOD_FMRI_PATH,
            image_dir=config.GOD_IMAGENET_PATH # This now points to dir with train/test/csvs
        )
        # get_data_splits performs normalization and averaging as defined in data_loading.py
        data_splits = handler.get_data_splits(
             normalize_runs=True, # Normalization happens inside get_data_splits
             test_split_size=config.TEST_SPLIT_SIZE,
             random_state=config.RANDOM_STATE
        )
        image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)

        # Use get_dataloaders from the provided data_loading.py
        dataloaders_feat_ext = data_loading.get_dataloaders(
            god_data_splits=data_splits,
            batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE,
            num_workers=config.FEATURE_EXTRACTION_NUM_WORKERS,
            image_transform=image_transform
        )

        # Get paths and numpy arrays for mapping model
        test_avg_gt_paths = data_splits['test_avg'][1]
        if not test_avg_gt_paths:
             print("Error: No averaged test set ground truth image paths found after data loading.")
             return

        X_train_fmri, _ = data_splits['train']
        X_val_fmri, _ = data_splits['val']
        X_test_avg_fmri, _ = data_splits['test_avg']

        # Basic check on shapes
        if X_train_fmri.ndim != 2 or X_val_fmri.ndim != 2 or X_test_avg_fmri.ndim != 2:
             print(f"Error: Incorrect fMRI data dimensions after loading. Check data_loading.py.")
             print(f"Shapes: Train={X_train_fmri.shape}, Val={X_val_fmri.shape}, TestAvg={X_test_avg_fmri.shape}")
             return
        if X_train_fmri.shape[0] == 0:
            print("Error: No training fMRI data loaded.")
            return

        print(f"fMRI data shapes: Train={X_train_fmri.shape}, Val={X_val_fmri.shape}, Test Avg={X_test_avg_fmri.shape}")


    except Exception as e:
        print(f"Error during data loading step: {e}")
        traceback.print_exc()
        return

    # --- 4. Extract GOD Image Embeddings (Targets for mapping) ---
    print(f"\n--- Extracting Target GOD Image Embeddings ({model_name}) ---")
    Z_train_img = Z_val_img = Z_test_avg_true_img = None
    try:
        embedding_model, _ = feature_extraction.load_embedding_model(model_name)
        if embedding_model is None: raise ValueError("Failed to load embedding model.")
        embedding_dim = config.EMBEDDING_MODELS[model_name]['embedding_dim']

        # --- Extract features using dataloaders ---
        # The FmriImageDataset yields (fmri_tensor, image_tensor)
        # feature_extraction.extract_features expects (data_batch, _) or (fmri_batch, data_batch)
        # It needs slight adaptation if FmriImageDataset returns dummy fmri

        print("Extracting Train target embeddings...")
        if dataloaders_feat_ext.get('train'):
             X_train_fmri_dl, Z_train_img = feature_extraction.extract_features(
                 embedding_model, dataloaders_feat_ext['train'], model_name, config.DEVICE
             )
             # Verify the fMRI from dataloader matches the one loaded earlier (optional sanity check)
             # if not np.allclose(X_train_fmri_dl, X_train_fmri):
             #     print("Warning: fMRI data mismatch between direct load and DataLoader iteration.")
        else:
             print("Error: Train dataloader for feature extraction is missing.")
             return

        print("Extracting Validation target embeddings...")
        if dataloaders_feat_ext.get('val') and len(X_val_fmri) > 0:
             _, Z_val_img = feature_extraction.extract_features(
                 embedding_model, dataloaders_feat_ext['val'], model_name, config.DEVICE
             )
             print(f"Extracted Validation target embeddings: Z={Z_val_img.shape}")
        else:
             Z_val_img = np.array([], dtype=np.float32).reshape(0, embedding_dim)
             print("No validation set found or loaded for feature extraction.")


        print("Extracting Averaged Test target embeddings...")
        if dataloaders_feat_ext.get('test_avg'):
             _, Z_test_avg_true_img = feature_extraction.extract_features(
                 embedding_model, dataloaders_feat_ext['test_avg'], model_name, config.DEVICE
             )
             print(f"Extracted Averaged Test target embeddings: Z_true={Z_test_avg_true_img.shape}")
        else:
             print("Error: Test (Averaged) dataloader for feature extraction is missing.")
             return

        # --- Sanity Check: Match fMRI samples with extracted embedding samples ---
        if X_train_fmri.shape[0] != Z_train_img.shape[0]:
            raise ValueError(f"Mismatch! Train fMRI samples ({X_train_fmri.shape[0]}) != Train Embeddings ({Z_train_img.shape[0]})")
        if X_val_fmri.shape[0] != Z_val_img.shape[0]:
            raise ValueError(f"Mismatch! Val fMRI samples ({X_val_fmri.shape[0]}) != Val Embeddings ({Z_val_img.shape[0]})")
        if X_test_avg_fmri.shape[0] != Z_test_avg_true_img.shape[0]:
            raise ValueError(f"Mismatch! Test Avg fMRI samples ({X_test_avg_fmri.shape[0]}) != Test Avg Embeddings ({Z_test_avg_true_img.shape[0]})")

    except Exception as e:
        print(f"Error during GOD target feature extraction: {e}")
        traceback.print_exc()
        return

    # --- 5. Train/Load Mapping Model (fMRI -> Embedding) ---
    # (Code remains the same as previous version - handles ridge/mlp selection)
    print(f"\n--- Training/Loading {mapping_model_type.upper()} Mapping Model ({model_name}) ---")
    mapping_model = None
    mapping_model_path = None
    predict_embeddings_func = None

    if mapping_model_type == 'ridge':
        mapping_model_path = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{config.RIDGE_ALPHA}.sav")
        predict_embeddings_func = mapping_models.predict_embeddings_ridge
        if args.force_retrain or not os.path.exists(mapping_model_path):
            print("Training new Ridge model...")
            try:
                mapping_model, saved_path = mapping_models.train_ridge_mapping(
                    X_train_fmri, Z_train_img, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name
                )
                if mapping_model is None: raise ValueError("Ridge training failed.")
                mapping_model_path = saved_path if saved_path else mapping_model_path
            except Exception as e:
                print(f"Error training Ridge model: {e}")
                traceback.print_exc(); return
        else:
            print(f"Loading existing Ridge model from: {mapping_model_path}")
            try:
                mapping_model = mapping_models.load_ridge_model(mapping_model_path)
                if mapping_model is None: raise FileNotFoundError("Failed to load ridge model.")
            except Exception as e:
                print(f"Error loading Ridge model: {e}")
                traceback.print_exc(); return

    elif mapping_model_type == 'mlp':
        mapping_model_path = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{model_name}_best.pt")
        predict_embeddings_func = mapping_models.predict_embeddings_mlp
        n_voxels = X_train_fmri.shape[1]

        if args.force_retrain or not os.path.exists(mapping_model_path):
            print("Training new MLP model...")
            try:
                 mapping_model, saved_path = mapping_models.train_mlp_mapping(
                      X_train_fmri, Z_train_img, X_val_fmri, Z_val_img, model_name, embedding_dim
                 )
                 if mapping_model is None: raise ValueError("MLP training failed.")
                 mapping_model_path = saved_path if saved_path else mapping_model_path
            except Exception as e:
                 print(f"Error training MLP model: {e}")
                 traceback.print_exc(); return
        else:
             print(f"Loading existing MLP model state from: {mapping_model_path}")
             try:
                  mapping_model = mapping_models.load_mlp_model(mapping_model_path, n_voxels, embedding_dim)
                  if mapping_model is None: raise FileNotFoundError("Failed to load MLP model.")
             except Exception as e:
                  print(f"Error loading MLP model: {e}")
                  traceback.print_exc(); return
    else:
        print(f"Error: Unknown mapping model type '{mapping_model_type}'")
        return


    # --- 6. Predict Embeddings from Test fMRI ---
    # (Code remains the same as previous version)
    print(f"\n--- Predicting Test Embeddings from fMRI ({mapping_model_type.upper()}) ---")
    prediction_metrics = {}
    Z_test_avg_pred_raw = None
    query_embeddings = None
    try:
        if mapping_model is None or predict_embeddings_func is None:
             raise ValueError("Mapping model or prediction function is not available.")

        Z_test_avg_pred_raw = predict_embeddings_func(mapping_model, X_test_avg_fmri)
        if Z_test_avg_pred_raw is None: raise ValueError("Prediction failed.")

        print("\nEvaluating RAW embedding prediction performance (RMSE, R2):")
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_img, Z_test_avg_pred_raw)
        prediction_metrics['rmse_raw'] = pred_rmse
        prediction_metrics['r2_raw'] = pred_r2

        print("\nApplying standardization adjustment to predicted embeddings...")
        epsilon = 1e-8
        try:
            train_mean = np.mean(Z_train_img, axis=0, keepdims=True)
            train_std = np.std(Z_train_img, axis=0, keepdims=True)
            pred_mean = np.mean(Z_test_avg_pred_raw, axis=0, keepdims=True)
            pred_std = np.std(Z_test_avg_pred_raw, axis=0, keepdims=True)

            zero_std_mask = pred_std < epsilon
            if np.any(zero_std_mask):
                print(f"Warning: {np.sum(zero_std_mask)} dimensions in prediction have near-zero std dev. Applying epsilon.")
                pred_std[zero_std_mask] = epsilon

            Z_test_avg_pred_adj = ((Z_test_avg_pred_raw - pred_mean) / pred_std) * train_std + train_mean
            print("Standardization complete.")

            print("Evaluating ADJUSTED embedding prediction performance (RMSE, R2):")
            adj_pred_rmse, adj_pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true_img, Z_test_avg_pred_adj)
            prediction_metrics['rmse_adj'] = adj_pred_rmse
            prediction_metrics['r2_adj'] = adj_pred_r2

        except Exception as std_e:
            print(f"Error during standardization: {std_e}. Skipping adjustment evaluation.")
            Z_test_avg_pred_adj = Z_test_avg_pred_raw
            prediction_metrics['rmse_adj'] = np.nan
            prediction_metrics['r2_adj'] = np.nan

        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"embedding_prediction_metrics_{mapping_model_type}_{model_name}.csv")
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

        # Decide which embeddings to use
        use_adjusted = True
        if not np.isnan(prediction_metrics['r2_adj']) and not np.isnan(prediction_metrics['r2_raw']):
             if prediction_metrics['r2_adj'] < prediction_metrics['r2_raw'] - 0.05:
                  use_adjusted = False
        elif np.isnan(prediction_metrics['r2_adj']):
             use_adjusted = False

        if use_adjusted:
            query_embeddings = Z_test_avg_pred_adj
            print("Using ADJUSTED predicted embeddings for retrieval.")
        else:
            query_embeddings = Z_test_avg_pred_raw
            print("Using RAW predicted embeddings for retrieval.")

        if np.isnan(query_embeddings).any():
            print(f"FATAL: Query embeddings contain NaNs ({np.isnan(query_embeddings).sum()} values). Cannot proceed.")
            return

    except Exception as e:
        print(f"Error during embedding prediction or evaluation: {e}")
        traceback.print_exc(); return

    # --- 7. Precompute/Load ImageNet-256 Features & Train/Load k-NN ---
    # (Code remains the same as previous version)
    print(f"\n--- Preparing ImageNet-256 Retrieval Database ({model_name}) ---")
    knn_model = None
    db_features = db_labels = db_class_map = None
    try:
        # This uses feature_extraction.py to get the retrieval DB features
        db_features, db_labels, db_class_map = feature_extraction.precompute_imagenet256_features(model_name)

        if db_features is None or db_labels is None or db_class_map is None:
             print("Failed to load or compute ImageNet-256 features. Cannot proceed with retrieval. Exiting.")
             return

        knn_model_filename = os.path.join(config.SAVED_KNN_MODELS_PATH, f"knn_{model_name}_k{config.KNN_N_NEIGHBORS}.sav")
        if args.force_retrain or not os.path.exists(knn_model_filename):
            print("Training new k-NN model...")
            knn_model, _ = retrieval.train_knn_retrieval(db_features, config.KNN_N_NEIGHBORS, model_name)
        else:
            print(f"Loading existing k-NN model from {knn_model_filename}...")
            knn_model = retrieval.load_knn_model(knn_model_filename)

        if knn_model is None:
            raise ValueError("Failed to train or load k-NN model.")

    except Exception as e:
        print(f"Error preparing retrieval database or k-NN model: {e}")
        traceback.print_exc(); return

    # --- 8. Retrieve Neighbor Labels from ImageNet-256 ---
    # (Code remains the same as previous version)
    print(f"\n--- Retrieving Semantic Labels using k-NN ({model_name}) ---")
    retrieved_readable_labels = None
    top1_prompts = []
    try:
        try:
            expected_dim = knn_model.n_features_in_
            if query_embeddings.shape[1] != expected_dim:
               print(f"FATAL Error: Query embedding dimension {query_embeddings.shape[1]} != k-NN dimension {expected_dim}")
               return
        except AttributeError:
             print("Warning: Cannot verify k-NN feature dimension compatibility.")


        indices, distances, retrieved_readable_labels = retrieval.retrieve_nearest_neighbors(
            knn_model, query_embeddings, db_labels, db_class_map
        )

        if retrieved_readable_labels is None:
             print("Label retrieval failed. Using default prompts.")
             top1_prompts = ["a blank image"] * len(query_embeddings)
        else:
            top1_prompts = []
            for i, labels in enumerate(retrieved_readable_labels):
                if labels and isinstance(labels[0], str) and labels[0]:
                    prompt = labels[0].replace("_", " ").strip()
                    if not prompt:
                        prompt = "object"
                        print(f"Warning: Empty label retrieved for sample {i}. Using '{prompt}'.")
                    # Enhanced prompt
                    prompt = f"high quality photograph of a {prompt}"
                    top1_prompts.append(prompt)
                else:
                    top1_prompts.append("a photograph of an object")
                    print(f"Warning: Invalid or missing label retrieved for sample {i}. Using default prompt.")


            if len(top1_prompts) != len(query_embeddings):
                print(f"Critical Warning: Number of prompts ({len(top1_prompts)}) doesn't match queries ({len(query_embeddings)}). Padding/truncating.")
                if len(top1_prompts) < len(query_embeddings):
                     top1_prompts.extend(["a photograph of an object"] * (len(query_embeddings) - len(top1_prompts)))
                else:
                     top1_prompts = top1_prompts[:len(query_embeddings)]

            print(f"Generated {len(top1_prompts)} top-1 prompts. Example: {top1_prompts[:5]}")

            try:
                retrieval_info = {
                    'query_index': list(range(len(query_embeddings))),
                    'retrieved_indices': indices.tolist() if indices is not None else [None]*len(query_embeddings),
                    'retrieved_distances': distances.tolist() if distances is not None else [None]*len(query_embeddings),
                    'retrieved_labels_all_k': retrieved_readable_labels,
                    'top1_prompt_generated': top1_prompts
                }
                retrieval_df = pd.DataFrame(retrieval_info)
                retrieval_output_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"retrieval_details_{mapping_model_type}_{model_name}.csv")
                retrieval_df.to_csv(retrieval_output_file, index=False)
                print(f"Saved retrieval details to {retrieval_output_file}")
            except Exception as save_e:
                print(f"Warning: Could not save retrieval details: {save_e}")

    except Exception as e:
        print(f"Error during label retrieval: {e}")
        traceback.print_exc()
        top1_prompts = ["a photograph of an object"] * len(query_embeddings)
        print("Using default prompts due to retrieval error.")

    # --- 9. Generate Images using Stable Diffusion ---
    # (Code remains the same as previous version)
    print(f"\n--- Generating Images using Stable Diffusion ({model_name}) ---")
    generated_images_pil = []
    eval_results_df = None

    num_expected_gt = len(test_avg_gt_paths)
    if len(top1_prompts) != num_expected_gt:
        print(f"Warning: Final number of prompts ({len(top1_prompts)}) differs from expected averaged test samples ({num_expected_gt}). Adjusting generation count.")
        if len(top1_prompts) < num_expected_gt:
             top1_prompts.extend(["a photograph of an object"] * (num_expected_gt - len(top1_prompts)))
        else:
             top1_prompts = top1_prompts[:num_expected_gt]

    try:
        generated_images_pil = generation.generate_images_from_prompts(
            top1_prompts,
            guidance_scale=config.STABLE_DIFFUSION_GUIDANCE_SCALE,
            num_inference_steps=config.NUM_INFERENCE_STEPS
        )

        evaluation_pairs = []
        valid_indices_generated = []
        for i, gen_img in enumerate(generated_images_pil):
             if gen_img is not None and i < len(test_avg_gt_paths):
                  evaluation_pairs.append((test_avg_gt_paths[i], gen_img))
                  valid_indices_generated.append(i)

        if not evaluation_pairs:
             print("Image generation failed for all prompts or alignment failed.")
             eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index', 'original_test_index'] + config.EVAL_METRICS)
        else:
            valid_gt_paths = [pair[0] for pair in evaluation_pairs]
            valid_generated_images = [pair[1] for pair in evaluation_pairs]
            print(f"Successfully generated and aligned {len(valid_generated_images)} images with ground truths.")

            # --- 10. Save Generated Images ---
            generation.save_generated_images(valid_generated_images, valid_gt_paths, f"{mapping_model_type}_{model_name}")

            # --- 11. Evaluate Reconstructions ---
            print(f"\n--- Evaluating Reconstructions ({model_name}) ---")
            eval_results_df = evaluation.evaluate_reconstructions(
                valid_gt_paths, valid_generated_images, config.EVAL_METRICS
            )
            if eval_results_df is not None and not eval_results_df.empty:
                 original_indices_map = {i: orig_idx for i, orig_idx in enumerate(valid_indices_generated)}
                 eval_results_df['original_test_index'] = eval_results_df['sample_index'].map(original_indices_map)
                 print("Added 'original_test_index' to evaluation results.")
            else:
                 print("Evaluation failed or resulted in empty dataframe.")
                 eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index', 'original_test_index'] + config.EVAL_METRICS)

    except Exception as e:
        print(f"Error during image generation, saving, or evaluation: {e}")
        traceback.print_exc()
        eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index', 'original_test_index'] + config.EVAL_METRICS)


    # --- 12. Save Evaluation Results ---
    if eval_results_df is not None:
         evaluation.save_evaluation_results(eval_results_df, f"{mapping_model_type}_{model_name}")
    else:
         print("Evaluation resulted in None DataFrame or generation failed. No final results saved.")


    # --- 13. Basic Visualization (Optional) ---
    if args.visualize and eval_results_df is not None and not eval_results_df.empty and 'original_test_index' in eval_results_df.columns:
        print("\n--- Visualizing Sample Results ---")
        if 'valid_gt_paths' in locals() and 'valid_generated_images' in locals():
            num_to_show = min(10, len(valid_gt_paths))
            if num_to_show > 0:
                 try:
                      fig, axes = plt.subplots(num_to_show, 2, figsize=(8, num_to_show * 4))
                      if num_to_show == 1: axes = np.array([axes])
                      fig.suptitle(f'Sample Reconstructions - {mapping_model_type.upper()} + {model_name.upper()}', fontsize=16)

                      for i in range(num_to_show):
                          gt_path_viz = valid_gt_paths[i]
                          gen_img_viz = valid_generated_images[i]
                          original_index = valid_indices_generated[i]
                          prompt_viz = top1_prompts[original_index] if original_index < len(top1_prompts) else "Prompt N/A"

                          try:
                              gt_img_pil = Image.open(gt_path_viz).convert("RGB")
                              axes[i, 0].imshow(gt_img_pil)
                              axes[i, 0].set_title(f"GT Idx: {original_index}")
                              axes[i, 0].axis("off")

                              axes[i, 1].imshow(gen_img_viz)
                              axes[i, 1].set_title(f"Gen (Prompt: {prompt_viz[:30]}...)")
                              axes[i, 1].axis("off")
                          except Exception as plot_e:
                              print(f"Error plotting sample {i} (Original Index: {original_index}): {plot_e}")
                              if i < len(axes):
                                   axes[i, 0].set_title(f"GT {original_index} Error")
                                   axes[i, 0].axis("off")
                                   axes[i, 1].set_title(f"Gen {original_index} Error")
                                   axes[i, 1].axis("off")

                      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                      vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{mapping_model_type}_{model_name}.png")
                      plt.savefig(vis_filename)
                      print(f"Saved visualization to {vis_filename}")
                      plt.close(fig)
                 except Exception as viz_e:
                      print(f"Error during visualization creation: {viz_e}")
                      traceback.print_exc()
            else:
                 print("No valid generated images available for visualization.")
        else:
            print("Could not find valid generated images/paths for visualization.")

    end_time = time.time()
    print(f"\n--- Experiment Finished ---")
    print(f"Mapping Model:   {mapping_model_type.upper()}")
    print(f"Embedding Model: {model_name.upper()}")
    print(f"Total Time:      {(end_time - start_time) / 60:.2f} minutes")
    print(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(config.EMBEDDING_MODELS.keys()),
        help="Name of the visual embedding model (feature extractor) to use."
    )
    parser.add_argument(
        "--mapping_model",
        type=str,
        default=config.MAPPING_MODEL_TYPE, # Default from config
        choices=['ridge', 'mlp'],
        help="Type of mapping model (fMRI -> embedding) to use."
    )
    # --- Argument reinstated ---
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run the data download and setup step first using download_data.py."
    )
    # -------------------------
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining of mapping and retrieval models, even if saved files exist."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save a visualization of sample reconstructions."
    )

    args = parser.parse_args()

    # Override config mapping type if provided via command line
    config.MAPPING_MODEL_TYPE = args.mapping_model

    main(args)

    # Example usage from command line:
    # python run_experiment.py --model_name resnet50 --mapping_model mlp --download --visualize
    # python run_experiment.py --model_name vit --mapping_model mlp --force_retrain
    # python run_experiment.py --model_name clip --mapping_model ridge --download