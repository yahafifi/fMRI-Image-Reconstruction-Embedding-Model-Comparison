# run_experiment.py (Corrected for Text2Img Only)
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from PIL import Image
import traceback

# Import project modules
try:
    import config
    import download_data
    import data_loading
    import feature_extraction
    import mapping_models
    import retrieval
    import generation
    import evaluation
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure all .py files (config.py, data_loading.py, etc.) are in the same directory or Python path.")
    sys.exit(1)

def run_main_logic(args):
    """Runs the fMRI decoding experiment (Text2Img generation only)."""
    start_time = time.time()
    # --- Get parameters from args ---
    model_name = args.model_name             # e.g., 'resnet50', 'vit', 'clip'
    mapping_model_type = args.mapping_model  # 'ridge' or 'mlp'
    # generation_mode is removed for now, assume text2img

    print(f"--- Starting Experiment ---")
    print(f"Visual Embedding Model : {model_name.upper()}")
    print(f"Mapping Model          : {mapping_model_type.upper()}")
    print(f"Generation Mode        : Text2Img (Default)") # Indicate mode
    print(f"Force Retrain          : {args.force_retrain}")
    print(f"Attempt Download       : {args.download}")
    print(f"Visualize              : {args.visualize}")
    print(f"-------------------------")

    # --- 1. Data Download (Optional) ---
    if args.download:
        print("\n--- Attempting Data Download ---")
        if not download_data.download_all_data():
            print("WARNING: Data download/setup failed or had errors. Check paths and URLs. Continuing...")
        else:
            print("Data download/setup completed.")
    else:
        print("\n--- Skipping Data Download ---")
        # Basic data checks
        if not os.path.exists(config.GOD_FMRI_FILE_PATH): print(f"WARNING: fMRI file not found: {config.GOD_FMRI_FILE_PATH}")
        if not os.path.isdir(config.GOD_IMAGENET_PATH) or not os.path.isdir(os.path.join(config.GOD_IMAGENET_PATH, 'training')): print(f"WARNING: GOD image directory not found: {config.GOD_IMAGENET_PATH}")
        if not os.path.isdir(os.path.join(config.TINY_IMAGENET_PATH, 'train')): print(f"WARNING: Tiny ImageNet train directory not found: {config.TINY_IMAGENET_PATH}")

    # --- 2. Load fMRI Data and Prepare Dataloaders ---
    print("\n--- Loading GOD fMRI Data ---")
    X_train, Z_train, X_val, Z_val, X_test_avg, Z_test_avg_true = None, None, None, None, None, None
    test_avg_gt_paths = []
    dataloaders = {}
    n_voxels = -1

    try:
        handler = data_loading.GodFmriDataHandler(
            subject_id=config.SUBJECT_ID, roi=config.ROI,
            fmri_file=config.GOD_FMRI_FILE_PATH, image_dir=config.GOD_IMAGENET_PATH)
        n_voxels = handler.n_voxels
        data_splits = handler.get_data_splits(normalize_runs=True, test_split_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
        image_transform = data_loading.get_default_image_transform(config.TARGET_IMAGE_SIZE)
        dataloaders = data_loading.get_dataloaders(
            god_data_splits=data_splits, batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS, image_transform=image_transform)
        test_avg_gt_paths = data_splits.get('test_avg', (None, []))[1]
        if not test_avg_gt_paths: print("WARNING: No averaged test set ground truth image paths found.")
    except FileNotFoundError as e: print(f"ERROR: Missing critical file: {e}"); return
    except Exception as e: print(f"ERROR during data loading: {e}"); traceback.print_exc(); return

    # --- 3. Extract GOD Image Embeddings ---
    print(f"\n--- Extracting GOD Image Embeddings ({model_name.upper()}) ---")
    try:
        if not dataloaders or 'train' not in dataloaders or dataloaders['train'] is None: raise ValueError("Train dataloader missing.")
        embedding_model, _ = feature_extraction.load_embedding_model(model_name)
        X_train, Z_train = feature_extraction.extract_features(embedding_model, dataloaders['train'], model_name, config.DEVICE)
        if dataloaders.get('val'): X_val, Z_val = feature_extraction.extract_features(embedding_model, dataloaders['val'], model_name, config.DEVICE)
        else:
             X_val = np.array([]).reshape(0, X_train.shape[1]) if X_train is not None else np.array([])
             Z_val = np.array([]).reshape(0, Z_train.shape[1]) if Z_train is not None else np.array([])
             print("No validation data found or loaded.")
        if dataloaders.get('test_avg'): X_test_avg, Z_test_avg_true = feature_extraction.extract_features(embedding_model, dataloaders['test_avg'], model_name, config.DEVICE)
        else: print("WARNING: Test (Averaged) dataloader missing.")
    except Exception as e: print(f"ERROR during GOD feature extraction: {e}"); traceback.print_exc(); return

    # --- 4. Train/Load Mapping Model ---
    print(f"\n--- Training/Loading {mapping_model_type.upper()} Mapping Model ({model_name.upper()}) ---")
    mapping_model = None; fmri_scaler = None
    try:
        if mapping_model_type == "ridge":
            ridge_model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{config.RIDGE_ALPHA}.sav")
            if args.force_retrain or not os.path.exists(ridge_model_filename):
                if X_train is None or Z_train is None: raise ValueError("Training data missing for Ridge.")
                mapping_model, _ = mapping_models.train_ridge_mapping(X_train, Z_train, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name)
            else: mapping_model = mapping_models.load_ridge_model(ridge_model_filename)
        elif mapping_model_type == "mlp":
            mlp_model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{model_name}_best.pt")
            mlp_scaler_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_scaler_{model_name}.sav")
            can_train_mlp = (X_train is not None and Z_train is not None)
            has_val_data = (X_val is not None and Z_val is not None and X_val.size > 0)
            if args.force_retrain or not os.path.exists(mlp_model_filename) or not os.path.exists(mlp_scaler_filename):
                 if not can_train_mlp: raise ValueError("Training data missing for MLP.")
                 if not has_val_data: print("WARNING: Validation data missing for MLP. Training without validation-based early stopping.")
                 embedding_dim = config.EMBEDDING_MODELS[model_name]['embedding_dim']
                 mapping_model, fmri_scaler, _ = mapping_models.train_mlp_mapping(X_train, Z_train, X_val, Z_val, model_name, embedding_dim)
            else:
                if n_voxels < 0: raise ValueError("Number of voxels not determined.")
                embedding_dim = config.EMBEDDING_MODELS[model_name]['embedding_dim']
                mapping_model, fmri_scaler = mapping_models.load_mlp_model(model_name, embedding_dim, n_voxels)
    except Exception as e: print(f"ERROR during mapping model training/loading: {e}"); traceback.print_exc(); return
    if mapping_model is None: print("ERROR: Failed to train or load mapping model."); return

    # --- 5. Predict Embeddings from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings using {mapping_model_type.upper()} ({model_name.upper()}) ---")
    Z_test_avg_pred = None; query_embeddings = None; prediction_metrics = {}
    if X_test_avg is None: print("WARNING: Test fMRI data missing. Cannot predict.")
    else:
        try:
            if mapping_model_type == "ridge": Z_test_avg_pred = mapping_models.predict_embeddings_ridge(mapping_model, X_test_avg)
            elif mapping_model_type == "mlp":
                if fmri_scaler is None: raise ValueError("MLP scaler not available.")
                Z_test_avg_pred = mapping_models.predict_embeddings_mlp(mapping_model, fmri_scaler, X_test_avg)

            if Z_test_avg_true is not None: # Evaluate raw if possible
                print("\nEvaluating RAW embedding prediction performance:")
                pred_rmse, pred_r2, pred_cos_sim = mapping_models.evaluate_embedding_prediction(Z_test_avg_true, Z_test_avg_pred)
                prediction_metrics['raw_rmse']=pred_rmse; prediction_metrics['raw_r2']=pred_r2; prediction_metrics['raw_cos_sim']=pred_cos_sim
            else: print("Skipping RAW embedding evaluation (missing true test embeddings).")

            print("\nApplying standardization adjustment...") # Apply adjustment
            epsilon = 1e-10
            if Z_train is not None and Z_train.size > 0:
                train_mean=np.mean(Z_train, axis=0); train_std=np.std(Z_train, axis=0)
                pred_mean=np.mean(Z_test_avg_pred, axis=0); pred_std=np.std(Z_test_avg_pred, axis=0)
                Z_test_avg_pred_adj = ((Z_test_avg_pred - pred_mean) / (pred_std + epsilon)) * train_std + train_mean
                print("Standardization complete.")
                if Z_test_avg_true is not None: # Evaluate adjusted if possible
                    print("\nEvaluating ADJUSTED embedding prediction performance:")
                    adj_rmse, adj_r2, adj_cos_sim = mapping_models.evaluate_embedding_prediction(Z_test_avg_true, Z_test_avg_pred_adj)
                    prediction_metrics['adj_rmse']=adj_rmse; prediction_metrics['adj_r2']=adj_r2; prediction_metrics['adj_cos_sim']=adj_cos_sim
                else: print("Skipping ADJUSTED embedding evaluation (missing true test embeddings).")
                query_embeddings = Z_test_avg_pred_adj
                print("\nUsing *adjusted* predicted embeddings for retrieval.")
            else: print("WARNING: Z_train missing. Using RAW predictions for retrieval."); query_embeddings = Z_test_avg_pred
        except Exception as e: print(f"ERROR during embedding prediction: {e}"); traceback.print_exc()
    if query_embeddings is None: print("ERROR: Query embeddings not generated."); return

    # --- 6. Prepare Tiny ImageNet DB & k-NN ---
    print(f"\n--- Preparing Tiny ImageNet Retrieval Database ({model_name.upper()}) ---")
    knn_model = None; db_features, db_labels, db_class_map = None, None, None
    # NOTE: db_image_paths are NOT needed for text2img, so we ignore the 4th return value
    try:
        db_features, db_labels, db_class_map, _ = feature_extraction.precompute_tiny_imagenet_features(model_name)
        if db_features is not None and db_labels is not None and db_class_map is not None:
            print(f"Loaded Tiny ImageNet DB: {db_features.shape[0]} samples.")
            knn_model_filename = os.path.join(config.SAVED_KNN_MODELS_PATH, f"knn_{model_name}_k{config.KNN_N_NEIGHBORS}.sav")
            if args.force_retrain or not os.path.exists(knn_model_filename):
                knn_model, _ = retrieval.train_knn_retrieval(db_features, config.KNN_N_NEIGHBORS, model_name)
            else: knn_model = retrieval.load_knn_model(knn_model_filename)
        else: print("ERROR: Failed to load Tiny ImageNet DB."); return
    except Exception as e: print(f"ERROR preparing retrieval DB: {e}"); traceback.print_exc(); return

    # --- 7. Retrieve Neighbor Labels ---
    print(f"\n--- Retrieving Semantic Labels using k-NN ({model_name.upper()}) ---")
    top1_prompts = []
    if knn_model is None: print("ERROR: k-NN model missing.")
    else:
        try:
            # NOTE: Assuming retrieve_nearest_neighbors DOES NOT require db_image_paths for text prompts
            # And it returns readable_labels as the 3rd element. We ignore the path return (4th) if it exists.
            # Adjust the call based on your final retrieval.py function signature
            results = retrieval.retrieve_nearest_neighbors(knn_model, query_embeddings, db_labels, db_class_map) # Call without paths
            # Unpack carefully based on what retrieve_nearest_neighbors returns
            if len(results) == 3: # Expects indices, distances, readable_labels
                _, _, retrieved_readable_labels = results
            elif len(results) == 4: # If it still returns paths, ignore the last one
                 _, _, retrieved_readable_labels, _ = results
            else:
                 raise ValueError(f"Unexpected number of return values ({len(results)}) from retrieve_nearest_neighbors")

            top1_prompts = [labels[0] for labels in retrieved_readable_labels if labels]
            print(f"Generated {len(top1_prompts)} top-1 prompts. Example: {top1_prompts[:5]}")
            if len(top1_prompts) != len(query_embeddings): print(f"WARNING: Prompt count mismatch.")
        except Exception as e: print(f"ERROR during label retrieval: {e}"); traceback.print_exc(); return

    # --- 8. Generate Images (Text2Img only) ---
    print(f"\n--- Generating Images using SD (Text2Img) ---")
    generated_images_pil = []; valid_generated_images = []; corresponding_gt_paths = []; valid_indices = []
    if not top1_prompts: print("No prompts available. Skipping generation.")
    else:
        num_expected_gt = len(test_avg_gt_paths)
        num_prompts_avail = len(top1_prompts)
        num_to_generate = min(num_prompts_avail, num_expected_gt)
        print(f"Attempting to generate {num_to_generate} images.")
        prompts_to_use = top1_prompts[:num_to_generate]
        gt_paths_to_use = test_avg_gt_paths[:num_to_generate]
        try:
            generated_images_pil = generation.generate_images_from_prompts(prompts_to_use) # Standard text2img function
            valid_indices = [i for i, img in enumerate(generated_images_pil) if img is not None]
            valid_generated_images = [generated_images_pil[i] for i in valid_indices]
            corresponding_gt_paths = [gt_paths_to_use[i] for i in valid_indices] # Align GT paths
            if not valid_generated_images: print("Image generation failed for all attempts.")
            else: print(f"Successfully generated {len(valid_generated_images)} images.")
        except Exception as e: print(f"ERROR during image generation: {e}"); traceback.print_exc()

    # --- 9. Save Generated Images ---
    if valid_generated_images and corresponding_gt_paths:
        print(f"\n--- Saving Generated Images ({model_name.upper()}) ---")
        generation.save_generated_images(valid_generated_images, corresponding_gt_paths, model_name)
    elif top1_prompts: print("\n--- Skipping Saving (No valid images generated) ---")

    # --- 10. Evaluate Reconstructions ---
    print(f"\n--- Evaluating Reconstructions ({model_name.upper()}) ---")
    eval_results_df = pd.DataFrame()
    if valid_generated_images and corresponding_gt_paths:
         try: eval_results_df = evaluation.evaluate_reconstructions(corresponding_gt_paths, valid_generated_images, config.EVAL_METRICS)
         except Exception as e: print(f"ERROR during evaluation: {e}"); traceback.print_exc()
    else: print("Skipping evaluation.")

    # --- 11. Save Evaluation Results ---
    if not eval_results_df.empty:
         evaluation.save_evaluation_results(eval_results_df, f"{model_name}_{mapping_model_type}") # Add mapping type to filename
         pred_metrics_path = os.path.join(config.EVALUATION_RESULTS_PATH, f"prediction_metrics_{model_name}_{mapping_model_type}.csv")
         try: pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_path, index=False); print(f"Saved prediction metrics to: {pred_metrics_path}")
         except Exception as e: print(f"Could not save prediction metrics: {e}")
    else: print("No evaluation results to save.")

    # --- 12. Basic Visualization ---
    if args.visualize and not eval_results_df.empty:
        print("\n--- Attempting Visualization ---")
        # ...(Visualization code remains largely the same as previous version)...
        # ...(Ensure it uses valid_generated_images, corresponding_gt_paths, valid_indices, top1_prompts)...
        # ...(Maybe add mapping_model_type to the title and filename)...
        try:
            num_samples_generated = len(valid_generated_images)
            num_to_show = min(5, num_samples_generated)
            print(f"Visualizing first {num_to_show} generated pairs.")
            if num_to_show > 0:
                fig, axes = plt.subplots(num_to_show, 2, figsize=(10, num_to_show * 5))
                if num_to_show == 1: axes = np.array([axes])
                fig.suptitle(f'Sample Reconstructions - {model_name.upper()} ({mapping_model_type.upper()})', fontsize=16, y=1.02)
                for i in range(num_to_show):
                    gt_path_viz = corresponding_gt_paths[i]
                    gen_img_viz = valid_generated_images[i]
                    original_query_index = valid_indices[i]
                    prompt_viz = top1_prompts[original_query_index] if original_query_index < len(top1_prompts) else "Prompt N/A"
                    ax_gt = axes[i, 0]
                    try: gt_img_pil = Image.open(gt_path_viz).convert("RGB"); ax_gt.imshow(gt_img_pil); ax_gt.set_title(f"Ground Truth {original_query_index}"); ax_gt.axis("off")
                    except Exception as e_gt: print(f"Viz Error GT {i}: {e_gt}"); ax_gt.set_title("GT Error"); ax_gt.axis("off")
                    ax_gen = axes[i, 1]
                    if gen_img_viz:
                         ax_gen.imshow(gen_img_viz)
                         wrapped_prompt = '\n'.join(prompt_viz[j:j+30] for j in range(0, len(prompt_viz), 30))
                         ax_gen.set_title(f"Generated (Prompt: {wrapped_prompt})"); ax_gen.axis("off")
                    else: ax_gen.set_title("Gen Error"); ax_gen.axis("off")
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{model_name}_{mapping_model_type}.png") # Added mapping type
                plt.savefig(vis_filename, bbox_inches='tight'); plt.show(); plt.close(fig)
                print(f"Saved visualization to {vis_filename}")
            else: print("No valid generated images available.")
        except Exception as e_plot: print(f"ERROR during visualization: {e_plot}"); traceback.print_exc()
    elif args.visualize: print("Visualization skipped (No evaluation results).")

    # --- End ---
    end_time = time.time()
    print(f"\n--- Experiment Finished ---")
    print(f"Visual Embedding Model : {model_name.upper()}")
    print(f"Mapping Model          : {mapping_model_type.upper()}")
    print(f"Total Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")

# ===============================================
# Main execution block
# ===============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment")
    parser.add_argument( "--model_name", type=str, required=True, choices=list(config.EMBEDDING_MODELS.keys()), help="Name of the visual embedding model.")
    parser.add_argument( "--mapping_model", type=str, default="ridge", choices=["ridge", "mlp"], help="Type of mapping model.")
    # REMOVED --generation_mode for now
    parser.add_argument( "--download", action="store_true", help="Run data download step.")
    parser.add_argument( "--force_retrain", action="store_true", help="Force retraining of models.")
    parser.add_argument( "--visualize", action="store_true", help="Generate visualization.")
    args = parser.parse_args()
    run_main_logic(args)
