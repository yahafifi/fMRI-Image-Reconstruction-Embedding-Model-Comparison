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
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 1. Data Download (Optional) ---
    if args.download:
        print("\n--- Attempting Data Download ---")
        if not download_data.download_all_data():
            print("Data download/setup failed. Please check URLs and paths. Exiting.")
            # It's safer to exit if download fails, as subsequent steps depend on it.
            return
        else:
            print("Data download/setup step completed.")
    else:
        # Basic check if essential data seems present
        print("\n--- Skipping Data Download ---")
        god_fmri_file = os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5")
        god_train_dir = os.path.join(config.GOD_IMAGENET_PATH, "training")
        imagenet256_dir = config.IMAGENET256_PATH # Check the base retrieval dir

        if not os.path.exists(god_fmri_file):
             print(f"Warning: GOD fMRI file not found at {god_fmri_file}. Check path or run with --download.")
             # Decide whether to proceed or exit
             # return
        if not os.path.exists(god_train_dir):
             print(f"Warning: GOD stimuli 'training' directory not found at {god_train_dir}. Check path or run with --download.")
             # return
        if not os.path.exists(imagenet256_dir):
             print(f"Warning: ImageNet-256 directory not found at {imagenet256_dir}. Check path/dataset name in config.")
             # return


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
             print("Error: No averaged test set ground truth image paths found after data loading.")
             return

    except Exception as e:
        print(f"Error during data loading: {e}")
        traceback.print_exc()
        return

    # --- 3. Extract GOD Image Embeddings (for mapping) ---
    # We need embeddings for train, val (optional), and test_avg ground truth images
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
        # Z_val = None # Initialize
        if dataloaders.get('val'):
             X_val, Z_val = feature_extraction.extract_features(
                 embedding_model, dataloaders['val'], model_name, config.DEVICE
             )
             print(f"Extracted Validation features: X={X_val.shape}, Z={Z_val.shape}")
        else:
             X_val = np.array([]) # Keep consistent type if no val set
             Z_val = np.array([])
             print("No validation set found or loaded.")


        # Extract for Averaged Test set (ground truth embeddings)
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
    # !! Placeholder for adding MLP later !! Current: Ridge Regression
    print(f"\n--- Training/Loading Ridge Mapping Model ({model_name}) ---")
    ridge_model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{config.RIDGE_ALPHA}.sav")

    ridge_model = None
    if args.force_retrain or not os.path.exists(ridge_model_filename):
        print("Training new Ridge model...")
        try:
            # Ensure X_train and Z_train have compatible shapes
            if X_train.shape[0] != Z_train.shape[0]:
                 raise ValueError(f"Training fMRI samples ({X_train.shape[0]}) != Training embedding samples ({Z_train.shape[0]})")

            ridge_model, saved_path = mapping_models.train_ridge_mapping(
                X_train, Z_train, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name
            )
            if ridge_model is None: raise ValueError("Ridge training failed.")
            ridge_model_filename = saved_path # Update filename in case it differs slightly

        except Exception as e:
            print(f"Error training Ridge model: {e}")
            traceback.print_exc()
            return
    else:
        print(f"Loading existing Ridge model from: {ridge_model_filename}")
        try:
            ridge_model = mapping_models.load_ridge_model(ridge_model_filename)
            if ridge_model is None: raise FileNotFoundError("Failed to load ridge model.")
        except Exception as e:
            print(f"Error loading Ridge model: {e}")
            traceback.print_exc()
            # Option: Fallback to training?
            # print("Attempting to train Ridge model instead...")
            # try:
            #     ridge_model, ridge_model_filename = mapping_models.train_ridge_mapping(...)
            # except: return
            return

    # --- 5. Predict Embeddings from Test fMRI ---
    print(f"\n--- Predicting Test Embeddings from fMRI ({model_name}) ---")
    prediction_metrics = {} # Store prediction eval results
    try:
        Z_test_avg_pred = mapping_models.predict_embeddings(ridge_model, X_test_avg)
        if Z_test_avg_pred is None: raise ValueError("Prediction failed.")

        # Evaluate the raw embedding prediction quality
        print("Evaluating RAW embedding prediction performance (RMSE, R2):")
        pred_rmse, pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true, Z_test_avg_pred)
        prediction_metrics['rmse_raw'] = pred_rmse
        prediction_metrics['r2_raw'] = pred_r2

        # --- Apply standardization adjustment (optional but kept from original logic) ---
        print("Applying standardization adjustment to predicted embeddings...")
        epsilon = 1e-9 # Slightly increased epsilon for stability
        train_mean = np.mean(Z_train, axis=0)
        train_std = np.std(Z_train, axis=0)
        pred_mean = np.mean(Z_test_avg_pred, axis=0)
        pred_std = np.std(Z_test_avg_pred, axis=0)

        # Add epsilon to denominator std dev to prevent division by zero
        Z_test_avg_pred_adj = ((Z_test_avg_pred - pred_mean) / (pred_std + epsilon)) * train_std + train_mean
        print("Standardization complete.")

        # Evaluate adjusted predictions too
        print("Evaluating ADJUSTED embedding prediction performance (RMSE, R2):")
        adj_pred_rmse, adj_pred_r2 = mapping_models.evaluate_prediction(Z_test_avg_true, Z_test_avg_pred_adj)
        prediction_metrics['rmse_adj'] = adj_pred_rmse
        prediction_metrics['r2_adj'] = adj_pred_r2

        # Save prediction metrics
        pred_metrics_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"embedding_prediction_metrics_{model_name}.csv")
        pd.DataFrame([prediction_metrics]).to_csv(pred_metrics_file, index=False)
        print(f"Saved embedding prediction metrics to {pred_metrics_file}")

        # --- Choose which predicted embeddings to use for retrieval ---
        # Using the adjusted ones based on original code's likely intent & potential benefit with Ridge
        query_embeddings = Z_test_avg_pred
        print("Using *adjusted* predicted embeddings for retrieval.")

    except Exception as e:
        print(f"Error during embedding prediction or evaluation: {e}")
        traceback.print_exc()
        return

    # --- 6. Precompute/Load ImageNet-256 Features & Train/Load k-NN ---
    print(f"\n--- Preparing ImageNet-256 Retrieval Database ({model_name}) ---")
    knn_model = None
    try:
        db_features, db_labels, db_class_map = feature_extraction.precompute_imagenet256_features(model_name)

        # Check if feature extraction was successful
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
        traceback.print_exc()
        return

    # --- 7. Retrieve Neighbor Labels from ImageNet-256 ---
    print(f"\n--- Retrieving Semantic Labels using k-NN ({model_name}) ---")
    retrieved_readable_labels = None
    try:
        # Ensure query embeddings match expected feature dimensions for k-NN
        # expected_dim = knn_model.n_features_in_ # Check attribute if available
        # if query_embeddings.shape[1] != expected_dim:
        #    print(f"Error: Query embedding dimension {query_embeddings.shape[1]} != k-NN dimension {expected_dim}")
        #    return

        indices, distances, retrieved_readable_labels = retrieval.retrieve_nearest_neighbors(
            knn_model, query_embeddings, db_labels, db_class_map
        )

        if retrieved_readable_labels is None:
             print("Label retrieval failed. Check logs.")
             # Decide whether to proceed with empty prompts or exit
             top1_prompts = [] # Create empty list to avoid generation error
             # return # Safer to exit if retrieval fails
        else:
            # --- Select Top-1 Prompt ---
            top1_prompts = [labels[0] for labels in retrieved_readable_labels if labels] # Get first label if list is not empty
            if len(top1_prompts) != len(query_embeddings):
                print(f"Warning: Number of prompts ({len(top1_prompts)}) doesn't match queries ({len(query_embeddings)}). Some retrieval might have failed.")
                # How to handle? Pad prompts? For now, proceed with generated prompts.
                # Make sure evaluation handles potential length mismatch.

            print(f"Generated {len(top1_prompts)} top-1 prompts. Example: {top1_prompts[:5]}")

            # --- Save retrieval info ---
            try:
                retrieval_info = {
                    'query_index': list(range(len(query_embeddings))),
                    'retrieved_indices': indices.tolist(),
                    'retrieved_distances': distances.tolist(),
                    'retrieved_labels': retrieved_readable_labels,
                    'top1_prompt': top1_prompts + [None]*(len(query_embeddings)-len(top1_prompts)) # Pad if needed
                }
                retrieval_df = pd.DataFrame(retrieval_info)
                retrieval_output_file = os.path.join(config.EVALUATION_RESULTS_PATH, f"retrieval_details_{model_name}.csv")
                retrieval_df.to_csv(retrieval_output_file, index=False)
                print(f"Saved retrieval details to {retrieval_output_file}")
            except Exception as save_e:
                print(f"Warning: Could not save retrieval details: {save_e}")


    except Exception as e:
        print(f"Error during label retrieval: {e}")
        traceback.print_exc()
        return

    # --- 8. Generate Images using Stable Diffusion ---
    print(f"\n--- Generating Images using Stable Diffusion ({model_name}) ---")
    generated_images_pil = [] # Initialize
    if not top1_prompts:
         print("No prompts available for generation. Skipping generation and evaluation.")
         eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS) # Create empty frame
    else:
        # Ensure number of prompts matches number of GT images expected
        num_expected_gt = len(test_avg_gt_paths)
        if len(top1_prompts) != num_expected_gt:
            print(f"Warning: Number of prompts ({len(top1_prompts)}) differs from expected averaged test samples ({num_expected_gt}). Generation/Evaluation might be misaligned.")
            # Strategy: Generate based on available prompts, but evaluate only against corresponding GTs.

        try:
            generated_images_pil = generation.generate_images_from_prompts(top1_prompts) # Returns list of PIL images or None

            # --- Align generated images with ground truth paths ---
            # Create pairs of (gt_path, gen_img) only for successful generations
            evaluation_pairs = []
            valid_indices_generated = []
            for i, gen_img in enumerate(generated_images_pil):
                 if gen_img is not None and i < len(test_avg_gt_paths): # Check both generation success and GT path availability
                      evaluation_pairs.append((test_avg_gt_paths[i], gen_img))
                      valid_indices_generated.append(i) # Keep track of original index

            if not evaluation_pairs:
                 print("Image generation failed for all prompts or alignment failed.")
                 eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)
            else:
                valid_gt_paths = [pair[0] for pair in evaluation_pairs]
                valid_generated_images = [pair[1] for pair in evaluation_pairs]
                print(f"Successfully generated and aligned {len(valid_generated_images)} images with ground truths.")

                # --- 9. Save Generated Images ---
                generation.save_generated_images(valid_generated_images, valid_gt_paths, model_name)

                # --- 10. Evaluate Reconstructions ---
                print(f"\n--- Evaluating Reconstructions ({model_name}) ---")
                eval_results_df = evaluation.evaluate_reconstructions(
                    valid_gt_paths, valid_generated_images, config.EVAL_METRICS
                )
                # Add original index back for clarity if needed
                if eval_results_df is not None and 'sample_index' in eval_results_df.columns:
                    eval_results_df['original_test_index'] = valid_indices_generated
                    print("Added 'original_test_index' to evaluation results.")

        except Exception as e:
            print(f"Error during image generation, saving, or evaluation: {e}")
            traceback.print_exc()
            # Create empty eval results if generation failed badly
            eval_results_df = pd.DataFrame(columns=['ground_truth_path', 'sample_index'] + config.EVAL_METRICS)


    # --- 11. Save Evaluation Results ---
    if eval_results_df is not None:
         evaluation.save_evaluation_results(eval_results_df, model_name)
    else:
         print("Evaluation resulted in None DataFrame or generation failed. No final results saved.")


    # --- 12. Basic Visualization (Optional) ---
    if args.visualize and eval_results_df is not None and not eval_results_df.empty and 'ground_truth_path' in eval_results_df.columns:
        print("\n--- Visualizing Sample Results ---")
        # Use the filtered lists from step 8/10 for visualization
        if 'valid_gt_paths' in locals() and 'valid_generated_images' in locals():
            num_to_show = min(5, len(valid_gt_paths)) # Show first few valid samples
            if num_to_show > 0:
                 try:
                      fig, axes = plt.subplots(num_to_show, 2, figsize=(8, num_to_show * 4))
                      if num_to_show == 1: axes = np.array([axes]) # Ensure axes is iterable even for 1 sample
                      fig.suptitle(f'Sample Reconstructions - {model_name.upper()}', fontsize=16)

                      for i in range(num_to_show):
                          gt_path_viz = valid_gt_paths[i]
                          gen_img_viz = valid_generated_images[i]
                          # Get corresponding prompt used for this image
                          original_index = valid_indices_generated[i] # Get index used for prompt list
                          prompt_viz = top1_prompts[original_index] if original_index < len(top1_prompts) else "N/A"

                          try:
                              gt_img_pil = Image.open(gt_path_viz).convert("RGB")

                              # Plot Ground Truth
                              axes[i, 0].imshow(gt_img_pil)
                              axes[i, 0].set_title(f"Ground Truth {original_index}") # Show original index
                              axes[i, 0].axis("off")

                              # Plot Generated Image
                              axes[i, 1].imshow(gen_img_viz)
                              axes[i, 1].set_title(f"Generated (Prompt: {prompt_viz})") # Show prompt used
                              axes[i, 1].axis("off")

                          except Exception as plot_e:
                              print(f"Error plotting sample {i} (Original Index: {original_index}): {plot_e}")
                              if i < len(axes): # Check if axes exist for this index
                                   axes[i, 0].set_title("Error Loading GT")
                                   axes[i, 0].axis("off")
                                   axes[i, 1].set_title("Error Loading Gen")
                                   axes[i, 1].axis("off")


                      plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
                      vis_filename = os.path.join(config.EVALUATION_RESULTS_PATH, f"visualization_{model_name}.png")
                      plt.savefig(vis_filename)
                      print(f"Saved visualization to {vis_filename}")
                      plt.close(fig) # Close the figure to free memory
                      # plt.show() # Uncomment if running interactively and want to display plot
                 except Exception as viz_e:
                      print(f"Error during visualization creation: {viz_e}")
                      traceback.print_exc()
            else:
                 print("No valid generated images available for visualization.")
        else:
            print("Could not find valid generated images/paths for visualization.")

    end_time = time.time()
    print(f"\n--- Experiment for {model_name.upper()} Finished ---")
    print(f"Total Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")
    print(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI Decoding Experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(config.EMBEDDING_MODELS.keys()),
        help="Name of the visual embedding model to use."
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
    main(args)

    # Example usage from command line:
    # python run_experiment.py --model_name resnet50 --download --visualize
    # python run_experiment.py --model_name vit --force_retrain
    # python run_experiment.py --model_name clip
