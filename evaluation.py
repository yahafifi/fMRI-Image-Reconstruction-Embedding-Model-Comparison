# evaluation.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips # Needs installation: pip install lpips
import open_clip # Needs installation: pip install open_clip_torch torch torchvision
from PIL import Image
import os
import pandas as pd
import tqdm

import config

# --- Metric Calculation Functions ---

def compute_ssim(img1_pil, img2_pil):
    """Computes SSIM between two PIL images."""
    try:
        # Convert to grayscale numpy arrays
        img1_gray = np.array(img1_pil.convert("L"))
        img2_gray = np.array(img2_pil.convert("L").resize(img1_pil.size)) # Resize to match

        # Compute SSIM
        # data_range can be important, assumes 8-bit images (0-255)
        score = ssim(img1_gray, img2_gray, data_range=img1_gray.max() - img1_gray.min())
        return score
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return np.nan # Return NaN on error

# --- LPIPS ---
_lpips_loss_fn = None

def get_lpips_fn(net_type='alex', device=config.DEVICE):
     """Initializes and returns the LPIPS loss function."""
     global _lpips_loss_fn
     if _lpips_loss_fn is None:
          print(f"Loading LPIPS model (net={net_type})...")
          _lpips_loss_fn = lpips.LPIPS(net=net_type).to(device).eval()
          print("LPIPS model loaded.")
     return _lpips_loss_fn

def compute_lpips(img1_pil, img2_pil, loss_fn):
     """Computes LPIPS distance between two PIL images."""
     try:
          # Convert PIL images to tensors expected by LPIPS (-1 to 1 range)
          img1_tensor = lpips.im2tensor(lpips.load_image(img1_pil)).to(loss_fn.device) # Use loss_fn's device
          img2_tensor = lpips.im2tensor(lpips.load_image(img2_pil)).to(loss_fn.device)

          with torch.no_grad():
               distance = loss_fn(img1_tensor, img2_tensor).item()
          return distance
     except Exception as e:
          print(f"Error calculating LPIPS: {e}")
          return np.nan


# --- CLIP Similarity ---
_clip_model = None
_clip_preprocess = None

def get_clip_score_model(model_name=config.CLIP_SCORE_MODEL_NAME, pretrained=config.CLIP_SCORE_PRETRAINED, device=config.DEVICE):
    """Initializes and returns the CLIP model and preprocessor for scoring."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        print(f"Loading CLIP model for scoring ({model_name} / {pretrained})...")
        try:
             _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                  model_name, pretrained=pretrained, device=device
             )
             _clip_model.eval()
             print("CLIP scoring model loaded.")
        except Exception as e:
             print(f"Error loading CLIP scoring model: {e}")
             _clip_model = None
             _clip_preprocess = None

    return _clip_model, _clip_preprocess

def compute_clip_similarity(img1_pil, img2_pil, model, preprocess):
    """Computes CLIP-based image similarity."""
    if model is None or preprocess is None:
         print("CLIP scoring model not available.")
         return np.nan
    try:
        # Preprocess images and move to model's device
        img1_tensor = preprocess(img1_pil).unsqueeze(0).to(model.device)
        img2_tensor = preprocess(img2_pil).unsqueeze(0).to(model.device)

        with torch.no_grad():
            # Encode images
            img1_feat = model.encode_image(img1_tensor).float()
            img2_feat = model.encode_image(img2_tensor).float()

            # Normalize features
            img1_feat /= img1_feat.norm(dim=-1, keepdim=True)
            img2_feat /= img2_feat.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = (img1_feat @ img2_feat.T).item() # Dot product for normalized vectors

        # Scale similarity to [0, 1] range (optional, CLIP score is often -1 to 1 or scaled)
        # score = (similarity + 1) / 2
        return similarity # Return raw cosine similarity

    except Exception as e:
        print(f"Error calculating CLIP similarity: {e}")
        return np.nan


# --- Main Evaluation Function ---
def evaluate_reconstructions(ground_truth_paths, generated_images, metrics=config.EVAL_METRICS):
    """
    Evaluates generated images against ground truth using specified metrics.

    Args:
        ground_truth_paths (list): List of paths to ground truth images.
        generated_images (list): List of generated PIL Images.
        metrics (list): List of metric names ('ssim', 'lpips', 'clip_sim').

    Returns:
        pandas.DataFrame: DataFrame containing scores for each image pair and metric.
    """
    if len(ground_truth_paths) != len(generated_images):
        print(f"Error: Mismatch in number of ground truth paths ({len(ground_truth_paths)}) and generated images ({len(generated_images)}).")
        return None

    results = []
    lpips_fn = None
    clip_scoring_model, clip_scoring_preprocess = None, None

    # Pre-load models if needed
    if 'lpips' in metrics:
        lpips_fn = get_lpips_fn()
    if 'clip_sim' in metrics:
        clip_scoring_model, clip_scoring_preprocess = get_clip_score_model()

    print(f"Evaluating {len(generated_images)} image pairs using metrics: {metrics}...")
    for i, (gt_path, gen_img) in enumerate(tqdm.tqdm(zip(ground_truth_paths, generated_images), total=len(generated_images), desc="Evaluating")):
        scores = {'ground_truth_path': gt_path, 'sample_index': i}

        if gen_img is None:
            print(f"Skipping evaluation for sample {i} (generated image is None).")
            for metric in metrics:
                scores[metric] = np.nan
            results.append(scores)
            continue

        try:
            gt_img = Image.open(gt_path).convert("RGB")

            if 'ssim' in metrics:
                scores['ssim'] = compute_ssim(gt_img, gen_img)

            if 'lpips' in metrics and lpips_fn:
                 # Resize generated image to match GT for LPIPS consistency if needed
                 gen_img_resized = gen_img.resize(gt_img.size)
                 scores['lpips'] = compute_lpips(gt_img, gen_img_resized, lpips_fn)
            elif 'lpips' in metrics:
                 scores['lpips'] = np.nan # Model failed to load

            if 'clip_sim' in metrics and clip_scoring_model:
                scores['clip_sim'] = compute_clip_similarity(gt_img, gen_img, clip_scoring_model, clip_scoring_preprocess)
            elif 'clip_sim' in metrics:
                 scores['clip_sim'] = np.nan # Model failed to load

        except FileNotFoundError:
            print(f"Error: Ground truth image not found: {gt_path}")
            for metric in metrics:
                scores[metric] = np.nan
        except Exception as e:
            print(f"Error evaluating sample {i} (GT: {gt_path}): {e}")
            for metric in metrics:
                scores[metric] = np.nan # Assign NaN for all metrics on error

        results.append(scores)

    results_df = pd.DataFrame(results)
    print("Evaluation complete.")

     # Print average scores
    print("\nAverage Scores:")
    for metric in metrics:
        if metric in results_df.columns:
             mean_score = results_df[metric].mean() # NaNs are automatically skipped
             print(f"- {metric}: {mean_score:.4f}")

    return results_df


def save_evaluation_results(results_df, model_name, filename=None):
    """Saves the evaluation results DataFrame to a CSV file."""
    if filename is None:
        filename = f"evaluation_results_{model_name}.csv"
    save_path = os.path.join(config.EVALUATION_RESULTS_PATH, filename)
    try:
        results_df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to: {save_path}")
    except Exception as e:
        print(f"Error saving evaluation results to {save_path}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Evaluation ---")
    # Requires generated images and corresponding ground truth paths

    # Simulate paths and images (replace with actual data)
    n_test_samples = 5
    sim_gt_paths = []
    sim_gen_images = []
    temp_eval_dir = os.path.join(config.OUTPUT_BASE_PATH, "temp_eval_test")
    os.makedirs(temp_eval_dir, exist_ok=True)

    print(f"Creating {n_test_samples} dummy image pairs in {temp_eval_dir}...")
    for i in range(n_test_samples):
         gt_path = os.path.join(temp_eval_dir, f"gt_{i}.png")
         gen_img = Image.new('RGB', (224, 224), color = (np.random.randint(0, 255), 0, 0)) # Randomish red
         gt_img = Image.new('RGB', (256, 256), color = (0, np.random.randint(0, 255), 0)) # Randomish green (diff size)

         try:
             gt_img.save(gt_path)
             sim_gt_paths.append(gt_path)
             sim_gen_images.append(gen_img)
         except Exception as e:
             print(f"Error creating dummy file {gt_path}: {e}")

    # Add a None case
    sim_gt_paths.append(os.path.join(temp_eval_dir, f"gt_{n_test_samples}.png"))
    Image.new('RGB', (256, 256), color = 'blue').save(sim_gt_paths[-1])
    sim_gen_images.append(None)


    if not sim_gt_paths:
         print("Could not create dummy files for testing.")
    else:
        print(f"Created {len(sim_gen_images)} dummy image pairs (one generated is None).")
        # Evaluate
        results_dataframe = evaluate_reconstructions(sim_gt_paths, sim_gen_images)

        if results_dataframe is not None:
            print("\nEvaluation Results DataFrame:")
            print(results_dataframe)

            # Save results
            save_evaluation_results(results_dataframe, "test_eval")

        # Clean up dummy files
        print(f"Cleaning up dummy files in {temp_eval_dir}...")
        # shutil.rmtree(temp_eval_dir) # Be careful with rmtree

    print("\n--- Evaluation Test Complete ---")
