# generation.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import tqdm
import time
import numpy as np # Added numpy import

import config

# Global variable to hold the pipeline to avoid reloading it repeatedly
_sd_pipeline = None
_sd_tokenizer = None # Store tokenizer for unconditional embeds
_sd_text_encoder = None # Store text encoder for unconditional embeds

def load_stable_diffusion_pipeline(model_id=config.STABLE_DIFFUSION_MODEL_ID, device=config.DEVICE):
    """Loads the Stable Diffusion pipeline, tokenizer and text encoder if not already loaded."""
    global _sd_pipeline, _sd_tokenizer, _sd_text_encoder
    if _sd_pipeline is None:
        print(f"Loading Stable Diffusion pipeline: {model_id}...")
        start_time = time.time()
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            # Optional: Use a faster scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
             # Optional: Enable memory-efficient attention if xformers is installed
            try:
                 import xformers
                 pipe.enable_xformers_memory_efficient_attention()
                 print("Enabled xformers memory efficient attention.")
            except ImportError:
                 print("xformers not installed. Running without memory efficient attention.")

            _sd_pipeline = pipe
            # Load tokenizer and text_encoder separately for unconditional embeddings
            _sd_tokenizer = getattr(pipe, 'tokenizer', None)
            _sd_text_encoder = getattr(pipe, 'text_encoder', None)
            if _sd_tokenizer is None or _sd_text_encoder is None:
                 print("Warning: Could not access tokenizer or text_encoder from the pipeline object.")
                 # Attempt to load them based on model ID if standard attributes fail
                 try:
                      from transformers import CLIPTokenizer, CLIPTextModel
                      print("Attempting to load Tokenizer/TextEncoder separately...")
                      _sd_tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
                      _sd_text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device).eval()
                      print("Loaded Tokenizer/TextEncoder separately.")
                 except Exception as load_err:
                      print(f"Failed to load Tokenizer/TextEncoder separately: {load_err}")
                      print("Will fallback to zero unconditional embeddings.")


            end_time = time.time()
            print(f"Stable Diffusion pipeline loaded in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading Stable Diffusion pipeline: {e}")
            print("Please ensure you have the necessary libraries (diffusers, transformers, accelerate) installed and are logged into Hugging Face Hub if needed.")
            _sd_pipeline = None # Ensure it remains None on failure
            return None
    # else:
        # print("Stable Diffusion pipeline already loaded.")
    return _sd_pipeline

@torch.no_grad() # Ensure no gradients are computed during generation
def get_unconditional_embedding(batch_size=1, device=config.DEVICE, dtype=torch.float16):
     """ Gets the unconditional embedding (typically for empty prompt). """
     global _sd_tokenizer, _sd_text_encoder
     uncond_embeddings = None

     if _sd_tokenizer is not None and _sd_text_encoder is not None:
          try:
               max_length = _sd_tokenizer.model_max_length
               uncond_input = _sd_tokenizer(
                    [""] * batch_size, # Empty string prompt
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
               )
               # Ensure text encoder is in eval mode and correct device/dtype
               _sd_text_encoder = _sd_text_encoder.to(device).to(dtype).eval()
               uncond_embeddings = _sd_text_encoder(uncond_input.input_ids.to(device))[0] # Get hidden states
               # print(f"Computed unconditional embedding shape: {uncond_embeddings.shape}") # Debug
          except Exception as e:
               print(f"Error computing unconditional embedding using text encoder: {e}")
               uncond_embeddings = None # Fallback

     if uncond_embeddings is None:
          print("Warning: Using zero vector for unconditional embedding.")
          # Determine expected embedding dimension from the pipeline if possible
          embed_dim = 768 # Default for SD v1.5 CLIP ViT-L/14
          if _sd_pipeline and hasattr(_sd_pipeline, 'unet') and hasattr(_sd_pipeline.unet, 'config'):
              # Try to infer from UNet cross-attention dimension
              unet_config = getattr(_sd_pipeline.unet, 'config', {})
              cross_attn_dim = unet_config.get('cross_attention_dim', embed_dim)
              embed_dim = cross_attn_dim if cross_attn_dim else embed_dim
              # SD v1.5 expects sequence length 77
              seq_len = 77 # Default max length for CLIP tokenizer in SD
              uncond_shape = (batch_size, seq_len, embed_dim)
              print(f"Using fallback uncond shape: {uncond_shape}")
              uncond_embeddings = torch.zeros(uncond_shape, device=device, dtype=dtype)

     return uncond_embeddings

# --- NEW FUNCTION for Direct Embedding Conditioning ---
def generate_images_from_embeddings(predicted_embeddings, # Numpy array or Tensor
                                    guidance_scale=config.STABLE_DIFFUSION_GUIDANCE_SCALE,
                                    num_inference_steps=30): # Slightly more steps might help
    """
    Generates images using Stable Diffusion conditioned DIRECTLY on predicted embeddings.

    Args:
        predicted_embeddings (np.ndarray or torch.Tensor): Predicted embeddings from mapping. Shape [num_samples, embedding_dim].
        guidance_scale (float): Guidance scale for generation (Classifier-Free Guidance).
        num_inference_steps (int): Number of diffusion steps.

    Returns:
        list: A list of generated PIL Images (or None on error).
    """
    pipe = load_stable_diffusion_pipeline()
    if pipe is None:
        print("Error: Stable Diffusion pipeline failed to load.")
        return [None] * len(predicted_embeddings)

    # --- Prepare Embeddings ---
    if isinstance(predicted_embeddings, np.ndarray):
         predicted_embeddings_tensor = torch.from_numpy(predicted_embeddings).to(config.DEVICE)
    else:
         predicted_embeddings_tensor = predicted_embeddings.to(config.DEVICE)

    # Ensure correct dtype (match pipeline's expected dtype, usually float16)
    target_dtype = torch.float16 # Common for SD pipelines
    if hasattr(pipe, 'dtype'): target_dtype = pipe.dtype # Use pipeline's dtype if available
    predicted_embeddings_tensor = predicted_embeddings_tensor.to(target_dtype)
    num_samples = predicted_embeddings_tensor.shape[0]

    # --- Get Unconditional Embedding ---
    # Compute once for all samples (batch_size=1, will be broadcast/repeated)
    uncond_embeddings = get_unconditional_embedding(batch_size=1, device=config.DEVICE, dtype=target_dtype)
    if uncond_embeddings is None:
         print("Error: Failed to obtain unconditional embedding. Cannot proceed.")
         return [None] * num_samples

    # --- Check/Adjust Embedding Dimensions ---
    # Predicted embeddings are likely [batch, dim]. SD often expects [batch, seq_len, dim].
    # We might need to replicate the single embedding across the sequence length dimension.
    expected_shape_part = uncond_embeddings.shape[1:] # Shape without batch dim (e.g., [77, 768])
    print(f"Pipeline expects embedding shape (excluding batch): {expected_shape_part}")
    print(f"Predicted embedding shape (excluding batch): {predicted_embeddings_tensor.shape[1:]}")

    if predicted_embeddings_tensor.ndim == 2 and uncond_embeddings.ndim == 3:
        # We have [batch, dim], need [batch, seq_len, dim]
        seq_len = uncond_embeddings.shape[1] # Get seq_len from uncond embed
        print(f"Repeating predicted embeddings across sequence length ({seq_len}) dimension.")
        # Repeat the embedding vector 'seq_len' times along a new dimension 1
        predicted_embeddings_tensor = predicted_embeddings_tensor.unsqueeze(1).repeat(1, seq_len, 1)
        print(f"Adjusted predicted embeddings shape: {predicted_embeddings_tensor.shape}")

    elif predicted_embeddings_tensor.ndim != uncond_embeddings.ndim:
         print(f"Error: Dimensionality mismatch between predicted ({predicted_embeddings_tensor.ndim}D) and unconditional ({uncond_embeddings.ndim}D) embeddings after potential adjustment. Cannot proceed.")
         return [None] * num_samples
    elif predicted_embeddings_tensor.shape[1:] != expected_shape_part:
         print(f"Warning: Shape mismatch after potential adjustment. Predicted: {predicted_embeddings_tensor.shape[1:]} vs Expected: {expected_shape_part}. Trying anyway...")
         # This might still fail later in the UNet


    # --- Generation Loop ---
    generated_images = []
    print(f"Generating {num_samples} images using predicted embeddings...")
    # Process one embedding at a time for memory safety
    for i in tqdm.tqdm(range(num_samples), desc="Generating Images"):
        cond_embedding = predicted_embeddings_tensor[i:i+1] # Keep batch dim: Shape [1, seq_len, embedding_dim]

        # Classifier-Free Guidance: Concatenate unconditional and conditional embeddings
        # Repeat unconditional embedding to match batch size (which is 1 here)
        final_embeddings = torch.cat([uncond_embeddings, cond_embedding]) # Shape [2, seq_len, embedding_dim]

        try:
            # Generate image using prompt_embeds
            # Use torch.autocast for potential mixed-precision speedup/memory saving
            with torch.autocast(config.DEVICE.type if config.DEVICE.type != 'mps' else 'cpu', dtype=target_dtype if config.DEVICE.type != 'mps' else torch.float32): # MPS doesn't support float16 autocast well
                 result = pipe(prompt_embeds=final_embeddings, # The core change!
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale, # CFG scale
                               # Optional: Add negative_prompt_embeds=uncond_embeddings if pipe supports it?
                               # Usually handled by CFG via guidance_scale.
                              )
                 image = result.images[0]
            generated_images.append(image)
        except Exception as e:
            print(f"\nError generating image for sample index {i}: {e}")
            import traceback
            traceback.print_exc() # Print detailed error
            generated_images.append(None) # Append placeholder on error

    print(f"\nGenerated {len([img for img in generated_images if img is not None])} images successfully.")
    return generated_images


# --- Example Usage (for testing this module) ---
if __name__ == "__main__":
    print("--- Testing Generation from Embeddings ---")

    # Simulate predicted embeddings (e.g., from Ridge/MLP)
    n_samples = 4
    embed_dim = 768 # Example for SD v1.5 base CLIP
    simulated_embeddings = np.random.rand(n_samples, embed_dim).astype(np.float32)
    print(f"Simulated embeddings shape: {simulated_embeddings.shape}")

    generated_images = generate_images_from_embeddings(simulated_embeddings, num_inference_steps=10) # Few steps for quick test

    # Check if images were generated
    valid_generated_images = [img for img in generated_images if img is not None]
    print(f"\nSuccessfully generated {len(valid_generated_images)} images.")

    if valid_generated_images:
        # Display the first valid generated image (if running in an environment that supports it)
        try:
            print("Displaying the first generated image:")
            valid_generated_images[0].show() # This might open in an external viewer
        except Exception as e:
            print(f"Could not display image automatically: {e}")

        # You would normally save these using the main script's logic

    print("\n--- Generation Test Complete ---")

# --- ADD THIS FUNCTION BACK ---
def save_generated_images(generated_images, ground_truth_paths, model_name, output_dir=config.GENERATED_IMAGES_PATH):
    """Saves generated images alongside their corresponding ground truth.

    Args:
        generated_images (list): List of generated PIL Images (or None).
        ground_truth_paths (list): List of paths to the corresponding ground truth images.
        model_name (str): Identifier for the model/method (used for subfolder).
        output_dir (str): Base directory to save images.
    """
    if len(generated_images) != len(ground_truth_paths):
        print(f"Warning: Mismatch between generated images ({len(generated_images)}) and GT paths ({len(ground_truth_paths)}). Cannot save reliably.")
        # Decide how to handle this - maybe save only the generated ones?
        # For now, we'll only save if lengths match for paired saving.
        return # Or adjust logic if you want to save unpaired images

    save_subdir = os.path.join(output_dir, model_name)
    os.makedirs(save_subdir, exist_ok=True)
    print(f"Saving generated images to: {save_subdir}")

    saved_count = 0
    for i, (gen_img, gt_path) in enumerate(zip(generated_images, ground_truth_paths)):
        if gen_img is None:
            # print(f"Skipping save for sample {i} as generated image is None.") # Optional print
            continue
        if gt_path is None:
             print(f"Skipping save for sample {i} as ground truth path is None.")
             continue

        try:
            # Create filenames
            # Extract base name safely, handle potential path issues
            base_gt_name = os.path.splitext(os.path.basename(gt_path))[0]
            if not base_gt_name: # Handle edge case of invalid path
                 base_gt_name = f"sample_{i}"

            gen_filename = os.path.join(save_subdir, f"{base_gt_name}_generated_{model_name}.png")
            gt_filename_copy = os.path.join(save_subdir, f"{base_gt_name}_ground_truth.png") # Save GT as PNG for consistency

            # Save generated image
            gen_img.save(gen_filename, "PNG")

            # Save copy of ground truth image (convert to RGB first)
            gt_img_pil = Image.open(gt_path).convert("RGB")
            gt_img_pil.save(gt_filename_copy, "PNG")

            saved_count += 1
        except FileNotFoundError:
             print(f"Error saving image pair for sample {i}: Ground truth file not found at {gt_path}")
        except Exception as e:
            print(f"Error saving image pair for sample {i} (GT: {gt_path}): {e}")

    print(f"Saved {saved_count} generated/ground_truth pairs.")