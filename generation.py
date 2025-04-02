# generation.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import tqdm
import time

import config

# Global variable to hold the pipeline to avoid reloading it repeatedly
_sd_pipeline = None

def load_stable_diffusion_pipeline(model_id=config.STABLE_DIFFUSION_MODEL_ID, device=config.DEVICE):
    """Loads the Stable Diffusion pipeline if not already loaded."""
    global _sd_pipeline
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

def generate_images_from_prompts(prompts, guidance_scale=config.STABLE_DIFFUSION_GUIDANCE_SCALE, num_inference_steps=25):
    """
    Generates images using Stable Diffusion based on a list of text prompts.

    Args:
        prompts (list): A list of text prompts (strings).
        guidance_scale (float): Guidance scale for generation.
        num_inference_steps (int): Number of diffusion steps.

    Returns:
        list: A list of generated PIL Images.
    """
    pipe = load_stable_diffusion_pipeline()
    if pipe is None:
        return [None] * len(prompts) # Return placeholders if loading failed

    generated_images = []
    print(f"Generating {len(prompts)} images using Stable Diffusion...")
    for prompt in tqdm.tqdm(prompts, desc="Generating Images"):
        if not isinstance(prompt, str) or not prompt:
             print(f"Warning: Skipping invalid prompt: {prompt}")
             generated_images.append(None) # Append placeholder for invalid prompt
             continue
        try:
            # Generate image
            with torch.autocast(config.DEVICE.type): # Use autocast for potential speedup with float16
                 result = pipe(prompt,
                               guidance_scale=guidance_scale,
                               num_inference_steps=num_inference_steps)
                 image = result.images[0]
            generated_images.append(image)
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            generated_images.append(None) # Append placeholder on error

    print(f"Generated {len([img for img in generated_images if img is not None])} images successfully.")
    return generated_images

def save_generated_images(generated_images, ground_truth_paths, model_name, output_dir=config.GENERATED_IMAGES_PATH):
    """Saves generated images alongside their corresponding ground truth."""
    if len(generated_images) != len(ground_truth_paths):
        print(f"Warning: Mismatch between generated images ({len(generated_images)}) and GT paths ({len(ground_truth_paths)}). Cannot save reliably.")
        # Decide how to handle this - maybe save only the generated ones?
        # For now, we'll only save if lengths match.
        return

    save_subdir = os.path.join(output_dir, model_name)
    os.makedirs(save_subdir, exist_ok=True)
    print(f"Saving generated images to: {save_subdir}")

    saved_count = 0
    for i, (gen_img, gt_path) in enumerate(zip(generated_images, ground_truth_paths)):
        if gen_img is None:
            print(f"Skipping save for sample {i} as generated image is None.")
            continue

        try:
            # Create filenames
            base_gt_name = os.path.splitext(os.path.basename(gt_path))[0]
            gen_filename = os.path.join(save_subdir, f"{base_gt_name}_generated_{model_name}.png")
            gt_filename_copy = os.path.join(save_subdir, f"{base_gt_name}_ground_truth.png") # Save GT as PNG for consistency

            # Save generated image
            gen_img.save(gen_filename)

            # Save copy of ground truth image (convert to RGB first)
            gt_img_pil = Image.open(gt_path).convert("RGB")
            gt_img_pil.save(gt_filename_copy)

            saved_count += 1
        except Exception as e:
            print(f"Error saving image pair for sample {i} (GT: {gt_path}): {e}")

    print(f"Saved {saved_count} generated/ground_truth pairs.")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Generation ---")

    # Example prompts (replace with actual retrieved prompts later)
    example_prompts = [
        "a golden retriever playing in a park",
        "a red sports car driving on a coastal road",
        "a futuristic cityscape at night with flying vehicles",
        "a tabby cat sleeping on a couch",
        "a sailboat on a calm blue ocean",
        None, # Test invalid prompt
        "a detailed portrait of an astronaut on Mars",
    ]

    # Simulate ground truth paths (replace with actual paths)
    sim_gt_paths = [f"/path/to/image_{i}.jpg" for i in range(len(example_prompts))]

    generated_images = generate_images_from_prompts(example_prompts)

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

        # Test saving (using simulated GT paths)
        test_model_name = "test_generation"
        # Make sure the paths exist for saving even if GT doesn't exist
        os.makedirs(os.path.dirname(sim_gt_paths[0]), exist_ok=True)
        # Create dummy GT files for saving test
        for p in sim_gt_paths:
             try: Image.new('RGB', (60, 30), color = 'red').save(p.replace(".jpg",".png")) # Save dummy as png
             except: pass # Ignore errors if path is invalid

        save_generated_images(generated_images,
                              [p.replace(".jpg",".png") for p in sim_gt_paths], # Use dummy png paths
                              test_model_name)

    print("\n--- Generation Test Complete ---")
