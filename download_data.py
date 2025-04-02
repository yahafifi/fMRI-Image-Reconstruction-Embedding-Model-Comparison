# download_data.py
import os
import requests
import zipfile
import tarfile # Added for potential future use
import shutil
import config  # Import your config file
import subprocess # For Kaggle API calls

# --- Helper Functions ---
def create_directory(path):
    os.makedirs(path, exist_ok=True)

def download_file(url, save_path):
    print(f"Downloading {os.path.basename(save_path)} from {url}...")
    # Add User-Agent header as Google Drive sometimes blocks default requests agent
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        # Use session for potential cookie handling
        session = requests.Session()
        response = session.get(url, stream=True, headers=headers, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes

        # Simple check for HTML content (often indicates login page or error)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type and len(response.content) < 2000: # Arbitrary small size check
             # Attempt to follow redirection if any (may not work for all GDrive links)
             if 'Location' in response.headers:
                  redirect_url = response.headers['Location']
                  print(f"Redirecting to: {redirect_url}")
                  response = session.get(redirect_url, stream=True, headers=headers, timeout=30)
                  response.raise_for_status()
                  if 'text/html' in response.headers.get('Content-Type', ''):
                       print(f"Error: Download URL for {os.path.basename(save_path)} likely points to an HTML page (login/error). URL might be expired or require authentication.")
                       return False
             else:
                  print(f"Error: Download URL for {os.path.basename(save_path)} likely points to an HTML page (login/error). URL might be expired or require authentication.")
                  # print(f"Content received:\n{response.text[:500]}...") # Debugging
                  return False


        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192 * 4): # Increased chunk size
                file.write(chunk)
        print(f"File downloaded and saved at: {save_path}")
        return True
    except requests.exceptions.Timeout:
         print(f"Error: Timeout occurred while downloading {url}")
         return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Add specific check for common GDrive quota error message if possible
        # if 'quota' in str(e).lower(): print("Google Drive Quota Exceeded?")
        return False
    except Exception as e:
        print(f"An error occurred while saving {save_path}: {e}")
        return False

def extract_zip_file(zip_path, extract_path, password=None):
    print(f"Extracting {zip_path} to {extract_path}...")
    try:
        os.makedirs(extract_path, exist_ok=True) # Ensure extract path exists
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pwd_bytes = password.encode() if password else None
            zip_ref.extractall(extract_path, pwd=pwd_bytes)
        print("Files extracted successfully.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        return False
    except RuntimeError as e:
        if 'password required' in str(e).lower() or 'bad password' in str(e).lower() :
             print(f"Error: Password required or incorrect for {zip_path}.")
        else:
            print(f"Error extracting {zip_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False

def organize_god_imagenet_files(base_path):
    """ Specific organization for the GOD dataset's ImageNet stimuli zip """
    print(f"Organizing GOD stimuli files in {base_path}...")
    images_subfolder = os.path.join(base_path, "images") # The zip might create this subdir

    source_base = base_path # Default source is the extraction base path
    # Check if the 'images' subfolder exists and seems valid
    if os.path.isdir(images_subfolder) and os.path.exists(os.path.join(images_subfolder, "training")):
        print("Found 'images' subfolder containing 'training'. Using it as source.")
        source_base = images_subfolder
    elif not os.path.exists(os.path.join(source_base, "training")):
        print(f"Critical: 'training' folder not found in {base_path} or {images_subfolder}. Cannot organize.")
        return False

    # --- Move CSV files ---
    moved_csv = True
    for file_name in ["image_test_id.csv", "image_training_id.csv"]:
        current_path = os.path.join(source_base, file_name)
        target_path = os.path.join(base_path, file_name) # Target is always the base path
        if os.path.exists(current_path):
            try:
                # Ensure target directory exists (should already by config)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.move(current_path, target_path)
                print(f"Moved {file_name} to {target_path}")
            except Exception as e:
                print(f"Error moving {file_name} from {current_path} to {target_path}: {e}")
                moved_csv = False
        elif not os.path.exists(target_path): # Check if already in target
            print(f"Warning: {file_name} not found at {current_path} and not at {target_path}")
            moved_csv = False # Consider this a failure? Maybe not critical if already moved.
    if not moved_csv:
        print("Potential issue moving CSV files.")
        # Decide if this is fatal return False

    # --- Move "training" and "test" folders ---
    moved_folders = True
    for folder_name in ["training", "test"]:
        current_path = os.path.join(source_base, folder_name)
        target_path = os.path.join(base_path, folder_name) # Target is the base path
        if os.path.exists(current_path) and current_path != target_path: # Avoid moving if source is target
             try:
                print(f"Moving folder {current_path} to {target_path}...")
                shutil.move(current_path, target_path)
                print(f"Moved '{folder_name}' folder to {target_path}")
             except Exception as e:
                print(f"Error moving folder {folder_name}: {e}")
                moved_folders = False
        elif not os.path.exists(target_path):
             print(f"Warning: '{folder_name}' folder not found at {current_path} or {target_path}")
             moved_folders = False
    if not moved_folders:
        print("Potential issue moving training/test folders.")
        # Decide if this is fatal return False

    # --- Remove redundant "images" folder ---
    # Only if it existed, is different from base_path, and is now potentially empty
    if source_base == images_subfolder and source_base != base_path and os.path.exists(images_subfolder):
        try:
            # Check if empty before removing
            if not os.listdir(images_subfolder):
                 print(f"Attempting to remove empty source folder: {images_subfolder}")
                 os.rmdir(images_subfolder) # Use rmdir for empty dir
                 print(f"Removed empty redundant folder: {images_subfolder}")
            else:
                 # Check if it only contains leftover MacOS hidden files
                 is_effectively_empty = all(f.startswith('.') for f in os.listdir(images_subfolder))
                 if is_effectively_empty:
                      print(f"Source folder {images_subfolder} contains only hidden files. Removing.")
                      shutil.rmtree(images_subfolder) # Use rmtree if hidden files exist
                      print(f"Removed redundant folder with hidden files: {images_subfolder}")
                 else:
                      print(f"Folder {images_subfolder} not empty after moving files ({os.listdir(images_subfolder)}), keeping it.")
        except Exception as e:
            print(f"Error removing folder {images_subfolder}: {e}")

    # --- Final Verification ---
    print("Verifying final structure...")
    final_train_path = os.path.join(base_path, "training")
    final_test_path = os.path.join(base_path, "test")
    final_train_csv = os.path.join(base_path, "image_training_id.csv")
    final_test_csv = os.path.join(base_path, "image_test_id.csv")

    train_ok = os.path.isdir(final_train_path) and len(os.listdir(final_train_path)) > 0
    test_ok = os.path.isdir(final_test_path) and len(os.listdir(final_test_path)) > 0
    csv_ok = os.path.exists(final_train_csv) and os.path.exists(final_test_csv)

    if train_ok and test_ok and csv_ok:
        print("Verified: 'training', 'test' folders, and CSV files exist in the target directory.")
        return True
    else:
        print(f"Error: Final structure verification failed.")
        if not train_ok: print(f"- Training folder missing or empty at {final_train_path}")
        if not test_ok: print(f"- Test folder missing or empty at {final_test_path}")
        if not csv_ok: print(f"- CSV file(s) missing at {base_path}")
        return False

# --- Kaggle Download Helper --- (Merged from previous suggestion)
def kaggle_download(dataset_slug, download_path):
    """Downloads a dataset using the Kaggle API."""
    print(f"Attempting to download Kaggle dataset '{dataset_slug}' to {download_path}...")
    os.makedirs(download_path, exist_ok=True)
    try:
        # Check for Kaggle API credentials
        api_key_path_root = "/root/.kaggle/kaggle.json"
        api_key_path_user = os.path.expanduser("~/.kaggle/kaggle.json")
        kaggle_json_path = None

        if os.path.exists(api_key_path_root):
            kaggle_json_path = api_key_path_root
        elif os.path.exists(api_key_path_user):
            kaggle_json_path = api_key_path_user

        if kaggle_json_path:
            print(f"Using Kaggle API credentials found at: {kaggle_json_path}")
            # Ensure correct permissions (required in some environments)
            os.chmod(kaggle_json_path, 0o600)
        else:
            print("Warning: Kaggle API key (kaggle.json) not found in standard locations (~/.kaggle/ or /root/.kaggle/).")
            print("Kaggle download might fail if credentials are required.")
            # Proceed anyway, might work in Kaggle notebook environment if dataset is public

        command = [
            'kaggle', 'datasets', 'download',
            '-d', dataset_slug,
            '-p', download_path,
            '--unzip', # Automatically unzip
            '--quiet'  # Suppress progress bars
        ]
        print(f"Running command: {' '.join(command)}")
        # Set KAGGLE_CONFIG_DIR environment variable if needed
        env = os.environ.copy()
        if kaggle_json_path:
            env['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_json_path)

        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        print(f"Kaggle dataset '{dataset_slug}' downloaded and extracted successfully.")
        # print("stdout:", result.stdout) # Uncomment for debugging
        return True
    except FileNotFoundError:
         print("Error: 'kaggle' command not found. Is the Kaggle API client installed (pip install kaggle) and in PATH?")
         return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing Kaggle command: {e}")
        print("Stderr:", e.stderr)
        print("Stdout:", e.stdout)
        print(f"Ensure the dataset slug '{dataset_slug}' is correct and you have permissions (e.g., accepted competition rules).")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Kaggle download: {e}")
        return False

# --- Main Download Function ---
def download_all_data():
    print("--- Starting Data Download and Setup ---")
    overall_success = True

    # Create base directories specified in config
    create_directory(config.DATA_BASE_PATH)
    create_directory(config.MODELS_BASE_PATH)
    create_directory(config.OUTPUT_BASE_PATH)
    # Specific subdirs needed by the pipeline
    create_directory(config.GOD_FMRI_PATH)
    create_directory(config.GOD_IMAGENET_PATH)
    create_directory(config.GOD_FEATURES_PATH)
    create_directory(config.IMAGENET256_FEATURES_PATH)
    create_directory(config.SAVED_KNN_MODELS_PATH)
    create_directory(config.GENERATED_IMAGES_PATH)
    create_directory(config.EVALUATION_RESULTS_PATH)


    # --- Files to download (URLs from original script) ---
    # WARNING: Google Drive URLs are highly likely to fail. Replace them if possible.
    files_to_download = {
        # GOD fMRI Data (Subject 3) - Figshare link seems more stable
        "https://figshare.com/ndownloader/files/24080126": os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5"),

        # GOD ImageNet Stimuli (Zip) - !! UNSTABLE GDrive link !!
        "https://drive.usercontent.google.com/download?id=1t4sjiG6fxtCEaFX2SGM6Qm6Yf6yzAICl&export=download&authuser=0&confirm=t&uuid=fc9f55ab-8c8a-48c8-ade5-f196cfcda715&at=APvzH3r2hWAMJmsgNC78q0WyDyhH%3A1735581470761": os.path.join(config.GOD_IMAGENET_PATH, "image_stimuli.zip"),

        # Precomputed ResNet50 Features (Optional) - !! UNSTABLE GDrive link !!
        "https://drive.usercontent.google.com/download?id=1ekPtyEnxgyncn64GPZ0iBpuMigq0_ijs&export=download&authuser=0&confirm=t&uuid=f18a1aaf-2211-4c18-a2fb-3fa590c4e56d&at=APvzH3rahhRmaSeEamw-dEvYcQo2:1735582682153": os.path.join(config.GOD_FEATURES_PATH, "feature_imagenet_500_resnet50.pt"),
        "https://drive.usercontent.google.com/download?id=1qwlk2UUUY8x013nDTBbklrULSiMEDzbn&export=download&authuser=0&confirm=t&uuid=60619370-c181-48ac-b3af-17ec6e56d291&at=APvzH3rlyvk5vB8vrUdtj79VDXdA:1735582741402": os.path.join(config.GOD_FEATURES_PATH, "labels_imagenet_500_resnet50.pt"),

        # Precomputed Ridge Model (Optional) - !! UNSTABLE GDrive link !!
        "https://drive.usercontent.google.com/download?id=199aEEhTt_MmTib1mCtah-ViPS10rXHmL&export=download&authuser=0&confirm=t&uuid=ac08cbd0-91b7-434f-a753-b87586c46719&at=APvzH3oqPAxTZZ7rEPsANanOeSLh:1735583273280": os.path.join(config.MODELS_BASE_PATH, f"ridge_sub{config.SUBJECT_ID}_original.sav"), # Rename slightly

        # Metadata/Mappings (GitHub links - more stable)
        "https://raw.githubusercontent.com/enomodnara/brain_decoding/refs/heads/main/class_to_wordnet.json": config.CLASS_TO_WORDNET_JSON,

        # GOD Captions (Optional) - GitHub link
        "https://raw.githubusercontent.com/YulongBonjour/BrainCLIP/refs/heads/main/GOD_caption/GOD_train_caps_hm_and_wdnet.json": os.path.join(config.KAGGLE_BASE_PATH, "GOD_train_caps_hm_and_wdnet.json"),
    }

    # --- Download loop ---
    print("\n--- Downloading required files ---")
    download_success_flags = {}
    any_download_failed = False
    for url, path in files_to_download.items():
        # Skip download if file already exists
        if os.path.exists(path):
            print(f"Skipping download, file exists: {path}")
            download_success_flags[path] = True
            continue

        print(f"\nAttempting to download: {os.path.basename(path)}")
        # Add specific warning for GDrive links
        if "drive.google.com" in url or "drive.usercontent.google.com" in url:
            print("WARNING: Attempting download from Google Drive URL. These links are often temporary and may fail due to permissions, quotas, or expiration.")

        if download_file(url, path):
            download_success_flags[path] = True
        else:
            download_success_flags[path] = False
            any_download_failed = True
            print(f"ERROR: Failed to download {os.path.basename(path)} from {url}")

    # --- Post-Download Processing ---
    print("\n--- Post-Download Processing ---")

    # 1. Extract GOD ImageNet Stimuli Zip
    zip_file_path = os.path.join(config.GOD_IMAGENET_PATH, "image_stimuli.zip")
    if download_success_flags.get(zip_file_path, False): # Check if download itself succeeded
        print(f"\nProcessing {zip_file_path}...")
        # Using password from original script
        if extract_zip_file(zip_file_path, config.GOD_IMAGENET_PATH, password="W9kaBybu"):
            # Organize the extracted files
           if not organize_god_imagenet_files(config.GOD_IMAGENET_PATH):
               print("ERROR: Failed to organize GOD stimuli files after extraction.")
               overall_success = False
           else:
                print("GOD stimuli organization successful.")
                # Optionally remove the zip file after successful extraction & organization
                try:
                    os.remove(zip_file_path)
                    print(f"Removed {zip_file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove {zip_file_path}: {e}")
        else:
            print(f"ERROR: Failed to extract {zip_file_path}. Stimuli data will be missing.")
            overall_success = False
    elif os.path.isdir(os.path.join(config.GOD_IMAGENET_PATH, "training")):
        print("GOD Stimuli zip not downloaded (or failed), but training/test folders seem to exist already.")
    else:
        # This case means download failed/skipped AND structure doesn't exist
        print(f"CRITICAL ERROR: GOD Stimuli ({zip_file_path}) not downloaded/extracted and final structure not found.")
        overall_success = False # Stimuli are essential

    # 2. Download ImageNet-256 (if needed) - Added Step
    print(f"\nChecking ImageNet-256 retrieval dataset at: {config.IMAGENET256_PATH}")
    imagenet_kaggle_slug = config.IMAGENET256_DATASET_NAME # Use name from config
    # Check if the target directory exists AND is not empty
    if not os.path.isdir(config.IMAGENET256_PATH) or not os.listdir(config.IMAGENET256_PATH):
        print(f"ImageNet-256 directory is missing or empty.")
        print(f"Attempting download using Kaggle API (slug: '{imagenet_kaggle_slug}')...")
        # Define where Kaggle should download files (often needs to be /kaggle/working or similar writable area)
        # The actual dataset will likely appear under /kaggle/input if linked via Kaggle UI
        # If downloading manually via API, choose a writable path.
        # Let's assume config.INPUT_BASE_PATH is appropriate if not using Kaggle UI linking.
        kaggle_download_target_dir = config.INPUT_BASE_PATH

        if not kaggle_download(imagenet_kaggle_slug, kaggle_download_target_dir):
             print(f"ERROR: Failed to download ImageNet-256 using Kaggle slug '{imagenet_kaggle_slug}'.")
             print(f"Please ensure the slug is correct, Kaggle API is set up, and the target path '{config.IMAGENET256_PATH}' is correct or linkable.")
             overall_success = False
        else:
             # Crucial check: After download, does config.IMAGENET256_PATH exist?
             # Kaggle might download to kaggle_download_target_dir/dataset_artifact_name
             # We need to ensure config.IMAGENET256_PATH points to the actual class folders
             if not os.path.isdir(config.IMAGENET256_PATH) or not os.listdir(config.IMAGENET256_PATH):
                  print(f"Warning: Kaggle download finished, but the expected path '{config.IMAGENET256_PATH}' is still missing or empty.")
                  print(f"Please check the contents of '{kaggle_download_target_dir}' and adjust config.IMAGENET256_PATH or create a symlink.")
                  # Example potential actual path:
                  potential_actual_path = os.path.join(kaggle_download_target_dir, imagenet_kaggle_slug.split('/')[-1])
                  print(f"Did the data download to a subdirectory like: {potential_actual_path}?")
                  overall_success = False # Mark as failure until path is correct
             else:
                  print("ImageNet-256 download via Kaggle successful and path seems correct.")

    else:
         print(f"ImageNet-256 data directory ({config.IMAGENET256_PATH}) already exists and is not empty.")


    # --- Final Summary ---
    print("\n--- Data Download and Setup Summary ---")
    if any_download_failed:
        print("WARNING: One or more file downloads failed. Check logs above.")
        overall_success = False # Ensure overall status reflects download failures

    if overall_success:
        print("Attempted download and setup seems completed. Subsequent checks will verify data presence.")
    else:
        print("ERROR: Data download and setup encountered critical errors. Pipeline might fail.")

    return overall_success


if __name__ == "__main__":
    # This allows running the script directly to download data
    if download_all_data():
         print("\nDownload script finished. Please review logs for any warnings or errors.")
    else:
         print("\nDownload script finished with errors.")
         print("Please check the download URLs (especially Google Drive), permissions, and Kaggle setup.")