# mapping_models.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge # Keep Ridge for comparison if needed later
import pickle
import os
import time
import copy # For deep copying best model state
import traceback # For error printing

import config

# --- Evaluation Function (Works for both Ridge and MLP) ---
def evaluate_prediction(Z_true, Z_pred, dataset_name="Test"):
    """
    Calculates RMSE and R2 score for predictions.
    Handles both numpy arrays and torch tensors (converts tensors to numpy).
    """
    if Z_true is None or Z_pred is None:
        print(f"Warning: Cannot evaluate prediction for {dataset_name} - data is None.")
        return np.nan, np.nan

    if isinstance(Z_true, torch.Tensor):
        Z_true = Z_true.detach().cpu().numpy()
    if isinstance(Z_pred, torch.Tensor):
        Z_pred = Z_pred.detach().cpu().numpy()

    if Z_true.shape != Z_pred.shape:
         print(f"Warning: Shape mismatch in evaluation ({dataset_name}). True: {Z_true.shape}, Pred: {Z_pred.shape}")
         # Attempt to evaluate if only batch dim differs (e.g., comparing single pred to single true)
         if Z_true.ndim == Z_pred.ndim and Z_true.shape[1:] == Z_pred.shape[1:] and Z_pred.shape[0] > 0 and Z_true.shape[0] > 0 :
              print("  Attempting evaluation despite different sample counts (using available predictions)...")
              min_samples = min(Z_true.shape[0], Z_pred.shape[0])
              Z_true = Z_true[:min_samples]
              Z_pred = Z_pred[:min_samples]
              if min_samples <= 1: # Cannot calculate R2 reliably for <= 1 sample
                    print("  Cannot calculate R2 score with <= 1 matching sample.")
                    try:
                         mse = mean_squared_error(Z_true, Z_pred)
                         rmse = np.sqrt(mse)
                         print(f"Evaluation ({dataset_name}) - RMSE: {rmse:.4f}, R2 Score: N/A")
                         return rmse, np.nan
                    except Exception as e:
                         print(f"  Error calculating RMSE for single sample: {e}")
                         return np.nan, np.nan

         else:
              print(f"  Cannot evaluate due to incompatible shapes.")
              return np.nan, np.nan # Return NaN if shapes are fundamentally incompatible

    if Z_true.size == 0 or Z_pred.size == 0:
        print(f"Warning: Cannot evaluate prediction for {dataset_name} - data arrays are empty.")
        return np.nan, np.nan

    try:
        mse = mean_squared_error(Z_true, Z_pred)
        rmse = np.sqrt(mse)
        # R2 score requires at least 2 samples if calculated per feature, or needs multioutput='uniform_average'
        if Z_true.shape[0] > 1:
             # variance_weighted is generally preferred for multioutput regression
             r2 = r2_score(Z_true, Z_pred, multioutput='variance_weighted')
        else:
             # R2 is ill-defined for a single sample; calculate relative MSE if needed, or return NaN/0
             r2 = np.nan # Indicate R2 is not applicable
             print(f"  (Cannot calculate R2 score with only {Z_true.shape[0]} sample)")

        print(f"Evaluation ({dataset_name}) - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        return rmse, r2
    except Exception as e:
        print(f"Error during evaluation ({dataset_name}): {e}")
        traceback.print_exc()
        return np.nan, np.nan


# === MLP Implementation ===

# 1. MLP Model Definition
class FmriMappingMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu', dropout_rate=0.5):
        super().__init__()
        # Add a print statement to show the dimensions being used
        print(f"Initializing FmriMappingMLP: Input={input_dim}, Output={output_dim}")
        layers = []
        last_dim = input_dim

        # Activation function mapping
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        # Add more activation functions as needed
        else:
            print(f"Warning: Unsupported activation '{activation}'. Using ReLU.")
            act_fn = nn.ReLU()

        print(f"Building MLP: Input={input_dim}", end="")
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_fn)
            print(f" -> H{i+1}({hidden_dim}, {activation})", end="")
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                print(f"[D={dropout_rate:.1f}]", end="")
            last_dim = hidden_dim

        # Output layer (no activation or dropout usually)
        layers.append(nn.Linear(last_dim, output_dim))
        print(f" -> Output({output_dim})")

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 2. Simple Dataset for fMRI -> Embedding pairs
class FmriEmbeddingDataset(Dataset):
    def __init__(self, fmri_data, embedding_data):
        # Ensure data are numpy arrays initially for checks
        if not isinstance(fmri_data, np.ndarray):
            raise TypeError("fmri_data must be a numpy array")
        if not isinstance(embedding_data, np.ndarray):
            raise TypeError("embedding_data must be a numpy array")

        if fmri_data.shape[0] != embedding_data.shape[0]:
            raise ValueError(f"Sample count mismatch: fMRI ({fmri_data.shape[0]}) vs Embeddings ({embedding_data.shape[0]})")
        if fmri_data.size == 0 or embedding_data.size == 0:
             raise ValueError("Input data arrays cannot be empty")

        # Convert to tensors
        self.fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)
        self.embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
        print(f"Created FmriEmbeddingDataset with {len(self.fmri_tensor)} samples.")

    def __len__(self):
        return len(self.fmri_tensor)

    def __getitem__(self, idx):
        return self.fmri_tensor[idx], self.embedding_tensor[idx]

# 3. MLP Training Function
def train_mlp_mapping(X_train, Z_train, X_val, Z_val, model_name, fmri_model_name):
    """Trains an MLP mapping model and saves the best version based on validation loss."""
    print(f"--- Starting MLP Training for {fmri_model_name} -> {model_name} ---")
    start_train_time = time.time()

    device = config.DEVICE
    input_dim = X_train.shape[1]
    output_dim = Z_train.shape[1] # Target dimension (can be full embedding or PCA)

    # Create datasets and dataloaders
    try:
        train_dataset = FmriEmbeddingDataset(X_train, Z_train)
        train_loader = DataLoader(train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=(device != torch.device('cpu')))
    except Exception as e:
        print(f"Error creating training dataset/loader: {e}")
        traceback.print_exc()
        return None, None

    val_loader = None
    if X_val is not None and Z_val is not None and X_val.size > 0 and Z_val.size > 0: # Check if validation data exists and is not empty
        try:
             val_dataset = FmriEmbeddingDataset(X_val, Z_val)
             val_loader = DataLoader(val_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=(device != torch.device('cpu')))
             print(f"Training with {len(train_dataset)} samples, Validating with {len(val_dataset)} samples.")
        except Exception as e:
             print(f"Error creating validation dataset/loader: {e}")
             traceback.print_exc()
             val_loader = None # Proceed without validation if it fails
             print(f"Training with {len(train_dataset)} samples. Validation loader creation failed.")
    else:
         print(f"Training with {len(train_dataset)} samples. No validation set provided or data is empty.")


    # Initialize model, loss, optimizer, scheduler
    try:
        model = FmriMappingMLP(
            input_dim=input_dim,
            output_dim=output_dim, # Use target dimension
            hidden_layers=config.MLP_HIDDEN_LAYERS,
            activation=config.MLP_ACTIVATION,
            dropout_rate=config.MLP_DROPOUT_RATE
        ).to(device)
    except Exception as e:
        print(f"Error initializing MLP model: {e}")
        traceback.print_exc()
        return None, None

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY)
    # Learning rate scheduler (ReduceLROnPlateau is good with validation)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.MLP_LR_FACTOR, patience=config.MLP_LR_PATIENCE, verbose=True)

    # Training loop with validation and early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}

    print(f"Starting training for {config.MLP_EPOCHS} epochs...")
    for epoch in range(config.MLP_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        for i, (fmri_batch, embed_batch) in enumerate(train_loader):
            fmri_batch = fmri_batch.to(device)
            embed_batch = embed_batch.to(device)

            optimizer.zero_grad()
            outputs = model(fmri_batch)
            loss = criterion(outputs, embed_batch)

            # Check for NaN loss
            if torch.isnan(loss):
                 print(f"ERROR: NaN loss detected at epoch {epoch+1}, batch {i+1}. Stopping training.")
                 return None, None # Indicate failure

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        avg_train_loss = running_loss / batch_count if batch_count > 0 else 0
        history['train_loss'].append(avg_train_loss)

        # Validation step
        avg_val_loss = float('inf') # Default if no validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for fmri_batch_val, embed_batch_val in val_loader:
                    fmri_batch_val = fmri_batch_val.to(device)
                    embed_batch_val = embed_batch_val.to(device)
                    outputs_val = model(fmri_batch_val)
                    loss_val = criterion(outputs_val, embed_batch_val)
                    if torch.isnan(loss_val):
                         print(f"Warning: NaN detected in validation loss at epoch {epoch+1}. Skipping validation loss update.")
                    else:
                        val_loss += loss_val.item()
                        val_batch_count += 1

            # Avoid division by zero if val_loader is empty or all batches had NaN
            if val_batch_count > 0:
                avg_val_loss = val_loss / val_batch_count
                history['val_loss'].append(avg_val_loss)

                # Reduce LR on plateau using calculated avg_val_loss
                scheduler.step(avg_val_loss)

                # Early stopping and best model saving logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # Save the best model state using deepcopy
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    print(f"Epoch {epoch+1}: Val loss ({avg_val_loss:.6f}) did not improve from {best_val_loss:.6f} ({epochs_no_improve}/{config.MLP_PATIENCE})")

                if epochs_no_improve >= config.MLP_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            else: # Validation failed (NaN or empty)
                 history['val_loss'].append(np.nan) # Record NaN if validation failed
                 print(f"Epoch {epoch+1}: Validation step skipped or resulted in NaN.")
                 # Cannot check for best model or early stopping without valid val_loss
                 # Save current model state as a fallback? Might not be desirable.
                 # best_model_state = copy.deepcopy(model.state_dict())

        else: # No validation loader
             history['val_loss'].append(np.nan)
             # If no validation, save the last model state (or based on train loss, less ideal)
             best_model_state = copy.deepcopy(model.state_dict()) # Save current/last state


        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
        print(f"Epoch {epoch+1}/{config.MLP_EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - LR: {current_lr:.1e} - Time: {epoch_time:.2f}s")


    # --- Post-Training ---
    final_val_rmse = np.nan
    final_val_r2 = np.nan

    # Load the best model state found during training (if any)
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state with validation loss: {best_val_loss:.6f}")

        # --- Evaluate the best model on the full validation set ---
        if val_loader: # Only if validation was used and training had a best state
            print("Evaluating best model on the full validation set...")
            model.eval() # Ensure eval mode
            all_val_preds = []
            all_val_true = []
            with torch.no_grad():
                for fmri_batch_val, embed_batch_val in val_loader:
                    fmri_batch_val = fmri_batch_val.to(device)
                    outputs_val = model(fmri_batch_val)
                    all_val_preds.append(outputs_val.cpu())
                    all_val_true.append(embed_batch_val.cpu()) # Keep true embeddings on CPU

            if all_val_preds:
                try:
                    Z_val_pred_final = torch.cat(all_val_preds).numpy()
                    Z_val_true_final = torch.cat(all_val_true).numpy()
                    print("Validation Set Evaluation (Best Model):")
                    final_val_rmse, final_val_r2 = evaluate_prediction(Z_val_true_final, Z_val_pred_final, dataset_name="Validation")
                except Exception as eval_e:
                    print(f"Error evaluating best model on validation set: {eval_e}")
                    traceback.print_exc()
            else:
                 print("No predictions collected for final validation evaluation.")
        else:
             print("No validation loader available to evaluate the final model.")

    elif not val_loader:
         print("Training finished without validation. Using final model state.")
         # best_model_state will contain the state from the last epoch if no validation
         if best_model_state:
              model.load_state_dict(best_model_state)
         else:
              print("Warning: No model state available after training.")
              return None, None # Cannot proceed without a model state

    else: # Validation loader existed, but no best_model_state (e.g., NaN errors early)
         print("Warning: No best model state found (e.g., NaN loss during training/validation). Cannot save or return a reliable model.")
         return None, None


    # Save the trained model state dictionary (using the loaded best state)
    # Determine filename suffix based on PCA config (handled in run_experiment now)
    # Here we just use the passed names for the filename core
    pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""
    model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{fmri_model_name}_{model_name}{pca_suffix}.pt")

    try:
        os.makedirs(config.MODELS_BASE_PATH, exist_ok=True)
        # Save model state dict, input/output dims, and maybe config used
        save_dict = {
            'model_state_dict': model.state_dict(), # Save the best state loaded above
            'input_dim': input_dim,
            'output_dim': output_dim, # Save the actual output dimension used
            'hidden_layers': config.MLP_HIDDEN_LAYERS,
            'activation': config.MLP_ACTIVATION,
            'dropout_rate': config.MLP_DROPOUT_RATE,
            'final_train_loss': avg_train_loss, # Last epoch train loss
            'best_val_loss': best_val_loss, # Best validation MSE loss achieved
            'final_val_rmse': final_val_rmse, # RMSE on val set using best model
            'final_val_r2': final_val_r2     # R2 on val set using best model
        }
        torch.save(save_dict, model_filename)
        print(f"Saved trained MLP model to: {model_filename}")
    except Exception as e:
        print(f"Error saving MLP model: {e}")
        traceback.print_exc()
        model_filename = None # Indicate saving failed

    total_train_time = time.time() - start_train_time
    print(f"--- MLP Training Finished. Final Val R2: {final_val_r2:.4f}. Total Time: {total_train_time/60:.2f} minutes ---")

    # Optionally return validation metrics if using for hyperparameter tuning external script
    # return model, model_filename, final_val_r2
    return model, model_filename


# 4. MLP Loading Function
def load_mlp_model(model_filename, device=config.DEVICE):
    """Loads a saved MLP model state dictionary."""
    if not os.path.exists(model_filename):
        print(f"Error: MLP model file not found: {model_filename}")
        return None
    try:
        print(f"Loading MLP model from: {model_filename}")
        checkpoint = torch.load(model_filename, map_location=device)

        # Check for essential keys
        required_keys = ['input_dim', 'output_dim', 'hidden_layers', 'model_state_dict']
        if not all(key in checkpoint for key in required_keys):
             raise KeyError(f"Checkpoint file {model_filename} missing required keys.")

        # Recreate model architecture using saved hyperparameters
        model = FmriMappingMLP(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            hidden_layers=checkpoint['hidden_layers'],
            # Use saved activation/dropout, fall back to config defaults if missing (older saves)
            activation=checkpoint.get('activation', config.MLP_ACTIVATION),
            dropout_rate=checkpoint.get('dropout_rate', config.MLP_DROPOUT_RATE)
        ).to(device)

        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set to evaluation mode IMPORTANT!
        print(f"Loaded MLP model successfully.")
        print(f"  - Architecture: In={checkpoint['input_dim']}, Out={checkpoint['output_dim']}, Hidden={checkpoint['hidden_layers']}, Act={checkpoint.get('activation','N/A')}, Dropout={checkpoint.get('dropout_rate','N/A')}")
        print(f"  - Best recorded validation loss: {checkpoint.get('best_val_loss', np.nan):.6f}")
        print(f"  - Final validation R2: {checkpoint.get('final_val_r2', np.nan):.4f}")
        return model
    except Exception as e:
        print(f"Error loading MLP model from {model_filename}: {e}")
        traceback.print_exc()
        return None


# 5. MLP Prediction Function
@torch.no_grad() # Essential for inference
def predict_embeddings_mlp(mlp_model, X_data, device=config.DEVICE, batch_size=config.MLP_BATCH_SIZE*2):
    """Predicts embeddings using the loaded MLP model, handling large data."""
    if mlp_model is None:
        print("Error: MLP model is None. Cannot predict.")
        return None

    mlp_model.eval() # Ensure model is in evaluation mode

    # Convert numpy data to tensor
    if isinstance(X_data, np.ndarray):
        # Handle empty array case
        if X_data.size == 0:
             print("Warning: Input data for MLP prediction is empty.")
             return np.array([]) # Return empty numpy array
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
    elif isinstance(X_data, torch.Tensor):
        if X_data.numel() == 0:
             print("Warning: Input tensor for MLP prediction is empty.")
             return np.array([])
        X_tensor = X_data.float() # Ensure correct type
    else:
        print("Error: Input data must be a numpy array or torch tensor.")
        return None

    print(f"Predicting embeddings for {X_tensor.shape[0]} samples using MLP (Batch Size: {batch_size})...")
    all_preds = []

    # Create a simple dataloader for prediction batches
    try:
        # Handle potential empty tensor for dataset creation
        if X_tensor.shape[0] == 0:
             print("Input tensor has 0 samples, returning empty prediction.")
             return np.array([])
             
        pred_dataset = torch.utils.data.TensorDataset(X_tensor)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error creating prediction dataloader: {e}")
        return None


    try:
        for batch_tensors in pred_loader:
            # Ensure batch_tensors is not empty and is indexable
            if not batch_tensors: continue
            batch_data = batch_tensors[0].to(device) # Get data from tuple and move to device
            if batch_data.numel() == 0: continue # Skip empty batches

            batch_preds = mlp_model(batch_data)
            all_preds.append(batch_preds.cpu()) # Collect predictions on CPU

        # Handle case where no predictions were made (e.g., all input batches were empty)
        if not all_preds:
             print("Warning: No predictions were generated.")
             # Determine expected output shape
             try:
                 output_dim = mlp_model.network[-1].out_features
                 return np.empty((0, output_dim), dtype=np.float32)
             except Exception:
                 return np.array([]) # Fallback


        Z_pred_tensor = torch.cat(all_preds, dim=0)

    except Exception as e:
        print(f"Error during MLP prediction loop: {e}")
        traceback.print_exc()
        return None

    print("Prediction complete.")
    # Return predictions as numpy array (common format for downstream tasks)
    return Z_pred_tensor.numpy()


# === Ridge Regression Implementation (Kept for reference/comparison) ===
def train_ridge_mapping(X_train, Z_train, alpha, max_iter, model_name, fmri_model_name):
    """Trains a Ridge regression model and saves it."""
    print(f"--- Starting Ridge Training for {fmri_model_name} -> {model_name} ---")
    start_train_time = time.time()
    try:
        if X_train.size == 0 or Z_train.size == 0:
             raise ValueError("Ridge training requires non-empty X_train and Z_train.")
             
        ridge = Ridge(alpha=alpha, max_iter=max_iter, random_state=config.RANDOM_STATE)
        print(f"Fitting Ridge model (alpha={alpha})...")
        ridge.fit(X_train, Z_train)

        # Evaluate on training data (optional, sanity check)
        print("Evaluating Ridge model on training data...")
        Z_train_pred = ridge.predict(X_train)
        train_rmse, train_r2 = evaluate_prediction(Z_train, Z_train_pred, dataset_name="Train") # Use updated eval func
        print(f"Ridge training complete. Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")

        # Save the trained model
        pca_suffix = f"_pca{config.PCA_N_COMPONENTS}" if config.USE_PCA_TARGET else ""
        model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{fmri_model_name}_{model_name}{pca_suffix}_alpha{alpha}.sav")
        os.makedirs(config.MODELS_BASE_PATH, exist_ok=True)
        with open(model_filename, 'wb') as f:
            pickle.dump(ridge, f)
        print(f"Saved trained Ridge model to: {model_filename}")
        total_train_time = time.time() - start_train_time
        print(f"--- Ridge Training Finished. Total Time: {total_train_time:.2f} seconds ---")
        return ridge, model_filename

    except Exception as e:
        print(f"Error training Ridge model: {e}")
        traceback.print_exc()
        return None, None


def load_ridge_model(model_filename):
    """Loads a saved Ridge model."""
    if not os.path.exists(model_filename):
        print(f"Error: Ridge model file not found: {model_filename}")
        return None
    try:
        with open(model_filename, 'rb') as f:
            ridge = pickle.load(f)
        print(f"Loaded Ridge model from: {model_filename}")
        return ridge
    except Exception as e:
         print(f"Error loading Ridge model: {e}")
         traceback.print_exc()
         return None

def predict_embeddings_ridge(ridge_model, X_data):
    """Predicts embeddings using the loaded Ridge model."""
    if ridge_model is None:
         print("Error: Ridge model is None.")
         return None
    if X_data.size == 0:
         print("Warning: Input data for Ridge prediction is empty.")
         # Determine expected output shape if possible (difficult for Ridge)
         return np.array([]) # Return empty array

    print(f"Predicting embeddings for {X_data.shape[0]} samples using Ridge...")
    try:
        Z_pred = ridge_model.predict(X_data)
        print("Prediction complete.")
        return Z_pred
    except Exception as e:
         print(f"Error during Ridge prediction: {e}")
         traceback.print_exc()
         return None


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Mapping Models ---")

    # --- Simulate Data ---
    emb_model_name = "resnet50" # Example embedding model name
    fmri_model_name = f"Subj{config.SUBJECT_ID}_{config.ROI}" # Example identifier for fMRI source
    dim_fmri = 4466 # Example dimensionality for VC ROI from logs
    dim_embedding = config.EMBEDDING_MODELS[emb_model_name]['embedding_dim']
    n_train = 1080 # Match logs
    n_val = 120   # Match logs
    n_test = 50    # Match logs

    # Determine if PCA is used for simulation targets
    if config.USE_PCA_TARGET:
        print(f"Simulating with PCA target (n_components={config.PCA_N_COMPONENTS})")
        dim_target = config.PCA_N_COMPONENTS
    else:
        print("Simulating with full embedding target")
        dim_target = dim_embedding


    print("Generating simulated data...")
    X_train_sim = np.random.rand(n_train, dim_fmri).astype(np.float32)
    # Simulate some structure: Z ~ X*W + noise
    W_sim = np.random.randn(dim_fmri, dim_target).astype(np.float32) * 0.01 # Target dimension
    Z_train_sim = X_train_sim @ W_sim + np.random.randn(n_train, dim_target).astype(np.float32) * 0.1

    X_val_sim = np.random.rand(n_val, dim_fmri).astype(np.float32)
    Z_val_sim = X_val_sim @ W_sim + np.random.randn(n_val, dim_target).astype(np.float32) * 0.1

    X_test_sim = np.random.rand(n_test, dim_fmri).astype(np.float32)
    Z_test_sim = X_test_sim @ W_sim + np.random.randn(n_test, dim_target).astype(np.float32) * 0.1 # Ground truth for test simulation

    print(f"Simulating data for {fmri_model_name} -> {emb_model_name} ({'PCA' if config.USE_PCA_TARGET else 'Full'} Target):")
    print(f"X_train: {X_train_sim.shape}, Z_train: {Z_train_sim.shape}")
    print(f"X_val: {X_val_sim.shape}, Z_val: {Z_val_sim.shape}")
    print(f"X_test: {X_test_sim.shape}, Z_test (true): {Z_test_sim.shape}")

    # --- Test MLP ---
    print("\n--- Testing MLP ---")
    config.MAPPING_MODEL_TYPE="mlp" # Ensure testing uses MLP path
    try:
        # Train MLP
        mlp_model, mlp_path = train_mlp_mapping(
            X_train_sim, Z_train_sim, X_val_sim, Z_val_sim,
            emb_model_name, fmri_model_name # Pass both model names
        )

        if mlp_model and mlp_path:
            # Load MLP
            loaded_mlp = load_mlp_model(mlp_path)

            if loaded_mlp:
                # Predict with MLP
                Z_pred_mlp = predict_embeddings_mlp(loaded_mlp, X_test_sim)

                if Z_pred_mlp is not None:
                    print(f"MLP Predicted test embeddings shape: {Z_pred_mlp.shape}")
                    # Evaluate MLP prediction
                    print("Evaluating MLP Prediction on Test Set:")
                    evaluate_prediction(Z_test_sim, Z_pred_mlp, dataset_name="Test")
            else:
                 print("MLP loading failed.")
        else:
             print("MLP training or saving failed.")

    except Exception as e:
        print(f"\nAn error occurred during MLP test: {e}")
        traceback.print_exc()

    # --- Test Ridge (Optional comparison) ---
    print("\n--- Testing Ridge (Comparison) ---")
    config.MAPPING_MODEL_TYPE="ridge" # Ensure testing uses Ridge path
    try:
        # Train Ridge
        ridge_model, ridge_path = train_ridge_mapping(
            X_train_sim, Z_train_sim, # Train on Z_train_sim (could be PCA or full)
            config.RIDGE_ALPHA, config.RIDGE_MAX_ITER,
            emb_model_name, fmri_model_name # Pass both model names
        )

        if ridge_model and ridge_path:
            # Load Ridge
            loaded_ridge = load_ridge_model(ridge_path)

            if loaded_ridge:
                # Predict with Ridge
                Z_pred_ridge = predict_embeddings_ridge(loaded_ridge, X_test_sim)
                if Z_pred_ridge is not None:
                    print(f"Ridge Predicted test embeddings shape: {Z_pred_ridge.shape}")
                    # Evaluate Ridge prediction
                    print("Evaluating Ridge Prediction on Test Set:")
                    evaluate_prediction(Z_test_sim, Z_pred_ridge, dataset_name="Test") # Compare against Z_test_sim
            else:
                 print("Ridge loading failed.")
        else:
            print("Ridge training or saving failed.")

    except Exception as e:
        print(f"\nAn error occurred during Ridge test: {e}")
        traceback.print_exc()


    print("\n--- Mapping Models Test Complete ---")
