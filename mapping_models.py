# mapping_models.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm
import copy # For saving best model state

import config

# === Ridge Regression Functions (Keep for comparison/fallback) ===

def train_ridge_mapping(X_train, Z_train, alpha, max_iter, model_name):
    """Trains a Ridge regression model and saves it."""
    print(f"Training Ridge model (alpha={alpha}) for {model_name}...")
    start_time = time.time()
    try:
        ridge = Ridge(alpha=alpha, max_iter=max_iter, random_state=config.RANDOM_STATE, solver='auto')
        ridge.fit(X_train, Z_train)
    except Exception as e:
         print(f"Error fitting Ridge model: {e}")
         return None, None

    # Evaluate on training data (optional, sanity check)
    try:
        Z_train_pred = ridge.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(Z_train, Z_train_pred))
        train_r2 = r2_score(Z_train, Z_train_pred)
        end_time = time.time()
        print(f"Ridge training complete ({end_time - start_time:.2f}s). Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")
    except Exception as e:
        print(f"Error evaluating Ridge on training data: {e}")
        return ridge, None # Return model but no path if eval fails

    # Save the trained model
    model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{alpha}.sav")
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(ridge, f)
        print(f"Saved trained Ridge model to: {model_filename}")
    except Exception as e:
        print(f"Error saving Ridge model: {e}")
        model_filename = None # Indicate saving failed

    return ridge, model_filename

def load_ridge_model(model_filename):
    """Loads a saved Ridge model."""
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Ridge model file not found: {model_filename}")
    try:
        with open(model_filename, 'rb') as f:
            ridge = pickle.load(f)
        print(f"Loaded Ridge model from: {model_filename}")
        return ridge
    except Exception as e:
         print(f"Error loading Ridge model from {model_filename}: {e}")
         return None

def predict_embeddings_ridge(ridge_model, X_data):
    """Predicts embeddings using the loaded Ridge model."""
    print(f"Predicting embeddings using Ridge for {X_data.shape[0]} samples...")
    start_time = time.time()
    try:
        Z_pred = ridge_model.predict(X_data)
        end_time = time.time()
        print(f"Ridge prediction complete ({end_time-start_time:.2f}s).")
        return Z_pred
    except Exception as e:
        print(f"Error during Ridge prediction: {e}")
        return None

# === Deep Residual SE MLP Components ===

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for MLPs"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pool along feature dim if input is (batch, channel, 1)? No, for MLP it's (batch, features)
        # Let's adapt SE for MLP features: operate directly on the feature dimension
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x: (batch_size, features)
        # We want weights per feature: (batch_size, features)
        y = self.fc(x) # Directly apply FC layers to learn weights per feature
        return x * y # Multiply input by learned weights

class ResidualSEBlock(nn.Module):
    """Residual Block with optional LayerNorm and Squeeze-and-Excitation"""
    def __init__(self, input_dim, output_dim, hidden_dim=None, use_layer_norm=True, dropout_rate=0.1, activation=nn.GELU, use_se=True, se_reduction=16):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim # Or input_dim? Let's use output_dim
        self.use_layer_norm = use_layer_norm
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_se = use_se

        # Layer Norm before Linear is often more stable (Pre-activation style)
        self.norm1 = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if use_se:
            self.se = SEBlock(output_dim, reduction=se_reduction)

        # Shortcut connection: Handle dimension mismatch
        if input_dim == output_dim:
            self.shortcut = nn.Identity()
        else:
            # Option 1: Projection shortcut
            self.shortcut = nn.Linear(input_dim, output_dim, bias=False)
            # Option 2: Zero-padding (less common in MLPs)
            # print(f"Warning: Residual block input dim ({input_dim}) != output dim ({output_dim}). Using linear projection shortcut.")

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.norm1(x)
        out = self.activation(out)
        out = self.fc1(out)
        out = self.dropout(out) # Dropout after first fc

        out = self.norm2(out)
        out = self.activation(out)
        out = self.fc2(out)
        # SE block applied *after* second linear layer, before residual add
        if self.use_se:
            out = self.se(out)

        out = self.dropout(out) # Dropout after second fc/SE

        out += shortcut
        return out

class DeepResidualSEMLP(nn.Module):
    """Deep MLP with Residual Blocks and optional SE"""
    def __init__(self, input_dim, output_dim, hidden_dims, use_layer_norm=True, dropout_rate=0.1, activation=nn.GELU, use_se=True, se_reduction=16):
        super().__init__()

        self.initial_layer = nn.Linear(input_dim, hidden_dims[0])
        self.initial_norm = nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity()
        self.initial_activation = activation()
        self.initial_dropout = nn.Dropout(dropout_rate)

        layers = []
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            layers.append(ResidualSEBlock(
                input_dim=current_dim,
                output_dim=h_dim,
                hidden_dim=h_dim, # Keep hidden dim same as output in block for simplicity
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
                activation=activation,
                use_se=use_se,
                se_reduction=se_reduction
            ))
            current_dim = h_dim # Output dim becomes input for next block

        self.residual_layers = nn.Sequential(*layers)

        self.final_norm = nn.LayerNorm(current_dim) if use_layer_norm else nn.Identity()
        self.final_activation = activation()
        self.final_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        x = self.initial_layer(x)
        # Apply norm/act/dropout after initial projection? Or let first residual block handle it?
        # Let's apply them here for consistency before blocks
        x = self.initial_norm(x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)

        x = self.residual_layers(x)

        x = self.final_norm(x)
        x = self.final_activation(x)
        x = self.final_layer(x)
        return x

# === MLP Training and Prediction Functions ===

def train_mlp_mapping(X_train, Z_train, X_val, Z_val, model_name, embedding_dim):
    """Trains the DeepResidualSEMLP model."""
    device = config.DEVICE
    n_voxels = X_train.shape[1]

    # --- Create Model ---
    model = DeepResidualSEMLP(
        input_dim=n_voxels,
        output_dim=embedding_dim,
        hidden_dims=config.MLP_HIDDEN_DIMS,
        use_layer_norm=config.MLP_USE_LAYER_NORM,
        dropout_rate=config.MLP_DROPOUT_RATE,
        activation=config.MLP_ACTIVATION,
        use_se=config.MLP_USE_SE_BLOCK,
        se_reduction=config.MLP_SE_REDUCTION
    ).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLP Model Parameters: {total_params:,}")

    # --- Create DataLoaders for Mapping Training ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Z_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.MAPPING_BATCH_SIZE, shuffle=True, num_workers=config.MAPPING_NUM_WORKERS, pin_memory=True)

    val_loader = None
    if X_val is not None and Z_val is not None and len(X_val) > 0:
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Z_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=config.MAPPING_BATCH_SIZE * 2, shuffle=False, num_workers=config.MAPPING_NUM_WORKERS, pin_memory=True) # Larger batch size for validation
        print(f"Using Validation set with {len(X_val)} samples.")
    else:
        print("No Validation set provided or it's empty. Training without validation-based early stopping or best model saving.")


    # --- Optimizer and Scheduler ---
    if config.MLP_OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY)
    elif config.MLP_OPTIMIZER.lower() == 'adam':
         optimizer = optim.Adam(model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=0) # Adam doesn't handle wd correctly like AdamW
    else:
         print(f"Warning: Unknown optimizer '{config.MLP_OPTIMIZER}'. Using AdamW.")
         optimizer = optim.AdamW(model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY)

    scheduler = None
    if config.MLP_SCHEDULER == 'CosineAnnealingLR':
         scheduler = CosineAnnealingLR(optimizer, T_max=config.MLP_SCHEDULER_T_MAX, eta_min=1e-7) # Use a small eta_min
         print(f"Using CosineAnnealingLR scheduler with T_max={config.MLP_SCHEDULER_T_MAX}")
    # Add other schedulers here if needed (e.g., ReduceLROnPlateau)

    criterion = nn.MSELoss()

    # --- Training Loop ---
    best_val_r2 = -np.inf # Initialize with a very low value
    epochs_no_improve = 0
    best_model_state = None
    model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{model_name}_best.pt")

    print(f"\n--- Starting MLP Training for {config.MLP_EPOCHS} epochs ---")
    train_start_time = time.time()

    for epoch in range(config.MLP_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        for fmri_batch, target_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.MLP_EPOCHS} [Train]", leave=False):
            fmri_batch, target_batch = fmri_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()
            output_batch = model(fmri_batch)
            loss = criterion(output_batch, target_batch)
            total_train_loss += loss.item() * fmri_batch.size(0) # Accumulate loss scaled by batch size

            loss.backward()

            # Optional Gradient Clipping
            if config.MLP_GRAD_CLIP_VALUE > 0:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), config.MLP_GRAD_CLIP_VALUE)

            optimizer.step()

            # Store predictions and targets for R2 calculation after epoch
            all_train_preds.append(output_batch.detach().cpu().numpy())
            all_train_targets.append(target_batch.detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_r2 = -1.0 # Default if calculation fails
        try:
             all_train_preds_np = np.concatenate(all_train_preds, axis=0)
             all_train_targets_np = np.concatenate(all_train_targets, axis=0)
             train_r2 = r2_score(all_train_targets_np, all_train_preds_np)
        except ValueError as e:
             print(f"Warning: Could not compute train R2 score for epoch {epoch+1}: {e}")


        # --- Validation Phase ---
        val_r2 = -1.0 # Default if no validation
        avg_val_loss = float('nan')
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            all_val_preds = []
            all_val_targets = []
            with torch.no_grad():
                for fmri_batch, target_batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.MLP_EPOCHS} [Val]", leave=False):
                    fmri_batch, target_batch = fmri_batch.to(device), target_batch.to(device)
                    output_batch = model(fmri_batch)
                    loss = criterion(output_batch, target_batch)
                    total_val_loss += loss.item() * fmri_batch.size(0)

                    all_val_preds.append(output_batch.cpu().numpy())
                    all_val_targets.append(target_batch.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            try:
                 all_val_preds_np = np.concatenate(all_val_preds, axis=0)
                 all_val_targets_np = np.concatenate(all_val_targets, axis=0)
                 val_r2 = r2_score(all_val_targets_np, all_val_preds_np)
            except ValueError as e:
                 print(f"Warning: Could not compute validation R2 score for epoch {epoch+1}: {e}")
                 val_r2 = -1.0 # Set to low value if calculation fails


        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{config.MLP_EPOCHS} | Time: {epoch_duration:.2f}s | LR: {current_lr:.1e} | "
              f"Train Loss: {avg_train_loss:.4f} | Train R2: {train_r2:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val R2: {val_r2:.4f}")

        # Learning Rate Scheduler Step (after validation)
        if scheduler:
             # Some schedulers like ReduceLROnPlateau need metrics
             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                  scheduler.step(avg_val_loss) # Or val_r2? Depends on mode
             else:
                  scheduler.step()

        # --- Early Stopping & Best Model Saving (based on Validation R2) ---
        if val_loader: # Only if we have validation data
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                epochs_no_improve = 0
                # Save the best model state
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, model_filename)
                print(f"** Validation R2 improved to {best_val_r2:.4f}. Saved model to {model_filename} **")
            else:
                epochs_no_improve += 1
                print(f"Validation R2 did not improve for {epochs_no_improve} epoch(s). Best: {best_val_r2:.4f}")

            if epochs_no_improve >= config.MLP_EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {config.MLP_EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break
        else:
             # If no validation, save the last epoch model? Or save periodically?
             # Let's save the last one for now if no validation exists.
             best_model_state = copy.deepcopy(model.state_dict())
             torch.save(best_model_state, model_filename)
             # print(f"Saved model state from epoch {epoch+1} (no validation set).")


    train_end_time = time.time()
    print(f"--- MLP Training Finished ({train_end_time - train_start_time:.2f}s) ---")

    # Load the best model state found during training (if validation was used)
    if best_model_state is not None:
        print(f"Loading best model state (Val R2: {best_val_r2:.4f}) from {model_filename}")
        model.load_state_dict(best_model_state)
    elif os.path.exists(model_filename):
         print(f"Loading model from last epoch (no validation used) from {model_filename}")
         model.load_state_dict(torch.load(model_filename, map_location=device))
    else:
         print("Warning: No best model state found or saved file exists. Using model from final training state.")


    return model, model_filename


def load_mlp_model(model_filename, n_voxels, embedding_dim):
    """Loads a saved MLP model state dict."""
    device = config.DEVICE
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"MLP model file not found: {model_filename}")

    # Recreate the model structure
    model = DeepResidualSEMLP(
        input_dim=n_voxels,
        output_dim=embedding_dim,
        hidden_dims=config.MLP_HIDDEN_DIMS,
        use_layer_norm=config.MLP_USE_LAYER_NORM,
        dropout_rate=config.MLP_DROPOUT_RATE,
        activation=config.MLP_ACTIVATION,
        use_se=config.MLP_USE_SE_BLOCK,
        se_reduction=config.MLP_SE_REDUCTION
    ).to(device)

    try:
        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.eval() # Set to evaluation mode
        print(f"Loaded MLP model state from: {model_filename}")
        return model
    except Exception as e:
        print(f"Error loading MLP model state from {model_filename}: {e}")
        return None


def predict_embeddings_mlp(mlp_model, X_data):
    """Predicts embeddings using the loaded MLP model, handling batching."""
    device = config.DEVICE
    mlp_model.eval() # Ensure evaluation mode

    print(f"Predicting embeddings using MLP for {X_data.shape[0]} samples...")
    start_time = time.time()

    # Create a DataLoader for the prediction data
    pred_dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    # Use a larger batch size for prediction if memory allows
    pred_loader = DataLoader(pred_dataset, batch_size=config.MAPPING_BATCH_SIZE * 4, shuffle=False, num_workers=0) # Workers=0 might be safer here

    all_preds = []
    with torch.no_grad():
        for batch in tqdm.tqdm(pred_loader, desc="Predicting", leave=False):
            fmri_batch = batch[0].to(device) # DataLoader wraps tensor in a list/tuple
            output_batch = mlp_model(fmri_batch)
            all_preds.append(output_batch.cpu().numpy())

    try:
        Z_pred = np.concatenate(all_preds, axis=0)
        end_time = time.time()
        print(f"MLP prediction complete ({end_time-start_time:.2f}s). Output shape: {Z_pred.shape}")
        return Z_pred
    except ValueError: # Handles case where prediction failed for all batches
         print("Error: MLP prediction resulted in empty output.")
         return None
    except Exception as e:
        print(f"Error during MLP prediction concatenation: {e}")
        return None


# --- General Evaluation Function (Can be used for both models) ---
def evaluate_prediction(Z_true, Z_pred):
    """Calculates RMSE and R2 score for predictions."""
    if Z_true is None or Z_pred is None:
         print("Cannot evaluate prediction: True or Pred is None.")
         return np.nan, np.nan
    if Z_true.shape != Z_pred.shape:
         print(f"Cannot evaluate prediction: Shape mismatch! True={Z_true.shape}, Pred={Z_pred.shape}")
         return np.nan, np.nan
         
    try:
        # Ensure NaNs in prediction don't break calculation (replace with mean?)
        # Or just let sklearn handle it, R2 might become very bad which is correct
        if np.isnan(Z_pred).any():
             print(f"Warning: Predictions contain NaNs ({np.isnan(Z_pred).sum()} values). R2 score might be affected.")
             # Option: Z_pred = np.nan_to_num(Z_pred, nan=np.nanmean(Z_pred)) # Replace NaN with mean

        rmse = np.sqrt(mean_squared_error(Z_true, Z_pred))
        r2 = r2_score(Z_true, Z_pred)
        print(f"Prediction Evaluation - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        return rmse, r2
    except Exception as e:
         print(f"Error during prediction evaluation: {e}")
         return np.nan, np.nan


# --- Example Usage (for testing this module) ---
if __name__ == "__main__":
    print("--- Testing Mapping Models (MLP Focus) ---")

    model_name_test = "resnet50" # Example model for embedding dim
    dim_fmri = 4000 # Example dimensionality for fMRI
    dim_embedding = config.EMBEDDING_MODELS[model_name_test]['embedding_dim']
    n_train = 2000
    n_val = 200
    n_test = 100

    # Simulate data
    print("Simulating data...")
    X_train_sim = np.random.rand(n_train, dim_fmri).astype(np.float32)
    # Simulate a non-linear relationship
    weights_sim = np.random.rand(dim_fmri, dim_embedding).astype(np.float32) * 0.1
    Z_train_sim = np.tanh(X_train_sim @ weights_sim) + np.random.normal(0, 0.1, size=(n_train, dim_embedding)).astype(np.float32)

    X_val_sim = np.random.rand(n_val, dim_fmri).astype(np.float32)
    Z_val_sim = np.tanh(X_val_sim @ weights_sim) + np.random.normal(0, 0.1, size=(n_val, dim_embedding)).astype(np.float32)

    X_test_sim = np.random.rand(n_test, dim_fmri).astype(np.float32)
    Z_test_true_sim = np.tanh(X_test_sim @ weights_sim) + np.random.normal(0, 0.1, size=(n_test, dim_embedding)).astype(np.float32)

    print(f"Simulated data shapes:")
    print(f"Train: X={X_train_sim.shape}, Z={Z_train_sim.shape}")
    print(f"Val:   X={X_val_sim.shape}, Z={Z_val_sim.shape}")
    print(f"Test:  X={X_test_sim.shape}, Z_true={Z_test_true_sim.shape}")

    try:
        print("\n--- Testing MLP Training ---")
        # Temporarily reduce epochs for quick testing
        original_epochs = config.MLP_EPOCHS
        config.MLP_EPOCHS = 5 # Low epoch count for testing
        mlp_model, model_path = train_mlp_mapping(
            X_train_sim, Z_train_sim, X_val_sim, Z_val_sim,
            model_name_test + "_sim", dim_embedding
        )
        config.MLP_EPOCHS = original_epochs # Restore config value

        if mlp_model and model_path:
            print("\n--- Testing MLP Loading ---")
            loaded_mlp = load_mlp_model(model_path, dim_fmri, dim_embedding)

            if loaded_mlp:
                print("\n--- Testing MLP Prediction ---")
                Z_pred_test_mlp = predict_embeddings_mlp(loaded_mlp, X_test_sim)

                print("\n--- Evaluating MLP Prediction ---")
                evaluate_prediction(Z_test_true_sim, Z_pred_test_mlp)
            else:
                 print("MLP Loading failed.")
        else:
             print("MLP Training failed.")

    except Exception as e:
        print(f"\nAn error occurred during MLP mapping model test: {e}")
        import traceback
        traceback.print_exc()

    # You could add a similar test block for Ridge here if needed

    print("\n--- Mapping Models Test Complete ---")