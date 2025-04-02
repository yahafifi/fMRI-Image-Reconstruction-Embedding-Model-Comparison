# mapping_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pickle
from sklearn.linear_model import Ridge # Keep Ridge imports
from sklearn.metrics import mean_squared_error, r2_score
import tqdm # For progress bars

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    """ Squeeze-and-Excitation block for channel-wise attention. """
    def __init__(self, channel, reduction=16):
        super().__init__()
        # AdaptiveAvgPool1d treats features as channels and averages over the 'length' dim (which is 1 here)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x shape: (batch, features)
        b, c = x.size()
        # Add pseudo-length dim for pooling: (b, c) -> (b, c, 1)
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c) # Squeeze: pool -> view removes length dim -> (b, c)
        y = self.fc(y) # Excitation -> (b, c)
        # Rescale original input: (b, c) * (b, c)
        return x * y


# --- Residual Block with SE ---
class ResidualBlock(nn.Module):
    """ Residual block containing Linear layers, Norm, Activation, Dropout, SE, and Skip Connection. """
    def __init__(self, hidden_dim, dropout_rate, se_reduction_ratio):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Normalize features before activation
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            # No ReLU right before SE block based on common implementations
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate) # Dropout before SE might be too aggressive
        )
        self.se_block = SEBlock(hidden_dim, reduction=se_reduction_ratio)
        self.dropout = nn.Dropout(dropout_rate) # Apply dropout after SE block and residual connection
        self.relu = nn.ReLU(inplace=True) # Activation after adding residual

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.se_block(out)
        out = out + residual # Add skip connection FIRST
        out = self.relu(out)   # Apply activation AFTER skip connection
        out = self.dropout(out) # Apply dropout AFTER activation and skip connection
        return out


# --- Advanced MLP Model ---
class AdvancedMLP(nn.Module):
    """ Deep MLP with Residual Blocks and Squeeze-and-Excitation. """
    def __init__(self, fmri_dim, embedding_dim, hidden_dim, num_blocks, dropout_rate, se_reduction_ratio):
        super().__init__()

        # Initial projection layer
        self.initial_layer = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Stack of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate, se_reduction_ratio) for _ in range(num_blocks)]
        )

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, embedding_dim)

        # Optional: Initialize weights (can sometimes help)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x


# --- MLP Training Function (Updated) ---
def train_mlp_mapping(X_train, Z_train, X_val, Z_val, model_name, device, cfg): # Pass config object
    """ Trains the AdvancedMLP mapping model with scheduling, gradient clipping, and early stopping. """
    fmri_dim = X_train.shape[1]
    embedding_dim = Z_train.shape[1]

    # --- Hyperparameters from Config ---
    hidden_dim = cfg.MLP_HIDDEN_DIM
    num_blocks = cfg.MLP_NUM_RESIDUAL_BLOCKS
    se_reduction = cfg.MLP_SE_REDUCTION_RATIO
    dropout_rate = cfg.MLP_DROPOUT_RATE
    learning_rate = cfg.MLP_LEARNING_RATE
    weight_decay = cfg.MLP_WEIGHT_DECAY
    epochs = cfg.MLP_EPOCHS
    batch_size = cfg.MLP_BATCH_SIZE
    grad_clip_norm = cfg.MLP_GRADIENT_CLIP_NORM
    scheduler_type = cfg.MLP_SCHEDULER_TYPE
    early_stop_patience = cfg.MLP_EARLY_STOPPING_PATIENCE

    # --- Model, Optimizer, Loss ---
    model = AdvancedMLP(fmri_dim, embedding_dim, hidden_dim, num_blocks, dropout_rate, se_reduction).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CosineEmbeddingLoss(margin=0.0) # Using Cosine Loss

    # --- Learning Rate Scheduler ---
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MLP_COSINE_T_MAX, eta_min=1e-7)
        print(f"Using Cosine Annealing scheduler (T_max={cfg.MLP_COSINE_T_MAX})")
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.MLP_PLATEAU_FACTOR, patience=cfg.MLP_PLATEAU_PATIENCE, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler (factor={cfg.MLP_PLATEAU_FACTOR}, patience={cfg.MLP_PLATEAU_PATIENCE})")
    else:
        print("No LR scheduler used.")


    # --- DataLoaders ---
    # Using float32 for stability, can experiment with float16 tensors later if needed
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Z_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    val_loader = None
    if X_val is not None and Z_val is not None and len(X_val) > 0:
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Z_val, dtype=torch.float32))
        # Use larger batch size for validation as gradients aren't computed
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("Warning: No validation data provided for MLP training. Early stopping based on validation loss disabled.")
        early_stop_patience = epochs # Effectively disable early stopping

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_save_path = os.path.join(cfg.MODELS_BASE_PATH, f"adv_mlp_mapping_{model_name}.pth")

    print(f"\n--- Training Advanced MLP ({num_blocks} blocks, hidden={hidden_dim}) ---")
    print(f"Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")

    # Use gradient scaler for mixed precision if on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for fmri_batch, embed_batch in pbar:
            fmri_batch = fmri_batch.to(device)
            embed_batch = embed_batch.to(device)

            optimizer.zero_grad(set_to_none=True) # More memory efficient

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                 predictions = model(fmri_batch)
                 # Target tensor for CosineEmbeddingLoss (all ones, same size as batch)
                 target = torch.ones(fmri_batch.size(0), device=device)
                 loss = criterion(predictions, embed_batch, target)

            scaler.scale(loss).backward()

            # Unscales gradients and clips (if enabled) before optimizer step
            if grad_clip_norm > 0:
                 scaler.unscale_(optimizer) # Unscale gradients before clipping
                 torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update() # Update scaler for next iteration

            train_loss_accum += loss.item() # Track loss per item
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss_per_batch = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        current_lr = optimizer.param_groups[0]['lr']
        avg_val_loss = float('inf') # Default if no validation loader

        if val_loader:
            model.eval()
            val_loss_accum = 0.0
            val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            with torch.no_grad():
                for fmri_batch, embed_batch in val_pbar:
                    fmri_batch = fmri_batch.to(device)
                    embed_batch = embed_batch.to(device)
                    # Use autocast for validation consistency
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                         predictions = model(fmri_batch)
                         target = torch.ones(fmri_batch.size(0), device=device)
                         loss = criterion(predictions, embed_batch, target)
                    val_loss_accum += loss.item()
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_val_loss = val_loss_accum / len(val_loader) # Avg loss per batch on validation
            print(f"Epoch {epoch+1}/{epochs} Summary: Train Loss: {avg_train_loss_per_batch:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")

            # --- LR Scheduling & Early Stopping (based on validation loss) ---
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif scheduler is not None:
                scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> Val loss improved. Saved model to {model_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  -> Val loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break
        else: # No validation loader
             print(f"Epoch {epoch+1}/{epochs} Summary: Train Loss: {avg_train_loss_per_batch:.6f} | LR: {current_lr:.1e} (No validation)")
             if scheduler is not None and not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                 scheduler.step()
             # Save model at the end if no validation
             if epoch == epochs - 1:
                  torch.save(model.state_dict(), model_save_path)
                  print(f"Saved model from last epoch to {model_save_path}")


    # --- Load Best Model ---
    if os.path.exists(model_save_path):
         print(f"\nLoading best performing model weights from: {model_save_path}")
         # Initialize model again to ensure correct structure before loading state dict
         final_model = AdvancedMLP(fmri_dim, embedding_dim, hidden_dim, num_blocks, dropout_rate, se_reduction).to(device)
         final_model.load_state_dict(torch.load(model_save_path, map_location=device))
         print("Loaded best weights successfully.")
         return final_model, model_save_path
    else:
         print("\nWarning: No saved model weights found. Returning model state from end of training.")
         return model, model_save_path # Return the model as is


# --- MLP Loading Function (Updated) ---
def load_mlp_model(model_path, fmri_dim, embedding_dim, device, cfg):
     """ Loads the trained AdvancedMLP model. """
     hidden_dim = cfg.MLP_HIDDEN_DIM
     num_blocks = cfg.MLP_NUM_RESIDUAL_BLOCKS
     se_reduction = cfg.MLP_SE_REDUCTION_RATIO
     # Use a default dropout for loading if not specified, but ideally it matches training
     dropout_rate = getattr(cfg, "MLP_DROPOUT_RATE", 0.5)

     # Initialize the model structure first
     model = AdvancedMLP(fmri_dim, embedding_dim, hidden_dim, num_blocks, dropout_rate, se_reduction).to(device)
     try:
          # Load the saved state dictionary
          model.load_state_dict(torch.load(model_path, map_location=device))
          model.eval() # Set to evaluation mode
          print(f"Loaded Advanced MLP model weights from: {model_path}")
          return model
     except FileNotFoundError:
          print(f"Error: Advanced MLP model file not found at {model_path}")
          return None
     except Exception as e:
          print(f"Error loading Advanced MLP model weights: {e}")
          import traceback
          traceback.print_exc()
          return None

# --- MLP Prediction Function (Updated for Autocast) ---
@torch.no_grad()
def predict_embeddings_mlp(mlp_model, X_data, device, batch_size=512): # Increased prediction batch size
    """ Predicts embeddings using the trained Advanced MLP model. """
    if mlp_model is None:
         print("Error: MLP model is None. Cannot predict.")
         return None
    mlp_model.eval() # Ensure evaluation mode
    predictions = []
    X_tensor = torch.tensor(X_data, dtype=torch.float32) # Input data as float32
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Use 0 workers for simple tensor dataset

    print(f"Predicting embeddings for {len(X_data)} samples using Advanced MLP...")
    pbar = tqdm.tqdm(loader, desc="MLP Prediction")
    for (batch,) in pbar:
         batch = batch.to(device)
         # Use autocast during prediction for consistency if trained with it
         with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
              preds = mlp_model(batch)
         # Ensure output is float32 when moving back to CPU/Numpy
         predictions.append(preds.cpu().float().numpy())

    if not predictions:
         print("Error: No predictions were generated.")
         return None

    return np.concatenate(predictions, axis=0)


# --- Ridge Functions (Kept for potential comparison) ---
def train_ridge_mapping(X_train, Z_train, alpha, max_iter, model_name):
    """Trains a Ridge regression model and saves it."""
    print(f"[Ridge] Training Ridge model (alpha={alpha}) for {model_name}...")
    ridge = Ridge(alpha=alpha, max_iter=max_iter, random_state=config.RANDOM_STATE)
    ridge.fit(X_train, Z_train)

    Z_train_pred = ridge.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(Z_train, Z_train_pred))
    train_r2 = r2_score(Z_train, Z_train_pred)
    print(f"[Ridge] Training complete. Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")

    model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{alpha}.sav")
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(ridge, f)
        print(f"[Ridge] Saved trained Ridge model to: {model_filename}")
    except Exception as e:
        print(f"[Ridge] Error saving Ridge model: {e}")

    return ridge, model_filename

def load_ridge_model(model_filename):
    """Loads a saved Ridge model."""
    if not os.path.exists(model_filename):
        print(f"[Ridge] Error: Ridge model file not found: {model_filename}")
        return None # Return None instead of raising error directly
    try:
        with open(model_filename, 'rb') as f:
            ridge = pickle.load(f)
        print(f"[Ridge] Loaded Ridge model from: {model_filename}")
        return ridge
    except Exception as e:
        print(f"[Ridge] Error loading Ridge model: {e}")
        return None


def predict_embeddings_ridge(ridge_model, X_data): # Renamed for clarity
    """Predicts embeddings using the loaded Ridge model."""
    if ridge_model is None:
        print("[Ridge] Error: Ridge model is None.")
        return None
    print(f"[Ridge] Predicting embeddings for {X_data.shape[0]} samples...")
    try:
        Z_pred = ridge_model.predict(X_data)
        print("[Ridge] Prediction complete.")
        return Z_pred
    except Exception as e:
        print(f"[Ridge] Error during prediction: {e}")
        return None


def evaluate_prediction(Z_true, Z_pred):
    """Calculates RMSE and R2 score for predictions."""
    if Z_true is None or Z_pred is None or len(Z_true) != len(Z_pred):
        print("Error: Invalid inputs for evaluation.")
        return np.nan, np.nan
    try:
        # Ensure numpy arrays for calculation
        Z_true_np = np.asarray(Z_true)
        Z_pred_np = np.asarray(Z_pred)

        # Check for NaNs or Infs which can cause errors
        if not np.all(np.isfinite(Z_true_np)) or not np.all(np.isfinite(Z_pred_np)):
            print("Warning: NaN or Inf found in prediction/truth data. Evaluation metrics may be invalid.")
            # Optional: handle NaNs, e.g., by removing corresponding samples or returning NaN
            return np.nan, np.nan

        rmse = np.sqrt(mean_squared_error(Z_true_np, Z_pred_np))
        r2 = r2_score(Z_true_np, Z_pred_np)
        print(f"Evaluation - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        return rmse, r2
    except Exception as e:
        print(f"Error during prediction evaluation: {e}")
        return np.nan, np.nan