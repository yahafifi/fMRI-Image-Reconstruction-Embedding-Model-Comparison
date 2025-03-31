# mapping_models.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge # Optional: for KRR alternative
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import config # Assuming config.py is in the same directory or path

# --- Configuration for MLP (Add these or move to config.py) ---
MLP_HIDDEN_LAYERS = [2048, 2048] # Sizes of hidden layers
MLP_DROPOUT_RATE = 0.3         # Dropout rate
MLP_LEARNING_RATE = 1e-4       # AdamW learning rate
MLP_WEIGHT_DECAY = 1e-5        # AdamW weight decay
MLP_EPOCHS = 100               # Number of training epochs
MLP_BATCH_SIZE = 64            # Batch size for MLP training
MLP_EARLY_STOPPING_PATIENCE = 10 # Stop if validation loss doesn't improve for N epochs

# --- Ridge Regression Functions (Keep for comparison) ---

def train_ridge_mapping(X_train, Z_train, alpha, max_iter, model_name):
    """Trains a Ridge regression model and saves it."""
    print(f"Training Ridge model (alpha={alpha}) for {model_name}...")
    # Optional: Scale fMRI data for Ridge
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=alpha, max_iter=max_iter, random_state=config.RANDOM_STATE)
    # ridge.fit(X_train_scaled, Z_train)
    ridge.fit(X_train, Z_train) # Fit on original for direct comparison start

    Z_train_pred = ridge.predict(X_train) # Predict on original
    train_rmse = np.sqrt(mean_squared_error(Z_train, Z_train_pred))
    train_r2 = r2_score(Z_train, Z_train_pred)
    print(f"Ridge Training complete. Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")

    model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{alpha}.sav")
    # Save model and scaler if used
    # save_payload = {'model': ridge, 'scaler': scaler}
    save_payload = {'model': ridge, 'scaler': None} # Save None if no scaler used
    with open(model_filename, 'wb') as f:
        pickle.dump(save_payload, f)
    print(f"Saved trained Ridge model to: {model_filename}")
    return ridge, model_filename # Return only model for consistency, scaler loaded separately if needed

def load_ridge_model(model_filename):
    """Loads a saved Ridge model (and optional scaler)."""
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Ridge model file not found: {model_filename}")
    with open(model_filename, 'rb') as f:
        payload = pickle.load(f)
    ridge_model = payload['model']
    # scaler = payload.get('scaler', None) # Get scaler, default to None if not found
    print(f"Loaded Ridge model from: {model_filename}")
    # return ridge_model, scaler
    return ridge_model # Return only model for consistency

def predict_embeddings_ridge(ridge_model, X_data): # Renamed for clarity
    """Predicts embeddings using the loaded Ridge model."""
    print(f"Predicting embeddings using Ridge for {X_data.shape[0]} samples...")
    # if scaler:
    #     X_data_scaled = scaler.transform(X_data)
    #     Z_pred = ridge_model.predict(X_data_scaled)
    # else:
    Z_pred = ridge_model.predict(X_data) # Predict on original data
    print("Ridge Prediction complete.")
    return Z_pred

# --- MLP Regression Functions ---

class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(MLPRegression, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim)) # Optional: BatchNorm
            layers.append(nn.ReLU()) # Or nn.GELU()
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim)) # Output layer (no activation for regression)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_mlp_mapping(X_train, Z_train, X_val, Z_val, model_name, embedding_dim):
    """Trains an MLP regression model with validation and early stopping."""

    input_dim = X_train.shape[1]
    output_dim = embedding_dim
    device = config.DEVICE

    print(f"Training MLP model for {model_name}...")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Hidden layers: {MLP_HIDDEN_LAYERS}, Dropout: {MLP_DROPOUT_RATE}")
    print(f"LR: {MLP_LEARNING_RATE}, Weight Decay: {MLP_WEIGHT_DECAY}, Epochs: {MLP_EPOCHS}, Batch Size: {MLP_BATCH_SIZE}")

    # --- Data Preparation ---
    # Scale fMRI data (important for NNs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val.size > 0 else X_val # Scale val set if it exists

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(Z_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)

    if X_val.size > 0:
        val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                    torch.tensor(Z_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=MLP_BATCH_SIZE * 2) # Can use larger batch for validation
    else:
        val_loader = None
        print("No validation set provided. Training without validation-based early stopping.")


    # --- Model, Loss, Optimizer ---
    model = MLPRegression(input_dim, output_dim, MLP_HIDDEN_LAYERS, MLP_DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    model_save_path = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{model_name}_best.pt")
    scaler_save_path = os.path.join(config.MODELS_BASE_PATH, f"mlp_scaler_{model_name}.sav")


    for epoch in range(MLP_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        start_epoch_time = time.time()

        for fmri_batch, target_batch in train_loader:
            fmri_batch, target_batch = fmri_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(fmri_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * fmri_batch.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # Validation Step
        avg_val_loss = float('inf') # Default if no validation
        if val_loader:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for fmri_batch, target_batch in val_loader:
                    fmri_batch, target_batch = fmri_batch.to(device), target_batch.to(device)
                    outputs = model(fmri_batch)
                    loss = criterion(outputs, target_batch)
                    epoch_val_loss += loss.item() * fmri_batch.size(0)
            avg_val_loss = epoch_val_loss / len(val_loader.dataset)
            history['val_loss'].append(avg_val_loss)

            print(f"Epoch [{epoch+1}/{MLP_EPOCHS}] | Time: {time.time()-start_epoch_time:.2f}s | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # Early Stopping and Model Saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"   Validation loss improved. Saved model to {model_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"   Validation loss did not improve for {epochs_no_improve} epochs.")
                if epochs_no_improve >= MLP_EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after epoch {epoch+1}.")
                    break
        else: # No validation set
             print(f"Epoch [{epoch+1}/{MLP_EPOCHS}] | Time: {time.time()-start_epoch_time:.2f}s | Train Loss: {avg_train_loss:.6f}")
             # Save model from last epoch if no validation
             if epoch == MLP_EPOCHS - 1:
                  torch.save(model.state_dict(), model_save_path)
                  print(f"Saved model from final epoch to {model_save_path}")


    # Save the scaler
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved fMRI scaler to: {scaler_save_path}")

    print("MLP Training finished.")
    # Load the best model found during training
    model.load_state_dict(torch.load(model_save_path))
    return model, scaler, model_save_path # Return trained model, scaler, and path to best weights

def load_mlp_model(model_name, embedding_dim, input_dim):
    """Loads the best saved MLP model state and the corresponding scaler."""
    device = config.DEVICE
    model_load_path = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{model_name}_best.pt")
    scaler_load_path = os.path.join(config.MODELS_BASE_PATH, f"mlp_scaler_{model_name}.sav")

    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"MLP model file not found: {model_load_path}")
    if not os.path.exists(scaler_load_path):
        raise FileNotFoundError(f"MLP scaler file not found: {scaler_load_path}")

    # Instantiate model architecture
    model = MLPRegression(input_dim, embedding_dim, MLP_HIDDEN_LAYERS, MLP_DROPOUT_RATE)
    # Load state dict
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model = model.to(device)
    model.eval() # Set to evaluation mode

    # Load scaler
    with open(scaler_load_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Loaded MLP model state from: {model_load_path}")
    print(f"Loaded fMRI scaler from: {scaler_load_path}")

    return model, scaler

@torch.no_grad()
def predict_embeddings_mlp(mlp_model, scaler, X_data):
    """Predicts embeddings using the loaded MLP model and scaler."""
    device = config.DEVICE
    mlp_model.eval()
    print(f"Predicting embeddings using MLP for {X_data.shape[0]} samples...")

    # Scale input data using the loaded scaler
    X_data_scaled = scaler.transform(X_data)
    X_data_tensor = torch.tensor(X_data_scaled, dtype=torch.float32).to(device)

    # Predict in batches to avoid OOM for large test sets
    all_preds = []
    pred_loader = DataLoader(TensorDataset(X_data_tensor), batch_size=MLP_BATCH_SIZE * 4) # Larger batch for inference

    for batch in pred_loader:
        fmri_batch = batch[0] # DataLoader wraps tensor in a tuple/list
        outputs = mlp_model(fmri_batch)
        all_preds.append(outputs.cpu().numpy())

    Z_pred = np.concatenate(all_preds, axis=0)
    print("MLP Prediction complete.")
    return Z_pred

# --- Common Evaluation Function (Can be used for Ridge or MLP preds) ---
def evaluate_embedding_prediction(Z_true, Z_pred):
    """Calculates RMSE and R2 score for embedding predictions."""
    if Z_true.shape[0] == 0 or Z_pred.shape[0] == 0:
        print("Cannot evaluate empty arrays.")
        return np.nan, np.nan
    if Z_true.shape != Z_pred.shape:
         print(f"Shape mismatch! True: {Z_true.shape}, Pred: {Z_pred.shape}. Cannot evaluate.")
         return np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(Z_true, Z_pred))
    r2 = r2_score(Z_true, Z_pred)

    # Calculate average cosine similarity (often insightful for embeddings)
    cos_sim = np.mean([np.dot(zt, zp) / (np.linalg.norm(zt) * np.linalg.norm(zp))
                       for zt, zp in zip(Z_true, Z_pred) if np.linalg.norm(zt) > 1e-6 and np.linalg.norm(zp) > 1e-6]) # Avoid div by zero

    print(f"Embedding Prediction Eval - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}, Cosine Sim: {cos_sim:.4f}")
    return rmse, r2, cos_sim

# --- Example Usage (Updated) ---
if __name__ == "__main__":
    print("--- Testing Mapping Models ---")

    model_name_test = "resnet50" # Example model
    # Need input_dim (n_voxels) and output_dim (embedding_dim)
    # These would normally come from loaded data
    n_voxels_sim = 4000
    embedding_dim_sim = config.EMBEDDING_MODELS[model_name_test]['embedding_dim']
    n_train = 1000
    n_val = 100
    n_test = 50

    # Simulate data
    print("Simulating data...")
    X_train_sim = np.random.rand(n_train, n_voxels_sim).astype(np.float32)
    Z_train_sim = np.random.rand(n_train, embedding_dim_sim).astype(np.float32)
    X_val_sim = np.random.rand(n_val, n_voxels_sim).astype(np.float32)
    Z_val_sim = np.random.rand(n_val, embedding_dim_sim).astype(np.float32)
    X_test_sim = np.random.rand(n_test, n_voxels_sim).astype(np.float32)
    Z_test_sim = np.random.rand(n_test, embedding_dim_sim).astype(np.float32) # Ground truth for test

    print(f"Simulated data shapes: X_train {X_train_sim.shape}, Z_train {Z_train_sim.shape}, X_val {X_val_sim.shape}, X_test {X_test_sim.shape}")

    # --- Test Ridge ---
    print("\n--- Testing Ridge ---")
    try:
        ridge_model, _ = train_ridge_mapping(X_train_sim, Z_train_sim, config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name_test)
        loaded_ridge = load_ridge_model(os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name_test}_alpha{config.RIDGE_ALPHA}.sav"))
        Z_pred_test_ridge = predict_embeddings_ridge(loaded_ridge, X_test_sim)
        evaluate_embedding_prediction(Z_test_sim, Z_pred_test_ridge)
    except Exception as e: print(f"Ridge test error: {e}")

    # --- Test MLP ---
    print("\n--- Testing MLP ---")
    try:
        mlp_model, scaler, model_path = train_mlp_mapping(X_train_sim, Z_train_sim, X_val_sim, Z_val_sim, model_name_test, embedding_dim_sim)
        loaded_mlp, loaded_scaler = load_mlp_model(model_name_test, embedding_dim_sim, n_voxels_sim)
        Z_pred_test_mlp = predict_embeddings_mlp(loaded_mlp, loaded_scaler, X_test_sim)
        evaluate_embedding_prediction(Z_test_sim, Z_pred_test_mlp)
    except Exception as e:
        print(f"MLP test error: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Mapping Models Test Complete ---")
