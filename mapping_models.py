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

import config

# --- Evaluation Function (Works for both Ridge and MLP) ---
def evaluate_prediction(Z_true, Z_pred):
    """
    Calculates RMSE and R2 score for predictions.
    Handles both numpy arrays and torch tensors (converts tensors to numpy).
    """
    if isinstance(Z_true, torch.Tensor):
        Z_true = Z_true.detach().cpu().numpy()
    if isinstance(Z_pred, torch.Tensor):
        Z_pred = Z_pred.detach().cpu().numpy()

    if Z_true.shape != Z_pred.shape:
         print(f"Warning: Shape mismatch in evaluation. True: {Z_true.shape}, Pred: {Z_pred.shape}")
         # Attempt to evaluate if only batch dim differs (e.g., comparing single pred to single true)
         if Z_true.shape[1:] == Z_pred.shape[1:] and Z_true.shape[0] == 1 and Z_pred.shape[0] == 1:
              pass # Allow evaluation
         else:
              return np.nan, np.nan # Return NaN if shapes are fundamentally incompatible

    try:
        mse = mean_squared_error(Z_true, Z_pred)
        rmse = np.sqrt(mse)
        # R2 score requires at least 2 samples if calculated per feature, or needs multioutput='uniform_average'
        if Z_true.shape[0] > 1:
             r2 = r2_score(Z_true, Z_pred, multioutput='variance_weighted') # More robust R2 for multi-output
        else:
             # R2 is ill-defined for a single sample; calculate relative MSE if needed, or return NaN/0
             r2 = 0.0 # Or np.nan
        print(f"Evaluation - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        return rmse, r2
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return np.nan, np.nan


# === MLP Implementation ===

# 1. MLP Model Definition
class FmriMappingMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu', dropout_rate=0.5):
        super().__init__()
        layers = []
        last_dim = input_dim

        # Activation function mapping
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        # Add more activation functions as needed (e.g., LeakyReLU)
        else:
            print(f"Warning: Unsupported activation '{activation}'. Using ReLU.")
            act_fn = nn.ReLU()

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        # Output layer (no activation or dropout usually)
        layers.append(nn.Linear(last_dim, output_dim))

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

        # Convert to tensors
        self.fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)
        self.embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)

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
    output_dim = Z_train.shape[1]

    # Create datasets and dataloaders
    train_dataset = FmriEmbeddingDataset(X_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    if X_val.size > 0 and Z_val.size > 0: # Check if validation data exists
         val_dataset = FmriEmbeddingDataset(X_val, Z_val)
         val_loader = DataLoader(val_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
         print(f"Training with {len(train_dataset)} samples, Validating with {len(val_dataset)} samples.")
    else:
         val_loader = None
         print(f"Training with {len(train_dataset)} samples. No validation set provided.")


    # Initialize model, loss, optimizer, scheduler
    model = FmriMappingMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=config.MLP_HIDDEN_LAYERS,
        activation=config.MLP_ACTIVATION,
        dropout_rate=config.MLP_DROPOUT_RATE
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY)
    # Learning rate scheduler (ReduceLROnPlateau is good with validation)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.MLP_LR_FACTOR, patience=config.MLP_LR_PATIENCE, verbose=True)

    # Training loop with validation and early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config.MLP_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (fmri_batch, embed_batch) in enumerate(train_loader):
            fmri_batch = fmri_batch.to(device)
            embed_batch = embed_batch.to(device)

            optimizer.zero_grad()
            outputs = model(fmri_batch)
            loss = criterion(outputs, embed_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        avg_val_loss = float('inf') # Default if no validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for fmri_batch_val, embed_batch_val in val_loader:
                    fmri_batch_val = fmri_batch_val.to(device)
                    embed_batch_val = embed_batch_val.to(device)
                    outputs_val = model(fmri_batch_val)
                    loss_val = criterion(outputs_val, embed_batch_val)
                    val_loss += loss_val.item()
            avg_val_loss = val_loss / len(val_loader)

            # Reduce LR on plateau
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
                print(f"Epoch {epoch+1}: Val loss ({avg_val_loss:.6f}) did not improve from {best_val_loss:.6f}")

            if epochs_no_improve >= config.MLP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        else:
             # If no validation, save the last model state (or best train loss, less ideal)
             best_model_state = copy.deepcopy(model.state_dict())


        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{config.MLP_EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - Time: {epoch_time:.2f}s")


    # Load the best model state found during training
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state with validation loss: {best_val_loss:.6f}")
    else:
         print("Warning: No best model state found (possibly no validation or training failed early). Using final model state.")


    # Save the trained model state dictionary
    model_filename = os.path.join(config.MODELS_BASE_PATH, f"mlp_mapping_{fmri_model_name}_{model_name}.pt")
    try:
        os.makedirs(config.MODELS_BASE_PATH, exist_ok=True)
        # Save model state dict, input/output dims, and maybe config used
        save_dict = {
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_layers': config.MLP_HIDDEN_LAYERS,
            'activation': config.MLP_ACTIVATION,
            'dropout_rate': config.MLP_DROPOUT_RATE,
            'best_val_loss': best_val_loss # Record performance
        }
        torch.save(save_dict, model_filename)
        print(f"Saved trained MLP model to: {model_filename}")
    except Exception as e:
        print(f"Error saving MLP model: {e}")
        model_filename = None # Indicate saving failed

    total_train_time = time.time() - start_train_time
    print(f"--- MLP Training Finished. Total Time: {total_train_time/60:.2f} minutes ---")

    return model, model_filename


# 4. MLP Loading Function
def load_mlp_model(model_filename, device=config.DEVICE):
    """Loads a saved MLP model state dictionary."""
    if not os.path.exists(model_filename):
        print(f"Error: MLP model file not found: {model_filename}")
        return None
    try:
        checkpoint = torch.load(model_filename, map_location=device)

        # Recreate model architecture
        model = FmriMappingMLP(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            hidden_layers=checkpoint['hidden_layers'],
            activation=checkpoint.get('activation', 'relu'), # Use get for backward compat
            dropout_rate=checkpoint.get('dropout_rate', 0.5)
        ).to(device)

        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set to evaluation mode
        print(f"Loaded MLP model from: {model_filename}")
        print(f"  - Best recorded validation loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
        return model
    except Exception as e:
        print(f"Error loading MLP model from {model_filename}: {e}")
        return None


# 5. MLP Prediction Function
@torch.no_grad() # Essential for inference
def predict_embeddings_mlp(mlp_model, X_data, device=config.DEVICE):
    """Predicts embeddings using the loaded MLP model."""
    if mlp_model is None:
        print("Error: MLP model is None. Cannot predict.")
        return None

    mlp_model.eval() # Ensure model is in evaluation mode

    # Convert numpy data to tensor and move to device
    if isinstance(X_data, np.ndarray):
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    elif isinstance(X_data, torch.Tensor):
        X_tensor = X_data.to(device)
    else:
        print("Error: Input data must be a numpy array or torch tensor.")
        return None

    print(f"Predicting embeddings for {X_tensor.shape[0]} samples using MLP...")

    # Predict in batches if data is large (optional, good practice)
    # For simplicity here, predict all at once if memory allows
    try:
        Z_pred_tensor = mlp_model(X_tensor)
    except Exception as e:
        print(f"Error during MLP prediction: {e}")
        return None

    print("Prediction complete.")
    # Return predictions as numpy array (common format for downstream tasks)
    return Z_pred_tensor.cpu().numpy()


# === Ridge Regression Implementation (Kept for reference/comparison) ===
def train_ridge_mapping(X_train, Z_train, alpha, max_iter, model_name, fmri_model_name):
    """Trains a Ridge regression model and saves it."""
    print(f"Training Ridge model (alpha={alpha}) for {fmri_model_name} -> {model_name}...")
    try:
        ridge = Ridge(alpha=alpha, max_iter=max_iter, random_state=config.RANDOM_STATE)
        ridge.fit(X_train, Z_train)

        # Evaluate on training data (optional, sanity check)
        Z_train_pred = ridge.predict(X_train)
        train_rmse, train_r2 = evaluate_prediction(Z_train, Z_train_pred) # Use updated eval func
        print(f"Ridge training complete. Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")

        # Save the trained model
        model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{fmri_model_name}_{model_name}_alpha{alpha}.sav")
        os.makedirs(config.MODELS_BASE_PATH, exist_ok=True)
        with open(model_filename, 'wb') as f:
            pickle.dump(ridge, f)
        print(f"Saved trained Ridge model to: {model_filename}")
        return ridge, model_filename

    except Exception as e:
        print(f"Error training Ridge model: {e}")
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
         return None

def predict_embeddings_ridge(ridge_model, X_data):
    """Predicts embeddings using the loaded Ridge model."""
    if ridge_model is None:
         print("Error: Ridge model is None.")
         return None
    print(f"Predicting embeddings for {X_data.shape[0]} samples using Ridge...")
    try:
        Z_pred = ridge_model.predict(X_data)
        print("Prediction complete.")
        return Z_pred
    except Exception as e:
         print(f"Error during Ridge prediction: {e}")
         return None


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Mapping Models ---")

    # --- Simulate Data ---
    emb_model_name = "resnet50" # Example embedding model name
    fmri_model_name = f"Subj{config.SUBJECT_ID}_{config.ROI}" # Example identifier for fMRI source
    dim_fmri = 5000 # Example dimensionality for VC ROI
    dim_embedding = config.EMBEDDING_MODELS[emb_model_name]['embedding_dim']
    n_train = 1000
    n_val = 200
    n_test = 50

    X_train_sim = np.random.rand(n_train, dim_fmri).astype(np.float32)
    Z_train_sim = np.random.rand(n_train, dim_embedding).astype(np.float32)
    X_val_sim = np.random.rand(n_val, dim_fmri).astype(np.float32)
    Z_val_sim = np.random.rand(n_val, dim_embedding).astype(np.float32)
    X_test_sim = np.random.rand(n_test, dim_fmri).astype(np.float32)
    Z_test_sim = np.random.rand(n_test, dim_embedding).astype(np.float32) # Ground truth for test simulation

    print(f"Simulating data for {fmri_model_name} -> {emb_model_name}:")
    print(f"X_train: {X_train_sim.shape}, Z_train: {Z_train_sim.shape}")
    print(f"X_val: {X_val_sim.shape}, Z_val: {Z_val_sim.shape}")
    print(f"X_test: {X_test_sim.shape}, Z_test (true): {Z_test_sim.shape}")

    # --- Test MLP ---
    print("\n--- Testing MLP ---")
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
                    print("Evaluating MLP Prediction:")
                    evaluate_prediction(Z_test_sim, Z_pred_mlp)
            else:
                 print("MLP loading failed.")
        else:
             print("MLP training failed.")

    except Exception as e:
        print(f"\nAn error occurred during MLP test: {e}")
        traceback.print_exc()

    # --- Test Ridge (Optional comparison) ---
    print("\n--- Testing Ridge (Comparison) ---")
    try:
        # Train Ridge
        ridge_model, ridge_path = train_ridge_mapping(
            X_train_sim, Z_train_sim,
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
                    print("Evaluating Ridge Prediction:")
                    evaluate_prediction(Z_test_sim, Z_pred_ridge)
            else:
                 print("Ridge loading failed.")
        else:
            print("Ridge training failed.")

    except Exception as e:
        print(f"\nAn error occurred during Ridge test: {e}")
        traceback.print_exc()


    print("\n--- Mapping Models Test Complete ---")
