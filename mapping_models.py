# mapping_models.py
 import numpy as np
 from sklearn.linear_model import Ridge
 from sklearn.metrics import mean_squared_error, r2_score
 import pickle
 import os
 
 import config
 
 def train_ridge_mapping(X_train, Z_train, alpha, max_iter, model_name):
     """Trains a Ridge regression model and saves it."""
     print(f"Training Ridge model (alpha={alpha}) for {model_name}...")
     ridge = Ridge(alpha=alpha, max_iter=max_iter, random_state=config.RANDOM_STATE)
     ridge.fit(X_train, Z_train)
 
     # Evaluate on training data (optional, sanity check)
     Z_train_pred = ridge.predict(X_train)
     train_rmse = np.sqrt(mean_squared_error(Z_train, Z_train_pred))
     train_r2 = r2_score(Z_train, Z_train_pred)
     print(f"Training complete. Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")
 
     # Save the trained model
     model_filename = os.path.join(config.MODELS_BASE_PATH, f"ridge_mapping_{model_name}_alpha{alpha}.sav")
     with open(model_filename, 'wb') as f:
         pickle.dump(ridge, f)
     print(f"Saved trained Ridge model to: {model_filename}")
 
     return ridge, model_filename
 
 def load_ridge_model(model_filename):
     """Loads a saved Ridge model."""
     if not os.path.exists(model_filename):
         raise FileNotFoundError(f"Ridge model file not found: {model_filename}")
     with open(model_filename, 'rb') as f:
         ridge = pickle.load(f)
     print(f"Loaded Ridge model from: {model_filename}")
     return ridge
 
 def predict_embeddings(ridge_model, X_data):
     """Predicts embeddings using the loaded Ridge model."""
     print(f"Predicting embeddings for {X_data.shape[0]} samples...")
     Z_pred = ridge_model.predict(X_data)
     print("Prediction complete.")
     return Z_pred
 
 def evaluate_prediction(Z_true, Z_pred):
     """Calculates RMSE and R2 score for predictions."""
     rmse = np.sqrt(mean_squared_error(Z_true, Z_pred))
     r2 = r2_score(Z_true, Z_pred)
     print(f"Evaluation - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
     return rmse, r2
 
 # --- Example Usage ---
 if __name__ == "__main__":
     print("--- Testing Mapping Models ---")
     # This requires feature extraction to have run and produced X_train, Z_train, X_test_avg, Z_test_avg
     # We'll simulate this with random data for demonstration if run directly
 
     model_name_test = "resnet50" # Example model
     dim_fmri = 1000 # Example dimensionality
     dim_embedding = config.EMBEDDING_MODELS[model_name_test]['embedding_dim']
     n_train = 500
     n_test = 50
 
     # Simulate data
     X_train_sim = np.random.rand(n_train, dim_fmri)
     Z_train_sim = np.random.rand(n_train, dim_embedding)
     X_test_sim = np.random.rand(n_test, dim_fmri)
     Z_test_sim = np.random.rand(n_test, dim_embedding) # Ground truth for test simulation
 
     print(f"Simulating data for {model_name_test}:")
     print(f"X_train: {X_train_sim.shape}, Z_train: {Z_train_sim.shape}")
     print(f"X_test: {X_test_sim.shape}, Z_test: {Z_test_sim.shape}")
 
     try:
         # Train
         ridge_model, model_path = train_ridge_mapping(
             X_train_sim, Z_train_sim,
             config.RIDGE_ALPHA, config.RIDGE_MAX_ITER, model_name_test
         )
 
         # Load (redundant here, but demonstrates loading)
         loaded_ridge = load_ridge_model(model_path)
 
         # Predict on test data
         Z_pred_test = predict_embeddings(loaded_ridge, X_test_sim)
         print(f"Predicted test embeddings shape: {Z_pred_test.shape}")
 
         # Evaluate prediction against simulated ground truth
         evaluate_prediction(Z_test_sim, Z_pred_test)
 
     except Exception as e:
         print(f"\nAn error occurred during mapping model test: {e}")
 
     print("\n--- Mapping Models Test Complete ---")
