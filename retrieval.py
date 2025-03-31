# retrieval.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import time

import config # Make sure config is imported

def train_knn_retrieval(database_features, n_neighbors, model_name):
    """Fits a k-NN model on the database features and saves it."""
    # Input validation
    if database_features is None or database_features.size == 0:
        print("Error: Database features are empty. Cannot train k-NN.")
        return None, None
    if not isinstance(database_features, np.ndarray):
         print("Error: Database features must be a numpy array.")
         return None, None
         
    print(f"Fitting k-NN model (k={n_neighbors}, metric={config.KNN_METRIC}, algo={config.KNN_ALGORITHM}) on {model_name} database features ({database_features.shape})...")
    start_time = time.time()
    try:
        knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=config.KNN_METRIC,      # Use metric from config
            algorithm=config.KNN_ALGORITHM, # Use algorithm from config
            n_jobs=-1                      # Use all available CPU cores
            )
        knn.fit(database_features)
    except Exception as e:
        print(f"Error fitting k-NN model: {e}")
        return None, None
        
    end_time = time.time()
    print(f"k-NN fitting complete. Time taken: {end_time - start_time:.2f} seconds")

    # Save the k-NN model
    knn_filename = os.path.join(config.SAVED_KNN_MODELS_PATH, f"knn_{model_name}_k{n_neighbors}.sav")
    try:
        os.makedirs(config.SAVED_KNN_MODELS_PATH, exist_ok=True) # Ensure dir exists
        with open(knn_filename, 'wb') as f:
            pickle.dump(knn, f)
        print(f"Saved k-NN model to: {knn_filename}")
    except Exception as e:
        print(f"Error saving k-NN model to {knn_filename}: {e}")
        # Return the model anyway, but warn about saving failure
        
    return knn, knn_filename

# --- Load k-NN Model ---
def load_knn_model(knn_filename):
    """Loads a saved k-NN model."""
    if not os.path.exists(knn_filename):
        print(f"Warning: k-NN model file not found: {knn_filename}")
        return None
    try:
        with open(knn_filename, 'rb') as f:
            knn = pickle.load(f)
        print(f"Loaded k-NN model from: {knn_filename}")
        return knn
    except Exception as e:
        print(f"Error loading k-NN model from {knn_filename}: {e}")
        return None

# --- Retrieve Nearest Neighbors ---
def retrieve_nearest_neighbors(knn_model, query_embeddings, database_labels, database_class_map):
    """
    Uses the k-NN model to find nearest neighbors for query embeddings.
    Returns the indices, distances, and readable labels of the neighbors.
    """
    # Input validation
    if knn_model is None:
         print("Error: k-NN model is None. Cannot retrieve neighbors.")
         return None, None, None
    if query_embeddings is None or query_embeddings.size == 0:
         print("Error: Query embeddings are empty. Cannot retrieve neighbors.")
         return None, None, None
    if database_labels is None or database_class_map is None:
         print("Error: Database labels or class map is missing.")
         return None, None, None
         
    # Ensure query embeddings have the correct shape and type
    if not isinstance(query_embeddings, np.ndarray):
         print("Error: Query embeddings must be a numpy array.")
         return None, None, None
    if query_embeddings.ndim == 1: # Handle single query embedding
        query_embeddings = query_embeddings.reshape(1, -1)
    # Check dimension compatibility (optional but good practice)
    # n_features = knn_model.n_features_in_ # Check attribute name, might vary
    # if query_embeddings.shape[1] != n_features:
    #      print(f"Error: Query dimension ({query_embeddings.shape[1]}) doesn't match k-NN feature dimension ({n_features}).")
    #      return None, None, None

    print(f"Retrieving nearest neighbors for {query_embeddings.shape[0]} query embeddings...")
    start_time = time.time()
    try:
        # Find k nearest neighbors
        distances, indices = knn_model.kneighbors(query_embeddings)
    except Exception as e:
        print(f"Error during k-NN search: {e}")
        return None, None, None
        
    end_time = time.time()
    print(f"k-NN search complete. Time taken: {end_time - start_time:.2f} seconds")

    # Get the corresponding labels from the database
    try:
        retrieved_db_labels_numeric = database_labels[indices] # Shape: (n_queries, k)
    except IndexError:
        print("Error: k-NN indices out of bounds for database labels. Check database consistency.")
        return indices, distances, None # Return indices/distances but signal label error
    except Exception as e:
        print(f"Error retrieving numeric labels using indices: {e}")
        return indices, distances, None


    # Convert numeric labels to readable names using the map
    retrieved_readable_labels = []
    for i in range(retrieved_db_labels_numeric.shape[0]): # Iterate through queries
         query_labels = []
         for j in range(retrieved_db_labels_numeric.shape[1]): # Iterate through neighbors (k)
              numeric_label = retrieved_db_labels_numeric[i, j]
              # Use .get() for safe dictionary lookup
              readable_label = database_class_map.get(numeric_label, f"UnknownLabel_{numeric_label}")
              query_labels.append(readable_label)
         retrieved_readable_labels.append(query_labels) # List of lists of strings

    print(f"Retrieved labels for {len(retrieved_readable_labels)} queries.")
    return indices, distances, retrieved_readable_labels


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Retrieval ---")
    # Requires precomputed ImageNet-256 features

    model_name_test = "vit" # Example model to test with
    n_neighbors_test = config.KNN_N_NEIGHBORS

    try:
        # Load precomputed ImageNet-256 features and labels
        feature_file = os.path.join(config.IMAGENET256_FEATURES_PATH, f"imagenet256_features_{model_name_test}.npy")
        labels_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_labels.npy")
        class_map_file = os.path.join(config.IMAGENET256_FEATURES_PATH, "imagenet256_idx_to_class.npy")

        if not (os.path.exists(feature_file) and os.path.exists(labels_file) and os.path.exists(class_map_file)):
             print(f"Error: Precomputed features/labels/map for {model_name_test} on ImageNet-256 not found.")
             print(f"Looked in: {config.IMAGENET256_FEATURES_PATH}")
             print("Run feature_extraction.py first to generate them.")
        else:
            print("Loading database features and labels...")
            db_features = np.load(feature_file)
            db_labels = np.load(labels_file)
            db_class_map = np.load(class_map_file, allow_pickle=True).item()
            print(f"Loaded ImageNet-256 database features ({db_features.shape}) and labels ({db_labels.shape}) for {model_name_test}")

            # Fit k-NN
            knn_model, knn_path = train_knn_retrieval(db_features, n_neighbors_test, model_name_test)

            # Load k-NN (demonstration)
            if knn_path: # Only load if saving was attempted
                 loaded_knn = load_knn_model(knn_path)
            else:
                 loaded_knn = knn_model # Use the model directly if saving failed

            if loaded_knn: # Proceed only if k-NN model is available
                # Simulate some query embeddings (e.g., predicted embeddings from fMRI)
                n_queries = 10
                if db_features.shape[1] > 0: # Check if feature dimension is valid
                    query_dim = db_features.shape[1]
                    query_embeddings_sim = np.random.rand(n_queries, query_dim).astype(db_features.dtype) # Match dtype

                    # Normalize query embeddings if using cosine distance
                    if config.KNN_METRIC == 'cosine':
                        print("Normalizing simulated query embeddings for cosine distance...")
                        query_norms = np.linalg.norm(query_embeddings_sim, axis=1, keepdims=True)
                        # Add epsilon to avoid division by zero for zero vectors
                        query_embeddings_sim = query_embeddings_sim / np.where(query_norms == 0, 1e-9, query_norms)


                    # Retrieve neighbors
                    indices, distances, readable_labels = retrieve_nearest_neighbors(
                        loaded_knn, query_embeddings_sim, db_labels, db_class_map
                    )

                    if indices is not None:
                        print(f"\nRetrieved neighbor indices shape: {indices.shape}") # (n_queries, k)
                        print(f"Retrieved neighbor distances shape: {distances.shape}") # (n_queries, k)
                    if readable_labels: # Check if not empty
                        print(f"\nExample retrieved readable labels (Query 0): {readable_labels[0]}")
                        print(f"Example retrieved readable labels (Query 1): {readable_labels[1]}")
                    else:
                        print("No readable labels retrieved (check query/database or retrieval function).")
                else:
                    print("Error: Database features have zero dimension.")
            else:
                 print("k-NN model could not be trained or loaded. Skipping retrieval.")


    except Exception as e:
        print(f"\nAn error occurred during retrieval test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Retrieval Test Complete ---")
