# retrieval.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import time

import config

def train_knn_retrieval(database_features, n_neighbors, model_name):
    """Fits a k-NN model on the database features and saves it."""
    print(f"Fitting k-NN model (k={n_neighbors}) on {model_name} database features ({database_features.shape})...")
    start_time = time.time()
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute') # Cosine is common for embeddings
    # Note: 'auto' might choose a faster algorithm if appropriate, but 'brute' guarantees correctness for cosine.
    knn.fit(database_features)
    end_time = time.time()
    print(f"k-NN fitting complete. Time taken: {end_time - start_time:.2f} seconds")

    # Save the k-NN model
    knn_filename = os.path.join(config.SAVED_KNN_MODELS_PATH, f"knn_{model_name}_k{n_neighbors}.sav")
    with open(knn_filename, 'wb') as f:
        pickle.dump(knn, f)
    print(f"Saved k-NN model to: {knn_filename}")

    return knn, knn_filename

def load_knn_model(knn_filename):
    """Loads a saved k-NN model."""
    if not os.path.exists(knn_filename):
        raise FileNotFoundError(f"k-NN model file not found: {knn_filename}")
    with open(knn_filename, 'rb') as f:
        knn = pickle.load(f)
    print(f"Loaded k-NN model from: {knn_filename}")
    return knn

def retrieve_nearest_neighbors(knn_model, query_embeddings, database_labels, database_class_map):
    """
    Uses the k-NN model to find nearest neighbors for query embeddings.
    Returns the indices, distances, and readable labels of the neighbors.
    """
    print(f"Retrieving nearest neighbors for {query_embeddings.shape[0]} query embeddings...")
    start_time = time.time()
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(query_embeddings)
    end_time = time.time()
    print(f"k-NN search complete. Time taken: {end_time - start_time:.2f} seconds")

    # Get the corresponding labels from the database
    retrieved_db_labels_numeric = database_labels[indices] # Shape: (n_queries, k)

    # Convert numeric labels to readable names using the map
    retrieved_readable_labels = []
    for i in range(retrieved_db_labels_numeric.shape[0]): # Iterate through queries
         query_labels = []
         for j in range(retrieved_db_labels_numeric.shape[1]): # Iterate through neighbors (k)
              numeric_label = retrieved_db_labels_numeric[i, j]
              readable_label = database_class_map.get(numeric_label, f"UnknownLabel_{numeric_label}")
              query_labels.append(readable_label)
         retrieved_readable_labels.append(query_labels) # List of lists of strings

    # retrieved_readable_labels will be like:
    # [ ['cat', 'dog', ...],   # Neighbors for query 0
    #   ['car', 'truck', ...], # Neighbors for query 1
    #   ... ]

    print(f"Retrieved labels for {len(retrieved_readable_labels)} queries.")
    return indices, distances, retrieved_readable_labels


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Retrieval ---")
    # Requires precomputed Tiny ImageNet features

    model_name_test = "vit" # Example
    n_neighbors_test = config.KNN_N_NEIGHBORS

    try:
        # Load precomputed Tiny ImageNet features and labels
        feature_file = os.path.join(config.TINY_IMAGENET_FEATURES_PATH, f"tiny_imagenet_features_{model_name_test}.npy")
        labels_file = os.path.join(config.TINY_IMAGENET_FEATURES_PATH, "tiny_imagenet_labels.npy")
        class_map_file = os.path.join(config.TINY_IMAGENET_FEATURES_PATH, "tiny_imagenet_class_map.npy")

        if not (os.path.exists(feature_file) and os.path.exists(labels_file) and os.path.exists(class_map_file)):
             print(f"Error: Precomputed features/labels/map for {model_name_test} not found.")
             print("Run feature_extraction.py first to generate them.")
        else:
            db_features = np.load(feature_file)
            db_labels = np.load(labels_file)
            db_class_map = np.load(class_map_file, allow_pickle=True).item()
            print(f"Loaded database features ({db_features.shape}) and labels ({db_labels.shape}) for {model_name_test}")

            # Fit k-NN
            knn_model, knn_path = train_knn_retrieval(db_features, n_neighbors_test, model_name_test)

            # Load k-NN (demonstration)
            loaded_knn = load_knn_model(knn_path)

            # Simulate some query embeddings (e.g., predicted embeddings from fMRI)
            n_queries = 10
            query_dim = db_features.shape[1]
            query_embeddings_sim = np.random.rand(n_queries, query_dim)
            # Normalize query embeddings similar to how db embeddings might be (e.g., L2 norm) for cosine
            query_embeddings_sim = query_embeddings_sim / np.linalg.norm(query_embeddings_sim, axis=1, keepdims=True)


            # Retrieve neighbors
            indices, distances, readable_labels = retrieve_nearest_neighbors(
                loaded_knn, query_embeddings_sim, db_labels, db_class_map
            )

            print(f"\nRetrieved neighbor indices shape: {indices.shape}") # (n_queries, k)
            print(f"Retrieved neighbor distances shape: {distances.shape}") # (n_queries, k)
            print(f"\nExample retrieved readable labels (Query 0): {readable_labels[0]}")
            print(f"Example retrieved readable labels (Query 1): {readable_labels[1]}")

    except Exception as e:
        print(f"\nAn error occurred during retrieval test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Retrieval Test Complete ---")
