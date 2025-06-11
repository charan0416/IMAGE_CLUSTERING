# clusterer.py (Tuned for 'face_recognition' library)
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import pickle

DB_PATH = "face_database"
EMBEDDINGS_FILE = os.path.join(DB_PATH, "face_embeddings.pkl")
CLUSTERS_FILE = os.path.join(DB_PATH, "face_clusters.pkl")

# The default distance for face_recognition is around 0.6.
# A good starting epsilon for clustering is lower than that.
# You can adjust this value to make clustering stricter (smaller value) or more lenient (larger value).
EPSILON = 0.4
MIN_SAMPLES = 2  # A person needs to appear at least twice to form a cluster


def run_clustering():
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: Embeddings file not found. Please run indexer.py first.");
        return
    with open(EMBEDDINGS_FILE, "rb") as f:
        all_face_data = pickle.load(f)
    if not all_face_data:
        print("Embedding database is empty. No faces to cluster.");
        return

    df = pd.DataFrame(all_face_data)
    embeddings = np.array(df['embedding'].tolist())
    print(f"Starting clustering on {len(embeddings)} faces with EPSILON = {EPSILON}...")

    # The 'face_recognition' library uses Euclidean distance by default.
    db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric='euclidean').fit(embeddings)
    df['cluster'] = db.labels_

    with open(CLUSTERS_FILE, "wb") as f:
        pickle.dump(df, f)

    num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    num_noise = list(db.labels_).count(-1)

    print(f"\nClustering complete.")
    print(f"  - Found {num_clusters} distinct people (clusters).")
    print(f"  - {num_noise} faces were considered unique and not clustered.")


if __name__ == "__main__":
    run_clustering()