# indexer.py (Final Version with Incremental Indexing)

import os
import pickle
import face_recognition
from tqdm import tqdm

# --- Configuration ---
IMAGE_LIBRARY_PATH = "/Users/saicharanbaru/Desktop/123"
DB_PATH = "face_database"
os.makedirs(DB_PATH, exist_ok=True)
db_file_path = os.path.join(DB_PATH, "face_embeddings.pkl")



def get_image_files(path):
    image_files = []
    if not os.path.isdir(path):
        print(f"!!! ERROR: Invalid directory: '{path}'")
        return []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files


def run_indexing():
    all_image_files = get_image_files(IMAGE_LIBRARY_PATH)
    if not all_image_files:
        print("No image files found. Please check your IMAGE_LIBRARY_PATH in indexer.py");
        return

    all_face_data = []
    face_id_counter = 0
    processed_files = set()  # A set to keep track of files we've already indexed

    # --- INCREMENTAL LOGIC: Load existing database if it exists ---
    if os.path.exists(db_file_path):
        try:
            with open(db_file_path, "rb") as f:
                all_face_data = pickle.load(f)
                face_id_counter = len(all_face_data)
            # Create a set of paths for fast checking
            processed_files = {data['image_path'] for data in all_face_data}
            print(f"Loaded existing database. {len(processed_files)} files have already been processed.")
        except (EOFError, pickle.UnpicklingError):
            print("Warning: Database file was corrupted. Starting from scratch.")
            all_face_data = []  # Reset if file is broken

    # Determine which files are new
    files_to_process = [f for f in all_image_files if f not in processed_files]

    if not files_to_process:
        print("No new images to index. Your database is up to date.")
        return

    print(f"Found {len(files_to_process)} new images to index.")

    for image_path in tqdm(files_to_process, desc="Processing New Images"):
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations: continue

            face_encodings = face_recognition.face_encodings(image, face_locations)
            tqdm.write(f"  [Found {len(face_encodings)} face(s)] in {os.path.basename(image_path)}")

            for encoding in face_encodings:
                all_face_data.append({
                    "face_id": face_id_counter,
                    "image_path": image_path,
                    "embedding": encoding
                })
                face_id_counter += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] on {os.path.basename(image_path)}: {e}")

    # Save the combined (old + new) data back to the file
    with open(db_file_path, "wb") as f:
        pickle.dump(all_face_data, f)

    print(f"\n--- Indexing complete. Database now contains a total of {len(all_face_data)} faces. ---")


if __name__ == "__main__":
    run_indexing()