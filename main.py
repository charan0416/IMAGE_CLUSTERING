# main.py (Upgraded with a new endpoint to get all photos in a cluster)

import os
import pandas as pd
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Configuration ---
app = FastAPI()
DB_PATH = "face_database"
CLUSTERS_FILE = os.path.join(DB_PATH, "face_clusters.pkl")
PEOPLE_FILE = os.path.join(DB_PATH, "people.pkl")


# --- Dynamic Image Serving ---
def get_image_library_root():
    if not os.path.exists(CLUSTERS_FILE): return None
    try:
        df = pd.read_pickle(CLUSTERS_FILE)
        if not df.empty: return os.path.commonpath(df['image_path'].tolist())
    except Exception:
        return None
    return None


image_root = get_image_library_root()
if image_root:
    app.mount("/images", StaticFiles(directory=image_root), name="images")


def load_people():
    if os.path.exists(PEOPLE_FILE):
        with open(PEOPLE_FILE, "rb") as f: return pickle.load(f)
    return {}


# --- API Endpoints ---
@app.get("/api/clusters")
async def get_clusters():
    if not os.path.exists(CLUSTERS_FILE):
        return JSONResponse(content={"error": "Clusters file not found."}, status_code=404)

    df = pd.read_pickle(CLUSTERS_FILE)
    people_names = load_people()
    df_clusters = df[df['cluster'] != -1]  # Exclude noise points

    clusters = {}
    for cluster_id, group in df_clusters.groupby('cluster'):
        rep_image_data = group.iloc[0]
        relative_path = os.path.relpath(rep_image_data['image_path'], image_root)

        clusters[int(cluster_id)] = {
            "name": people_names.get(cluster_id, f"Person {cluster_id + 1}"),
            "face_count": len(group),
            "representative_image": f"/images/{relative_path.replace(os.path.sep, '/')}"
        }
    return JSONResponse(content=clusters)


# --- NEW ENDPOINT TO GET ALL PHOTOS FOR A SPECIFIC CLUSTER ---
@app.get("/api/cluster/{cluster_id}")
async def get_cluster_details(cluster_id: int):
    if not os.path.exists(CLUSTERS_FILE):
        return JSONResponse(content={"error": "Clusters file not found."}, status_code=404)

    df = pd.read_pickle(CLUSTERS_FILE)

    # Filter the DataFrame to get only the photos for the requested cluster
    cluster_photos_df = df[df['cluster'] == cluster_id]

    if cluster_photos_df.empty:
        return JSONResponse(content={"error": "Cluster ID not found."}, status_code=404)

    # Create a list of web-accessible image URLs
    image_urls = []
    for image_path in cluster_photos_df['image_path'].unique():
        relative_path = os.path.relpath(image_path, image_root)
        image_urls.append(f"/images/{relative_path.replace(os.path.sep, '/')}")

    return JSONResponse(content={"images": image_urls})


@app.post("/api/name-cluster")
async def name_cluster(request: Request):
    data = await request.json()
    people = load_people()
    people[int(data["cluster_id"])] = data["name"]
    with open(PEOPLE_FILE, "wb") as f: pickle.dump(people, f)
    return JSONResponse(content={"status": "success"})


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f: return HTMLResponse(content=f.read())