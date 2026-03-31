import os
import requests
import tarfile
import urllib.request

DATASET_VERSION_ID = 10251170
API_URL = f"https://zenodo.org/api/records/{DATASET_VERSION_ID}"
DATA_DIR = r"D:\RemoteSensing-Project\Dataset"

os.makedirs(DATA_DIR, exist_ok=True)

print(f"Fetching record {DATASET_VERSION_ID} from Zenodo...")
response = requests.get(API_URL)
if response.status_code == 200:
    data = response.json()
    files = data.get("files", [])
    print(f"Found {len(files)} files.")
    for f in files:
        file_name = f.get("key", f.get("filename"))
        download_url = f.get("links", {}).get("self", "")
        if "labels.geojson" in file_name or "features.tar.gz" in file_name:
            out_path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(out_path):
                print(f"Downloading {file_name}...")
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(out_path, 'wb') as outFile:
                        for chunk in r.iter_content(chunk_size=8192):
                            outFile.write(chunk)
                print(f"Saved {file_name}")
            else:
                print(f"{file_name} already exists.")
            
            # extract features
            if file_name == "features.tar.gz":
                print(f"Extracting {file_name}...")
                with tarfile.open(out_path, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR)
                print("Extraction complete.")
            
    print("Dataset download complete.")
else:
    print(f"Failed to fetch metadata from Zenodo: {response.status_code}")
