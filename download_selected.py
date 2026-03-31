import os
import requests
import tarfile
import urllib.request
import time

DATASET_VERSION_ID = 10251170
API_URL = f"https://zenodo.org/api/records/{DATASET_VERSION_ID}"
DATA_DIR = r"D:\RemoteSensing-Project\Dataset"

os.makedirs(DATA_DIR, exist_ok=True)

class ProgressHook:
    def __init__(self):
        self.last_time = time.time()
        self.downloaded = 0
        
    def __call__(self, block_num, block_size, total_size):
        self.downloaded += block_size
        current_time = time.time()
        if current_time - self.last_time > 5:
            percent = self.downloaded * 100 / total_size if total_size > 0 else 0
            print(f"Downloaded {self.downloaded / (1024*1024):.2f} MB of {total_size / (1024*1024):.2f} MB ({percent:.1f}%)")
            self.last_time = current_time

print(f"Fetching record {DATASET_VERSION_ID} from Zenodo...")
response = requests.get(API_URL)
if response.status_code == 200:
    data = response.json()
    files = data.get("files", [])
    
    # We only want features.tar.gz and labels.geojson to keep it reasonable
    target_files = ["features.tar.gz", "labels.geojson"]
    
    for f in files:
        file_name = f.get("key", f.get("filename"))
        if file_name in target_files:
            download_url = f.get("links", {}).get("self", "")
            out_path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(out_path):
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(download_url, out_path, reporthook=ProgressHook())
                print(f"\nSaved {file_name}")
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
