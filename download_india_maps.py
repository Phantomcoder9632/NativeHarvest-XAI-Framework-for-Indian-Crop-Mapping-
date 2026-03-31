import os
import urllib.request
import zipfile

url = "https://github.com/datameet/maps/archive/refs/heads/master.zip"
data_dir = r"D:\RemoteSensing-Project\Dataset"
zip_path = os.path.join(data_dir, "India_Maps.zip")
extract_dir = os.path.join(data_dir, "India_Maps")

os.makedirs(data_dir, exist_ok=True)

print(f"Downloading DataMeet India Maps...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete. Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Clean up zip file
os.remove(zip_path)

print(f"DataMeet India Maps extracted to {extract_dir}.")
