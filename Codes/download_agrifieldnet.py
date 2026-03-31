import os
import sys

codes_dir = os.path.abspath(r"D:\RemoteSensing-Project\Codes")
os.environ["PATH"] += os.pathsep + codes_dir

try:
    from torchgeo.datasets import AgriFieldNet
except ImportError as e:
    print(f"Error importing torchgeo: {e}")
    exit(1)

out_dir = r"D:\RemoteSensing-Project\Dataset\AgriFieldNet"
os.makedirs(out_dir, exist_ok=True)

print("Connecting to Source Cooperative via TorchGeo API...")
print("Starting download of the native Indian AgriFieldNet dataset (This dataset is extremely large, please wait)...")

try:
    dataset = AgriFieldNet(paths=out_dir, download=True)
    print("Download completed successfully!")
    print(f"Dataset securely formatted and saved to: {out_dir}")
except Exception as e:
    print(f"Download failed. TorchGeo encountered an error: {str(e)}")
    print("If it mentions azcopy, we will install the Microsoft azcopy binary next.")
