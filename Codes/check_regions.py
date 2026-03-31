import json

labels_file = r"D:\RemoteSensing-Project\Dataset\CropHarvest\labels.geojson"

print("Loading labels...")
with open(labels_file, "r") as f:
    geojson = json.load(f)

datasets = set()
labels = set()

for feat in geojson["features"]:
    props = feat["properties"]
    dataset = str(props.get("dataset", "")).lower()
    label = str(props.get("label", "")).lower()
    
    datasets.add(dataset)
    labels.add(label)

print("Available Datasets (Regions/Projects):")
print(sorted(list(datasets)))

print("\nAvailable Crop Labels:")
print(sorted(list(labels)))
