import os
import glob
import time
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
import ee
from tqdm import tqdm

GCP_PROJECT_ID = 'remote-sensing-harvest'
agrifieldnet_labels_dir = r"D:\RemoteSensing-Project\Dataset\AgriFieldNet\train_labels"
output_h5 = r"D:\RemoteSensing-Project\Dataset\native_india_arrays.h5"

def setup_gee():
    try:
        ee.Initialize(project=GCP_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GCP_PROJECT_ID)

def get_wgs84_polygon(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        src_crs = src.crs
    left, bottom, right, top = transform_bounds(src_crs, 'EPSG:4326', *bounds)
    return box(left, bottom, right, top)

def fetch_sar_time_series(geometry, start_date, end_date):
    """Fetches VV and VH backscatter from Earth Engine"""
    coords = list(geometry.exterior.coords)
    ee_polygon = ee.Geometry.Polygon(coords)
    
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterBounds(ee_polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW')))
                     
    def extract_means(image):
        date = image.date().format('YYYY-MM-dd')
        means = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=ee_polygon, scale=10, maxPixels=1e9)
        return ee.Feature(None, {'date': date, 'VV': means.get('VV'), 'VH': means.get('VH')})
        
    features = s1_collection.map(extract_means).getInfo().get('features', [])
    records = [[f.get('properties', {}).get(k) for k in ('date', 'VV', 'VH')] for f in features]
    df = pd.DataFrame(records, columns=['Date', 'VV', 'VH']).dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def extract_label_from_tif(label_tif):
    """Read the dominant crop class from the LABEL tif (pixel values = AgriFieldNet crop codes).
    Background/no-crop pixels are 0 and are ignored."""
    try:
        with rasterio.open(label_tif) as src:
            data = src.read(1)
            unique, counts = np.unique(data, return_counts=True)
            valid = unique != 0
            if not np.any(valid):
                return 0
            return int(unique[valid][np.argmax(counts[valid])])
    except:
        return 0

def build_tensors(limit=50):
    setup_gee()
    tif_files = [f for f in glob.glob(os.path.join(agrifieldnet_labels_dir, "*.tif")) if not f.endswith("field_ids.tif")]
    
    print(f"Located {len(tif_files)} total Farm arrays. Building H5 Tensor for first {limit} farms to avoid GEE API Rate Limits...")
    
    start_date, end_date = '2021-01-01', '2021-12-31'  # FULL YEAR: captures both Kharif + Rabi crops
    date_range = pd.date_range(start=start_date, end=end_date)
    num_days = len(date_range)
    
    # Preallocate H5 variables
    features_list = []
    labels_list = []
    
    for i in tqdm(range(min(limit, len(tif_files)))):
        tif_path = tif_files[i]
        field_id_path = tif_path.replace(".tif", "_field_ids.tif")
        
        try:
            # 1. Get Geometry & SAR Data
            geom = get_wgs84_polygon(tif_path)
            sar_df = fetch_sar_time_series(geom, start_date, end_date)
            
            if sar_df.empty:
                continue
                
            # 2. Extract Crop Label from MAIN label tif (same file as the geometry source)
            label = extract_label_from_tif(tif_path)   # <-- correct file!
            if label == 0:
                continue
                
            # 3. Interpolate SAR into [360, 2] structure (padded to num_days)
            sar_df.set_index('Date', inplace=True)
            # Reindex to encompass every single day in the sequence
            full_series = sar_df.reindex(date_range)
            # Interpolate missing days linearly
            full_series = full_series.interpolate(method='linear').bfill().ffill()
            
            # Since AgriFieldNet doesn't supply Native Optical Time-Series directly via Torchgeo easily here,
            # We construct a synthetic NDVI (zeros) to match the [360, 3] LSTM requirement
            # The model will naturally ignore channel 0 (NDVI) and only use channel 1, 2 (VV, VH)
            tensor_3d = np.zeros((num_days, 3))
            tensor_3d[:, 1] = full_series['VV'].values
            tensor_3d[:, 2] = full_series['VH'].values
            
            features_list.append(tensor_3d)
            labels_list.append(label)
            
            # Sleep briefly to avoid Google API Rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Failed on {os.path.basename(tif_path)}: {e}")
            
    # Save to H5
    print("\nArray compilation complete. Saving to H5 PyTorch Base...")
    features_np = np.array(features_list)
    labels_np = np.array(labels_list)
    
    with h5py.File(output_h5, 'w') as hf:
        hf.create_dataset('features', data=features_np)
        hf.create_dataset('labels', data=labels_np)
        
    print(f"SUCCESS: Saved Matrix {features_np.shape} to {output_h5}")

if __name__ == "__main__":
    build_tensors(limit=10000) # Testing on just 10 farms first
