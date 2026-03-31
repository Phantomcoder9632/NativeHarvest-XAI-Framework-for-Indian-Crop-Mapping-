import os
import glob
import ee
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

# To run this script securely, you must have an active Google Cloud Project (GCP)
GCP_PROJECT_ID = 'remote-sensing-harvest'

agrifieldnet_labels_dir = r"D:\RemoteSensing-Project\Dataset\AgriFieldNet\train_labels"

def setup_gee():
    print("Initiating Google Earth Engine Authentication...")
    try:
        ee.Initialize(project=GCP_PROJECT_ID)
        print("GEE Successfully Initialized.")
    except Exception as e:
        print("Initialization failed: ", e)
        print("Please authenticate via terminal first.")
        ee.Authenticate()
        ee.Initialize(project=GCP_PROJECT_ID)

def fetch_sar_time_series(geometry, start_date, end_date):
    """
    Fetches a time-series of VV and VH radar backscatter for a given polygon via Google Earth Engine.
    """
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
            reducer=ee.Reducer.mean(),
            geometry=ee_polygon,
            scale=10,
            maxPixels=1e9
        )
        return ee.Feature(None, {
            'date': date,
            'VV': means.get('VV'),
            'VH': means.get('VH')
        })
    
    time_series_features = s1_collection.map(extract_means).getInfo().get('features', [])
    records = []
    for feat in time_series_features:
        props = feat.get('properties', {})
        records.append([props.get('date'), props.get('VV'), props.get('VH')])
        
    df = pd.DataFrame(records, columns=['Date', 'VV', 'VH']).dropna()
    return df

def get_wgs84_polygon(tif_path):
    """Reads a Sentinel-2 TIF file, extracts spatial bounds, and converts to Lat/Lon EPSG:4326 Polygon."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        src_crs = src.crs
        
    # Transform coordinates to WGS84 for Google Earth Engine
    left, bottom, right, top = transform_bounds(src_crs, 'EPSG:4326', *bounds)
    return box(left, bottom, right, top)

if __name__ == "__main__":
    setup_gee()
    
    # Get all actual label tifs (excluding the field_ids arrays)
    tif_files = [f for f in glob.glob(os.path.join(agrifieldnet_labels_dir, "*.tif")) if not f.endswith("field_ids.tif")]
    
    if not tif_files:
        print(f"No .tif files found in {agrifieldnet_labels_dir}")
    else:
        print(f"Loaded {len(tif_files)} image bounding chips.")
        print(f"Testing GEE HARVEST on the first field: {os.path.basename(tif_files[0])}")
        
        # 1. Parse bounding box logic natively
        first_farm_geom = get_wgs84_polygon(tif_files[0])
        print(f"Extracted Native Geocoordinates: {first_farm_geom.bounds}")
        
        # 2. Kharif Monsoon season
        START_DATE = '2021-06-01' 
        END_DATE   = '2021-11-30'
        
        print("Querying Google Earth Engine Servers Arrays...")
        sar_df = fetch_sar_time_series(first_farm_geom, START_DATE, END_DATE)
        
        print("\n------------- GEE NATIVE SAR DATA (FARM 1) -------------")
        print(sar_df.head(10))
        print("---------------------------------------------------------")
        print("Success! You can now map this logic over all 17,000 arrays!")
