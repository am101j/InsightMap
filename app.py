"""
InsightMap - Geospatial Price Prediction

Predicts Toronto Airbnb prices using spatial feature engineering, neighborhood analysis, and machine learning.

To train the model, run: python train_model.py

"""

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import json
import os
import warnings

warnings.filterwarnings('ignore')


# Configuration
LISTINGS_PATH = "listings.csv"
SUBWAY_SHAPEFILE_PATH = "subway_data/TTC_SUBWAY_LINES_WGS84.shp"
NEIGHBOURHOOD_SHAPEFILE_PATH = "neighbourhood_data/Neighbourhoods - 4326.shp"
MODEL_DIR = "model"
UNION_STATION_LAT = 43.6456
UNION_STATION_LON = -79.3806

MAX_PRICE = 1000
TORONTO_CENTER = [43.6532, -79.3832]
TOP_N_NEIGHBOURHOODS = 15


def apply_custom_styling():
    """Apply clean, minimal light mode CSS styling."""
    st.markdown("""
    <style>
        /* ===== CLEAN LIGHT THEME ===== */
        .stApp {
            background-color: #ffffff;
        }
        
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1a1a1a;
        }
        
        /* ===== MAIN HEADER ===== */
        .main-header {
            border-bottom: 1px solid #e5e5e5;
            padding: 1.5rem 0;
            margin-bottom: 1.5rem;
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        .main-header p {
            margin: 0.25rem 0 0 0;
            color: #666666;
            font-size: 0.95rem;
        }
        
        /* ===== METRIC CARDS ===== */
        .metric-card {
            background: #f9f9f9;
            border: 1px solid #e5e5e5;
            padding: 1.25rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        .metric-label {
            color: #666666;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-top: 0.25rem;
        }
        
        /* ===== PREDICTION BOX ===== */
        .prediction-box {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .prediction-price {
            font-size: 2.5rem;
            font-weight: 700;
            color: #166534;
        }
        
        .prediction-label {
            color: #166534;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }
        
        /* ===== SIDEBAR STYLING ===== */
        [data-testid="stSidebar"] {
            background-color: #fafafa;
            border-right: 1px solid #e5e5e5;
        }
        
        [data-testid="stSidebar"] .stMarkdown h2 {
            color: #1a1a1a;
            font-weight: 600;
            font-size: 0.95rem;
        }
        
        /* ===== TABS STYLING ===== */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            border-bottom: 1px solid #e5e5e5;
            gap: 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #666666;
            border-radius: 0;
            font-weight: 500;
            padding: 0.75rem 1.25rem;
            border-bottom: 2px solid transparent;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #1a1a1a;
        }
        
        .stTabs [aria-selected="true"] {
            background: transparent !important;
            color: #1a1a1a !important;
            border-bottom: 2px solid #1a1a1a !important;
        }
        
        /* ===== BUTTONS ===== */
        .stButton > button {
            background: #1a1a1a;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        
        .stButton > button:hover {
            background: #333333;
        }
        
        /* ===== METRICS ===== */
        [data-testid="stMetricValue"] {
            font-weight: 600;
            color: #1a1a1a;
        }
        
        [data-testid="stMetricLabel"] {
            color: #666666;
            font-weight: 500;
        }
        
        /* ===== SECTION HEADERS ===== */
        .stMarkdown h3 {
            color: #1a1a1a;
            font-weight: 600;
            font-size: 1.1rem;
            margin-top: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e5e5;
        }
        
        /* ===== LOCATION DETAILS BOX ===== */
        .location-details {
            background: #fafafa;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .location-details h4 {
            color: #666666;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            border-bottom: 1px solid #eeeeee;
        }
        
        .detail-row:last-child {
            border-bottom: none;
        }
        
        .detail-label {
            color: #666666;
            font-size: 0.85rem;
        }
        
        .detail-value {
            color: #1a1a1a;
            font-weight: 500;
        }
        
        /* ===== FOOTER ===== */
        .app-footer {
            border-top: 1px solid #e5e5e5;
            padding: 1.5rem;
            text-align: center;
            margin-top: 2rem;
        }
        
        .app-footer p {
            color: #999999;
            font-size: 0.85rem;
            margin: 0.15rem 0;
        }
        
        /* ===== HIDE STREAMLIT BRANDING ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* ===== MAP CONTAINER ===== */
        iframe {
            border-radius: 8px !important;
            border: 1px solid #e5e5e5 !important;
        }
        
        .leaflet-container {
            cursor: crosshair !important;
        }
        
        /* ===== DIVIDER ===== */
        hr {
            border: none;
            height: 1px;
            background: #e5e5e5;
            margin: 1.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)



@st.cache_data
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    #Load and clean the Airbnb listings data with feature extraction.
    print("Loading listings data...")
    
    df = pd.read_csv(filepath, on_bad_lines='skip')
    print(f"  Loaded {len(df)} rows")
    
    
    # Clean Price
    if df['price'].dtype == object:
        df['price'] = (
            df['price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    
    print(f"  After price cleaning: {len(df)} rows")
    
    # Clean Coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"  After coordinate cleaning: {len(df)} rows")
    
    # Clean Additional Features
    df['number_of_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce').fillna(0)
    
    df['reviews_per_month'] = pd.to_numeric(df['reviews_per_month'], errors='coerce').fillna(0)
    
    df['minimum_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce').fillna(1)
    df['minimum_nights'] = df['minimum_nights'].clip(upper=365)
    
    df['availability_365'] = pd.to_numeric(df['availability_365'], errors='coerce').fillna(0)
    
    df['calculated_host_listings_count'] = pd.to_numeric(
        df['calculated_host_listings_count'], errors='coerce'
    ).fillna(1)
    

    
    df['neighbourhood'] = df['neighbourhood'].fillna('Unknown')
    
    df = df[df['price'] < MAX_PRICE]
    
    print(f"  After outlier filtering (price < {MAX_PRICE}): {len(df)} rows")
    
    df = df.reset_index(drop=True)
    
    return df



@st.cache_data
def load_neighbourhood_boundaries(filepath: str) -> gpd.GeoDataFrame:
    #Load Toronto neighbourhood boundary polygons
    print("Loading neighbourhood boundaries...")
    neighbourhoods_gdf = gpd.read_file(filepath)
    print(f"  Loaded {len(neighbourhoods_gdf)} neighbourhoods")
    return neighbourhoods_gdf


def detect_neighbourhood(lat: float, lon: float, neighbourhoods_gdf: gpd.GeoDataFrame) -> str:
    """
    Detect which neighbourhood a point falls within using point-in-polygon.
    Returns the neighbourhood name or 'Unknown' if outside all boundaries.
    """
    point = gpd.GeoDataFrame(
        {'geometry': [Point(lon, lat)]},
        crs="EPSG:4326"
    )
    
    # Spatial join to find containing polygon
    result = gpd.sjoin(point, neighbourhoods_gdf, how='left', predicate='within')
    
    if len(result) > 0 and 'AREA_NA7' in result.columns and pd.notna(result.iloc[0]['AREA_NA7']):
        return result.iloc[0]['AREA_NA7']
    
    if len(result) > 0 and 'AREA_NAME' in result.columns and pd.notna(result.iloc[0]['AREA_NAME']):
        return result.iloc[0]['AREA_NAME']
    
    return 'Unknown'

@st.cache_data
def load_subway_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Load subway shapefile and extract station points from line geometries.
    Converts to EPSG:32617 (UTM Zone 17N) for meter-based distances.
    """    
    subway_gdf = gpd.read_file(filepath)
    
    # Extract all vertices (points) from line geometries
    all_points = []
    
    for geom in subway_gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            all_points.extend([Point(c) for c in coords])
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                all_points.extend([Point(c) for c in coords])
        elif geom.geom_type == 'Point':
            all_points.append(geom)
    
    print(f"  Extracted {len(all_points)} vertices from subway lines")
    
    stations_gdf = gpd.GeoDataFrame(
        {'geometry': all_points},
        crs="EPSG:4326"
    )
    
    # Project to UTM Zone 17N for meter-based distances
    stations_gdf = stations_gdf.to_crs("EPSG:32617")
    
    return stations_gdf


def create_housing_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert housing DataFrame to GeoDataFrame with UTM projection."""
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:32617")
    return gdf


def build_subway_kdtree(stations_gdf: gpd.GeoDataFrame) -> tuple:
    """Build a KDTree for O(log n) nearest-neighbor queries."""
    station_coords = np.array([
        [geom.x, geom.y] for geom in stations_gdf.geometry
    ])
    tree = cKDTree(station_coords)
    return tree, station_coords


def get_union_station_coords() -> tuple:
    """Get Union Station coordinates in EPSG:32617."""
    union_point = gpd.GeoDataFrame(
        {'geometry': [Point(UNION_STATION_LON, UNION_STATION_LAT)]},
        crs="EPSG:4326"
    )
    union_point = union_point.to_crs("EPSG:32617")
    return union_point.geometry.iloc[0].x, union_point.geometry.iloc[0].y


def engineer_spatial_features(
    housing_gdf: gpd.GeoDataFrame,
    subway_tree: cKDTree,
    union_coords: tuple
) -> gpd.GeoDataFrame:
    
    housing_coords = np.array([
        [geom.x, geom.y] for geom in housing_gdf.geometry
    ])
    
    # Distance to nearest subway
    distances, _ = subway_tree.query(housing_coords)
    housing_gdf['dist_to_subway_meters'] = distances
    
    # Distance to Union Station
    union_x, union_y = union_coords
    dist_to_union = np.sqrt(
        (housing_coords[:, 0] - union_x) ** 2 +
        (housing_coords[:, 1] - union_y) ** 2
    )
    housing_gdf['dist_to_union_meters'] = dist_to_union
    
    # Create distance bins for interpretability
    housing_gdf['subway_proximity'] = pd.cut(
        housing_gdf['dist_to_subway_meters'],
        bins=[0, 500, 1000, 2000, 5000, float('inf')],
        labels=['<500m', '500m-1km', '1-2km', '2-5km', '>5km']
    )
    
    return housing_gdf


@st.cache_resource
def load_trained_model() -> tuple:
    """
    Load the pre-trained XGBoost model and metadata from disk.
    
    This follows industry best practices: train once, deploy forever.
    To retrain the model, run: python train_model.py
    
    Returns:
        tuple: (model, feature_names, top_neighbourhoods, top_props, metrics, importances)
    """
    model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    
    if not os.path.exists(model_path):
        st.error("""
        **Model not found!**
        
        Please train the model first by running:
        ```
        python train_model.py
        ```
        """)
        st.stop()
    
    print("Loading pre-trained model from disk...")
    
    # Load XGBoost booster (we saved using get_booster().save_model())
    booster = xgb.Booster()
    booster.load_model(model_path)
    print("   + Model loaded")
    
    # Load metadata
    with open(os.path.join(MODEL_DIR, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)
    
    with open(os.path.join(MODEL_DIR, "top_neighbourhoods.json"), 'r') as f:
        top_neighbourhoods = json.load(f)
    print(f"   + {len(top_neighbourhoods)} neighbourhoods loaded")
    
    # Try to load top property types if available
    top_props = []
    if os.path.exists(os.path.join(MODEL_DIR, "top_property_types.json")):
        with open(os.path.join(MODEL_DIR, "top_property_types.json"), 'r') as f:
            top_props = json.load(f)
    
    with open(os.path.join(MODEL_DIR, "metrics.json"), 'r') as f:
        metrics = json.load(f)
    print(f"   + Metrics loaded (R^2: {metrics['r2']:.1%})")
    
    with open(os.path.join(MODEL_DIR, "feature_importances.json"), 'r') as f:
        importances = json.load(f)
    
    return booster, feature_names, top_neighbourhoods, top_props, metrics, importances


def plot_feature_importance(importances: dict, top_n: int = 15) -> str:
    """Create feature importance plot with clean light theme."""
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]
    
    # Clean feature names for display
    display_names = []
    for f in features:
        if f.startswith('hood_'):
            display_names.append(f.replace('hood_', ''))
        elif f.startswith('room_'):
            display_names.append(f.replace('room_', ''))
        elif f.startswith('prop_'):
            display_names.append(f.replace('prop_', ''))
        elif f == 'dist_to_subway_meters':
            display_names.append('Subway Distance')
        elif f == 'dist_to_union_meters':
            display_names.append('Union Station Dist')
        elif f == 'number_of_reviews':
            display_names.append('Review Count')
        elif f == 'reviews_per_month':
            display_names.append('Reviews/Month')
        elif f == 'minimum_nights':
            display_names.append('Min Nights')
        elif f == 'availability_365':
            display_names.append('Availability')
        elif f == 'calculated_host_listings_count':
            display_names.append('Host Listings')
        elif f == 'review_scores_rating':
            display_names.append('Rating')
        elif f == 'host_is_superhost':
            display_names.append('Superhost')
        elif f == 'privacy_index':
            display_names.append('Privacy Index')
        elif f == 'bathroom_ratio':
            display_names.append('Bath Ratio')
        else:
            display_names.append(f)
    
    bg_color = '#ffffff'
    text_color = '#1a1a1a'
    grid_color = '#e5e5e5'
    bar_color = '#4a4a4a'
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    bars = ax.barh(display_names[::-1], scores[::-1], color=bar_color, edgecolor='none')
    
    ax.set_xlabel('Importance Score', fontsize=11, color=text_color)
    ax.set_title('Feature Importance', fontsize=14, fontweight='600', color=text_color, pad=15)
    ax.tick_params(colors=text_color, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.xaxis.grid(True, color=grid_color, alpha=0.7)
    
    plt.tight_layout()
    save_path = "feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
    plt.close()
    
    return save_path


def predict_price_for_location(
    lat: float,
    lon: float,
    property_type: str,
    room_type: str,
    neighbourhood: str,
    model,  # xgb.Booster
    subway_tree: cKDTree,
    union_coords: tuple,
    feature_names: list,
    top_neighbourhoods: list,
    top_props: list,
    number_of_reviews: int = 10,
    reviews_per_month: float = 1.0,
    minimum_nights: int = 2,
    availability_365: int = 200,
    calculated_host_listings_count: int = 1,
    accommodates: int = 2,
    bedrooms: int = 1,
    beds: int = 1,
    bathrooms: float = 1.0,
    has_pool: int = 0,
    has_ac: int = 0,
    has_fparking: int = 0,
    has_wifi: int = 1,
    review_scores_rating: float = 4.8,
    host_is_superhost: int = 0,
    host_identity_verified: int = 1,
    instant_bookable: int = 0
) -> tuple:
    """
    Predict price for a given location with enhanced features.
    Returns (predicted_price, dist_subway, dist_union, X_pred)
    """
    # Project coordinates
    point_gdf = gpd.GeoDataFrame(
        {'geometry': [Point(lon, lat)]},
        crs="EPSG:4326"
    ).to_crs("EPSG:32617")
    
    x = point_gdf.geometry.iloc[0].x
    y = point_gdf.geometry.iloc[0].y
    
    # Calculate spatial features
    dist_subway, _ = subway_tree.query([[x, y]])
    dist_subway = dist_subway[0]
    
    union_x, union_y = union_coords
    dist_union = np.sqrt((x - union_x) ** 2 + (y - union_y) ** 2)
    
    # Interactions
    privacy_index = bedrooms / (accommodates + 0.1)
    bathroom_ratio = bathrooms / (bedrooms + 0.1)

    # Build feature vector
    feature_dict = {
        'dist_to_subway_meters': dist_subway,
        'dist_to_union_meters': dist_union,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'minimum_nights': minimum_nights,
        'availability_365': availability_365,
        'calculated_host_listings_count': calculated_host_listings_count,
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'beds': beds,
        'bathrooms': bathrooms,
        'has_pool': has_pool,
        'has_ac': has_ac,
        'has_fparking': has_fparking,
        'has_wifi': has_wifi,
        'review_scores_rating': review_scores_rating,
        'review_scores_cleanliness': review_scores_rating, # Proxy if not set
        'review_scores_location': review_scores_rating,    # Proxy if not set
        'review_scores_value': review_scores_rating,       # Proxy if not set
        'host_is_superhost': host_is_superhost,
        'host_identity_verified': host_identity_verified,
        'instant_bookable': instant_bookable,
        'privacy_index': privacy_index,
        'bathroom_ratio': bathroom_ratio
    }
    
    # One-hot encode property type
    prop_grouped = property_type if property_type in top_props else 'Other'
    all_props = top_props + ['Other'] if top_props else []
    for p in all_props:
        col_name = f'prop_{p}'
        if col_name in feature_names:
            feature_dict[col_name] = 1 if p == prop_grouped else 0

    # One-hot encode room type
    room_types = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
    for rt in room_types:
        col_name = f'room_{rt}'
        if col_name in feature_names:
            feature_dict[col_name] = 1 if rt == room_type else 0
    
    # One-hot encode neighbourhood
    neighbourhood_grouped = neighbourhood if neighbourhood in top_neighbourhoods else 'Other'
    all_neighbourhoods = top_neighbourhoods + ['Other']
    for n in all_neighbourhoods:
        col_name = f'hood_{n}'
        if col_name in feature_names:
            feature_dict[col_name] = 1 if n == neighbourhood_grouped else 0
    
    # Create DataFrame
    X_pred = pd.DataFrame([feature_dict])
    
    # Ensure all columns exist
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[feature_names]
    
    # Model predicts in LOG space using DMatrix, convert back to dollars with expm1
    dmatrix = xgb.DMatrix(X_pred)
    prediction_log = model.predict(dmatrix)[0]
    prediction_dollars = np.expm1(prediction_log)
    
    return max(0, prediction_dollars), dist_subway, dist_union, X_pred


def create_shap_waterfall_plot(model, X_pred: pd.DataFrame, predicted_price: float) -> str:
    """
    Create a SHAP waterfall plot explaining why the price is what it is.
    Returns path to saved image.
    """
    # Create SHAP explainer for the XGBoost booster
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_pred)
    
    # Get the base value (average prediction in log space)
    base_value_log = explainer.expected_value
    base_value_dollars = np.expm1(base_value_log)
    
    # Create clean feature names for display
    feature_display_names = []
    for name in X_pred.columns:
        if name.startswith('hood_'):
            feature_display_names.append(name.replace('hood_', ''))
        elif name.startswith('room_'):
            feature_display_names.append(name.replace('room_', ''))
        elif name.startswith('prop_'):
            feature_display_names.append(name.replace('prop_', ''))
        elif name == 'dist_to_subway_meters':
            feature_display_names.append('Subway Dist')
        elif name == 'dist_to_union_meters':
            feature_display_names.append('Downtown Dist')
        elif name == 'number_of_reviews':
            feature_display_names.append('Reviews')
        elif name == 'reviews_per_month':
            feature_display_names.append('Reviews/Mo')
        elif name == 'minimum_nights':
            feature_display_names.append('Min Nights')
        elif name == 'availability_365':
            feature_display_names.append('Availability')
        elif name == 'calculated_host_listings_count':
            feature_display_names.append('Host Listings')
        elif name == 'review_scores_rating':
            feature_display_names.append('Rating')
        elif name == 'host_is_superhost':
            feature_display_names.append('Superhost')
        elif name == 'privacy_index':
            feature_display_names.append('Privacy Index')
        elif name == 'bathroom_ratio':
            feature_display_names.append('Bath Ratio')
        else:
            feature_display_names.append(name)
    
    # Convert SHAP values from log to approximate dollar impact
    shap_values_scaled = shap_values.values[0] * predicted_price / 2
    
    # Get top contributors (positive and negative)
    indices = np.argsort(np.abs(shap_values_scaled))[::-1][:10]
    
    # Create a simple bar chart showing contributions
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    top_names = [feature_display_names[i] for i in indices]
    top_values = [shap_values_scaled[i] for i in indices]
    
    colors = ['#22c55e' if v > 0 else '#ef4444' for v in top_values]
    
    bars = ax.barh(range(len(top_names)), top_values, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.invert_yaxis()
    
    ax.set_xlabel('Impact on Price ($)', fontsize=10, color='#1a1a1a')
    ax.set_title('Why This Price?', fontsize=12, fontweight='600', color='#1a1a1a', pad=10)
    ax.tick_params(colors='#1a1a1a', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#e5e5e5')
    ax.spines['left'].set_color('#e5e5e5')
    ax.axvline(x=0, color='#e5e5e5', linewidth=1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_values)):
        label = f'+${val:.0f}' if val > 0 else f'-${abs(val):.0f}'
        x_pos = val + (2 if val > 0 else -2)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                 va='center', ha='left' if val > 0 else 'right',
                 fontsize=8, color='#1a1a1a')
    
    plt.tight_layout()
    save_path = "shap_explanation.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    
    return save_path


def create_map_with_heatmap(center: list, df: pd.DataFrame, zoom: int = 12) -> folium.Map:
    """Create a Folium map with clean light tiles."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='cartodbpositron'
    )
    
    # Add Union Station marker
    folium.Marker(
        location=[UNION_STATION_LAT, UNION_STATION_LON],
        popup="Union Station (Transit Hub)",
        icon=folium.Icon(color='red', icon='train', prefix='fa'),
        tooltip="Union Station"
    ).add_to(m)
    
    return m


def main():
    """Main Streamlit application with enhanced UI."""
    st.set_page_config(
        page_title="InsightMap",
        page_icon="",
        layout="wide"
    )
    
    apply_custom_styling()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>InsightMap</h1>
        <p>ML-Powered Airbnb Price Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load Model & Data
    with st.spinner("Loading model and data..."):
        model, feature_names, top_neighbourhoods, top_props, metrics, importances = load_trained_model()
        
        # Load spatial data for predictions
        df = load_and_clean_data(LISTINGS_PATH)
        stations_gdf = load_subway_data(SUBWAY_SHAPEFILE_PATH)
        neighbourhoods_gdf = load_neighbourhood_boundaries(NEIGHBOURHOOD_SHAPEFILE_PATH)
        subway_tree, _ = build_subway_kdtree(stations_gdf)
        union_coords = get_union_station_coords()
        
        # Generate plots (cached)
        importance_plot = plot_feature_importance(importances)
    
    # ==========================
    # SESSION STATE MGMT
    # ==========================
    if 'clicked_coords' not in st.session_state:
        st.session_state.clicked_coords = TORONTO_CENTER

    # ==========================
    # SIDEBAR: Inputs
    # ==========================
    with st.sidebar:
        st.header("Settings")
        
        # Room Type First (as per screenshot)
        room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
        
        st.markdown("### Property Details")
        col_g, col_b = st.columns(2)
        with col_g:
            accommodates = st.number_input("Guests", 1, 16, 2)
        with col_b:
            beds = st.number_input("Beds", 0, 15, 1)
            
        col_bed, col_bath = st.columns(2)
        with col_bed:
            bedrooms = st.number_input("Bedrooms", 0, 10, 1)
        with col_bath:
            bathrooms = st.number_input("Baths", 0.0, 10.0, 1.0, 0.5)

        st.markdown("### Amenities")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            has_wifi = st.checkbox("Wifi", True)
            has_ac = st.checkbox("A/C", False)
        with col_a2:
            has_pool = st.checkbox("Pool", False)
            has_fparking = st.checkbox("Free Parking", False)

        with st.expander("Advanced Settings"):
            prop_type = st.selectbox("Property Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"] + (top_props if top_props else []))
            is_superhost = st.toggle("Superhost", False)
            rating = st.slider("Guest Rating", 0.0, 5.0, 4.8)
            n_reviews = st.number_input("Total Reviews", 0, 1000, 50)
            reviews_per_mo = st.number_input("Reviews/Month", 0.0, 20.0, 1.5)


    # ==========================
    # MAIN TABS
    # ==========================
    tab_pred, tab_insights, tab_data = st.tabs(["Price Predictor", "Model Insights", "Data Analysis"])
    
    # --- TAB 1: Price Predictor ---
    with tab_pred:
        st.markdown("### Click to Predict Price")
        st.caption("Click anywhere on the map to get a predicted nightly price for that location.")
        
        col_map, col_res = st.columns([2, 1], gap="medium")
        
        with col_map:
            # Create Basemap
            m = create_map_with_heatmap(st.session_state.clicked_coords, df)
            
            # Add a marker for the current selected location
            folium.Marker(
                location=st.session_state.clicked_coords,
                popup="Selected Location",
                icon=folium.Icon(color="green", icon="home", prefix='fa')
            ).add_to(m)

            # Render Map & Logic
            map_output = st_folium(m, height=500, width="100%")
            
            if map_output['last_clicked']:
                new_lat = map_output['last_clicked']['lat']
                new_lng = map_output['last_clicked']['lng']
                if (new_lat != st.session_state.clicked_coords[0] or 
                    new_lng != st.session_state.clicked_coords[1]):
                    st.session_state.clicked_coords = [new_lat, new_lng]
                    st.rerun()

        with col_res:
            st.markdown("#### Prediction Result")
            
            # Use coords from state
            clicked_lat = st.session_state.clicked_coords[0]
            clicked_lon = st.session_state.clicked_coords[1]
                
            # Detect Neighbourhood
            detected_hood = detect_neighbourhood(clicked_lat, clicked_lon, neighbourhoods_gdf)
            
            # Run Prediction
            predicted_price, dist_subway, dist_union, X_pred = predict_price_for_location(
                lat=clicked_lat,
                lon=clicked_lon,
                property_type=prop_type,
                room_type=room_type,
                neighbourhood=detected_hood,
                model=model,
                subway_tree=subway_tree,
                union_coords=union_coords,
                feature_names=feature_names,
                top_neighbourhoods=top_neighbourhoods,
                top_props=top_props if top_props else [],
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                accommodates=accommodates,
                beds=beds,
                review_scores_rating=rating,
                number_of_reviews=n_reviews,
                reviews_per_month=reviews_per_mo,
                host_is_superhost=1 if is_superhost else 0,
                has_wifi=1 if has_wifi else 0,
                has_ac=1 if has_ac else 0,
                has_pool=1 if has_pool else 0,
                has_fparking=1 if has_fparking else 0
            )
            
            # Display Prediction
            st.markdown(f"""
            <div class="prediction-box">
                <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Estimated Nightly Price
                </div>
                <div class="prediction-price">${predicted_price:.0f}</div>
                <div style="color: #1a1a1a; font-weight: 500; margin-top: 0.5rem;">
                    {detected_hood}
                </div>
                <div style="color: #888; font-size: 0.8rem; margin-top: 0.25rem;">
                    {dist_subway/1000:.1f}km to Subway • {dist_union/1000:.1f}km to Downtown
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation (SHAP) - Optional in this view or collapsed
            with st.expander("See Price Breakdown"):
                 shap_plot_path = create_shap_waterfall_plot(model, X_pred, predicted_price)
                 st.image(shap_plot_path)

    # --- TAB 2: Model Insights ---
    with tab_insights:
        st.markdown("### Model Performance")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${metrics['rmse']:.0f}</div>
                <div class="metric-label">RMSE (Root Mean Square Error)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${metrics['mae']:.0f}</div>
                <div class="metric-label">MAE (Mean Absolute Error)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['r2']:.1%}</div>
                <div class="metric-label">R² (Variance Explained)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### Feature Importance")
        st.image(importance_plot, use_column_width=True)

    # --- TAB 3: Data Analysis ---
    with tab_data:
        st.markdown("### Data Overview")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("#### Price Distribution")
            # Create a histogram using Streamlit
            price_hist = np.histogram(df['price'], bins=list(range(0, 501, 50)) + [1000])
            hist_df = pd.DataFrame({
                'Price Range': [f"${i}-{i+50}" if i < 500 else "$500+" for i in range(0, 500, 50)] + ["$500+"],
                'Count': price_hist[0]
            })
            st.bar_chart(hist_df.set_index('Price Range'), color="#0068c9")
            
            st.caption(f"Mean Price: ${df['price'].mean():.0f} | Median Price: ${df['price'].median():.0f}")
            
        with col_d2:
            st.markdown("#### Top Neighbourhoods by Count")
            hood_counts = df['neighbourhood'].value_counts().head(10)
            st.bar_chart(hood_counts, color="#0068c9")

        st.markdown("---")
        st.markdown("#### Dataset Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Total Listings", f"{len(df):,}")
        stat_col2.metric("Neighbourhoods", f"{df['neighbourhood'].nunique()}")
        stat_col3.metric("Avg Price", f"${df['price'].mean():.0f}")
        stat_col4.metric("Avg Reviews", f"{df['number_of_reviews'].mean():.0f}")

if __name__ == "__main__":
    main()
