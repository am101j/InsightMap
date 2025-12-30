"""
InsightMap - Model Training Script
===================================

Run this script ONCE to train and save the model.
The saved model will be loaded by app.py for predictions.

Usage:
    python train_model.py

Output:
    - model/xgboost_model.json (trained model)
    - model/feature_names.json (feature names for prediction)
    - model/top_neighbourhoods.json (neighbourhood mapping)
    - model/metrics.json (model performance metrics)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os
import warnings

warnings.filterwarnings('ignore')


# Configuration

LISTINGS_PATH = "listings.csv"
SUBWAY_SHAPEFILE_PATH = "subway_data/TTC_SUBWAY_LINES_WGS84.shp"
MODEL_DIR = "model"
MAX_PRICE = 1000
TOP_N_NEIGHBOURHOODS = 15
UNION_STATION_LAT = 43.6456
UNION_STATION_LON = -79.3806


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean the Airbnb listings data."""
    print("[1/5] Loading listings data...")
    
    df = pd.read_csv(filepath, on_bad_lines='skip')
    print(f"   Loaded {len(df):,} rows")
    
    # Clean price column
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
    
    # Clean coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Clean additional features
    df['number_of_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce').fillna(0)
    df['reviews_per_month'] = pd.to_numeric(df['reviews_per_month'], errors='coerce').fillna(0)
    df['minimum_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce').fillna(1)
    df['minimum_nights'] = df['minimum_nights'].clip(upper=365)
    df['availability_365'] = pd.to_numeric(df['availability_365'], errors='coerce').fillna(0)
    df['calculated_host_listings_count'] = pd.to_numeric(
        df['calculated_host_listings_count'], errors='coerce'
    ).fillna(1)
    df['neighbourhood'] = df['neighbourhood'].fillna('Unknown')
    
    # Filter outliers
    df = df[df['price'] < MAX_PRICE]
    print(f"   After cleaning: {len(df):,} rows")
    
    return df.reset_index(drop=True)


def load_subway_data(filepath: str) -> gpd.GeoDataFrame:
    """Load subway shapefile and extract station points."""
    print("[2/5] Loading subway data...")
    
    subway_gdf = gpd.read_file(filepath)
    all_points = []
    
    for geom in subway_gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == 'LineString':
            all_points.extend([Point(c) for c in list(geom.coords)])
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                all_points.extend([Point(c) for c in list(line.coords)])
        elif geom.geom_type == 'Point':
            all_points.append(geom)
    
    stations_gdf = gpd.GeoDataFrame({'geometry': all_points}, crs="EPSG:4326")
    stations_gdf = stations_gdf.to_crs("EPSG:32617")
    print(f"   Extracted {len(all_points)} subway vertices")
    
    return stations_gdf


def engineer_spatial_features(df: pd.DataFrame, stations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Engineer spatial features for the model."""
    print("[3/5] Engineering spatial features...")
    
    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    housing_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    housing_gdf = housing_gdf.to_crs("EPSG:32617")
    
    # Build subway KDTree
    station_coords = np.array([[geom.x, geom.y] for geom in stations_gdf.geometry])
    subway_tree = cKDTree(station_coords)
    
    # Calculate distances
    housing_coords = np.array([[geom.x, geom.y] for geom in housing_gdf.geometry])
    
    # Distance to nearest subway
    distances, _ = subway_tree.query(housing_coords)
    housing_gdf['dist_to_subway_meters'] = distances
    
    # Distance to Union Station
    union_point = gpd.GeoDataFrame(
        {'geometry': [Point(UNION_STATION_LON, UNION_STATION_LAT)]},
        crs="EPSG:4326"
    ).to_crs("EPSG:32617")
    union_x, union_y = union_point.geometry.iloc[0].x, union_point.geometry.iloc[0].y
    
    dist_to_union = np.sqrt(
        (housing_coords[:, 0] - union_x) ** 2 +
        (housing_coords[:, 1] - union_y) ** 2
    )
    housing_gdf['dist_to_union_meters'] = dist_to_union
    
    print(f"   Added spatial features to {len(housing_gdf):,} listings")
    return housing_gdf


def prepare_features(gdf: gpd.GeoDataFrame) -> tuple:
    """Prepare feature matrix for training."""
    print("[4/5] Preparing features...")
    
    # Numeric features
    numeric_features = [
        'dist_to_subway_meters',
        'dist_to_union_meters',
        'number_of_reviews',
        'reviews_per_month',
        'minimum_nights',
        'availability_365',
        'calculated_host_listings_count'
    ]
    
    X = gdf[numeric_features].copy()
    
    # One-hot encode room_type
    if 'room_type' in gdf.columns:
        room_dummies = pd.get_dummies(gdf['room_type'], prefix='room')
        X = pd.concat([X, room_dummies], axis=1)
    
    # One-hot encode top neighbourhoods
    top_neighbourhoods = gdf['neighbourhood'].value_counts().head(TOP_N_NEIGHBOURHOODS).index.tolist()
    gdf['neighbourhood_grouped'] = gdf['neighbourhood'].apply(
        lambda x: x if x in top_neighbourhoods else 'Other'
    )
    neighbourhood_dummies = pd.get_dummies(gdf['neighbourhood_grouped'], prefix='hood')
    X = pd.concat([X, neighbourhood_dummies], axis=1)
    
    y = gdf['price']
    
    # Drop NaN values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Log transformation for better model performance
    y_log = np.log1p(y)
    
    print(f"   Feature matrix: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    return X, y_log, y, list(X.columns), top_neighbourhoods


def train_model(X: pd.DataFrame, y_log: pd.Series, y_original: pd.Series) -> tuple:
    """Train XGBoost model."""
    print("\n[5/5] Training XGBoost model...")
    print("   Algorithm: XGBoost (Gradient Boosting)")
    print("   Target: Log-transformed prices")
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # XGBoost configuration
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"   Training on {len(X_train):,} samples...")
    model.fit(X_train, y_train_log)
    
    # Evaluate
    y_pred_log = model.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test_log)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    r2 = r2_score(y_test_dollars, y_pred_dollars)
    
    print(f"\n>> Model Performance:")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   R^2: {r2:.1%}")
    
    metrics = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
    importances = dict(zip(X.columns, [float(x) for x in model.feature_importances_]))
    
    return model, metrics, importances


def save_model(model, feature_names, top_neighbourhoods, metrics, importances):
    """Save model and metadata to disk."""
    print(f"\nSaving model to '{MODEL_DIR}/' directory...")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save XGBoost model
    model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    model.get_booster().save_model(model_path)
    print(f"   + Model saved: {model_path}")
    
    # Save feature names
    features_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   + Features saved: {features_path}")
    
    # Save top neighbourhoods
    neighbourhoods_path = os.path.join(MODEL_DIR, "top_neighbourhoods.json")
    with open(neighbourhoods_path, 'w') as f:
        json.dump(top_neighbourhoods, f, indent=2)
    print(f"   + Neighbourhoods saved: {neighbourhoods_path}")
    
    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   + Metrics saved: {metrics_path}")
    
    # Save feature importances
    importances_path = os.path.join(MODEL_DIR, "feature_importances.json")
    with open(importances_path, 'w') as f:
        json.dump(importances, f, indent=2)
    print(f"   + Importances saved: {importances_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("TorontoGeoHome - Model Training Pipeline")
    print("=" * 60)
    print()
    
    # Load data
    df = load_and_clean_data(LISTINGS_PATH)
    stations_gdf = load_subway_data(SUBWAY_SHAPEFILE_PATH)
    
    # Engineer features
    housing_gdf = engineer_spatial_features(df, stations_gdf)
    
    # Prepare features
    X, y_log, y_original, feature_names, top_neighbourhoods = prepare_features(housing_gdf)
    
    # Train model
    model, metrics, importances = train_model(X, y_log, y_original)
    
    # Save everything
    save_model(model, feature_names, top_neighbourhoods, metrics, importances)
    
    print("\nTraining complete! Model saved to 'model/' directory.")
    print("Run 'streamlit run app.py' to start the prediction app.")


if __name__ == "__main__":
    main()
