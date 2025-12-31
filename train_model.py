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
    - model/top_property_types.json (property type mapping)
    - model/metrics.json (model performance metrics)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# Configuration

LISTINGS_PATH = "listings.csv.gz"
SUBWAY_SHAPEFILE_PATH = "subway_data/TTC_SUBWAY_LINES_WGS84.shp"
MODEL_DIR = "model"
MAX_PRICE = 1000
TOP_N_NEIGHBOURHOODS = 15
TOP_N_PROPERTY_TYPES = 10
UNION_STATION_LAT = 43.6456
UNION_STATION_LON = -79.3806


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean the Airbnb listings data."""
    print("[1/5] Loading listings data...")
    
    # Check if gzip
    if filepath.endswith('.gz'):
        df = pd.read_csv(filepath, compression='gzip', on_bad_lines='skip', low_memory=False)
    else:
        df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
    
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
    df['calculated_host_listings_count'] = pd.to_numeric(df['calculated_host_listings_count'], errors='coerce').fillna(1)
    
    # NEW: Review Scores
    review_cols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_value']
    for col in review_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill missing review scores with median (neutral assumption better than 0)
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 4.5  # Fallback default

    # NEW: Host Features
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)
    else:
        df['host_is_superhost'] = 0
        
    if 'host_identity_verified' in df.columns:
        df['host_identity_verified'] = df['host_identity_verified'].apply(lambda x: 1 if x == 't' else 0)
    else:
        df['host_identity_verified'] = 0

    if 'instant_bookable' in df.columns:
        df['instant_bookable'] = df['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)
    else:
        df['instant_bookable'] = 0

    # New Features from Detailed Data
    df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce').fillna(2)
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(1)
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce').fillna(1)
    
    # Extract bathrooms from text (e.g. "1.5 baths" -> 1.5)
    if 'bathrooms_text' in df.columns:
        df['bathrooms'] = df['bathrooms_text'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    df['bathrooms'] = df['bathrooms'].fillna(1)

    # Simplified Amenities Parsing
    # We look for keywords like "Air conditioning", "Pool", "Wifi"
    if 'amenities' in df.columns:
        df['amenities'] = df['amenities'].astype(str).str.lower()
        df['has_pool'] = df['amenities'].apply(lambda x: 1 if 'pool' in x else 0)
        df['has_ac'] = df['amenities'].apply(lambda x: 1 if 'air conditioning' in x or 'ac' in x else 0)
        df['has_fparking'] = df['amenities'].apply(lambda x: 1 if 'free parking' in x else 0)
        df['has_wifi'] = df['amenities'].apply(lambda x: 1 if 'wifi' in x else 0)
    else:
        df['has_pool'] = 0
        df['has_ac'] = 0
        df['has_fparking'] = 0
        df['has_wifi'] = 0
        df['has_wifi'] = 0

    # FIX: Use cleansed neighbourhood name for accurate location (Re-applied)
    if 'neighbourhood_cleansed' in df.columns:
        df['neighbourhood'] = df['neighbourhood_cleansed']

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
    print("[4/5] Preparing features & engineering interactions...")
    
    # NEW: Interaction Features
    # Privacy Index: Low value = crowded/dorm, High = luxury/private
    gdf['privacy_index'] = gdf['bedrooms'] / (gdf['accommodates'] + 0.1)
    
    # Bathroom Ratio: Bathrooms per bedroom
    gdf['bathroom_ratio'] = gdf['bathrooms'] / (gdf['bedrooms'] + 0.1)

    numeric_features = [
        'dist_to_subway_meters',
        'dist_to_union_meters',
        'number_of_reviews',
        'reviews_per_month',
        'minimum_nights',
        'availability_365',
        'calculated_host_listings_count',
        'accommodates',
        'bedrooms',
        'beds',
        'bathrooms',
        'has_pool',
        'has_ac',
        'has_fparking',
        'has_wifi',
        # New Numeric/Boolean
        'review_scores_rating',
        'review_scores_cleanliness',
        'review_scores_location',
        'review_scores_value',
        'host_is_superhost',
        'host_identity_verified',
        'instant_bookable',
        # New Interactions
        'privacy_index',
        'bathroom_ratio'
    ]
    
    X = gdf[numeric_features].copy()
    
    # One-hot encode property_type (Top N + Other) instead of just room_type
    if 'property_type' in gdf.columns:
        top_props = gdf['property_type'].value_counts().head(TOP_N_PROPERTY_TYPES).index.tolist()
        gdf['property_group'] = gdf['property_type'].apply(lambda x: x if x in top_props else 'Other')
        prop_dummies = pd.get_dummies(gdf['property_group'], prefix='prop')
        X = pd.concat([X, prop_dummies], axis=1)

    # Keep room_type as well, it's very predictive
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
    
    return X, y_log, y, list(X.columns), top_neighbourhoods, top_props if 'property_type' in gdf.columns else []


def train_model(X: pd.DataFrame, y_log: pd.Series, y_original: pd.Series) -> tuple:
    """Train XGBoost model with RandomizedSearchCV."""
    print("\n[5/5] Training XGBoost model with Randomized Search...")
    print("   Algorithm: XGBoost (Gradient Boosting)")
    print("   Target: Log-transformed prices")
    print("   Tuning: Finding best hyperparameters (this takes 30-60s)...")
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # Hyperparameter Grid
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9, 11],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    # Base Model
    xgb_reg = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        n_iter=20,  # 20 random combinations
        scoring='r2',
        cv=3,       # 3-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train_log)
    
    print(f"\n   >> Best Parameters Found: {random_search.best_params_}")
    
    model = random_search.best_estimator_
    
    # Evaluate
    y_pred_log = model.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test_log)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    r2 = r2_score(y_test_dollars, y_pred_dollars)
    
    # Generate plots
    plot_prediction_scatter(y_test_dollars, y_pred_dollars)
    plot_price_distribution(y_original)
    
    print(f"\n>> New Model Performance:")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   R^2: {r2:.1%}")
    
    metrics = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
    importances = dict(zip(X.columns, [float(x) for x in model.feature_importances_]))
    
    return model, metrics, importances


def plot_prediction_scatter(y_test, y_pred):
    """Generate and save actual vs predicted scatter plot."""
    plt.figure(figsize=(8, 8), facecolor='#ffffff')
    plt.scatter(y_test, y_pred, alpha=0.4, s=12, c='#22c55e', edgecolors='none')
    
    # Perfect prediction line
    max_val = max(y_test.max(), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='#ef4444', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Actual vs Predicted Prices', fontsize=14, fontweight='bold', pad=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(MODEL_DIR, "prediction_scatter.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   + Scatter plot saved: {save_path}")


def plot_price_distribution(prices):
    """Generate and save price distribution plot."""
    plt.figure(figsize=(10, 6), facecolor='#ffffff')
    plt.hist(prices, bins=50, color='#22c55e', alpha=0.7, edgecolor='none')
    
    plt.axvline(prices.mean(), color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: ${prices.mean():.0f}')
    plt.axvline(prices.median(), color='#3b82f6', linestyle='--', linewidth=2, label=f'Median: ${prices.median():.0f}')
    
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Price Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    save_path = os.path.join(MODEL_DIR, "price_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   + Distribution plot saved: {save_path}")


def save_model(model, feature_names, top_neighbourhoods, top_props, metrics, importances):
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

    # Save top property types
    props_path = os.path.join(MODEL_DIR, "top_property_types.json")
    with open(props_path, 'w') as f:
        json.dump(top_props, f, indent=2)
    print(f"   + Property types saved: {props_path}")
    
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
    print("TorontoGeoHome - Advanced Model Training")
    print("=" * 60)
    print()
    
    # Load data
    df = load_and_clean_data(LISTINGS_PATH)
    stations_gdf = load_subway_data(SUBWAY_SHAPEFILE_PATH)
    
    # Engineer features
    housing_gdf = engineer_spatial_features(df, stations_gdf)
    
    # Prepare features
    X, y_log, y_original, feature_names, top_neighbourhoods, top_props = prepare_features(housing_gdf)
    
    # Train model
    model, metrics, importances = train_model(X, y_log, y_original)
    
    # Save everything
    save_model(model, feature_names, top_neighbourhoods, top_props, metrics, importances)
    
    print("\nTraining complete! Model saved to 'model/' directory.")
    print("Run 'streamlit run app.py' to start the prediction app.")


if __name__ == "__main__":
    main()
