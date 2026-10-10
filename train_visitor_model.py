import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

PROC_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\datasets\processed"
MODEL_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\models"

def main():
    print("Loading preprocessed Recruit data...")
    df = pd.read_parquet(os.path.join(PROC_DIR, "ds1_recruit_merged.parquet"))
    
    print(f"Data shape: {df.shape}")
    
    # Fill NAs
    df['holiday_flg'] = df['holiday_flg'].fillna(0)
    
    # Drop rows without visitors (if any)
    df = df.dropna(subset=['visitors'])
    
    # Feature Engineering
    print("Engineering features...")
    # Add is_weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # We want to predict visitors. What features do we have?
    # air_store_id, visit_date, visitors, air_genre_name, air_area_name, latitude, longitude
    # day_of_week, holiday_flg, month, year
    
    features = [
        'day_of_week', 'is_weekend', 'holiday_flg', 'month', 'year',
        'latitude', 'longitude'
    ]
    
    categorical_features = ['air_genre_name', 'air_area_name']
    
    # Label encode categoricals
    le_dict = {}
    for cat in categorical_features:
        df[cat] = df[cat].fillna('Unknown')
        le = LabelEncoder()
        df[cat + '_encoded'] = le.fit_transform(df[cat])
        le_dict[cat] = le
        features.append(cat + '_encoded')
        
    print(f"Features used: {features}")
    
    # We will use log1p for the target to handle right-skewed visitor counts
    X = df[features]
    y = np.log1p(df['visitors'])
    
    # Time-based split: Use 2017 data for testing, 2016 for training
    train_mask = df['year'] == 2016
    test_mask = df['year'] == 2017
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Train XGBoost Model
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_true = np.expm1(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
    mae = mean_absolute_error(y_test_true, y_pred)
    r2 = r2_score(y_test_true, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.4f}")
    
    # Save model and encoders
    print("Saving model and encoders...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'visitor_demand_model.joblib'))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, 'visitor_label_encoders.joblib'))
    joblib.dump(features, os.path.join(MODEL_DIR, 'visitor_model_features.joblib'))
    
    # Feature importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feat_imp.to_string(index=False))
    feat_imp.to_csv(os.path.join(MODEL_DIR, 'visitor_feature_importance.csv'), index=False)
    
    print("Model 1 training complete!")

if __name__ == "__main__":
    main()
