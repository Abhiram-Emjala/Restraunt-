import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

PROC_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\datasets\processed"
MODEL_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\models"

def main():
    print("Loading datasets for Model 2 (Spending)...")
    pizza = pd.read_parquet(os.path.join(PROC_DIR, "ds12_pizza_sales.parquet"))
    tips = pd.read_parquet(os.path.join(PROC_DIR, "ds3_tips.parquet"))
    
    # 1. Party Size Distribution from Tips
    party_size_dist = tips['size'].value_counts(normalize=True).sort_index().to_dict()
    print("Party Size Distribution:", party_size_dist)
    joblib.dump(party_size_dist, os.path.join(MODEL_DIR, "party_size_dist.joblib"))
    
    # 2. Order Value Modeling from Pizza
    print(f"Pizza Data shape: {pizza.shape}")
    
    # Aggregate to order level
    order_level = pizza.groupby('order_id').agg({
        'total_price_inr': 'sum',
        'quantity': 'sum',
        'order_dayofweek': 'first',
        'order_month': 'first',
        'order_hour': 'first'
    }).reset_index()
    
    print(f"Aggregated to {order_level.shape[0]} unique orders.")
    
    # Drop NAs
    order_level = order_level.dropna(subset=['order_hour'])
    
    # Features for Order Value Model
    features = ['order_dayofweek', 'order_month', 'order_hour']
    X = order_level[features]
    y = np.log1p(order_level['total_price_inr']) # log transform due to skew
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Order Value Regressor...")
    # Use RandomForest for simplicity and non-linearity
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_true = np.expm1(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
    mae = mean_absolute_error(y_test_true, y_pred)
    
    print(f"RMSE: {rmse:.2f} INR")
    print(f"MAE:  {mae:.2f} INR")
    
    # The R2 will be low because order value has high inherent variance not explained just by time
    print(f"R2:   {r2_score(y_test_true, y_pred):.4f}")
    
    # Save the order value model
    joblib.dump(model, os.path.join(MODEL_DIR, 'order_value_model.joblib'))
    joblib.dump(features, os.path.join(MODEL_DIR, 'order_value_features.joblib'))
    
    # 3. Fit a Gamma distribution to the residuals (for realistic simulation noise)
    residuals = y_test_true - y_pred
    std_dev = np.std(residuals)
    print(f"Residual Std Dev: {std_dev:.2f} INR")
    joblib.dump(std_dev, os.path.join(MODEL_DIR, 'order_value_residual_std.joblib'))
    
    print("Model 2 (Spending) training complete!")

if __name__ == "__main__":
    main()
