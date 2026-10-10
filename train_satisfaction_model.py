import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

PROC_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\datasets\processed"
MODEL_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\models"

def main():
    print("Loading datasets for Model 4 (Satisfaction/Retention)...")
    zomato = pd.read_parquet(os.path.join(PROC_DIR, "ds9_zomato.parquet"))
    olist_orders = pd.read_parquet(os.path.join(PROC_DIR, "ds13_olist_orders.parquet"))
    olist_reviews = pd.read_parquet(os.path.join(PROC_DIR, "ds13_olist_reviews.parquet"))
    
    # ---------------------------------------------------------
    # 1. Zomato Base Satisfaction Model (Price, Booking, Delivery)
    # ---------------------------------------------------------
    print("Training Zomato Base Satisfaction Model...")
    z_df = zomato[['Aggregate rating', 'Price range', 'Has Table booking', 'Has Online delivery']].dropna()
    z_df['Has Table booking'] = (z_df['Has Table booking'] == 'Yes').astype(int)
    z_df['Has Online delivery'] = (z_df['Has Online delivery'] == 'Yes').astype(int)
    
    X_z = z_df[['Price range', 'Has Table booking', 'Has Online delivery']]
    y_z = z_df['Aggregate rating']
    
    zomato_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    zomato_model.fit(X_z, y_z)
    joblib.dump(zomato_model, os.path.join(MODEL_DIR, "satisfaction_base_model.joblib"))
    print("Zomato model trained and saved.")
    
    # ---------------------------------------------------------
    # 2. Olist Wait Time Penalty Model (Wait Time -> Stars)
    # ---------------------------------------------------------
    print("Training Olist Wait Time Penalty Model...")
    # Merge orders and reviews
    olist = olist_orders[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']].merge(
        olist_reviews[['order_id', 'review_score']], on='order_id', how='inner'
    )
    olist = olist.dropna()
    
    olist['purchase_time'] = pd.to_datetime(olist['order_purchase_timestamp'])
    olist['delivery_time'] = pd.to_datetime(olist['order_delivered_customer_date'])
    olist['wait_time_days'] = (olist['delivery_time'] - olist['purchase_time']).dt.total_seconds() / (24 * 3600)
    
    # Filter reasonable wait times (0 to 60 days)
    olist = olist[(olist['wait_time_days'] >= 0) & (olist['wait_time_days'] <= 60)]
    
    # Fit an OLS model to get the linear penalty of wait time on stars
    X_o = sm.add_constant(olist['wait_time_days'])
    y_o = olist['review_score']
    wait_time_model = sm.OLS(y_o, X_o).fit()
    print("Wait Time OLS Summary:")
    print(wait_time_model.summary().tables[1])
    
    # Extract the slope (penalty per day of wait time)
    wait_time_penalty_per_day = wait_time_model.params['wait_time_days']
    print(f"Wait Time Penalty per Day: {wait_time_penalty_per_day:.4f} stars")
    joblib.dump(wait_time_penalty_per_day, os.path.join(MODEL_DIR, "satisfaction_wait_penalty.joblib"))
    
    print("Model 4 (Satisfaction/Retention) training complete!")

if __name__ == "__main__":
    main()
