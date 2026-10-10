import pandas as pd
import numpy as np
import os
import joblib

PROC_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\datasets\processed"
MODEL_DIR = r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\models"

def main():
    print("Loading datasets for Model 3 (Menu/Product Demand)...")
    pizza = pd.read_parquet(os.path.join(PROC_DIR, "ds12_pizza_sales.parquet"))
    bb = pd.read_parquet(os.path.join(PROC_DIR, "ds11_bread_basket.parquet"))
    
    # 1. Category Popularity from Pizza
    cat_pop = pizza['pizza_category'].value_counts(normalize=True).to_dict()
    print("Pizza Category Popularity:", cat_pop)
    joblib.dump(cat_pop, os.path.join(MODEL_DIR, "menu_category_pop.joblib"))
    
    # 2. Size Popularity from Pizza
    size_pop = pizza['pizza_size'].value_counts(normalize=True).to_dict()
    print("Pizza Size Popularity:", size_pop)
    joblib.dump(size_pop, os.path.join(MODEL_DIR, "menu_size_pop.joblib"))
    
    # 3. Time of day demand shifts from Bread Basket (Morning vs Afternoon vs Evening)
    # Using 'hour' to define periods
    bb['period'] = pd.cut(bb['hour'], bins=[0, 11, 16, 24], labels=['Morning', 'Afternoon', 'Evening/Night'])
    time_pop = bb['period'].value_counts(normalize=True).to_dict()
    print("Time of Day Popularity (Bakery):", time_pop)
    joblib.dump(time_pop, os.path.join(MODEL_DIR, "menu_time_pop.joblib"))
    
    print("Model 3 (Menu Demand) empirical models saved successfully!")

if __name__ == "__main__":
    main()
