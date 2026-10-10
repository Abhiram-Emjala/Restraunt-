"""
Indian city demographic profiles.
Primary sources (where real data is used):
  - Top 500 Indian Cities dataset (user-provided, loaded if available at datasets/14_india_cities/)
  - Living Cost India dataset (user-provided, loaded if available)
Fallback: Curated static profiles — clearly labelled as MODEL ASSUMPTION / Census 2021 Estimate
"""
import os
import pandas as pd
import numpy as np

_LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "14_india_cities"))
DATA_DIR = _LOCAL_DATA_DIR if os.path.exists(_LOCAL_DATA_DIR) else r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\datasets\14_india_cities"

# ---------------------------------------------------------------------------
# Static baseline profiles (MODEL ASSUMPTION — Census 2021 + RBI estimates)
# These are used as the fallback when user datasets are not yet loaded.
# ---------------------------------------------------------------------------
_STATIC_CITY_PROFILES = {
    "Hyderabad": {
        "city": "Hyderabad", "state": "Telangana", "lat": 17.385, "lng": 78.487,
        "population": 10_500_000, "density_per_km2": 18_172,
        "median_income_inr": 420_000, "avg_household_size": 3.8,
        "working_population_pct": 0.42, "youth_pct": 0.34,
        "rent_index": 1.0, "wage_index": 1.0,
        "competition_density": "Very High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Bengaluru": {
        "city": "Bengaluru", "state": "Karnataka", "lat": 12.972, "lng": 77.594,
        "population": 13_200_000, "density_per_km2": 11_900,
        "median_income_inr": 540_000, "avg_household_size": 3.5,
        "working_population_pct": 0.48, "youth_pct": 0.38,
        "rent_index": 1.30, "wage_index": 1.25,
        "competition_density": "Very High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Mumbai": {
        "city": "Mumbai", "state": "Maharashtra", "lat": 19.076, "lng": 72.877,
        "population": 20_700_000, "density_per_km2": 27_350,
        "median_income_inr": 620_000, "avg_household_size": 3.5,
        "working_population_pct": 0.50, "youth_pct": 0.30,
        "rent_index": 2.10, "wage_index": 1.50,
        "competition_density": "Very High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Delhi": {
        "city": "Delhi", "state": "Delhi", "lat": 28.613, "lng": 77.209,
        "population": 32_000_000, "density_per_km2": 11_320,
        "median_income_inr": 510_000, "avg_household_size": 4.0,
        "working_population_pct": 0.46, "youth_pct": 0.31,
        "rent_index": 1.60, "wage_index": 1.30,
        "competition_density": "Very High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Chennai": {
        "city": "Chennai", "state": "Tamil Nadu", "lat": 13.083, "lng": 80.270,
        "population": 11_500_000, "density_per_km2": 26_900,
        "median_income_inr": 450_000, "avg_household_size": 3.6,
        "working_population_pct": 0.45, "youth_pct": 0.32,
        "rent_index": 1.10, "wage_index": 1.05,
        "competition_density": "High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Pune": {
        "city": "Pune", "state": "Maharashtra", "lat": 18.521, "lng": 73.856,
        "population": 7_200_000, "density_per_km2": 5_600,
        "median_income_inr": 480_000, "avg_household_size": 3.5,
        "working_population_pct": 0.47, "youth_pct": 0.38,
        "rent_index": 1.15, "wage_index": 1.10,
        "competition_density": "High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Kolkata": {
        "city": "Kolkata", "state": "West Bengal", "lat": 22.573, "lng": 88.364,
        "population": 15_000_000, "density_per_km2": 24_300,
        "median_income_inr": 360_000, "avg_household_size": 4.2,
        "working_population_pct": 0.40, "youth_pct": 0.28,
        "rent_index": 0.80, "wage_index": 0.85,
        "competition_density": "High",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Ahmedabad": {
        "city": "Ahmedabad", "state": "Gujarat", "lat": 23.030, "lng": 72.587,
        "population": 8_400_000, "density_per_km2": 9_000,
        "median_income_inr": 390_000, "avg_household_size": 4.0,
        "working_population_pct": 0.44, "youth_pct": 0.30,
        "rent_index": 0.85, "wage_index": 0.90,
        "competition_density": "Medium",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Jaipur": {
        "city": "Jaipur", "state": "Rajasthan", "lat": 26.912, "lng": 75.787,
        "population": 4_200_000, "density_per_km2": 6_500,
        "median_income_inr": 310_000, "avg_household_size": 4.5,
        "working_population_pct": 0.38, "youth_pct": 0.29,
        "rent_index": 0.65, "wage_index": 0.75,
        "competition_density": "Medium",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Lucknow": {
        "city": "Lucknow", "state": "Uttar Pradesh", "lat": 26.847, "lng": 80.947,
        "population": 3_800_000, "density_per_km2": 7_200,
        "median_income_inr": 280_000, "avg_household_size": 4.8,
        "working_population_pct": 0.37, "youth_pct": 0.30,
        "rent_index": 0.55, "wage_index": 0.65,
        "competition_density": "Medium",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Coimbatore": {
        "city": "Coimbatore", "state": "Tamil Nadu", "lat": 11.017, "lng": 76.957,
        "population": 2_200_000, "density_per_km2": 4_800,
        "median_income_inr": 340_000, "avg_household_size": 3.5,
        "working_population_pct": 0.44, "youth_pct": 0.31,
        "rent_index": 0.60, "wage_index": 0.78,
        "competition_density": "Low",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
    "Visakhapatnam": {
        "city": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.687, "lng": 83.218,
        "population": 2_100_000, "density_per_km2": 5_200,
        "median_income_inr": 310_000, "avg_household_size": 3.9,
        "working_population_pct": 0.41, "youth_pct": 0.32,
        "rent_index": 0.58, "wage_index": 0.72,
        "competition_density": "Low",
        "data_source": "MODEL ASSUMPTION — Census 2021 Estimate + RBI KLEMS 2023",
    },
}

# Loaded real data (populated at module init if files exist)
_REAL_PROFILES: dict = {}


def _try_load_user_datasets():
    """
    Attempt to load city demographics from user-provided datasets.
    Expects files in datasets/14_india_cities/.
    """
    global _REAL_PROFILES
    if not os.path.isdir(DATA_DIR):
        return

    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(fpath, encoding="utf-8")
            elif fname.endswith((".xlsx", ".xls")):
                df = pd.read_excel(fpath)
            else:
                continue

            # Detect city name column (case-insensitive)
            city_col = next(
                (c for c in df.columns if c.strip().lower() in ("city", "city_name", "name")),
                None,
            )
            if city_col is None:
                continue

            df[city_col] = df[city_col].astype(str).str.strip().str.title()

            # Map columns flexibly
            COL_MAP = {
                "population": ["population", "pop", "total_pop", "total population"],
                "density_per_km2": ["density", "density_per_km2", "population density"],
                "median_income_inr": ["median_income", "avg_income", "income", "per_capita_income", "monthly_income"],
                "rent_index": ["rent_index", "rent", "avg_rent"],
                "lat": ["lat", "latitude"],
                "lng": ["lng", "lon", "longitude"],
                "state": ["state", "state_name"],
            }

            def _find_col(df, candidates):
                for c in candidates:
                    match = next((col for col in df.columns if col.strip().lower() == c), None)
                    if match:
                        return match
                return None

            for city, grp in df.groupby(city_col):
                row = grp.iloc[0]
                profile = _STATIC_CITY_PROFILES.get(city, {}).copy()
                profile["city"] = city
                profile["data_source"] = f"REAL DATA — {fname}"

                for key, candidates in COL_MAP.items():
                    col = _find_col(df, candidates)
                    if col and pd.notna(row[col]):
                        try:
                            val = float(str(row[col]).replace(",", ""))
                            profile[key] = val
                        except ValueError:
                            pass

                _REAL_PROFILES[city] = profile
        except Exception as e:
            print(f"[location_data] Warning: could not load {fname}: {e}")


_try_load_user_datasets()


def get_all_cities() -> list:
    """Returns sorted list of all available city names."""
    combined = {**_STATIC_CITY_PROFILES, **_REAL_PROFILES}
    return sorted(combined.keys())


def get_city_profile(city_name: str) -> dict:
    """
    Return city demographic profile.
    Real data takes priority over static fallback.
    """
    if city_name in _REAL_PROFILES:
        return _REAL_PROFILES[city_name]
    if city_name in _STATIC_CITY_PROFILES:
        return _STATIC_CITY_PROFILES[city_name]
    # Fallback to Hyderabad as default
    p = _STATIC_CITY_PROFILES["Hyderabad"].copy()
    p["city"] = city_name
    p["data_source"] = "MODEL ASSUMPTION — No data available, using Hyderabad proxy"
    return p


def get_all_profiles_df() -> pd.DataFrame:
    """Returns all city profiles as a DataFrame (for map rendering)."""
    combined = {**_STATIC_CITY_PROFILES, **_REAL_PROFILES}
    rows = list(combined.values())
    return pd.DataFrame(rows)


def reload_user_datasets():
    """Call this after user drops new files into datasets/14_india_cities/."""
    global _REAL_PROFILES
    _REAL_PROFILES = {}
    _try_load_user_datasets()
