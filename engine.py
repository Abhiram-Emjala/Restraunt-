"""
Refactored Simulation Engine — reads from BusinessState, outputs structured daily results.
All ML models loaded once at startup and reused across simulations.
"""
import os
import datetime
import numpy as np
import pandas as pd
import joblib

from src.simulation.state import BusinessState
from src.simulation.location_data import get_city_profile

_LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
MODEL_DIR = _LOCAL_MODEL_DIR if os.path.exists(_LOCAL_MODEL_DIR) else r"c:\Users\abhir\.gemini\antigravity\scratch\data-driven VRBS\models"

# ---------------------------------------------------------------------------
# Marketing Engine (MODEL ASSUMPTION — parametric)
# ---------------------------------------------------------------------------

def simulate_marketing(state: BusinessState) -> dict:
    """
    Compute monthly marketing outcomes from budget allocation.
    Returns impressions, clicks, new_customers_from_marketing, total_cac, channel_breakdown.
    MODEL ASSUMPTION: channel parameters based on IAMAI India 2023 benchmarks.
    """
    mk = state.marketing
    total_budget = mk.monthly_budget_inr
    params = mk.channel_params
    channels = mk.channels

    channel_results = {}
    total_new_customers = 0
    total_spend = 0

    for ch, fraction in channels.items():
        if ch == "word_of_mouth":
            continue  # Organic — not budget-driven
        p = params.get(ch, {})
        spend = total_budget * fraction
        if spend <= 0 or not p:
            continue

        cpm = p.get("cpm_inr", 100)
        ctr = p.get("ctr", 0.02)
        cvr = p.get("conversion_rate", 0.03)

        impressions = int((spend / max(cpm, 1)) * 1000)
        clicks = int(impressions * ctr)
        conversions = int(clicks * cvr)

        channel_results[ch] = {
            "spend_inr": round(spend, 0),
            "impressions": impressions,
            "clicks": clicks,
            "new_customers": conversions,
            "cac_inr": round(spend / max(conversions, 1), 0),
        }
        total_new_customers += conversions
        total_spend += spend

    # Word-of-mouth: proportional to satisfaction (added by caller)
    overall_cac = round(total_spend / max(total_new_customers, 1), 0)

    return {
        "channel_breakdown": channel_results,
        "total_new_customers_from_marketing": total_new_customers,
        "overall_cac_inr": overall_cac,
        "total_marketing_spend_inr": total_spend,
        "marketing_roi_pct": 0.0,  # filled after revenue is computed
        "data_source": mk.data_source,
    }


# ---------------------------------------------------------------------------
# Model Loader (singleton)
# ---------------------------------------------------------------------------

_models_cache = None


def _load_models():
    global _models_cache
    if _models_cache is not None:
        return _models_cache
    _models_cache = {
        "visitor_model": joblib.load(os.path.join(MODEL_DIR, "visitor_demand_model.joblib")),
        "visitor_le": joblib.load(os.path.join(MODEL_DIR, "visitor_label_encoders.joblib")),
        "visitor_features": joblib.load(os.path.join(MODEL_DIR, "visitor_model_features.joblib")),
        "party_size_dist": joblib.load(os.path.join(MODEL_DIR, "party_size_dist.joblib")),
        "order_value_model": joblib.load(os.path.join(MODEL_DIR, "order_value_model.joblib")),
        "order_value_std": joblib.load(os.path.join(MODEL_DIR, "order_value_residual_std.joblib")),
        "sat_base_model": joblib.load(os.path.join(MODEL_DIR, "satisfaction_base_model.joblib")),
        "wait_penalty": joblib.load(os.path.join(MODEL_DIR, "satisfaction_wait_penalty.joblib")),
        "menu_cat_pop": joblib.load(os.path.join(MODEL_DIR, "menu_category_pop.joblib")),
        "menu_time_pop": joblib.load(os.path.join(MODEL_DIR, "menu_time_pop.joblib")),
    }
    return _models_cache


# ---------------------------------------------------------------------------
# Daily step
# ---------------------------------------------------------------------------

def _simulate_day(date: pd.Timestamp, state: BusinessState, models: dict,
                  daily_new_from_marketing: int) -> dict:
    ops = state.operations
    cust = state.customers
    loc = get_city_profile(state.location.city)

    # --- 1. DEMAND (Model 1: XGBoost visitor model) ---
    dow = date.dayofweek
    month = date.month
    is_weekend = int(dow in [5, 6])
    holiday_flg = 0  # Could be extended with holiday calendar

    try:
        genre_encoded = models["visitor_le"]["air_genre_name"].transform([state.identity.genre])[0]
    except ValueError:
        genre_encoded = 0
    try:
        area_encoded = models["visitor_le"]["air_area_name"].transform([state.location.city])[0]
    except ValueError:
        area_encoded = 0

    input_df = pd.DataFrame([{
        "day_of_week": dow, "is_weekend": is_weekend, "holiday_flg": holiday_flg,
        "month": month, "year": 2017,
        "latitude": state.location.lat, "longitude": state.location.lng,
        "air_genre_name_encoded": genre_encoded,
        "air_area_name_encoded": area_encoded,
    }])[models["visitor_features"]]

    base_visitors = max(0, int(np.expm1(models["visitor_model"].predict(input_df)[0])))

    # Location income modifier: higher income city → higher spending but similar base footfall
    income_mod = np.clip(loc["median_income_inr"] / 420_000, 0.6, 2.0)

    # Marketing daily uplift
    marketing_uplift = daily_new_from_marketing

    # Retention visitors (returning customers)
    retention_visitors = int(base_visitors * cust.retention_rate * income_mod * 0.3)

    total_potential = base_visitors + marketing_uplift + retention_visitors

    # --- 2. CAPACITY & WAIT TIME ---
    max_daily_seats = ops.capacity_seats * int(ops.operating_hours)
    ideal_staff = total_potential / 15.0
    staff_ratio = ops.staff_count / max(1, ideal_staff)

    wait_hours = 0.0
    if staff_ratio < 1.0:
        wait_hours += (1.0 - staff_ratio) * 1.5
    if total_potential > max_daily_seats:
        wait_hours += (total_potential - max_daily_seats) / ops.capacity_seats

    balking_rate = min(1.0, wait_hours / 2.0)
    effective_visitors = max(0, int(total_potential * (1 - balking_rate)))
    lost_customers = total_potential - effective_visitors
    capacity_utilization_pct = min(100.0, (effective_visitors / max(1, max_daily_seats)) * 100)

    # --- 3. REVENUE (Model 2: RF order value model) ---
    num_parties = max(1, int(effective_visitors / cust.avg_party_size))

    order_input = pd.DataFrame([{
        "order_dayofweek": dow, "order_month": month, "order_hour": 13
    }])
    base_order_log = models["order_value_model"].predict(order_input)[0]
    base_order_value = np.expm1(base_order_log)

    # Price positioning modifier
    price_mods = {"Budget": 0.55, "Mid-Range": 1.0, "Premium": 1.65, "Luxury": 2.80}
    price_mod = price_mods.get(state.products.price_positioning, 1.0)

    # Income modifier on spending
    spend_mod = price_mod * np.clip(income_mod, 0.7, 1.8)

    order_values = np.random.normal(
        loc=base_order_value * spend_mod,
        scale=models["order_value_std"] * spend_mod * 0.4,
        size=num_parties,
    )
    order_values = np.clip(order_values, 80, None)
    revenue = float(np.sum(order_values))
    avg_order_value = float(np.mean(order_values))

    # --- 4. COSTS ---
    cogs = revenue * ops.cogs_pct
    labor_cost_daily = (ops.staff_count * 150 * ops.operating_hours * loc.get("wage_index", 1.0))
    fixed_daily = (ops.rent_monthly_inr + ops.utilities_monthly_inr +
                   ops.equipment_monthly_inr + ops.other_fixed_monthly_inr) / 30.0
    marketing_daily = state.marketing.monthly_budget_inr / 30.0
    total_cost = cogs + labor_cost_daily + fixed_daily + marketing_daily

    gross_profit = revenue - cogs
    operating_profit = revenue - cogs - labor_cost_daily - fixed_daily - marketing_daily
    net_profit = operating_profit  # simplified (no tax/depreciation for now)
    gross_margin_pct = (gross_profit / max(revenue, 1)) * 100
    net_margin_pct = (net_profit / max(revenue, 1)) * 100

    # --- 5. SATISFACTION (Model 4) ---
    price_range = {"Budget": 1, "Mid-Range": 2, "Premium": 3, "Luxury": 4}.get(
        state.products.price_positioning, 2
    )
    sat_input = pd.DataFrame([{
        "Price range": price_range,
        "Has Table booking": 1,
        "Has Online delivery": 1,
    }])
    base_stars = models["sat_base_model"].predict(sat_input)[0]
    wait_penalty = (wait_hours / 24.0) * models["wait_penalty"]
    quality_bonus = (3.5 - 3.0) * 0.3 + (3.5 - 3.0) * 0.2  # placeholder quality
    satisfaction = float(np.clip(base_stars + wait_penalty + quality_bonus, 1.0, 5.0))

    # --- 6. CUSTOMER METRICS ---
    new_customers = int(effective_visitors * cust.new_customer_fraction)
    returning_customers = effective_visitors - new_customers
    clv = avg_order_value * cust.avg_visit_frequency_monthly * cust.customer_lifetime_months

    return {
        "date": date.strftime("%Y-%m-%d"),
        "day_of_week": dow,
        "is_weekend": is_weekend,
        # Visitors
        "potential_visitors": total_potential,
        "effective_visitors": effective_visitors,
        "lost_customers": lost_customers,
        "new_customers": new_customers,
        "returning_customers": returning_customers,
        "wait_time_hours": round(wait_hours, 2),
        "capacity_utilization_pct": round(capacity_utilization_pct, 1),
        # Revenue
        "revenue_inr": round(revenue, 0),
        "avg_order_value_inr": round(avg_order_value, 0),
        # Costs
        "cogs_inr": round(cogs, 0),
        "labor_cost_inr": round(labor_cost_daily, 0),
        "fixed_cost_inr": round(fixed_daily, 0),
        "marketing_cost_inr": round(marketing_daily, 0),
        "total_cost_inr": round(total_cost, 0),
        # Profit
        "gross_profit_inr": round(gross_profit, 0),
        "operating_profit_inr": round(operating_profit, 0),
        "net_profit_inr": round(net_profit, 0),
        "gross_margin_pct": round(gross_margin_pct, 1),
        "net_margin_pct": round(net_margin_pct, 1),
        # Quality
        "satisfaction_stars": round(satisfaction, 2),
        "clv_inr": round(clv, 0),
    }


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(state: BusinessState) -> dict:
    """
    Run the full simulation for state.sim_config.simulation_days days.
    Returns a dict with daily_df (DataFrame) + summary + marketing_results.
    """
    models = _load_models()
    start = pd.to_datetime(state.sim_config.start_date)
    days = state.sim_config.simulation_days

    # Marketing runs monthly — compute once and distribute daily
    marketing_results = simulate_marketing(state)
    daily_marketing_customers = marketing_results["total_new_customers_from_marketing"] / 30.0

    daily_rows = []
    for i in range(days):
        date = start + datetime.timedelta(days=i)
        row = _simulate_day(date, state, models, int(daily_marketing_customers))
        daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows)

    # Compute marketing ROI
    total_revenue = daily_df["revenue_inr"].sum()
    mkt_spend = marketing_results["total_marketing_spend_inr"]
    # Attribute marketing revenue = new customers * avg order value * marketing share
    mkt_attributed_rev = (
        marketing_results["total_new_customers_from_marketing"]
        * daily_df["avg_order_value_inr"].mean()
    )
    marketing_results["marketing_roi_pct"] = round(
        ((mkt_attributed_rev - mkt_spend) / max(mkt_spend, 1)) * 100, 1
    )

    # Summary
    summary = {
        "total_revenue_inr": round(daily_df["revenue_inr"].sum(), 0),
        "total_cost_inr": round(daily_df["total_cost_inr"].sum(), 0),
        "total_gross_profit_inr": round(daily_df["gross_profit_inr"].sum(), 0),
        "total_net_profit_inr": round(daily_df["net_profit_inr"].sum(), 0),
        "avg_gross_margin_pct": round(daily_df["gross_margin_pct"].mean(), 1),
        "avg_net_margin_pct": round(daily_df["net_margin_pct"].mean(), 1),
        "total_customers_served": int(daily_df["effective_visitors"].sum()),
        "total_new_customers": int(daily_df["new_customers"].sum()),
        "total_lost_customers": int(daily_df["lost_customers"].sum()),
        "avg_satisfaction": round(daily_df["satisfaction_stars"].mean(), 2),
        "avg_wait_time_hours": round(daily_df["wait_time_hours"].mean(), 2),
        "avg_capacity_utilization_pct": round(daily_df["capacity_utilization_pct"].mean(), 1),
        "avg_order_value_inr": round(daily_df["avg_order_value_inr"].mean(), 0),
        "simulation_days": days,
        "location": state.location.city,
        "business_name": state.identity.business_name,
        "health": _health_signal(daily_df),
    }

    return {
        "daily_df": daily_df,
        "summary": summary,
        "marketing": marketing_results,
    }


def _health_signal(df: pd.DataFrame) -> str:
    avg_margin = df["net_margin_pct"].mean()
    trend = df["net_profit_inr"].iloc[-7:].mean() - df["net_profit_inr"].iloc[:7].mean()
    if avg_margin < 0:
        return "At Risk"
    elif avg_margin < 8:
        return "Stable"
    elif trend > 0 and avg_margin >= 8:
        return "Growing"
    else:
        return "Stable"
