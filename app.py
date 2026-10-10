"""
VRBS v2 — Main Dash Application
Full product: Home, Build Business, Output, Explain, Financial Report, Advisor, Challenge Me
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, no_update
import dash_leaflet as dl
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.simulation.state import BusinessState, default_state
from src.simulation.engine import run_simulation
from src.simulation.location_data import get_city_profile, get_all_profiles_df
from src.simulation.explainer import get_all_explanations
from src.advisor.advisor import BusinessAdvisor, generate_challenge, evaluate_answer

# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
app = dash.Dash(
    __name__,
    assets_folder=ASSETS_DIR,
    suppress_callback_exceptions=True,
    title="VRBS — Business Simulator",
)
server = app.server

_advisor = BusinessAdvisor()

# ─────────────────────────────────────────────
# Color / chart template
# ─────────────────────────────────────────────
CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=10, r=10, t=36, b=10),
)

# ─────────────────────────────────────────────
# Sidebar definition
# ─────────────────────────────────────────────
NAV_ITEMS = [
    ("home",          "🏠", "Home",             None),
    (None,            None, "BUILD",             "section"),
    ("build-identity","🏗️", "Business",          None),
    ("build-location","📍", "Location",          None),
    ("build-ops",     "⚙️", "Operations",        None),
    ("build-products","🍽️", "Products & Pricing",None),
    ("build-customers","👥","Customers",          None),
    ("build-marketing","📣","Marketing",          None),
    (None,            None, "UNDERSTAND",        "section"),
    ("output",        "📊", "Business Output",   None),
    ("explain",       "🔍", "Explain the Business",None),
    ("financials",    "💰", "Financial Report",  None),
    (None,            None, "THINK",             "section"),
    ("advisor",       "🤖", "Business Advisor",  None),
    ("challenge",     "🧠", "Challenge Me",      None),
]

# Section save order — used for auto-redirect
SECTION_FLOW = [
    "build-identity",
    "build-location",
    "build-ops",
    "build-products",
    "build-customers",
    "build-marketing",
]

SECTION_KEYS = {
    "build-identity": "identity",
    "build-location": "location",
    "build-ops": "ops",
    "build-products": "products",
    "build-customers": "customers",
    "build-marketing": "marketing",
}


def make_sidebar():
    items = []
    for page_id, icon, label, kind in NAV_ITEMS:
        if kind == "section":
            items.append(html.Div(label, className="sidebar-section-label"))
        else:
            items.append(
                html.Div(
                    [html.Span(icon, className="nav-item-icon"),
                     html.Span(label, className="nav-item-label")],
                    className="nav-item",
                    id={"type": "nav-item", "page": page_id},
                    n_clicks=0,
                )
            )
    return html.Div(
        [
            html.Div([
                html.Div("VR", className="sidebar-logo-mark"),
                html.Span("VRBS", className="sidebar-logo-text"),
            ], className="sidebar-logo"),
            html.Div(items, className="sidebar-nav"),
        ],
        id="sidebar",
        className="",
    )


# ─────────────────────────────────────────────
# Saved-status badge helper
# ─────────────────────────────────────────────
def _saved_badge(saved: bool):
    """Returns a small ✔ Configured badge if the section was saved."""
    if not saved:
        return None
    return html.Span(
        "✔ Configured",
        style={
            "background": "rgba(34,197,94,0.15)",
            "color": "#22c55e",
            "border": "1px solid rgba(34,197,94,0.3)",
            "borderRadius": "20px",
            "fontSize": "11px",
            "fontWeight": "600",
            "padding": "2px 10px",
            "marginLeft": "12px",
            "verticalAlign": "middle",
            "letterSpacing": "0.3px",
        }
    )


# ─────────────────────────────────────────────
# Page Layouts
# ─────────────────────────────────────────────

def page_home(state: BusinessState, sim=None):
    summ = sim["summary"] if sim else {}
    health = summ.get("health", "—")
    health_class = {"Growing": "status-growing", "Stable": "status-stable", "At Risk": "status-atrisk"}.get(health, "status-stable")

    kpis = [
        ("REVENUE", f"₹{summ.get('total_revenue_inr',0):,.0f}", "positive" if summ.get('total_revenue_inr',0) > 0 else "", ""),
        ("NET PROFIT", f"₹{summ.get('total_net_profit_inr',0):,.0f}", "positive" if summ.get('total_net_profit_inr',0) > 0 else "negative", ""),
        ("CUSTOMERS", f"{summ.get('total_customers_served',0):,}", "", ""),
        ("SATISFACTION", f"{summ.get('avg_satisfaction',0):.1f} / 5.0", "", ""),
    ]

    return html.Div([
        html.Div([
            html.Div(f"Welcome back", className="home-greeting"),
            html.Div(state.identity.business_name, className="home-business-name"),
            html.Div([
                html.Span(f"📍 {state.location.city}, {state.location.state}",
                          style={"marginRight": "16px", "color": "#93c5fd", "fontSize": "13px"}),
                html.Span(health, className=f"status-badge {health_class}"),
            ], className="home-location"),
            html.Div([
                html.Button("⚙ Configure Business", id="btn-config-home", className="btn-primary"),
            ]),
        ], className="home-hero"),
        # KPI row
        html.Div([
            html.Div([
                html.Div(label, className="kpi-label"),
                html.Div(value, className=f"kpi-value {cls}"),
            ], className="kpi-card")
            for label, value, cls, _ in kpis
        ], className="kpi-grid"),
        # Quick summary paragraph
        html.Div([
            html.Div([
                html.Div("LAST SIMULATION SUMMARY", className="vrbs-card-title"),
            ], className="vrbs-card-header"),
            html.Div(
                html.P(
                    f"Over {summ.get('simulation_days',30)} simulated days, your {state.identity.genre} "
                    f"in {state.location.city} generated ₹{summ.get('total_revenue_inr',0):,.0f} in revenue. "
                    f"You served {summ.get('total_customers_served',0):,} customers with an average satisfaction "
                    f"of {summ.get('avg_satisfaction',0):.1f}/5.0. "
                    f"Capacity utilisation averaged {summ.get('avg_capacity_utilization_pct',0):.0f}% "
                    f"and {summ.get('total_lost_customers',0)} customers were lost to wait times."
                    if summ else "No simulation run yet. Configure your business and click 'Configure Business' to get started.",
                    style={"color": "#cbd5e1", "fontSize": "14px", "lineHeight": "1.7"}
                ),
                className="vrbs-card-body"
            ),
        ], className="vrbs-card"),
    ])


SLIDER_UNITS = {
    "ops-capacity": "seats", "ops-staff": "staff", "ops-hours": "h", "ops-days": "days",
    "ops-rent": "₹", "ops-utilities": "₹", "ops-cogs": "%", "mkt-budget": "₹",
    "cust-party": "people", "cust-new-frac": "%", "cust-retention": "%",
    "cust-freq": "x/mo", "cust-lifetime": "mo", "prod-elasticity": ""
}

def _slider_row(label, id_, min_, max_, step, value, fmt=None):
    unit = SLIDER_UNITS.get(id_, "")
    unit_pre = unit if unit == "₹" else ""
    unit_post = unit if unit != "₹" else ""
    display_value = int(round(value * 100)) if unit == "%" else value
    return html.Div([
        html.Div([
            html.Span(label, className="vrbs-label", style={"marginBottom": 0}),
            html.Div([
                html.Span(unit_pre, style={"color": "#64748b", "fontSize": "12px", "marginRight": "4px"}),
                dcc.Input(
                    id=f"{id_}-input", type="number", value=display_value,
                    min=(min_*100 if unit=="%" else min_),
                    max=(max_*100 if unit=="%" else max_),
                    step=(step*100 if unit=="%" else step),
                    style={"fontFamily": "'JetBrains Mono',monospace",
                           "fontSize": "13px", "color": "#93c5fd",
                           "background": "rgba(255,255,255,0.05)", "border": "1px solid rgba(255,255,255,0.1)",
                           "textAlign": "right", "width": "75px", "outline": "none",
                           "padding": "2px 6px", "borderRadius": "4px"}
                ),
                html.Span(unit_post, style={"color": "#64748b", "fontSize": "12px", "marginLeft": "4px"}),
            ], style={"marginLeft": "auto", "display": "flex", "alignItems": "center"})
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        dcc.Slider(min=min_, max=max_, step=step, value=value, id=id_,
                   marks=None, tooltip={"always_visible": False},
                   className="vrbs-slider"),
    ], style={"marginBottom": "20px"})


def page_build_identity(state: BusinessState, saved: bool = False):
    return html.Div([
        html.Div([
            html.Span("Business Identity", className="section-title",
                      style={"display": "inline"}),
            _saved_badge(saved),
        ], style={"marginBottom": "4px"}),
        html.Div("Define what your business is.", className="section-sub"),
        html.Div([
            html.Div([
                html.Label("Business Name", className="vrbs-label"),
                dcc.Input(id="bi-name", value=state.identity.business_name,
                          className="vrbs-input", style={"width": "100%"}),
            ], style={"marginBottom": "16px"}),
            html.Div([
                html.Label("Cuisine / Genre", className="vrbs-label"),
                dcc.Dropdown(id="bi-genre",
                             options=[{"label": g, "value": g} for g in [
                                 "Indian Cuisine", "Italian Food", "Chinese Cuisine",
                                 "Fast Food", "Cafe/Coffee Shop", "Bakery",
                                 "Continental", "South Indian", "North Indian",
                                 "Street Food", "Fine Dining", "Biryani House"]],
                             value=state.identity.genre,
                             className="vrbs-select",
                             clearable=False, searchable=False,
                             style={"color": "black"}),
            ], style={"marginBottom": "16px"}),
            html.Div([
                html.Label("Concept", className="vrbs-label"),
                dcc.Dropdown(id="bi-concept",
                             options=[{"label": c, "value": c} for c in
                                      ["Quick Service", "Casual Dining", "Fine Dining",
                                       "Cafe", "Takeaway Only", "Cloud Kitchen"]],
                             value=state.identity.concept,
                             clearable=False, searchable=False,
                             style={"color": "black"}),
            ], style={"marginBottom": "16px"}),
            html.Div([
                html.Label("Target Customer", className="vrbs-label"),
                dcc.Dropdown(id="bi-target",
                             options=[{"label": c, "value": c} for c in
                                      ["Students", "Young Professionals", "Families",
                                       "Business Travelers", "Seniors", "All Demographics"]],
                             value=state.identity.target_customer,
                             clearable=False, searchable=False,
                             style={"color": "black"}),
            ], style={"marginBottom": "20px"}),
            html.Button("Save Identity →", id="btn-save-identity", className="btn-primary"),
        ], style={"maxWidth": "560px"}),
    ])


def _city_map():
    df = get_all_profiles_df()
    markers = []
    for _, row in df.iterrows():
        lat, lng = row.get("lat"), row.get("lng")
        if pd.isna(lat) or pd.isna(lng):
            continue
        income = row.get("median_income_inr", 400000)
        color = "#22c55e" if income > 500000 else "#fbbf24" if income > 350000 else "#ef4444"
        markers.append(
            dl.CircleMarker(
                center=[lat, lng],
                radius=10, color=color, fill=True, fillOpacity=0.7,
                id={"type": "city-marker", "city": row["city"]},
                children=dl.Tooltip(row["city"]),
            )
        )
    return dl.Map(
        [dl.TileLayer(url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                      attribution="© OpenStreetMap © CARTO"),
         dl.LayerGroup(markers)],
        center=[20, 78], zoom=5,
        style={"height": "400px", "borderRadius": "10px", "border": "1px solid rgba(255,255,255,0.07)"},
        id="india-map",
    )


def page_build_location(state: BusinessState, saved: bool = False):
    profile = get_city_profile(state.location.city)
    return html.Div([
        html.Div([
            html.Span("Location", className="section-title",
                      style={"display": "inline"}),
            _saved_badge(saved),
        ], style={"marginBottom": "4px"}),
        html.Div("Select your city to see the demographic environment your business will operate in.", className="section-sub"),
        html.Div([
            html.Div([
                html.Label("Select City", className="vrbs-label"),
                dcc.Dropdown(
                    id="loc-city-select",
                    options=[{"label": c, "value": c} for c in
                             ["Hyderabad","Bengaluru","Mumbai","Delhi","Chennai",
                              "Pune","Kolkata","Ahmedabad","Jaipur","Lucknow",
                              "Coimbatore","Visakhapatnam"]],
                    value=state.location.city,
                    className="vrbs-select",
                    clearable=False, searchable=False,
                    style={"color": "black", "marginBottom": "16px"},
                ),
                _city_map(),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div(id="city-demo-panel", children=_city_panel(profile),
                     style={"width": "280px", "flexShrink": "0"}),
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-start"}),
        html.Div(
            f"⚠ Data source: {profile.get('data_source','Model Assumption')}",
            style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "12px"}
        ),
        html.Button("Save Location →", id="btn-save-location", className="btn-primary",
                    style={"marginTop": "20px"}),
    ])


def _city_panel(profile: dict):
    def row(label, value):
        return html.Div([
            html.Span(label, style={"color": "#64748b", "fontSize": "12px"}),
            html.Span(value, style={"color": "#e2e8f0", "fontSize": "13px",
                                    "fontFamily": "'JetBrains Mono',monospace", "marginLeft": "auto"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "8px 0", "borderBottom": "1px solid rgba(255,255,255,0.04)"})

    pop = profile.get("population", 0)
    income = profile.get("median_income_inr", 0)
    density = profile.get("density_per_km2", 0)
    rent_i = profile.get("rent_index", 1.0)
    competition = profile.get("competition_density", "N/A")

    return html.Div([
        html.Div(profile.get("city", ""), style={"fontSize": "16px", "fontWeight": "700",
                                                  "color": "#fff", "marginBottom": "4px"}),
        html.Div(profile.get("state", ""), style={"fontSize": "12px", "color": "#94a3b8", "marginBottom": "16px"}),
        row("Population", f"{pop/1_000_000:.1f}M"),
        row("Density /km²", f"{density:,}"),
        row("Median Income", f"₹{income/1000:.0f}K/yr"),
        row("Rent Index", f"{rent_i:.2f}x"),
        row("Youth (18-35)", f"{profile.get('youth_pct',0)*100:.0f}%"),
        row("Working Pop", f"{profile.get('working_population_pct',0)*100:.0f}%"),
        row("Competition", competition),
    ], className="vrbs-card", style={"padding": "16px"})


def page_build_operations(state: BusinessState, saved: bool = False):
    ops = state.operations
    return html.Div([
        html.Div([
            html.Span("Operations", className="section-title",
                      style={"display": "inline"}),
            _saved_badge(saved),
        ], style={"marginBottom": "4px"}),
        html.Div("Configure how your business physically operates.", className="section-sub"),
        html.Div([
            _slider_row("Seating Capacity", "ops-capacity", 10, 300, 5, ops.capacity_seats),
            _slider_row("Staff Count", "ops-staff", 1, 30, 1, ops.staff_count),
            _slider_row("Operating Hours / Day", "ops-hours", 4, 18, 0.5, ops.operating_hours),
            _slider_row("Operating Days / Week", "ops-days", 1, 7, 1, ops.operating_days_per_week),
            html.Hr(style={"border": "1px solid rgba(255,255,255,0.06)", "margin": "8px 0 20px"}),
            html.Div("Monthly Fixed Costs", style={"fontSize": "13px", "fontWeight": "600",
                                                    "color": "#94a3b8", "marginBottom": "16px",
                                                    "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            _slider_row("Monthly Rent (₹)", "ops-rent", 10000, 500000, 5000, ops.rent_monthly_inr),
            _slider_row("Monthly Utilities (₹)", "ops-utilities", 2000, 100000, 1000, ops.utilities_monthly_inr),
            _slider_row("COGS % of Revenue", "ops-cogs", 0.15, 0.65, 0.01, ops.cogs_pct),
            html.Button("Save Operations →", id="btn-save-ops", className="btn-primary"),
        ], style={"maxWidth": "600px"}),
    ])


def page_build_products(state: BusinessState, saved: bool = False):
    prod = state.products
    pos_options = ["Budget", "Mid-Range", "Premium", "Luxury"]
    return html.Div([
        html.Div([
            html.Span("Products & Pricing", className="section-title",
                      style={"display": "inline"}),
            _saved_badge(saved),
        ], style={"marginBottom": "4px"}),
        html.Div("Set your pricing strategy. Price affects both revenue per order and customer demand.", className="section-sub"),
        html.Div([
            html.Label("Price Positioning", className="vrbs-label"),
            dcc.Dropdown(id="prod-positioning",
                         options=[{"label": p, "value": p} for p in pos_options],
                         value=prod.price_positioning,
                         clearable=False, searchable=False,
                         style={"color": "black", "marginBottom": "20px"}),
        ], style={"maxWidth": "400px", "marginBottom": "8px"}),
        html.Div([
            html.Div([
                html.Div("Positioning", style={"fontWeight": "700", "color": "#fff",
                                               "marginBottom": "4px", "fontSize": "14px"}),
                html.Div("Order Value Multiplier", style={"fontSize": "12px", "color": "#94a3b8"}),
            ]),
            html.Div([
                html.Div([
                    html.Div(p, className="vrbs-label", style={"marginBottom": "4px"}),
                    html.Div(f"{m}x", style={"fontFamily": "'JetBrains Mono',monospace",
                                             "color": "#93c5fd", "fontSize": "16px"}),
                ], style={"textAlign": "center", "padding": "12px 16px", "flex": "1",
                          "background": "rgba(37,99,235,0.1)" if p == prod.price_positioning else "transparent",
                          "borderRadius": "8px"})
                for p, m in [("Budget", "0.55"), ("Mid-Range", "1.0"), ("Premium", "1.65"), ("Luxury", "2.8")]
            ], style={"display": "flex", "gap": "8px", "background": "rgba(255,255,255,0.03)",
                      "borderRadius": "10px", "padding": "8px", "marginBottom": "24px"}),
        ]),
        html.Div([
            html.Div("Price Elasticity (MODEL ASSUMPTION)", style={"fontSize": "13px", "color": "#94a3b8", "marginBottom": "4px"}),
            html.Div("Represents how sensitive your customers are to price changes. "
                     "−1.2 means a 10% price increase reduces demand by 12%.",
                     style={"fontSize": "12px", "color": "#64748b", "marginBottom": "16px"}),
            _slider_row("Price Elasticity", "prod-elasticity", -3.0, -0.1, 0.1, prod.price_elasticity),
        ], style={"maxWidth": "500px"}),
        html.Button("Save Pricing →", id="btn-save-products", className="btn-primary"),
    ])


def page_build_customers(state: BusinessState, saved: bool = False):
    cust = state.customers
    return html.Div([
        html.Div([
            html.Span("Customers", className="section-title",
                      style={"display": "inline"}),
            _saved_badge(saved),
        ], style={"marginBottom": "4px"}),
        html.Div("Configure your customer behaviour model.", className="section-sub"),
        html.Div([
            html.Label("Primary Customer Segment", className="vrbs-label"),
            dcc.Dropdown(id="cust-segment",
                         options=[{"label": s, "value": s} for s in
                                  ["Students", "Young Professionals", "Families",
                                   "Business Travelers", "Seniors", "All Demographics"]],
                         value=cust.primary_segment,
                         style={"color": "black", "marginBottom": "20px"}),
        ], style={"maxWidth": "400px"}),
        html.Div([
            _slider_row("Avg Party Size", "cust-party", 1.0, 8.0, 0.5, cust.avg_party_size),
            _slider_row("New Customer Fraction", "cust-new-frac", 0.1, 1.0, 0.05, cust.new_customer_fraction),
            _slider_row("Monthly Retention Rate", "cust-retention", 0.05, 0.9, 0.05, cust.retention_rate),
            _slider_row("Avg Monthly Visit Frequency", "cust-freq", 0.5, 8.0, 0.5, cust.avg_visit_frequency_monthly),
            _slider_row("Customer Lifetime (months)", "cust-lifetime", 1.0, 36.0, 1.0, cust.customer_lifetime_months),
        ], style={"maxWidth": "600px"}),
        html.Div(id="clv-preview", style={"marginBottom": "20px"}),
        html.Button("Save Customers →", id="btn-save-customers", className="btn-primary"),
    ])


def page_build_marketing(state: BusinessState, saved: bool = False):
    mk = state.marketing
    return html.Div([
        html.Div([
            html.Span("Marketing", className="section-title",
                      style={"display": "inline"}),
            _saved_badge(saved),
        ], style={"marginBottom": "4px"}),
        html.Div([
            "Channel parameters are ",
            html.Strong("model assumptions"),
            f" based on IAMAI India 2023 digital marketing benchmarks. Source: {mk.data_source}",
        ], className="section-sub"),
        html.Div([
            _slider_row("Monthly Marketing Budget (₹)", "mkt-budget", 0, 200000, 2000, mk.monthly_budget_inr),
        ], style={"maxWidth": "600px"}),
        html.Div("Budget Allocation by Channel", className="vrbs-label",
                 style={"marginBottom": "12px", "marginTop": "8px"}),
        html.Div([
            html.Div([
                html.Div([
                    html.Span(ch.replace("_", " ").title(), style={"color": "#e2e8f0", "fontSize": "13px"}),
                    html.Span(f"{int(frac*100)}%",
                              style={"color": "#93c5fd", "fontFamily": "'JetBrains Mono',monospace",
                                     "fontSize": "13px", "marginLeft": "auto"}),
                ], style={"display": "flex", "marginBottom": "6px"}),
                html.Div([
                    html.Div(style={"width": f"{int(frac*100)}%", "height": "6px",
                                    "background": "#2563eb", "borderRadius": "3px"}),
                ], style={"background": "rgba(255,255,255,0.06)", "borderRadius": "3px", "height": "6px",
                          "marginBottom": "16px"}),
                html.Div([
                    html.Span(f"CPM: ₹{mk.channel_params.get(ch, {}).get('cpm_inr', 0)}",
                              style={"color": "#64748b", "fontSize": "11px", "marginRight": "12px"}),
                    html.Span(f"CTR: {mk.channel_params.get(ch, {}).get('ctr', 0)*100:.1f}%",
                              style={"color": "#64748b", "fontSize": "11px", "marginRight": "12px"}),
                    html.Span(f"CVR: {mk.channel_params.get(ch, {}).get('conversion_rate', 0)*100:.1f}%",
                              style={"color": "#64748b", "fontSize": "11px"}),
                ]),
            ], style={"padding": "14px", "background": "rgba(255,255,255,0.03)",
                      "borderRadius": "8px", "marginBottom": "8px"})
            for ch, frac in mk.channels.items()
        ], style={"maxWidth": "560px"}),
        # Save + Run Simulation buttons side by side
        html.Div([
            html.Button("Save Marketing", id="btn-save-marketing", className="btn-primary",
                        style={"marginRight": "12px"}),
            html.Button("▶ Run Simulation", id="btn-run-marketing", className="btn-primary",
                        style={"background": "linear-gradient(135deg, #059669 0%, #10b981 100%)",
                               "boxShadow": "0 0 20px rgba(16,185,129,0.3)"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "16px"}),
    ])


def page_output(sim):
    if not sim:
        return html.Div("Run a simulation first to see output.", className="section-sub")
    df = pd.DataFrame(sim["daily_df"])
    summ = sim["summary"]
    mkt = sim["marketing"]

    def fig_line(x, ys, colors, title):
        fig = go.Figure()
        for y, color, name in ys:
            fig.add_trace(go.Scatter(x=x, y=df[y], mode="lines", name=name,
                                     line=dict(color=color, width=2)))
        fig.update_layout(**CHART_TEMPLATE, title=dict(text=title, font=dict(size=13, color="#94a3b8")), height=260)
        return fig

    # Financials
    fig_fin = go.Figure()
    fig_fin.add_trace(go.Scatter(x=df["date"], y=df["revenue_inr"], name="Revenue",
                                  line=dict(color="#3b82f6", width=2), fill="tozeroy",
                                  fillcolor="rgba(59,130,246,0.07)"))
    fig_fin.add_trace(go.Scatter(x=df["date"], y=df["total_cost_inr"], name="Total Cost",
                                  line=dict(color="#ef4444", width=1.5, dash="dash")))
    fig_fin.add_trace(go.Bar(x=df["date"], y=df["net_profit_inr"], name="Net Profit",
                              marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in df["net_profit_inr"]],
                              opacity=0.7, yaxis="y2"))
    fig_fin.update_layout(**CHART_TEMPLATE, title=dict(text="Revenue / Cost / Profit", font=dict(size=13, color="#94a3b8")),
                           height=280, yaxis2=dict(overlaying="y", side="right", showgrid=False))

    # Visitors
    fig_vis = fig_line(df["date"],
                        [("potential_visitors", "#64748b", "Potential"),
                         ("effective_visitors", "#3b82f6", "Served"),
                         ("lost_customers", "#ef4444", "Lost")],
                        None, "Customer Traffic")

    # Satisfaction
    fig_sat = go.Figure()
    fig_sat.add_trace(go.Scatter(x=df["date"], y=df["satisfaction_stars"], name="Satisfaction",
                                  line=dict(color="#fbbf24", width=2), fill="tozeroy",
                                  fillcolor="rgba(251,191,36,0.06)"))
    fig_sat.add_hline(y=3.0, line_dash="dot", line_color="rgba(239,68,68,0.4)",
                       annotation_text="Min Acceptable", annotation_font_color="#94a3b8")
    fig_sat.update_layout(**CHART_TEMPLATE, title=dict(text="Daily Satisfaction Score", font=dict(size=13, color="#94a3b8")), height=220)
    fig_sat.update_yaxes(range=[1, 5.2], gridcolor="rgba(255,255,255,0.05)")

    # Marketing channel table
    ch_rows = []
    for ch, cd in mkt.get("channel_breakdown", {}).items():
        ch_rows.append(html.Tr([
            html.Td(ch.replace("_", " ").title()),
            html.Td(f"₹{cd['spend_inr']:,.0f}", className="money"),
            html.Td(f"{cd['impressions']:,}"),
            html.Td(f"{cd['clicks']:,}"),
            html.Td(f"{cd['new_customers']:,}"),
            html.Td(f"₹{cd['cac_inr']:,.0f}", className="money"),
        ]))

    return html.Div([
        html.Div("Business Output", className="section-title"),
        html.Div(f"Simulation results for {summ.get('simulation_days', 30)} days | {summ.get('location')} | {summ.get('business_name')}", className="section-sub"),
        # KPIs
        html.Div([
            html.Div([html.Div(l, className="kpi-label"), html.Div(v, className=f"kpi-value {c}")], className="kpi-card")
            for l, v, c in [
                ("REVENUE",        f"₹{summ['total_revenue_inr']:,.0f}",       "positive"),
                ("GROSS PROFIT",   f"₹{summ['total_gross_profit_inr']:,.0f}",  "positive" if summ['total_gross_profit_inr'] > 0 else "negative"),
                ("NET PROFIT",     f"₹{summ['total_net_profit_inr']:,.0f}",    "positive" if summ['total_net_profit_inr'] > 0 else "negative"),
                ("GROSS MARGIN",   f"{summ['avg_gross_margin_pct']:.1f}%",     ""),
                ("NET MARGIN",     f"{summ['avg_net_margin_pct']:.1f}%",       "positive" if summ['avg_net_margin_pct'] > 5 else "warning" if summ['avg_net_margin_pct'] > 0 else "negative"),
                ("CAPACITY UTIL",  f"{summ['avg_capacity_utilization_pct']:.0f}%", ""),
                ("AVG SAT",        f"{summ['avg_satisfaction']:.2f}/5.0",      ""),
                ("LOST CUSTOMERS", f"{summ['total_lost_customers']:,}",        "negative" if summ['total_lost_customers'] > 0 else ""),
            ]
        ], className="kpi-grid"),
        # Charts
        html.Div([dcc.Graph(figure=fig_fin, config={"displayModeBar": False})], className="vrbs-card", style={"marginBottom": "16px"}),
        html.Div([dcc.Graph(figure=fig_vis, config={"displayModeBar": False})], className="vrbs-card", style={"marginBottom": "16px"}),
        html.Div([dcc.Graph(figure=fig_sat, config={"displayModeBar": False})], className="vrbs-card", style={"marginBottom": "16px"}),
        # Marketing table
        html.Div([
            html.Div("MARKETING CHANNEL PERFORMANCE", className="vrbs-card-title"),
            html.Div(f"CAC: ₹{mkt.get('overall_cac_inr', 0):,.0f} | ROI: {mkt.get('marketing_roi_pct', 0):.1f}% | {mkt.get('data_source', '')}",
                     style={"fontSize": "11px", "color": "#64748b"}),
        ], className="vrbs-card-header"),
        html.Div(
            html.Table([
                html.Thead(html.Tr([html.Th(h) for h in ["Channel", "Spend", "Impressions", "Clicks", "New Customers", "CAC"]])),
                html.Tbody(ch_rows),
            ], className="fin-table"),
            className="vrbs-card-body",
        ),
    ], className="vrbs-card")


def page_explain(sim, state_dict):
    if not sim:
        return html.Div("Run a simulation first to get business concept explanations.", className="section-sub")
    summ = {**sim["summary"], "marketing": sim["marketing"]}
    explanations = get_all_explanations(summ)
    if not explanations:
        return html.Div([
            html.Div("Explain the Business", className="section-title"),
            html.Div("✅ Your business metrics look healthy — no major concerns detected!", className="section-sub"),
        ])
    cards = []
    for exp in explanations:
        c = exp.get("concept", {})
        vals = exp.get("current_values", {})
        cards.append(html.Div([
            html.Div(c.get("name", "Concept"), className="concept-title"),
            html.Div(c.get("formula", ""), className="concept-formula"),
            html.Div(exp.get("observation", ""), className="concept-observation"),
            html.Div(c.get("definition", ""), className="concept-body", style={"marginBottom": "8px"}),
            html.Div(exp.get("implication", ""), className="concept-body"),
            html.Div([
                html.Span(f"{k}: {v}", className="concept-value-chip")
                for k, v in vals.items()
            ], className="concept-values"),
        ], className="concept-card"))
    return html.Div([
        html.Div("Explain the Business", className="section-title"),
        html.Div("Based on your simulation, here are the key business concepts at work in your business.", className="section-sub"),
        *cards,
    ])


def page_financials(sim):
    if not sim:
        return html.Div("Run a simulation first.", className="section-sub")
    summ = sim["summary"]
    df = pd.DataFrame(sim["daily_df"])

    rev = summ["total_revenue_inr"]
    cogs = df["cogs_inr"].sum()
    labor = df["labor_cost_inr"].sum()
    fixed = df["fixed_cost_inr"].sum()
    mkt = df["marketing_cost_inr"].sum()
    gross = summ["total_gross_profit_inr"]
    net = summ["total_net_profit_inr"]

    def trow(label, value, cls=""):
        return html.Tr([html.Td(label), html.Td(f"₹{value:,.0f}", className=f"money {cls}")])

    income_stmt = html.Table([
        html.Thead(html.Tr([html.Th("Item"), html.Th("Amount")])),
        html.Tbody([
            trow("Revenue", rev, "positive"),
            trow("Cost of Goods Sold (COGS)", -cogs, "negative"),
            html.Tr([html.Td("Gross Profit", style={"fontWeight": "700"}),
                     html.Td(f"₹{gross:,.0f}", className="money positive total-row")], className="total-row"),
            trow("Labor Cost", -labor, "negative"),
            trow("Rent & Fixed Costs", -fixed, "negative"),
            trow("Marketing", -mkt, "negative"),
            html.Tr([html.Td("Net Profit (Operating)", style={"fontWeight": "700"}),
                     html.Td(f"₹{net:,.0f}", className=f"money {'positive' if net >= 0 else 'negative'} total-row")], className="total-row"),
        ]),
    ], className="fin-table")

    return html.Div([
        html.Div("Financial Report", className="section-title"),
        html.Div(f"{summ['simulation_days']}-day Income Statement | {summ.get('business_name')} | {summ.get('location')}", className="section-sub"),
        html.Div([
            html.Div("INCOME STATEMENT", className="vrbs-card-title"),
        ], className="vrbs-card-header"),
        html.Div(income_stmt, className="vrbs-card-body"),
    ], className="vrbs-card", style={"maxWidth": "640px"})


def page_advisor():
    return html.Div([
        html.Div("Business Advisor", className="section-title"),
        html.Div("Ask me anything about your business. I have full access to your simulation data.", className="section-sub", style={"marginBottom": "4px"}),
        html.Div("Note: Please run a simulation first so I have your business data to work with.", style={"fontSize": "13px", "color": "#93c5fd", "marginBottom": "24px", "fontStyle": "italic"}),
        html.Div([
            html.Div(id="advisor-messages", className="chat-messages", children=[]),
            html.Div([
                dcc.Input(id="advisor-input", placeholder="Ask about your business...",
                           className="chat-input", debounce=False,
                           style={"flex": "1"}),
                html.Button("Send", id="btn-advisor-send", className="btn-primary"),
            ], className="chat-input-bar"),
        ], className="chat-container"),
    ])


def page_challenge(sim, state_dict):
    if not sim:
        return html.Div("Run a simulation first to unlock challenges.", className="section-sub")
    q = generate_challenge(sim["summary"], state_dict, level=1)
    level = q["level"]
    return html.Div([
        html.Div("Challenge Me", className="section-title"),
        html.Div("Answer based on your actual business data. The AI will evaluate your reasoning.", className="section-sub"),
        html.Div([
            html.Div(q["level_label"], className="kpi-label", style={"marginBottom": "8px"}),
            html.Div([
                html.Div(style={"width": "32px", "height": "8px", "borderRadius": "4px",
                                "background": "#2563eb" if i < level else "rgba(255,255,255,0.1)"})
                for i in range(5)
            ], className="challenge-level"),
            html.Div(q["question"], className="challenge-question"),
            html.Div(f"💡 Hint: {q['hint']}", className="challenge-hint"),
            dcc.Textarea(id="challenge-answer", placeholder="Type your answer here...",
                          className="challenge-textarea"),
            html.Div([
                html.Button("Submit Answer", id="btn-challenge-submit", className="btn-primary",
                            style={"marginRight": "10px"}),
                html.Button("Next Question", id="btn-challenge-next", className="btn-ghost"),
            ], style={"marginTop": "16px"}),
            html.Div(id="challenge-feedback", style={"marginTop": "20px"}),
        ], className="vrbs-card", style={"padding": "24px"}),
        dcc.Store(id="current-challenge", data=q),
    ])


# ─────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="business-state",    data=default_state().to_dict()),
    dcc.Store(id="simulation-result", data=None),
    dcc.Store(id="current-page",      data="home"),
    dcc.Store(id="saved-sections",    data={}),
    html.Div([
        make_sidebar(),
        html.Div([
            html.Div([
                html.Button("☰", id="sidebar-toggle", className="sidebar-hamburger", style={"margin": "0 16px 0 0"}),
                html.Div(id="topbar-title", className="topbar-title"),
                html.Div(id="topbar-right", className="topbar-actions"),
            ], id="topbar"),
            html.Div(id="page-wrapper"),
        ], id="main-content"),
    ], id="app-shell"),
], id="app-root")


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

# Sidebar toggle
@app.callback(
    Output("sidebar", "className"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar", "className"),
    prevent_initial_call=True,
)
def toggle_sidebar(n, cls):
    return "collapsed" if cls == "" else ""


# Navigation → page routing
@app.callback(
    Output("current-page", "data"),
    Input({"type": "nav-item", "page": ALL}, "n_clicks"),
    State({"type": "nav-item", "page": ALL}, "id"),
    prevent_initial_call=True,
)
def navigate(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered_id = ctx.triggered[0]["prop_id"]
    import re as _re
    m = _re.search(r'"page":"([^"]+)"', triggered_id)
    if m:
        return m.group(1)
    return no_update

@app.callback(
    Output({"type": "nav-item", "page": ALL}, "className"),
    Input("current-page", "data"),
    State({"type": "nav-item", "page": ALL}, "id"),
)
def update_nav_active(page, ids):
    return [("nav-item active" if id_["page"] == page else "nav-item") for id_ in ids]


# Home buttons — Configure only (Run Simulation moved to Marketing section)
@app.callback(
    Output("current-page", "data", allow_duplicate=True),
    Input("btn-config-home", "n_clicks"),
    prevent_initial_call=True,
)
def home_buttons(config):
    if not config:
        return no_update
    return "build-identity"


# City selection → update demographic panel + state
@app.callback(
    [Output("city-demo-panel", "children"),
     Output("business-state", "data")],
    Input("loc-city-select", "value"),
    State("business-state", "data"),
    prevent_initial_call=True,
)
def update_city(city, state_data):
    if not city:
        return no_update, no_update
    profile = get_city_profile(city)
    state = BusinessState.from_dict(state_data)
    state.location.city = city
    state.location.state = profile.get("state", "")
    state.location.lat = profile.get("lat", 20.0)
    state.location.lng = profile.get("lng", 78.0)
    state.location.population = int(profile.get("population", 1_000_000))
    state.location.median_income_inr = int(profile.get("median_income_inr", 400_000))
    state.location.density_per_km2 = int(profile.get("density_per_km2", 5_000))
    state.location.rent_index = float(profile.get("rent_index", 1.0))
    state.location.competition_density = profile.get("competition_density", "Medium")
    state.location.data_source = profile.get("data_source", "")
    return _city_panel(profile), state.to_dict()


# Save identity → navigate to Location
@app.callback(
    Output("business-state", "data", allow_duplicate=True),
    Output("current-page", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-save-identity", "n_clicks"),
    State("bi-name", "value"),
    State("bi-genre", "value"),
    State("bi-concept", "value"),
    State("bi-target", "value"),
    State("business-state", "data"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def save_identity(n, name, genre, concept, target, state_data, saved):
    if not n:
        return no_update, no_update, no_update
    state = BusinessState.from_dict(state_data)
    state.identity.business_name = name or state.identity.business_name
    state.identity.genre = genre or state.identity.genre
    state.identity.concept = concept or state.identity.concept
    state.identity.target_customer = target or state.identity.target_customer
    saved = dict(saved or {})
    saved["identity"] = True
    return state.to_dict(), "build-location", saved


# Save location → navigate to Operations
@app.callback(
    Output("current-page", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-save-location", "n_clicks"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def save_location(n, saved):
    if not n:
        return no_update, no_update
    saved = dict(saved or {})
    saved["location"] = True
    return "build-ops", saved


# Save operations → navigate to Products & Pricing
@app.callback(
    Output("business-state", "data", allow_duplicate=True),
    Output("current-page", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-save-ops", "n_clicks"),
    State("ops-capacity", "value"), State("ops-staff", "value"),
    State("ops-hours", "value"), State("ops-days", "value"),
    State("ops-rent", "value"), State("ops-utilities", "value"),
    State("ops-cogs", "value"),
    State("business-state", "data"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def save_ops(n, cap, staff, hours, days, rent, util, cogs, state_data, saved):
    if not n:
        return no_update, no_update, no_update
    state = BusinessState.from_dict(state_data)
    state.operations.capacity_seats = cap
    state.operations.staff_count = staff
    state.operations.operating_hours = hours
    state.operations.operating_days_per_week = days
    state.operations.rent_monthly_inr = rent
    state.operations.utilities_monthly_inr = util
    state.operations.cogs_pct = cogs
    saved = dict(saved or {})
    saved["ops"] = True
    return state.to_dict(), "build-products", saved


# Save products → navigate to Customers
@app.callback(
    Output("business-state", "data", allow_duplicate=True),
    Output("current-page", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-save-products", "n_clicks"),
    State("prod-positioning", "value"),
    State("prod-elasticity", "value"),
    State("business-state", "data"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def save_products(n, pos, elast, state_data, saved):
    if not n:
        return no_update, no_update, no_update
    state = BusinessState.from_dict(state_data)
    state.products.price_positioning = pos or state.products.price_positioning
    state.products.price_elasticity = elast
    saved = dict(saved or {})
    saved["products"] = True
    return state.to_dict(), "build-customers", saved


# Save customers → navigate to Marketing
@app.callback(
    Output("business-state", "data", allow_duplicate=True),
    Output("current-page", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-save-customers", "n_clicks"),
    State("cust-segment", "value"), State("cust-party", "value"),
    State("cust-new-frac", "value"), State("cust-retention", "value"),
    State("cust-freq", "value"), State("cust-lifetime", "value"),
    State("business-state", "data"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def save_customers(n, seg, party, new_frac, ret, freq, lifetime, state_data, saved):
    if not n:
        return no_update, no_update, no_update
    state = BusinessState.from_dict(state_data)
    state.customers.primary_segment = seg or state.customers.primary_segment
    state.customers.avg_party_size = party
    state.customers.new_customer_fraction = new_frac
    state.customers.retention_rate = ret
    state.customers.avg_visit_frequency_monthly = freq
    state.customers.customer_lifetime_months = lifetime
    saved = dict(saved or {})
    saved["customers"] = True
    return state.to_dict(), "build-marketing", saved


# Save marketing (stays on Marketing page — user then clicks Run Simulation)
@app.callback(
    Output("business-state", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-save-marketing", "n_clicks"),
    State("mkt-budget", "value"),
    State("business-state", "data"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def save_marketing(n, budget, state_data, saved):
    if not n:
        return no_update, no_update
    state = BusinessState.from_dict(state_data)
    state.marketing.monthly_budget_inr = budget
    saved = dict(saved or {})
    saved["marketing"] = True
    return state.to_dict(), saved


# Run simulation from Marketing page button
@app.callback(
    Output("current-page", "data", allow_duplicate=True),
    Output("saved-sections", "data", allow_duplicate=True),
    Input("btn-run-marketing", "n_clicks"),
    State("mkt-budget", "value"),
    State("business-state", "data"),
    State("saved-sections", "data"),
    prevent_initial_call=True,
)
def run_from_marketing(n, budget, state_data, saved):
    """Saves marketing (if not already) then navigates to output to trigger simulation."""
    if not n:
        return no_update, no_update
    saved = dict(saved or {})
    saved["marketing"] = True
    return "output", saved


# Run simulation (triggered when navigating to output/explain/financials/advisor/challenge)
@app.callback(
    Output("simulation-result", "data"),
    Input("current-page", "data"),
    State("business-state", "data"),
    prevent_initial_call=True,
)
def maybe_run_simulation(page, state_data):
    if page not in ("output", "explain", "financials", "advisor", "challenge"):
        return no_update
    state = BusinessState.from_dict(state_data)
    result = run_simulation(state)
    # Serialise (df → list of dicts)
    serialised = {
        "daily_df": result["daily_df"].to_dict(orient="records"),
        "summary": result["summary"],
        "marketing": result["marketing"],
    }
    return serialised


# Page render
@app.callback(
    [Output("page-wrapper", "children"),
     Output("topbar-title", "children")],
    [Input("current-page", "data"),
     Input("simulation-result", "data")],
    State("business-state", "data"),
    State("saved-sections", "data"),
)
def render_page(page, sim_data, state_data, saved_sections):
    state = BusinessState.from_dict(state_data)
    sv = saved_sections or {}

    titles = {
        "home": "Home",
        "build-identity": "Build Your Business — Identity",
        "build-location": "Build Your Business — Location",
        "build-ops": "Build Your Business — Operations",
        "build-products": "Build Your Business — Products & Pricing",
        "build-customers": "Build Your Business — Customers",
        "build-marketing": "Build Your Business — Marketing",
        "output": "Business Output",
        "explain": "Explain the Business",
        "financials": "Financial Report",
        "advisor": "Business Advisor",
        "challenge": "Challenge Me",
    }
    title = titles.get(page, "VRBS")

    sim = None
    if sim_data:
        sim = {**sim_data, "daily_df": sim_data["daily_df"]}

    if page == "home":             content = page_home(state, sim)
    elif page == "build-identity": content = page_build_identity(state, saved=sv.get("identity", False))
    elif page == "build-location": content = page_build_location(state, saved=sv.get("location", False))
    elif page == "build-ops":      content = page_build_operations(state, saved=sv.get("ops", False))
    elif page == "build-products": content = page_build_products(state, saved=sv.get("products", False))
    elif page == "build-customers":content = page_build_customers(state, saved=sv.get("customers", False))
    elif page == "build-marketing":content = page_build_marketing(state, saved=sv.get("marketing", False))
    elif page == "output":         content = page_output(sim)
    elif page == "explain":        content = page_explain(sim, state_data)
    elif page == "financials":     content = page_financials(sim)
    elif page == "advisor":        content = page_advisor()
    elif page == "challenge":      content = page_challenge(sim, state_data)
    else:                          content = html.Div("Page not found.")

    return content, title


# Advisor send message
@app.callback(
    [Output("advisor-messages", "children"),
     Output("advisor-input", "value")],
    [Input("btn-advisor-send", "n_clicks"),
     Input("advisor-input", "n_submit")],
    State("advisor-input", "value"),
    State("advisor-messages", "children"),
    State("simulation-result", "data"),
    State("business-state", "data"),
    prevent_initial_call=True,
)
def advisor_send(n_clicks, n_submit, msg, history, sim_data, state_data):
    if not msg or (not n_clicks and not n_submit):
        return no_update, no_update
    state = BusinessState.from_dict(state_data)
    summ = sim_data["summary"] if sim_data else {}
    mkt = sim_data.get("marketing", {}) if sim_data else {}
    _advisor.load_business(state.to_dict(), summ, mkt)
    reply = _advisor.ask(msg)
    history = list(history or [])
    history.append(html.Div(msg, className="chat-msg chat-msg-user"))
    history.append(dcc.Markdown(reply, className="chat-msg chat-msg-advisor"))
    return history, ""


# Challenge submit
@app.callback(
    Output("challenge-feedback", "children"),
    Input("btn-challenge-submit", "n_clicks"),
    State("challenge-answer", "value"),
    State("current-challenge", "data"),
    State("simulation-result", "data"),
    State("business-state", "data"),
    prevent_initial_call=True,
)
def submit_challenge(n, answer, question, sim_data, state_data):
    if not n or not answer:
        return no_update
    state = BusinessState.from_dict(state_data)
    summ = sim_data["summary"] if sim_data else {}
    mkt = sim_data.get("marketing", {}) if sim_data else {}
    _advisor.load_business(state.to_dict(), summ, mkt)
    feedback = evaluate_answer(question, answer, _advisor)
    return html.Div([
        html.Div("AI Evaluation", className="kpi-label", style={"marginBottom": "8px"}),
        html.Div(feedback, style={"color": "#cbd5e1", "fontSize": "13.5px", "lineHeight": "1.7",
                                   "whiteSpace": "pre-wrap"}),
    ], className="concept-card", style={"borderLeftColor": "#22c55e"})


# Bidirectional sync between slider and input
for _id, _unit in SLIDER_UNITS.items():
    app.clientside_callback(
        """
        function(slider_val, input_val) {
            var trigger = dash_clientside.callback_context.triggered;
            if (!trigger || trigger.length === 0) {
                return window.dash_clientside.no_update;
            }
            var prop_id = trigger[0].prop_id;
            var is_pct = %s;
            
            if (prop_id.indexOf('input') !== -1) {
                var s_val = is_pct ? input_val / 100.0 : input_val;
                return [s_val, input_val];
            }
            var i_val = is_pct ? Math.round(slider_val * 100) : slider_val;
            return [slider_val, i_val];
        }
        """ % ('true' if _unit == "%" else 'false'),
        Output(_id, "value"),
        Output(f"{_id}-input", "value"),
        Input(_id, "value"),
        Input(f"{_id}-input", "value"),
        prevent_initial_call=True
    )

# Visual feedback for save buttons (button text flashes "✔ Saved!" then resets)
for _btn_id, _orig in [
    ("btn-save-identity", "Save Identity →"),
    ("btn-save-location", "Save Location →"),
    ("btn-save-ops", "Save Operations →"),
    ("btn-save-products", "Save Pricing →"),
    ("btn-save-customers", "Save Customers →"),
    ("btn-save-marketing", "Save Marketing"),
]:
    app.clientside_callback(
        """
        function(n) {
            if (!n) return window.dash_clientside.no_update;
            return "✔ Saved!";
        }
        """,
        Output(_btn_id, "children"),
        Input(_btn_id, "n_clicks"),
        prevent_initial_call=True
    )

if __name__ == "__main__":
    app.run(debug=True, port=8050)
