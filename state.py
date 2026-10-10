"""
BusinessState — Single source of truth for all VRBS modules.
Every page reads from this. Build Your Business writes to it.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

@dataclass
class LocationProfile:
    city: str = "Hyderabad"
    state: str = "Telangana"
    country: str = "India"
    lat: float = 17.385
    lng: float = 78.4867
    population: int = 10_000_000
    density_per_km2: int = 18_000
    median_income_inr: int = 420_000    # annual
    avg_household_size: float = 3.8
    working_population_pct: float = 0.42
    youth_pct: float = 0.32             # 18-35 age group fraction
    rent_index: float = 1.0             # 1.0 = Hyderabad baseline
    wage_index: float = 1.0
    competition_density: str = "High"   # Low / Medium / High / Very High
    data_source: str = "Census 2021 Estimate"  # transparency label


@dataclass
class IdentityConfig:
    business_name: str = "My Restaurant"
    business_type: str = "Restaurant"
    genre: str = "Indian Cuisine"
    concept: str = "Casual Dining"
    target_customer: str = "Young Professionals"
    business_size: str = "Small"        # Micro / Small / Medium


@dataclass
class OperationsConfig:
    capacity_seats: int = 50
    staff_count: int = 6
    operating_hours: float = 12.0
    operating_days_per_week: int = 7
    rent_monthly_inr: int = 80_000
    utilities_monthly_inr: int = 15_000
    equipment_monthly_inr: int = 5_000
    cogs_pct: float = 0.32              # Cost of goods sold as % of revenue
    other_fixed_monthly_inr: int = 10_000


@dataclass
class ProductItem:
    name: str = "Main Course"
    price_inr: float = 350.0
    cogs_inr: float = 105.0
    demand_share: float = 0.4           # fraction of orders that include this


@dataclass
class ProductConfig:
    items: list = field(default_factory=lambda: [
        {"name": "Starter", "price_inr": 180, "cogs_inr": 50, "demand_share": 0.6},
        {"name": "Main Course", "price_inr": 350, "cogs_inr": 110, "demand_share": 0.9},
        {"name": "Dessert", "price_inr": 150, "cogs_inr": 40, "demand_share": 0.3},
        {"name": "Beverages", "price_inr": 120, "cogs_inr": 20, "demand_share": 0.75},
    ])
    price_positioning: str = "Mid-Range"    # Budget / Mid-Range / Premium / Luxury
    price_elasticity: float = -1.2          # % change in demand per 1% price change


@dataclass
class CustomerConfig:
    primary_segment: str = "Young Professionals"
    avg_party_size: float = 2.5
    new_customer_fraction: float = 0.6      # fraction of visitors who are new
    retention_rate: float = 0.35            # fraction who return within 30 days
    avg_visit_frequency_monthly: float = 1.8
    customer_lifetime_months: float = 14.0


@dataclass
class MarketingConfig:
    monthly_budget_inr: int = 30_000
    channels: dict = field(default_factory=lambda: {
        "social_media": 0.40,    # fraction of budget
        "search_ads": 0.25,
        "influencer": 0.15,
        "offline_flyers": 0.10,
        "word_of_mouth": 0.10,   # organic — budget has no effect, auto-calculated
    })
    # These are Model Assumptions based on industry benchmarks for Indian SMB restaurants
    # Source: "Digital Marketing Benchmarks India 2023" (IAMAI / industry estimates)
    channel_params: dict = field(default_factory=lambda: {
        # channel: {cpm_inr, ctr, conversion_rate, avg_cac_inr}
        "social_media":   {"cpm_inr": 80,  "ctr": 0.015, "conversion_rate": 0.03, "avg_cac_inr": 450},
        "search_ads":     {"cpm_inr": 200, "ctr": 0.065, "conversion_rate": 0.08, "avg_cac_inr": 280},
        "influencer":     {"cpm_inr": 120, "ctr": 0.035, "conversion_rate": 0.05, "avg_cac_inr": 380},
        "offline_flyers": {"cpm_inr": 30,  "ctr": 0.008, "conversion_rate": 0.02, "avg_cac_inr": 600},
        "word_of_mouth":  {"cpm_inr": 0,   "ctr": 0.0,  "conversion_rate": 0.0,  "avg_cac_inr": 0},
    })
    data_source: str = "MODEL ASSUMPTION — IAMAI India 2023 digital marketing benchmarks"


@dataclass
class SimulationConfig:
    start_date: str = "2024-01-01"
    simulation_days: int = 30


# ---------------------------------------------------------------------------
# Master BusinessState
# ---------------------------------------------------------------------------

@dataclass
class BusinessState:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    location: LocationProfile = field(default_factory=LocationProfile)
    operations: OperationsConfig = field(default_factory=OperationsConfig)
    products: ProductConfig = field(default_factory=ProductConfig)
    customers: CustomerConfig = field(default_factory=CustomerConfig)
    marketing: MarketingConfig = field(default_factory=MarketingConfig)
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    # Populated after simulation runs
    last_simulation_summary: Optional[dict] = None
    simulation_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BusinessState":
        bs = cls()
        if "identity" in d:
            bs.identity = IdentityConfig(**d["identity"])
        if "location" in d:
            bs.location = LocationProfile(**d["location"])
        if "operations" in d:
            bs.operations = OperationsConfig(**d["operations"])
        if "products" in d:
            bs.products = ProductConfig(**d["products"])
        if "customers" in d:
            bs.customers = CustomerConfig(**d["customers"])
        if "marketing" in d:
            mc = d["marketing"]
            bs.marketing = MarketingConfig(
                monthly_budget_inr=mc.get("monthly_budget_inr", 30000),
                channels=mc.get("channels", {}),
                channel_params=mc.get("channel_params", {}),
                data_source=mc.get("data_source", ""),
            )
        if "sim_config" in d:
            bs.sim_config = SimulationConfig(**d["sim_config"])
        bs.last_simulation_summary = d.get("last_simulation_summary")
        bs.simulation_history = d.get("simulation_history", [])
        return bs

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, s: str) -> "BusinessState":
        return cls.from_dict(json.loads(s))


def default_state() -> BusinessState:
    return BusinessState()
