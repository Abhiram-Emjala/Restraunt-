"""
Business concept explainer — maps simulation output patterns to educational explanations.
Used by "Explain the Business" page and "Challenge Me" question generator.
"""
from typing import Optional


# ---------------------------------------------------------------------------
# Concept library
# ---------------------------------------------------------------------------

CONCEPTS = {
    "contribution_margin": {
        "name": "Contribution Margin",
        "formula": "Revenue − Variable Costs",
        "definition": (
            "The amount left after covering variable costs, which 'contributes' to "
            "covering fixed costs and generating profit."
        ),
        "why_it_matters": (
            "If your contribution margin is low, even small drops in revenue can push you into loss."
        ),
    },
    "operating_leverage": {
        "name": "Operating Leverage",
        "formula": "% Change in Profit / % Change in Revenue",
        "definition": (
            "A measure of how sensitive profit is to changes in revenue. "
            "High fixed costs create high operating leverage."
        ),
        "why_it_matters": (
            "High leverage means revenue growth amplifies profit, but revenue decline is devastating."
        ),
    },
    "break_even": {
        "name": "Break-Even Point",
        "formula": "Fixed Costs / Contribution Margin per Customer",
        "definition": (
            "The number of customers (or revenue) needed to cover all costs, resulting in zero profit."
        ),
        "why_it_matters": "Knowing your break-even tells you the minimum viable scale for your business.",
    },
    "cac": {
        "name": "Customer Acquisition Cost (CAC)",
        "formula": "Total Marketing Spend / New Customers Acquired",
        "definition": "The average cost to acquire one new paying customer through marketing.",
        "why_it_matters": (
            "If CAC > CLV, you are paying more to acquire customers than they are worth. Unsustainable."
        ),
    },
    "clv": {
        "name": "Customer Lifetime Value (CLV)",
        "formula": "Avg Order Value × Visit Frequency × Customer Lifetime (months)",
        "definition": "The total revenue a business can expect from a single customer over their relationship.",
        "why_it_matters": "CLV > CAC is the fundamental rule of sustainable customer economics.",
    },
    "gross_margin": {
        "name": "Gross Margin",
        "formula": "(Revenue − COGS) / Revenue × 100",
        "definition": "The percentage of revenue remaining after subtracting the direct cost of goods sold.",
        "why_it_matters": (
            "Low gross margin leaves little room to cover operating expenses and earn profit."
        ),
    },
    "capacity_utilization": {
        "name": "Capacity Utilization",
        "formula": "Effective Customers / Maximum Possible Customers × 100",
        "definition": "The percentage of your maximum service capacity that is actually being used.",
        "why_it_matters": (
            "Below ~60% utilization, fixed costs dominate and profitability suffers. "
            "Above ~90%, quality and wait time suffer."
        ),
    },
    "price_elasticity": {
        "name": "Price Elasticity of Demand",
        "formula": "% Change in Demand / % Change in Price",
        "definition": (
            "Measures how sensitive customer demand is to price changes. "
            "Elastic demand drops sharply with price increases."
        ),
        "why_it_matters": "Setting the wrong price can destroy demand, even if per-unit margin improves.",
    },
    "marketing_roi": {
        "name": "Marketing ROI / ROAS",
        "formula": "(Marketing-Attributed Revenue − Marketing Spend) / Marketing Spend × 100",
        "definition": "The return generated on every rupee spent on marketing.",
        "why_it_matters": "Negative marketing ROI means you are spending more to acquire customers than they bring in.",
    },
    "fixed_vs_variable": {
        "name": "Fixed vs Variable Costs",
        "formula": "Fixed: Rent + Salaries | Variable: COGS + Utilities",
        "definition": (
            "Fixed costs stay constant regardless of revenue. "
            "Variable costs scale with sales volume."
        ),
        "why_it_matters": (
            "A high fixed-cost structure is risky during slow periods but very profitable at high volume."
        ),
    },
    "retention": {
        "name": "Customer Retention Rate",
        "formula": "(Returning Customers / Total Customers) × 100",
        "definition": "The percentage of customers who come back after their first visit.",
        "why_it_matters": (
            "Retaining a customer costs 5–7× less than acquiring a new one. "
            "Low retention forces permanent dependence on expensive marketing."
        ),
    },
    "churn": {
        "name": "Customer Churn",
        "formula": "1 − Retention Rate",
        "definition": "The percentage of customers who do not return.",
        "why_it_matters": (
            "High churn is a silent revenue leak — you constantly lose the value of previously acquired customers."
        ),
    },
}


# ---------------------------------------------------------------------------
# Pattern detector
# ---------------------------------------------------------------------------

def detect_patterns(summary: dict, prev_summary: Optional[dict] = None) -> list:
    """
    Analyse simulation summary and return a list of triggered concept explanations.
    Each item: {concept_key, observation, implication, current_values}
    """
    findings = []
    rev = summary.get("total_revenue_inr", 0)
    profit = summary.get("total_net_profit_inr", 0)
    gross_margin = summary.get("avg_gross_margin_pct", 0)
    net_margin = summary.get("avg_net_margin_pct", 0)
    cap_util = summary.get("avg_capacity_utilization_pct", 0)
    lost = summary.get("total_lost_customers", 0)
    sat = summary.get("avg_satisfaction", 0)
    wait = summary.get("avg_wait_time_hours", 0)
    cac = summary.get("marketing", {}).get("overall_cac_inr", 0) if "marketing" in summary else 0

    # --- Gross Margin check ---
    if gross_margin < 30:
        findings.append({
            "concept_key": "gross_margin",
            "observation": f"Your gross margin is {gross_margin:.1f}%, which is below the healthy 35–45% range for restaurants.",
            "implication": "This means your direct food costs are consuming too much of revenue, leaving little for other expenses.",
            "current_values": {"Gross Margin": f"{gross_margin:.1f}%", "Revenue": f"₹{rev:,.0f}"},
        })

    # --- Net margin / profitability ---
    if profit < 0:
        findings.append({
            "concept_key": "break_even",
            "observation": f"Your business is operating at a loss of ₹{abs(profit):,.0f} over the simulation period.",
            "implication": "You have not reached break-even. Fixed costs are larger than your contribution margin.",
            "current_values": {"Net Profit": f"₹{profit:,.0f}", "Net Margin": f"{net_margin:.1f}%"},
        })
    elif net_margin < 5:
        findings.append({
            "concept_key": "operating_leverage",
            "observation": f"Your net margin is very thin at {net_margin:.1f}%.",
            "implication": "High fixed costs relative to revenue create high operating leverage — any revenue drop will hurt profits significantly.",
            "current_values": {"Net Margin": f"{net_margin:.1f}%"},
        })

    # --- Revenue up, profit down (if previous run available) ---
    if prev_summary:
        prev_rev = prev_summary.get("total_revenue_inr", 0)
        prev_profit = prev_summary.get("total_net_profit_inr", 0)
        if rev > prev_rev and profit < prev_profit:
            findings.append({
                "concept_key": "fixed_vs_variable",
                "observation": f"Revenue increased by ₹{rev - prev_rev:,.0f} but profit fell by ₹{prev_profit - profit:,.0f}.",
                "implication": "Your variable or fixed costs grew faster than revenue. This is a classic cost-structure warning.",
                "current_values": {"Revenue Δ": f"+₹{rev - prev_rev:,.0f}", "Profit Δ": f"₹{profit - prev_profit:,.0f}"},
            })

    # --- Capacity utilization ---
    if cap_util < 40:
        findings.append({
            "concept_key": "capacity_utilization",
            "observation": f"Capacity utilization is only {cap_util:.1f}%. You are using less than half your potential.",
            "implication": "Fixed costs (rent, staff) are being spread over very few customers, dramatically hurting unit economics.",
            "current_values": {"Capacity Utilization": f"{cap_util:.1f}%"},
        })
    elif cap_util > 85:
        findings.append({
            "concept_key": "capacity_utilization",
            "observation": f"Capacity utilization is very high at {cap_util:.1f}%.",
            "implication": "This drives wait times, hurts satisfaction, and causes lost customers. Expansion or operational efficiency is needed.",
            "current_values": {"Capacity Utilization": f"{cap_util:.1f}%", "Wait Time": f"{wait:.1f}h"},
        })

    # --- Lost customers ---
    if lost > 50:
        findings.append({
            "concept_key": "retention",
            "observation": f"{lost} customers left without being served due to wait times or over-capacity.",
            "implication": "Lost customers are permanent revenue leakage. Each lost customer also represents a negative word-of-mouth risk.",
            "current_values": {"Lost Customers": f"{lost}", "Wait Time": f"{wait:.1f}h"},
        })

    # --- Satisfaction ---
    if sat < 3.0:
        findings.append({
            "concept_key": "churn",
            "observation": f"Average satisfaction is low at {sat:.1f} / 5.0.",
            "implication": "Low satisfaction drives high churn. Customers who have a bad experience rarely return, and often discourage others.",
            "current_values": {"Satisfaction": f"{sat:.2f} / 5.0"},
        })

    # --- CAC check ---
    if cac > 500:
        findings.append({
            "concept_key": "cac",
            "observation": f"Your marketing Cost to Acquire a Customer (CAC) is ₹{cac:,.0f}.",
            "implication": "If your average customer spends less than this per visit, you are spending more to acquire them than they immediately bring.",
            "current_values": {"CAC": f"₹{cac:,.0f}"},
        })

    return findings


def explain_finding(finding: dict) -> dict:
    """
    Enrich a finding with full concept details from the library.
    """
    key = finding.get("concept_key", "")
    concept = CONCEPTS.get(key, {})
    return {**finding, "concept": concept}


def get_all_explanations(summary: dict, prev_summary: Optional[dict] = None) -> list:
    findings = detect_patterns(summary, prev_summary)
    return [explain_finding(f) for f in findings]
