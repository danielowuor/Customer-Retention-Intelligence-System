"""Business-layer functions shared by notebook 06 and the Streamlit dashboard.

Every function is pure (arrays in, arrays out) so the notebook, the app and the
tests all run exactly the same code.
"""
import numpy as np

DEFAULTS = dict(
    margin=0.40,          # contribution margin on MonthlyCharges
    horizon=24,           # months of future margin counted in CLV
    annual_disc=0.10,     # annual discount rate
    hazard_mult=1.0,      # scales the long-run churn hazard of retained customers
    save_rate=0.22,       # share of would-be churners an offer retains
    contact_cost=5.0,     # fixed cost per contacted customer ($)
    discount_pct=0.10,    # discount given to contacted customers who remain
    discount_months=6,    # duration of that discount (months)
)


def clv(monthly_charges, hazard, P):
    """Margin-basis CLV over a finite horizon, discounted, with a constant monthly hazard."""
    h = np.clip(np.asarray(hazard) * P["hazard_mult"], 0, 0.99)
    r = (1 + P["annual_disc"]) ** (1 / 12) - 1
    q = (1 - h) / (1 + r)
    annuity = np.where(np.abs(1 - q) < 1e-12, P["horizon"], q * (1 - q ** P["horizon"]) / (1 - q))
    return P["margin"] * np.asarray(monthly_charges) * annuity


def economics(p, monthly_charges, V, P):
    """Expected net value N of contacting each customer, break-even probability p*, and discount cost D."""
    p, mc, V = map(np.asarray, (p, monthly_charges, V))
    D = P["discount_pct"] * mc * P["discount_months"]
    s = P["save_rate"]
    N = s * p * V - P["contact_cost"] - (1 - p + s * p) * D
    p_star = (P["contact_cost"] + D) / (s * V + (1 - s) * D)
    return N, p_star, D


def profit_curve(N, score):
    """Cumulative expected profit when contacting customers in descending `score` order (starts at 0)."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(N)[np.argsort(-np.asarray(score))])])


def campaign_kpis(selected, p, monthly_charges, V, P):
    """Expected KPIs for contacting the customers at row positions `selected`."""
    selected = np.asarray(selected, dtype=int)   # an empty list would otherwise become float and fail as an index
    p, mc, V = (np.asarray(a)[selected] for a in (p, monthly_charges, V))
    s = P["save_rate"]
    D = P["discount_pct"] * mc * P["discount_months"]
    cost = P["contact_cost"] * len(selected) + ((1 - p + s * p) * D).sum()
    saves = (s * p).sum()
    revenue_protected = (s * p * mc).sum()
    value_protected = (s * p * V).sum()
    profit = value_protected - cost
    if len(selected) == 0 or cost == 0 or saves == 0:
        return dict(customers_contacted=len(selected), total_cost=cost, expected_saves=saves,
                    monthly_revenue_protected=revenue_protected, margin_value_protected=value_protected,
                    net_profit=profit, roi=np.nan, cost_per_retained=np.nan, saves_per_100_contacts=np.nan)
    return dict(customers_contacted=len(selected), total_cost=cost, expected_saves=saves,
                monthly_revenue_protected=revenue_protected, margin_value_protected=value_protected,
                net_profit=profit, roi=profit / cost, cost_per_retained=cost / saves,
                saves_per_100_contacts=100 * saves / len(selected))


def breakeven_save_rate(fraction, p, monthly_charges, V, P, iters=40):
    """Smallest save rate at which contacting the best `fraction` of customers breaks even."""
    k = int(fraction * len(p))
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        N_mid, _, _ = economics(p, monthly_charges, V, {**P, "save_rate": mid})
        profit = np.sort(N_mid)[::-1][:k].sum()
        lo, hi = (lo, mid) if profit > 0 else (mid, hi)
    return hi
