"""SmartTel Retention Intelligence - decision-support dashboard.

Run from the repository root:   streamlit run app/app.py
The app only READS saved artifacts; it never retrains anything.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets `import src...` work when launched as `streamlit run app/app.py`

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from src.roi_simulator import (DEFAULTS, breakeven_save_rate, campaign_kpis, clv,
                               economics, profit_curve)

FS = ROOT / "data" / "feature_store"
st.set_page_config(page_title="📊 SmartTel Retention Intelligence", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- data layer
@st.cache_data
def load_artifacts():
    t = pd.read_parquet(FS / "telco_prioritization_latest.parquet")
    f = pd.read_parquet(FS / "telco_features_scored_latest.parquet")
    f = f.merge(t[["customerID", "churn_probability_oof"]], on="customerID", how="left")
    cfg = json.load(open(ROOT / "configs" / "roi_assumptions.json"))
    shap_path = FS / "telco_shap_reasons_latest.parquet"
    reasons = pd.read_parquet(shap_path) if shap_path.exists() else None      # optional
    return t, f, cfg, reasons


@st.cache_resource
def load_model():
    paths = sorted(glob.glob(str(ROOT / "models" / "churn_model_xgboost_calibrated_*.joblib")))
    return joblib.load(paths[-1]) if paths else None                          # optional


@st.cache_data
def baseline_model_scores(_bundle, features):
    return rescore(_bundle, features)


def rescore(bundle, features):
    X = bundle["preprocessor"].transform(features[bundle["feature_cols"]])
    return bundle["calibrated_model"].predict_proba(X)[:, 1]


table, feats, config, reasons = load_artifacts()
bundle = load_model()
hazards = config["persona_monthly_hazard"]
base_defaults = {**DEFAULTS, **config.get("base_assumptions", {})}

# --------------------------------------------------------------------------- sidebar: assumptions
SLIDERS = [  # key, label, min, max, step
    ("save_rate", "Save rate (share of would-be churners an offer retains)", 0.02, 0.60, 0.01),
    ("discount_pct", "Discount depth (for customers who stay)", 0.0, 0.30, 0.01),
    ("discount_months", "Discount length (months)", 1, 18, 1),
    ("contact_cost", "Contact cost per customer ($)", 0.0, 50.0, 1.0),
    ("margin", "Contribution margin", 0.10, 0.80, 0.05),
    ("horizon", "CLV horizon (months)", 6, 48, 6),
    ("annual_disc", "Annual discount rate", 0.0, 0.30, 0.01),
    ("hazard_mult", "Long-run hazard multiplier", 0.25, 3.0, 0.25),
]
for key, *_ in SLIDERS:
    st.session_state.setdefault(f"a_{key}", base_defaults[key])


def reset_assumptions():
    for key, *_ in SLIDERS:
        st.session_state[f"a_{key}"] = base_defaults[key]


st.sidebar.title("Business assumptions")
st.sidebar.caption("Analyst-set placeholders, not measured values. Every dollar figure in this app depends on them.")
for key, label, lo, hi, step in SLIDERS:
    cast = int if isinstance(base_defaults[key], int) else float
    st.sidebar.slider(label, cast(lo), cast(hi), key=f"a_{key}", step=cast(step))
st.sidebar.button("Reset to defaults", on_click=reset_assumptions)
P = {key: st.session_state[f"a_{key}"] for key, *_ in SLIDERS}

# --------------------------------------------------------------------------- core calculation
p = table["churn_probability_oof"].to_numpy()
mc = table["MonthlyCharges"].to_numpy()
V = clv(mc, table["persona_name"].map(hazards).to_numpy(), P)
N, p_star, D = economics(p, mc, V, P)
n = len(table)
live = table.assign(clv=V, net_value=N, break_even_p=p_star, evar=p * V)

st.title("SmartTel Retention Intelligence")
st.caption("Decision-support for retention campaigns. Dollar figures are scenario-conditional, not forecasts.")

tabs = st.tabs(["📊 Overview",
    "👤 Customer Explorer",
    "🎯 Prioritization",
    "💰 Campaign Simulator",
    "🔮 What-if Scenarios",
    "⚙️ Model & Assumptions"])

# --------------------------------------------------------------------------- 1. overview
with tabs[0]:
    labels = (feats["Churn"] == "Yes").astype(int).to_numpy()
    top10 = np.argsort(-feats["churn_probability_oof"].to_numpy())[: int(0.10 * len(feats))]
    c = st.columns(5)
    c[0].metric("👥 Total Customers", f"{n:,}")
    c[1].metric(" ⚠️ Observed churn rate", f"{labels.mean():.1%}")
    c[2].metric(" 💸 Monthly revenue at risk", f"${(mc * p).sum():,.0f}", f"{(mc * p).sum() / mc.sum():.1%} of revenue", delta_color="off")
    c[3].metric(" 🎯 High-risk customers (p ≥ 0.5)", f"{(p >= 0.5).sum():,}")
    c[4].metric("Precision, top 10% highest risk", f"{labels[top10].mean():.1%}", f"base rate {labels.mean():.1%}", delta_color="off")

    st.subheader("Profit-optimal campaign under the current assumptions")
    opt_k = int((N > 0).sum())
    kp = campaign_kpis(np.where(N > 0)[0], p, mc, V, P)
    c = st.columns(4)
    c[0].metric("Customers worth contacting", f"{opt_k:,}", f"{opt_k / n:.1%} of base", delta_color="off")
    c[1].metric("Expected net profit (profit-optimal)", f"${kp['net_profit']:,.0f}")
    c[2].metric("ROI (profit-optimal)", f"{kp['roi']:.2f}" if not np.isnan(kp["roi"]) else "n/a")
    c[3].metric("Cost per retained customer (profit-optimal)", f"${kp['cost_per_retained']:,.0f}" if not np.isnan(kp["cost_per_retained"]) else "n/a")

    st.subheader("Customer personas")
    persona = live.groupby("persona_name").agg(
        customers=("customerID", "size"), mean_churn_probability=("churn_probability_oof", "mean"),
        monthly_revenue_at_risk=("MonthlyCharges", lambda s: (s * live.loc[s.index, "churn_probability_oof"]).sum()),
        expected_value_at_risk=("evar", "sum"))
    persona["share_of_value_at_risk"] = persona["expected_value_at_risk"] / persona["expected_value_at_risk"].sum()
    left, right = st.columns([3, 2])
    left.dataframe(persona.style.format({"mean_churn_probability": "{:.1%}", "monthly_revenue_at_risk": "${:,.0f}",
                                         "expected_value_at_risk": "${:,.0f}", "share_of_value_at_risk": "{:.1%}"}),
                   width="stretch")
    right.plotly_chart(px.bar(persona.reset_index(), x="persona_name", y="share_of_value_at_risk",
                              title="Share of expected value at risk").update_layout(xaxis_title="", yaxis_tickformat=".0%"),
                       width="stretch")

# --------------------------------------------------------------------------- 2. customer explorer
with tabs[1]:
    st.caption(" 📈 Customers are listed best-first by expected net value of contact.")
    cid = st.selectbox("Customer ID", live.sort_values("net_value", ascending=False)["customerID"], key="customer_id")
    row = live.loc[live["customerID"] == cid].iloc[0]
    c = st.columns(5)
    c[0].metric("Churn probability", f"{row['churn_probability_oof']:.1%}")
    c[1].metric("CLV (margin basis)", f"${row['clv']:,.0f}")
    c[2].metric("Break-even probability", f"{row['break_even_p']:.1%}")
    c[3].metric("Expected net value of contact", f"${row['net_value']:,.0f}")
    c[4].metric("Recommendation", "Contact" if row["net_value"] > 0 else "Do not contact")
    st.write(f"**Persona:** {row['persona_name']}  |  **Contract:** {row['Contract']}  |  "
             f"**Tenure:** {row['tenure']} months  |  **Monthly charges:** ${row['MonthlyCharges']:,.2f}")
    st.write(f"**Robustness (from the notebook's 243-scenario grid, not the sliders):** {row['contact_robustness_class']} "
             f"({row['contact_robustness_share']:.0%} of scenarios favour contact)")
    if reasons is not None:
        r = reasons.loc[reasons["customerID"] == cid].iloc[0]
        st.subheader(" 🔎 What the model weighs for this customer")
        for k in (1, 2, 3):
            arrow = "▲" if r[f"driver_{k}_effect"] == "raises risk" else "▼"
            st.write(f"{arrow} **{r[f'driver_{k}']}** — {r[f'driver_{k}_effect']}")
        st.caption("Explains the Module 3 model's weighting (SHAP). The probability shown above is the out-of-fold estimate.")
    else:
        st.info("SHAP reasons not found. Run `python scripts/compute_shap_reasons.py` to enable this panel.")

# --------------------------------------------------------------------------- 3. matrix
with tabs[2]:
    st.caption("Value tiers are terciles of CLV; risk tiers use fixed cut-offs (Low < 20%, Medium 20-50%, High ≥ 50%). "
               "The matrix is descriptive; the contact decision uses expected net value.")
    live["value_tier"] = pd.qcut(live["clv"], 3, labels=["Low value", "Medium value", "High value"]).astype(str)
    live["risk_tier"] = pd.cut(live["churn_probability_oof"], [-0.001, 0.20, 0.50, 1.001],
                               labels=["Low risk", "Medium risk", "High risk"]).astype(str)
    rows, cols = ["High value", "Medium value", "Low value"], ["Low risk", "Medium risk", "High risk"]
    counts = pd.crosstab(live["value_tier"], live["risk_tier"]).reindex(index=rows, columns=cols, fill_value=0)
    evar = live.pivot_table(index="value_tier", columns="risk_tier", values="evar", aggfunc="sum").reindex(index=rows, columns=cols).fillna(0)
    evar = 100 * evar / evar.to_numpy().sum()
    net = live.pivot_table(index="value_tier", columns="risk_tier", values="net_value", aggfunc="mean").reindex(index=rows, columns=cols)
    fig_cols = st.columns(3)
    for col, (title, data, fmt, scale) in zip(fig_cols, [("Customers", counts, ".0f", "Blues"),
                                                         ("% of expected value at risk", evar, ".1f", "Reds"),
                                                         ("Mean expected net value of contact ($)", net, ".0f", "RdYlGn")]):
        fig = px.imshow(data, text_auto=fmt, color_continuous_scale=scale, aspect="auto", title=title,
                        color_continuous_midpoint=0 if scale == "RdYlGn" else None)
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        col.plotly_chart(fig, width="stretch")
    ranked = live.sort_values("net_value", ascending=False).assign(
        recommended_action=lambda d: np.where(d["net_value"] > 0, "Contact", "Do not contact"))
    st.download_button("Download ranked target list (CSV)",
                       ranked[["customerID", "persona_name", "MonthlyCharges", "churn_probability_oof", "clv",
                               "break_even_p", "net_value", "value_tier", "risk_tier", "recommended_action"]].to_csv(index=False),
                       file_name="retention_target_list.csv", mime="text/csv")

# --------------------------------------------------------------------------- 4. campaign simulator
with tabs[3]:
    left, right = st.columns([1, 2])
    with left:
        strategy = st.radio("Rank customers by", ["Net value (CLV × risk × cost)", "Churn risk only",
                                                  "Revenue at risk (charges × risk)"], key="strategy")
        score = {"Net value (CLV × risk × cost)": N, "Churn risk only": p,
                 "Revenue at risk (charges × risk)": mc * p}[strategy]
        order = np.argsort(-score)
        mode = st.radio("Choose campaign size by", ["Budget (% of customers)", "Top N customers", "Probability threshold"],
                        key="size_mode")
        if mode == "Budget (% of customers)":
            k = int(st.slider("Customers contacted (%)", 0, 100, 20, key="budget_pct") / 100 * n)
            selected = order[:k]
        elif mode == "Top N customers":
            k = int(st.number_input("Save the top N customers", 0, n, 1000, step=100, key="top_n"))
            selected = order[:k]
        else:
            tau = st.slider("Contact if churn probability ≥", 0.0, 1.0, 0.32, 0.01, key="tau")
            selected = np.where(p >= tau)[0]
        kp = campaign_kpis(selected, p, mc, V, P)
        fmt = lambda v, f: "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f.format(v)
        st.metric("Customers contacted", f"{kp['customers_contacted']:,}", f"{kp['customers_contacted'] / n:.1%} of base", delta_color="off")
        st.metric("Expected net profit", f"${kp['net_profit']:,.0f}")
        st.metric("ROI (profit / cost)", fmt(kp["roi"], "{:.2f}"))
        st.metric("Total cost", f"${kp['total_cost']:,.0f}")
        st.metric("Expected customers saved", f"{kp['expected_saves']:,.0f}")
        st.metric("Cost per retained customer", fmt(kp["cost_per_retained"], "${:,.0f}"))
        st.metric("Monthly revenue protected", f"${kp['monthly_revenue_protected']:,.0f}")
    with right:
        curve = profit_curve(N, score)
        best_k = int(curve.argmax())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=100 * np.arange(n + 1) / n, y=curve, name=strategy, line=dict(color="#2E74B5")))
        fig.add_trace(go.Scatter(x=[100 * len(selected) / n], y=[curve[len(selected)]], mode="markers",
                                 marker=dict(size=12, color="#C0504D"), name="Your campaign"))
        fig.add_trace(go.Scatter(x=[100 * best_k / n], y=[curve[best_k]], mode="markers",
                                 marker=dict(size=12, color="#70AD47", symbol="diamond"), name="Profit-optimal size"))
        fig.update_layout(title="Expected profit vs. campaign size", xaxis_title="Customers contacted (% of base)",
                          yaxis_title="Cumulative expected profit ($)")
        st.plotly_chart(fig, width="stretch")
        fracs = np.arange(0.05, 0.651, 0.05)
        be = [100 * breakeven_save_rate(f_, p, mc, V, P) for f_ in fracs]
        fig2 = go.Figure(go.Scatter(x=100 * fracs, y=be, mode="lines+markers", name="Break-even save rate"))
        fig2.add_hline(y=100 * P["save_rate"], line_dash="dash", line_color="#C0504D",
                       annotation_text=f"Assumed save rate ({P['save_rate']:.0%})")
        fig2.update_layout(title="How effective must the offer be?", xaxis_title="Campaign size (% of customers)",
                           yaxis_title="Save rate needed to break even (%)")
        st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------------------------------- 5. what-if
with tabs[4]:
    st.warning("**Associative, not causal.** These scenarios edit a customer attribute and re-score it with the Module 3 "
               "model. The model learned that customers with these attributes tend to churn less; it does not establish "
               "that *changing* the attribute would cause the same change. Treat results as hypotheses for a pilot.")
    if bundle is None:
        st.info("Model bundle not found in models/, so what-if scenarios are disabled.")
    else:
        base_scores = baseline_model_scores(bundle, feats)
        p_base = feats["churn_probability_oof"].to_numpy()
        mc_base = feats["MonthlyCharges"].to_numpy()

        def summarize(f2, mc_new):
            # Apply the model's CHANGE in probability to the out-of-fold baseline, so the baseline stays honest.
            p_new = np.clip(p_base + rescore(bundle, f2) - base_scores, 0, 1)
            return p_new, dict(
                churners_base=p_base.sum(), churners_new=p_new.sum(),
                rate_base=p_base.mean(), rate_new=p_new.mean(),
                rev_base=(mc_base * (1 - p_base)).sum(), rev_new=(mc_new * (1 - p_new)).sum())

        sub = st.radio("Scenario", ["Contract mix", "Price discount"], horizontal=True, key="whatif")
        f2 = feats.copy()
        if sub == "Contract mix":
            share = st.slider("Share of month-to-month customers moved to a one-year contract (%)", 0, 100, 25, key="mix_share")
            mtm = np.where(feats["Contract"] == "Month-to-month")[0]
            moved = np.random.default_rng(42).choice(mtm, size=int(len(mtm) * share / 100), replace=False)
            f2.loc[f2.index[moved], "Contract"] = "One year"
            f2["contract_risk_flag"] = ((f2["Contract"] == "Month-to-month") & (f2["PaymentMethod"] == "Electronic check")).astype(int)
            _, s = summarize(f2, mc_base)
            st.caption(f"{len(moved):,} of {len(mtm):,} month-to-month customers re-contracted (random, fixed seed).")
        else:
            cut = st.slider("Monthly charge reduction (%)", 0, 30, 10, key="price_cut")
            who = st.selectbox("Apply to", ["All customers", "High-risk customers (p ≥ 0.5)", "Month-to-month customers"], key="price_who")
            mask = {"All customers": np.ones(len(feats), bool), "High-risk customers (p ≥ 0.5)": p_base >= 0.5,
                    "Month-to-month customers": (feats["Contract"] == "Month-to-month").to_numpy()}[who]
            mc_new = np.where(mask, mc_base * (1 - cut / 100), mc_base)
            f2["MonthlyCharges"] = mc_new
            _, s = summarize(f2, mc_new)
            st.caption(f"{int(mask.sum()):,} customers receive the reduction.")
        c = st.columns(4)
        c[0].metric("Expected churn rate", f"{s['rate_new']:.1%}", f"{100 * (s['rate_new'] - s['rate_base']):+.1f} pts", delta_color="inverse")
        c[1].metric("Expected churners", f"{s['churners_new']:,.0f}", f"{s['churners_new'] - s['churners_base']:+,.0f}", delta_color="inverse")
        c[2].metric("Expected monthly revenue", f"${s['rev_new']:,.0f}", f"{s['rev_new'] - s['rev_base']:+,.0f}")
        c[3].metric("Baseline expected monthly revenue", f"${s['rev_base']:,.0f}")

# --------------------------------------------------------------------------- 6. model & assumptions
with tabs[5]:
    st.subheader("Assumptions in use")
    st.dataframe(pd.DataFrame({"parameter": list(P), "current": list(P.values()),
                               "notebook default": [base_defaults[k] for k in P]}).set_index("parameter"), width="stretch")
    st.subheader("Calibration of the probabilities used here (out-of-fold)")
    frac, mean_pred = calibration_curve(labels, feats["churn_probability_oof"], n_bins=10, strategy="quantile")
    cal = go.Figure()
    cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
    cal.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name="Out-of-fold"))
    cal.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed churn frequency", height=380)
    st.plotly_chart(cal, width="stretch")
    st.write(f"Out-of-fold ROC-AUC: **{roc_auc_score(labels, feats['churn_probability_oof']):.3f}**")
    st.subheader("Fairness check: predicted vs. observed churn by group")
    fair = []
    for attr in ["gender", "SeniorCitizen"]:
        g = feats.assign(y=labels).groupby(attr).agg(customers=("y", "size"), observed_churn=("y", "mean"),
                                                    mean_predicted=("churn_probability_oof", "mean"))
        g.index = [f"{attr} = {i}" for i in g.index]
        fair.append(g)
    st.dataframe(pd.concat(fair).style.format({"observed_churn": "{:.1%}", "mean_predicted": "{:.1%}"}), width="stretch")
    st.subheader("Limitations")
    st.markdown(
        "- Margin, offer cost, discount terms, and the **save rate** are placeholders. Only a randomized pilot can measure the save rate.\n"
        "- The churn flag and tenure imply different time scales; the hazard multiplier tests this, but the inconsistency is in the data.\n"
        "- The IBM data is a single snapshot; churn drivers may differ in other markets.\n"
        "- Survival analysis and uplift modeling (Modules 4-5) are experimental, are **not** used here, and rest on an "
        "approximation and a simulated pilot respectively.\n"
        "- What-if scenarios are associative, not causal."
    )
