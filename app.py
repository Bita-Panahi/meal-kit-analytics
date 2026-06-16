import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats


st.set_page_config(page_title="Meal-Kit Analytics", layout="wide")
st.title("🥗 Meal-Kit Subscription Analytics")
st.caption("A small, fully explainable dashboard: overview, churn, A/B testing, and demand.")

st.sidebar.header("Settings")
CUTOFF = st.sidebar.slider(
    "Cutoff week ('today')",
    min_value=10, max_value=45, value=25,
    help="Pretend it's the end of this week. Churn and the A/B test are measured from here."
)
HORIZON = st.sidebar.slider(
    "Look-ahead weeks",
    min_value=2, max_value=10, value=6,
    help="A customer churns if they place no order within this many weeks after the cutoff."
)

@st.cache_data
def load_data():
    import os, generate_data
    if not os.path.exists("data/orders.csv"):
        generate_data.main()
    orders = pd.read_csv("data/orders.csv")
    customers = pd.read_csv("data/customers.csv")
    return orders, customers

orders, customers = load_data()

n = st.sidebar.slider("Customers to include", 200, len(customers), len(customers), step=100)
customers = customers.sample(n, random_state=42)
orders = orders[orders["customer_id"].isin(customers["customer_id"])]


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Churn", "A/B Test", "Demand"])

with tab1:
    st.header("Business overview")

    con = duckdb.connect()
    con.register("orders", orders)
    con.register("customers", customers)

    kpi = con.execute("""
        SELECT COUNT(*) AS total_orders,
               SUM(box_price_nok) AS revenue,
               AVG(delivered_on_time::INT) AS on_time
        FROM orders
    """).df()
    n_customers = con.execute("SELECT COUNT(*) AS c FROM customers").df()["c"][0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{int(n_customers):,}")
    c2.metric("Orders (year)", f"{int(kpi['total_orders'][0]):,}")
    c3.metric("Revenue", f"{kpi['revenue'][0]:,.0f} NOK")
    c4.metric("On-time rate", f"{kpi['on_time'][0]*100:.1f}%")

    by_brand = con.execute("""
        SELECT c.brand, SUM(o.box_price_nok) AS revenue
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.brand ORDER BY revenue DESC
    """).df()
    by_region = con.execute("""
        SELECT region, COUNT(*) AS customers
        FROM customers
        GROUP BY region ORDER BY customers DESC
    """).df()
    con.close()

    st.subheader("Revenue per brand")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#2a9d8f', '#e76f51', '#264653']
    bars = ax.bar(by_brand['brand'], by_brand['revenue'] / 1e6, color=colors)
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set_title('Revenue per brand (NOK)')
    ax.set_ylabel('Million NOK')
    ax.tick_params(axis='x', rotation=15)
    ax.margins(y=0.12)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Customers by region")
    by_region = by_region.sort_values("customers")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(by_region["region"], by_region["customers"], color="#2a9d8f")
    ax2.bar_label(ax2.containers[0], padding=3)
    ax2.set_xlabel("Customers")
    ax2.margins(x=0.10)
    fig2.tight_layout()
    st.pyplot(fig2)
    
    
with tab2:
    st.header("Who is about to cancel?")
    st.write("Logistic regression on the **active base**, framed point-in-time so the "
             "future can't leak into the features.")

    history = orders[orders["week"] <= CUTOFF]
    recent_ids = history[history["week"] > CUTOFF - 6]["customer_id"].unique()
    h = history[history["customer_id"].isin(recent_ids)]
    last8 = h[h["week"] > CUTOFF - 8]

    feat = pd.DataFrame({"customer_id": recent_ids}).set_index("customer_id")
    feat["weeks_since_last_order"] = CUTOFF - h.groupby("customer_id")["week"].max()
    feat["orders_last_8w"] = last8.groupby("customer_id")["order_id"].count()
    feat["late_last_8w"] = last8.assign(late=(~last8["delivered_on_time"]).astype(int)) \
                                .groupby("customer_id")["late"].sum()
    feat["tenure_weeks"] = CUTOFF - customers.set_index("customer_id")["signup_week"]
    feat["avg_meals"] = h.groupby("customer_id")["n_meals"].mean()
    feat = feat.fillna(0)

    future_ids = set(orders[(orders["week"] > CUTOFF) &
                            (orders["week"] <= CUTOFF + HORIZON)]["customer_id"].unique())
    feat["churned"] = (~feat.index.isin(future_ids)).astype(int)

    names = ["weeks_since_last_order", "orders_last_8w", "late_last_8w",
             "tenure_weeks", "avg_meals"]
    X = (feat[names] - feat[names].mean()) / feat[names].std()
    y = feat["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = LogisticRegression().fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Active customers scored", f"{len(feat):,}")
    m2.metric("Churn rate", f"{feat['churned'].mean()*100:.1f}%")
    m3.metric("Model AUC", f"{auc:.2f}")

    st.subheader("What drives churn (model weights)")
    coef = pd.DataFrame({"feature": names, "weight": model.coef_[0]}) \
             .sort_values("weight")

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#e76f51" if w > 0 else "#2a9d8f" for w in coef["weight"]]
    ax.barh(coef["feature"], coef["weight"], color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight  (red = raises churn risk, green = lowers it)")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Top 10 customers at risk")
    feat["churn_risk"] = model.predict_proba(X)[:, 1]
    top = feat.sort_values("churn_risk", ascending=False).head(10)
    st.dataframe(top[["churn_risk", "weeks_since_last_order",
                      "orders_last_8w", "late_last_8w"]].round(2))
    
    
with tab3:
    st.header("Did the discount work?")

    active_ids = orders[(orders["week"] > CUTOFF - 6) &
                        (orders["week"] <= CUTOFF)]["customer_id"].unique()
    exp = customers[customers["in_experiment"] &
                    customers["customer_id"].isin(active_ids)].copy()

    future_ids = set(orders[(orders["week"] > CUTOFF) &
                            (orders["week"] <= CUTOFF + HORIZON)]["customer_id"].unique())
    exp["reordered"] = exp["customer_id"].isin(future_ids).astype(int)

    treat = exp[exp["got_discount"]]
    ctrl = exp[~exp["got_discount"]]
    p1, n1 = treat["reordered"].mean(), len(treat)
    p2, n2 = ctrl["reordered"].mean(), len(ctrl)

    p_pool = (treat["reordered"].sum() + ctrl["reordered"].sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Discount reorder rate", f"{p1*100:.1f}%")
    c2.metric("Control reorder rate", f"{p2*100:.1f}%")
    c3.metric("p-value", f"{p_value:.4f}")


    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(["No discount", "Discount"], [p2, p1], color=["#999999", "#2a9d8f"])
    ax.set_ylabel("Reorder rate")
    ax.set_title("A/B test: discount vs control")
    ax.set_ylim(0, 1)
    for i, v in enumerate([p2, p1]):
        ax.text(i, v + 0.02, f"{v*100:.0f}%", ha="center")
    fig.tight_layout()
    st.pyplot(fig)

    if p_value < 0.05:
        st.success(f"The discount lifted reorders by {(p1-p2)*100:.1f} percentage points "
                   f"(statistically significant, p < 0.05). Recommendation: roll it out wider "
                   f"and keep measuring.")
    else:
        st.warning("We can't yet conclude the discount helped (p ≥ 0.05). "
                   "Recommendation: keep testing with a larger group.")

with tab4:
    st.header("How many boxes next week?")

    weekly = orders.groupby("week").size().rename("orders").reset_index()
    weekly["moving_avg_4w"] = weekly["orders"].rolling(window=4).mean()
    forecast = weekly["moving_avg_4w"].iloc[-1]

    st.metric("Forecast for next week", f"{forecast:.0f} orders")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weekly["week"], weekly["orders"], alpha=0.5, label="Actual orders")
    ax.plot(weekly["week"], weekly["moving_avg_4w"], linewidth=2.5, label="4-week average")
    ax.scatter([weekly["week"].iloc[-1] + 1], [forecast], color="red", zorder=5, label="Forecast")
    ax.set_xlabel("Week"); ax.set_ylabel("Orders"); ax.legend()
    st.pyplot(fig)

    st.caption("A 4-week moving average is a simple baseline. A seasonal model "
               "(for example, Prophet) is the natural next step, but baseline first.")
