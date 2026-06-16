import numpy as np
import pandas as pd
from pathlib import Path
import duckdb


SEED = 42
N_CUSTOMERS = 2000
N_WEEKS = 52
BRANDS = ["Linas Matkasse", "GodtLevert", "RetNemt"]
BRAND_COUNTRY = {"Linas Matkasse": "Sweden", "GodtLevert": "Norway", "RetNemt": "Denmark"}
CITIES = {
    "Sweden":  ["Stockholm", "Gothenburg", "Malmo", "Uppsala"],
    "Norway":  ["Oslo", "Bergen", "Trondheim", "Stavanger"],
    "Denmark": ["Copenhagen", "Aarhus", "Odense", "Aalborg"],
}

rng = np.random.default_rng(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def make_customers():
    customer_id = np.arange(1, N_CUSTOMERS + 1)
    brand = rng.choice(BRANDS, size=N_CUSTOMERS)
    country = np.array([BRAND_COUNTRY[b] for b in brand])
    region = np.array([rng.choice(CITIES[c]) for c in country])
    plan_meals = rng.choice([3, 4, 5], size=N_CUSTOMERS)

    # Most customers sign up at week 0, but some join later in the year.
    # That keeps total demand from collapsing as early customers drift away.
    signup_week = rng.integers(0, N_WEEKS - 8, size=N_CUSTOMERS)

    loyalty = np.clip(rng.normal(0.88, 0.06, size=N_CUSTOMERS), 0.70, 0.985)

    # A/B TEST SETUP
    in_experiment = rng.random(N_CUSTOMERS) < 0.5
    got_discount = in_experiment & (rng.random(N_CUSTOMERS) < 0.5)

    customers = pd.DataFrame({
        "customer_id": customer_id,
        "brand": brand,
        "region": region,
        "plan_meals": plan_meals,
        "signup_week": signup_week,
        "in_experiment": in_experiment,
        "got_discount": got_discount,
        "loyalty": loyalty,
    })
    return customers


def make_orders(customers):
    rows = []

    active = np.zeros(N_CUSTOMERS, dtype=bool)
    last_late = np.zeros(N_CUSTOMERS, dtype=bool)

    loyalty = customers["loyalty"].to_numpy()
    signup = customers["signup_week"].to_numpy()
    plan = customers["plan_meals"].to_numpy()
    discount = customers["got_discount"].to_numpy()
    cust_ids = customers["customer_id"].to_numpy()

    for week in range(N_WEEKS):
        newly_joined = (signup == week)
        active = active | newly_joined
        p_reorder = loyalty.copy()
        p_reorder = p_reorder - 0.18 * last_late
        p_reorder = p_reorder + 0.08 * discount
        p_reorder = np.clip(p_reorder, 0.0, 0.99)

        draw = rng.random(N_CUSTOMERS)
        orders_this_week = active & (draw < p_reorder)

        active = active & orders_this_week

        idx = np.where(orders_this_week)[0]
        late_flags = rng.random(len(idx)) < 0.12
        last_late[:] = False
        last_late[idx] = late_flags

        for j, i in enumerate(idx):
            meals = plan[i]
            # Box price in NOK: roughly 99 kr per meal, with small noise.
            price = round(meals * 99 * rng.normal(1.0, 0.03), 0)
            rows.append({
                "customer_id": int(cust_ids[i]),
                "week": int(week),
                "n_meals": int(meals),
                "box_price_nok": float(price),
                "delivered_on_time": bool(not late_flags[j]),
            })

    orders = pd.DataFrame(rows)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))
    return orders

def main():
    customers = make_customers()
    orders = make_orders(customers)

    customers_to_save = customers.drop(columns=["loyalty"])

    customers_to_save.to_csv(DATA_DIR / "customers.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    db_path = DATA_DIR / "data.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE customers AS SELECT * FROM customers_to_save")
    con.execute("CREATE TABLE orders AS SELECT * FROM orders")
    con.close()

    print("Data created in:", DATA_DIR)
    print(f"  customers.csv : {len(customers_to_save):>6} rows")
    print(f"  orders.csv    : {len(orders):>6} rows")
    print(f"  data.duckdb   : tables 'customers' and 'orders'")


if __name__ == "__main__":
    main()
