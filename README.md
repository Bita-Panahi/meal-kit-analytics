# 🥗 Meal-Kit Subscription Analytics

> A small analytics dashboard for a simplified Nordic meal-kit subscription business, covering overview metrics, churn risk, A/B testing, and demand forecasting.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-FFF000?logo=duckdb&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)


<!-- After deploying on Streamlit Community Cloud, paste your link here: -->
**🔗 Live demo:** [Check Here](https://meal-kit-analyticsgit-gmkeksvebjfmjw8hsqb4zx.streamlit.app/) •&nbsp; **📓 Step-by-step notebooks:** `01`–`04` &nbsp;

---

## Why this project

I wanted this project to stay practical and easy to follow. Instead of using the most complex model possible, I focused on building a clear workflow: define churn properly, create useful customer features, train a simple baseline model, and turn the results into something everyone could understand.
The dashboard is interactive, so the user can change the cutoff week, the prediction window, and the number of customers. The metrics and customer list update automatically, which makes it easier to see how different churn definitions affect the results.

## What it covers

| Tab | Business question | Method  | Skills shown |
|---|---|---|---|
| **Overview** | How big is the business, and where's the revenue? | SQL on DuckDB (`JOIN`, `GROUP BY`) | SQL, local analytics database |
| **Churn** | Which active customers may churn? And why? | Logistic regression, leakage-safe framing | Churn prediction, predictive modeling |
| **A/B Test** | Did the discount make people reorder? | Two-proportion z-test | A/B testing, experimentation |
| **Demand** | How many boxes next week? | 4-week moving average | Demand forecasting |

## A few things worth highlighting

- **Leakage-safe churn framing.** The model only uses customer behaviour up to a selected cutoff week, while the churn label is based on what happens after that point. This keeps the prediction setup realistic and avoids using future information by mistake.
- **Simple, readable model.** I used logistic regression as the main model so the results are easier to interpret. The feature weights give a clear view of which factors increase churn risk, such as longer time since last order or recent late deliveries.
- **Practical evaluation.** The A/B testing section reports the statistical result and translates it into a short recommendation. This is a simple first-pass experiment analysis, not a full causal analysis. The demand section starts with a simple baseline forecast, then points to seasonal forecasting as a natural next step instead of overcomplicating the first version.

## The data is synthetic (on purpose)

There's no real customer data here, since that would not be appropriate for a public portfolio project. `generate_data.py` simulates a realistic meal-kit business (~2,000 customers, a year of weekly orders across three Nordic brands) with real structure baked in: loyal vs fragile customers, late deliveries that nudge people to quit, and a discount experiment. The simulation lets the analyses demonstrate the method end-to-end; the skills transfer directly to real data, with only the source changes.

## Run it locally

```bash
pip install -r requirements.txt
python generate_data.py                # run once to create the data/ folder
python -m streamlit run app.py         # opens the dashboard in your browser
```

To explore the analyses one step at a time, open the numbered notebooks (`01`–`04`) in Jupyter.

## Project structure

```
.
├── app.py                  # the Streamlit dashboard (four tabs)
├── generate_data.py        # creates the synthetic dataset
├── queries.sql             # the SQL behind the Overview tab
├── 01_overview_sql.ipynb   # Overview   (SQL)
├── 02_churn.ipynb          # Churn      (logistic regression)
├── 03_ab_test.ipynb        # A/B test   (two-proportion z-test)
├── 04_demand.ipynb         # Demand     (moving average)
└── requirements.txt
```

## Tech stack

Python · Streamlit · DuckDB (SQL) · pandas · NumPy · scikit-learn · SciPy · Matplotlib

---
