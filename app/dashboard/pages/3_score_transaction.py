import json
import streamlit as st
from pathlib import Path
from api import post, model_info
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"

st.title("Score a Transaction")

try:
    info = model_info()
except Exception as e:
    st.error(f"Could not reach the API: {e}")
    st.stop()

st.caption(
    f"The model expects {info['n_features']} features. Fill in what you have — "
    "anything omitted is sent as missing, which the model handles natively."
)
@st.cache_data
def samples():
    return pd.read_csv(DATA)

if st.button("Load a random transaction"):
    row = samples().sample(1).iloc[0]
    label = row.pop("isFraud")
    st.session_state["loaded"] = {
        "amount": float(row["TransactionAmt"]),
        "product": str(row["ProductCD"]),
        "extra": row.drop(["TransactionAmt", "ProductCD"]).dropna().to_dict(),
        "true_label": int(label),
    }
    st.rerun()

loaded = st.session_state.get("loaded")

c1, c2, c3 = st.columns(3)
amount = c1.number_input("Transaction amount ($)", 0.01, 100_000.0,
                         loaded["amount"] if loaded else 75.0, step=1.0)

products = ["W", "C", "R", "H", "S"]
product = c2.selectbox("Product code", products,
                       index=products.index(loaded["product"]) if loaded else 0)
threshold = c3.number_input("Threshold", 0.0, 1.0,
                            float(info["default_threshold"]), step=0.01)

extra_default = json.dumps(loaded["extra"], indent=2, default=str) if loaded else '{}'
extra_raw = st.text_area("Additional features (JSON)", value=extra_default, height=160)

if loaded and (amount != loaded["amount"] or product != loaded["product"]):
    del st.session_state["loaded"]
    loaded = None

if st.button("Score", type="primary"):
    try:
        extra = json.loads(extra_raw) if extra_raw.strip() else {}
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    unknown = sorted(set(extra) - set(info["features"]))
    if unknown:
        st.error(f"Not features of this model: {unknown[:5]}")
        st.stop()

    payload = {
        "transaction": {"TransactionAmt": amount, "ProductCD": product, "features": extra},
        "threshold": threshold,
    }

    try:
        result = post("/predict", payload)
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    prob = result["fraud_probability"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud probability", f"{prob:.1%}")
    m2.metric("Decision", "FLAG" if result["is_fraud"] else "PASS")
    m3.metric("Features missing", f"{result['n_features_missing']} / {info['n_features']}")

    st.progress(min(prob, 1.0))

    baseline = info["fraud_prevalence_baseline"]
    st.caption(
        f"Base rate is {baseline:.1%}, so this transaction scores {prob / baseline:.1f}x "
        f"the average. {result['n_features_missing']} features were not supplied — with "
        "most of the feature set missing, treat the score as illustrative rather than accurate."
    )

    if loaded:
        truth = "FRAUD" if loaded["true_label"] else "LEGITIMATE"
        correct = result["is_fraud"] == bool(loaded["true_label"])
        (st.success if correct else st.warning)(f"Actual label: {truth}")

    with st.expander("Raw response"):
        st.json(result)
