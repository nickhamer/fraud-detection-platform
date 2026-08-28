from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from api import post, model_info

st.title("Evaluate on Labelled Data")

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_eval.csv"

try:
    info = model_info()
except Exception as e:
    st.error(f"Could not reach the API: {e}")
    st.stop()

st.caption(
    "Upload transactions with an `isFraud` column to measure how the model performs on "
    "your data. Useful for checking whether performance holds up on a different time "
    "period than the model was validated on."
)

src = st.radio("Data source", ["Bundled sample", "Upload a CSV"], horizontal=True)

df = None
if src == "Bundled sample":
    if SAMPLE.exists():
        df = pd.read_csv(SAMPLE)
        st.caption(f"{len(df):,} held-out test transactions at natural prevalence.")
    else:
        st.warning("Sample file not found in the repo.")
else:
    upload = st.file_uploader("CSV with an isFraud column", type="csv")
    if upload is not None:
        df = pd.read_csv(upload)

if df is None:
    st.stop()

if "isFraud" not in df.columns:
    st.error("No isFraud column found — labels are required to compute metrics.")
    st.stop()

c1, c2 = st.columns(2)
mode = c1.radio("Choose the operating point by", ["default threshold", "precision", "recall"])
target = None
if mode != "default threshold":
    target = c2.slider(f"Target {mode}", 0.05, 0.95, 0.5, step=0.05)

if not st.button("Evaluate", type="primary"):
    st.stop()

labels = df["isFraud"].astype(int).tolist()
features = df.drop(columns=["isFraud"])

known = set(info["features"])
rows = []
for _, r in features.iterrows():
    d = {k: v for k, v in r.items() if k in known and pd.notna(v)}
    rows.append({
        "TransactionAmt": float(d.pop("TransactionAmt", 0.0)) or 0.01,
        "ProductCD": str(d.pop("ProductCD", "W")),
        "features": d,
    })

payload = {"transactions": rows, "labels": labels}
if mode == "precision":
    payload["target_precision"] = target
elif mode == "recall":
    payload["target_recall"] = target

with st.spinner(f"Scoring {len(rows):,} transactions..."):
    try:
        res = post("/evaluate", payload)
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

op = res["operating_point"]

m = st.columns(4)
m[0].metric("Average Precision", f"{res['average_precision']:.3f}",
            delta=f"{res['average_precision'] / res['prevalence']:.0f}x baseline")
m[1].metric("ROC-AUC", f"{res['roc_auc']:.3f}")
m[2].metric("Prevalence", f"{res['prevalence']:.2%}")
m[3].metric("Transactions", f"{res['n']:,}")

st.subheader(f"At threshold {op['threshold']:.3f}")

k = st.columns(4)
k[0].metric("Precision", f"{op['precision']:.1%}")
k[1].metric("Recall", f"{op['recall']:.1%}")
k[2].metric("Flagged", f"{op['n_flagged']:,}")
cm = op["confusion_matrix"]
k[3].metric("False alarms per fraud", f"{cm['fp'] / max(cm['tp'], 1):.1f}")

st.dataframe(
    pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["actually legitimate", "actually fraud"],
        columns=["predicted legitimate", "predicted fraud"],
    ),
    use_container_width=True,
)

if op["requested_precision"] is not None:
    st.caption(
        f"Requested precision {op['requested_precision']:.0%}, achieved "
        f"{op['precision']:.1%}. The threshold is chosen from this same data, so the two "
        "match closely by construction — on genuinely new data the achieved value would "
        "typically be lower."
    )
elif op["requested_recall"] is not None:
    st.caption(
        f"Requested recall {op['requested_recall']:.0%}, achieved {op['recall']:.1%}."
    )

with st.expander("Raw response"):
    st.json(res)
