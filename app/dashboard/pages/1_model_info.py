import streamlit as st
from api import model_info

st.title("Model Info")

try:
    info = model_info()
except Exception as e:
    st.error(f"Could not reach the API: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Average Precision", f"{info['metrics']['average_precision']:.3f}",
          delta=f"{info['metrics']['average_precision'] / info['fraud_prevalence_baseline']:.0f}x baseline")
c2.metric("ROC-AUC", f"{info['metrics']['roc_auc']:.3f}")
c3.metric("Features", info["n_features"])

st.caption(f"Trained {info['trained_at'][:10]} · default threshold {info['default_threshold']}")

st.subheader("Features")
st.write(f"{len(info['categorical_features'])} categorical, "
         f"{info['n_features'] - len(info['categorical_features'])} numeric")

with st.expander("Full feature list"):
    st.write(info["features"])
