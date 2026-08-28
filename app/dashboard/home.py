import streamlit as st

from api import BASE_URL, get

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide",
)


st.title("Transaction Fraud Detection")

st.markdown(
    """
A gradient-boosted model that scores card transactions for fraud risk, trained on the
IEEE-CIS dataset. Fraud is rare here — about 3.4% of transactions — so the interesting
question isn't *how accurate* the model is, but where you choose to set the alert
threshold: catch more fraud and review more false alarms, or keep the queue small and
miss more.

Use the pages in the sidebar to explore that trade-off.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pages")
    st.markdown(
        """
- **Model Info** — headline metrics and the feature set
- **Threshold Explorer** — move the threshold and see what it costs
- **Score Transaction** — score a single transaction
- **Evaluate Upload** — upload labelled data and measure performance
"""
    )

with col2:
    st.subheader("API status")
    try:
        get("/health")
        st.success(f"Connected to {BASE_URL}")
        st.link_button("Open API docs", f"{BASE_URL}/docs")
    except Exception as e:
        st.error(f"Cannot reach the API at {BASE_URL}")
        st.caption(str(e))
        st.markdown("Start it with `uvicorn app.api.main:app --port 8000`")

st.divider()
st.caption(
    "Validation uses a time-based split (earlier transactions train, later ones test) "
    "to avoid leaking future information. Reported metrics are not comparable to Kaggle "
    "leaderboard scores, which use a different held-out set."
)
