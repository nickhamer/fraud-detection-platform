import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.colors as mcolors

from api import get, model_info

st.title("Threshold Explorer")

st.set_page_config(
    page_title="Threshold Explorer",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.8rem; padding-bottom: 1rem; max-width: 1100px;}
[data-testid="stMetricValue"] { font-size: 1.4rem; }
h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>

    [data-testid="stVerticalBlock"] { gap: 0.4rem; }
    [data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    [data-testid="stWidgetLabel"] p { font-size: 0.8rem; margin-bottom: 0.1rem; }
    [data-testid="stSlider"] { padding-top: 0.2rem; padding-bottom: 0.2rem; }
    [data-testid="stNumberInput"] input { padding-top: 0.2rem; padding-bottom: 0.2rem; }

    [data-testid="stMetric"] { padding: 0.2rem 0; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricLabel"] p { font-size: 0.75rem; }

    [data-testid="stCaptionContainer"] p { font-size: 0.75rem; margin-bottom: 0.2rem; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=300)
def curve(n_points=500):
    return get("/thresholds", n_points=n_points)

try:
    c = curve(n_points=500)
    info = model_info()
except Exception as e:
    st.error(f"Could not reach the API: {e}")
    st.stop()


BUNDLE_PATH = Path(__file__).resolve().parents[2] / "models" / "fraud_model.joblib"
print('looking for model in ', BUNDLE_PATH)
b = joblib.load(BUNDLE_PATH)
pc = b["pr_curve"]
print("n:", len(pc["thresholds"]))
print("thr range:", min(pc["thresholds"]), max(pc["thresholds"]))
print("prec range:", min(pc["precision"]), max(pc["precision"]))
print("AP in metrics:", b["metrics"]["average_precision"])

pc = b["pr_curve"]
t = np.array(pc["thresholds"]); p = np.array(pc["precision"]); r = np.array(pc["recall"])
for i in [0, 1, len(t)//2, len(t)-2, len(t)-1]:
    print(i, f"thr={t[i]:.4f} prec={p[i]:.4f} rec={r[i]:.4f}")
print("thr sorted ascending?", np.all(np.diff(t) > 0))
print("argmax precision at index", p.argmax(), "of", len(p))


thr = np.array(c["thresholds"])
prec = np.array(c["precision"])
rec = np.array(c["recall"])
prevalence = info["fraud_prevalence_baseline"]


c1, c2 = st.columns([1, 3])
volume  = c1.number_input("Transactions per day", 1_000, 10_000_000, 100_000, step=1_000)
t = c2.slider("Alert threshold", float(thr.min()), float(thr.max()),
              float(info["default_threshold"]), step=0.01)


i = int(np.argmin(np.abs(thr - t)))
p, r = prec[i], rec[i]

n_fraud = volume * prevalence
caught = n_fraud * r
missed = n_fraud - caught
flagged = caught / p if p > 0 else 0
false_alarms = flagged - caught


st.caption(
    f"At this threshold, roughly **{caught:,.0f}** of the ~{n_fraud:,.0f} daily frauds are "
    f"caught and **{missed:,.0f}** slip through, at the cost of **{false_alarms:,.0f}** "
    f"legitimate transactions sent for review — about "
    f"{false_alarms / max(caught, 1):.1f} false alarms per fraud found."
)

def _thousands(x, pos):
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


n_fraud_all = volume * prevalence
caught_all = n_fraud_all * rec
missed_all = n_fraud_all - caught_all
flagged_all = np.divide(caught_all, prec, out=np.zeros_like(prec), where=prec > 0)
fp_all = np.maximum(flagged_all - caught_all, 0)


metrics_area_above = st.container()

left, right = st.columns([1, 3])
with left:
    st.metric("Precision", f"{p:.1%}")
    st.metric("Recall", f"{r:.1%}")
    st.metric("Flagged / day", f"{flagged:,.0f}")
    st.metric("False alarms / day", f"{false_alarms:,.0f}")
    
    cost_fn = st.number_input("Cost of a missed fraud ($)", 1, 100_000, 200, step=10)
    cost_fp = st.number_input("Cost of a false alarm ($)", 0, 10_000, 5, step=1)
with right:
    plot_area = st.container()



total_cost = missed_all * cost_fn + fp_all * cost_fp

best = int(np.argmin(total_cost))
cost_here = total_cost[i]
p_best, r_best = prec[best], rec[best]


fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(8, 5), sharex=True,
    height_ratios=[3, 1.3], gridspec_kw={"hspace": 0}
)

# --- top: PR curve ---
sc = ax.scatter(rec, prec, c=thr, cmap="viridis", s=8)
fig.colorbar(sc, ax=[ax, ax2], label="threshold")

color = sc.cmap(sc.norm(thr[i]))
color_best = sc.cmap(sc.norm(thr[best]))
ax.axhline(prevalence, ls=":", color="grey", lw=1, label=f"random (precision {prevalence:.3f})")
ax.hlines(p, 0, r, color=color, ls="--", lw=1.2)
ax.vlines(r, 0, p, color=color, ls="--", lw=1.2)
ax.hlines(p_best, 0, r_best, color=color_best, ls="--", lw=1.2)
ax.vlines(r_best, 0, p_best, color=color_best, ls="--", lw=1.2)
ax.plot(r, p, "o", color=color, ms=8, mec="black", mew=0.5)
ax.plot(r_best, p_best, "*", color=color_best, ms=14, mec="white", mew=0.8)
ax.text(0.1, p_best + 0.01, 'optimum', color=color_best, fontsize=12)
ax.set_ylabel("Precision")
ax.set_ylim(0, 1)
ax.xaxis.set_tick_params(bottom=True, direction="in", labelbottom=False)
ax.grid(alpha=0.4)


# --- bottom: cost ---
ax2.scatter(rec, total_cost, c=thr, cmap="viridis", s=8, norm=sc.norm)
ax2.axvline(rec[best], ls="--", color=color_best, lw=1.2,
            label=f"optimum (recall {rec[best]:.2f}, thr {thr[best]:.3f})")
ax2.axvline(rec[i], ls="--", color=color, lw=1.2,
            label=f"selected (recall {rec[i]:.2f}, thr {t:.2f})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Total cost ($/day)")
ax2.set_xlim(0, 1)
ax2.grid(alpha=0.4)
ax2.yaxis.set_major_formatter(FuncFormatter(_thousands))
ax2.tick_params(axis="y", colors="steelblue", labelsize=9)
ax2.yaxis.label.set_color("steelblue")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

with metrics_area_above:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cost at your threshold", f"${cost_here:,.0f}/day")
    k2.metric("Optimal threshold", f"{thr[best]:.3f}")
    k3.metric("Cost at optimum", f"${total_cost[best]:,.0f}/day")
    k4.metric("","",delta=f"-${cost_here - total_cost[best]:,.0f} at optimum",
              delta_color="inverse")

with plot_area:
    st.pyplot(fig, width="content")


st.divider()
with st.expander("Solve for a target", expanded=True):
    mode = st.radio("Target", ["precision", "recall"], horizontal=True)
    target = st.slider(f"Desired {mode}", 0.05, 0.95, 0.5, step=0.05)
    try:
        st.json(get("/threshold-for", **{mode: target}))
    except Exception as e:
        st.warning(f"Unreachable on this curve: {e}")


st.caption(
    "Precision and recall come from a held-out test set drawn from a later time period "
    "than the training data. They estimate future performance but don't guarantee it — "
    "fraud patterns drift, and realised precision on new transactions is typically lower. "
    "Cost figures are illustrative, not measured."
)
