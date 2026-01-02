# dashboard.py

import streamlit as st
import plotly.graph_objects as go
from core_engine import get_live_snapshot

st.set_page_config(layout="wide")
st.title("📊 Real-Time Volatility & Tail-Risk Engine")

snapshot = get_live_snapshot()

if snapshot is None:
    st.warning("Waiting for market data...")
    st.stop()

# ---------- TOP METRICS ----------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Spot Price", f"{snapshot['spot']:.2f}")
col2.metric("Rolling Vol", f"{snapshot['rolling_vol']:.2%}")
col3.metric("GARCH Vol", f"{snapshot['garch_vol']:.2%}")
col4.metric("Call Price", f"{snapshot['call_price']:.2f}")

# ---------- REGIME ----------
st.subheader("Volatility Regime")
regime = snapshot["regime"]

if regime == "STRESS_VOL":
    st.error("STRESS REGIME – Elevated Tail Risk")
elif regime == "LOW_VOL":
    st.success("LOW VOLATILITY – Compressed Risk")
else:
    st.info("NORMAL VOLATILITY REGIME")

# ---------- TAIL RISK ----------
st.subheader("Tail Risk (95%)")
st.write(f"**VaR:** {snapshot['var']:.4%}")
st.write(f"**Expected Shortfall:** {snapshot['es']:.4%}")

# ---------- PRICE CHART ----------
st.subheader("Live Price")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=snapshot["df"].index,
    y=snapshot["df"]["close"],
    mode="lines",
    name="Price"
))
st.plotly_chart(fig, use_container_width=True)

st.caption("Updates on refresh. This is a risk engine, not a trading signal.")
