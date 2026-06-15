"""
dashboard/app.py — Layer 6: Streamlit dashboard
===============================================
Enter a ticker → calls the FastAPI backend (/analyze) → shows the live
candlestick chart, the Grad-CAM explainability overlay, the multimodal
prediction, uncertainty band, regime, and live news sentiment.

Run (with the API already running on :8000):
    streamlit run dashboard/app.py
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("MMIS_API_URL", "http://127.0.0.1:8000")
TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "SPY"]

st.set_page_config(page_title="MMIS — Market Intelligence", page_icon="📊", layout="wide")

st.title("📊 Multimodal Market Intelligence System")
st.caption("Price + News (FinBERT) + Chart images (EfficientNet) → cross-modal "
           "attention → regime-conditioned, uncertainty-aware next-day direction.")

st.warning(
    "⚠️ **Research / interpretability demo — NOT financial advice.** "
    "Out-of-sample this model has **no demonstrated directional edge (≈ coin-flip)**. "
    "It is a working multimodal ML/MLOps + explainability system, not an alpha engine."
)

# ── Sidebar controls ────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    api_url = st.text_input("API URL", API_URL)
    ticker = st.selectbox("Ticker", TICKERS, index=0)
    custom = st.text_input("…or custom ticker", "")
    do_gradcam = st.checkbox("Compute Grad-CAM", value=True)
    run = st.button("🔮 Analyze", use_container_width=True, type="primary")
    st.divider()
    try:
        h = requests.get(f"{api_url}/health", timeout=3).json()
        st.success("API: online" if h.get("models_loaded") else "API: loading…")
    except Exception:
        st.error("API offline — start it with:\n`uvicorn api.main:app --port 8000`")


def fetch_image(api_url, filename):
    r = requests.get(f"{api_url}/image/{filename}", timeout=30)
    return r.content if r.status_code == 200 else None


# ── Main ────────────────────────────────────────────────────────
if run:
    sym = (custom.strip() or ticker).upper()
    with st.spinner(f"Running live multimodal inference for {sym}…"):
        try:
            resp = requests.get(f"{api_url}/analyze",
                                params={"ticker": sym, "gradcam": do_gradcam},
                                timeout=180)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
        st.stop()

    d = resp.json()
    arrow = {"UP": "🟢 ▲ UP", "DOWN": "🔴 ▼ DOWN", "FLAT": "🟡 ▬ FLAT"}[d["direction"]]

    st.subheader(f"{d['ticker']} — as of {d['as_of_date']}  (last close ${d['last_close']})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted direction", arrow)
    c2.metric("Calibrated P(up)", f"{d['up_probability_calibrated']*100:.1f}%")
    c3.metric("Market regime", d["regime"])
    # Higher variance = less confident. Show a simple confidence proxy.
    conf = max(d["probabilities"].values())
    c4.metric("Model confidence", f"{conf*100:.1f}%",
              help=f"MC-Dropout predictive variance: {d['uncertainty']:.5f}")

    st.markdown("**3-class probabilities** (Down / Flat / Up)")
    pcols = st.columns(3)
    for col, key, label in zip(pcols, ["down", "flat", "up"], ["Down", "Flat", "Up"]):
        col.progress(min(d["probabilities"][key], 1.0), text=f"{label}: {d['probabilities'][key]*100:.1f}%")

    # Charts: raw candlesticks + Grad-CAM
    g1, g2 = st.columns(2)
    chart_bytes = fetch_image(api_url, d["chart_file"]) if d.get("chart_file") else None
    if chart_bytes:
        g1.image(chart_bytes, caption="Live 20-day candlestick chart", use_container_width=True)
    if d.get("gradcam_file"):
        gc_bytes = fetch_image(api_url, d["gradcam_file"])
        if gc_bytes:
            g2.image(gc_bytes, caption="Grad-CAM: candles that drove this prediction",
                     use_container_width=True)

    # Sentiment + headlines
    s = d["sentiment"]
    st.markdown(f"**Live news sentiment** (FinBERT, {s['headline_count']} headlines): "
                f"🟢 pos {s['pos']:.2f} · ⚪ neu {s['neu']:.2f} · 🔴 neg {s['neg']:.2f}")
    if d.get("headlines"):
        with st.expander("Recent headlines used"):
            for h in d["headlines"]:
                st.write("• " + h)

    st.info(d["disclaimer"])
else:
    st.info("Pick a ticker in the sidebar and hit **Analyze** to run the live pipeline.")
