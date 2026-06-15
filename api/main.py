"""
api/main.py — Layer 6: FastAPI inference server
===============================================
Wraps the live MMIS pipeline behind a REST API. The heavy models load ONCE at
startup (via lifespan) and are reused across requests.

Run:
    uvicorn api.main:app --reload --port 8000
Then:
    GET http://localhost:8000/analyze?ticker=AAPL
    GET http://localhost:8000/image/AAPL_2026-06-15_gradcam.png
    GET http://localhost:8000/docs        (interactive Swagger UI)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CHART_DIR = Path("data/images/charts").resolve()
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once at startup
    from live_inference import LiveMMIS, SUPPORTED_TICKERS
    logger.info("Starting MMIS API — loading models...")
    STATE["engine"] = LiveMMIS()
    STATE["tickers"] = SUPPORTED_TICKERS
    logger.info("MMIS API ready.")
    yield
    STATE.clear()


app = FastAPI(
    title="MMIS — Multimodal Market Intelligence API",
    description=("Live multimodal next-day direction inference. "
                 "Research/interpretability demo — NOT financial advice."),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "MMIS inference API",
        "supported_tickers": STATE.get("tickers", []),
        "endpoints": ["/analyze?ticker=AAPL", "/image/{filename}", "/health", "/docs"],
        "disclaimer": "Research/interpretability demo — NOT financial advice.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "engine" in STATE}


@app.get("/analyze")
def analyze(ticker: str = Query(..., description="Stock ticker, e.g. AAPL"),
            gradcam: bool = Query(True, description="Also compute Grad-CAM overlay")):
    """Run the full live pipeline and (optionally) the Grad-CAM overlay."""
    engine = STATE.get("engine")
    if engine is None:
        raise HTTPException(503, "Models not loaded yet.")
    try:
        result = engine.predict(ticker)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed for '{ticker}': {e}")

    gradcam_file = None
    if gradcam:
        try:
            from gradcam import compute_gradcam
            path = compute_gradcam(engine, result)
            gradcam_file = Path(path).name
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")

    result.pop("_artifacts", None)   # tensors are not JSON-serialisable
    result["chart_file"] = Path(result["chart_path"]).name
    result["gradcam_file"] = gradcam_file
    return result


@app.get("/image/{filename}")
def image(filename: str):
    """Serve a generated chart / Grad-CAM image (path-traversal guarded)."""
    target = (CHART_DIR / filename).resolve()
    if CHART_DIR not in target.parents or not target.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(target, media_type="image/png")
