"""
gradcam.py — Layer 6 explainability: Grad-CAM on the live candlestick chart
===========================================================================
Standard Grad-CAM (Selvaraju et al. 2017), but end-to-end through the whole
visual→fusion chain: we backprop the fusion model's *predicted-class logit*
w.r.t. EfficientNet-B0's last convolutional feature maps. The resulting heatmap
highlights the regions of the chart (which candles / which part of the price
action) that most drove THIS prediction — not a generic saliency map.

Usage (with a LiveMMIS instance and a prediction result):
    from live_inference import LiveMMIS
    from gradcam import compute_gradcam
    live = LiveMMIS()
    res = live.predict("AAPL")
    overlay_path = compute_gradcam(live, res)
"""

import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)


def compute_gradcam(live, result: dict, output_path: str = None) -> str:
    """
    Args:
        live:    a LiveMMIS instance (provides efficientnet, fusion, visual scaler)
        result:  the dict returned by LiveMMIS.predict() (must include _artifacts)
        output_path: where to save the overlay PNG (defaults next to the chart)

    Returns: path to the saved Grad-CAM overlay image.
    """
    art = result["_artifacts"]
    img_tensor   = art["img_tensor"].to(live.device)   # [1,3,224,224]
    market_t     = art["market_t"].to(live.device)
    sentiment_t  = art["sentiment_t"].to(live.device)
    target_class = result["direction_class"]

    efficientnet = live.efficientnet.eval()
    fusion       = live.fusion.eval()

    # Visual scaler as torch tensors so scaling stays in the autograd graph.
    vmean = torch.tensor(live.scalers["visual"].mean_, dtype=torch.float32,
                         device=live.device)
    vscale = torch.tensor(live.scalers["visual"].scale_, dtype=torch.float32,
                          device=live.device)

    # ── Forward with gradient tracking on the conv feature maps ──────
    efficientnet.zero_grad(set_to_none=True)
    fusion.zero_grad(set_to_none=True)

    feats = efficientnet.features(img_tensor)     # [1, 1280, 7, 7]
    feats.retain_grad()
    pooled = efficientnet.avgpool(feats).flatten(1)   # [1, 1280]
    visual_scaled = (pooled - vmean) / vscale

    logits, _, _ = fusion(market_t, sentiment_t, visual_scaled)
    score = logits[0, target_class]

    # ── Backward → Grad-CAM weights ─────────────────────────────────
    score.backward()
    grads = feats.grad                              # [1,1280,7,7]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1,1280,1,1]
    cam = torch.relu((weights * feats).sum(dim=1)).squeeze()  # [7,7]
    cam = cam.detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    # ── Overlay heatmap on the original chart ───────────────────────
    chart = Image.open(result["chart_path"]).convert("RGB")
    W, H = chart.size
    cam_img = np.array(Image.fromarray(np.uint8(cam * 255)).resize((W, H), Image.BICUBIC)) / 255.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(chart)
    ax.imshow(cam_img, cmap="jet", alpha=0.45)
    ax.axis("off")
    ax.set_title(f"{result['ticker']} — Grad-CAM ({result['direction']} drivers)",
                 fontsize=10)

    if output_path is None:
        output_path = result["chart_path"].replace(".png", "_gradcam.png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=110)
    plt.close(fig)
    logger.info(f"Grad-CAM saved → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    from live_inference import LiveMMIS
    parser = argparse.ArgumentParser(description="Grad-CAM for a live MMIS prediction")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    args = parser.parse_args()

    live = LiveMMIS()
    res = live.predict(args.ticker)
    path = compute_gradcam(live, res)
    print(f"Prediction: {res['direction']} | up_prob(cal)={res['up_probability_calibrated']}")
    print(f"Grad-CAM overlay: {path}")
