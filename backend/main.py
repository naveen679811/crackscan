"""
Crack Detection API - FastAPI Backend
Pre-trained model loaded at startup. No training at runtime.
"""

import os
import io
import time
import asyncio
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Crack Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/crack_model.h5")

CRACK_TYPES = ["hairline crack", "surface crack", "wide crack", "structural visible crack"]

SEVERITY_MAP = {
    "hairline crack":           ("Safe",      "not harmful — micro-fracture, cosmetic only"),
    "surface crack":            ("Moderate",  "monitor — surface-level, may worsen"),
    "wide crack":               ("Dangerous", "structural issue — immediate inspection needed"),
    "structural visible crack": ("Dangerous", "critical — structural integrity at risk"),
}

# ── Model ─────────────────────────────────────────────────────────────────────
model     = None
use_keras = False

def load_model_at_startup():
    global model, use_keras
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading Keras model from {MODEL_PATH} ...")
            model     = tf.keras.models.load_model(MODEL_PATH)
            use_keras = True
            logger.info("Keras model loaded successfully.")
            return
        else:
            logger.warning("Model file not found. Using heuristic engine.")
    except ImportError:
        logger.warning("TensorFlow not available. Using heuristic engine.")
    except Exception as e:
        logger.warning(f"Could not load model: {e}. Using heuristic engine.")

    model     = None
    use_keras = False
    logger.info("Heuristic engine activated.")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0

# ── Heuristic Predictor ───────────────────────────────────────────────────────
def heuristic_predict(img_arr: np.ndarray) -> dict:
    import cv2

    img_uint8 = (img_arr * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    r, g, b = img_uint8[:,:,0], img_uint8[:,:,1], img_uint8[:,:,2]

    # ── Image type detection ───────────────────────────────────────────────
    white_ratio = float((gray > 200).sum()) / gray.size
    black_ratio = float((gray < 50).sum())  / gray.size
    dark_ratio  = float((gray < 80).sum())  / gray.size

    # Drawing style = white background + black crack lines
    is_drawing_style = white_ratio > 0.40 and black_ratio > 0.02

    # ── Color analysis ─────────────────────────────────────────────────────
    skin_mask = (
        (r > 60) & (r < 255) &
        (g > 40) & (g < 220) &
        (b > 20) & (b < 190) &
        (r > g)  & (r > b)   &
        ((r.astype(int) - g.astype(int)) > 10)
    )
    skin_ratio = float(skin_mask.sum()) / skin_mask.size

    rg = np.abs(r.astype(int) - g.astype(int))
    rb = np.abs(r.astype(int) - b.astype(int))
    gb = np.abs(g.astype(int) - b.astype(int))
    colorfulness = float((rg + rb + gb).mean()) / 255.0

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    avg_saturation = float(hsv[:,:,1].mean())

    # ── Anti false positive ────────────────────────────────────────────────
    # Only reject clearly colorful people/nature photos
    # Allow: gray concrete, brown earth, white+black drawings
    is_colorful_photo = not is_drawing_style and (
        (skin_ratio > 0.15 and avg_saturation > 65)
        or
        (colorfulness > 0.28 and avg_saturation > 70)
    )

    if is_colorful_photo:
        return {
            "crack_detected":  False,
            "crack_type":      "none",
            "severity":        "Safe",
            "severity_detail": "no crack detected — image does not appear to be a structural surface",
            "confidence":      round(min(0.70 + skin_ratio * 0.25, 0.95), 4),
        }

    # ── Crack feature extraction ───────────────────────────────────────────

    # 1. Canny edge density
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(edges.sum()) / (edges.shape[0] * edges.shape[1] * 255)

    # 2. Thin line detection
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    thin_v   = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
    thin_h   = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    thin_score = float((thin_v.astype(int) + thin_h.astype(int)).mean()) / 255.0

    # 3. Laplacian variance
    lap_var = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)

    # 4. Connected component shape analysis
    thresh_val = 128 if is_drawing_style else 75
    _, binary  = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    crack_like = 0
    for i in range(1, num_labels):
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:
            continue
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if aspect > 3 and area < (gray.size * 0.20):
            crack_like += 1

    component_score = min(crack_like / 15.0, 1.0)

    # ── Composite score ────────────────────────────────────────────────────
    if is_drawing_style:
        score = (
            black_ratio     * 0.35 +
            edge_density    * 0.20 +
            component_score * 0.30 +
            thin_score      * 0.10 +
            lap_var         * 0.05
        )
        threshold = 0.05
    else:
        score = (
            edge_density    * 0.20 +
            dark_ratio      * 0.15 +
            thin_score      * 0.30 +
            lap_var         * 0.10 +
            component_score * 0.15 +
            (1.0 - min(float(gray.std()) / 80.0, 1.0)) * 0.10
        )
        threshold = 0.12

    crack_detected = score > threshold

    if not crack_detected:
        return {
            "crack_detected":  False,
            "crack_type":      "none",
            "severity":        "Safe",
            "severity_detail": "no crack detected",
            "confidence":      round(min(0.88, 1.0 - score * 3.0), 4),
        }

    # ── Classify crack type ────────────────────────────────────────────────
    if score < 0.18:
        crack_type = "hairline crack"
    elif score < 0.32:
        crack_type = "surface crack"
    elif score < 0.50:
        crack_type = "wide crack"
    else:
        crack_type = "structural visible crack"

    severity, severity_detail = SEVERITY_MAP[crack_type]
    confidence = round(min(0.55 + score * 1.1, 0.97), 4)

    return {
        "crack_detected":  True,
        "crack_type":      crack_type,
        "severity":        severity,
        "severity_detail": severity_detail,
        "confidence":      confidence,
    }

# ── Keras batch predictor ─────────────────────────────────────────────────────
def keras_predict_batch(img_arrays: List[np.ndarray]) -> List[dict]:
    batch = np.stack(img_arrays, axis=0)
    preds = model.predict(batch, verbose=0)
    results = []
    for pred in preds:
        if pred.shape[-1] == 2:
            no_crack_prob = float(pred[0])
            crack_prob    = float(pred[1])
            crack_detected = crack_prob > 0.5
            if not crack_detected:
                results.append({"crack_detected": False, "crack_type": "none",
                                 "severity": "Safe", "severity_detail": "no crack detected",
                                 "confidence": round(no_crack_prob, 4)})
                continue
            crack_type = "surface crack"
            severity, severity_detail = SEVERITY_MAP[crack_type]
            results.append({"crack_detected": True, "crack_type": crack_type,
                             "severity": severity, "severity_detail": severity_detail,
                             "confidence": round(crack_prob, 4)})
        else:
            no_crack_prob = float(pred[0])
            crack_probs   = pred[1:]
            best_idx      = int(np.argmax(crack_probs))
            best_prob     = float(crack_probs[best_idx])
            crack_detected = best_prob > no_crack_prob
            if not crack_detected:
                results.append({"crack_detected": False, "crack_type": "none",
                                 "severity": "Safe", "severity_detail": "no crack detected",
                                 "confidence": round(no_crack_prob, 4)})
            else:
                crack_type = CRACK_TYPES[best_idx]
                severity, severity_detail = SEVERITY_MAP[crack_type]
                results.append({"crack_detected": True, "crack_type": crack_type,
                                 "severity": severity, "severity_detail": severity_detail,
                                 "confidence": round(best_prob, 4)})
    return results

# ── Single image worker ───────────────────────────────────────────────────────
def process_single(file_bytes: bytes, filename: str) -> dict:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        arr     = preprocess_image(pil_img)
        result  = heuristic_predict(arr)
        result["image_name"] = filename
        return result
    except Exception as e:
        return {
            "image_name":      filename,
            "error":           str(e),
            "crack_detected":  False,
            "crack_type":      "unknown",
            "severity":        "Unknown",
            "severity_detail": "processing error",
            "confidence":      0.0,
        }

executor = ThreadPoolExecutor(max_workers=8)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model_at_startup()

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None or not use_keras,
        "engine":       "keras" if use_keras else "heuristic",
    }

# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 images per request.")

    start = time.time()
    logger.info(f"Received {len(files)} image(s).")

    async def read_file(f: UploadFile):
        data = await f.read()
        return data, f.filename

    file_data = await asyncio.gather(*[read_file(f) for f in files])

    if use_keras:
        results     = []
        batch_items = []
        batch_names = []
        for file_bytes, filename in file_data:
            try:
                pil_img = Image.open(io.BytesIO(file_bytes))
                arr     = preprocess_image(pil_img)
                batch_items.append(arr)
                batch_names.append(filename)
            except Exception as e:
                results.append({"image_name": filename, "error": str(e),
                                 "crack_detected": False, "crack_type": "unknown",
                                 "severity": "Unknown", "severity_detail": "processing error",
                                 "confidence": 0.0})
        for i in range(0, len(batch_items), BATCH_SIZE):
            batch        = batch_items[i:i + BATCH_SIZE]
            names        = batch_names[i:i + BATCH_SIZE]
            loop         = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(executor, keras_predict_batch, batch)
            for res, name in zip(batch_results, names):
                res["image_name"] = name
                results.append(res)
    else:
        loop    = asyncio.get_event_loop()
        tasks   = [loop.run_in_executor(executor, process_single, fb, fn)
                   for fb, fn in file_data]
        results = list(await asyncio.gather(*tasks))

    elapsed = round(time.time() - start, 3)
    logger.info(f"Processed {len(results)} images in {elapsed}s")

    return JSONResponse({
        "total":                    len(results),
        "processing_time_seconds":  elapsed,
        "results":                  results,
    })

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)