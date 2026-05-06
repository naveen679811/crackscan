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
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Crack Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/crack_model.h5")

CRACK_TYPES = ["hairline crack", "surface crack", "wide crack", "structural visible crack"]
SEVERITY_MAP = {
    "hairline crack": ("Safe", "not harmful — micro-fracture, cosmetic only"),
    "surface crack": ("Moderate", "monitor — surface-level, may worsen"),
    "wide crack": ("Dangerous", "structural issue — immediate inspection needed"),
    "structural visible crack": ("Dangerous", "critical — structural integrity at risk"),
}

# ─── Model Singleton ──────────────────────────────────────────────────────────
model = None
use_keras = False

def load_model_at_startup():
    global model, use_keras

    # Try Keras / TensorFlow first
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading Keras model from {MODEL_PATH} ...")
            model = tf.keras.models.load_model(MODEL_PATH)
            use_keras = True
            logger.info("✅ Keras model loaded successfully.")
            return
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Falling back to rule-based engine.")
    except ImportError:
        logger.warning("TensorFlow not available. Falling back to rule-based engine.")
    except Exception as e:
        logger.warning(f"Could not load Keras model: {e}. Falling back to rule-based engine.")

    # Fallback: rule-based heuristic engine (works without any model file)
    model = None
    use_keras = False
    logger.info("✅ Rule-based heuristic engine activated (no model file required).")


# ─── Image Preprocessing ──────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ─── Heuristic Predictor (fallback) ──────────────────────────────────────────
def heuristic_predict(img_arr: np.ndarray) -> dict:
    import cv2

    img_uint8 = (img_arr * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    r, g, b = img_uint8[:,:,0], img_uint8[:,:,1], img_uint8[:,:,2]

    # ── ANTI-FALSE-POSITIVE FILTERS (run first) ────────────────────────────

    # 1. Skin tone detector — if image has lots of skin tones, likely a person photo
    skin_mask = (
        (r > 60) & (r < 255) &
        (g > 40) & (g < 220) &
        (b > 20) & (b < 190) &
        (r > g) & (r > b) &
        ((r.astype(int) - g.astype(int)) > 10)
    )
    skin_ratio = float(skin_mask.sum()) / skin_mask.size

    # 2. Colorfulness score — real crack images are mostly gray/brown
    rg = np.abs(r.astype(int) - g.astype(int))
    rb = np.abs(r.astype(int) - b.astype(int))
    gb = np.abs(g.astype(int) - b.astype(int))
    colorfulness = float((rg + rb + gb).mean()) / 255.0

    # 3. Color variance across channels — cracks are near-grayscale
    channel_std = float(np.std([r.mean(), g.mean(), b.mean()]))

    # 4. Check if image looks like concrete/stone (low saturation, gray tones)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1]
    avg_saturation = float(saturation.mean())

    # High saturation = colorful photo (person, nature) = NOT a crack image
    # Low saturation = gray/concrete surface = likely crack image
    is_colorful_photo = (
        skin_ratio > 0.08 or          # has skin tones
        colorfulness > 0.18 or        # very colorful
        avg_saturation > 55 or        # highly saturated colors
        channel_std > 18              # big difference between RGB channels
    )

    if is_colorful_photo:
        return {
            "crack_detected": False,
            "crack_type": "none",
            "severity": "Safe",
            "severity_detail": "no crack detected — image does not appear to be a structural surface",
            "confidence": round(min(0.70 + skin_ratio * 0.25, 0.95), 4),
        }

    # ── CRACK DETECTION (only runs on gray/concrete-looking images) ────────

    # 1. Canny edge density with tight thresholds
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges.sum()) / (edges.shape[0] * edges.shape[1] * 255)

    # 2. Dark pixel ratio
    dark_ratio = float((gray < 70).sum()) / gray.size

    # 3. Thin elongated line detection (crack signature)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    thin_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
    thin_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    thin_score = float((thin_v.astype(int) + thin_h.astype(int)).mean()) / 255.0

    # 4. Laplacian variance — crack surfaces have sharp local transitions
    lap_var = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 3000.0, 1.0)

    # 5. Connected component shape analysis — cracks are long and thin
    _, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    crack_like_components = 0
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:
            continue
        aspect = max(w, h) / (min(w, h) + 1e-5)
        # Crack components: very elongated (aspect > 4) and not too large
        if aspect > 4 and area < (gray.size * 0.15):
            crack_like_components += 1
    component_score = min(crack_like_components / 20.0, 1.0)

    # 6. Texture uniformity — crack surfaces (concrete) are mostly uniform
    #    People/scenes have varied texture everywhere
    local_std = cv2.meanStdDev(gray)[1][0][0]
    surface_uniformity = 1.0 - min(float(local_std) / 80.0, 1.0)

    # ── Composite crack score ──────────────────────────────────────────────
    score = (
        edge_density    * 0.20 +
        dark_ratio      * 0.15 +
        thin_score      * 0.30 +
        lap_var         * 0.10 +
        component_score * 0.15 +
        surface_uniformity * 0.10
    )

    crack_detected = score > 0.12

    if not crack_detected:
        return {
            "crack_detected": False,
            "crack_type": "none",
            "severity": "Safe",
            "severity_detail": "no crack detected",
            "confidence": round(min(0.88, 1.0 - score * 3.5), 4),
        }

    # ── Classify crack type ────────────────────────────────────────────────
    if score < 0.18:
        crack_type = "hairline crack"
    elif score < 0.30:
        crack_type = "surface crack"
    elif score < 0.45:
        crack_type = "wide crack"
    else:
        crack_type = "structural visible crack"

    severity, severity_detail = SEVERITY_MAP[crack_type]
    confidence = round(min(0.55 + score * 1.1, 0.97), 4)

    return {
        "crack_detected": True,
        "crack_type": crack_type,
        "severity": severity,
        "severity_detail": severity_detail,
        "confidence": confidence,
    }
  

# ─── Keras Predictor ──────────────────────────────────────────────────────────
def keras_predict_batch(img_arrays: List[np.ndarray]) -> List[dict]:
    batch = np.stack(img_arrays, axis=0)
    preds = model.predict(batch, verbose=0)

    results = []
    for pred in preds:
        # Assume output: [no_crack_prob, hairline, surface, wide, structural]
        # or binary [no_crack, crack] — handle both
        if pred.shape[-1] == 2:
            no_crack_prob, crack_prob = float(pred[0]), float(pred[1])
            crack_detected = crack_prob > 0.5
            if not crack_detected:
                results.append({
                    "crack_detected": False,
                    "crack_type": "none",
                    "severity": "Safe",
                    "severity_detail": "no crack detected",
                    "confidence": round(no_crack_prob, 4),
                })
                continue
            # Default to surface crack for binary model
            crack_type = "surface crack"
            severity, severity_detail = SEVERITY_MAP[crack_type]
            results.append({
                "crack_detected": True,
                "crack_type": crack_type,
                "severity": severity,
                "severity_detail": severity_detail,
                "confidence": round(crack_prob, 4),
            })
        else:
            # 5-class: [no_crack, hairline, surface, wide, structural]
            no_crack_prob = float(pred[0])
            crack_probs = pred[1:]
            best_idx = int(np.argmax(crack_probs))
            best_prob = float(crack_probs[best_idx])
            crack_detected = best_prob > no_crack_prob

            if not crack_detected:
                results.append({
                    "crack_detected": False,
                    "crack_type": "none",
                    "severity": "Safe",
                    "severity_detail": "no crack detected",
                    "confidence": round(no_crack_prob, 4),
                })
            else:
                crack_type = CRACK_TYPES[best_idx]
                severity, severity_detail = SEVERITY_MAP[crack_type]
                results.append({
                    "crack_detected": True,
                    "crack_type": crack_type,
                    "severity": severity,
                    "severity_detail": severity_detail,
                    "confidence": round(best_prob, 4),
                })
    return results


# ─── Single Image Worker ──────────────────────────────────────────────────────
def process_single(file_bytes: bytes, filename: str) -> dict:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        arr = preprocess_image(pil_img)
        result = heuristic_predict(arr)
        result["image_name"] = filename
        return result
    except Exception as e:
        return {
            "image_name": filename,
            "error": str(e),
            "crack_detected": False,
            "crack_type": "unknown",
            "severity": "Unknown",
            "severity_detail": "processing error",
            "confidence": 0.0,
        }


executor = ThreadPoolExecutor(max_workers=8)


# ─── Startup Event ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model_at_startup()


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None or not use_keras,
        "engine": "keras" if use_keras else "heuristic",
    }


# ─── Predict Endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 images per request.")

    start = time.time()
    logger.info(f"Received {len(files)} image(s) for prediction.")

    # Read all files concurrently
    async def read_file(f: UploadFile):
        data = await f.read()
        return data, f.filename

    file_data = await asyncio.gather(*[read_file(f) for f in files])

    if use_keras:
        # Keras batch prediction
        results = []
        batch_items = []
        batch_names = []

        for file_bytes, filename in file_data:
            try:
                pil_img = Image.open(io.BytesIO(file_bytes))
                arr = preprocess_image(pil_img)
                batch_items.append(arr)
                batch_names.append(filename)
            except Exception as e:
                results.append({
                    "image_name": filename,
                    "error": str(e),
                    "crack_detected": False,
                    "crack_type": "unknown",
                    "severity": "Unknown",
                    "severity_detail": "processing error",
                    "confidence": 0.0,
                })

        # Process in batches
        for i in range(0, len(batch_items), BATCH_SIZE):
            batch = batch_items[i:i + BATCH_SIZE]
            names = batch_names[i:i + BATCH_SIZE]
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(executor, keras_predict_batch, batch)
            for res, name in zip(batch_results, names):
                res["image_name"] = name
                results.append(res)

    else:
        # Heuristic: parallel processing via thread pool
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_single, file_bytes, filename)
            for file_bytes, filename in file_data
        ]
        results = await asyncio.gather(*tasks)
        results = list(results)

    elapsed = round(time.time() - start, 3)
    logger.info(f"Processed {len(results)} images in {elapsed}s")

    return JSONResponse({
        "total": len(results),
        "processing_time_seconds": elapsed,
        "results": results,
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
