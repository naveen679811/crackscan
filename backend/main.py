"""
Crack Detection API - FastAPI Backend v2
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Crack Detection API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/crack_model.h5")

CRACK_TYPES = ["hairline crack", "surface crack", "wide crack", "structural visible crack"]
SEVERITY_MAP = {
    "hairline crack":           ("Safe",      "not harmful — micro-fracture, cosmetic only"),
    "surface crack":            ("Moderate",  "monitor — surface-level, may worsen over time"),
    "wide crack":               ("Dangerous", "structural issue — immediate inspection needed"),
    "structural visible crack": ("Dangerous", "critical — structural integrity at risk"),
}

model     = None
use_keras = False

def load_model_at_startup():
    global model, use_keras
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading Keras model from {MODEL_PATH} ...")
            model = tf.keras.models.load_model(MODEL_PATH)
            use_keras = True
            logger.info("Keras model loaded.")
            return
        logger.warning("No model file found. Using heuristic engine.")
    except ImportError:
        logger.warning("TensorFlow not available. Using heuristic engine.")
    except Exception as e:
        logger.warning(f"Model load failed: {e}. Using heuristic engine.")
    model = None
    use_keras = False
    logger.info("Heuristic engine v2 activated.")

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0

def heuristic_predict(img_arr: np.ndarray) -> dict:
    import cv2

    img_uint8 = (img_arr * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    r = img_uint8[:,:,0]
    g = img_uint8[:,:,1]
    b = img_uint8[:,:,2]

    # ── Stage 1: Image type ────────────────────────────────────────────────
    white_ratio     = float((gray > 210).sum()) / gray.size
    black_ratio     = float((gray < 40).sum())  / gray.size
    dark_ratio      = float((gray < 85).sum())  / gray.size
    very_dark_ratio = float((gray < 30).sum())  / gray.size
    mean_brightness = float(gray.mean())

    is_drawing = white_ratio > 0.35 and black_ratio > 0.015
    is_dark_bg = mean_brightness < 60 and white_ratio < 0.10

    # ── Stage 2: Color / photo filter ─────────────────────────────────────
    skin_mask = (
        (r > 60) & (r < 255) & (g > 40) & (g < 220) & (b > 20) & (b < 190) &
        (r > g)  & (r > b)   & ((r.astype(int) - g.astype(int)) > 10)
    )
    skin_ratio     = float(skin_mask.sum()) / skin_mask.size
    colorfulness   = float((
        np.abs(r.astype(int)-g.astype(int)) +
        np.abs(r.astype(int)-b.astype(int)) +
        np.abs(g.astype(int)-b.astype(int))
    ).mean()) / 255.0
    hsv            = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    avg_saturation = float(hsv[:,:,1].mean())
    avg_brightness = float(hsv[:,:,2].mean())

    is_colorful_photo = not is_drawing and not is_dark_bg and (
        (skin_ratio > 0.12 and avg_saturation > 60) or
        (colorfulness > 0.25 and avg_saturation > 68 and avg_brightness > 80)
    )

    if is_colorful_photo:
        return {
            "crack_detected": False, "crack_type": "none", "severity": "Safe",
            "severity_detail": "no crack detected — not a structural surface",
            "confidence": round(min(0.72 + skin_ratio * 0.20, 0.95), 4),
        }

    # ── Stage 3: Feature extraction ────────────────────────────────────────
    blurred          = cv2.GaussianBlur(gray, (3,3), 0)
    edges_tight      = cv2.Canny(blurred, 50, 150)
    edges_loose      = cv2.Canny(blurred, 20, 80)
    edge_tight       = float(edges_tight.sum()) / (gray.size * 255)
    edge_loose       = float(edges_loose.sum()) / (gray.size * 255)

    # Thin line score
    thin_score = float((
        cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(1,11))).astype(np.int32) +
        cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(11,1))).astype(np.int32) +
        cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))).astype(np.int32)
    ).mean()) / 255.0

    lap_var = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 2500.0, 1.0)

    # Connected components
    thresh_val = 140 if is_drawing else (40 if is_dark_bg else 80)
    inv_flag   = cv2.THRESH_BINARY if is_dark_bg else cv2.THRESH_BINARY_INV
    _, binary  = cv2.threshold(gray, thresh_val, 255, inv_flag)
    binary     = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    crack_like        = 0
    total_crack_area  = 0
    max_comp          = 0
    for i in range(1, len(stats)):
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20: continue
        aspect = max(w,h) / (min(w,h) + 1e-5)
        if aspect > 2.5 and area < gray.size * 0.25:
            crack_like       += 1
            total_crack_area += area
            max_comp          = max(max_comp, area)

    component_score  = min(crack_like / 12.0, 1.0)
    crack_area_ratio = total_crack_area / (gray.size + 1e-5)
    large_comp_ratio = max_comp / (gray.size + 1e-5)
    mean_crack_width = float(np.mean(binary.astype(np.float32), axis=1).mean()) / 255.0
    network_density  = float(cv2.dilate(binary, np.ones((3,3),np.uint8)).sum()) / (gray.size * 255)

    # ── Stage 4: Composite score ───────────────────────────────────────────
    if is_drawing:
        score     = (black_ratio*0.30 + edge_tight*0.15 + component_score*0.25 +
                     thin_score*0.15 + network_density*0.10 + lap_var*0.05)
        threshold = 0.04
    elif is_dark_bg:
        score     = (very_dark_ratio*0.08 + edge_loose*0.25 + component_score*0.25 +
                     thin_score*0.20 + network_density*0.12 + lap_var*0.10)
        threshold = 0.09
    else:
        score     = (edge_loose*0.15 + dark_ratio*0.12 + thin_score*0.28 +
                     lap_var*0.12 + component_score*0.18 + network_density*0.10 +
                     crack_area_ratio*0.05)
        threshold = 0.10

    if score <= threshold:
        return {
            "crack_detected": False, "crack_type": "none", "severity": "Safe",
            "severity_detail": "no crack detected",
            "confidence": round(min(0.92, 1.0 - score * 4.5), 4),
        }

    # ── Stage 5: Type + severity classification ────────────────────────────
    type_score = (
        mean_crack_width * 3.5 * 0.35 +
        crack_area_ratio * 4.0 * 0.35 +
        score            * 1.0 * 0.30
    )

    if   type_score < 0.10: crack_type = "hairline crack"
    elif type_score < 0.22: crack_type = "surface crack"
    elif type_score < 0.42: crack_type = "wide crack"
    else:                   crack_type = "structural visible crack"

    # Overrides for strong signals
    if large_comp_ratio > 0.08:                          crack_type = "structural visible crack"
    if is_drawing and black_ratio > 0.12:                crack_type = "wide crack"
    if is_drawing and black_ratio > 0.25:                crack_type = "structural visible crack"
    if not is_drawing and dark_ratio > 0.35 and score > 0.35: crack_type = "structural visible crack"

    severity, severity_detail = SEVERITY_MAP[crack_type]
    confidence = round(min(0.62 + type_score * 0.85, 0.98), 4)

    return {
        "crack_detected": True, "crack_type": crack_type,
        "severity": severity, "severity_detail": severity_detail,
        "confidence": confidence,
    }

def keras_predict_batch(img_arrays):
    batch = np.stack(img_arrays, axis=0)
    preds = model.predict(batch, verbose=0)
    results = []
    for pred in preds:
        if pred.shape[-1] == 2:
            c = float(pred[1])
            if c <= 0.5:
                results.append({"crack_detected":False,"crack_type":"none","severity":"Safe","severity_detail":"no crack","confidence":round(float(pred[0]),4)})
            else:
                sev,det = SEVERITY_MAP["surface crack"]
                results.append({"crack_detected":True,"crack_type":"surface crack","severity":sev,"severity_detail":det,"confidence":round(c,4)})
        else:
            ncp = float(pred[0]); cp = pred[1:]; bi = int(np.argmax(cp)); bp = float(cp[bi])
            if bp <= ncp:
                results.append({"crack_detected":False,"crack_type":"none","severity":"Safe","severity_detail":"no crack","confidence":round(ncp,4)})
            else:
                ct = CRACK_TYPES[bi]; sev,det = SEVERITY_MAP[ct]
                results.append({"crack_detected":True,"crack_type":ct,"severity":sev,"severity_detail":det,"confidence":round(bp,4)})
    return results

def process_single(file_bytes: bytes, filename: str) -> dict:
    try:
        arr    = preprocess_image(Image.open(io.BytesIO(file_bytes)))
        result = heuristic_predict(arr)
        result["image_name"] = filename
        return result
    except Exception as e:
        return {"image_name":filename,"error":str(e),"crack_detected":False,
                "crack_type":"unknown","severity":"Unknown","severity_detail":"error","confidence":0.0}

executor = ThreadPoolExecutor(max_workers=8)

@app.on_event("startup")
async def startup_event():
    load_model_at_startup()

@app.get("/health")
async def health():
    return {"status":"ok","model_loaded":model is not None or not use_keras,
            "engine":"keras" if use_keras else "heuristic v2"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 images.")

    start     = time.time()
    file_data = await asyncio.gather(*[
        asyncio.coroutine(lambda f: (f.read(), f.filename))(fl) for fl in files
    ]) if False else [(await fl.read(), fl.filename) for fl in files]

    if use_keras:
        results = []; items = []; names = []
        for fb, fn in file_data:
            try:
                items.append(preprocess_image(Image.open(io.BytesIO(fb)))); names.append(fn)
            except Exception as e:
                results.append({"image_name":fn,"error":str(e),"crack_detected":False,
                                 "crack_type":"unknown","severity":"Unknown","severity_detail":"error","confidence":0.0})
        loop = asyncio.get_event_loop()
        for i in range(0, len(items), BATCH_SIZE):
            br = await loop.run_in_executor(executor, keras_predict_batch, items[i:i+BATCH_SIZE])
            for res,name in zip(br, names[i:i+BATCH_SIZE]):
                res["image_name"] = name; results.append(res)
    else:
        loop    = asyncio.get_event_loop()
        results = list(await asyncio.gather(*[
            loop.run_in_executor(executor, process_single, fb, fn) for fb,fn in file_data
        ]))

    elapsed = round(time.time()-start, 3)
    logger.info(f"Processed {len(results)} images in {elapsed}s")
    return JSONResponse({"total":len(results),"processing_time_seconds":elapsed,"results":results})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
