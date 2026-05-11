"""
leaf_validator.py  —  Strict multi-signal leaf validator
=========================================================
Rejects: selfies, sky, screenshots, random photos, cartoons.
Accepts: genuine plant leaf photographs only.

Signals (ALL must pass):
  1. Skin-tone rejection   — selfies / portrait / hand photos
  2. Sky / blue rejection  — outdoor landscape shots
  3. Saturation gate       — screenshots / white graphics
  4. Leaf hue coverage     — must have enough green/yellow-green pixels
  5. Edge-density gate     — crisp UI screenshots have too many hard edges
  6. Texture / variance    — real leaves have organic mid-range variance
"""

import cv2
import numpy as np
from PIL import Image


# ─── Tunable thresholds ────────────────────────────────────────────────────────

SKIN_RATIO_MAX   = 0.18    # selfie rejected if > 18% skin pixels
SKY_RATIO_MAX    = 0.35    # landscape rejected if > 35% sky-blue pixels
MIN_MEAN_SAT     = 35      # out of 255 — rejects grey/white screenshots
LEAF_RATIO_MIN   = 0.12    # ≥ 12% of pixels must be leaf-coloured
EDGE_DENSITY_MAX = 0.25    # < 25% canny edge pixels
TEXTURE_VAR_MIN  = 180     # below → solid colour / plain gradient
TEXTURE_VAR_MAX  = 8500    # above → busy screenshot / collage

# HSV hue ranges (OpenCV 0–179)
LEAF_HUE_MIN, LEAF_HUE_MAX = 18, 88   # green → yellow-green → yellow
LEAF_SAT_MIN, LEAF_VAL_MIN = 40, 40

SKY_HUE_MIN,  SKY_HUE_MAX  = 90, 130  # cyan-blue
SKY_SAT_MIN,  SKY_VAL_MIN  = 40, 80

# Skin tone in YCrCb colour space (more robust than HSV for skin)
SKIN_Y_MIN,  SKIN_Y_MAX  =  80, 235
SKIN_CR_MIN, SKIN_CR_MAX = 133, 173
SKIN_CB_MIN, SKIN_CB_MAX =  77, 127


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def validate_leaf_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    is_valid=True  → proceed with disease model.
    is_valid=False → show rejection_reason to user.
    """
    bgr   = _pil_to_bgr(pil_image)
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    h, s, v      = cv2.split(hsv)
    y_ch, cr, cb = cv2.split(ycrcb)
    total        = float(h.size)

    # ── [1] Skin-tone check (selfie / face / hand) ────────────────────────────
    skin_mask = (
        (y_ch >= SKIN_Y_MIN)  & (y_ch <= SKIN_Y_MAX) &
        (cr   >= SKIN_CR_MIN) & (cr   <= SKIN_CR_MAX) &
        (cb   >= SKIN_CB_MIN) & (cb   <= SKIN_CB_MAX)
    )
    if skin_mask.sum() / total > SKIN_RATIO_MAX:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            "(Detected skin tones — this looks like a selfie or portrait photo.)"
        )

    # ── [2] Sky / blue landscape detection ───────────────────────────────────
    sky_mask = (
        (h >= SKY_HUE_MIN) & (h <= SKY_HUE_MAX) &
        (s >= SKY_SAT_MIN) & (v >= SKY_VAL_MIN)
    )
    if sky_mask.sum() / total > SKY_RATIO_MAX:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            "(Detected sky or blue landscape tones — not a leaf photo.)"
        )

    # ── [3] Saturation gate (screenshots / white-bg graphics) ────────────────
    if float(np.mean(s)) < MIN_MEAN_SAT:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            "(Image is too grey or washed-out — looks like a screenshot or PDF.)"
        )

    # ── [4] Leaf hue coverage ─────────────────────────────────────────────────
    leaf_mask  = (h >= LEAF_HUE_MIN) & (h <= LEAF_HUE_MAX) & (s >= LEAF_SAT_MIN) & (v >= LEAF_VAL_MIN)
    brown_mask = (h >= 4) & (h <= 17) & (s >= 50) & (s <= 210) & (v >= 40)  # dry/diseased brown
    leaf_ratio = (leaf_mask.sum() + brown_mask.sum()) / total

    if leaf_ratio < LEAF_RATIO_MIN:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            f"(Only {leaf_ratio*100:.1f}% leaf-coloured pixels detected — "
            f"minimum is {LEAF_RATIO_MIN*100:.0f}%.)"
        )

    # ── [5] Edge density (screenshot / UI detection) ──────────────────────────
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 60, 150)
    if float(np.count_nonzero(edges)) / total > EDGE_DENSITY_MAX:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            "(Too many sharp edges detected — looks like a screenshot or diagram.)"
        )

    # ── [6] Texture variance (solid colour / complex UI) ─────────────────────
    gray_f  = gray.astype(np.float32)
    mean_sq = cv2.boxFilter(gray_f ** 2, -1, (7, 7))
    sq_mean = cv2.boxFilter(gray_f,      -1, (7, 7)) ** 2
    mean_var = float(np.mean(np.maximum(mean_sq - sq_mean, 0)))

    if mean_var < TEXTURE_VAR_MIN:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            "(Image surface is too uniform — looks like a solid colour or gradient.)"
        )
    if mean_var > TEXTURE_VAR_MAX:
        return False, (
            "Invalid image. Please upload a clear plant leaf image. "
            "(Image is too visually complex — looks like a screenshot or collage.)"
        )

    return True, "Image validated as a plant leaf."


def get_validation_debug_info(pil_image: Image.Image) -> dict:
    """Raw signal values for sidebar debug panel."""
    bgr   = _pil_to_bgr(pil_image)
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    h, s, v      = cv2.split(hsv)
    y_ch, cr, cb = cv2.split(ycrcb)
    total = float(h.size)

    skin_mask  = (y_ch>=SKIN_Y_MIN)&(y_ch<=SKIN_Y_MAX)&(cr>=SKIN_CR_MIN)&(cr<=SKIN_CR_MAX)&(cb>=SKIN_CB_MIN)&(cb<=SKIN_CB_MAX)
    sky_mask   = (h>=SKY_HUE_MIN)&(h<=SKY_HUE_MAX)&(s>=SKY_SAT_MIN)&(v>=SKY_VAL_MIN)
    leaf_mask  = (h>=LEAF_HUE_MIN)&(h<=LEAF_HUE_MAX)&(s>=LEAF_SAT_MIN)&(v>=LEAF_VAL_MIN)
    brown_mask = (h>=4)&(h<=17)&(s>=50)&(s<=210)&(v>=40)

    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 60, 150)
    gray_f  = gray.astype(np.float32)
    mean_sq = cv2.boxFilter(gray_f**2, -1, (7,7))
    sq_mean = cv2.boxFilter(gray_f,    -1, (7,7))**2
    mean_var = float(np.mean(np.maximum(mean_sq - sq_mean, 0)))

    return {
        "1_skin_ratio":       round(skin_mask.sum()/total, 4),
        "2_sky_ratio":        round(sky_mask.sum()/total,  4),
        "3_mean_saturation":  round(float(np.mean(s)), 2),
        "4_leaf_pixel_ratio": round((leaf_mask.sum()+brown_mask.sum())/total, 4),
        "5_edge_density":     round(float(np.count_nonzero(edges))/total, 4),
        "6_texture_variance": round(mean_var, 2),
        "--- thresholds ---": {
            "skin_max":        SKIN_RATIO_MAX,
            "sky_max":         SKY_RATIO_MAX,
            "sat_min":         MIN_MEAN_SAT,
            "leaf_min":        LEAF_RATIO_MIN,
            "edge_max":        EDGE_DENSITY_MAX,
            "texture_min/max": f"{TEXTURE_VAR_MIN} / {TEXTURE_VAR_MAX}",
        },
    }
