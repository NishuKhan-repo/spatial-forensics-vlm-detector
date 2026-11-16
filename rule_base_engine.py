import os
import json
import cv2
import numpy as np
from PIL import Image, ImageChops, ExifTags

def compute_fft_high_freq_ratio(gray_img):
    """2D Fast Fourier Transform (FFT) se synthetic high-frequency artifacts check karta hai."""
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    
    # Low frequency center mask out karke outer high-frequency energy naapna
    mask = np.ones((h, w), dtype=bool)
    y, x = np.ogrid[:h, :w]
    mask[(y - cy)**2 + (x - cx)**2 <= r**2] = False
    
    high_freq_mean = float(np.mean(magnitude_spectrum[mask]))
    total_mean = float(np.mean(magnitude_spectrum))
    return round(high_freq_mean / (total_mean + 1e-5), 4)

def evaluate_image_rules(image_path):
    orig_pil = Image.open(image_path).convert("RGB")
    orig_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. Edge-Masked Flat Noise
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    flat_mask = edges < 25.0

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = np.abs(gray - blurred)
    flat_sensor_grain = float(np.mean(residual[flat_mask])) if np.sum(flat_mask) > 100 else float(np.mean(residual))

    # 2. ELA Compression Signature
    temp_ela = "temp_rule_ela.jpg"
    orig_pil.save(temp_ela, "JPEG", quality=90)
    recompressed = Image.open(temp_ela)
    ela_arr = np.array(ImageChops.difference(orig_pil, recompressed), dtype=np.float32)
    os.remove(temp_ela)

    ela_mean = float(np.mean(ela_arr))
    ela_99 = float(np.percentile(ela_arr, 99))
    ela_spike_ratio = round(ela_99 / (ela_mean + 1e-5), 2)

    # 3. 2D FFT Spectral Ratio
    fft_ratio = compute_fft_high_freq_ratio(gray)

    # 4. EXIF Camera Hardware Check
    has_exif = False
    camera_model = "None"
    try:
        exif = orig_pil._getexif()
        if exif:
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            if tags.get("Make") or tags.get("Model"):
                has_exif = True
                camera_model = f"{tags.get('Make', '')} {tags.get('Model', '')}".strip()
    except Exception:
        pass

    # ---------------------------------------------------------
    # CALIBRATED DETERMINISTIC RULE ENGINE
    # ---------------------------------------------------------
    ai_risk_score = 30  # Neutral baseline
    triggered_rules = []

    # RULE 1: Social Media / Web Compressed Real Photo Guard
    # Jab ELA error bahut low (< 0.50) aur spike ratio high (> 6.5) ho, toh low flat grain WhatsApp/web JPEG compression ki wajah se hota hai, AI se nahi!
    if ela_mean < 0.50 and ela_spike_ratio > 6.5 and flat_sensor_grain < 1.15:
        ai_risk_score -= 20
        triggered_rules.append(
            f"SOCIAL_MEDIA_REAL_PHOTO: Low uniform ELA ({ela_mean:.3f}) + natural edge compression ratio ({ela_spike_ratio}) indicates web/WhatsApp compressed real capture, not AI."
        )
    else:
        # Standard ELA & Flat Grain evaluation
        if ela_mean > 0.75:
            ai_risk_score += 40
            triggered_rules.append(f"HIGH_SYNTHETIC_ELA: ELA mean ({ela_mean:.3f}) > 0.75 indicates non-camera synthetic rendering.")
        elif ela_mean < 0.50:
            ai_risk_score -= 15
            triggered_rules.append(f"NATURAL_CAMERA_ELA: Low uniform ELA ({ela_mean:.3f}) matches authentic JPEG pipeline.")

        if flat_sensor_grain < 1.25:
            ai_risk_score += 25
            triggered_rules.append(f"WAXY_AI_SMOOTHING: Flat regions lack physical sensor grain ({flat_sensor_grain:.3f} < 1.25).")
        elif 1.30 <= flat_sensor_grain <= 2.80:
            ai_risk_score -= 15
            triggered_rules.append(f"NATURAL_SENSOR_PRNU: Flat regions contain healthy camera sensor grain ({flat_sensor_grain:.3f}).")

    # RULE 2: Synthetic Uniform ELA Density (AI generators spread error broadly across synthetic textures)
    if ela_mean > 0.80 and ela_spike_ratio < 6.0:
        ai_risk_score += 20
        triggered_rules.append(f"SYNTHETIC_TEXTURE_DENSITY: High ELA ({ela_mean:.3f}) with low spike ratio ({ela_spike_ratio}) is a strong diffusion signature.")

    # RULE 3: Hardware EXIF
    if has_exif:
        ai_risk_score -= 25
        triggered_rules.append(f"VERIFIED_HARDWARE_EXIF: Camera metadata found ({camera_model}).")
    else:
        ai_risk_score += 5
        triggered_rules.append("NO_HARDWARE_EXIF: Metadata stripped (neutral penalty).")

    ai_risk_score = max(0, min(100, ai_risk_score))
    verdict_hint = "LIKELY_FAKE" if ai_risk_score >= 55 else "LIKELY_REAL"

    return {
        "image": image_path,
        "metrics": {
            "flat_sensor_grain": round(flat_sensor_grain, 3),
            "ela_mean_error": round(ela_mean, 3),
            "ela_spike_ratio": ela_spike_ratio,
            "fft_high_freq_ratio": fft_ratio,
            "camera_exif": camera_model
        },
        "rule_engine_verdict": verdict_hint,
        "ai_risk_score": ai_risk_score,
        "triggered_rules": triggered_rules
    }

if __name__ == "__main__":
    for img_file in ["test2.jpg", "test1.jpg"]:
        res = evaluate_image_rules(img_file)
        print(json.dumps(res, indent=2))