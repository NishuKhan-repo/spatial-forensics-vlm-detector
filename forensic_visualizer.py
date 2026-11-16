import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def generate_clean_forensics(image_path, output_dir="forensic_output"):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' nahi mili!")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Image load karna
    orig_pil = Image.open(image_path).convert("RGB")
    orig_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ---------------------------------------------------------
    # 1. ELA COMPRESSION MAP (90% Re-save Difference)
    # ---------------------------------------------------------
    temp_jpg = os.path.join(output_dir, "temp_90.jpg")
    orig_pil.save(temp_jpg, "JPEG", quality=90)
    recompressed = Image.open(temp_jpg)

    ela_diff = ImageChops.difference(orig_pil, recompressed)
    extrema = ela_diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_enhanced = ImageEnhance.Brightness(ela_diff).enhance(255.0 / max_diff)
    
    ela_path = os.path.join(output_dir, f"{base_name}_1_ELA.jpg")
    ela_enhanced.save(ela_path)
    os.remove(temp_jpg)

    # ---------------------------------------------------------
    # 2. LOCAL NOISE HEATMAP (Camera Sensor Grain)
    # ---------------------------------------------------------
    # Smooth colors hata kar sirf micro-grain nikalna
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred

    # Local variance (5x5 window)
    mean = cv2.blur(residual, (5, 5))
    sq_mean = cv2.blur(residual ** 2, (5, 5))
    local_std = np.sqrt(np.maximum(sq_mean - (mean ** 2), 0))

    # Heatmap banana (Blue = Smooth/No Grain, Yellow/Red = High Grain)
    norm_noise = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(norm_noise, cv2.COLORMAP_JET)

    heatmap_path = os.path.join(output_dir, f"{base_name}_2_NOISE_HEATMAP.jpg")
    cv2.imwrite(heatmap_path, heatmap_bgr)

    # ---------------------------------------------------------
    # 3. EDGE GRADIENT MAP (Structural Boundaries)
    # ---------------------------------------------------------
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobelx, sobely)
    edge_map = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edge_path = os.path.join(output_dir, f"{base_name}_3_EDGE_MAP.jpg")
    cv2.imwrite(edge_path, edge_map)

    # ---------------------------------------------------------
    # SMART METRICS (Bokeh-Safe Subject Center Analysis)
    # ---------------------------------------------------------
    h, w = local_std.shape
    # Sirf beech ka 40% hissa (jahan main subject hota hai) check karna
    center_noise = local_std[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    
    subject_grain_score = float(np.mean(center_noise))
    ela_mean_error = float(np.mean(np.array(ela_diff)))

    print("\n" + "="*55)
    print(f" FORENSIC EXTRACTION COMPLETE: {image_path}")
    print("="*55)
    print(f"1. ELA Map Saved      : {ela_path}")
    print(f"2. Noise Heatmap Saved: {heatmap_path}")
    print(f"3. Edge Map Saved     : {edge_path}")
    print("-" * 55)
    print(f"-> Subject Center Grain Score : {subject_grain_score:.3f}  (Real usually > 1.8)")
    print(f"-> Global ELA Mean Error      : {ela_mean_error:.3f}")
    print("="*55 + "\n")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    generate_clean_forensics(img)