import os
import sys
import json
import ollama
from forensic_visualizer import generate_clean_forensics
from rule_base_engine import evaluate_image_rules
from forensic_rag import ForensicRAGStore

def run_full_forensic_pipeline(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    print("\n" + "=" * 65)
    print(f" STARTING HYBRID SPATIAL FORENSIC PIPELINE: {image_path}")
    print("=" * 65)

    # 1. Layer 1: Generate Clean Forensic Maps
    print("[Layer 1] Generating ELA, Noise Heatmap, and Edge Gradient maps...")
    generate_clean_forensics(image_path, output_dir="forensic_output")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ela_map_path = os.path.join("forensic_output", f"{base_name}_1_ELA.jpg")

    # 2. Layer 2: Deterministic Rule Engine & Signal Telemetry
    print("[Layer 2] Running Rule-Base Engine & Spectral/ELA calibration...")
    rule_report = evaluate_image_rules(image_path)
    metrics = rule_report["metrics"]
    triggered_rules = rule_report["triggered_rules"]
    risk_score = rule_report["ai_risk_score"]
    rule_hint = rule_report["rule_engine_verdict"]

    print(f"    -> Rule Engine Hint : {rule_hint} (AI Risk Score: {risk_score}/100)")
    for r in triggered_rules:
        print(f"    -> Rule Triggered   : {r}")

    # 3. Layer 3: Forensic Vector RAG Retrieval
    print("[Layer 3] Retrieving matching forensic profiles from ChromaDB RAG...")
    rag_store = ForensicRAGStore(db_path="./forensic_rag_db")
    retrieved_docs = rag_store.retrieve_context(triggered_rules, n_results=2)
    rag_context_str = "\n".join([f"- {doc}" for doc in retrieved_docs])

    # 4. Layer 4: Grounded Multimodal VLM Inference (Ollama)
    print("[Layer 4] Querying Ollama 'spatial-detector' with Image + ELA + RAG...")
    prompt = f"""
Analyze Image 1 (Original Image) and Image 2 (ELA Compression Map) using the telemetry and forensic knowledge below:

=== DETERMINISTIC RULE ENGINE TELEMETRY ===
- Rule Engine Preliminary Hint: {rule_hint}
- Calculated AI Risk Score: {risk_score} / 100
- Edge-Masked Flat Sensor Grain: {metrics['flat_sensor_grain']}
- Global ELA Mean Error: {metrics['ela_mean_error']}
- ELA Spike Ratio: {metrics['ela_spike_ratio']}
- 2D FFT High-Freq Ratio: {metrics['fft_high_freq_ratio']}
- Hardware EXIF: {metrics['camera_exif']}
- Triggered Rules:
{chr(10).join(['  * ' + r for r in triggered_rules])}

=== RETRIEVED FORENSIC RAG KNOWLEDGE ===
{rag_context_str}

=== TASK ===
Cross-verify the visual spatial consistency of Image 1 (shadows, light direction, reflections, skin/fabric textures, background geometry) and Image 2 (ELA uniformity) against the telemetry and RAG guidelines above. Output strictly valid JSON.
"""

    try:
        response = ollama.generate(
            model="spatial-detector",
            prompt=prompt,
            images=[image_path, ela_map_path],
            format="json",
            options={"temperature": 0.1,
                      "top_p": 0.5,
                      "num_ctx": 16384
                      }
        )

        final_result = json.loads(response["response"])

        print("\n" + "=" * 65)
        print("              FINAL HYBRID FORENSIC VERDICT")
        print("=" * 65)
        print(f"IMAGE ANALYZED    : {image_path}")
        print(f"FINAL VERDICT     : {final_result.get('verdict')}")
        print(f"CONFIDENCE SCORE  : {final_result.get('confidence_score')}%")
        print(f"PRIMARY EVIDENCE  : {final_result.get('primary_evidence')}")
        print("-" * 65)
        print("SPATIAL CHECKS:")
        for k, v in final_result.get("spatial_checks", {}).items():
            print(f"  • {k.replace('_', ' ').title():<32}: {v}")
        print("-" * 65)
        print(f"REASONING:\n{final_result.get('reasoning')}")
        print("=" * 65 + "\n")
        return final_result

    except Exception as e:
        print(f"[!] VLM Inference Error: {e}")

if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else "dp.jpg"
    run_full_forensic_pipeline(target_img)