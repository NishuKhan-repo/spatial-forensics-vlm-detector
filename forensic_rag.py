import chromadb
from chromadb.utils import embedding_functions

class ForensicRAGStore:
    def __init__(self, db_path="./forensic_rag_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="calibrated_spatial_forensics",
            embedding_function=self.emb_fn
        )

    def build_knowledge_base(self):
        """Calibrated forensic rules aur generator signatures ko vector DB mein store karta hai."""
        documents = [
            {
                "id": "rule_social_media_compression",
                "text": (
                    "SOCIAL_MEDIA_REAL_PHOTO PROFILE: When an authentic smartphone or DSLR photo is shared over "
                    "WhatsApp, Instagram, or web platforms, 8x8 DCT JPEG re-compression strips EXIF metadata and "
                    "smooths flat regions (lowering flat_sensor_grain below 1.15). However, Global ELA Mean stays "
                    "very low (< 0.50) while ELA Spike Ratio remains high (> 6.5) because compression error concentrates "
                    "strictly along real structural edges. Do NOT misclassify this as AI smoothing; verify physical shadow coherence."
                ),
                "metadata": {"category": "authentic_compressed", "bias": "REAL"}
            },
            {
                "id": "rule_raw_camera_sensor",
                "text": (
                    "RAW_CAMERA_HARDWARE PROFILE: Uncompressed or lightly compressed authentic camera captures retain "
                    "EXIF hardware tags and natural Photo Response Non-Uniformity (PRNU) sensor grain (1.30 to 2.80) "
                    "even in edge-masked flat areas like skin or walls. Spatial lighting vectors and cast shadows follow strict 3D geometry."
                ),
                "metadata": {"category": "authentic_raw", "bias": "REAL"}
            },
            {
                "id": "rule_diffusion_synthetic_ela",
                "text": (
                    "HIGH_SYNTHETIC_ELA & DIFFUSION TEXTURE DENSITY: Modern diffusion generators (Imagen, Flux, Midjourney, SDXL) "
                    "synthesize high-frequency micro-textures across the entire pixel grid rather than capturing light through a lens. "
                    "When re-compressed at 90% JPEG, this produces elevated Global ELA Mean (> 0.75) with a low ELA Spike Ratio (< 6.0). "
                    "Inspect the image for waxy facial skin contrasted against hyper-sharp clothing/background details, inconsistent window/studio light sources, or corner AI watermarks."
                ),
                "metadata": {"category": "ai_diffusion", "bias": "FAKE"}
            },
            {
                "id": "rule_deepfake_splicing",
                "text": (
                    "LOCALIZED_SPLICING_AND_INPAINTING: Face-swaps and regional AI edits create abrupt spatial discontinuities. "
                    "In the ELA map, the edited face or object appears as a distinct bright or dark anomaly island compared to the surrounding frame. "
                    "In the Edge Gradient Map, look for double halos or melted boundaries along the jawline, hairline, or fingers."
                ),
                "metadata": {"category": "manipulation", "bias": "FAKE"}
            },
            {
                "id": "rule_macro_spatial_physics",
                "text": (
                    "MACRO SPATIAL CONSISTENCY CHECKLIST: Regardless of pixel metrics, verify three physical laws: "
                    "(1) Shadow Direction: All shadows must project away from the dominant light source. "
                    "(2) Specular Catchlights: Reflections in left and right eye pupils must match in angle and shape. "
                    "(3) Perspective Lines: Architectural lines, furniture edges, and background patterns must converge naturally without warping."
                ),
                "metadata": {"category": "spatial_physics", "bias": "NEUTRAL"}
            }
        ]

        self.collection.upsert(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            metadatas=[d["metadata"] for d in documents]
        )
        print(f"[+] Forensic RAG Database Ready! Total indexed profiles: {self.collection.count()}")

    def retrieve_context(self, triggered_rules, n_results=2):
        """Rule Engine ke output ke basis par sabse relevant forensic knowledge fetch karta hai."""
        query_text = " ".join(triggered_rules)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results["documents"][0]

if __name__ == "__main__":
    rag = ForensicRAGStore()
    rag.build_knowledge_base()

    # Quick Test: Social Media rule ke liye kya retrieve hota hai?
    print("\n--- TEST RETRIEVAL FOR SOCIAL MEDIA COMPRESSION ---")
    sample_rules = ["SOCIAL_MEDIA_REAL_PHOTO: Low uniform ELA (0.358) + natural edge compression ratio (8.38)"]
    for idx, doc in enumerate(rag.retrieve_context(sample_rules), 1):
        print(f"{idx}. {doc}\n")