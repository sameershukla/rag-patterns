"""
indexer.py — Phase A: Indexing
Run once to embed clinic documents and save the FAISS index to disk.
Re-run only when documents change.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Documents (Clinic Knowledge Base) ────────────────────────────────────────
#
# These are the actual clinic policies and medical guidelines.
# The LLM has no knowledge of these — they are private, specific,
# and would be hallucinated if asked without context.

documents = [
    {
        "id": "doc_1",
        "text": (
            "Paracetamol Dosage Policy: Adult patients may take 500mg to 1000mg "
            "of paracetamol every 4 to 6 hours. The maximum daily dose must not "
            "exceed 4000mg (4g) in 24 hours. Patients with liver disease must not "
            "exceed 2000mg per day. Do not combine with other paracetamol-containing "
            "medications. Overdose can cause severe liver damage."
        ),
    },
    {
        "id": "doc_2",
        "text": (
            "Post-Surgery Wound Care: Keep the wound dry for 48 hours after surgery. "
            "Change the dressing every 2 days using sterile gauze. "
            "Do not submerge the wound in water for 14 days. "
            "Signs of infection include redness, swelling, warmth, or discharge. "
            "If you develop a fever above 38.5 degrees Celsius, contact the clinic immediately."
        ),
    },
    {
        "id": "doc_3",
        "text": (
            "Appointment Cancellation Policy: Appointments must be cancelled at least "
            "24 hours in advance to avoid a cancellation fee of $50. "
            "Cancellations made within 2 hours of the appointment will be charged "
            "the full consultation fee. To cancel, call 1-800-CLINIC or use the "
            "patient portal. Emergency cancellations due to hospitalisation are exempt."
        ),
    },
    {
        "id": "doc_4",
        "text": (
            "Antibiotic Usage Guidelines: Complete the full course of antibiotics as "
            "prescribed even if symptoms improve. Never share antibiotics with others. "
            "Amoxicillin 500mg is prescribed three times daily for 7 days for most "
            "bacterial infections. If you experience rash, difficulty breathing, or "
            "swelling after taking antibiotics, stop immediately and seek emergency care "
            "as these are signs of an allergic reaction."
        ),
    },
    {
        "id": "doc_5",
        "text": (
            "Fasting Before Blood Tests: Patients must fast for 8 to 12 hours before "
            "cholesterol and glucose blood tests. Water is permitted during the fasting "
            "period. Diabetic patients on insulin must consult their doctor before "
            "fasting. Blood tests are available Monday to Friday, 7am to 11am only. "
            "Walk-in blood tests are accepted before 9am. After 9am, an appointment "
            "is required."
        ),
    },
    {
        "id": "doc_6",
        "text": (
            "Children's Fever Management: For children aged 3 months to 12 years, "
            "paracetamol syrup is dosed at 15mg per kilogram of body weight every "
            "4 to 6 hours. Do not give aspirin to children under 16 years. "
            "A fever above 38 degrees Celsius in a baby under 3 months requires "
            "immediate emergency care. Tepid sponging can help reduce discomfort "
            "but does not treat the underlying cause."
        ),
    },
]

# ── Indexing Pipeline ─────────────────────────────────────────────────────────

def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(documents)} clinic documents...")
    texts      = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "index.faiss")
    print("Saved → index.faiss")

    with open("documents.json", "w") as f:
        json.dump(documents, f, indent=2)
    print("Saved → documents.json")

    print(f"\nIndexing complete. {index.ntotal} documents indexed and ready.")


if __name__ == "__main__":
    build_index()
