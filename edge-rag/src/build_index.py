import platform
import sys

# Laptop-only: torch/sentence-transformers crashes on Raspberry Pi 3
machine = platform.machine().lower()
if machine in ("armv7l", "aarch64", "arm64"):
    print(
        "\n[STOP] build_index.py is laptop-only.\n"
        "Reason: it requires torch/sentence-transformers, which crashes on Raspberry Pi 3.\n"
        "Run build_index.py on your laptop, then copy the generated index/ folder to the Pi.\n"
    )
    sys.exit(2)

import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus.jsonl"
QUERIES_PATH = "data/queries.jsonl"
OUT_DIR = "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def to_binary_signbits(x: np.ndarray) -> np.ndarray:
    # x: float32 normalized embeddings (N, D)
    # returns packed bits: (N, ceil(D/8)) uint8
    bits = (x >= 0).astype(np.uint8)
    # pad to multiple of 8
    pad = (-bits.shape[1]) % 8
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    packed = np.packbits(bits, axis=1)
    return packed

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append((obj["doc_id"], obj["text"]))

    print(f"Loaded {len(docs)} docs")

    model = SentenceTransformer(MODEL_NAME)
    texts = [t for _, t in docs]

    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)
    emb = l2_normalize(emb)

    packed = to_binary_signbits(emb)

    doc_ids = np.array([d for d, _ in docs])
    np.save(os.path.join(OUT_DIR, "doc_ids.npy"), doc_ids)
    np.save(os.path.join(OUT_DIR, "emb_f32.npy"), emb)
    np.save(os.path.join(OUT_DIR, "emb_bin_u8.npy"), packed)
    np.save(os.path.join(OUT_DIR, "meta.npy"), np.array([emb.shape[1]], dtype=np.int32))

    # ---- Precompute query embeddings and qrels (Pi never needs torch) ----
    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    if queries:
        q_texts = [q["query"] for q in queries]
        q_emb = model.encode(q_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
        q_emb = l2_normalize(np.asarray(q_emb, dtype=np.float32))
        np.save(os.path.join(OUT_DIR, "queries_emb_f32.npy"), q_emb)
        np.save(os.path.join(OUT_DIR, "queries.npy"), np.array(q_texts, dtype=object))
        np.save(os.path.join(OUT_DIR, "query_ids.npy"), np.array([q["qid"] for q in queries], dtype=object))
        qrels = {q["qid"]: q["relevant_doc_ids"] for q in queries}
        with open(os.path.join(OUT_DIR, "qrels.json"), "w", encoding="utf-8") as f:
            json.dump(qrels, f, indent=0)
        print(f"Precomputed embeddings for {len(queries)} queries -> index/queries_emb_f32.npy, index/qrels.json")

    print("Saved index (copy this folder to Pi):")
    print(" - index/doc_ids.npy")
    print(" - index/emb_f32.npy")
    print(" - index/emb_bin_u8.npy")
    print(" - index/queries_emb_f32.npy  (Pi: no torch)")
    print(" - index/queries.npy (optional, raw query strings)")
    print(" - index/query_ids.npy (qid per row, matches queries_emb_f32.npy)")
    print(" - index/qrels.json (ground truth)")

if __name__ == "__main__":
    main()
