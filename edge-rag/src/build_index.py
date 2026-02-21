import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus.jsonl"
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
    np.save(os.path.join(OUT_DIR, "emb_f32.npy"), emb)           # keep float index too (baseline)
    np.save(os.path.join(OUT_DIR, "emb_bin_u8.npy"), packed)     # binary signbits index
    np.save(os.path.join(OUT_DIR, "meta.npy"), np.array([emb.shape[1]], dtype=np.int32))

    print("Saved index:")
    print(" - index/doc_ids.npy")
    print(" - index/emb_f32.npy")
    print(" - index/emb_bin_u8.npy")

if __name__ == "__main__":
    main()
