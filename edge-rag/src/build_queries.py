import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERIES_PATH = "data/queries.jsonl"

OUT_Q_F32 = "index/query_emb_f32.npy"
OUT_Q_BIN = "index/query_bin_u8.npy"

def load_queries(path: str):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qs.append(obj["query"])
    return qs

def pack_sign_bits(x: np.ndarray) -> np.ndarray:
    # x: (M,D) float32
    bits = (x >= 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1)  # (M, D/8)
    return packed

def main():
    qs = load_queries(QUERIES_PATH)
    m = SentenceTransformer(MODEL_NAME)
    emb = m.encode(qs, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    qbin = pack_sign_bits(emb).astype(np.uint8)

    np.save(OUT_Q_F32, emb)
    np.save(OUT_Q_BIN, qbin)

    print(f"Wrote {OUT_Q_F32} shape={emb.shape}")
    print(f"Wrote {OUT_Q_BIN} shape={qbin.shape}")

if __name__ == "__main__":
    main()
