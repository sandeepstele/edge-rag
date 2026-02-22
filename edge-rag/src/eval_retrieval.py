import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from math import log2

QUERIES_PATH = "data/queries.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

K_LIST = [1, 3, 5, 10, 20, 50]  # stress test: see where binary degrades at higher k

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def cosine_topk(q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    # q: (D,), X: (N, D) normalized -> cosine = dot
    scores = X @ q
    idx = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx

def hamming_topk(q_bin: np.ndarray, X_bin: np.ndarray, k: int) -> np.ndarray:
    # q_bin: (B,) uint8 packed bits, X_bin: (N,B) uint8
    # Hamming distance via XOR + popcount
    xor = np.bitwise_xor(X_bin, q_bin)
    # popcount each row
    dist = np.unpackbits(xor, axis=1).sum(axis=1)  # simple, not fastest; OK for eval
    idx = np.argpartition(dist, kth=min(k, len(dist)-1))[:k]
    idx = idx[np.argsort(dist[idx])]
    return idx

def dcg(rels):
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1) / log2(i + 1)
    return s

def ndcg_at_k(ranked_ids, relevant_set, k):
    rels = [1 if d in relevant_set else 0 for d in ranked_ids[:k]]
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(rels) / denom

def mrr(ranked_ids, relevant_set):
    for i, d in enumerate(ranked_ids, start=1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0

def recall_at_k(ranked_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    hit = sum(1 for d in ranked_ids[:k] if d in relevant_set)
    return hit / len(relevant_set)

def precision_at_k(ranked_ids, relevant_set, k):
    if k == 0:
        return 0.0
    hit = sum(1 for d in ranked_ids[:k] if d in relevant_set)
    return hit / k

def to_query_bin(q: np.ndarray) -> np.ndarray:
    bits = (q >= 0).astype(np.uint8)
    pad = (-bits.shape[0]) % 8
    if pad:
        bits = np.pad(bits, (0, pad), mode="constant", constant_values=0)
    return np.packbits(bits)

def main():
    doc_ids = np.load("index/doc_ids.npy", allow_pickle=True)
    X = np.load("index/emb_f32.npy").astype(np.float32)
    X_bin = np.load("index/emb_bin_u8.npy").astype(np.uint8)

    model = SentenceTransformer(MODEL_NAME)

    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj)

    print(f"Loaded {len(queries)} queries")

    # Encode queries once
    q_texts = [q["query"] for q in queries]
    Q = model.encode(q_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    Q = l2_normalize(np.asarray(Q, dtype=np.float32))

    results = {
        "float": {f"recall@{k}": [] for k in K_LIST} | {f"prec@{k}": [] for k in K_LIST} | {"mrr": [], "ndcg@10": []},
        "bin":   {f"recall@{k}": [] for k in K_LIST} | {f"prec@{k}": [] for k in K_LIST} | {"mrr": [], "ndcg@10": []},
    }

    for i, qobj in enumerate(tqdm(queries)):
        rel = set(qobj["relevant_doc_ids"])
        qv = Q[i]

        # float retrieval
        idx_f = cosine_topk(qv, X, k=max(K_LIST))
        ranked_f = [str(doc_ids[j]) for j in idx_f]
        results["float"]["mrr"].append(mrr(ranked_f, rel))
        results["float"]["ndcg@10"].append(ndcg_at_k(ranked_f, rel, 10))
        for k in K_LIST:
            results["float"][f"recall@{k}"].append(recall_at_k(ranked_f, rel, k))
            results["float"][f"prec@{k}"].append(precision_at_k(ranked_f, rel, k))

        # binary retrieval
        qbin = to_query_bin(qv)
        idx_b = hamming_topk(qbin, X_bin, k=max(K_LIST))
        ranked_b = [str(doc_ids[j]) for j in idx_b]
        results["bin"]["mrr"].append(mrr(ranked_b, rel))
        results["bin"]["ndcg@10"].append(ndcg_at_k(ranked_b, rel, 10))
        for k in K_LIST:
            results["bin"][f"recall@{k}"].append(recall_at_k(ranked_b, rel, k))
            results["bin"][f"prec@{k}"].append(precision_at_k(ranked_b, rel, k))

    def summarize(name):
        print(f"\n== {name} ==")
        for k in K_LIST:
            print(f"Recall@{k}:   {np.mean(results[name][f'recall@{k}']):.4f}")
            print(f"Prec@{k}:     {np.mean(results[name][f'prec@{k}']):.4f}")
        print(f"MRR:        {np.mean(results[name]['mrr']):.4f}")
        print(f"nDCG@10:    {np.mean(results[name]['ndcg@10']):.4f}")

    summarize("float")
    summarize("bin")

if __name__ == "__main__":
    main()
