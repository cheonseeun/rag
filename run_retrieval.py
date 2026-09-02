"""
1단계: 각 retrieval 모델로 top-k 검색 + IR 지표 계산 (API 비용 0).

출력: runs/{model}/retrieved.json, runs/{model}/ir_metrics.json
"""
import argparse
import json
import time

import numpy as np

from config import DATA_DIR, DEFAULT_MODELS, EVAL_KS, RUN_DIR, TOP_K
from encoders import Encoder


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def ir_metrics(ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks):
    """AutoRAG는 query당 정답 1개이므로 Recall@k == HitRate@k."""
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([
            len(set(ranked[q][:k]) & set(qrels[q])) / len(qrels[q]) for q in qrels
        ]))
    rr, ndcg = [], []
    for q, gold in qrels.items():
        r = ranked[q]
        rr.append(next((1 / (i + 1) for i, d in enumerate(r) if d in gold), 0.0))
        dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(r[:10]) if d in gold)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg.append(dcg / idcg)
    out["mrr"] = float(np.mean(rr))
    out["ndcg@10"] = float(np.mean(ndcg))
    return out


def run_one(name: str, corpus, queries, qrels, top_k: int):
    out_dir = RUN_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = [c["_id"] for c in corpus]
    # title이 있으면 붙여줍니다 (AutoRAG는 대부분 비어 있음)
    doc_texts = [(c["title"] + "\n" + c["text"]).strip() if c.get("title") else c["text"]
                 for c in corpus]
    q_ids = [q["_id"] for q in queries]
    q_texts = [q["text"] for q in queries]

    enc = Encoder(name)
    t0 = time.time()
    D = enc.encode_corpus(doc_texts)
    Q = enc.encode_queries(q_texts)
    elapsed = time.time() - t0
    enc.free()

    scores = Q @ D.T                                   # (nq, nd), 이미 정규화됨
    order = np.argsort(-scores, axis=1)[:, :max(top_k, max(EVAL_KS))]

    retrieved, ranked = {}, {}
    for i, qid in enumerate(q_ids):
        idxs = order[i]
        ranked[qid] = [doc_ids[j] for j in idxs]
        retrieved[qid] = {
            "query": q_texts[i],
            "gold_ids": list(qrels.get(qid, {})),
            "docs": [{"doc_id": doc_ids[j],
                      "text": corpus[j]["text"],
                      "score": float(scores[i, j]),
                      "is_gold": doc_ids[j] in qrels.get(qid, {})}
                     for j in idxs[:top_k]],
        }

    m = ir_metrics(ranked, qrels, EVAL_KS)
    m["encode_sec"] = round(elapsed, 1)
    m["embedding_dim"] = int(D.shape[1])

    json.dump(retrieved, open(out_dir / "retrieved.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(m, open(out_dir / "ir_metrics.json", "w", encoding="utf-8"), indent=2)
    np.save(out_dir / "doc_emb.npy", D)   # 하드 네거티브 마이닝에 재사용
    np.save(out_dir / "query_emb.npy", Q)
    print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()
                                   if isinstance(v, float)))
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--top-k", type=int, default=TOP_K)
    a = p.parse_args()

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    qrels = json.load(open(DATA_DIR / "qrels.json", encoding="utf-8"))

    summary = {}
    for name in a.models:
        try:
            summary[name] = run_one(name, corpus, queries, qrels, a.top_k)
        except Exception as e:                      # 한 모델 실패로 전체가 죽지 않게
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    json.dump(summary, open(RUN_DIR / "ir_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)