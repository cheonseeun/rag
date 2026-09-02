r"""
2단계 파이프라인 평가: 1단계(bi-encoder)로 top-N을 추리고, 2단계(cross-encoder)로
그 N개만 재정렬합니다. Test split(타깃 도메인 held-out)에서 recall/mrr을 재서,
"bi-encoder 단독"과 "bi-encoder + cross-encoder reranker" 성능을 직접 비교합니다.

사용:
  python evaluate_reranker_pipeline.py --bi-encoder kure-v1-resplit-topkpercpos --top-n 20
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from config import DATA_DIR, ROOT
from encoders import Encoder


def load_test_corpus_and_queries(triplets_path):
    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]
    docs = {}
    queries = []
    for r in rows:
        docs[r["positive_id"]] = r["positive"]
        for n in r["negatives"]:
            docs[n["doc_id"]] = n["text"]
        queries.append({"query": r["query"], "gold_id": r["positive_id"]})
    return docs, queries


def recall_mrr(rankings: list[list[str]], golds: list[str], ks=(1, 3, 5, 10)):
    recalls = {k: [] for k in ks}
    rr = []
    for ranked, gold in zip(rankings, golds):
        for k in ks:
            recalls[k].append(1.0 if gold in ranked[:k] else 0.0)
        pos = ranked.index(gold) if gold in ranked else None
        rr.append(1 / (pos + 1) if pos is not None else 0.0)
    return {f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()} | \
        {"mrr": float(np.mean(rr))}


def main(bi_encoder_name: str, cross_encoder_path: str, triplets_base: str, top_n: int):
    test_path = DATA_DIR / f"aihub_triplets_{triplets_base}_Test.jsonl"
    docs, queries = load_test_corpus_and_queries(test_path)
    doc_ids = list(docs)
    doc_texts = [docs[d] for d in doc_ids]
    q_texts = [q["query"] for q in queries]
    golds = [q["gold_id"] for q in queries]
    print(f"코퍼스 {len(doc_ids)}개 문서, 쿼리 {len(queries)}개")

    # 1단계: bi-encoder로 전체 순위
    print(f"\n[1단계] {bi_encoder_name}로 1차 검색...")
    enc = Encoder(bi_encoder_name)
    D = enc.encode_corpus(doc_texts)
    Q = enc.encode_queries(q_texts)
    enc.free()
    sims = Q @ D.T
    order = np.argsort(-sims, axis=1)

    stage1_rankings = [[doc_ids[j] for j in order[i]] for i in range(len(queries))]
    stage1_metrics = recall_mrr(stage1_rankings, golds)
    print("1단계(bi-encoder 단독) 성능:", {k: round(v, 4) for k, v in stage1_metrics.items()})

    # 2단계: cross-encoder로 top_n만 재정렬
    print(f"\n[2단계] cross-encoder로 top-{top_n} 재정렬...")
    ce = CrossEncoder(cross_encoder_path, device="cuda" if torch.cuda.is_available() else "cpu")

    stage2_rankings = []
    for i, ranked in enumerate(stage1_rankings):
        top_candidates = ranked[:top_n]
        pairs = [[q_texts[i], docs[d]] for d in top_candidates]
        ce_scores = ce.predict(pairs, show_progress_bar=False)
        reordered = [top_candidates[j] for j in np.argsort(-ce_scores)]
        # top_n 밖의 문서는 원래 bi-encoder 순위 그대로 뒤에 붙임
        rest = ranked[top_n:]
        stage2_rankings.append(reordered + rest)

    stage2_metrics = recall_mrr(stage2_rankings, golds)
    print("2단계(bi-encoder + cross-encoder) 성능:",
          {k: round(v, 4) for k, v in stage2_metrics.items()})

    df = pd.DataFrame([
        {"stage": "bi-encoder만", **stage1_metrics},
        {"stage": f"bi-encoder+reranker(top-{top_n})", **stage2_metrics},
    ])
    print("\n=== 종합 ===")
    print(df.round(4).to_string(index=False))

    out_path = ROOT / "runs" / "reranker_comparison.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi-encoder", required=True, help="config.py REGISTRY 키")
    ap.add_argument("--cross-encoder", default="trained_models/cross_encoder_reranker/final",
                     help="학습된 cross-encoder 경로")
    ap.add_argument("--triplets", default="bge-m3_topk-percpos_k4")
    ap.add_argument("--top-n", type=int, default=20,
                     help="cross-encoder로 재정렬할 상위 후보 수")
    a = ap.parse_args()
    main(a.bi_encoder, a.cross_encoder, a.triplets, a.top_n)