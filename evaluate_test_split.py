r"""
재분할(Train/Val/Test) 중 Test triplet으로 "타깃 도메인 내부, 완전히 못 본 문서"에서의
검색 성능을 평가합니다. AutoRAGRetrieval/KorFinSTS는 도메인이 달라 일반화만 봤는데,
이건 학습 도메인(회계기준) 안에서 실제로 성능이 좋아졌는지를 직접 잽니다.

코퍼스: Test triplet에 등장하는 모든 문서(positive + negatives) 전체.
        Test 문서는 정의상 학습 때 전혀 노출되지 않았습니다 (build_train_val_test_split.py
        에서 문서 단위로 분리했으므로).

사용:
  python evaluate_test_split.py --models kure-v1 kure-v1-resplit-topkpercpos kure-v1-resplit-meghwani
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from config import DATA_DIR
from encoders import Encoder


def _tokenize(t: str) -> list[str]:
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", t)


def build_bm25_ranks(doc_texts: list[str], q_texts: list[str]) -> np.ndarray:
    """BM25 기준 각 쿼리별 문서 순위(0=1등)를 (n_q, n_docs) 배열로 반환."""
    from rank_bm25 import BM25Okapi

    print("  BM25 인덱스 구축 중 (hybrid용)...")
    bm25 = BM25Okapi([_tokenize(t) for t in doc_texts])
    n_docs = len(doc_texts)
    ranks = np.zeros((len(q_texts), n_docs), dtype=np.int32)
    for i, q in enumerate(q_texts):
        scores = bm25.get_scores(_tokenize(q))
        order = np.argsort(-scores)
        rank_of = np.empty(n_docs, dtype=np.int32)
        rank_of[order] = np.arange(n_docs)          # rank_of[j] = 문서 j의 순위(0-base)
        ranks[i] = rank_of
    return ranks


def rrf_fuse(dense_order: np.ndarray, bm25_ranks: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: 두 방법의 순위를 합쳐 최종 순위를 만듭니다.

    RRF(문서) = 1/(k + dense_순위) + 1/(k + bm25_순위)
    점수 스케일이 완전히 다른 dense(코사인)와 BM25(비유계 TF-IDF)를 직접 더할 수
    없으므로, "몇 등이었는가"라는 공통 잣대로 바꿔서 결합합니다. k=60은 RRF
    원 논문(Cormack et al., 2009)의 관례적 권장값입니다.
    """
    n_q, n_docs = dense_order.shape
    dense_rank = np.empty_like(dense_order)
    for i in range(n_q):
        dense_rank[i, dense_order[i]] = np.arange(n_docs)   # dense 순위(0-base)로 변환

    rrf_score = 1.0 / (k + dense_rank + 1) + 1.0 / (k + bm25_ranks + 1)
    return np.argsort(-rrf_score, axis=1)                    # (n_q, n_docs) 최종 순위(문서 인덱스)


def load_test_corpus_and_queries(triplets_path):
    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]
    docs = {}          # doc_id -> text
    queries = []        # {"query":.., "gold_id":.., "negative_ids":[...]}
    for r in rows:
        docs[r["positive_id"]] = r["positive"]
        neg_ids = []
        for n in r["negatives"]:
            docs[n["doc_id"]] = n["text"]
            neg_ids.append(n["doc_id"])
        queries.append({"query": r["query"], "gold_id": r["positive_id"],
                         "negative_ids": neg_ids})
    return docs, queries


def evaluate(enc: Encoder, docs: dict, queries: list, ks=(1, 3, 5, 10),
             use_hybrid: bool = False):
    doc_ids = list(docs)
    doc_texts = [docs[d] for d in doc_ids]
    D = enc.encode_corpus(doc_texts)
    Q = enc.encode_queries([q["query"] for q in queries])

    sims = Q @ D.T                                   # (n_q, n_docs), 코사인 유사도(임베딩 거리)
    order = np.argsort(-sims, axis=1)
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    hybrid_order = None
    if use_hybrid:
        bm25_ranks = build_bm25_ranks(doc_texts, [q["query"] for q in queries])
        hybrid_order = rrf_fuse(order, bm25_ranks)

    def _score(rank_order):
        recalls = {k: [] for k in ks}
        rr = []
        for i, q in enumerate(queries):
            ranked = [doc_ids[j] for j in rank_order[i]]
            gold = q["gold_id"]
            for k in ks:
                recalls[k].append(1.0 if gold in ranked[:k] else 0.0)
            rank_pos = ranked.index(gold) if gold in ranked else None
            rr.append(1 / (rank_pos + 1) if rank_pos is not None else 0.0)
        return {f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()} | \
            {"mrr": float(np.mean(rr))}

    dense_metrics = _score(order)

    sim_to_gold = []       # 질문-정답 문서 임베딩 거리
    sim_top1 = []          # 질문-1위 문서 임베딩 거리 (정답이든 오답이든)
    miss_gaps = []         # 정답을 놓친 경우: (1위 오답 유사도 - 정답 유사도). 클수록 심하게 헷갈림
    sim_to_negatives = []  # 질문-마이닝된 negative(오답) 문서들의 임베딩 거리 (전체 평균)
    for i, q in enumerate(queries):
        ranked = [doc_ids[j] for j in order[i]]
        gold = q["gold_id"]
        gold_idx = doc_id_to_idx[gold]

        rank_pos = ranked.index(gold) if gold in ranked else None
        s_gold = float(sims[i, gold_idx])
        s_top1 = float(sims[i, order[i, 0]])
        sim_to_gold.append(s_gold)
        sim_top1.append(s_top1)
        if rank_pos != 0:   # 1위가 정답이 아니면(못 맞혔으면) 격차 기록
            miss_gaps.append(s_top1 - s_gold)

        for neg_id in q.get("negative_ids", []):
            neg_idx = doc_id_to_idx.get(neg_id)
            if neg_idx is not None:
                sim_to_negatives.append(float(sims[i, neg_idx]))

    result = dict(dense_metrics) | {
        "n_queries": len(queries), "n_docs": len(doc_ids),
        # --- 임베딩 거리(코사인 유사도) 수치화 ---
        "mean_sim_to_gold": float(np.mean(sim_to_gold)),
        "mean_sim_top1": float(np.mean(sim_top1)),
        "mean_sim_to_negatives": float(np.mean(sim_to_negatives)) if sim_to_negatives else 0.0,
        "sim_gold_minus_neg": float(np.mean(sim_to_gold) - np.mean(sim_to_negatives))
                              if sim_to_negatives else 0.0,
        "mean_miss_gap": float(np.mean(miss_gaps)) if miss_gaps else 0.0,
        "n_misses": len(miss_gaps),
    }
    if use_hybrid:
        hybrid_metrics = _score(hybrid_order)
        result |= {f"hybrid_{k}": v for k, v in hybrid_metrics.items()}
    return result

def load_qa_corpus_and_queries(qa_path):
    """qa_test.jsonl(question/golden_answers/metadata 스키마) 로더.
    negatives가 없는 순수 QA 데이터라, 코퍼스는 전체 golden_answers를
    doc_id 기준으로 모은 것이 되고, negative_ids는 빈 리스트로 둡니다
    (즉 sim_to_negatives/sim_gold_minus_neg는 계산되지 않고, recall/mrr은
    전체 코퍼스 기준으로 정확히 계산됩니다)."""
    rows = [json.loads(l) for l in open(qa_path, encoding="utf-8")]
    docs = {}
    queries = []
    skipped = 0
    for r in rows:
        doc_id = r["metadata"]["doc_id"]
        answers = r.get("golden_answers", [])
        if not answers:
            skipped += 1
            continue
        # 같은 doc_id가 여러 QA에 등장할 수 있음 -> 처음 것으로 고정(중복 등록 방지)
        docs.setdefault(doc_id, answers[0])
        queries.append({"query": r["question"], "gold_id": doc_id, "negative_ids": []})
    if skipped:
        print(f"경고: golden_answers 없는 row {skipped}개 스킵")
    return docs, queries

def main(models: list[str], triplets_base: str, hybrid: bool, qa_file: str | None):
    if qa_file:
        qa_path = DATA_DIR / qa_file
        if not qa_path.exists():
            raise FileNotFoundError(f"{qa_path} 없음")
        docs, queries = load_qa_corpus_and_queries(qa_path)
        print(f"QA 코퍼스({qa_file}): 문서 {len(docs)}개, 쿼리 {len(queries)}개")
    else:
        test_path = DATA_DIR / f"aihub_triplets_{triplets_base}_Test.jsonl"
        if not test_path.exists():
            raise FileNotFoundError(
                f"{test_path} 없음 -> mine_hard_negatives.py --split Test 로 생성하세요")
        docs, queries = load_test_corpus_and_queries(test_path)
        print(f"Test 코퍼스: 문서 {len(docs)}개, 쿼리 {len(queries)}개 (전부 학습 중 미노출)")

    if hybrid:
        print("Hybrid(dense+BM25, RRF 결합) 지표도 함께 계산합니다 (쿼리마다 BM25 재계산이라 느릴 수 있음).")

    rows = []
    for name in models:
        print(f"\n[{name}] 평가 중...")
        enc = Encoder(name)
        metrics = evaluate(enc, docs, queries, use_hybrid=hybrid)
        enc.free()
        rows.append({"model": name, **metrics})
        print(f"  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                 if isinstance(v, float)))
    # ... 이하 동일 (df 생성, 출력, csv 저장) ...
        
    df = pd.DataFrame(rows).sort_values("mrr", ascending=False)
    pd.set_option("display.width", 160)
    print("\n=== 종합 (Test split, 학습 도메인 내부 held-out) ===")
    print(df.round(4).to_string(index=False))

    print("\n=== 임베딩 거리(코사인 유사도) 요약 ===")
    print("mean_sim_to_gold: 질문-정답 문서 평균 유사도 (높을수록 정답을 확신 있게 끌어당김)")
    print("mean_sim_top1   : 질문-1위 문서 평균 유사도 (정답이든 오답이든)")
    print("mean_miss_gap   : 정답을 못 맞춘 경우, (1위 오답 유사도 - 정답 유사도)의 평균.")
    print("                  0에 가까울수록 '거의 맞출 뻔한' 실수, 클수록 확실히 헷갈린 것")
    cols = ["model", "mean_sim_to_gold", "mean_sim_top1", "mean_miss_gap", "n_misses"]
    print(df[cols].round(4).to_string(index=False))

    if hybrid:
        print("\n=== Dense 단독 vs Hybrid(RRF) 비교 ===")
        hybrid_cols = ["model", "recall@1", "hybrid_recall@1", "mrr", "hybrid_mrr"]
        print(df[hybrid_cols].round(4).to_string(index=False))
        print("(hybrid_* 가 더 높으면, 이 모델은 BM25와 결합했을 때 이득을 보는 경우입니다.")
        print(" 파인튜닝으로 의미 판별력이 좋아졌다면 dense 단독만으로 충분해 hybrid 이득이")
        print(" 줄어드는 경향이 나올 수 있습니다.)")

    out_path = DATA_DIR.parent / "runs" / "test_split_comparison.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                     help="config.py REGISTRY 키들 (baseline과 파인튜닝 모델 함께 지정)")
    ap.add_argument("--triplets", default="bge-m3_topk-percpos_k4",
                     help="Test 코퍼스를 만들 triplet 파일 베이스 이름")
    ap.add_argument("--hybrid", action="store_true",
                     help="dense 단독 지표에 더해, dense+BM25를 RRF로 결합한 hybrid 지표도 계산")
    ap.add_argument("--qa-file", default=None,
                     help="data/ 아래 QA 형식(question/golden_answers/metadata) jsonl 파일명. "
                          "지정하면 기존 triplet Test 파일 대신 이 파일로 평가합니다.")
    a = ap.parse_args()
    main(a.models, a.triplets, a.hybrid, a.qa_file)