"""
하드 네거티브 마이닝: 같은 sub_category 내에서, NV-Retriever 논문
(Moreira et al., 2024)의 positive-aware 방법으로 negative를 선정합니다.

논리:
  - 카테고리 제한: 세은님 도메인(회계기준 소분류)에 맞춘 후보 풀 제한.
    같은 카테고리 안에서만 골라야 "개념은 다른데 표현이 유사해 혼동되는"
    진짜 하드 네거티브가 됩니다.
  - Positive-aware 필터(논문 핵심 기여): 단순 유사도 상위(Naive Top-K)를
    negative로 쓰면, 사실 답으로 인정되어야 할 문서가 negative로 잘못
    라벨링되는 "false negative"가 많이 섞입니다. 논문은 이를 정답 점수를
    기준(anchor)으로 삼아 걸러내는 TopK-PercPos/TopK-MarginPos를 제안했고,
    BEIR 15개 데이터셋 전체 실험에서 TopK-PercPos(정답 점수의 95%를 임계값)가
    가장 높은 정확도(NDCG@10 60.55)를 기록했습니다 (논문 Table 3, 5).

Teacher 모델:
  - dense (기본): config.py REGISTRY에 등록된 임베딩 모델 (bge-m3, kure-v1 등)
  - bm25: 어휘 중복 기반 희소 검색. 논문 Table 1에서 BM25로 마이닝한 negative가
    가장 낮은 성능(NDCG@10 0.5002, random negative보다도 낮음)을 보였다고
    보고되어 있어, 이 실험은 "왜 dense teacher를 썼는가"에 대한 대조군으로 의미가 있습니다.
    (형태소 분석기 없이 공백 기준 토큰화만 사용하므로 한국어 특성상 실제 성능은
    더 낮게 나올 가능성이 있습니다. 이 한계는 논문 명시)

  TopK-PercPos: negative 점수가 (정답 점수 * PERC_THRESHOLD) 이상이면 제외
  TopK-MarginPos: negative 점수가 (정답 점수 - MARGIN_THRESHOLD) 이상이면 제외
  Meghwani (Oracle AI, ACL 2025, arXiv:2505.18366): 임계값 상수 없이 세 유사도를
    서로 비교. sim(Q,D)가 sim(Q,PD)보다 커야 하고(조건1, negative가 질문에는
    정답보다도 가까움 -> 강하게 헷갈림), 동시에 sim(PD,D)보다도 커야 함(조건2,
    negative 문서 자체가 정답 문서와는 확실히 달라야 함 -> 패러프레이즈/false negative 방지).
    TopK-PercPos보다 공격적인 조건 1과, 그걸 보완하는 조건 2가 세트로 작동합니다.

Negative 선택 전략 (--select):
  top       (기본): 필터를 통과한 후보 중 유사도 상위 k개를 그대로 채택.
  sampled   : 필터를 통과한 후보 중 상위 SAMPLE_POOL개를 후보군으로 넓게 잡고,
    유사도가 높을수록 뽑힐 확률이 큰 softmax 가중치로 k개를 무작위 샘플링합니다.
    단, 가장 강한 negative(1위)는 항상 포함시켜 안정성을 확보합니다(Top-1+sampled
    top-k, NV-Retriever 논문 4.3.2절). 매번 같은 "가장 어려운 조합"만 뽑는 대신
    난이도가 섞인 조합을 만들어, negative 다양성을 늘리는 효과를 노립니다.
    필터 자체는 안 건드리므로 false negative 위험이나 스킵률에는 영향이 없습니다.

출력: data/aihub_triplets_{teacher}_{method}_k{n}_{select}_{split}.jsonl
  {"query":.., "positive":.., "positive_id":.., "positive_score":..,
   "negatives":[{"doc_id":.., "text":.., "score":.., "sim_to_positive":..}, ...],
   "sub_category":.., "main_category":.., "split":..}
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

from config import DATA_DIR
from encoders import Encoder

PERC_THRESHOLD = 0.95    # TopK-PercPos: 논문에서 최적으로 확인된 값
MARGIN_THRESHOLD = 0.05  # TopK-MarginPos: 논문에서 최적으로 확인된 값
CANDIDATE_POOL = 50       # 이 안에서 후보를 뒤져 필터링 (카테고리가 작으면 자동으로 줄어듦)
SAMPLE_POOL = 10          # Sampled Top-k: 필터 통과 후보 중 이 안에서 샘플링 (논문 최적값)


def select_negatives(candidates: list[dict], n_negatives: int, select: str,
                      seed_rng: random.Random) -> list[dict]:
    """필터를 이미 통과한 후보 리스트(유사도 내림차순 정렬됨)에서 최종 k개를 고릅니다.

    candidates: [{"doc_id":.., "text":.., "score":.., ...}, ...] (내림차순 정렬됨)
    """
    if select == "top" or len(candidates) <= n_negatives:
        return candidates[:n_negatives]

    if select == "sampled":
        pool = candidates[:SAMPLE_POOL]
        top1, rest = pool[0], pool[1:]
        if not rest:
            return pool[:n_negatives]
        scores = np.array([c["score"] for c in rest], dtype=np.float64)
        # softmax 가중치: 유사도가 높을수록 뽑힐 확률이 커짐
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        n_sample = min(n_negatives - 1, len(rest))
        idxs = seed_rng.choices(range(len(rest)), weights=weights.tolist(), k=n_sample) \
            if n_sample < len(rest) else list(range(len(rest)))
        # choices는 복원추출이라 중복 가능 -> 논문 관찰(중복 negative가 오히려 학습에
        # 유리)과 일치하므로 그대로 둠
        sampled = [rest[i] for i in idxs]
        return [top1] + sampled

    raise ValueError(select)


def load_pairs(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def dedup_docs(pairs):
    """doc_id -> {"text", "sub", "main"} 로 문서 풀을 중복 제거."""
    docs = {}
    for r in pairs:
        did = r["doc_id"]
        if did not in docs:
            docs[did] = {"text": r["positive"], "sub": r["sub_category"],
                         "main": r["main_category"]}
    return docs


def filter_negative(method: str, pos_score: float, neg_score: float,
                     sim_pos_neg: float | None = None) -> bool:
    """True면 이 negative를 채택(=false negative 위험이 낮음).

    sim_pos_neg: positive 문서와 negative 후보 문서 사이의 유사도.
                 method="meghwani"에서만 필요합니다.
    """
    if method == "topk-percpos":
        return neg_score < pos_score * PERC_THRESHOLD
    if method == "topk-marginpos":
        return neg_score < pos_score - MARGIN_THRESHOLD
    if method == "meghwani":
        if sim_pos_neg is None:
            raise ValueError("meghwani 방식은 sim_pos_neg(positive-negative 유사도)가 필요합니다")
        cond1 = neg_score > pos_score          # negative가 질문에 정답보다도 가까움
        cond2 = neg_score > sim_pos_neg         # negative가 질문과 갖는 관계가
                                                 # negative-정답 관계보다 강함
        return cond1 and cond2
    if method == "naive":
        return True   # 필터 없음 (논문의 baseline, 비교용)
    raise ValueError(method)


def build_bm25_scores(doc_texts: list[str], q_texts: list[str]):
    """BM25로 (n_queries, n_docs) 점수 행렬을 만듭니다. bm25 인덱스와 토큰화 함수도
    같이 반환합니다(meghwani 방식에서 positive 문서를 쿼리처럼 재사용하기 위함).

    형태소 분석기 없이 간단한 정규식 토큰화만 사용합니다. 한국어는 조사/어미가
    붙어 어휘가 그대로 겹치지 않는 경우가 많아, 제대로 된 형태소 분석 없이는
    BM25가 실제보다 더 불리하게 나올 수 있습니다. 이 한계를 감안하고 보세요.
    """
    import re

    from rank_bm25 import BM25Okapi

    def tokenize(t: str) -> list[str]:
        return re.findall(r"[가-힣]+|[A-Za-z0-9]+", t)

    print("BM25 인덱스 구축 중...")
    tokenized_docs = [tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

    print("BM25 쿼리 점수 계산 중 (CPU, 오래 걸릴 수 있습니다)...")
    n_docs = len(doc_texts)
    scores = np.zeros((len(q_texts), n_docs), dtype=np.float32)
    for i, q in enumerate(q_texts):
        scores[i] = bm25.get_scores(tokenize(q))
        if i % 2000 == 0:
            print(f"  {i}/{len(q_texts)}", end="\r")
    print()
    return scores, bm25, tokenize


# 참고: dense teacher 로직은 main() 안에 GPU 배치 처리로 인라인되어 있습니다.


def main(teacher: str, split_filter: str | None, method: str, n_negatives: int,
         source_file: str = "aihub_pairs.jsonl", select: str = "top", seed: int = 0,
         global_pool: bool = False):
    pairs = load_pairs(DATA_DIR / source_file)
    if split_filter:
        pairs = [r for r in pairs if r["split"] == split_filter]
    print(f"쿼리 {len(pairs)}개 / teacher: {teacher} / 마이닝 방법: {method} / k={n_negatives} "
          f"/ 선택전략: {select} / 후보풀: {'전체 코퍼스' if global_pool else '카테고리 제한'} "
          f"/ 원본: {source_file}")

    docs = dedup_docs(pairs)
    doc_ids = list(docs)
    id2idx = {d: i for i, d in enumerate(doc_ids)}
    print(f"고유 문서 {len(doc_ids)}개")

    # 카테고리별 문서 인덱스 (sub_category 내부로만 negative 후보 제한)
    cat_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(doc_ids):
        cat_to_idxs[docs[d]["sub"]].append(i)

    doc_texts = [docs[d]["text"] for d in doc_ids]
    q_texts = [r["query"] for r in pairs]

    bm25 = tokenize = None
    bm25_pos_scores: dict[str, np.ndarray] = {}   # meghwani + bm25 전용 캐시
    if teacher == "bm25":
        scores, bm25, tokenize = build_bm25_scores(doc_texts, q_texts)  # (n_q, n_docs), numpy, CPU
        get_row = lambda i: scores[i]
    else:
        # dense는 GPU 배치 연산이 가능하므로 아래 루프에서 D_t/Q_t로 직접 계산
        print(f"[{teacher}] 문서/쿼리 인코딩 중...")
        enc = Encoder(teacher)
        D = enc.encode_corpus(doc_texts)
        Q = enc.encode_queries(q_texts)
        enc.free()
        D_t = torch.from_numpy(D)
        Q_t = torch.from_numpy(Q)
        if torch.cuda.is_available():
            D_t, Q_t = D_t.cuda(), Q_t.cuda()
        get_row = None

    out = []
    skipped_small_cat = 0
    skipped_all_filtered = 0
    rng = random.Random(seed)
    all_idxs = list(range(len(doc_ids)))   # global_pool용: 카테고리 무시하고 전체 문서
    BATCH = 256
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start:start + BATCH]
        if teacher != "bm25":
            q_batch = Q_t[start:start + BATCH]

        for i, r in enumerate(batch):
            sub = r["sub_category"]
            cand_idxs = all_idxs if global_pool else cat_to_idxs[sub]
            pos_idx = id2idx[r["doc_id"]]
            if len(cand_idxs) <= 1:
                skipped_small_cat += 1
                continue

            if teacher == "bm25":
                row = get_row(start + i)                           # (n_docs,) numpy
                pos_score = float(row[pos_idx])
                sims = row[cand_idxs]
            else:
                pos_score = float(q_batch[i] @ D_t[pos_idx])
                cand_idxs_t = torch.tensor(cand_idxs, device=q_batch.device)
                sims = (D_t[cand_idxs_t] @ q_batch[i]).cpu().numpy()

            # meghwani/hybrid 방식에만 필요: positive 문서와 각 후보 문서 사이의 유사도.
            # dense는 이미 만든 문서 임베딩끼리 내적하면 되어 추가 비용이 거의 없지만,
            # bm25는 positive 문서를 별도의 "쿼리"로 취급해 재계산해야 해서 비용이 큽니다.
            sim_pos_negs = None
            if method in ("meghwani", "hybrid"):
                if teacher == "bm25":
                    pos_row = bm25_pos_scores.get(r["doc_id"])
                    if pos_row is None:
                        pos_row = bm25.get_scores(tokenize(docs[r["doc_id"]]["text"]))
                        bm25_pos_scores[r["doc_id"]] = pos_row
                    sim_pos_negs = pos_row[cand_idxs]
                else:
                    cand_idxs_t2 = torch.tensor(cand_idxs, device=D_t.device)
                    sim_pos_negs = (D_t[cand_idxs_t2] @ D_t[pos_idx]).cpu().numpy()

            order = np.argsort(-sims)
            ranked_idxs = [cand_idxs[j] for j in order]
            ranked_sims = sims[order]
            ranked_pos_negs = sim_pos_negs[order] if sim_pos_negs is not None else None

            negatives = []
            n_from_meghwani = 0
            if method == "hybrid":
                # 1단계: meghwani의 엄격한 이중조건으로 "진짜 극단적으로 헷갈리는" negative부터 확보
                for rank, (idx, sim) in enumerate(zip(ranked_idxs[:CANDIDATE_POOL],
                                                        ranked_sims[:CANDIDATE_POOL])):
                    if doc_ids[idx] == r["doc_id"]:
                        continue
                    spn = float(ranked_pos_negs[rank]) if ranked_pos_negs is not None else None
                    if not filter_negative("meghwani", pos_score, float(sim), spn):
                        continue
                    negatives.append({"doc_id": doc_ids[idx], "text": doc_texts[idx],
                                       "score": float(sim), "sim_to_positive": spn,
                                       "source": "meghwani"})
                    if len(negatives) >= n_negatives:
                        break
                n_from_meghwani = len(negatives)

                # 2단계: 부족한 자리는 topk-percpos(더 관대한 조건)로 채움.
                # 이미 1단계에서 뽑힌 문서는 중복으로 다시 뽑지 않음.
                if len(negatives) < n_negatives:
                    picked_ids = {n["doc_id"] for n in negatives}
                    for rank, (idx, sim) in enumerate(zip(ranked_idxs[:CANDIDATE_POOL],
                                                            ranked_sims[:CANDIDATE_POOL])):
                        if doc_ids[idx] == r["doc_id"] or doc_ids[idx] in picked_ids:
                            continue
                        if not filter_negative("topk-percpos", pos_score, float(sim)):
                            continue
                        negatives.append({"doc_id": doc_ids[idx], "text": doc_texts[idx],
                                           "score": float(sim), "source": "topk-percpos"})
                        picked_ids.add(doc_ids[idx])
                        if len(negatives) >= n_negatives:
                            break
            else:
                for rank, (idx, sim) in enumerate(zip(ranked_idxs[:CANDIDATE_POOL],
                                                        ranked_sims[:CANDIDATE_POOL])):
                    if doc_ids[idx] == r["doc_id"]:
                        continue                      # 정답 자신 제외
                    spn = float(ranked_pos_negs[rank]) if ranked_pos_negs is not None else None
                    if not filter_negative(method, pos_score, float(sim), spn):
                        continue                      # positive-aware 필터 (false negative 위험 제거)
                    neg_entry = {"doc_id": doc_ids[idx], "text": doc_texts[idx], "score": float(sim)}
                    if spn is not None:
                        neg_entry["sim_to_positive"] = spn
                    negatives.append(neg_entry)
                    if len(negatives) >= CANDIDATE_POOL:   # 필터 통과 후보를 넉넉히 모아둠
                        break

                negatives = select_negatives(negatives, n_negatives, select, rng)

            if not negatives:
                skipped_all_filtered += 1
                continue

            out.append({
                "query": r["query"], "positive": r["positive"],
                "positive_id": r["doc_id"], "positive_score": pos_score,
                "negatives": negatives,
                "sub_category": sub, "main_category": r["main_category"],
                "split": r["split"],
                **({"n_from_meghwani": n_from_meghwani} if method == "hybrid" else {}),
            })

        print(f"  {min(start + BATCH, len(pairs))}/{len(pairs)}", end="\r")

    split_tag = split_filter if split_filter else "all"
    select_tag = "" if select == "top" else f"_{select}"   # 기존 파일명 규칙과 호환 유지
    pool_tag = "_global" if global_pool else ""
    out_path = DATA_DIR / f"aihub_triplets_{teacher}_{method}_k{n_negatives}{select_tag}{pool_tag}_{split_tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n\n최종 triplet {len(out)}개 -> {out_path}")
    print(f"(같은 카테고리 문서가 1개뿐이라 스킵: {skipped_small_cat}개)")
    print(f"(모든 후보가 필터에 걸려 negative 0개인 쿼리: {skipped_all_filtered}개)")

    if method == "hybrid" and out:
        meghwani_counts = [r.get("n_from_meghwani", 0) for r in out]
        full_meghwani = sum(1 for c in meghwani_counts if c >= n_negatives)
        partial = sum(1 for c in meghwani_counts if 0 < c < n_negatives)
        none_meghwani = sum(1 for c in meghwani_counts if c == 0)
        print(f"\n[hybrid 구성] meghwani만으로 {n_negatives}개 다 채운 쿼리: {full_meghwani}개 "
              f"({full_meghwani/len(out)*100:.1f}%)")
        print(f"[hybrid 구성] meghwani 일부 + topk-percpos로 보충: {partial}개 "
              f"({partial/len(out)*100:.1f}%)")
        print(f"[hybrid 구성] meghwani 0개, 전부 topk-percpos: {none_meghwani}개 "
              f"({none_meghwani/len(out)*100:.1f}%)")
        print(f"[hybrid 구성] 쿼리당 평균 meghwani negative 수: {np.mean(meghwani_counts):.2f} / {n_negatives}")

    # 진단: negative 유사도 분포 및 정답 대비 상대 위치
    all_scores = [n["score"] for r in out for n in r["negatives"]]
    all_gaps = [r["positive_score"] - n["score"] for r in out for n in r["negatives"]]
    if all_scores:
        print(f"negative 점수: mean={np.mean(all_scores):.3f} "
              f"median={np.median(all_scores):.3f} min={np.min(all_scores):.3f}")
        print(f"정답과의 점수 격차(양수=negative가 더 낮음): "
              f"mean={np.mean(all_gaps):.3f} min={np.min(all_gaps):.3f}")
        n_per_query = [len(r["negatives"]) for r in out]
        print(f"쿼리당 negative 개수: mean={np.mean(n_per_query):.2f} "
              f"(목표 k={n_negatives})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="bge-m3",
                     help="config.py REGISTRY의 인코더 이름 (bge-m3, kure-v1 등) 또는 'bm25'")
    ap.add_argument("--split", default=None,
                     choices=[None, "Training", "Validation", "Train", "Val", "Test"])
    ap.add_argument("--source", default="aihub_pairs.jsonl",
                     help="data/ 아래 원본 pairs 파일명. 3단 분할을 쓰려면 "
                          "aihub_pairs_resplit.jsonl 로 지정")
    ap.add_argument("--method", default="topk-percpos",
                     choices=["topk-percpos", "topk-marginpos", "meghwani", "hybrid", "naive"],
                     help="topk-percpos: 정답 점수의 95%% 임계 (논문 최적 성능). "
                          "topk-marginpos: 정답 점수 - 0.05. "
                          "meghwani: Oracle AI 논문의 이중 조건 (positive-negative 유사도도 확인). "
                          "hybrid: meghwani로 먼저 채우고, k개를 못 채우면 나머지는 "
                          "topk-percpos로 채움 (meghwani의 데이터 부족 문제 완화용). "
                          "naive: 필터 없음 (비교용 baseline)")
    ap.add_argument("--k", type=int, default=4,
                     help="쿼리당 채택할 하드 네거티브 수 (논문 기본값 4)")
    ap.add_argument("--select", default="top", choices=["top", "sampled"],
                     help="top: 필터 통과 후보 중 유사도 상위 k개 그대로 채택 (기본). "
                          "sampled: 상위 10개 중 softmax 가중 샘플링 (top-1은 항상 포함), "
                          "negative 다양성 확보용")
    ap.add_argument("--seed", type=int, default=0, help="--select sampled 사용 시 재현성용 시드")
    ap.add_argument("--global-pool", action="store_true",
                     help="negative 후보를 같은 sub_category로 제한하지 않고 전체 코퍼스에서 찾음. "
                          "meghwani처럼 후보가 부족해 스킵률이 높은 방법에서 후보 풀을 넓혀보는 "
                          "ablation용. 카테고리 밖 무관한 문서가 섞일 위험이 있으니 결과를 "
                          "반드시 정성적으로 확인하세요.")
    a = ap.parse_args()
    main(a.teacher, a.split, a.method, a.k, a.source, a.select, a.seed, a.global_pool)