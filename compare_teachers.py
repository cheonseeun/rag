"""
서로 다른 teacher 모델(인코더)로 마이닝한 하드 네거티브를 비교합니다.
NV-Retriever 논문 Appendix A(Table 6)의 Jaccard 유사도 분석에 대응합니다.

teacher 모델이 서로 다른 negative를 뽑을수록(낮은 Jaccard) 정보가 상호보완적이라,
논문은 이런 경우 앙상블(intra-sample ensembling)이 도움이 될 수 있다고 보고합니다.

사용:
  python compare_teachers.py --encoders bge-m3 kure-v1 --method topk-percpos
"""
import argparse
import json
from itertools import combinations

import numpy as np

from config import DATA_DIR


def load_triplets(teacher: str, method: str, k: int) -> dict[str, dict]:
    path = DATA_DIR / f"aihub_triplets_{teacher}_{method}_k{k}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 -> 먼저 mine_hard_negatives.py --teacher {teacher} 실행")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    # query 텍스트를 key로 사용 (쿼리는 teacher와 무관하게 동일)
    return {r["query"]: r for r in rows}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main(encoders: list[str], method: str, k: int):
    data = {enc: load_triplets(enc, method, k) for enc in encoders}
    common_queries = set.intersection(*(set(d) for d in data.values()))
    print(f"공통 쿼리 {len(common_queries)}개로 비교\n")

    # 1) Jaccard 유사도 행렬 (top-4 negative doc_id 집합 기준)
    print("=== Negative 집합 Jaccard 유사도 (평균) ===")
    header = "".ljust(14) + "".join(e[:12].ljust(14) for e in encoders)
    print(header)
    for e1 in encoders:
        row = [e1.ljust(14)]
        for e2 in encoders:
            if e1 == e2:
                row.append("-".ljust(14))
                continue
            jaccards = []
            for q in common_queries:
                s1 = {n["doc_id"] for n in data[e1][q]["negatives"]}
                s2 = {n["doc_id"] for n in data[e2][q]["negatives"]}
                jaccards.append(jaccard(s1, s2))
            row.append(f"{np.mean(jaccards):.3f}".ljust(14))
        print("".join(row))

    # 2) 각 teacher의 negative 품질 지표 (정답과의 점수 격차)
    print("\n=== Teacher별 negative 품질 (정답과의 점수 격차, 작을수록 강한 hard negative) ===")
    for e in encoders:
        gaps = [data[e][q]["positive_score"] - n["score"]
                 for q in common_queries for n in data[e][q]["negatives"]]
        n_neg = [len(data[e][q]["negatives"]) for q in common_queries]
        print(f"{e:14s} gap_mean={np.mean(gaps):.3f}  gap_min={np.min(gaps):.3f}  "
              f"avg_negatives/query={np.mean(n_neg):.2f}")

    print("\n해석: Jaccard가 낮을수록(<0.3) 두 teacher가 서로 다른 negative를 찾는다는 뜻이며,")
    print("이 경우 논문처럼 두 teacher의 negative를 합쳐 쓰는 앙상블도 고려할 수 있습니다.")
    print("gap_mean이 작고 min이 양수인 teacher일수록 false negative 없이 강한 negative를 뽑은 것입니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--method", default="topk-percpos")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    main(a.encoders, a.method, a.k)