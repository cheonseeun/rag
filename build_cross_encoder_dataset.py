"""
기존 bi-encoder triplet(aihub_triplets_*.jsonl)을 cross-encoder 학습 형식으로 변환합니다.

bi-encoder: {"query":.., "positive":.., "negatives":[...]}
cross-encoder 필요 형식: (query, doc, label) 쌍의 목록
  label=1 : query와 doc이 관련 있음 (positive)
  label=0 : query와 doc이 무관함 (negative)

Cross-encoder는 in-batch negative 개념이 없으므로(질문-문서 쌍 하나하나를
독립적으로 채점), 명시적으로 매 (query, negative) 조합을 다 학습 샘플로 만듭니다.
"""
import argparse
import json
import random

from config import DATA_DIR


def convert(triplets_path, seed=0):
    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]
    rng = random.Random(seed)

    samples = []
    for r in rows:
        samples.append({"query": r["query"], "doc": r["positive"], "label": 1})
        for n in r["negatives"]:
            samples.append({"query": r["query"], "doc": n["text"], "label": 0})

    rng.shuffle(samples)
    return samples


def main(triplets_base: str, split: str):
    in_path = DATA_DIR / f"aihub_triplets_{triplets_base}_{split}.jsonl"
    out_path = DATA_DIR / f"cross_encoder_{triplets_base}_{split}.jsonl"

    samples = convert(in_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"{split}: 총 {len(samples)}개 (positive {n_pos}, negative {n_neg}) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", default="bge-m3_topk-percpos_k4",
                     help="변환할 triplet 파일 베이스 이름")
    ap.add_argument("--split", required=True, choices=["Train", "Val", "Test"])
    a = ap.parse_args()
    main(a.triplets, a.split)