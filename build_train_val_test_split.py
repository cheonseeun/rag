"""
AI-Hub가 원래 나눠준 Training/Validation을 하나로 합친 뒤, 문서 단위로
train/val/test 3단 분할을 새로 만듭니다.

핵심: 쿼리 단위로 무작위 분할하면 같은 문서에서 나온 다른 QA가 서로 다른
split에 흩어져 들어가 정보 누수가 생깁니다 (예전에 겪었던 문제와 동일).
그래서 "문서(doc_id)"를 기준으로 먼저 그룹을 나누고, 그 그룹째로
train/val/test에 배정합니다 — 같은 문서의 QA는 항상 같은 split에만 있습니다.

사용:
  python build_train_val_test_split.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

출력:
  data/aihub_pairs_resplit.jsonl   (split 필드가 "Train"/"Val"/"Test"로 갱신됨)
"""
import argparse
import json
import random
from collections import defaultdict

from config import DATA_DIR


def load_pairs(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main(train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합은 1.0이어야 합니다"

    pairs = load_pairs(DATA_DIR / "aihub_pairs.jsonl")
    print(f"원본(Training+Validation 합계) 쿼리: {len(pairs)}개")

    # 문서(doc_id) 단위로 쿼리를 묶음
    doc_to_queries: dict[str, list[dict]] = defaultdict(list)
    for r in pairs:
        doc_to_queries[r["doc_id"]].append(r)

    doc_ids = list(doc_to_queries)
    print(f"고유 문서: {len(doc_ids)}개")

    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 나머지는 전부 test로 (반올림 오차 흡수)

    train_docs = set(doc_ids[:n_train])
    val_docs = set(doc_ids[n_train:n_train + n_val])
    test_docs = set(doc_ids[n_train + n_val:])

    out = []
    for r in pairs:
        d = r["doc_id"]
        if d in train_docs:
            new_split = "Train"
        elif d in val_docs:
            new_split = "Val"
        else:
            new_split = "Test"
        r2 = dict(r)
        r2["split"] = new_split          # 기존 Training/Validation 값을 덮어씀
        r2["orig_split"] = r["split"]    # 원래 AI-Hub 분류도 참고용으로 보존
        out.append(r2)

    out_path = DATA_DIR / "aihub_pairs_resplit.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    doc_counts = defaultdict(set)
    for r in out:
        counts[r["split"]] += 1
        doc_counts[r["split"]].add(r["doc_id"])

    print(f"\n-> {out_path}")
    print("\n=== 분할 결과 ===")
    for split in ("Train", "Val", "Test"):
        print(f"  {split:6s}: 쿼리 {counts[split]:6d}개  |  문서 {len(doc_counts[split]):5d}개  "
              f"({counts[split] / len(out) * 100:.1f}%)")

    # 문서 중복 여부 최종 확인
    overlap_tv = doc_counts["Train"] & doc_counts["Val"]
    overlap_tt = doc_counts["Train"] & doc_counts["Test"]
    overlap_vt = doc_counts["Val"] & doc_counts["Test"]
    print(f"\n문서 중복 확인: Train∩Val={len(overlap_tv)}  "
          f"Train∩Test={len(overlap_tt)}  Val∩Test={len(overlap_vt)}")
    if overlap_tv or overlap_tt or overlap_vt:
        print("경고: 중복이 있습니다. 코드를 다시 확인하세요.")
    else:
        print("문서 중복 없음. 세 split이 서로 독립적입니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.train_ratio, a.val_ratio, a.test_ratio, a.seed)