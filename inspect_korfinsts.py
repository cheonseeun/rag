"""
nmixx-fin/korfinSTS 데이터셋 구조 확인.

컬럼: data_type, sentence1, sentence2, score (int64)
score가 이진(0/1)인지 연속 등급(0~5 등)인지, data_type이 몇 종류인지
(논문에 따르면 뉴스/공시/리서치/규정 네 출처가 있다고 함) 먼저 확인합니다.
"""
from datasets import load_dataset

ds = load_dataset("nmixx-fin/korfinSTS", split="train")
print(f"전체 행 수: {len(ds)}")
print(f"컬럼: {ds.column_names}\n")

print("=== data_type 별 개수 ===")
from collections import Counter
dtypes = Counter(ds["data_type"])
for k, v in dtypes.most_common():
    print(f"  {k}: {v}")

print("\n=== score 분포 (전체) ===")
scores = Counter(ds["score"])
for k, v in sorted(scores.items()):
    print(f"  {k}: {v}")

print("\n=== data_type 별 score 분포 ===")
import numpy as np
data_types = sorted(set(ds["data_type"]))
for dt in data_types:
    sub_scores = [s for s, d in zip(ds["score"], ds["data_type"]) if d == dt]
    print(f"  {dt}: n={len(sub_scores)}, unique_scores={sorted(set(sub_scores))}, "
          f"mean={np.mean(sub_scores):.3f}")

print("\n=== score별 샘플 문장 쌍 (각 score값에서 1개씩) ===")
seen_scores = set()
for row in ds:
    if row["score"] not in seen_scores:
        seen_scores.add(row["score"])
        print(f"\n[score={row['score']}] data_type={row['data_type']}")
        print("S1:", row["sentence1"][:80])
        print("S2:", row["sentence2"][:80])