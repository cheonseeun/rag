"""
4단계: 전 모델 지표 취합 + 부트스트랩 신뢰구간 + 조건부 faithfulness 분해.

쿼리가 20~30개뿐이라 점추정만 보면 안 됩니다.
CI가 겹치면 "차이 없음"으로 보고하는 게 정직합니다.
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import DEFAULT_MODELS, RUN_DIR


def boot_ci(values, n=2000, seed=0):
    v = np.array([x for x in values if x is not None and not np.isnan(x)])
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect(models):
    rows = []
    for m in models:
        d = RUN_DIR / m
        if not (d / "ir_metrics.json").exists():
            continue
        ir = json.load(open(d / "ir_metrics.json", encoding="utf-8"))
        row = {"model": m, "recall@1": ir.get("recall@1"),
               "recall@5": ir.get("recall@5"), "mrr": ir.get("mrr"),
               "ndcg@10": ir.get("ndcg@10"), "dim": ir.get("embedding_dim"),
               "enc_sec": ir.get("encode_sec")}

        jp = d / "judge.json"
        if jp.exists():
            J = json.load(open(jp, encoding="utf-8"))
            vals = list(J.values())
            ans = [r for r in vals if not r["abstained"]]
            f, lo, hi = boot_ci([r["faithfulness"] for r in ans])
            row.update({
                "faithfulness": f, "faith_ci": f"[{lo:.3f}, {hi:.3f}]",
                "answer_relevancy": boot_ci([r["answer_relevancy"] for r in vals])[0],
                "gold_agreement": boot_ci([r.get("gold_agreement") for r in vals])[0],
                "abstain": np.mean([r["abstained"] for r in vals]),
                # 진단용: 정답 문서가 검색됐을 때 vs 아닐 때의 faithfulness
                "faith_hit": boot_ci([r["faithfulness"] for r in ans
                                      if r["n_gold_in_context"] > 0])[0],
                "faith_miss": boot_ci([r["faithfulness"] for r in ans
                                       if r["n_gold_in_context"] == 0])[0],
            })
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    a = p.parse_args()

    df = collect(a.models).sort_values("ndcg@10", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(df.round(4).to_string(index=False))

    df.to_csv(RUN_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (RUN_DIR / "comparison.md").write_text(
        df.round(4).to_markdown(index=False), encoding="utf-8")
    print(f"\n-> {RUN_DIR/'comparison.csv'}")
    print("\n해석 팁: faith_miss(정답 문서 미검색 시 faithfulness)가 높은데 "
          "gold_agreement가 낮다면, 그 모델은 '엉뚱한 문서를 충실히 요약'하고 있는 것입니다.")