r"""
KorFinSTS로 파인튜닝 전/후 비교.

score=1 : 진짜 패러프레이즈 (의미 동일, 표현만 다름) -> 대조군, 유사도 유지가 이상적
score=0 : 의미변화(semantic-shift) 쌍 (표면은 비슷하나 함의가 다르거나 반대) -> confusable 그룹,
          파인튜닝 후 유사도가 낮아져 더 잘 구별되길 기대

핵심 지표: AUROC (score를 정답 라벨, 코사인 유사도를 예측 점수로 삼아
코사인 유사도가 "진짜 패러프레이즈 vs 의미변화 쌍"을 얼마나 잘 구별하는지 측정).
1.0에 가까울수록 완벽 구별, 0.5는 무작위 수준.

data_type별(법률/DART공시/리서치/뉴스)로도 나눠서 봅니다. 세은님의 회계기준
도메인과 결이 가장 가까운 건 law/DART_report 쪽일 가능성이 높습니다.

사용:
  python compare_korfinsts.py --base kure-v1 --finetuned trained_models/kure-v1_full/final
"""
import argparse

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from compare_term_distances import load_encoder


def encode_pairs(enc, sentences1, sentences2):
    A = enc.encode_corpus(sentences1)
    B = enc.encode_corpus(sentences2)
    return np.sum(A * B, axis=1)


def report(df: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")
    auc = roc_auc_score(df["score"], df["sim"])
    print(f"전체 AUROC: {auc:.4f}  (1.0=완벽 구별, 0.5=무작위)")
    for dt, sub in df.groupby("data_type"):
        try:
            auc_dt = roc_auc_score(sub["score"], sub["sim"])
        except ValueError:
            auc_dt = float("nan")   # 한 그룹에 라벨이 하나뿐인 경우
        mean1 = sub.loc[sub["score"] == 1, "sim"].mean()
        mean0 = sub.loc[sub["score"] == 0, "sim"].mean()
        print(f"  {dt.split('/')[-1]:30s} AUROC={auc_dt:.4f}  "
              f"mean_sim(score=1)={mean1:.4f}  mean_sim(score=0)={mean0:.4f}  "
              f"gap={mean1 - mean0:.4f}")
    return auc


def main(base_ref: str, finetuned_ref: str, base_spec_name: str | None):
    ds = load_dataset("nmixx-fin/korfinSTS", split="train")
    df_base = pd.DataFrame({
        "data_type": ds["data_type"], "sentence1": ds["sentence1"],
        "sentence2": ds["sentence2"], "score": ds["score"],
    })
    print(f"총 {len(df_base)}쌍 로드")

    print(f"\n[before] {base_ref} 로딩...")
    enc_before = load_encoder(base_ref, base_spec_name)
    df_before = df_base.copy()
    df_before["sim"] = encode_pairs(enc_before, df_base["sentence1"].tolist(),
                                     df_base["sentence2"].tolist())
    enc_before.free()
    auc_before = report(df_before, f"BEFORE ({base_ref})")

    print(f"\n[after] {finetuned_ref} 로딩...")
    enc_after = load_encoder(finetuned_ref, base_spec_name or base_ref)
    df_after = df_base.copy()
    df_after["sim"] = encode_pairs(enc_after, df_base["sentence1"].tolist(),
                                    df_base["sentence2"].tolist())
    enc_after.free()
    auc_after = report(df_after, f"AFTER ({finetuned_ref})")

    print(f"\n=== 종합 ===")
    print(f"전체 AUROC: {auc_before:.4f} -> {auc_after:.4f}  "
          f"({'개선' if auc_after > auc_before else '악화'} {abs(auc_after-auc_before):.4f})")

    if auc_after > auc_before:
        print("파인튜닝이 '표면은 비슷하나 함의가 다른 문장'을 구별하는 능력을")
        print("실제로 향상시켰다는 증거입니다. 회계기준 도메인 데이터로 학습했음에도")
        print("일반 금융 텍스트(뉴스/공시/법률)로 일반화되는지 확인하는 지표이기도 합니다.")
    else:
        print("주의: 회계기준 도메인 특화 학습이 오히려 일반 금융 텍스트에서의")
        print("의미변화 탐지 능력을 떨어뜨렸을 수 있습니다 (도메인 과적합 가능성).")

    out = pd.DataFrame({
        "data_type": df_base["data_type"], "sentence1": df_base["sentence1"],
        "sentence2": df_base["sentence2"], "score": df_base["score"],
        "sim_before": df_before["sim"], "sim_after": df_after["sim"],
    })
    out["delta"] = out["sim_after"] - out["sim_before"]
    out_path = "korfinsts_comparison.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="파인튜닝 전 모델 (config.py REGISTRY 키)")
    ap.add_argument("--finetuned", required=True, help="파인튜닝 후 모델 경로")
    ap.add_argument("--base-spec", default=None,
                     help="--finetuned가 로컬 경로일 때 원본 전처리 규약을 가져올 REGISTRY 키")
    a = ap.parse_args()
    main(a.base, a.finetuned, a.base_spec)