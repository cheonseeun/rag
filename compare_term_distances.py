r"""
파인튜닝 전/후 임베딩 거리 비교: "유사 단어 간 벡터 거리 수치화" 실험.

측정 그룹 4개 (term_pairs.py):
  confusable : 표면적으로 비슷하나 개념이 다른 쌍 -> 파인튜닝 후 유사도 하락 기대
  antonym    : 정반대 의미인데 표면이 비슷한 금융 거래 용어(매수/매도 등)
               -> confusable보다 더 엄격하게 하락 기대 (헷갈리면 결론이 반대가 되는 위험한 케이스)
  synonym    : 실질적으로 같은 개념인 쌍 (대조군)   -> 파인튜닝 후에도 유사도 유지 기대
  unrelated  : 무관한 쌍 (기준선)                  -> 항상 낮게 유지

사용 예:
  python compare_term_distances.py --base kure-v1 --finetuned trained_models/kure-v1_full/final
"""
import argparse

import numpy as np
import pandas as pd

from config import REGISTRY, EncoderSpec
from encoders import Encoder
from term_pairs import ANTONYM_PAIRS, CONFUSABLE_PAIRS, SYNONYM_PAIRS, UNRELATED_PAIRS


def load_encoder(model_ref: str, base_spec_name: str | None) -> Encoder:
    """model_ref가 REGISTRY 키(예: kure-v1)면 그대로, 로컬 경로(파인튜닝 결과물)면
    base_spec_name의 kind/prefix 규약을 재사용해 hf_id만 로컬 경로로 바꾼 스펙을 만듭니다.
    (같은 인코더를 파인튜닝했으므로 전처리 규약 자체는 바뀌지 않습니다.)
    """
    if model_ref in REGISTRY:
        return Encoder(model_ref)
    if not base_spec_name:
        raise ValueError("로컬 경로를 쓰려면 --base-spec으로 원본 인코더 이름을 지정하세요 "
                          "(예: --base-spec kure-v1)")
    base = REGISTRY[base_spec_name]
    local_spec = EncoderSpec(
        hf_id=model_ref, kind=base.kind, max_seq_length=base.max_seq_length,
        batch_size=base.batch_size, dtype=base.dtype,
        trust_remote_code=base.trust_remote_code, padding_side=base.padding_side,
        query_prefix=base.query_prefix, doc_prefix=base.doc_prefix,
        st_kwargs=base.st_kwargs,
    )
    return Encoder(model_ref, spec=local_spec)


def measure(enc: Encoder, pairs: list[tuple[str, str]]) -> np.ndarray:
    """각 쌍의 코사인 유사도. 짧은 용어이므로 query/doc 구분 없이 둘 다
    encode_corpus 경로로 인코딩합니다(대칭적인 용어-용어 비교이므로)."""
    a_texts = [p[0] for p in pairs]
    b_texts = [p[1] for p in pairs]
    A = enc.encode_corpus(a_texts)
    B = enc.encode_corpus(b_texts)
    return np.sum(A * B, axis=1)   # 이미 정규화되어 있으므로 내적 = 코사인 유사도


def run_group(enc_before: Encoder, enc_after: Encoder, pairs, group_name: str) -> pd.DataFrame:
    before = measure(enc_before, pairs)
    after = measure(enc_after, pairs)
    return pd.DataFrame({
        "group": group_name,
        "term_a": [p[0] for p in pairs],
        "term_b": [p[1] for p in pairs],
        "sim_before": before,
        "sim_after": after,
        "delta": after - before,   # 음수 = 파인튜닝 후 더 멀어짐 (confusable/antonym엔 좋은 신호)
    })


def main(base_ref: str, finetuned_ref: str, base_spec_name: str | None):
    print(f"[before] {base_ref} 로딩...")
    enc_before = load_encoder(base_ref, base_spec_name)
    print(f"[after]  {finetuned_ref} 로딩...")
    enc_after = load_encoder(finetuned_ref, base_spec_name or base_ref)

    dfs = [
        run_group(enc_before, enc_after, CONFUSABLE_PAIRS, "confusable"),
        run_group(enc_before, enc_after, ANTONYM_PAIRS, "antonym"),
        run_group(enc_before, enc_after, SYNONYM_PAIRS, "synonym"),
        run_group(enc_before, enc_after, UNRELATED_PAIRS, "unrelated"),
    ]
    df = pd.concat(dfs, ignore_index=True)

    pd.set_option("display.width", 140, "display.max_colwidth", 20)
    print("\n" + df.round(4).to_string(index=False))

    print("\n=== 그룹별 요약 ===")
    summary = df.groupby("group")[["sim_before", "sim_after", "delta"]].mean().round(4)
    print(summary)

    print("\n=== 해석 ===")
    conf_delta = summary.loc["confusable", "delta"]
    ant_delta = summary.loc["antonym", "delta"] if "antonym" in summary.index else None
    syn_delta = summary.loc["synonym", "delta"]
    if conf_delta < 0 and abs(syn_delta) < abs(conf_delta) * 0.5:
        print(f"바람직한 패턴: confusable 유사도가 {conf_delta:+.4f} 하락했고, "
              f"synonym은 {syn_delta:+.4f}로 거의 유지됨.")
        print("-> 파인튜닝이 '무차별적으로 밀어낸 것'이 아니라, 실제로 헷갈리는 개념을")
        print("   선택적으로 구별하도록 학습됐다는 증거입니다.")
    elif conf_delta < 0 and syn_delta < -0.03:
        print(f"주의: confusable({conf_delta:+.4f})뿐 아니라 synonym({syn_delta:+.4f})도")
        print("   같이 떨어졌습니다. 모델이 선택적 구별이 아니라 전반적으로 임베딩을")
        print("   흩어놓았을 가능성이 있습니다. 학습률/epoch을 낮추는 조정을 고려하세요.")
    else:
        print(f"confusable delta={conf_delta:+.4f} -> 기대와 다른 방향이거나 효과가 약합니다.")
        print("   개별 쌍(delta 컬럼)을 직접 살펴보고, 어떤 쌍에서 효과가 없었는지 확인하세요.")

    if ant_delta is not None:
        print(f"\nantonym(매수/매도류 반의어) delta={ant_delta:+.4f}")
        if ant_delta < conf_delta:
            print("-> confusable보다도 더 크게 멀어짐. 반의어처럼 결과가 반대로 뒤집히는")
            print("   위험한 케이스를 특히 잘 구별하게 됐다는 신호입니다 (바람직함).")
        elif ant_delta >= 0:
            print("-> 경고: 반의어 쌍의 유사도가 오히려 유지되거나 증가했습니다. 실거래")
            print("   맥락(매수/매도 등)에 대한 구별력은 이번 파인튜닝으로 개선되지 않은")
            print("   것으로 보입니다 (학습 데이터가 회계기준 조문 위주라 실거래 반의어")
            print("   패턴을 충분히 다루지 못했을 가능성).")
        else:
            print("-> 하락했지만 confusable보다는 폭이 작습니다. 어느 정도 개선은 있으나")
            print("   반의어 특화 학습 데이터를 보강하면 더 나아질 여지가 있습니다.")

    out_path = "term_distance_comparison.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                     help="파인튜닝 전 모델. config.py REGISTRY 키 (예: kure-v1)")
    ap.add_argument("--finetuned", required=True,
                     help="파인튜닝 후 모델 경로 (예: trained_models/kure-v1_full/final)")
    ap.add_argument("--base-spec", default=None,
                     help="--finetuned가 로컬 경로일 때, 원본 전처리 규약을 가져올 "
                          "REGISTRY 키. 생략 시 --base 값을 그대로 사용")
    a = ap.parse_args()
    main(a.base, a.finetuned, a.base_spec)