"""
aihub_triplets_*.jsonl -> sentence-transformers 학습용 Dataset 변환.

MultipleNegativesRankingLoss는 각 row가 (anchor, positive, negative_1, ...,
negative_N) 컬럼을 갖는 형태를 기대합니다. negative가 N_NEGATIVES보다 적은
row는 마지막 negative를 복제해 채웁니다 (NV-Retriever 논문이 발견한 바,
동일 negative를 중복시키는 것이 오히려 그 negative의 중요도를 높여
학습에 유리하다는 관찰과도 일치합니다).

n_negatives=0이면 명시적 하드 네거티브를 전혀 쓰지 않고 anchor-positive만
반환합니다 (순수 in-batch negative 실험용).

인코더별 query/doc prefix 규약(encoders.py의 EncoderSpec)을 그대로 적용해서,
학습 시 입력 포맷이 평가·추론 때와 어긋나지 않도록 합니다.
"""
import json
import random

from datasets import Dataset

from config import DATA_DIR, REGISTRY, TASK_INSTRUCTION


def _fmt_query(text: str, spec) -> str:
    if spec.kind == "prefix":
        return spec.query_prefix + text
    if spec.kind == "instruct":
        return f"Instruct: {TASK_INSTRUCTION}\nQuery: {text}"
    return text


def _fmt_doc(text: str, spec) -> str:
    return spec.doc_prefix + text if spec.kind == "prefix" else text


def build_dataset(triplets_path, encoder_name: str, n_negatives: int = 4,
                   seed: int = 0) -> Dataset:
    spec = REGISTRY[encoder_name]
    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]
    rng = random.Random(seed)

    records = {"anchor": [], "positive": []}
    for i in range(n_negatives):
        records[f"negative_{i+1}"] = []

    for r in rows:
        if n_negatives == 0:
            # 순수 in-batch negative 실험용: 명시적 하드 네거티브를 전혀 쓰지 않음
            records["anchor"].append(_fmt_query(r["query"], spec))
            records["positive"].append(_fmt_doc(r["positive"], spec))
            continue

        negs = [n["text"] for n in r["negatives"]]
        if not negs:
            continue
        while len(negs) < n_negatives:
            negs.append(rng.choice(negs))   # 부족하면 기존 negative 중복 (논문 관찰과 일치)
        negs = negs[:n_negatives]

        records["anchor"].append(_fmt_query(r["query"], spec))
        records["positive"].append(_fmt_doc(r["positive"], spec))
        for i, n in enumerate(negs):
            records[f"negative_{i+1}"].append(_fmt_doc(n, spec))

    return Dataset.from_dict(records)


def load_train_eval(encoder_name: str, triplets_base: str = "bge-m3_topk-percpos_k4",
                     n_negatives: int = 4, seed: int = 0,
                     train_split: str = "Training", eval_split: str = "Validation"):
    """train/eval에 쓸 triplet 파일을 각각 독립적으로 마이닝된 파일에서 가져옵니다
    (같은 문서 풀에서 무작위로 떼어낸 게 아니라, 처음부터 분리된 데이터라
    진짜 held-out 검증이 됩니다).

    train_split/eval_split: 기존 AI-Hub 분할이면 "Training"/"Validation",
                             재분할한 3단 분할이면 "Train"/"Val" (또는 "Test") 사용.

    필요 파일:
      data/aihub_triplets_{triplets_base}_{train_split}.jsonl
      data/aihub_triplets_{triplets_base}_{eval_split}.jsonl
    """
    train_path = DATA_DIR / f"aihub_triplets_{triplets_base}_{train_split}.jsonl"
    eval_path = DATA_DIR / f"aihub_triplets_{triplets_base}_{eval_split}.jsonl"
    for p, name in ((train_path, train_split), (eval_path, eval_split)):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} 없음 -> 먼저 mine_hard_negatives.py --split {name} 로 생성하세요")

    train_ds = build_dataset(train_path, encoder_name, n_negatives, seed)
    eval_ds = build_dataset(eval_path, encoder_name, n_negatives, seed)
    return train_ds, eval_ds


if __name__ == "__main__":
    tr, ev = load_train_eval("bge-m3")
    print(f"train={len(tr)}  eval={len(ev)}")
    print(tr[0])