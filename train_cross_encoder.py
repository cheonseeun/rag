r"""
Cross-encoder reranker 학습.

Bi-encoder(지금까지 만든 kure-v1/arctic-embed-l-ko)와 구조가 다릅니다.
질문과 문서를 각각 따로 벡터화하는 게 아니라, "질문 [SEP] 문서"를 하나로
이어붙여 모델에 통째로 넣고, 관련도 점수 하나를 직접 출력하게 학습합니다.
질문-문서 쌍마다 모델을 새로 돌려야 해서 느리지만, 훨씬 정교한 판단이 가능합니다.

그래서 실전에서는 2단계로 씁니다:
  1단계 (bi-encoder, 빠름): 전체 코퍼스에서 top-N 후보를 빠르게 추림
  2단계 (cross-encoder, 느리지만 정교함): 그 N개만 정밀 재정렬

사용:
  python train_cross_encoder.py --base-model klue/roberta-base --epochs 2
"""
import argparse
import json

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (CrossEncoderTrainer,
                                                    CrossEncoderTrainingArguments)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from datasets import Dataset

from config import DATA_DIR, ROOT


def load_dataset(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return Dataset.from_dict({
        "query": [r["query"] for r in rows],
        "doc": [r["doc"] for r in rows],
        "label": [float(r["label"]) for r in rows],
    })


def main(base_model: str, triplets_base: str, epochs: int, batch_size: int,
         lr: float, max_length: int, max_samples: int | None, grad_accum: int,
         weight_decay: float):
    train_path = DATA_DIR / f"cross_encoder_{triplets_base}_Train.jsonl"
    eval_path = DATA_DIR / f"cross_encoder_{triplets_base}_Val.jsonl"
    for p in (train_path, eval_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음 -> 먼저 build_cross_encoder_dataset.py 실행하세요")

    train_ds = load_dataset(train_path)
    eval_ds = load_dataset(eval_path)
    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        eval_ds = eval_ds.select(range(min(max_samples // 5 or 1, len(eval_ds))))
    print(f"train={len(train_ds)}  eval={len(eval_ds)}")

    # 주의: 학습 시에는 activation_fn을 지정하지 않습니다.
    # BinaryCrossEntropyLoss는 내부적으로 BCEWithLogitsLoss를 쓰는데, 이 손실함수는
    # "sigmoid 적용 전 원시 로짓"을 받아 내부에서 sigmoid를 처리하도록 설계돼 있습니다.
    # 모델 출력에 미리 sigmoid를 씌우면 sigmoid가 두 번 적용되어 gradient가 크게
    # 왜곡되고 학습이 불안정해집니다(grad_norm 폭주, eval_loss 악화).
    # sigmoid는 추론 시 점수를 0~1로 매핑할 때만 적용하면 됩니다.
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    loss = BinaryCrossEntropyLoss(model)

    out_dir = ROOT / "trained_models" / "cross_encoder_reranker" / base_model
    smoke_test = max_samples is not None
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        max_grad_norm=1.0,   # gradient 폭주 방지
        fp16=True,   # TITAN RTX(Turing)는 bf16 미지원, fp16 사용
        eval_strategy="steps" if not smoke_test else "no",
        eval_steps=500,
        save_strategy="epoch" if not smoke_test else "no",
        save_total_limit=epochs + 1,
        save_only_model=True,
        logging_steps=20,
    )

    trainer = CrossEncoderTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, loss=loss,
    )
    trainer.train()

    if smoke_test:
        print("\n스모크 테스트 완료 (디스크 절약을 위해 최종 모델은 저장하지 않았습니다).")
    else:
        final_dir = out_dir / "final"
        model.save_pretrained(str(final_dir))
        print(f"\n학습 완료 -> {final_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Dongjin-kr/ko-reranker",
                     choices=["Dongjin-kr/ko-reranker", "dragonkue/bge-reranker-v2-m3-ko",
                              "upskyy/ko-reranker-8k"],
                     help="cross-encoder로 파인튜닝할 기반 한국어 reranker 모델")
    ap.add_argument("--triplets", default="bge-m3_topk-percpos_k4")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6,
                     help="Dongjin-kr/ko-reranker 저자가 이 모델을 만들 때 쓴 값(5e-6)과 동일. "
                          "이미 두 번 학습된 모델(BAAI -> 한국어)에 세 번째 학습을 얹는 것이므로 "
                          "보수적인 값이 안전합니다")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4,
                     help="유효 배치 = batch_size * grad_accum. 저자는 유효 배치 32를 사용했으므로 "
                          "batch_size 8 x grad_accum 4 = 32로 맞추는 것을 권장")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                     help="저자 설정과 동일. 정규화로 과적합/불안정 완화")
    ap.add_argument("--max-samples", type=int, default=None,
                     help="소량 스모크 테스트용: 이 개수만큼만 잘라서 학습 (체크포인트 저장 안 함)")
    a = ap.parse_args()
    main(a.base_model, a.triplets, a.epochs, a.batch_size, a.lr, a.max_length,
          a.max_samples, a.grad_accum, a.weight_decay)