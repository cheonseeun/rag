r"""
  Full-FT : KURE-v1, bge-m3        (작고 한국어 특화 -> 전체 파라미터 학습)
  LoRA    : qwen3-embed-4b/0.6b, nemotron-8b  (크고 SOTA -> 저랭크 어댑터만 학습)

사용:
  python train.py --encoder kure-v1 --mode full
  python train.py --encoder qwen3-embed-4b --mode lora
"""
import argparse

import torch
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                    SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from build_training_dataset import load_train_eval
from config import REGISTRY, ROOT

OUT_ROOT = ROOT / "trained_models"


def load_model(encoder_name: str) -> SentenceTransformer:
    spec = REGISTRY[encoder_name]
    dtype = torch.bfloat16 if spec.dtype == "bfloat16" else torch.float32
    model = SentenceTransformer(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        model_kwargs={"dtype": dtype, **spec.st_kwargs},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.max_seq_length = spec.max_seq_length
    return model


def apply_lora(model: SentenceTransformer, r: int = 16, alpha: int = 32):
    """모델의 트랜스포머 백본에 LoRA 어댑터를 부착합니다 (NV-Retriever 논문 설정과 동일: r=16, alpha=32)."""
    from peft import LoraConfig, get_peft_model

    auto_model = model[0].auto_model
    target_modules = set()
    for name, module in auto_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "query", "key", "value", "dense"):
                target_modules.add(leaf)
    if not target_modules:
        raise RuntimeError("LoRA target_modules를 자동으로 찾지 못했습니다. "
                            "모델 구조를 확인하고 target_modules를 직접 지정하세요.")
    print(f"LoRA target_modules: {sorted(target_modules)}")

    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=list(target_modules),
                       lora_dropout=0.05, bias="none")
    peft_model = get_peft_model(auto_model, cfg)
    peft_model.print_trainable_parameters()
    model[0].auto_model = peft_model
    return model


def main(encoder_name: str, mode: str, epochs: int, batch_size: int,
          lr: float, n_negatives: int, triplets_base: str, max_samples: int | None,
          grad_accum: int, max_seq_length: int | None, grad_checkpointing: bool,
          resume_from: str | None, hardness_mode: str | None, hardness_strength: float,
          train_split_name: str, eval_split_name: str, weight_decay: float):
    model = load_model(encoder_name)
    if resume_from:
        print(f"체크포인트에서 가중치를 불러옵니다: {resume_from}")
        model = SentenceTransformer(resume_from,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
    if max_seq_length:
        model.max_seq_length = max_seq_length   # 학습 시에만 짧게 잘라 메모리 절약
    if mode == "lora":
        model = apply_lora(model)

    train_ds, eval_ds = load_train_eval(
        encoder_name, triplets_base=triplets_base, n_negatives=n_negatives,
        train_split=train_split_name, eval_split=eval_split_name)
    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        eval_ds = eval_ds.select(range(min(max_samples // 5 or 1, len(eval_ds))))
    print(f"train={len(train_ds)}  eval={len(eval_ds)}")

    loss = MultipleNegativesRankingLoss(
        model, hardness_mode=hardness_mode, hardness_strength=hardness_strength,
    )
    if hardness_mode:
        print(f"hardness weighting 적용: mode={hardness_mode} strength={hardness_strength}")

    out_dir = OUT_ROOT / f"{encoder_name}_{mode}"
    # TITAN RTX(Turing, sm_75)는 bf16 텐서코어를 지원하지 않으므로 fp16 혼합정밀도를 씁니다.
    # bf16은 Ampere(sm_80) 이상에서만 의미가 있습니다.
    use_fp16 = torch.cuda.is_available()
    smoke_test = max_samples is not None
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=grad_checkpointing,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        # 같은 배치 안에 같은 anchor의 positive가 두 번 들어가지 않게 함
        # (in-batch negative를 오염시키는 것 방지)
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        fp16=use_fp16,
        eval_strategy="steps" if not smoke_test else "no",
        eval_steps=500,
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        logging_steps=20,
    )

    trainer = SentenceTransformerTrainer(
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
    ap.add_argument("--encoder", required=True,
                     help="config.py REGISTRY 키 (bge-m3, kure-v1, qwen3-embed-4b, nemotron-8b 등)")
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                     help="AdamW weight decay (L2 정규화). 기본값 0.01. "
                          "과적합이 심하면 0.05~0.1까지 올려보고, "
                          "0으로 주면 정규화 없이 학습(기존 동작과 동일)")
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--triplets", default="bge-m3_topk-percpos_k4",
                     help="data/aihub_triplets_{이 값}_Training.jsonl 과 "
                          "_Validation.jsonl 두 파일을 각각 train/eval로 사용")
    ap.add_argument("--max-samples", type=int, default=None,
                     help="소량 스모크 테스트용: 이 개수만큼만 잘라서 학습")
    ap.add_argument("--grad-accum", type=int, default=1,
                     help="배치를 나눠 누적 (메모리 절약, 유효 배치는 batch_size*grad_accum)")
    ap.add_argument("--max-seq-length", type=int, default=None,
                     help="학습 시 시퀀스 길이 상한 (미지정 시 config.py 기본값 사용, "
                          "메모리 부족하면 384~512로 낮추세요)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                     help="활성값을 다시 계산해 메모리를 아끼는 대신 속도가 느려짐")
    ap.add_argument("--resume-from", default=None,
                     help="이전 체크포인트 경로(예: trained_models/kure-v1_full/checkpoint-1000)에서 "
                          "가중치를 불러와 이어서 학습 (warm-start, optimizer state는 초기화됨)")
    ap.add_argument("--hardness-mode", default=None,
                     choices=[None, "in_batch_negatives", "hard_negatives", "all_negatives"],
                     help="negative의 '어려움 정도'를 손실에 직접 반영 (Lan et al. 2025 / "
                          "EmbeddingGemma 기법). 미지정 시 기본 MultipleNegativesRankingLoss와 동일. "
                          "'hard_negatives': 마이닝한 hard negative만 가중, "
                          "'in_batch_negatives': in-batch negative만 가중, "
                          "'all_negatives': 둘 다 가중")
    ap.add_argument("--hardness-strength", type=float, default=5.0,
                     help="hardness 가중치 강도(alpha). --hardness-mode 지정 시에만 적용. "
                          "EmbeddingGemma 논문은 hard_negatives에 5.0을 사용")
    ap.add_argument("--train-split", default="Training",
                     help="triplet 파일명의 train 쪽 접미사. AI-Hub 원본 분할이면 'Training', "
                          "재분할한 3단 분할이면 'Train'")
    ap.add_argument("--eval-split", default="Validation",
                     help="triplet 파일명의 eval 쪽 접미사. AI-Hub 원본 분할이면 'Validation', "
                          "재분할한 3단 분할이면 'Val'")
    a = ap.parse_args()
    main(a.encoder, a.mode, a.epochs, a.batch_size, a.lr, a.n_negatives,
          a.triplets, a.max_samples, a.grad_accum, a.max_seq_length,
          a.gradient_checkpointing, a.resume_from, a.hardness_mode, a.hardness_strength,
          a.train_split, a.eval_split, a.weight_decay)