"""
모델별 query/doc 전처리 규약을 흡수하는 통합 인코더 래퍼.

여기가 이 실험에서 제일 버그 나기 쉬운 곳입니다.
e5에 prefix를 안 붙이거나, Qwen3 문서 쪽에 instruct를 붙이거나,
jina의 task LoRA를 안 켜면 모델이 억울하게 낮은 점수를 받습니다.
"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import REGISTRY, TASK_INSTRUCTION, EncoderSpec

_DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}


class Encoder:
    def __init__(self, name: str, spec: EncoderSpec | None = None):
        self.name = name
        self.spec = spec or REGISTRY[name]
        s = self.spec

        # transformers 최신 버전은 dtype, 구버전은 torch_dtype을 받습니다.
        tokenizer_kwargs = {}
        if s.padding_side:
            tokenizer_kwargs["padding_side"] = s.padding_side

        def _build(dtype_key: str):
            return SentenceTransformer(
                s.hf_id,
                trust_remote_code=s.trust_remote_code,
                model_kwargs={dtype_key: _DTYPE[s.dtype], **s.st_kwargs},
                tokenizer_kwargs=tokenizer_kwargs or None,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        try:
            self.model = _build("dtype")
        except (TypeError, ValueError):
            self.model = _build("torch_dtype")
        self.model.max_seq_length = s.max_seq_length
        self.model.eval()

    # ---------- 텍스트 가공 ----------
    def _fmt_query(self, q: str) -> str:
        k = self.spec.kind
        if k == "prefix":
            return self.spec.query_prefix + q
        if k == "instruct":
            return f"Instruct: {TASK_INSTRUCTION}\nQuery: {q}"
        return q

    def _fmt_doc(self, d: str) -> str:
        # instruct / jina / plain 계열은 문서에 아무것도 붙이지 않습니다.
        return self.spec.doc_prefix + d if self.spec.kind == "prefix" else d

    # ---------- 인코딩 ----------
    @torch.inference_mode()
    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        kw = dict(
            batch_size=self.spec.batch_size,
            normalize_embeddings=True,      # cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        if self.spec.kind == "jina":
            kw["task"] = "retrieval.query" if is_query else "retrieval.passage"
        emb = self.model.encode(texts, **kw)
        return emb.astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self._fmt_query(q) for q in queries], True)

    def encode_corpus(self, docs: list[str]) -> np.ndarray:
        return self._encode([self._fmt_doc(d) for d in docs], False)

    def free(self):
        del self.model
        torch.cuda.empty_cache()


def sanity_check(name: str):
    """모델이 제대로 로드/포맷되는지 3문장으로 즉석 확인."""
    enc = Encoder(name)
    q = ["신용스프레드가 확대되면 회사채 시장에 어떤 영향이 있나요?"]
    d = ["신용스프레드가 확대되면 회사채 발행 비용이 상승하여 기업의 자금 조달이 어려워진다.",
         "여신전문금융회사의 레버리지 배율 규제는 자기자본 대비 총자산 기준으로 적용된다.",
         "오늘 서울 날씨는 맑고 기온은 영상 12도입니다."]
    sim = enc.encode_queries(q) @ enc.encode_corpus(d).T
    print(f"[{name}] dim={enc.model.get_sentence_embedding_dimension()} "
          f"sims={np.round(sim[0], 4).tolist()}")
    assert sim[0].argmax() == 0, "관련 문서가 1등이 아님 -> prefix/pooling 설정 의심"
    enc.free()


if __name__ == "__main__":
    import sys
    failed = []
    for n in (sys.argv[1:] or ["bge-m3"]):
        try:
            sanity_check(n)
        except Exception as e:
            failed.append(n)
            print(f"[{n}] FAILED: {type(e).__name__}: {e}")
    if failed:
        print(f"\n실패한 모델: {failed}  -> 이 모델들은 빼고 진행하거나 개별 해결하세요.")
    else:
        print("\n전체 통과. run_retrieval.py 로 진행하세요.")