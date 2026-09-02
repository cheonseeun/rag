"""
실험 설정: 모델 레지스트리 + 경로.

각 임베딩 모델은 query/document 전처리 규약이 다릅니다.
이걸 틀리면 성능이 5~15pp씩 날아가므로 여기서 중앙 관리합니다.

- plain    : prefix 없음 (bge-m3, KURE-v1)
- prefix   : "query: " / "passage: " (multilingual-e5 계열 non-instruct)
- instruct : "Instruct: {task}\nQuery: {q}" (query만), doc은 원문
             -> Qwen3-Embedding, multilingual-e5-*-instruct, llama-embed-nemotron-8b
                셋 다 동일 템플릿을 씁니다.
- jina     : SentenceTransformer.encode(task="retrieval.query"/"retrieval.passage")
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RUN_DIR = ROOT / "runs"
CACHE_DIR = ROOT / "cache"
for _d in (DATA_DIR, RUN_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HF_DATASET = "mteb/AutoRAGRetrieval"
DOMAIN = "finance"          # 문서/쿼리 _id 접두사
TOP_K = 5                   # 생성기에 넣을 문서 수
EVAL_KS = (1, 3, 5, 10)

# 검색용 task instruction (instruct 계열 모델에만 적용)
TASK_INSTRUCTION = (
    "Given a question about Korean financial documents, "
    "retrieve the passage that answers the question"
)

GEN_MODEL = "gemini-2.5-pro"
JUDGE_MODEL = "gemini-2.5-pro"   # self-preference가 걱정되면 다른 모델로 교체


@dataclass
class EncoderSpec:
    hf_id: str
    kind: str                      # plain | prefix | instruct | jina
    max_seq_length: int = 1024
    batch_size: int = 8
    dtype: str = "float32"         # float32 | bfloat16
    trust_remote_code: bool = False
    padding_side: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    st_kwargs: dict = field(default_factory=dict)


REGISTRY: dict[str, EncoderSpec] = {
    "bge-m3": EncoderSpec(
        hf_id="BAAI/bge-m3", kind="plain",
        max_seq_length=1024, batch_size=8,
    ),
    "kure-v1": EncoderSpec(
        hf_id="nlpai-lab/KURE-v1", kind="plain",
        max_seq_length=1024, batch_size=8,
    ),
    # ME5-large: 512 토큰 하드 리밋. 긴 한국어 청크는 잘립니다(논문에 명시할 것).
    "me5-large": EncoderSpec(
        hf_id="intfloat/multilingual-e5-large", kind="prefix",
        max_seq_length=512, batch_size=16,
        query_prefix="query: ", doc_prefix="passage: ",
    ),
    "me5-large-instruct": EncoderSpec(
        hf_id="intfloat/multilingual-e5-large-instruct", kind="instruct",
        max_seq_length=512, batch_size=16,
    ),
    "jina-v3": EncoderSpec(
        hf_id="jinaai/jina-embeddings-v3", kind="jina",
        max_seq_length=1024, batch_size=8, trust_remote_code=True,
    ),
    # TITAN RTX(24GB) 기준: 0.6B 권장, 4B는 bf16+batch 2로 겨우, 8B는 어려움
    "qwen3-embed-0.6b": EncoderSpec(
        hf_id="Qwen/Qwen3-Embedding-0.6B", kind="instruct",
        max_seq_length=1024, batch_size=8, dtype="bfloat16", padding_side="left",
    ),
    "qwen3-embed-4b": EncoderSpec(
        hf_id="Qwen/Qwen3-Embedding-4B", kind="instruct",
        max_seq_length=1024, batch_size=2, dtype="bfloat16", padding_side="left",
    ),
    "nemotron-8b": EncoderSpec(
        hf_id="nvidia/llama-embed-nemotron-8b", kind="instruct",
        max_seq_length=1024, batch_size=1, dtype="bfloat16",
        trust_remote_code=True, padding_side="left",
        st_kwargs={"attn_implementation": "eager"},   # flash_attn 있으면 flash_attention_2
    ),
    # arctic-embed-l-v2.0을 한국어로 추가학습한 모델. 제작자 보고에 따르면
    # 여러 도메인에서 bge-m3보다 우수. 문서 길이가 1300 토큰(~2500자)을
    # 넘으면 성능 저하 가능 (모델 카드 권장사항).
    "arctic-embed-l-ko": EncoderSpec(
        hf_id="dragonkue/snowflake-arctic-embed-l-v2.0-ko", kind="prefix",
        max_seq_length=1300, batch_size=8,
        query_prefix="query: ", doc_prefix="",
    ),

    "arctic-embed-l-ko-batch32": EncoderSpec(
        hf_id="trained_models/arctic-embed-l-ko_full_batch32/final", kind="prefix",
        max_seq_length=1300, batch_size=8,
        query_prefix="query: ", doc_prefix="",
    ),
    "arctic-embed-l-ko-inbatch-batch32": EncoderSpec(
        hf_id="trained_models/arctic-embed-l-ko_full_inbatch_batch32/final", kind="prefix",
        max_seq_length=1300, batch_size=8,
        query_prefix="query: ", doc_prefix="",
    ),
    "arctic-embed-l-ko-maxseq768-batch32": EncoderSpec(
        hf_id="trained_models/arctic-embed-l-ko_full_maxseq768_batch32/final", kind="prefix",
        max_seq_length=1300, batch_size=8,
        query_prefix="query: ", doc_prefix="",
    ),
    "arctic-embed-l-ko-maxseq500-batch32": EncoderSpec(
        hf_id="trained_models/arctic-embed-l-ko_full_maxseq500_batch32/final", kind="prefix",
        max_seq_length=1300, batch_size=8,
        query_prefix="query: ", doc_prefix="",
    ),

    # 파인튜닝 결과물 (dense-embedding teacher로 마이닝한 negative로 학습).
    # kure-v1과 동일한 전처리 규약(prefix 없음)을 그대로 씁니다.
    "kure-v1-finetuned-dense": EncoderSpec(
        hf_id="trained_models/kure-v1_full_dense_embedding/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    # BM25 teacher로 마이닝한 negative로 학습한 결과물.
    "kure-v1-finetuned-bm25": EncoderSpec(
        hf_id="trained_models/kure-v1_full_bm25/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    # arctic-embed-l-ko 파인튜닝 결과물. 원본과 동일한 prefix 규약(query: 접두사)을 그대로 씁니다.
    "arctic-embed-l-ko-finetuned": EncoderSpec(
        hf_id="trained_models/arctic-embed-l-ko_full_v2/final", kind="prefix",
        max_seq_length=1300, batch_size=8,
        query_prefix="query: ", doc_prefix="",
    ),
    # 재분할(문서 단위 8:1:1) 데이터로 학습한 결과물 (bge-m3 teacher, topk-percpos).
    "kure-v1-resplit-topkpercpos": EncoderSpec(
        hf_id="trained_models/kure-v1_full_dense_embedding_v2/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    # 재분할 데이터로 학습한 결과물 (bge-m3 teacher, meghwani 이중조건).
    "kure-v1-resplit-meghwani": EncoderSpec(
        hf_id="trained_models/kure-v1_full_meghwani_v2/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    # 재분할 데이터로 학습한 결과물 (bge-m3 teacher, hybrid: meghwani 우선 + topk-percpos 보충).
    "kure-v1-resplit-hybrid": EncoderSpec(
        hf_id="trained_models/kure-v1_full_hybrid/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    "kure-v1-inbatch-only-batch32": EncoderSpec(
        hf_id="trained_models/kure-v1_full_in_batch/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    "kure-v1-percpos-batch32": EncoderSpec(
        hf_id="trained_models/kure-v1_full_batch32/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    "kure-v1-meghwani-batch32": EncoderSpec(
        hf_id="trained_models/kure-v1_full_meghwani_batch32/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    "kure-v1-hybrid-batch32": EncoderSpec(
        hf_id="trained_models/kure-v1_full_hybrid_batch32/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
    "kure-v1-marginpos-batch32": EncoderSpec(
        hf_id="trained_models/kure-v1_full_marginpos_batch32/final", kind="plain",
        max_seq_length=1024, batch_size=32,
    ),
}

DEFAULT_MODELS = [
    "bge-m3", "kure-v1", "qwen3-embed-4b", "nemotron-8b",
]