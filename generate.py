"""
2단계: 고정된 생성기(Gemini 2.5 Pro)로 답변 생성.

retriever만 바꾸고 생성기/프롬프트/temperature는 완전히 고정해야
관측된 차이를 retrieval 탓으로 돌릴 수 있습니다.

--gold 를 주면 정답 문서(qrels)만 넣은 upper-bound 답변을 만듭니다.
이건 나중에 Gold-Answer Agreement 계산에 씁니다.
"""
import argparse
import json

from tqdm import tqdm

from config import DATA_DIR, DEFAULT_MODELS, GEN_MODEL, RUN_DIR

SYSTEM = """당신은 한국 금융 문서 기반 질의응답 어시스턴트입니다.
반드시 아래 규칙을 지키세요.
1. 제공된 문서에 있는 내용만으로 답변합니다. 사전 지식이나 추측을 더하지 마세요.
2. 문서로 답할 수 없으면 정확히 "주어진 문서로는 답변할 수 없습니다."라고만 쓰세요.
3. 3문장 이내의 한국어 평서문으로 답하세요. 인사말이나 출처 표기는 넣지 마세요."""

TEMPLATE = """[문서]
{contexts}

[질문]
{question}

[답변]"""

ABSTAIN = "주어진 문서로는 답변할 수 없습니다"


def fmt_contexts(docs):
    return "\n\n".join(f"<문서 {i+1}>\n{d['text']}" for i, d in enumerate(docs))


def generate_for(model_name: str, use_gold: bool, top_k: int, limit: int | None):
    from gemini_client import call

    run_dir = RUN_DIR / model_name
    retrieved = json.load(open(run_dir / "retrieved.json", encoding="utf-8"))

    if use_gold:
        corpus = {json.loads(l)["_id"]: json.loads(l)
                  for l in open(DATA_DIR / "corpus.jsonl", encoding="utf-8")}

    qids = list(retrieved)[:limit] if limit else list(retrieved)
    out = {}
    for qid in tqdm(qids, desc=f"gen[{model_name}{'/gold' if use_gold else ''}]"):
        item = retrieved[qid]
        docs = ([corpus[g] for g in item["gold_ids"]] if use_gold
                else item["docs"][:top_k])
        prompt = TEMPLATE.format(contexts=fmt_contexts(docs), question=item["query"])
        answer = call(prompt, model=GEN_MODEL, system=SYSTEM, temperature=0.0)
        out[qid] = {
            "query": item["query"],
            "answer": answer,
            "contexts": [d["text"] for d in docs],
            "context_ids": [d["_id"] if use_gold else d["doc_id"] for d in docs],
            "n_gold_in_context": sum(1 for d in docs
                                     if (d["_id"] if use_gold else d["doc_id"])
                                     in item["gold_ids"]),
            "abstained": ABSTAIN in answer,
        }

    fname = "answers_gold.json" if use_gold else "answers.json"
    json.dump(out, open(run_dir / fname, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    n_ab = sum(v["abstained"] for v in out.values())
    print(f"[{model_name}] {len(out)}건 생성, 기권 {n_ab}건 ({n_ab/len(out):.1%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--gold", action="store_true", help="정답 문서만으로 upper-bound 생성")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    for m in a.models:
        generate_for(m, a.gold, a.top_k, a.limit)