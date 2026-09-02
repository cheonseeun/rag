"""
AutoRAGRetrieval 전처리.

_id 포맷: "{domain} - {source.pdf} - {chunk_idx}"  (예: "finance - ...pdf - 12")
따라서 도메인 필터는 _id prefix로 합니다.

출력:
  data/corpus.jsonl   {"_id", "text", "title", "domain"}
  data/queries.jsonl  {"_id", "text", "domain"}
  data/qrels.json     {query_id: {doc_id: score}}
"""
import argparse
import json
import re
from collections import Counter

from datasets import load_dataset

from config import DATA_DIR, DOMAIN, HF_DATASET


def _domain_of(_id: str) -> str:
    """도메인 추출.

    corpus  _id: "finance - 파일명.pdf - 12"   -> finance
    queries _id: "0_finance - ..."             -> finance  (앞의 인덱스 제거)
    """
    head = re.split(r"\s*-\s*", str(_id).strip(), maxsplit=1)[0].strip().lower()
    return re.sub(r"^\d+_", "", head)


def _load_subset(name: str):
    try:
        return load_dataset(HF_DATASET, name, split="test")
    except Exception:
        # mteb 리포는 subset 구성이 바뀌는 경우가 있어 fallback
        return load_dataset(HF_DATASET, name)["test"]


def main(domain: str, corpus_scope: str):
    corpus = _load_subset("corpus")
    queries = _load_subset("queries")
    qrels = _load_subset("qrels")

    doc_domains = Counter(_domain_of(r) for r in corpus["_id"])
    q_domains = Counter(_domain_of(r) for r in queries["_id"])
    print("[corpus domains]", dict(doc_domains))
    print("[query  domains]", dict(q_domains))
    if domain not in q_domains:
        raise SystemExit(
            f"'{domain}' 도메인이 없습니다. 위 목록에서 골라 --domain 으로 넘기세요."
        )

    # 쿼리는 항상 해당 도메인만 평가
    q_rows = [
        {"_id": r["_id"], "text": r["text"], "domain": _domain_of(r["_id"])}
        for r in queries
        if _domain_of(r["_id"]) == domain
    ]
    q_ids = {r["_id"] for r in q_rows}

    # qrels 컬럼명은 리포에 따라 query-id/corpus-id 또는 query_id/corpus_id
    cols = qrels.column_names
    qcol = "query-id" if "query-id" in cols else "query_id"
    ccol = "corpus-id" if "corpus-id" in cols else "corpus_id"
    qrels_map: dict[str, dict[str, int]] = {}
    gold_ids = set()
    for r in qrels:
        qid, cid, s = str(r[qcol]), str(r[ccol]), int(r.get("score", 1))
        if qid in q_ids and s > 0:
            qrels_map.setdefault(qid, {})[cid] = s
            gold_ids.add(cid)

    # 검색 풀 범위
    #   all    : 720개 전체 (타 도메인이 hard-ish distractor 역할) <- 권장
    #   domain : 해당 도메인 문서만 (쉬워서 지표가 포화됨, ablation용)
    if corpus_scope == "all":
        c_rows = [
            {"_id": r["_id"], "text": r["text"], "title": r.get("title") or "",
             "domain": _domain_of(r["_id"])}
            for r in corpus
        ]
    else:
        c_rows = [
            {"_id": r["_id"], "text": r["text"], "title": r.get("title") or "",
             "domain": _domain_of(r["_id"])}
            for r in corpus
            if _domain_of(r["_id"]) == domain
        ]

    # 정답 문서가 풀에서 빠지면 Recall 상한이 1 미만이 됩니다. 반드시 확인.
    have = {r["_id"] for r in c_rows}
    missing = gold_ids - have
    assert not missing, f"정답 문서 {len(missing)}개가 검색 풀에 없습니다: {list(missing)[:3]}"

    # 빈 텍스트 / 공백 정리 (min_document_length=8인 짧은 청크 존재)
    for r in c_rows:
        r["text"] = re.sub(r"\s+", " ", r["text"]).strip()
    c_rows = [r for r in c_rows if len(r["text"]) >= 8 or r["_id"] in gold_ids]

    _dump(DATA_DIR / "corpus.jsonl", c_rows)
    _dump(DATA_DIR / "queries.jsonl", q_rows)
    (DATA_DIR / "qrels.json").write_text(
        json.dumps(qrels_map, ensure_ascii=False, indent=2), encoding="utf-8")

    lens = [len(r["text"]) for r in c_rows]
    print(f"\n[{domain}] queries={len(q_rows)}  corpus={len(c_rows)} "
          f"(scope={corpus_scope})  qrels={len(qrels_map)}")
    print(f"doc chars: mean={sum(lens)/len(lens):.0f} max={max(lens)}")
    print(f"-> {DATA_DIR}")


def _dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default=DOMAIN)
    p.add_argument("--corpus-scope", default="all", choices=["all", "domain"])
    a = p.parse_args()
    main(a.domain, a.corpus_scope)