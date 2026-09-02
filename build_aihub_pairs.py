r"""
AI-Hub "기업 회계처리 기준 데이터" -> 임베딩 파인튜닝용 (query, positive) 쌍 생성.

구조:
  01.원천데이터/TS_{대분류}_{소분류}/*.json
      document.metadata.document_name  예: "K-IFRS_기타_0003"
      document.content[].content_text  문서 원문 (여러 조각을 이어붙임)
  02.라벨링데이터/TL_{대분류}_{소분류}/*_QA{n}.json
      metadata.document_name           예: "K-IFRS_기타_0002_QA1"  (뒤에 _QA번호)
      qa_content.turns[].question/answer  멀티턴 QA

매칭: 파일명은 인코딩이 깨져 있으므로 JSON 내부 document_name 필드를 기준으로
      TL의 "_QA\d+" 접미사를 떼어 TS와 매칭합니다.

멀티턴 중 첫 번째 턴만 사용합니다. 후속 턴은 이전 턴을 전제로 한 질문이라
그 자체로는 독립된 검색 쿼리가 아니기 때문입니다.

출력: data/aihub_pairs.jsonl
  {"query": ..., "positive": ..., "doc_id": ..., "main_category": ..., 
   "sub_category": ..., "split": "Training"|"Validation"}
"""
import argparse
import glob
import json
import re
from pathlib import Path

from config import ROOT

RAW_ROOT = ROOT / "27.기업_회계처리_기준_데이터" / "3.개방데이터" / "1.데이터"
OUT_PATH = ROOT / "data" / "aihub_pairs.jsonl"

QA_SUFFIX = re.compile(r"_QA\d+$")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(name: str) -> str:
    """TS/TL 간 표기 불일치(예: '일반 기업 회계 기준' vs '일반기업회계기준') 흡수."""
    return re.sub(r"\s+", "", name)


SPLIT_PREFIX = {"Training": ("TS_", "TL_"), "Validation": ("VS_", "VL_")}


def build_source_index(split: str) -> dict[str, dict]:
    """정규화된 document_name -> {"text":.., "main":.., "sub":..} 인덱스."""
    src_prefix, _ = SPLIT_PREFIX[split]
    pattern = str(RAW_ROOT / split / "01.원천데이터" / f"{src_prefix}*" / "*.json")
    index = {}
    for p in glob.glob(pattern):
        try:
            d = load_json(p)["document"]
        except Exception as e:
            print(f"  [경고] 원천 파싱 실패: {p} ({e})")
            continue
        name = d["metadata"]["document_name"]
        text = "\n\n".join(c["content_text"] for c in d.get("content", []))
        cat = d["metadata"].get("topic_category", {})
        index[_norm(name)] = {
            "text": text,
            "main": cat.get("main_category", ""),
            "sub": cat.get("sub_category", ""),
        }
    return index


def build_pairs(split: str, source_index: dict, first_turn_only: bool):
    _, label_prefix = SPLIT_PREFIX[split]
    pattern = str(RAW_ROOT / split / "02.라벨링데이터" / f"{label_prefix}*" / "*.json")
    pairs, missing = [], []
    for p in glob.glob(pattern):
        try:
            d = load_json(p)
        except Exception as e:
            print(f"  [경고] 라벨 파싱 실패: {p} ({e})")
            continue
        label_name = d["metadata"]["document_name"]
        src_name = _norm(QA_SUFFIX.sub("", label_name))
        src = source_index.get(src_name)
        if src is None:
            missing.append(src_name)
            continue

        turns = d.get("qa_content", {}).get("turns", [])
        use_turns = turns[:1] if first_turn_only else turns
        for i, t in enumerate(use_turns):
            q = (t.get("question") or "").strip()
            if not q:
                continue
            pairs.append({
                "query": q,
                "answer": (t.get("answer") or "").strip(),
                "positive": src["text"],
                "doc_id": src_name,
                "main_category": src["main"],
                "sub_category": src["sub"],
                "turn_index": i,
                "split": split,
            })
    return pairs, missing


def main(first_turn_only: bool):
    all_pairs = []
    for split in ("Training", "Validation"):
        print(f"[{split}] 원천 문서 인덱싱 중...")
        src_idx = build_source_index(split)
        print(f"  원천 문서 {len(src_idx)}개")

        pairs, missing = build_pairs(split, src_idx, first_turn_only)
        all_pairs.extend(pairs)
        print(f"  QA 쌍 {len(pairs)}개 생성, 매칭 실패 {len(missing)}건")
        if missing:
            print(f"  매칭 실패 예시: {missing[:3]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 카테고리 분포 출력 (하드 네거티브 마이닝 설계에 참고)
    from collections import Counter
    subs = Counter(r["sub_category"] for r in all_pairs)
    mains = Counter(r["main_category"] for r in all_pairs)
    print(f"\n총 {len(all_pairs)}쌍 -> {OUT_PATH}")
    print("대분류 분포:", dict(mains))
    print("소분류 분포:", dict(subs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-turns", action="store_true",
                     help="멀티턴 전체 사용 (기본은 첫 턴만, 비권장이나 규모 필요시)")
    a = ap.parse_args()
    main(first_turn_only=not a.all_turns)