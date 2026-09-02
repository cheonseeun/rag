"""
AI-Hub Training/Validation 문서가 실제로 독립적인지 확인합니다.

의심되는 시나리오: Validation 문서가 Training 문서의 부분집합이거나,
내용이 거의 동일한 문서(같은 조문을 다른 파일로 중복 배포)일 수 있습니다.
doc_id만 비교하면 안 되는 이유: 원천 데이터 인덱싱이 split별로 독립적이라
(TS_*, VS_* 각각 자체 번호 체계) ID 형식은 겹치지 않을 수 있지만, 그 안의
'실제 텍스트 내용'이 같은 문서일 가능성은 별개로 확인해야 합니다.
"""
import json
from collections import Counter

from config import DATA_DIR


def load_docs(triplets_path):
    """triplet 파일에서 고유 문서(doc_id -> text) 추출."""
    docs = {}
    for line in open(triplets_path, encoding="utf-8"):
        r = json.loads(line)
        docs[r["positive_id"]] = r["positive"]
        for n in r["negatives"]:
            docs[n["doc_id"]] = n["text"]
    return docs


def main():
    train_path = DATA_DIR / "aihub_triplets_bge-m3_topk-percpos_k4_Training.jsonl"
    val_path = DATA_DIR / "aihub_triplets_bge-m3_topk-percpos_k4_Validation.jsonl"

    train_docs = load_docs(train_path)
    val_docs = load_docs(val_path)
    print(f"Training 고유 문서: {len(train_docs)}개")
    print(f"Validation 고유 문서: {len(val_docs)}개")

    # 1) doc_id 자체가 겹치는지 (이름 체계가 같다면 직접 비교 가능)
    train_ids = set(train_docs)
    val_ids = set(val_docs)
    id_overlap = train_ids & val_ids
    print(f"\n[1] doc_id 완전 일치: {len(id_overlap)}개 "
          f"({len(id_overlap)/len(val_ids)*100:.1f}% of Validation)")
    if id_overlap:
        print(f"  예시: {list(id_overlap)[:5]}")

    # 2) 텍스트 내용 자체가 100% 동일한 문서가 있는지 (id는 다르지만 내용이 같은 경우 대비)
    train_text_set = set(train_docs.values())
    val_text_set = set(val_docs.values())
    text_overlap = train_text_set & val_text_set
    print(f"\n[2] 텍스트 완전 일치(다른 id라도): {len(text_overlap)}개 "
          f"({len(text_overlap)/len(val_text_set)*100:.1f}% of Validation)")

    # 3) 텍스트 앞부분(제목/도입부) 유사도로 "거의 같은 문서" 대략 탐지
    #    (완전全문 비교는 아니지만, 같은 조문이 살짝 다른 chunk로 잘렸을 가능성 체크)
    def prefix(t, n=60):
        return t[:n].strip()

    train_prefixes = Counter(prefix(t) for t in train_docs.values())
    val_prefixes = [prefix(t) for t in val_docs.values()]
    near_dup = sum(1 for p in val_prefixes if p in train_prefixes)
    print(f"\n[3] 앞 60자 기준 근접 중복 의심: {near_dup}개 "
          f"({near_dup/len(val_prefixes)*100:.1f}% of Validation)")

    # 4) main_category / sub_category 분포가 서로 비슷한지 (완전 다른 주제로
    #    나뉜 게 아니라 같은 카테고리 안에서 나뉜 것인지 확인용 참고 정보)
    def load_categories(path):
        cats = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            cats.append((r["main_category"], r["sub_category"]))
        return Counter(cats)

    print("\n[4] 카테고리 분포 비교 (참고용)")
    train_cats = load_categories(train_path)
    val_cats = load_categories(val_path)
    print("  Training 상위 5:", train_cats.most_common(5))
    print("  Validation 상위 5:", val_cats.most_common(5))

    print("\n=== 결론 ===")
    if id_overlap or text_overlap:
        print("경고: Training과 Validation에 겹치는 문서가 있습니다. 진짜 held-out이 아닙니다.")
    elif near_dup > len(val_prefixes) * 0.1:
        print(f"주의: 완전 중복은 아니지만 {near_dup}개 문서가 Training과 도입부가 같습니다. "
              "같은 조문이 다르게 청크됐을 가능성이 있어 완전 독립은 아닐 수 있습니다.")
    else:
        print("Training과 Validation 문서 내용이 겹치지 않습니다. 독립적인 held-out으로 보입니다.")


if __name__ == "__main__":
    main()