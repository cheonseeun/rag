"""
term_pairs.py의 특정 카테고리에 있는 단어들이 실제 학습 triplet 데이터
(positive/negatives 전체 텍스트)에 몇 번 등장하는지 확인합니다.

사용법:
    python check_term_coverage.py --source data/aihub_triplets_bge-m3_topk-percpos_k4_Train.jsonl --category antonym
"""
import argparse
import json
import term_pairs

CATEGORY_MAP = {
    "confusable": term_pairs.CONFUSABLE_PAIRS,
    "antonym": term_pairs.ANTONYM_PAIRS,
    "synonym": term_pairs.SYNONYM_PAIRS,
    "unrelated": term_pairs.UNRELATED_PAIRS,
    "acct-antonym": term_pairs.ACCOUNTING_ANTONYM_PAIRS,
    "acct-antonym-curated": term_pairs.ACCOUNTING_ANTONYM_PAIRS_CURATED,
    "acct-synonym-curated": term_pairs.ACCOUNTING_SYNONYM_PAIRS_CURATED,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", required=True, help="triplet jsonl 경로(들)")
    parser.add_argument("--category", required=True, choices=list(CATEGORY_MAP.keys()))
    return parser.parse_args()


def load_all_text(paths):
    """positive/negatives 전체 텍스트를 하나의 큰 문자열로 합침 (부분 문자열 매칭용)."""
    chunks = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                chunks.append(r["positive"])
                for neg in r.get("negatives", []):
                    chunks.append(neg["text"])
    return chunks


if __name__ == "__main__":
    args = parse_args()
    pairs = CATEGORY_MAP[args.category]
    docs = load_all_text(args.source)
    full_text = "\n".join(docs)

    print(f"카테고리: {args.category} ({len(pairs)}쌍) / 문서 수: {len(docs)}\n")
    print(f"{'단어 A':20s} {'A 등장 문서수':>12s}   {'단어 B':20s} {'B 등장 문서수':>12s}   {'둘 다 등장 문서수':>14s}")
    print("-" * 100)

    zero_both = 0
    for a, b in pairs:
        count_a = sum(1 for d in docs if a in d)
        count_b = sum(1 for d in docs if b in d)
        count_both = sum(1 for d in docs if a in d and b in d)
        if count_a == 0 and count_b == 0:
            zero_both += 1
        print(f"{a:20s} {count_a:12d}   {b:20s} {count_b:12d}   {count_both:14d}")

    print(f"\n둘 다 전혀 등장하지 않은 쌍: {zero_both} / {len(pairs)}")