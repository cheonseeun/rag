"""
term_pairs.py의 특정 카테고리에 있는 단어쌍이 실제 학습 triplet 안에서
positive-negative로 직접 짝지어진 적이 있는지 확인합니다.

check_term_coverage.py는 "같은 문서에 둘 다 등장하는지"만 봤지만, 이 스크립트는
한 걸음 더 들어가 "실제로 모델이 이 둘을 서로 다른 것으로 대조하며 학습했는지"
(즉 하나가 positive, 다른 하나가 negative로 같은 triplet에 실제로 짝지어졌는지)를
직접 확인합니다. 이게 되어야 InfoNCE 손실이 두 표현을 실제로 밀어내는 방향으로
학습됩니다.

사용법:
    python check_triplet_pairing.py --source data/aihub_triplets_bge-m3_topk-percpos_k4_Train.jsonl --category antonym
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
    parser.add_argument("--show-examples", type=int, default=2, help="쌍마다 보여줄 실제 매칭 예시 수 (기본 2)")
    return parser.parse_args()


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    args = parse_args()
    pairs = CATEGORY_MAP[args.category]
    rows = load_rows(args.source)

    print(f"카테고리: {args.category} ({len(pairs)}쌍) / triplet row 수: {len(rows)}\n")
    print(f"{'단어 A':22s} {'단어 B':22s} {'positive-negative 직접 짝지어진 횟수':>20s}")
    print("-" * 90)

    zero_paired = 0
    examples_by_pair = {}

    for a, b in pairs:
        matches = []  # (방향, positive_text, negative_text)
        for r in rows:
            pos = r["positive"]
            negs = [n["text"] for n in r.get("negatives", [])]
            # A가 positive이고 B가 그 triplet의 negative 중 하나로 등장
            if a in pos and any(b in n for n in negs):
                matched_neg = next(n for n in negs if b in n)
                matches.append(("A(pos) vs B(neg)", pos, matched_neg))
            # 반대 방향: B가 positive이고 A가 negative로 등장
            if b in pos and any(a in n for n in negs):
                matched_neg = next(n for n in negs if a in n)
                matches.append(("B(pos) vs A(neg)", pos, matched_neg))

        count = len(matches)
        if count == 0:
            zero_paired += 1
        print(f"{a:22s} {b:22s} {count:20d}")
        examples_by_pair[(a, b)] = matches

    print(f"\n실제로 positive-negative로 한 번도 안 짝지어진 쌍: {zero_paired} / {len(pairs)}")

    if args.show_examples > 0:
        print("\n" + "=" * 90)
        print("실제 매칭 예시 (앞부분 60자만 표시)")
        print("=" * 90)
        for (a, b), matches in examples_by_pair.items():
            if not matches:
                continue
            print(f"\n[{a} / {b}] 총 {len(matches)}건, {min(args.show_examples, len(matches))}건 예시:")
            for direction, pos, neg in matches[: args.show_examples]:
                print(f"  방향: {direction}")
                print(f"    positive: {pos[:60].strip()}...")
                print(f"    negative: {neg[:60].strip()}...")