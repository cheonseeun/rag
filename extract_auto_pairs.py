"""
학습 데이터에서 회계 도메인 반의어/유의어 후보를 자동 추출해서 JSON으로 저장합니다.
결과는 전부 미검수(reviewed=false) 상태로 저장되며, 시각화 등에 쓰려면 JSON을 열어
좋은 쌍만 "reviewed": true로 바꿔주세요 (나쁜 쌍은 항목 통째로 삭제해도 됩니다).

사용법:
    python extract_auto_pairs.py --source data/aihub_triplets_bge-m3_topk-percpos_k4_Train.jsonl
"""
import argparse
import os
from term_pairs import (
    mine_accounting_antonym_pairs_from_data,
    mine_accounting_synonym_pairs_from_data,
    save_auto_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="triplet jsonl 경로")
    parser.add_argument("--n-antonym", type=int, default=20)
    parser.add_argument("--n-synonym", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="data/auto_pairs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    antonym_pairs = mine_accounting_antonym_pairs_from_data(args.source, n=args.n_antonym, seed=args.seed)
    synonym_pairs = mine_accounting_synonym_pairs_from_data(args.source, n=args.n_synonym, seed=args.seed)

    print(f"반의어 후보 {len(antonym_pairs)}쌍:")
    for a, b in antonym_pairs:
        print(f"  {a}  /  {b}")

    print(f"\n유의어 후보 {len(synonym_pairs)}쌍:")
    for a, b in synonym_pairs:
        print(f"  {a}  /  {b}")

    save_auto_pairs(antonym_pairs, "antonym", os.path.join(args.output_dir, "acct_antonym_auto.json"))
    save_auto_pairs(synonym_pairs, "synonym", os.path.join(args.output_dir, "acct_synonym_auto.json"))