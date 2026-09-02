import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from encoders import Encoder
from term_pairs import (
    CONFUSABLE_PAIRS,
    ANTONYM_PAIRS,
    SYNONYM_PAIRS,
    UNRELATED_PAIRS,
    ACCOUNTING_ANTONYM_PAIRS,
    ACCOUNTING_ANTONYM_PAIRS_CURATED,
    ACCOUNTING_SYNONYM_PAIRS_CURATED,
)
import umap  # pip install umap-learn

# 한글 라벨이 깨지지 않도록 한글 폰트 지정 (Windows 기본 맑은 고딕)
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="임베딩 공간 시각화 (baseline vs finetuned 등 비교)")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="비교할 모델명 리스트 (config.py REGISTRY 키 기준)."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["confusable", "antonym", "synonym", "unrelated", "acct-antonym",
                "acct-antonym-curated", "acct-synonym-curated"],
        choices=["confusable", "antonym", "synonym", "unrelated", "acct-antonym",
                "acct-antonym-curated", "acct-synonym-curated"],
        help="포함할 카테고리 (기본값: 전체). 예: --categories confusable antonym"
    )
    parser.add_argument("--method", choices=["umap", "pca"], default="umap")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--output", default="embedding_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-points", action="store_true", default=True,
        help="각 점 옆에 단어 텍스트 표시 (기본값: True)"
    )
    parser.add_argument(
        "--no-label-points", dest="label_points", action="store_false",
        help="점 라벨 끄기 (카테고리 많을 때 너무 빽빽하면 사용)"
    )
    parser.add_argument(
        "--label-fontsize", type=float, default=7,
        help="점 라벨 폰트 크기 (기본값: 7)"
    )
    return parser.parse_args()


CATEGORY_MAP = {
    "confusable": CONFUSABLE_PAIRS,
    "antonym": ANTONYM_PAIRS,
    "synonym": SYNONYM_PAIRS,
    "unrelated": UNRELATED_PAIRS,
    "acct-antonym": ACCOUNTING_ANTONYM_PAIRS,
    "acct-antonym-curated": ACCOUNTING_ANTONYM_PAIRS_CURATED,
    "acct-synonym-curated": ACCOUNTING_SYNONYM_PAIRS_CURATED,
}


def collect_terms_and_labels(categories):
    terms, labels, pair_ids = [], [], []
    pair_counter = 0
    for cat_name in categories:
        for a, b in CATEGORY_MAP[cat_name]:
            terms += [a, b]
            labels += [cat_name, cat_name]
            pair_ids += [pair_counter, pair_counter]
            pair_counter += 1
    return terms, labels, pair_ids


def reduce_dim(embs, method, n_neighbors, min_dist, seed):
    if method == "umap":
        n_neighbors = min(n_neighbors, max(2, len(embs) - 1))
        reducer = umap.UMAP(metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(embs)


def embed_and_reduce(model_name, terms, method, n_neighbors, min_dist, seed):
    enc = Encoder(model_name)
    try:
        embs = enc.encode_corpus(terms)
    finally:
        enc.free()
    coords = reduce_dim(embs, method, n_neighbors, min_dist, seed)
    return coords, embs


def pair_cosine_sims(embs, pair_ids):
    pair_ids = np.array(pair_ids)
    sims = {}
    for pid in np.unique(pair_ids):
        idx = np.where(pair_ids == pid)[0]
        if len(idx) == 2:
            i, j = idx
            sims[pid] = float(np.dot(embs[i], embs[j]))
    return sims


def plot_comparison(model_names, terms, labels, pair_ids, method, n_neighbors, min_dist,
                     seed, output, label_points, label_fontsize):
    fig, axes = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 7))
    if len(model_names) == 1:
        axes = [axes]

    label_set = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_set)))
    color_map = dict(zip(label_set, colors))
    pair_ids_arr = np.array(pair_ids)

    for ax, model_name in zip(axes, model_names):
        print(f"임베딩 계산 중: {model_name} ({len(terms)}개 용어)")
        coords, embs = embed_and_reduce(model_name, terms, method, n_neighbors, min_dist, seed)
        sims = pair_cosine_sims(embs, pair_ids)

        cat_sims = {}
        for pid, sim in sims.items():
            i = np.where(pair_ids_arr == pid)[0][0]
            cat = labels[i]
            cat_sims.setdefault(cat, []).append(sim)

        # 같은 쌍끼리 연결선
        for pid in np.unique(pair_ids_arr):
            idx = np.where(pair_ids_arr == pid)[0]
            if len(idx) == 2:
                i, j = idx
                cat = labels[i]
                ax.plot(
                    [coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                    color=color_map[cat], alpha=0.35, linewidth=1.2, zorder=1,
                )

        # 점 찍기
        for lbl in label_set:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            avg_sim = np.mean(cat_sims.get(lbl, [np.nan]))
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                label=f"{lbl} (avg cos={avg_sim:.3f})",
                color=color_map[lbl], s=40, alpha=0.85, zorder=2,
            )

        # 각 점 옆에 단어 텍스트 표시
        if label_points:
            for i, term in enumerate(terms):
                ax.annotate(
                    term,
                    (coords[i, 0], coords[i, 1]),
                    fontsize=label_fontsize,
                    color=color_map[labels[i]],
                    xytext=(3, 3), textcoords="offset points",
                    zorder=3,
                )

        ax.set_title(model_name, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(f"저장 완료: {output}")


if __name__ == "__main__":
    args = parse_args()
    terms, labels, pair_ids = collect_terms_and_labels(args.categories)
    print(f"총 용어 수: {len(terms)} (카테고리: {args.categories})")
    plot_comparison(
        args.models, terms, labels, pair_ids,
        method=args.method, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        seed=args.seed, output=args.output,
        label_points=args.label_points, label_fontsize=args.label_fontsize,
    )