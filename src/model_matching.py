"""Reproducible TF-IDF baseline for matching governance articles to executive-regulation articles.

Example:
    python -m src.model_matching --input data_clean/legal_articles_full.csv --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TWINS = [
    (8, 47), (10, 49), (79, 107), (13, 52), (6, 45), (39, 69),
    (37, 67), (31, 61), (11, 50), (14, 53), (84, 98), (77, 94),
    (12, 51), (87, 112), (15, 54), (40, 70), (4, 44), (32, 63),
    (33, 64), (74, 91), (81, 96), (38, 68), (9, 48), (76, 93),
    (7, 46), (86, 111),
]


def load_articles(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"doc_key", "article_no", "text"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    governance = data[data["doc_key"].eq("gov")]
    regulation = data[data["doc_key"].eq("reg")]
    if len(data) != 176 or len(governance) != 47 or len(regulation) != 129:
        raise ValueError(
            "Expected 176 rows (47 gov + 129 reg); got "
            f"{len(data)} ({len(governance)} gov + {len(regulation)} reg)."
        )
    if data["text"].isna().any():
        raise ValueError("Column 'text' contains missing values.")
    return data


def match_articles(data: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    governance = data[data["doc_key"].eq("gov")].reset_index(drop=True)
    regulation = data[data["doc_key"].eq("reg")].reset_index(drop=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(
        pd.concat([governance["text"], regulation["text"]], ignore_index=True).tolist()
    )
    scores = (matrix[: len(governance)] @ matrix[len(governance) :].T).toarray()
    twins = {n: f"{a}/{b}" for a, b in TWINS for n in (a, b)}
    results: list[dict[str, object]] = []
    for i, gov in governance.iterrows():
        seen: set[str] = set()
        rank = 0
        for j in np.argsort(-scores[i], kind="stable"):
            article_no = int(regulation.iloc[j]["article_no"])
            group = twins.get(article_no, str(article_no))
            if group in seen:
                continue
            seen.add(group)
            rank += 1
            results.append(
                {
                    "gov_article": int(gov["article_no"]),
                    "rank": rank,
                    "reg_group": group,
                    "score": round(float(scores[i, j]), 4),
                }
            )
            if rank >= top_k:
                break
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Cleaned articles CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    data = load_articles(args.input)
    results = match_articles(data, top_k=args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "tfidf_matches.csv"
    results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "input": str(args.input),
        "articles_total": int(len(data)),
        "governance_articles": int((data["doc_key"] == "gov").sum()),
        "regulation_articles": int((data["doc_key"] == "reg").sum()),
        "top_k": int(args.top_k),
        "match_rows": int(len(results)),
        "governance_articles_with_matches": int(results["gov_article"].nunique()),
        "max_similarity": round(float(results["score"].max()), 4),
        "mean_similarity": round(float(results["score"].mean()), 4),
        "median_similarity": round(float(results["score"].median()), 4),
        "note": "Similarity is lexical ranking, not legal correctness or validated accuracy.",
    }
    summary_path = args.output_dir / "tfidf_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
