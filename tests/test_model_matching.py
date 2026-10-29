import pandas as pd
import pytest

from src.model_matching import load_articles, match_articles


@pytest.fixture
def article_frame():
    rows = []
    for i in range(1, 48):
        rows.append({"doc_key": "gov", "article_no": i, "text": f"قاعدة حوكمة رقم {i} تتعلق بالإدارة والشفافية"})
    for i in range(1, 130):
        rows.append({"doc_key": "reg", "article_no": i, "text": f"مادة لائحة رقم {i} تتعلق بالجمعيات والإدارة والالتزام"})
    return pd.DataFrame(rows)


def test_load_articles_validates_expected_document_counts(tmp_path, article_frame):
    path = tmp_path / "articles.csv"
    article_frame.to_csv(path, index=False, encoding="utf-8-sig")
    loaded = load_articles(path)
    assert len(loaded) == 176
    assert (loaded["doc_key"] == "gov").sum() == 47
    assert (loaded["doc_key"] == "reg").sum() == 129


def test_load_articles_rejects_wrong_counts(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"doc_key": "gov", "article_no": 1, "text": "نص"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected 176 rows"):
        load_articles(path)


def test_match_articles_returns_top_three_for_each_governance_article(article_frame):
    results = match_articles(article_frame, top_k=3)
    assert len(results) == 47 * 3
    assert results["gov_article"].nunique() == 47
    assert results.groupby("gov_article")["rank"].apply(list).apply(lambda ranks: ranks == [1, 2, 3]).all()
    assert results["score"].between(0, 1).all()
