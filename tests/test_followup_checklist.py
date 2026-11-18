import json

import pandas as pd
import pytest

from src.config import CLEAN_ARTICLES
from src.followup_checklist import SNIPPETS_CSV, build_checklist, write_seed_js, write_template_csv


def _snippets(**overrides):
    row = {"indicator_id": "gov-0001", "doc_key": "gov", "article_no": 5, "article_label": "المادة الخامسه",
           "page_start": 3, "requirement_type": "obligation", "topics": "اللجان; الجمعية العمومية",
           "snippet_15_words": "يجب على الجمعية"}
    row.update(overrides)
    return pd.DataFrame([row])


ARTICLES = pd.DataFrame([{"doc_key": "gov", "article_no": 5, "text": "نص منظف", "text_raw": "نصّ المادة الأصلي"}])


def test_build_checklist_links_requirement_to_article_text():
    out = build_checklist(_snippets(), ARTICLES)
    item = out["items"][0]
    assert item["typeLabel"] == "التزام"
    assert item["topics"] == ["اللجان", "الجمعية العمومية"]
    assert out["articles"][item["articleKey"]] == "نصّ المادة الأصلي"


def test_build_checklist_rejects_unknown_type_and_missing_article():
    with pytest.raises(ValueError, match="Unknown requirement types"):
        build_checklist(_snippets(requirement_type="advice"), ARTICLES)
    with pytest.raises(ValueError, match="No article text"):
        build_checklist(_snippets(article_no=99), ARTICLES)


def test_real_data_gives_124_unique_requirements(tmp_path):
    snippets = pd.read_csv(SNIPPETS_CSV, encoding="utf-8-sig")
    articles = pd.read_csv(CLEAN_ARTICLES, encoding="utf-8-sig")
    out = build_checklist(snippets, articles)
    ids = [i["id"] for i in out["items"]]
    assert len(ids) == 124 == len(set(ids))
    assert {i["doc"] for i in out["items"]} == {"gov", "reg"}
    assert all(i["articleKey"] in out["articles"] for i in out["items"])

    js = write_seed_js(out, tmp_path / "seed.js").read_text(encoding="utf-8")
    payload = json.loads(js.split("window.MIYAR_FOLLOWUP_SEED=", 1)[1].rstrip().rstrip(";"))
    assert len(payload["items"]) == 124

    csv = pd.read_csv(write_template_csv(out, tmp_path / "t.csv"), encoding="utf-8-sig")
    assert len(csv) == 124 and (csv["الحالة"] == "لم يبدأ").all()
