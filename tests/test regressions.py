"""Regression tests for errors found while building the project.
Tests that need the source PDFs or CSV are skipped when those files are absent."""
import json

import pandas as pd
import pytest

from src import config
from src.legal_text_extraction import parse_ordinal


@pytest.mark.parametrize("text,value", [
    ("الثالئة عشرة", 13),   # OCR: ئ read as ي after normalisation
    ("التانية عشرة", 12),   # OCR: ت for ث
    ("التالتة والأريعون", 43),
    ("الأريعون", 40),       # OCR: dropped letter
])
def test_ocr_variant_ordinals(text, value):
    assert parse_ordinal(text) == value


def test_out_of_sequence_heading_is_rejected_not_renumbered():
    from src.legal_text_extraction import parse_articles
    pages = ["المادة الأولى: نص.\nالمادة الثالثة: عنوان خاطئ.\nالمادة الثانية: نص ثان."]
    arts, rej = parse_articles(pages, "reg")
    assert list(arts["article_no"]) == [1, 2]
    assert rej.iloc[0]["parsed"] == 3 and rej.iloc[0]["expected"] == 2


@pytest.mark.skipif(not (config.OUTPUTS_DIR / "run_metrics.json").exists(), reason="run the pipeline first")
def test_article_counts_match_documents():
    m = json.loads((config.OUTPUTS_DIR / "run_metrics.json").read_text(encoding="utf-8"))
    assert m["legal"]["sequence_checks"]["reg"]["found"] == config.EXPECTED_ARTICLES["reg"]
    assert m["legal"]["sequence_checks"]["gov"]["found"] == config.EXPECTED_ARTICLES["gov"]


@pytest.mark.skipif(not config.CHARITIES_CSV.exists(), reason="raw CSV not present (not redistributed)")
def test_real_csv_row_count_preserved():
    from src.data_cleaning import clean_charities, load_raw
    raw = load_raw(config.CHARITIES_CSV)
    assert len(clean_charities(raw)) == len(raw) == 2723
