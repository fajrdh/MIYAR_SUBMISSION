"""اختبارات أساسية: صحة الدوال الصغيرة، وسلامة المخرجات إن وُجدت.

تعمل الاختبارات دون بيانات المصدر (بيانات مصغّرة مُنشأة داخل الاختبار).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import OUTPUTS_DIR
from src.data_cleaning import clean_charities, find_header_row, load_raw
from src.legal_text_extraction import nz, parse_articles, parse_ordinal
from src.legal_topic_classification import extract_requirements, tag_topics


@pytest.mark.parametrize("text,value", [
    ("الأولى", 1), ("الثالثة والعشرون", 23), ("الحادية عشرة", 11),
    ("الحادية والعشرون", 21), ("الأولى بعد المائة", 101), ("الخامسة والأربعون", 45),
    ("كلام عادي", None),
])
def test_parse_ordinal(text, value):
    assert parse_ordinal(text) == value


def test_nz_normalizes_forms():
    assert nz("المادة الأولى") == nz("الماده الاولي")
    assert nz("مِنْ ٢٠٢٢") == "من 2022"


def test_parse_articles_strict_sequence():
    pages = ["المادة الأولى: يجب على الجمعية تعيين مدير.\nسطر ثاني من المادة.",
             "المادة الثالثة: هذا عنوان يكسر التسلسل.\nالمادة الثانية: نص آخر يكفي."]
    arts, rej = parse_articles(pages, "reg")
    assert list(arts["article_no"]) == [1, 2]
    assert len(rej) == 1 and rej.iloc[0]["parsed"] == 3
    # العنوان المرفوض يبقى ضمن نص المادة السابقة، لذا تمتد الصفحة إلى 2
    assert arts.iloc[0]["page_end"] == 2
    assert "هذا عنوان يكسر" in arts.iloc[0]["text"]


def test_topic_tagging():
    hits = tag_topics("يجب على مجلس الإدارة تشكيل لجنتين إحداهما للمراجعة")
    assert "مجلس الإدارة ومجلس الأمناء" in hits
    assert "اللجان" in hits


def test_requirement_types():
    df = pd.DataFrame([{"doc_key": "gov", "article_no": 1, "article_label": "المادة الأولى",
                        "page_start": 1, "text": "لا يجوز لمجلس الإدارة التصرف في الأموال. "
                        "يجب على المجلس إعداد الموازنة التقديرية للجمعية."}])
    req = extract_requirements(df)
    assert list(req["requirement_type"]) == ["prohibition", "obligation"]
    assert req["snippet_15_words"].str.split().str.len().max() <= 15


def test_header_detection_and_clean_preserves_rows(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "Charities - Dashboard,,,\n"
        "رقم التسجيل,اسم الجمعية,نوع الجهة,المنطقة\n"
        "1,الجمعية الخيرية لتحفيظ القرآن الكريم بمحافظة…,جمعية أهلية,منطقة الرياض\n"
        "1,مؤسسة تجريبية,مؤسسة أهلية,منطقة مكة المكرمة\n"
        "2,مؤسسة تجريبية,مؤسسة أهلية,منطقة مكة المكرمة\n",
        encoding="utf-8-sig")
    assert find_header_row(path) == 1
    raw = load_raw(path)
    clean = clean_charities(raw)
    assert len(clean) == len(raw) == 3
    assert clean["name_truncated"].sum() == 1
    assert clean["reg_number_shared"].sum() == 2
    assert clean["exact_duplicate_group"].sum() == 2


@pytest.mark.skipif(not (OUTPUTS_DIR / "run_metrics.json").exists(),
                    reason="يتطلب تشغيل run_pipeline.py أولاً")
def test_outputs_consistent():
    index = pd.read_csv(OUTPUTS_DIR / "legal_articles_index.csv")
    assert index["article_no"].min() == 1
    assert index.groupby("doc_key")["article_no"].apply(lambda s: s.is_unique).all()
