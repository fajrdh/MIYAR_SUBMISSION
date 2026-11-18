"""مِعيار للمتابعة: توليد بنود قائمة المتابعة التنظيمية من مخرجات التحليل.

يقرأ جمل الالتزام والحظر والشرط (data_clean/requirement_snippets.csv) والنص الكامل
للمواد (data_clean/legal_articles_full.csv)، ويكتب:

- followup_seed.js — بنود القائمة التي تقرؤها صفحة «مِعيار للمتابعة» في index.html.
- outputs/followup_checklist_template.csv — القالب نفسه بصيغة جدول لمن يتابع في Excel.

التشغيل من جذر المستودع:
    python -m src.followup_checklist
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import CLEAN_ARTICLES, DATA_PROCESSED, DOC_LABELS, OUTPUTS_DIR, ROOT

SNIPPETS_CSV = DATA_PROCESSED / "requirement_snippets.csv"
SEED_JS = ROOT / "followup_seed.js"
TEMPLATE_CSV = OUTPUTS_DIR / "followup_checklist_template.csv"

TYPE_LABELS = {"obligation": "التزام", "prohibition": "حظر", "condition": "شرط"}
SEED_VERSION = 1


def _split_topics(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


def article_key(doc_key: str, article_no) -> str:
    return f"{doc_key}-{int(article_no)}"


def build_checklist(snippets: pd.DataFrame, articles: pd.DataFrame) -> dict:
    """يرجع بنود القائمة ونصوص المواد المرتبطة بها (كل مادة مرة واحدة)."""
    required = {"indicator_id", "doc_key", "article_no", "article_label", "page_start",
                "requirement_type", "topics", "snippet_15_words"}
    missing = required - set(snippets.columns)
    if missing:
        raise ValueError(f"Missing columns in requirement snippets: {sorted(missing)}")
    if snippets["indicator_id"].duplicated().any():
        raise ValueError("Duplicate indicator_id values in requirement snippets")
    unknown = set(snippets["requirement_type"]) - set(TYPE_LABELS)
    if unknown:
        raise ValueError(f"Unknown requirement types: {sorted(unknown)}")

    text_col = "text_raw" if "text_raw" in articles.columns else "text"
    texts = {article_key(r.doc_key, r.article_no): str(getattr(r, text_col)).strip()
             for r in articles.itertuples(index=False)}

    items, used = [], {}
    # قواعد الحوكمة أولًا ثم اللائحة، وداخل كل وثيقة حسب رقم المادة
    ordered = snippets.assign(_doc=snippets["doc_key"].map({"gov": 0, "reg": 1}).fillna(2))
    ordered = ordered.sort_values(["_doc", "article_no", "indicator_id"]).drop(columns="_doc")
    for r in ordered.itertuples(index=False):
        key = article_key(r.doc_key, r.article_no)
        if key not in texts:
            raise ValueError(f"No article text for {r.indicator_id} ({key})")
        used[key] = texts[key]
        items.append({
            "id": r.indicator_id,
            "doc": r.doc_key,
            "docLabel": DOC_LABELS.get(r.doc_key, r.doc_key),
            "articleNo": int(r.article_no),
            "articleLabel": str(r.article_label),
            "page": int(r.page_start),
            "type": r.requirement_type,
            "typeLabel": TYPE_LABELS[r.requirement_type],
            "topics": _split_topics(r.topics),
            "snippet": str(r.snippet_15_words).strip(),
            "articleKey": key,
        })
    return {"version": SEED_VERSION, "source": "requirement_snippets.csv", "items": items, "articles": used}


def write_seed_js(checklist: dict, path: Path = SEED_JS) -> Path:
    body = json.dumps(checklist, ensure_ascii=False, separators=(",", ":"))
    path.write_text(
        "/* مولَّد آليًا من src/followup_checklist.py — لا تعدّليه يدويًا. */\n"
        f"window.MIYAR_FOLLOWUP_SEED={body};\n",
        encoding="utf-8",
    )
    return path


def write_template_csv(checklist: dict, path: Path = TEMPLATE_CSV) -> Path:
    rows = [{
        "رقم المتطلب": i["id"],
        "الوثيقة": i["docLabel"],
        "المادة": i["articleNo"],
        "الصفحة": i["page"],
        "نوع المتطلب": i["typeLabel"],
        "الموضوعات": "؛ ".join(i["topics"]),
        "مقتطف": i["snippet"],
        "الحالة": "لم يبدأ",
        "المسؤول": "",
        "تاريخ الاستحقاق": "",
        "الأدلة": "",
        "ملاحظات": "",
    } for i in checklist["items"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> dict:
    snippets = pd.read_csv(SNIPPETS_CSV, encoding="utf-8-sig")
    articles = pd.read_csv(CLEAN_ARTICLES, encoding="utf-8-sig")
    checklist = build_checklist(snippets, articles)
    write_seed_js(checklist)
    write_template_csv(checklist)
    print(f"followup: {len(checklist['items'])} requirements, {len(checklist['articles'])} articles "
          f"-> {SEED_JS.name}, {TEMPLATE_CSV.relative_to(ROOT)}")
    return checklist


if __name__ == "__main__":
    main()
