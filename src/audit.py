"""Extraction audit: per-article checks and a human-review queue.
An article is never marked 'verified' automatically; the audit only flags what needs a person to look."""
import pandas as pd

from src.legal_text_extraction import arabic_share

SHORT_BODY_CHARS = 40          # bodies shorter than this are suspicious (possible truncation)
MAX_PAGE_SPAN = 3              # articles spanning more pages than this are suspicious (possible merge)


def article_audit(arts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in arts.iterrows():
        text = str(r["text"]) if pd.notna(r["text"]) else ""
        reasons = []
        if bool(r["header_fuzzy"]):
            reasons.append("ترويسة طابقها OCR تقريبياً")
        if len(text.strip()) < SHORT_BODY_CHARS:
            reasons.append("نص قصير جداً")
        if int(r["page_end"]) - int(r["page_start"]) + 1 > MAX_PAGE_SPAN:
            reasons.append("يمتد على صفحات كثيرة")
        if arabic_share(text) < 85:
            reasons.append("نسبة حروف عربية منخفضة")
        rows.append({"doc_key": r["doc_key"], "article_no": int(r["article_no"]), "article_label": r["article_label"],
                     "page_start": int(r["page_start"]), "page_end": int(r["page_end"]),
                     "header_fuzzy": bool(r["header_fuzzy"]), "body_chars": len(text),
                     "needs_human_review": bool(reasons), "review_reasons": "؛ ".join(reasons)})
    return pd.DataFrame(rows)


def sequence_summary(arts: pd.DataFrame, rejected: pd.DataFrame, expected: dict) -> pd.DataFrame:
    rows = []
    for key, n_expected in expected.items():
        sub = arts[arts["doc_key"] == key].sort_values("article_no")
        nums = sub["article_no"].tolist()
        found = set(nums)
        pages_ordered = bool(sub["page_start"].is_monotonic_increasing)
        rows.append({"doc_key": key, "expected": n_expected, "extracted": len(found),
                     "missing": ", ".join(str(x) for x in sorted(set(range(1, n_expected + 1)) - found)) or "لا يوجد",
                     "duplicates": len(nums) - len(found),
                     "rejected_headers": int((rejected["doc_key"] == key).sum()) if len(rejected) else 0,
                     "pages_in_order": pages_ordered,
                     "fuzzy_headers": int(sub["header_fuzzy"].sum()),
                     "flagged_for_review": int(0)})
    return pd.DataFrame(rows)


def classification_review_sample(arts: pd.DataFrame, per_group: int = 10, seed: int = 7) -> pd.DataFrame:
    """Sample for HUMAN review. Human-label columns are left empty on purpose: no reference
    labels exist yet, so no precision/recall/F1 can be computed from this file."""
    from src.legal_topic_classification import TOPICS
    df = arts.copy()
    df["agree"] = df["primary_topic"].notna() & (df["tfidf_topic"] == df["primary_topic"])
    def reason(r):
        if pd.isna(r["primary_topic"]):
            return "لا توجد كلمة مفتاحية مطابقة (قد يكون النص عاماً أو OCR أفسد الكلمة)"
        if len(r["topics"]) > 1 and not r["agree"]:
            return "عدة مواضيع بالكلمات؛ التشابه يرجّح موضوعاً آخر"
        if not r["agree"]:
            return "الكلمات والتشابه يرجحان موضوعين مختلفين"
        return "اتفاق آلي (يحتاج مراجعة بشرية أيضاً)"
    df["reason_auto"] = df.apply(reason, axis=1)
    parts = []
    for (doc, agree), g in df.groupby(["doc_key", "agree"]):
        parts.append(g.sample(min(per_group, len(g)), random_state=seed))
    s = pd.concat(parts)
    out = pd.DataFrame({
        "doc_key": s["doc_key"], "article_no": s["article_no"], "page_start": s["page_start"],
        "predicted_primary_by_rules": s["primary_topic"], "all_topics_by_rules": s["topics"].map(lambda t: "; ".join(t)),
        "predicted_by_tfidf": s["tfidf_topic"], "agreement": s["agree"], "auto_reason": s["reason_auto"],
        "text_first_20_words": s["text"].map(lambda t: " ".join(str(t).split()[:20])),
        "human_expected_topic": "", "human_notes": "", "reviewer": ""})
    return out.sort_values(["doc_key", "article_no"]).reset_index(drop=True)
