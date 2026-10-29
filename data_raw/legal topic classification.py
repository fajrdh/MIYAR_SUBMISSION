"""Classify articles into governance topics and extract obligation statements.

Primary method: transparent keyword rules (each decision traceable to a keyword).
Secondary check: character n-gram TF-IDF similarity to topic descriptions, which
tolerates OCR errors. Agreement between the two is reported as a quality metric.
Outputs are candidates for human review, not legal conclusions."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.legal_text_extraction import nz

TOPICS = {
    "مجلس الإدارة ومجلس الأمناء": ["مجلس الإدارة", "مجلس الأمناء", "رئيس مجلس", "أعضاء مجلس"],
    "اللجان": ["لجنة", "لجان", "لجنتين", "لجنتان"],
    "الرقابة الداخلية": ["الرقابة الداخلية", "المراجع الداخلي", "مراجع الحسابات", "مراجعا للحسابات", "المراجعة", "الرقابة المالية"],
    "الإفصاح والشفافية": ["الإفصاح", "الشفافية", "يفصح", "الإعلان"],
    "تعارض المصالح": ["تعارض المصالح", "تعارض", "مصلحة شخصية", "مصلحة مباشرة"],
    "العضوية": ["العضوية", "عضوية", "العضو", "الأعضاء"],
    "المالية والموارد": ["المالية", "الموازنة", "الزكاة", "المحاسبية", "الأموال", "الحساب الختامي",
                         "الإيرادات", "الموارد المالية", "التقارير المالية", "الأوقاف"],
    "المكافآت": ["مكافأة", "مكافآت"],
    "الانتخابات والترشح": ["انتخاب", "انتخابات", "ترشح", "الترشيح", "التصويت", "اقتراع", "ناخب", "المرشحين", "مرشح"],
    "الجمعية العمومية": ["الجمعية العمومية"],
    "الإدارة التنفيذية والموارد البشرية": ["الرئيس التنفيذي", "المسؤول التنفيذي", "الوظائف القيادية",
                                          "الجهاز التنفيذي", "الموارد البشرية", "الموظفين", "موظفي", "موظفين"],
    "التأسيس والترخيص والتسجيل": ["التأسيس", "الترخيص", "التسجيل", "السجل"],
    "الحل والدمج": ["حل الجمعية", "حل المؤسسة", "حل المجلس", "حل", "دمج", "التصفية", "المصفي", "تصفية"],
    "الحوكمة والسياسات": ["حوكمة", "السياسات", "سياسة", "لوائح"],
    "الأهداف والأنشطة": ["الأهداف", "الأنشطة", "النشاطات"],
    "أحكام عامة وختامية": ["الجريدة الرسمية", "يعمل بها", "تسري", "التفسير", "تفسير", "تلغي", "مواءمة", "القواعد التنفيذية", "وسائل التقنية", "أحكام هذه القواعد", "أحكام هذه اللائحة"],
    "صفة النفع العام": ["النفع العام"],
    "التمثيل والشخصية الاعتبارية": ["الشخصية الاعتبارية", "ممثل", "ممثلة"],
    "الصناديق الأهلية والعائلية": ["الصندوق الأهلي", "الصندوق العائلي", "صندوق", "الصناديق"],
    "المجالس الوطنية والفرعية": ["مجلس الجمعيات", "مجلس المؤسسات", "المجالس الفرعية", "المجلس الفرعي"],
    "العلاقات والتمويل الخارجي": ["خارج المملكة", "الجهات الدولية", "الدول والمنظمات", "المنظمات الدولية", "الإعانات", "الإعانة"],
}

_AR = "\u0621-\u064A"
# clitics: و ف ب ك ل, definite article ال, and the double لل
_PREFIX = r"(?:[وفبك]?ال|ل?ل|[وفبلك])?"


_PRONOUN = r"(?:هما|هم|هن|ها|نا|كما|كم|ه)?"


def _word_regex(word: str) -> str:
    """Regex for one keyword word: optional 'ال', and 'ه' also matches 'ت' (e.g. لجنة / لجنتها)."""
    w = word[2:] if word.startswith("ال") and len(word) > 3 else word
    stem = re.escape(w[:-1]) + "(?:ه|ت)" if w.endswith("ه") else re.escape(w)
    return r"(?:ال)?" + stem


def _pattern(kw: str):
    """Match a keyword with: optional clitics before it (و ف ب ك ل، لل)، optional 'ال' on each word,
    and an attached pronoun after the last word (ها، هم، هما، ه، نا، كم). Whole words only."""
    words = nz(kw).split()
    body = r"\s+".join(_word_regex(w) for w in words[:-1] + [words[-1]])
    return re.compile(rf"(?<![{_AR}]){_PREFIX}{body}{_PRONOUN}(?![{_AR}])")


_PATTERNS = {topic: [_pattern(k) for k in kws] for topic, kws in TOPICS.items()}


def topic_hits(text: str) -> dict:
    t = nz(text)
    return {topic: sum(len(p.findall(t)) for p in pats) for topic, pats in _PATTERNS.items()}


def tag_topics(text: str) -> list:
    return [topic for topic, c in topic_hits(text).items() if c > 0]


def classify_articles(arts: pd.DataFrame) -> pd.DataFrame:
    df = arts.copy()
    hits = df["text"].map(topic_hits)
    df["topic_hits"] = hits
    df["topics"] = hits.map(lambda h: [t for t, c in h.items() if c > 0])
    df["primary_topic"] = hits.map(lambda h: max(h, key=h.get) if max(h.values()) > 0 else None)
    return df


def tfidf_topic_check(df: pd.DataFrame) -> pd.DataFrame:
    """Nearest topic prototype by char n-gram TF-IDF cosine similarity (secondary check)."""
    out = df.copy()
    names = list(TOPICS)
    protos = [nz(" ".join(TOPICS[t])) for t in names]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    mat = vec.fit_transform(out["text"].tolist() + protos)
    sims = (mat[:len(out)] @ mat[len(out):].T).toarray()
    out["tfidf_topic"] = [names[i] if sims[r].max() > 0 else None for r, i in enumerate(sims.argmax(axis=1))]
    out["tfidf_score"] = np.round(sims.max(axis=1), 4)
    labelled = out["primary_topic"].notna()
    out["agrees_with_rules"] = np.where(labelled, out["tfidf_topic"] == out["primary_topic"], False)
    return out


_REQ_PATTERNS = [
    ("prohibition", re.compile(rf"(?<![{_AR}])(?:لا يجوز|لايجوز|يحظر|يمنع)(?![{_AR}])")),
    ("obligation", re.compile(rf"(?<![{_AR}])يجب(?![{_AR}])")),
    ("condition", re.compile(rf"(?<![{_AR}])(?:بشرط|شريطة|على أن|إذا|اذا)(?![{_AR}])")),
]


def extract_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """Sentences stating an obligation, prohibition or condition. Snippets are capped at 15 words."""
    rows = []
    for _, r in df.iterrows():
        for sentence in re.split(r"[.؛\n]", r["text"]):
            s = sentence.strip()
            if not s:
                continue
            ns = nz(s)
            kind = next((k for k, pat in _REQ_PATTERNS if pat.search(ns)), None)
            if kind is None:
                continue
            rows.append({"doc_key": r["doc_key"], "article_no": r["article_no"], "article_label": r["article_label"],
                         "page_start": r["page_start"], "requirement_type": kind,
                         "topics": "; ".join(tag_topics(s)) or "; ".join(r.get("topics", []) or []),
                         "snippet_15_words": " ".join(s.split()[:15])})
    return pd.DataFrame(rows, columns=["doc_key", "article_no", "article_label", "page_start",
                                       "requirement_type", "topics", "snippet_15_words"])


# Reviewed overrides of the primary topic. Each one is justified by the article text (see reason).
# The keyword rules cannot see the word 'المجلس' alone, so these are applied after classification.
TOPIC_OVERRIDES = [
    {"doc_key": "reg", "article_no": 73, "topic": "المجالس الوطنية والفرعية",
     "reason": "نص المادة يعدّد اختصاصات «المجلس» (مجلس المؤسسات الأهلية): تمثيل المؤسسات، والتنسيق، وتقديم التوصيات؛ "
               "وهذه مهام المجلس لا أهداف الجهات، والكلمتان «الأهداف» و«الأنشطة» لم تكونا السبب الحقيقي"},
]


def apply_overrides(df):
    """Apply reviewed overrides to primary_topic and record them in a separate column."""
    out = df.copy()
    out["topic_override"] = ""
    for o in TOPIC_OVERRIDES:
        m = (out["doc_key"] == o["doc_key"]) & (out["article_no"] == o["article_no"])
        out.loc[m, "primary_topic"] = o["topic"]
        out.loc[m, "topic_override"] = o["reason"]
    return out
