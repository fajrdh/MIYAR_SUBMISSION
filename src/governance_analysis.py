"""Cross-document comparison (path 1) and the governance-score feasibility check (path 2)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.legal_text_extraction import nz

ARABIC_STOPWORDS = {nz(w) for w in [
    "من", "على", "في", "الى", "إلى", "عن", "او", "أو", "و", "ما", "لا", "ان", "أن", "التي", "الذي", "الذين",
    "هذه", "هذا", "ذلك", "مع", "يجب", "يجوز", "لها", "له", "لهم", "بها", "به", "بعد", "قبل", "حسب",
    "وفق", "وفقا", "وفقاً", "كل", "ايا", "اي", "تكون", "يكون", "يتم", "كما", "بين", "عند", "اذا", "إذا",
    "خلال", "ضمن", "حيث", "ثم", "اما", "أما", "بما", "كان", "تكن", "ليس", "لدى", "لدي", "المادة", "القواعد",
    "اللائحة", "الاساسية", "الأساسية", "المركز", "النظام", "الجمعية", "المؤسسة", "الجمعيات", "المؤسسات"]}


def word_tokens(text: str):
    return [w for w in re.findall(r"[\u0621-\u064A]{3,}", nz(text)) if w not in ARABIC_STOPWORDS]


def top_terms(arts: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Mean TF-IDF weight of terms per document (fitted on all articles of both documents)."""
    vec = TfidfVectorizer(tokenizer=word_tokens, lowercase=False, token_pattern=None)
    mat = vec.fit_transform(arts["text"].tolist())
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for doc, idx in arts.groupby("doc_key").indices.items():
        means = np.asarray(mat[idx].mean(axis=0)).ravel()
        for rank, j in enumerate(np.argsort(-means)[:n], start=1):
            rows.append({"doc_key": doc, "rank": rank, "term": terms[j], "mean_tfidf": round(float(means[j]), 4)})
    return pd.DataFrame(rows)


def governance_to_regulation(arts: pd.DataFrame, threshold: float = 0.25):
    """For each governance article, the most similar regulation article (char n-gram TF-IDF).
    Returns (matches, sensitivity). Similarity is a lexical signal, not proof of equivalence."""
    gov = arts[arts["doc_key"] == "gov"].reset_index(drop=True)
    reg = arts[arts["doc_key"] == "reg"].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    mat = vec.fit_transform(pd.concat([gov["text"], reg["text"]]).tolist())
    sims = (mat[:len(gov)] @ mat[len(gov):].T).toarray()
    best = sims.argmax(axis=1)
    rows = []
    for i, g in gov.iterrows():
        j = best[i]
        s = float(sims[i, j])
        shared = bool(set(g["topics"]) & set(reg.loc[j, "topics"]))
        rows.append({"governance_article_no": g["article_no"], "governance_page": g["page_start"],
                     "governance_topics": "; ".join(g["topics"]),
                     "closest_regulation_article_no": reg.loc[j, "article_no"],
                     "closest_regulation_page": reg.loc[j, "page_start"],
                     "similarity": round(s, 4), "above_threshold": s >= threshold, "shared_topic": shared})
    matches = pd.DataFrame(rows)
    sens = pd.DataFrame([{"threshold": t, "governance_articles_above": int((matches["similarity"] >= t).sum()),
                          "share_pct": round(100 * (matches["similarity"] >= t).mean(), 1),
                          "of_which_shared_topic": int(((matches["similarity"] >= t) & matches["shared_topic"]).sum())}
                         for t in [0.15, 0.2, 0.25, 0.3]])
    return matches, sens


def governance_scores_status(path_scores: Path, charities_has_score_column: bool):
    """Explain whether governance scores can be analysed. Never infers scores."""
    rows = [{"source": "ملف البيانات المفتوحة", "score_columns_found": "لا يوجد" if not charities_has_score_column else "موجود",
             "status": "غير متاح" if not charities_has_score_column else "متاح",
             "note": "لم يُحلَّل توزيع الدرجات لأنها غير موجودة في الملف"}]
    if Path(path_scores).exists():
        rows.append({"source": "ملف درجات منفصل", "score_columns_found": "يُفحص", "status": "متاح",
                     "note": "يُحلَّل بعد التحقق من الأعمدة الرقمية وصلاحية القيم"})
    else:
        rows.append({"source": "ملف درجات منفصل", "score_columns_found": pd.NA, "status": "غير متاح",
                     "note": "لم يُقدَّم ملف درجات منفصل"})
    return pd.DataFrame(rows)
