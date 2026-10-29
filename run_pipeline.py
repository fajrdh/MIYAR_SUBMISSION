"""Analysis pipeline:  python run_pipeline.py   (run from the project root)
Inputs : data_raw/Charities_-_Dashboard_2.csv (open data on non-profit entities)
         data_clean/legal_articles_full.csv (cleaned articles, from notebooks/01_clean_legal_articles.ipynb)
Outputs: outputs/ (tables, run_metrics.json)   Figures: outputs/figures/   Cleaned entities: data_clean/"""
import ast
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src import config
from src import data_cleaning as dc
from src import governance_analysis as ga
from src import legal_topic_classification as ltc
from src import visualizations as viz
from src import audit


def ensure_dirs():
    for d in [config.DATA_PROCESSED, config.OUTPUTS_DIR, config.FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_charities(metrics: dict):
    raw = dc.load_raw(config.CHARITIES_CSV)
    clean = dc.clean_charities(raw)
    clean.to_csv(config.DATA_PROCESSED / "charities_clean.csv", index=False, encoding="utf-8-sig")  # git-ignored

    comp = dc.completeness(raw)
    checks = dc.quality_checks(clean)
    names = dc.name_stats(clean)
    by_type, by_region = dc.by_type(clean), dc.by_region(clean)
    cross = dc.region_by_type(clean)
    out = config.OUTPUTS_DIR
    comp.to_csv(out / "charities_completeness.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(out / "charities_quality_checks.csv", index=False, encoding="utf-8-sig")
    names.to_csv(out / "charities_name_stats.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(out / "charities_by_type.csv", index=False, encoding="utf-8-sig")
    by_region.to_csv(out / "charities_by_region.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(out / "charities_region_by_type.csv", index=False, encoding="utf-8-sig")

    status = ga.governance_scores_status(config.SCORES_CSV, charities_has_score_column=False)
    status.to_csv(out / "governance_scores_status.csv", index=False, encoding="utf-8-sig")

    f = config.FIGURES_DIR
    viz.hbar(by_region["region"].str.replace(r"^منطقة\s+", "", regex=True), by_region["total"],
             "توزيع الجهات حسب المنطقة", "عدد الجهات", f / "charities_by_region.png")
    viz.hbar(by_type["entity_type"], by_type["total"], "توزيع الجهات حسب نوع الكيان", "عدد الجهات",
             f / "charities_by_type.png", color="#c9742f")
    d = by_region.assign(region=by_region["region"].str.replace(r"^منطقة\s+", "", regex=True))
    viz.stacked_hbar(d["region"], {"جمعيات أهلية": d["associations"], "مؤسسات أهلية": d["foundations"]},
                     "الجهات حسب المنطقة ونوع الكيان", "عدد الجهات", f / "charities_region_by_type.png")
    viz.completeness(comp, f / "charities_completeness.png")

    metrics["charities"] = {
        "rows": int(len(clean)),
        "entity_counts": by_type.set_index("entity_type")["total"].to_dict(),
        "top_regions": by_region.head(3).set_index("region")["total"].to_dict(),
        "reg_number_shared_rows": int(clean["reg_number_shared"].sum()),
        "reg_number_unique_values": int(clean["reg_number"].nunique()),
        "truncated_names": int(clean["name_truncated"].sum()),
        "exact_duplicate_group_rows": int(clean["exact_duplicate_group"].sum()),
        "governance_score_columns": 0,
        "governance_score_analysis": "لم يُنفَّذ: الملف لا يحتوي على درجات حوكمة، ولا يوجد ملف درجات منفصل.",
    }


def _as_list(v):
    """The cleaned file stores the topic list as text, e.g. "['العضوية', 'اللجان']"."""
    return list(ast.literal_eval(v)) if isinstance(v, str) and v.strip() else []


def run_legal(metrics: dict):
    """Legal-articles analysis on the cleaned file data_clean/legal_articles_full.csv (176 articles),
    the output of notebooks/01_clean_legal_articles.ipynb. The OCR extraction that produced the raw
    file (data_raw/legal_articles_full_updated.csv) is not re-run here."""
    out = config.OUTPUTS_DIR
    arts = pd.read_csv(config.CLEAN_ARTICLES, encoding="utf-8-sig")
    assert len(arts) == 176 and arts["primary_topic"].notna().all(), "يُتوقع 176 مادة كلها بموضوع"
    arts["topics"] = arts["topics"].map(_as_list)
    found = arts.groupby("doc_key").size().to_dict()
    for key, n in config.EXPECTED_ARTICLES.items():
        assert found.get(key) == n, (key, found.get(key), n)
    seq = {k: {"found": int(found[k]), "expected": n} for k, n in config.EXPECTED_ARTICLES.items()}

    index = arts[["doc_key", "article_no", "article_label", "page_start", "page_end", "topics",
                  "primary_topic", "topic_hits", "tfidf_topic", "tfidf_score", "agrees_with_rules", "topic_override"]].copy()
    index["topics"] = index["topics"].map(lambda t: "; ".join(t))
    index["topic_hits"] = index["topic_hits"].map(lambda h: json.dumps(ast.literal_eval(h), ensure_ascii=False))
    index.to_csv(out / "legal_articles_index.csv", index=False, encoding="utf-8-sig")

    review = audit.classification_review_sample(arts.assign(primary_topic=arts["primary_topic"]))
    review.to_csv(config.DATA_PROCESSED / "classification_review_sample.csv", index=False, encoding="utf-8-sig")
    review.drop(columns=["text_first_20_words"]).to_csv(out / "classification_review_sample.csv", index=False, encoding="utf-8-sig")

    req = ltc.extract_requirements(arts)
    req.insert(0, "indicator_id", [f"{d}-{i:04d}" for i, d in enumerate(req["doc_key"], start=1)])
    req.to_csv(config.DATA_PROCESSED / "requirement_snippets.csv", index=False, encoding="utf-8-sig")
    req.drop(columns=["snippet_15_words"]).to_csv(out / "requirement_indicators.csv", index=False, encoding="utf-8-sig")

    # Primary topic (final, after the documented reviews) per document
    counts = arts.groupby(["primary_topic", "doc_key"]).size().unstack(fill_value=0).reindex(columns=["gov", "reg"], fill_value=0)
    topic_table = pd.DataFrame({"topic": counts.index, "gov_articles": counts["gov"].astype(int).values,
                                "reg_articles": counts["reg"].astype(int).values})
    topic_table["total_articles"] = topic_table["gov_articles"] + topic_table["reg_articles"]
    topic_table["gov_share_pct"] = (100 * topic_table["gov_articles"] / seq["gov"]["found"]).round(1)
    topic_table["reg_share_pct"] = (100 * topic_table["reg_articles"] / seq["reg"]["found"]).round(1)
    topic_table = topic_table.sort_values(["total_articles", "topic"], ascending=[False, True]).reset_index(drop=True)
    topic_table.to_csv(out / "topic_counts_by_document.csv", index=False, encoding="utf-8-sig")

    terms = ga.top_terms(arts)
    terms.to_csv(out / "top_terms_by_document.csv", index=False, encoding="utf-8-sig")

    # Statistics reported in report/executive_summary.pdf
    vc = arts["primary_topic"].value_counts()
    cum = vc.cumsum() / vc.sum()
    pareto_n = int((cum < 0.8).sum() + 1)
    ct = counts.T                                   # documents x topics
    chi2, p, dof, expected = chi2_contingency(ct)
    cramers_v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
    disagree = arts["agrees_with_rules"] == False  # noqa: E712
    stats = {"topics": int(len(vc)), "largest_topic": vc.index[0], "largest_topic_articles": int(vc.iloc[0]),
             "largest_topic_pct": round(100 * vc.iloc[0] / vc.sum(), 1),
             "top2_pct": round(100 * vc.iloc[:2].sum() / vc.sum(), 1),
             "pareto_topics_for_80pct": pareto_n, "pareto_cover_pct": round(100 * float(cum.iloc[pareto_n - 1]), 1),
             "chi2": round(float(chi2), 1), "chi2_dof": int(dof), "chi2_p": float(p), "cramers_v": round(cramers_v, 2),
             "chi2_cells_expected_below_5_pct": round(100 * float((expected < 5).mean()), 0),
             "rule_vs_tfidf_disagree": int(disagree.sum()),
             "rule_vs_tfidf_disagree_without_documented_override": int((disagree & arts["topic_override"].isna()).sum()),
             "articles_with_documented_override": int(arts["topic_override"].notna().sum())}

    f = config.FIGURES_DIR
    viz.grouped(topic_table["topic"], {"قواعد الحوكمة": topic_table["gov_articles"],
                                      "اللائحة التنفيذية": topic_table["reg_articles"]},
                "عدد المواد لكل موضوع في كل وثيقة", "عدد المواد", f / "legal_topics_by_document.png")
    rt = req.groupby(["doc_key", "requirement_type"]).size().unstack(fill_value=0)
    viz.grouped(rt.columns.tolist(), {config.DOC_LABELS[d]: rt.loc[d].tolist() for d in rt.index},
                "أنواع المتطلبات في كل وثيقة", "عدد الجمل", f / "legal_requirement_types.png")

    metrics["legal"] = {
        "source": "data_clean/legal_articles_full.csv",
        "sequence_checks": seq,
        "articles_total": int(len(arts)),
        "articles_with_primary_topic": int(arts["primary_topic"].notna().sum()),
        "topic_stats": stats,
        "requirements_total": int(len(req)),
        "requirements_by_type": req.groupby(["doc_key", "requirement_type"]).size().reset_index(name="n")
                                  .to_dict(orient="records"),
        "topic_table": topic_table.to_dict(orient="records"),
        "topics_absent_in_one_document": sorted(
            set(topic_table.loc[topic_table["gov_articles"] == 0, "topic"]) |
            set(topic_table.loc[topic_table["reg_articles"] == 0, "topic"])),
    }


def run_entity_stats(metrics: dict):
    """Concentration statistics reported in report/executive_summary.pdf (from the charities tables)."""
    out = config.OUTPUTS_DIR
    reg = pd.read_csv(out / "charities_by_region.csv")
    share = reg["total"] / reg["total"].sum()
    cross = pd.read_csv(out / "charities_region_by_type.csv").set_index("region_clean")
    cross = cross.drop(index="المجموع", errors="ignore").drop(columns="المجموع", errors="ignore")
    chi2, p, dof, _ = chi2_contingency(cross)
    metrics["charities"].update({
        "hhi": round(float((share ** 2).sum() * 10000)), "hhi_min_equal": round(10000 / len(reg)),
        "top3_regions_pct": round(100 * float(reg["total"].nlargest(3).sum() / reg["total"].sum()), 1),
        "region_type_cramers_v": round(float(np.sqrt(chi2 / (cross.values.sum() * (min(cross.shape) - 1)))), 2),
        "regions_with_foundations": int((cross["مؤسسة أهلية"] > 0).sum()),
        "foundations_in_riyadh": int(reg.loc[reg["region"] == "الرياض", "foundations"].iloc[0]),
    })


def main():
    os.chdir(config.ROOT)                  # scripts/make_figures.py uses paths relative to the project root
    ensure_dirs()
    metrics = {}
    run_charities(metrics)
    run_entity_stats(metrics)
    run_legal(metrics)
    # The five dashboard charts (outputs/figures/01..05)
    script = str(config.ROOT / "scripts" / "make_figures.py")
    argv, sys.argv = sys.argv, [script]    # default paths, whatever arguments the caller (e.g. a notebook kernel) has
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv
    (config.OUTPUTS_DIR / "run_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
