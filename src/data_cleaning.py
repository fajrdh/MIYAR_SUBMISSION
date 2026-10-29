"""Load, validate and clean the charities open-data file.
The original file is read-only; cleaning returns new frames and adds flags.
No row is dropped silently."""
import re
from pathlib import Path

import pandas as pd

ORIGINAL_COLUMNS = ["رقم التسجيل", "اسم الجمعية", "نوع الجهة", "المنطقة"]
COLUMN_NAMES = {"رقم التسجيل": "reg_number", "اسم الجمعية": "name",
                "نوع الجهة": "entity_type", "المنطقة": "region"}
TRUNCATION_MARK = "…"


def find_header_row(path) -> int:
    """Return the 0-based line index of the header row (the file starts with a title line)."""
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if line.split(",")[0].strip() == ORIGINAL_COLUMNS[0]:
                return i
    raise ValueError("Header row with 'رقم التسجيل' not found")


def load_raw(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Charities CSV not found: {path}")
    header = find_header_row(path)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, skiprows=header)
    if list(df.columns) != ORIGINAL_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    return df.rename(columns=COLUMN_NAMES)


def _squash(s):
    if pd.isna(s):
        return pd.NA
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or pd.NA


def clean_charities(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].map(_squash)
    df["region_clean"] = df["region"].str.replace(r"^منطقة\s+", "", regex=True)
    df["name_truncated"] = df["name"].fillna("").str.contains(TRUNCATION_MARK, regex=False)
    df["reg_number_shared"] = df["reg_number"].notna() & df["reg_number"].duplicated(keep=False)
    df["exact_duplicate_group"] = df.duplicated(subset=["name", "entity_type", "region"], keep=False)
    return df


def completeness(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    rows = []
    for c in ["reg_number", "name", "entity_type", "region"]:
        non_empty = int(raw[c].fillna("").str.strip().ne("").sum())
        rows.append({"column": c, "non_empty": non_empty,
                     "completeness_pct": round(100 * non_empty / n, 2), "missing": n - non_empty})
    return pd.DataFrame(rows)


def quality_checks(clean: pd.DataFrame) -> pd.DataFrame:
    n = len(clean)
    def row(check, value, note, share=True):
        return {"check": check, "value": value, "note": note,
                "share_pct": round(100 * value / n, 2) if share else pd.NA}
    numeric = clean["reg_number"].fillna("").str.fullmatch(r"\d+")
    return pd.DataFrame([
        row("إجمالي السجلات", n, "لا يُحذف أي سجل", share=False),
        row("عناوين الأعمدة مطابقة للمتوقع", len(ORIGINAL_COLUMNS), ", ".join(ORIGINAL_COLUMNS), share=False),
        row("سجلات مكررة تماماً (الاسم والنوع والمنطقة)", int(clean["exact_duplicate_group"].sum()),
            "تُعرض ولا تُحذف؛ قد تكون جهات مختلفة بأسماء متطابقة"),
        row("سجلات يشترك رقم تسجيلها مع سجل آخر", int(clean["reg_number_shared"].sum()),
            "رقم التسجيل غير فريد؛ لا يصلح مفتاحاً للربط"),
        row("أرقام تسجيل غير رقمية", int((~numeric).sum()), "التحقق من الشكل فقط"),
        row("أسماء مقطوعة (تنتهي بـ …)", int(clean["name_truncated"].sum()),
            "القطع من المصدر؛ لا يمكن استرجاع الاسم الكامل"),
    ])


def name_stats(clean: pd.DataFrame) -> pd.DataFrame:
    lens = clean["name"].fillna("").str.len()
    return pd.DataFrame([{"mean_len": round(lens.mean(), 1), "median_len": float(lens.median()),
                          "max_len": int(lens.max()), "rows_at_max_len": int((lens == lens.max()).sum())}])


def by_type(clean: pd.DataFrame) -> pd.DataFrame:
    t = clean["entity_type"].value_counts().rename("total").reset_index()
    t.columns = ["entity_type", "total"]
    t["share_pct"] = (100 * t["total"] / len(clean)).round(2)
    return t


def by_region(clean: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(clean["region_clean"], clean["entity_type"])
    out = pd.DataFrame({"region": ct.index, "total": ct.sum(axis=1).values})
    out["associations"] = ct.get("جمعية أهلية", 0).values
    out["foundations"] = ct.get("مؤسسة أهلية", 0).values
    out["share_pct"] = (100 * out["total"] / len(clean)).round(2)
    return out.sort_values("total", ascending=False).reset_index(drop=True)


def region_by_type(clean: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(clean["region_clean"], clean["entity_type"], margins=True,
                       margins_name="المجموع").reset_index()
