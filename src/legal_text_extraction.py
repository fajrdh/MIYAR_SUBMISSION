"""Arabic legal-document extraction helpers used by the Mِعيار pipeline.

OCR requires the Tesseract executable and its Arabic language data (ara).
The original PDF pages are never modified; OCR text is cached under data/processed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import fitz
import pandas as pd

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def nz(text) -> str:
    """Normalize Arabic text for comparison while preserving source text elsewhere."""
    if not isinstance(text, str):
        return ""
    t = _DIACRITICS.sub("", text).translate(_ARABIC_DIGITS)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("\u200f", "").replace("\u200e", "")
    return re.sub(r"\s+", " ", t).strip()


normalize_ar = nz

_ONES = {
    "الأولى": 1, "الاولى": 1, "الأول": 1, "الاول": 1,
    "الثانية": 2, "الثاني": 2, "الثالثة": 3, "الثالث": 3,
    "الرابعة": 4, "الرابع": 4, "الخامسة": 5, "الخامس": 5,
    "السادسة": 6, "السادس": 6, "السابعة": 7, "السابع": 7,
    "الثامنة": 8, "الثامن": 8, "التاسعة": 9, "التاسع": 9,
    "العاشرة": 10, "العاشر": 10, "الحادية": 1, "الحادي": 1,
    "الثانية عشرة": 12, "الثاني عشر": 12,
}
_TENS = {
    "العشرون": 20, "العشرين": 20, "الثلاثون": 30, "الثلاثين": 30,
    "الأربعون": 40, "الاربعون": 40, "الأربعين": 40, "الاربعين": 40,
    "الخمسون": 50, "الخمسين": 50, "الستون": 60, "الستين": 60,
    "السبعون": 70, "السبعين": 70, "الثمانون": 80, "الثمانين": 80,
    "التسعون": 90, "التسعين": 90,
}


def parse_ordinal(label):
    """Convert common Arabic ordinal labels (including compound ordinals) to integers."""
    if not isinstance(label, str):
        return None
    s = nz(label).strip(" :.-")
    s = re.sub(r"^الماده\s*", "", s)
    s = re.sub(r"^المادة\s*", "", s)
    # Normalize ordinal spelling variants for lookup.
    direct = {nz(k): v for k, v in _ONES.items()}
    direct.update({nz(k): v for k, v in _TENS.items()})
    if s in direct:
        return direct[s]
    words = s.split()
    if "بعد" in words and "المائه" in words:
        try:
            return 100 + parse_ordinal(" ".join(words[:words.index("بعد")]))
        except (TypeError, ValueError):
            return None
    if len(words) >= 2:
        # Compound forms: الخامسة والأربعون / الثالثة والعشرون
        for i, word in enumerate(words):
            if word in ("والعشرون", "والعشرين", "والثلاثون", "والثلاثين",
                        "والأربعون", "والاربعون", "والأربعين", "والاربعين",
                        "والخمسون", "والخمسين", "والستون", "والستين",
                        "والسبعون", "والسبعين", "والثمانون", "والثمانين",
                        "والتسعون", "والتسعين"):
                tens = _TENS.get(word[1:])
                ones = parse_ordinal(" ".join(words[:i]))
                return tens + ones if tens is not None and ones is not None else None
    if len(words) == 2 and words[0] in ("الحاديه", "الحادي") and words[1] in ("عشره", "عشر"):
        return 11
    if len(words) == 2 and words[0] in ("الثانيه", "الثاني") and words[1] in ("عشره", "عشر"):
        return 12
    if s.isdigit():
        return int(s)
    return None


_HEADER = re.compile(
    r"(?im)^\s*(?:المادة|ماده)\s*[:：]?\s*"
    r"(الأولى|الاولى|الأول|الاول|الثانية|الثاني|الثالثة|الثالث|الرابعة|الرابع|"
    r"الخامسة|الخامس|السادسة|السادس|السابعة|السابع|الثامنة|الثامن|التاسعة|التاسع|"
    r"العاشرة|العاشر|الحادية\s+عشرة|الحادية\s+عشر|الحادية\s+والعشرون|"
    r"الحادية\s+والعشرين|[\u0621-\u064A\s]+?|\d{1,3})"
    r"(?=\s*[:：.\-\u2013\u2014\n]|\s*$)"
)


def _ordinal_number(text):
    return parse_ordinal(text)


def ocr_document(pdf_path, doc_key):
    """Return one OCR/text string per PDF page, caching text in processed/ocr_pages."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    cache = path.parent.parent / "processed" / "ocr_pages" / f"{doc_key}_pages.txt"
    if cache.exists():
        cached = cache.read_text(encoding="utf-8")
        pages = cached.split("\n\f\n")
        if pages and any(p.strip() for p in pages):
            return pages
    if not shutil.which("tesseract"):
        raise RuntimeError("Tesseract OCR غير مثبت. ثبّت Tesseract وحزمة اللغة العربية (ara).")
    try:
        with fitz.open(path) as pdf:
            pages = []
            for page in pdf:
                native = page.get_text("text").strip()
                if len(re.sub(r"\s", "", native)) >= 40:
                    pages.append(native)
                    continue
                pix = page.get_pixmap(dpi=250, alpha=False)
                proc = subprocess.run(
                    ["tesseract", "stdin", "stdout", "-l", "ara", "--psm", "6"],
                    input=pix.tobytes("png"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True,
                )
                pages.append(proc.stdout.decode("utf-8", errors="replace").strip())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"OCR failed for {path.name}: {exc.stderr.decode(errors='replace')}") from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n\f\n".join(pages), encoding="utf-8")
    return pages


def parse_articles(page_texts, doc_key):
    """Extract sequentially numbered articles; suspicious headers remain in prior text and are logged."""
    rows, rejected = [], []
    current = None
    pending = []
    for page_no, page_text in enumerate(page_texts, start=1):
        text = str(page_text or "")
        # Header matching works on a line at a time to avoid false positives in body text.
        lines = text.splitlines()
        for line in lines:
            match = re.match(r"^\s*(?:المادة|ماده)\s*[:：]?\s*([^:：.\-]{1,50}?)(?:\s*[:：.\-]\s*|\s*$)(.*)$", line.strip())
            parsed = parse_ordinal(match.group(1)) if match else None
            if match and parsed is not None:
                if current is None:
                    if parsed != 1:
                        rejected.append({"doc_key": doc_key, "page": page_no, "candidate": line, "parsed": parsed})
                        pending.append(line)
                        continue
                    current = {"doc_key": doc_key, "article_no": 1, "article_label": line.strip(),
                               "page_start": page_no, "page_end": page_no, "text": " ".join(pending), "text_raw": " ".join(pending)}
                    pending = []
                    body = match.group(2).strip()
                    if body:
                        current["text"] = (current["text"] + " " + body).strip()
                        current["text_raw"] = current["text"]
                    continue
                expected = current["article_no"] + 1
                if parsed == expected:
                    rows.append(current)
                    current = {"doc_key": doc_key, "article_no": parsed, "article_label": line.strip(),
                               "page_start": page_no, "page_end": page_no, "text": "", "text_raw": ""}
                    body = match.group(2).strip()
                    if body:
                        current["text"] = body
                        current["text_raw"] = body
                    continue
                rejected.append({"doc_key": doc_key, "page": page_no, "candidate": line, "parsed": parsed})
            if current is not None:
                current["page_end"] = page_no
                current["text"] = (current["text"] + " " + line.strip()).strip()
                current["text_raw"] = current["text"]
            else:
                pending.append(line.strip())
    if current is not None:
        rows.append(current)
    cols = ["doc_key", "article_no", "article_label", "page_start", "page_end", "text", "text_raw"]
    arts = pd.DataFrame(rows, columns=cols)
    if not arts.empty:
        arts["article_no"] = arts["article_no"].astype(int)
        arts["header_fuzzy"] = False
    else:
        arts["header_fuzzy"] = pd.Series(dtype=bool)
    rej = pd.DataFrame(rejected, columns=["doc_key", "page", "candidate", "parsed"])
    return arts, rej


def sequence_report(articles, expected_count):
    numbers = set(articles["article_no"].astype(int).tolist()) if not articles.empty else set()
    expected = set(range(1, int(expected_count) + 1))
    return {"expected": int(expected_count), "found": len(numbers),
            "missing": sorted(expected - numbers), "unexpected": sorted(numbers - expected)}


def arabic_share(text):
    """Percentage of letters in text that are Arabic-script letters."""
    letters = [c for c in str(text or "") if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum("\u0600" <= c <= "\u06ff" for c in letters)
    return round(100 * arabic / len(letters), 2)
