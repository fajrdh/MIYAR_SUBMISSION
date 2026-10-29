"""نسخة مختصرة من src/legal_text_extraction.py في مشروع زميلة 1: الدالة nz وثوابتها فقط، منقولة حرفيًا
(يحتاجها legal topic classification.py). بقية الملف (استخراج النص من PDF) لا تلزم للتنظيف."""
import re

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def nz(text) -> str:
    """Normalise Arabic for comparison: strip diacritics/tatweel, unify alef, ya, ta marbuta,
    map Arabic-Indic digits to ASCII, collapse spaces. The original text is kept separately."""
    if not isinstance(text, str):
        return ""
    t = _DIACRITICS.sub("", text).translate(_ARABIC_DIGITS)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("\u200f", "").replace("\u200e", "")
    return re.sub(r"\s+", " ", t).strip()


normalize_ar = nz
