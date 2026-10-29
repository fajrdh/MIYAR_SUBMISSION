"""Central configuration. Paths are relative to the project root so the same
code runs locally and in Google Colab (override with NONPROFIT_PROJECT_ROOT)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NONPROFIT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_clean"
OUTPUTS_DIR = ROOT / "outputs"
OCR_DIR = DATA_PROCESSED / "ocr_pages"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "report"

# Original files in data_raw/ under these names (see README).
REG_PDF = DATA_RAW / "regulation_executive.pdf"   # اللائحة التنفيذية لنظام الجمعيات والمؤسسات الأهلية
GOV_PDF = DATA_RAW / "governance_rules.pdf"       # قواعد حوكمة الجمعيات والمؤسسات الأهلية
CHARITIES_CSV = DATA_RAW / "Charities_-_Dashboard_2.csv"
SCORES_CSV = DATA_RAW / "governance_scores.csv"   # optional, only if such a file is provided
# Cleaned articles (output of notebooks/01_clean_legal_articles.ipynb): input of the legal analysis
CLEAN_ARTICLES = ROOT / "data_clean" / "legal_articles_full.csv"

DOC_LABELS = {"reg": "اللائحة التنفيذية", "gov": "قواعد الحوكمة"}
EXPECTED_ARTICLES = {"reg": 129, "gov": 47}
OCR_DPI = 250
OCR_LANG = "ara"
OCR_PSM = "6"
EXPECTED_PAGES = {"reg": 56, "gov": 17}
