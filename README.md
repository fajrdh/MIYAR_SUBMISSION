# مِعيار — تحليل البيانات القانونية للقطاع غير الربحي السعودي

تحليل مواد **اللائحة التنفيذية لنظام الجمعيات والمؤسسات الأهلية** (129 مادة) و**قواعد حوكمة الجمعيات والمؤسسات الأهلية** (47 مادة)، وبيانات **2,723 جهة غير ربحية** من منصة البيانات المفتوحة، مع نموذج يطابق كل مادة من قواعد الحوكمة بأقرب مواد اللائحة في المعنى.

> المؤشرات والتصنيفات مخرجات تحليلية للمراجعة، وليست استشارة قانونية أو حكمًا على امتثال أي جهة.

النتائج مختصرة في `report/executive_summary.pdf`، وآخر خلية في `analysis.ipynb` تتحقق أن كل رقم فيه مطابق لناتج الكود.

## ترتيب الملفات

```text
.
├── README.md                 # هذا الملف
├── ai_usage_log.md           # توثيق استخدام الذكاء الاصطناعي
├── analysis.ipynb            # دفتر التحليل: يشغّل run_pipeline.py ويعرض النتائج ويتحقق من أرقام الملخص
├── notebooks/
│   ├── 01_clean_legal_articles.ipynb    # تنظيف ملف المواد (Google Colab)
│   └── 02_model_matching.ipynb          # نموذج المطابقة والتجميع (Google Colab + GPU)
├── data_raw/                 # البيانات الأصلية + ملفات المراجعة + كود التصنيف الذي يطلبه دفتر التنظيف
├── data_clean/               # البيانات بعد التنظيف
│   ├── legal_articles_full.csv          # المواد المنظفة (ناتج 01) = مدخل النموذج والتحليل
│   └── charities_clean.csv …            # الجهات بعد التنظيف (ناتج run_pipeline.py)
├── outputs/                  # الجداول والنتائج
│   └── figures/              # الرسوم
├── report/
│   └── executive_summary.pdf # الملخص التنفيذي
├── run_pipeline.py           # التحليل كاملًا بأمر واحد
├── scripts/make_figures.py   # رسوم لوحة Excel الخمس (يستدعيه run_pipeline.py)
├── src/                      # وحدات الكود
├── tests/                    # الاختبارات
├── requirements.txt
└── index.html, netlify.toml, SUPABASE_SETUP.sql   # واجهة الموقع
```

## التشغيل

الخطوات 1 و2 تُشغَّل في Google Colab، ونواتجهما محفوظة في المستودع. لذلك يمكن تشغيل الخطوة 3 مباشرة.

### 1) التنظيف — `notebooks/01_clean_legal_articles.ipynb`
في Google Colab: Runtime ← Run all، ثم ارفعي معًا (تحديد متعدد) هذه الملفات السبعة من `data_raw/`:
- `legal_articles_full_updated.csv` — ملف المواد المستخرج من ملفات PDF بالقراءة الآلية (OCR) في مشروع الفريق.
- `legal_articles_errors_additions.xlsx` — الأسطر الدخيلة، و«ء» الزائدة، والأسطر الساقطة من القراءة الآلية.
- `مراجعة_المواضيع.xlsx` — المواضيع «الواضحة» المعتمدة.
- `مراجعة_القانوني_المواضيع.xlsx` و`مفتاح_مراجعة_القانوني_لا_يُرسل.csv` — قرارات المختص القانوني، والمفتاح الذي يربط خيارَيه (أ) و(ب) بالموضوعين.
- `legal topic classification.py` و`legal_text_extraction.py` — كود تصنيف المواضيع من مشروع الفريق.

**الناتج:** `legal_articles_full.csv`، وهو المحفوظ في `data_clean/`.

### 2) النموذج — `notebooks/02_model_matching.ipynb`
في Colab: Runtime ← Change runtime type ← GPU، ثم Run all، وارفعي `data_clean/legal_articles_full.csv` (المدخل الوحيد).
- المطابقة بنموذج BAAI/bge-m3 (الطريقة المعتمدة) مع حد التشابه 0.75، وخط الأساس TF-IDF، والتجميع الموضوعي، وأداة عرض في آخر خلية.
- نتائج التقييم بالمراجعة البشرية في `outputs/نتائج_النموذج.xlsx`، ومعها `المطابقات_النهائية_bge_m3.csv` و`clusters_bge_m3.csv` و`clusters_vs_topics.xlsx`.

### 3) التحليل — `analysis.ipynb` (أو `run_pipeline.py`)
من جذر المستودع:
```bash
python -m pip install -r requirements.txt
python run_pipeline.py        # أو افتحي analysis.ipynb وشغّلي كل الخلايا
python -m pytest tests/
```
يحتاج Python 3.12 أو أحدث، والمدة أقل من دقيقة. المدخلات:
- `data_raw/Charities_-_Dashboard_2.csv` — بيانات الجهات.
- `data_clean/legal_articles_full.csv` — المواد المنظفة.

**ما يكتبه:**
- **الجهات:** `outputs/charities_*.csv` (الاكتمال، الجودة، التوزيع حسب النوع والمنطقة)، و`governance_scores_status.csv`، و`data_clean/charities_clean.csv`.
- **المواد:** `outputs/legal_articles_index.csv` (الموضوع النهائي لكل مادة وسبب تعديله إن عُدّل)، و`topic_counts_by_document.csv` (عدد المواد لكل موضوع رئيسي في كل وثيقة)، و`requirement_indicators.csv` (جمل الالتزام والحظر والشرط)، و`top_terms_by_document.csv`، و`classification_review_sample.csv`.
- **الإحصاءات:** `outputs/run_metrics.json` (مؤشر التركز HHI، مربع كاي، كرامر V، باريتو).
- **الرسوم:** في `outputs/figures/`، ومنها رسوم لوحة `outputs/statistical_report_dashboard.xlsx` الخمس (`01`–`05`) التي يرسمها `scripts/make_figures.py`.

**ملاحظة:** استخراج المواد من ملفات PDF بالقراءة الآلية (الدالة `ocr_document` في `src/legal_text_extraction.py`، ويحتاج Tesseract بالعربية) هو المرحلة التي أنتجت `data_raw/legal_articles_full_updated.csv`، ولا يعيدها `run_pipeline.py`.
