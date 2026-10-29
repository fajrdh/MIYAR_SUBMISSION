"""يرسم الرسومات الخمس في outputs/figures/ (نفس رسوم outputs/statistical_report_dashboard.xlsx):
الرسوم 02–04 (الكيانات) من أوراق ملف Excel، والرسمان 01 و05 (المواضيع) من ملف المواد المنظف
data_clean/legal_articles_full.csv (176 مادة).

التشغيل من جذر المشروع:  python scripts/make_figures.py
أو بمسارات صريحة:        python scripts/make_figures.py <xlsx> <legal_articles_full.csv> <مجلد الرسوم>
المتطلبات: pandas، openpyxl، matplotlib، arabic-reshaper، python-bidi (في requirements.txt)."""
import sys, openpyxl, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
XLSX, CSV, OUT = sys.argv[1:4] if len(sys.argv) == 4 else (
    'outputs/statistical_report_dashboard.xlsx', 'data_clean/legal_articles_full.csv', 'outputs/figures')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
ar = lambda s: get_display(arabic_reshaper.reshape(str(s)))
BLUE, RED = '#4F81BD', '#C0504D'
wb = openpyxl.load_workbook(XLSX, data_only=False)
def sheet(name):
    rows = [r for r in wb[name].iter_rows(min_row=2, values_only=True) if r[0] and r[0] != 'الإجمالي']
    return [r[0] for r in rows], [r[1] for r in rows]
d = pd.read_csv(CSV, encoding='utf-8-sig')
vc = d.primary_topic.value_counts(); assert vc.sum() == 176 and d.primary_topic.notna().all()

# 1 عدد المواد حسب الموضوع الرئيسي (أفقي)
fig, ax = plt.subplots(figsize=(9, 9))
y = range(len(vc))[::-1]
ax.barh(list(y), vc.values, color=BLUE)
ax.set_yticks(list(y)); ax.set_yticklabels([ar(t) for t in vc.index])
for yi, v in zip(y, vc.values): ax.text(v + 0.3, yi, str(v), va='center', fontsize=9)
ax.set_title(ar('عدد المواد حسب الموضوع الرئيسي (176 مادة، 25 موضوعًا)'))
ax.set_xlabel(ar('عدد المواد')); fig.tight_layout(); fig.savefig(f'{OUT}/01_articles_by_topic.png', dpi=200); plt.close(fig)

# 2 الكيانات حسب المنطقة
reg, n = sheet('التوزيع الجغرافي للكيانات'); assert sum(n) == 2723
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(range(len(reg)), n, color=BLUE)
ax.set_xticks(range(len(reg))); ax.set_xticklabels([ar(r) for r in reg], rotation=45, ha='right')
for i, v in enumerate(n): ax.text(i, v + 8, f'{v:,}', ha='center', fontsize=9)
ax.set_title(ar('الكيانات حسب المنطقة')); ax.set_ylabel(ar('عدد الكيانات'))
fig.tight_layout(); fig.savefig(f'{OUT}/02_entities_by_region.png', dpi=200); plt.close(fig)

# 3 نسبة أنواع الكيانات (دائري مفرغ)
typ, tn = sheet('أنواع الكيانات'); assert sum(tn) == 2723
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(tn, colors=[BLUE, RED], startangle=90, counterclock=False, wedgeprops={'width': 0.42},
       labels=[ar(t) + f'\n{v:,} ({v/sum(tn):.1%})' for t, v in zip(typ, tn)])
ax.set_title(ar('نسبة أنواع الكيانات')); fig.tight_layout(); fig.savefig(f'{OUT}/03_entity_types_share.png', dpi=200); plt.close(fig)

# 4 مقارنة عدد الجمعيات والمؤسسات
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(range(2), tn, color=BLUE, width=0.55)
ax.set_xticks(range(2)); ax.set_xticklabels([ar(t) for t in typ])
for i, v in enumerate(tn): ax.text(i, v + 25, f'{v:,}', ha='center')
ax.set_title(ar('مقارنة عدد الجمعيات والمؤسسات')); ax.set_ylabel(ar('العدد'))
fig.tight_layout(); fig.savefig(f'{OUT}/04_entity_types_count.png', dpi=200); plt.close(fig)

# 5 تحليل باريتو
cum = vc.cumsum() / vc.sum()
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(range(len(vc)), vc.values, color=BLUE)
ax.set_xticks(range(len(vc))); ax.set_xticklabels([ar(t) for t in vc.index], rotation=60, ha='right', fontsize=8)
ax.set_ylabel(ar('عدد المواد'))
ax2 = ax.twinx(); ax2.plot(range(len(vc)), cum.values * 100, color='#BE4B48', marker='o', ms=3)
ax2.axhline(80, color='gray', ls='--', lw=0.8); ax2.set_ylim(0, 105); ax2.set_ylabel(ar('النسبة التراكمية %'))
ax2.spines['top'].set_visible(False)
ax.set_title(ar('تحليل باريتو: المواد والنسبة التراكمية'))
fig.tight_layout(); fig.savefig(f'{OUT}/05_topics_pareto.png', dpi=200); plt.close(fig)
print('ok', list(zip(reg, n))[:2], typ, tn)
