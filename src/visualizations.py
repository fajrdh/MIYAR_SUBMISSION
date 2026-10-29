"""Arabic-labelled charts. Every figure is drawn from the data passed to it."""
from pathlib import Path

import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from matplotlib import font_manager

_PREFERRED = ["Noto Sans Arabic", "Amiri", "Arial", "Tahoma", "DejaVu Sans"]


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in _PREFERRED if n in names), "DejaVu Sans")


plt.rcParams["font.family"] = _font()
plt.rcParams["axes.unicode_minus"] = False


def ar(text) -> str:
    """Shape and reorder Arabic for matplotlib, which does not do it itself."""
    return get_display(arabic_reshaper.reshape(str(text)))


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def hbar(labels, values, title, xlabel, path, color="#2a7f8e"):
    labels, values = list(labels)[::-1], list(values)[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(labels))))
    ax.barh([ar(l) for l in labels], values, color=color)
    for y, v in enumerate(values):
        ax.text(v, y, f" {v:,}", va="center", fontsize=9)
    ax.set_title(ar(title), fontsize=13)
    ax.set_xlabel(ar(xlabel))
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def stacked_hbar(labels, parts: dict, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(labels))))
    left = np.zeros(len(labels))
    for name, vals in parts.items():
        ax.barh([ar(l) for l in labels], vals, left=left, label=ar(name))
        left += np.array(vals)
    ax.set_title(ar(title), fontsize=13)
    ax.set_xlabel(ar(xlabel))
    ax.legend(prop={"size": 9})
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def grouped(categories, series: dict, title, ylabel, path):
    x = np.arange(len(categories))
    w = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=ar(name))
    ax.set_xticks(x)
    ax.set_xticklabels([ar(c) for c in categories], rotation=35, ha="right")
    ax.set_title(ar(title), fontsize=13)
    ax.set_ylabel(ar(ylabel))
    ax.legend(prop={"size": 9})
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def completeness(df, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([ar(c) for c in df["column"]], df["completeness_pct"], color="#3c9d6e")
    for i, v in enumerate(df["completeness_pct"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ar("نسبة الاكتمال %"))
    ax.set_title(ar("نسبة اكتمال الأعمدة"), fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def similarity_hist(sims, threshold, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sims, bins=15, color="#2a7f8e")
    ax.axvline(threshold, color="#c9742f", ls="--", label=ar(f"عتبة {threshold}"))
    ax.set_xlabel(ar("درجة التشابه مع أقرب مادة في اللائحة"))
    ax.set_ylabel(ar("عدد مواد قواعد الحوكمة"))
    ax.set_title(ar("تشابه مواد قواعد الحوكمة مع اللائحة"), fontsize=13)
    ax.legend(prop={"size": 9})
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)
