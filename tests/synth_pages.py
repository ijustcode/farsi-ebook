#!/usr/bin/env python3
"""Synthetic Persian page benchmark with exact word truth (plan §6.1).

Run as ``./venv/bin/python tests/synth_pages.py <command>``. No API calls, no
downloads, no new dependencies (PyMuPDF + numpy only).

What it does
------------
1. Lays out real Persian text into a one-page PDF with PyMuPDF
   ``insert_htmlbox`` (MuPDF's HTML engine shapes with HarfBuzz and applies the
   Unicode bidi algorithm, so joining and right-to-left order come from a real
   shaper, not from us).
2. Reads the exact glyph rectangles back from that PDF (``get_text("rawdict")``)
   and assigns every glyph to the source word it came from (see "word identity"
   below). Word rectangle = union of its glyph rectangles. These are *layout*
   rectangles: advance width x (ascender..descender), not ink bounds.
3. Rasterises the page at the project's hi-res size (long edge 2576 px).
4. Degrades the image: print-side ink spread/dropout, then a geometric warp
   (slope ramp down the page, bow, rotation, perspective), then camera-side
   uneven light, blur, noise, JPEG. Non-text overlays (stamp, signature,
   underlines, margin marks) are drawn on the paper before the warp. Every
   geometric step is applied identically to the truth geometry; overlays and
   photometric steps never change truth.
5. Writes ``<name>.png`` and ``<name>.truth.json`` under ``out/synth/``
   (gitignored). Only specs + seeds + hashes are tracked
   (``tests/data/synth_manifest.json``).

Word identity (why it is exact)
-------------------------------
MuPDF's own word segmentation is NOT usable as truth for Persian: it splits
words at ZWNJ (reported as a space character), at bidi-run boundaries and
sometimes after non-joining letters, and its extracted text is presentation
forms plus wrong code points for glyphs that have no cmap entry. So every
source token is wrapped in its own ``<span>`` whose colour encodes the token
id in the low 4 bits of each channel (``#0R0G0B``, id+1 = R*256+G*16+B, at
most 4095 tokens; the darkest-to-lightest "black" is 0..15/255, i.e. at most
6% grey, invisible after any degradation). The rawdict span colour of every
glyph therefore names its source token, in the SAME document that produces
the image -- no second layout is needed. (A second all-black layout was tried
first and rejected: MuPDF merges adjacent identically-styled spans and its
justification then drifts up to 0.64 pt from the coloured layout.) The page is
also rendered in RGB with anti-aliasing off, before any non-text ink is
drawn, giving a per-pixel word-id ink map used by smoke test 1a and to
compute each word's ink bounding box after degradation.

Coordinates
-----------
Truth geometry is normalised to the degraded image: x in [0,1] left->right,
y in [0,1] top->bottom, continuous coordinates (pixel i spans [i, i+1)/W).

Reading order
-------------
``reading_index`` is the logical (source) order: top-to-bottom by line, and
within a line right-to-left, EXCEPT inside an embedded left-to-right run
(Latin words, and European/Persian digits that follow a Latin word, per bidi
rule W7), where logical order runs left-to-right. ``visual_index`` is the
plain right-to-left position within the printed line. Verse rows are one
printed line whose right hemistich precedes the left one.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "synth"
MANIFEST_PATH = ROOT / "tests" / "data" / "synth_manifest.json"
GENERATOR_VERSION = 1

PAGE_W, PAGE_H = 595.0, 842.0           # A4 in PDF points
LONG_EDGE = 2576                        # project hi-res render long edge
ZOOM = LONG_EDGE / PAGE_H

# Private documents: never used as a text source, never copied anywhere.
PRIVATE_SLUGS = {"hadpi-articles", "jafarian-factory-sale"}
PERSIAN_SPECIFIC = "پچژگکی"

# ---------------------------------------------------------------------------
# Fonts. Referenced by path at generation time; never copied into the repo.
# The first available candidate of a class is its default.
# ---------------------------------------------------------------------------
_VZ = str(ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf")
_VZB = str(ROOT / "assets" / "fonts" / "Vazirmatn-Bold.ttf")
_SYS = "/System/Library/Fonts"
_SUP = "/System/Library/Fonts/Supplemental"
FONT_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "modern_naskh": [("Vazirmatn Regular", _VZ),
                     ("Geeza Pro", f"{_SYS}/GeezaPro.ttc"),
                     ("Tahoma", f"{_SUP}/Tahoma.ttf")],
    "book_naskh": [("DecoType Naskh", f"{_SUP}/DecoTypeNaskh.ttc"),
                   ("Times New Roman", f"{_SUP}/Times New Roman.ttf"),
                   ("Nadeem", f"{_SUP}/Nadeem.ttc"),
                   ("Baghdad", f"{_SUP}/Baghdad.ttc")],
    # Courier New is a genuine fixed-pitch (typewriter-design) face whose
    # Arabic block is also fixed-pitch (every isolated letter advance = 0.6 em).
    "typewriter": [("Courier New", f"{_SUP}/Courier New.ttf")],
    "nastaliq": [("Noto Nastaliq Urdu", f"{_SYS}/NotoNastaliq.ttc"),
                 ("DecoType Nastaleeq Urdu UI", f"{_SYS}/DecoTypeNastaleeqUrdu.ttc")],
    "bold_heading": [("Vazirmatn Bold", _VZB)],
    "mixed": [("Vazirmatn Regular", _VZ)],
}

CLASS_DEFAULTS: dict[str, dict] = {
    "modern_naskh": dict(layout="prose", justify=True, size=13.0, leading=1.75),
    "book_naskh": dict(layout="footnote", justify=True, size=14.0, leading=1.65),
    "typewriter": dict(layout="articles", justify=False, size=11.5, leading=1.9),
    "nastaliq": dict(layout="verse", justify=False, size=15.0, leading=2.4),
    # Headings in the class font (bold); body text in Vazirmatn Regular.
    "bold_heading": dict(layout="headings", justify=True, size=13.0, leading=1.75,
                         body_font=_VZ),
    "mixed": dict(layout="mixed", justify=True, size=12.5, leading=1.85),
}

LAYOUTS = ("prose", "verse", "headings", "articles", "footnote", "mixed")

DEGRADE_PRESETS: dict[str, dict] = {
    "clean": {},
    # Roughly the hadpi photographed pages (E18): slope 0.1 deg at top to
    # 3.3 deg at the bottom, plus camera effects and office overlays.
    "photo": dict(warp_top_deg=0.1, warp_mid_deg=2.2, warp_bottom_deg=3.3, warp_bow_px=6.0,
                  rotate_deg=0.4, perspective=0.008,
                  ink_spread_px=1, ink_dropout=0.12,
                  light_gradient=0.35, blur_sigma_px=1.1, noise_sigma=0.03,
                  jpeg_quality=70,
                  stamp=True, signature=True, underlines=4, margin_marks=3),
}
DEGRADE_KEYS = {
    "warp_top_deg": 0.0, "warp_bottom_deg": 0.0, "warp_mid_deg": None, "warp_bow_px": 0.0,
    "rotate_deg": 0.0, "perspective": 0.0,
    "ink_spread_px": 0, "ink_dropout": 0.0,
    "light_gradient": 0.0, "blur_sigma_px": 0.0, "noise_sigma": 0.0,
    "jpeg_quality": 0,
    "stamp": False, "signature": False, "underlines": 0, "margin_marks": 0,
}

# ---------------------------------------------------------------------------
# Embedded text. Prose and verse: Saadi (Golestan, d. 1291/92), Ferdowsi
# (d. c. 1020), Hafez (d. 1390) -- public domain. Headings: Golestan chapter
# titles. Articles, footnotes and mixed-script sentences were written for this
# file. Nothing here comes from any book in books/.
# ---------------------------------------------------------------------------
PROSE = [
    "منت خدای را عز و جل که طاعتش موجب قربت است و به شکر اندرش مزید نعمت. هر نفسی که فرو می‌رود ممد حیات است و چون برمی‌آید مفرح ذات. پس در هر نفسی دو نعمت موجود است و بر هر نعمتی شکری واجب.",
    "باران رحمت بی‌حسابش همه را رسیده و خوان نعمت بی‌دریغش همه جا کشیده. پرده ناموس بندگان به گناه فاحش ندرد و وظیفه روزی به خطای منکر نبرد.",
    "فراش باد صبا را گفته تا فرش زمردین بگسترد و دایه ابر بهاری را فرموده تا بنات نبات در مهد زمین بپرورد. درختان را به خلعت نوروزی قبای سبز ورق در بر گرفته و اطفال شاخ را به قدوم موسم ربیع کلاه شکوفه بر سر نهاده. عصاره تاکی به قدرت او شهد فایق شده و تخم خرمایی به تربیتش نخل باسق گشته.",
    "یکی از صاحبدلان سر به جیب مراقبت فرو برده بود و در بحر مکاشفت مستغرق شده. آنگه که از این معاملت باز آمد، یکی از یاران به طریق انبساط گفت: از این بوستان که بودی ما را چه تحفه کرامت کردی؟ گفت: به خاطر داشتم که چون به درخت گل رسم دامنی پر کنم هدیه اصحاب را. چون برسیدم بوی گلم چنان مست کرد که دامنم از دست برفت.",
    "پادشاهی را شنیدم به کشتن اسیری اشارت کرد. بیچاره در آن حالت نومیدی ملک را دشنام دادن گرفت و سقط گفتن، که گفته‌اند هر که دست از جان بشوید هر چه در دل دارد بگوید. ملک پرسید چه می‌گوید؟ یکی از وزرای نیک‌محضر گفت: ای خداوند همی‌گوید که خشم خود را فرو خورندگان و از مردم درگذرندگان را خدای دوست دارد. ملک را رحمت آمد و از سر خون او درگذشت.",
    "وزیر دیگر که ضد او بود گفت: ابنای جنس ما را نشاید در حضرت پادشاهان جز به راستی سخن گفتن. این ملک را دشنام داد و ناسزا گفت. ملک روی از این سخن درهم آورد و گفت: آن دروغ وی پسندیده‌تر آمد مرا زین راست که تو گفتی، که روی آن در مصلحتی بود و بنای این بر خبثی.",
    "یکی را از ملوک خراسان به خواب دیدند که جمله وجود او ریخته بود و خاک شده مگر چشمان او که همچنان در چشمخانه همی‌گردید و نظر می‌کرد. سایر حکما از تأویل آن فرو ماندند مگر درویشی که به جای آورد و گفت: هنوز نگران است که ملکش با دگران است.",
]
VERSES = [
    ("از دست و زبان که برآید", "کز عهده شکرش به درآید"),
    ("بنده همان به که ز تقصیر خویش", "عذر به درگاه خدای آورد"),
    ("ورنه سزاوار خداوندیش", "کس نتواند که به جای آورد"),
    ("ای کریمی که از خزانه غیب", "گبر و ترسا وظیفه‌خور داری"),
    ("دوستان را کجا کنی محروم", "تو که با دشمن این نظر داری"),
    ("ابر و باد و مه و خورشید و فلک در کارند", "تا تو نانی به کف آری و به غفلت نخوری"),
    ("همه از بهر تو سرگشته و فرمانبردار", "شرط انصاف نباشد که تو فرمان نبری"),
    ("به نام خداوند جان و خرد", "کزین برتر اندیشه برنگذرد"),
    ("خداوند نام و خداوند جای", "خداوند روزی‌ده رهنمای"),
    ("توانا بود هر که دانا بود", "ز دانش دل پیر برنا بود"),
    ("بنی آدم اعضای یک پیکرند", "که در آفرینش ز یک گوهرند"),
    ("چو عضوی به درد آورد روزگار", "دگر عضوها را نماند قرار"),
    ("تو کز محنت دیگران بی‌غمی", "نشاید که نامت نهند آدمی"),
    ("الا یا ایها الساقی ادر کاسا و ناولها", "که عشق آسان نمود اول ولی افتاد مشکل‌ها"),
    ("به بوی نافه‌ای کاخر صبا زان طره بگشاید", "ز تاب جعد مشکینش چه خون افتاد در دل‌ها"),
    ("هر که نان از عمل خویش خورد", "منت حاتم طائی نبرد"),
    ("دو چیز طیره عقل است دم فروبستن", "به وقت گفتن و گفتن به وقت خاموشی"),
]
HEADINGS = ["دیباچه", "باب اول در سیرت پادشاهان", "باب دوم در اخلاق درویشان",
            "باب سوم در فضیلت قناعت", "باب چهارم در فواید خاموشی", "حکایت"]
ARTICLES = [
    "در صورتی که خریدار تا تاریخ {date} مبلغ {amount} ریال را به حساب فروشنده واریز نکند، فروشنده حق فسخ قرارداد را خواهد داشت:",
    "کلیه هزینه‌های نقل و انتقال، اعم از عوارض شهرداری و مالیات، بر عهده طرفین به نسبت مساوی است و هیچ‌یک از طرفین حق رجوع ندارد.",
    "مدت این قرارداد از تاریخ امضا به مدت {n} ماه شمسی است و با توافق کتبی طرفین قابل تمدید خواهد بود.",
    "هرگونه اختلاف ناشی از تفسیر یا اجرای این قرارداد ابتدا از طریق مذاکره و در صورت عدم توافق از طریق داوری حل و فصل می‌شود.",
    "تحویل کالا در محل انبار فروشنده واقع در کیلومتر {n} جاده قدیم انجام می‌گیرد و خریدار موظف است ظرف {m} روز نسبت به حمل آن اقدام نماید.",
    "این سند در {n} نسخه تنظیم شده که هر کدام حکم واحد دارد و از تاریخ امضای طرفین معتبر خواهد بود.",
    "فروشنده اقرار می‌نماید که مورد معامله در رهن یا توقیف هیچ شخص حقیقی یا حقوقی نیست و مسئولیت هرگونه ادعای ثالث با اوست.",
]
FOOTNOTES = [
    "این حکایت در برخی نسخه‌ها با اندکی اختلاف آمده است.",
    "نگاه کنید به: شرح گلستان، چاپ دوم، ۱۳۴۸، ص ۲۷.",
    "در نسخه خطی شماره ۵۲۱۴، برگ ۱۴، این بیت نیامده است.",
    "ضبط متن مطابق نسخه اساس است و بدل‌ها در حاشیه آمده‌اند.",
]
MIXED = [
    "نسخه 2.4 نرم‌افزار PyMuPDF در تاریخ ۱۴۰۳/۰۲/۱۵ منتشر شد و حجم آن حدود ۳۸ مگابایت است.",
    "برای اطلاعات بیشتر به نشانی www.example.org مراجعه کنید یا با شماره ۰۲۱-۸۸۴۴۵۵۶۶ تماس بگیرید.",
    "در سال 1998 میلادی، استاندارد ISO 9001 بازنگری شد؛ اما ترجمه فارسی آن (چاپ دوم) تا سال ۱۳۸۰ منتشر نشد!",
    "قیمت هر واحد ۱۲٬۵۰۰ تومان و تخفیف ویژه ۱۵٪ است، یعنی مبلغ نهایی برابر ۱۰٬۶۲۵ تومان خواهد بود.",
    "فصل ۳ کتاب «Introduction to Algorithms» درباره الگوریتم‌های مرتب‌سازی است؛ صفحه‌های 45 تا 67 را بخوانید.",
    "آیا فایل report_final.pdf را دریافت کردید؟ لطفاً تا ساعت ۱۸:۳۰ پاسخ دهید.",
    "دمای هوا در روز ۲۱ اسفند به 12.5 درجه رسید و سرعت باد (km/h) حدود ۴۰ بود.",
]

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(n) -> str:
    return str(n).translate(FA_DIGITS)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def sub_seed(seed: int, purpose: str) -> int:
    """Independent deterministic stream per purpose, so changing one
    degradation does not reshuffle the others."""
    h = hashlib.sha256(f"{seed}:{purpose}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{a.shape}|{a.dtype}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def font_status(path: str) -> dict:
    """Does the file exist, load, and contain the Persian-specific letters?"""
    st = {"path": path, "exists": os.path.exists(path), "name": None,
          "persian_letters": False, "missing": None}
    if not st["exists"]:
        return st
    try:
        f = fitz.Font(fontfile=path)
    except Exception as e:  # pragma: no cover - diagnostic only
        st["error"] = str(e)
        return st
    st["name"] = f.name
    missing = [c for c in PERSIAN_SPECIFIC + "ابتثجحخدذرزسشصضطظعغفقلمنوه" if not f.has_glyph(ord(c))]
    st["missing"] = "".join(missing)
    st["persian_letters"] = not missing
    return st


def resolve_class_font(font_class: str) -> tuple[str, str] | None:
    for name, path in FONT_CANDIDATES.get(font_class, []):
        if font_status(path)["persian_letters"]:
            return name, path
    return None


def make_spec(font_class: str, *, preset: str = "clean", seed: int = 1,
              layout: str | None = None, name: str | None = None, **over) -> dict:
    """Build a full, JSON-serialisable page spec from class defaults."""
    found = resolve_class_font(font_class)
    if found is None:
        raise FontUnavailable(font_class)
    d = copy.deepcopy(CLASS_DEFAULTS[font_class])
    spec = {
        "generator_version": GENERATOR_VERSION,
        "name": name or f"{font_class}_{layout or d['layout']}_{preset}_s{seed}",
        "font_class": font_class,
        "font_name": found[0],
        "font": _rel(found[1]),
        "body_font": _rel(d.pop("body_font", found[1])),
        "layout": layout or d.pop("layout"),
        "justify": d.pop("justify"),
        "size": d.pop("size"),
        "leading": d.pop("leading"),
        "margins": [62.0, 62.0, 64.0, 64.0],   # left, right, top, bottom (pt)
        "text_source": "embedded",
        "seed": int(seed),
        "preset": preset,
        "degrade": dict(DEGRADE_PRESETS[preset]) if preset in DEGRADE_PRESETS else {},
    }
    d.pop("layout", None)
    for k, v in over.items():
        if k == "degrade":
            spec["degrade"].update(v)
        else:
            spec[k] = v
    return spec


class FontUnavailable(RuntimeError):
    pass


def _rel(path: str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _abs(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


# ---------------------------------------------------------------------------
# Text sources and tokenisation
# ---------------------------------------------------------------------------
def book_paragraphs(slug: str) -> list[str]:
    """Paragraphs from books/<slug>/text/*.md, read at generation time only.
    Never copied into tracked files. Private books are refused."""
    if slug in PRIVATE_SLUGS:
        raise ValueError(f"{slug} is a private document and may not be used as a text source")
    paras: list[str] = []
    for p in sorted((ROOT / "books" / slug / "text").glob("[0-9][0-9][0-9][0-9].md")):
        for block in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")):
            t = re.sub(r"^[#>*\-\s]+", "", block.strip(), flags=re.M)
            t = re.sub(r"\[\^?\d+\]:?", "", t)
            t = t.replace("---", " ").replace("*", "")
            t = " ".join(t.split())
            if len(t.split()) >= 8:
                paras.append(t)
    return paras


_STRONG_L = re.compile(r"[A-Za-zÀ-ɏ]")
_STRONG_R = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_DIGIT = re.compile(r"[0-9۰-۹٠-٩]")
_DIGITS_ONLY = re.compile(r"^[0-9۰-۹٠-٩]+$")


def _is_strong_r(ch: str) -> bool:
    # Arabic-block letters are strong R/AL; Arabic-Indic digits (06F0..06F9)
    # are EN and the Arabic punctuation marks are neutral.
    return bool(_STRONG_R.match(ch)) and unicodedata.bidirectional(ch) in ("R", "AL")


def token_tags(tok: str) -> list[str]:
    tags = []
    if "‌" in tok:
        tags.append("zwnj")
    if _DIGIT.search(tok):
        tags.append("digit")
    if _STRONG_L.search(tok):
        tags.append("latin")
    if any(unicodedata.category(c).startswith("P") or c in "٪%/" for c in tok):
        tags.append("punct")
    if any(_is_strong_r(c) for c in tok):
        tags.append("persian")
    return tags


def ltr_flags(tokens: list[str]) -> list[bool]:
    """Approximate bidi: a token sits in an embedded LTR run if it contains a
    strong L character, or it has no strong character and the previous strong
    character in the paragraph is L (rule W7 turns EN after L into L)."""
    out, last = [], "R"
    for t in tokens:
        has_l = bool(_STRONG_L.search(t))
        has_r = any(_is_strong_r(c) for c in t)
        if has_l:
            out.append(True)
        elif has_r:
            out.append(False)
        else:
            out.append(last == "L")
        for c in t:
            if _STRONG_L.match(c):
                last = "L"
            elif _is_strong_r(c):
                last = "R"
    return out


# ---------------------------------------------------------------------------
# Composition: page plan of blocks (each block = one insert_htmlbox call)
# ---------------------------------------------------------------------------
class TextPool:
    """Cycles through a seeded shuffle of a list."""

    def __init__(self, items, rng: random.Random):
        self.items = list(items)
        self.rng = rng
        self.order: list = []

    def next(self):
        if not self.order:
            self.order = self.items[:]
            self.rng.shuffle(self.order)
        return self.order.pop()


def _article_text(tpl: str, rng: random.Random) -> str:
    return tpl.format(
        date=f"{fa(rng.randint(1395, 1403))}/{fa(rng.randint(1, 12)):0>2}/{fa(rng.randint(1, 29)):0>2}".replace("0", "۰"),
        amount=fa(f"{rng.randint(12, 980) * 1000:,}").replace(",", "٬"),
        n=fa(rng.randint(2, 36)), m=fa(rng.randint(3, 15)))


def _stream(spec: dict, rng: random.Random):
    """Infinite generator of body paragraphs (strings) for the layout."""
    lay = spec["layout"]
    src = spec.get("text_source", "embedded")
    prose = PROSE
    if src.startswith("book:"):
        prose = book_paragraphs(src.split(":", 1)[1]) or PROSE
    if lay == "mixed":
        pool = TextPool(MIXED, rng)
        while True:
            yield " ".join(pool.next() for _ in range(rng.randint(2, 3)))
    elif lay == "articles":
        pool = TextPool(ARTICLES, rng)
        n = rng.randint(3, 40)
        while True:
            yield f"ماده ({fa(n)}) - " + _article_text(pool.next(), rng)
            n += 1
    else:
        pool = TextPool(prose, rng)
        while True:
            yield pool.next()


def _css(font_path: str, family: str, size: float, leading: float, align: str,
         indent: float = 0.0) -> str:
    base = os.path.basename(font_path)
    return (f"@font-face{{font-family:{family};src:url('{base}');}}"
            f"body{{font-family:{family};font-size:{size:.2f}pt;line-height:{leading:.3f};"
            f"margin:0;padding:0;}}"
            f"p{{margin:0;padding:0;text-align:{align};text-indent:{indent:.1f}pt;}}")


MAX_TOKENS = 4095
FIT_SLACK = 1.0   # pt of extra room below each planned block (float safety)


def id_color(tid: int) -> str:
    v = tid + 1
    if v > MAX_TOKENS:
        raise ValueError("too many tokens on one page for the colour id code")
    return f"#{(v >> 8) & 15:02x}{(v >> 4) & 15:02x}{v & 15:02x}"


def decode_color(c: int) -> int:
    """rawdict sRGB int or pixel RGB int -> token id (-1 if not a token)."""
    r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
    if r > 15 or g > 15 or b > 15:
        return -1
    return (r << 8 | g << 4 | b) - 1


def _block_html(tokens: list[str], first_id: int) -> str:
    parts = []
    for i, t in enumerate(tokens):
        col = id_color(first_id + i)
        parts.append(f'<span style="color:{col}">{html.escape(t)}</span>')
    return '<p dir="rtl">' + " ".join(parts) + "</p>"


def _try_place(page: fitz.Page, rect: fitz.Rect, block: dict, tokens: list[str],
               first_id: int):
    css = _css(block["font"], "F", block["size"], block["leading"], block["align"], block.get("indent", 0.0))
    arch = fitz.Archive(os.path.dirname(block["font"]))
    spare, _scale = page.insert_htmlbox(rect, _block_html(tokens, first_id),
                                        css=css, archive=arch, scale_low=1)
    return spare


def _fits(rect: fitz.Rect, block: dict, tokens: list[str]) -> float:
    """Height used if `tokens` fit in `rect`, else -1 (trial on a scratch page)."""
    d = fitz.open()
    p = d.new_page(width=PAGE_W, height=PAGE_H)
    spare = _try_place(p, rect, block, tokens, 0)
    d.close()
    return -1.0 if spare < 0 else rect.height - spare


def _fit_prefix(rect: fitz.Rect, block: dict, tokens: list[str], min_words: int = 4):
    """Largest token prefix that fits; returns (n, used_height)."""
    used = _fits(rect, block, tokens)
    if used >= 0:
        return len(tokens), used
    lo, hi, best = min_words, len(tokens) - 1, (0, 0.0)
    while lo <= hi:
        mid = (lo + hi) // 2
        u = _fits(rect, block, tokens[:mid])
        if u >= 0:
            best = (mid, u)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def plan_page(spec: dict) -> list[dict]:
    """Decide block rects and token lists. Deterministic given the spec."""
    rng = random.Random(sub_seed(spec["seed"], "text"))
    left, right, top, bottom = spec["margins"]
    x0, x1 = left, PAGE_W - right
    y, ybot = top, PAGE_H - bottom
    size, lead = spec["size"], spec["leading"]
    body_font = _abs(spec["body_font"])
    cls_font = _abs(spec["font"])
    align = "justify" if spec["justify"] else "right"
    lay = spec["layout"]
    blocks: list[dict] = []
    gap = size * 0.6

    def body_block(font=body_font, kind="body", sz=size, ld=lead, al=align, indent=size * 1.5):
        return dict(kind=kind, font=font, size=sz, leading=ld, align=al, indent=indent)

    # Footnote block reserved at the page bottom first; laid out last in
    # reading order.
    foot = None
    if lay == "footnote":
        fsz = round(size * 0.78, 2)
        nf = rng.randint(2, 3)
        ftext = " ".join(f"({fa(i + 1)}) {FOOTNOTES[(i + rng.randint(0, 9)) % len(FOOTNOTES)]}" for i in range(nf))
        fb = dict(kind="footnote", font=body_font, size=fsz, leading=1.5, align=align, indent=0.0)
        toks = ftext.split()
        h = _fits(fitz.Rect(x0, 0, x1, PAGE_H), fb, toks)
        fy0 = ybot - h
        fb.update(rect=[x0, fy0 - FIT_SLACK, x1, ybot], tokens=toks)
        foot = fb
        rule_y = fy0 - size * 0.5
        ybot = rule_y - size * 0.6
        foot["rule"] = [x1 - (x1 - x0) * 0.33, rule_y, x1, rule_y]

    if lay == "verse":
        hb = dict(kind="heading", font=cls_font, size=size * 1.3, leading=1.6, align="center", indent=0.0)
        toks = HEADINGS[rng.randrange(len(HEADINGS))].split()
        u = _fits(fitz.Rect(x0, y, x1, ybot), hb, toks)
        hb.update(rect=[x0, y, x1, y + u + FIT_SLACK], tokens=toks)
        blocks.append(hb)
        y += u + gap
        mid = (x0 + x1) / 2
        gutter = (x1 - x0) * 0.08
        pool = TextPool(VERSES, rng)
        vb = dict(kind="verse", font=cls_font, size=size, leading=lead, align="center", indent=0.0)
        one_line = None
        misses = 0
        while misses < 2 * len(VERSES):
            h1, h2 = pool.next()
            r1 = fitz.Rect(mid + gutter / 2, y, x1, ybot)
            r2 = fitz.Rect(x0, y, mid - gutter / 2, ybot)
            u1 = _fits(r1, vb, h1.split())
            u2 = _fits(r2, vb, h2.split())
            if u1 < 0 or u2 < 0:
                break   # no vertical room left
            if one_line is None:
                one_line = _fits(fitz.Rect(x0, 0, x1, PAGE_H), vb, ["و"])
            if abs(u1 - one_line) > 0.5 or abs(u2 - one_line) > 0.5:
                misses += 1
                continue   # a hemistich wraps in its column at this size: skip couplet
            blocks.append(dict(vb, rect=[r1.x0, y, r1.x1, y + u1 + FIT_SLACK], tokens=h1.split(), hemistich=1))
            blocks.append(dict(vb, rect=[r2.x0, y, r2.x1, y + u2 + FIT_SLACK], tokens=h2.split(), hemistich=2))
            y += u1 + size * 0.15
    else:
        stream = _stream(spec, rng)
        hpool = TextPool(HEADINGS, rng)
        first = True
        while y < ybot - size * lead:
            if lay == "headings" and (first or rng.random() < 0.35):
                big = first
                hb = dict(kind="heading", font=cls_font, size=size * (1.6 if big else 1.25),
                          leading=1.5, align="center" if big else "right", indent=0.0)
                toks = hpool.next().split()
                u = _fits(fitz.Rect(x0, y, x1, ybot), hb, toks)
                if u < 0:
                    break
                hb.update(rect=[x0, y, x1, y + u + FIT_SLACK], tokens=toks)
                blocks.append(hb)
                y += u + gap
            first = False
            fnt = cls_font if lay not in ("headings",) else body_font
            b = body_block(font=fnt)
            toks = next(stream).split()
            n, u = _fit_prefix(fitz.Rect(x0, y, x1, ybot), b, toks)
            if n == 0:
                break
            b.update(rect=[x0, y, x1, y + u + FIT_SLACK], tokens=toks[:n])
            blocks.append(b)
            y += u + gap
            if n < len(toks):
                break
    if foot is not None:
        blocks.append(foot)
    # Global token ids in reading order.
    nid = 0
    for b in blocks:
        b["first_id"] = nid
        nid += len(b["tokens"])
    return blocks


def render_plan(blocks: list[dict]) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    for b in blocks:
        spare = _try_place(page, fitz.Rect(b["rect"]), b, b["tokens"], b["first_id"])
        if spare < 0:
            raise RuntimeError(f"planned block no longer fits: {b['kind']} @ {b['rect']}")
    return doc


def draw_rules(page: fitz.Page, blocks: list[dict]) -> None:
    """Non-text layout ink (footnote separator). Drawn after the id render."""
    for b in blocks:
        if b.get("rule"):
            r = b["rule"]
            page.draw_line((r[0], r[1]), (r[2], r[3]), color=(0, 0, 0), width=0.6)


# ---------------------------------------------------------------------------
# Truth extraction from the PDF
# ---------------------------------------------------------------------------
def _flat_chars(page: fitz.Page):
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                for c in s["chars"]:
                    out.append(dict(c=c["c"], bbox=c["bbox"], origin=c["origin"],
                                    color=s["color"], font=s["font"], size=s["size"]))
    return out


def extract_words(page: fitz.Page, blocks: list[dict]) -> tuple[list[dict], dict]:
    ntok = sum(len(b["tokens"]) for b in blocks)
    per: list[list[dict]] = [[] for _ in range(ntok)]
    notes = {"zwnj_reported_as_space": 0, "untagged_glyphs": 0}
    for a in _flat_chars(page):
        tid = decode_color(a["color"])
        if 0 <= tid < ntok:
            per[tid].append(a)
        elif not a["c"].isspace():
            notes["untagged_glyphs"] += 1
    tok_block = []
    for bi, b in enumerate(blocks):
        tok_block += [bi] * len(b["tokens"])
    words = []
    for tid in range(ntok):
        b = blocks[tok_block[tid]]
        text = b["tokens"][tid - b["first_id"]]
        chars = per[tid]
        if "‌" in text:
            notes["zwnj_reported_as_space"] += sum(1 for c in chars if c["c"] == " ")
        ink = [c for c in chars if not c["c"].isspace() and c["bbox"][2] - c["bbox"][0] > 1e-3]
        w = dict(id=tid, text=text, block=tok_block[tid], kind=b["kind"],
                 hemistich=b.get("hemistich"), size=b["size"])
        if not ink:
            w.update(rect=None, baseline=None, fonts=[], missing=True)
        else:
            xs0 = min(c["bbox"][0] for c in ink); ys0 = min(c["bbox"][1] for c in ink)
            xs1 = max(c["bbox"][2] for c in ink); ys1 = max(c["bbox"][3] for c in ink)
            bases = sorted(c["origin"][1] for c in ink)
            # Layout slot: which line of its block the word sits on. Glyph
            # origins carry GPOS y-offsets (nastaliq cascades move them by up
            # to ~0.8 em), so a nearest-baseline clustering is unreliable;
            # the block's line pitch (size * leading) is exact.
            pitch = b["size"] * b["leading"]
            med = bases[len(bases) // 2]
            slot = int((med - b["rect"][1]) // pitch)
            w.update(rect=[xs0, ys0, xs1, ys1], baseline=med,
                     baseline_spread=bases[-1] - bases[0],
                     slot_y=b["rect"][1] + slot * pitch,
                     origins=[round(c["origin"][1], 1) for c in ink],
                     fonts=sorted({c["font"] for c in ink}), missing=False,
                     extracted=" ".join("".join(c["c"] for c in chars).split()),
                     _ink=ink)
        words.append(w)
    _visual_segments(words)
    return words, notes


def _visual_segments(words: list[dict]) -> None:
    """Split each word into visually contiguous glyph runs.

    Bidi reordering can place a word's glyphs in more than one place on the
    line: in "«Introduction to Algorithms»" inside RTL text the guillemet of
    the first token is displayed at the far end of the LTR run, so the union
    of its glyph boxes spans the whole phrase. A segment is a maximal run of
    the word's glyphs in x order, on its own layout slot, not interrupted by
    another word's glyph."""
    byslot: dict = {}
    for w in words:
        if w["missing"]:
            continue
        key = (w["block"], round(w["slot_y"], 2))
        for c in w["_ink"]:
            byslot.setdefault(key, []).append((c["bbox"][0], c["bbox"][2], w["id"], c["bbox"]))
    segs: dict[int, list[list[float]]] = {}
    for items in byslot.values():
        items.sort(key=lambda t: (t[0], t[1]))
        prev = None
        for _x0, _x1, tid, bb in items:
            if prev == tid:
                r = segs[tid][-1]
                r[0] = min(r[0], bb[0]); r[1] = min(r[1], bb[1]); r[2] = max(r[2], bb[2]); r[3] = max(r[3], bb[3])
            else:
                segs.setdefault(tid, []).append(list(bb))
            prev = tid
    for w in words:
        if w["missing"]:
            continue
        ss = sorted(segs.get(w["id"], [w["rect"]]), key=lambda r: -r[2])   # right -> left
        w["segments"] = ss
        # The "main" segment (most glyph width) for ordering purposes.
        w["main_rect"] = max(ss, key=lambda r: r[2] - r[0])
        del w["_ink"]


def group_lines(words: list[dict], blocks: list[dict]) -> list[dict]:
    """Cluster words into printed lines by baseline; verse hemistichs sharing
    a baseline become one line with two segments."""
    present = [w for w in words if not w["missing"]]
    present.sort(key=lambda w: (w["slot_y"], w["id"]))
    lines: list[list[dict]] = []
    for w in present:
        if lines:
            ref = lines[-1]
            tol = 0.3 * min(w["size"], ref[0]["size"])
            if abs(w["slot_y"] - ref[0]["slot_y"]) <= tol:
                ref.append(w)
                continue
        lines.append([w])
    for ws in lines:
        # Nominal baseline of the printed line = most common glyph origin y
        # (in nastaliq most glyphs are displaced; the mode is the line).
        vals, cnt = np.unique(np.round([o for w in ws for o in w["origins"]], 1), return_counts=True)
        base = float(vals[np.argmax(cnt)])
        for w in ws:
            w["line_baseline"] = base
    out = []
    for li, ws in enumerate(lines):
        ws.sort(key=lambda w: w["id"])
        for vis, w in enumerate(sorted(ws, key=lambda w: -(w["main_rect"][0] + w["main_rect"][2]) / 2)):
            w["visual_index"] = vis
        for w in ws:
            w["line"] = li
        segs = []
        for w in ws:
            if not segs or w["block"] != segs[-1][0]:
                segs.append((w["block"], []))
            segs[-1][1].append(w["id"])
        kinds = sorted({w["kind"] for w in ws})
        out.append(dict(index=li, kind=kinds[0] if len(kinds) == 1 else "+".join(kinds),
                        word_ids=[w["id"] for w in ws],
                        segments=[s for _b, s in segs],
                        text=" ".join(w["text"] for w in ws),
                        rect=[min(w["rect"][0] for w in ws), min(w["rect"][1] for w in ws),
                              max(w["rect"][2] for w in ws), max(w["rect"][3] for w in ws)],
                        baseline=ws[0]["line_baseline"]))
    for w in words:
        w.setdefault("line", None)
        w.setdefault("visual_index", None)
    return out


# ---------------------------------------------------------------------------
# Geometric transform (pixel space). forward: clean -> degraded.
# ---------------------------------------------------------------------------
class PageTransform:
    """Composition P o R o W in pixel coordinates of a W x H image.

    W (warp): y' = y + tan(theta(y)) * (x - cx) + bow * ((x - cx)/(W/2))^2 * (y/H)
              with theta(y) linear from warp_top_deg (y=0) to warp_bottom_deg
              (y=H), or, when warp_mid_deg is given, the quadratic through
              (0, top), (H/2, mid), (H, bottom) -- e.g. E18's hadpi p2 profile
              0.1 / 2.2 / 3.3 deg. Local line slope varies down the page.
    R: rotation by rotate_deg about the image centre.
    P: homography moving each image corner by up to perspective*W (seeded).
    """

    def __init__(self, width: int, height: int, deg: dict, seed: int):
        self.W, self.H = float(width), float(height)
        self.cx, self.cy = self.W / 2, self.H / 2
        self.t0 = math.radians(deg.get("warp_top_deg", 0.0))
        self.t1 = math.radians(deg.get("warp_bottom_deg", 0.0))
        mid = deg.get("warp_mid_deg")
        self.tm = None if mid is None else math.radians(mid)
        self.bow = float(deg.get("warp_bow_px", 0.0))
        a = math.radians(deg.get("rotate_deg", 0.0))
        self.ca, self.sa = math.cos(a), math.sin(a)
        p = float(deg.get("perspective", 0.0))
        self.Hm = None
        if p > 0:
            rng = np.random.default_rng(sub_seed(seed, "perspective"))
            src = np.array([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]], float)
            dst = src + rng.uniform(-p, p, size=(4, 2)) * self.W
            self.Hm = _homography(src, dst)
            self.Hinv = np.linalg.inv(self.Hm)
        self.identity = (self.t0 == 0 and self.t1 == 0 and not self.tm and self.bow == 0 and a == 0 and self.Hm is None)

    def _slope(self, y):
        t = y / self.H
        if self.tm is None:
            return np.tan(self.t0 + (self.t1 - self.t0) * t)
        # Lagrange quadratic through t = 0, 0.5, 1.
        th = (self.t0 * 2 * (t - 0.5) * (t - 1) - self.tm * 4 * t * (t - 1) + self.t1 * 2 * t * (t - 0.5))
        return np.tan(th)

    def _warp_dy(self, x, y):
        u = x - self.cx
        return self._slope(y) * u + self.bow * (u / (self.W / 2)) ** 2 * (y / self.H)

    def forward(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        y1 = y + self._warp_dy(x, y)
        x1 = x
        x2 = self.cx + self.ca * (x1 - self.cx) - self.sa * (y1 - self.cy)
        y2 = self.cy + self.sa * (x1 - self.cx) + self.ca * (y1 - self.cy)
        if self.Hm is not None:
            x2, y2 = _apply_h(self.Hm, x2, y2)
        return x2, y2

    def inverse(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if self.Hm is not None:
            x, y = _apply_h(self.Hinv, x, y)
        x1 = self.cx + self.ca * (x - self.cx) + self.sa * (y - self.cy)
        y1 = self.cy - self.sa * (x - self.cx) + self.ca * (y - self.cy)
        yy = y1.copy()
        for _ in range(12):          # contraction: |d(dy)/dy| ~ 1e-2
            yy = y1 - self._warp_dy(x1, yy)
        return x1, yy


def _homography(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def _apply_h(Hm, x, y):
    d = Hm[2, 0] * x + Hm[2, 1] * y + Hm[2, 2]
    return (Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) / d, (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2]) / d


def warp_image(img: np.ndarray, tf: PageTransform, fill: float, nearest: bool = False) -> np.ndarray:
    """Resample `img` (H x W) under tf: out(p) = img(tf.inverse(p))."""
    if tf.identity:
        return img.copy()
    H, W = img.shape
    out = np.empty_like(img)
    xs = np.arange(W, dtype=np.float64) + 0.5
    step = 256
    for r0 in range(0, H, step):
        ys = np.arange(r0, min(H, r0 + step), dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        sx, sy = tf.inverse(gx, gy)
        sx -= 0.5; sy -= 0.5                     # continuous -> index space
        if nearest:
            ix = np.rint(sx).astype(np.int64); iy = np.rint(sy).astype(np.int64)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            blk = np.full(ix.shape, fill, dtype=img.dtype)
            blk[ok] = img[iy[ok], ix[ok]]
        else:
            x0 = np.floor(sx).astype(np.int64); y0 = np.floor(sy).astype(np.int64)
            fx = (sx - x0).astype(np.float32); fy = (sy - y0).astype(np.float32)

            def px(yy, xx):
                ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                v = np.full(xx.shape, fill, dtype=np.float32)
                v[ok] = img[yy[ok], xx[ok]]
                return v
            blk = ((px(y0, x0) * (1 - fx) + px(y0, x0 + 1) * fx) * (1 - fy)
                   + (px(y0 + 1, x0) * (1 - fx) + px(y0 + 1, x0 + 1) * fx) * fy)
        out[r0:r0 + len(ys)] = blk
    return out


# ---------------------------------------------------------------------------
# Photometric degradations (float32 image, 1 = white paper, 0 = black ink)
# ---------------------------------------------------------------------------
def _shift_min(a: np.ndarray, r: int) -> np.ndarray:
    out = a.copy()
    for _ in range(r):
        m = out.copy()
        m[:, 1:] = np.minimum(m[:, 1:], out[:, :-1]); m[:, :-1] = np.minimum(m[:, :-1], out[:, 1:])
        n = m.copy()
        n[1:, :] = np.minimum(n[1:, :], m[:-1, :]); n[:-1, :] = np.minimum(n[:-1, :], m[1:, :])
        out = n
    return out


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[axis] = (r, r)
        p = np.pad(a, pad, mode="edge")
        acc = np.zeros_like(a)
        n = a.shape[axis]
        for i, kv in enumerate(k):
            sl = [slice(None), slice(None)]; sl[axis] = slice(i, i + n)
            acc += kv * p[tuple(sl)]
        a = acc
    return a


def _smooth_field(shape, cells: tuple[int, int], rng) -> np.ndarray:
    """Low-frequency random field in [0,1] by bilinear upsampling a grid."""
    H, W = shape
    g = rng.random((cells[0] + 1, cells[1] + 1)).astype(np.float32)
    yy = np.linspace(0, cells[0], H, dtype=np.float32)
    xx = np.linspace(0, cells[1], W, dtype=np.float32)
    y0 = np.minimum(yy.astype(int), cells[0] - 1); x0 = np.minimum(xx.astype(int), cells[1] - 1)
    fy = (yy - y0)[:, None]; fx = (xx - x0)[None, :]
    return ((g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx) * (1 - fy)
            + (g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx) * fy)


def print_side(img: np.ndarray, deg: dict, seed: int) -> np.ndarray:
    r = int(deg.get("ink_spread_px", 0))
    if r > 0:
        img = _shift_min(img, r)
    a = float(deg.get("ink_dropout", 0.0))
    if a > 0:
        rng = np.random.default_rng(sub_seed(seed, "dropout"))
        f = _smooth_field(img.shape, (60, 42), rng)
        m = np.clip((f - (1 - a)) / max(a, 1e-6), 0, 1) ** 0.7
        img = img + (1 - img) * m * 0.9
        speck = rng.random(img.shape, dtype=np.float32) < a * 0.04
        img = np.where(speck, np.maximum(img, 0.85), img)
    return img


def camera_side(img: np.ndarray, deg: dict, seed: int) -> np.ndarray:
    H, W = img.shape
    g = float(deg.get("light_gradient", 0.0))
    if g > 0:
        rng = np.random.default_rng(sub_seed(seed, "light"))
        ang = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        lin = ((xx / W - 0.5) * math.cos(ang) + (yy / H - 0.5) * math.sin(ang)) + 0.5
        rad = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2
        L = 1.0 - g * np.clip(0.65 * lin + 0.35 * rad, 0, 1)
        img = img * L
    img = gaussian_blur(img, float(deg.get("blur_sigma_px", 0.0)))
    s = float(deg.get("noise_sigma", 0.0))
    if s > 0:
        rng = np.random.default_rng(sub_seed(seed, "noise"))
        img = img + rng.normal(0, s, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def jpeg_roundtrip(u8: np.ndarray, q: int) -> np.ndarray:
    H, W = u8.shape
    pm = fitz.Pixmap(fitz.csGRAY, W, H, np.ascontiguousarray(u8).tobytes(), False)
    back = fitz.Pixmap(pm.tobytes("jpeg", jpg_quality=int(q)))
    return np.frombuffer(back.samples, np.uint8).reshape(back.h, back.w).copy()


# ---------------------------------------------------------------------------
# Overlays (drawn on the paper before the warp; never change truth)
# ---------------------------------------------------------------------------
def draw_overlays(page: fitz.Page, deg: dict, words: list[dict], lines: list[dict], seed: int) -> list[dict]:
    rng = random.Random(sub_seed(seed, "overlays"))
    out = []
    body_lines = [ln for ln in lines if ln["kind"] in ("body", "verse")] or lines
    if deg.get("stamp") and body_lines:
        ln = body_lines[max(0, len(body_lines) - 1 - rng.randint(1, 5))]
        r = rng.uniform(42, 56)
        c = fitz.Point(rng.uniform(150, 300), (ln["rect"][1] + ln["rect"][3]) / 2)
        col = (0.22, 0.28, 0.62)
        page.draw_circle(c, r, color=col, width=2.2, stroke_opacity=0.85)
        page.draw_circle(c, r * 0.74, color=col, width=1.0, stroke_opacity=0.85)
        for k in range(36):
            t = 2 * math.pi * k / 36
            p0 = c + (math.cos(t) * r * 0.80, math.sin(t) * r * 0.80)
            p1 = c + (math.cos(t) * r * 0.92, math.sin(t) * r * 0.92)
            page.draw_line(p0, p1, color=col, width=0.9, stroke_opacity=0.85)
        star = []
        for k in range(10):
            t = -math.pi / 2 + math.pi * k / 5
            rr = r * (0.42 if k % 2 == 0 else 0.18)
            star.append(c + (math.cos(t) * rr, math.sin(t) * rr))
        page.draw_polyline(star + [star[0]], color=col, width=1.2, stroke_opacity=0.85)
        out.append(dict(kind="stamp", rect_pt=[c.x - r, c.y - r, c.x + r, c.y + r]))
    if deg.get("signature") and body_lines:
        tail = body_lines[-3:]
        ya, yb = tail[0]["rect"][1], tail[-1]["rect"][3]
        xa = rng.uniform(90, 200)
        pts = []
        for k in range(3):
            p0 = fitz.Point(xa + k * 40, yb - rng.uniform(0, 8))
            p1 = fitz.Point(xa + k * 40 + rng.uniform(20, 60), ya - rng.uniform(5, 20))
            p2 = fitz.Point(xa + k * 40 + rng.uniform(40, 90), yb + rng.uniform(5, 20))
            p3 = fitz.Point(xa + k * 40 + rng.uniform(70, 130), ya + rng.uniform(0, 10))
            page.draw_bezier(p0, p1, p2, p3, color=(0.05, 0.05, 0.12), width=1.4)
            pts += [p0, p3]
        page.draw_line(fitz.Point(xa - 10, yb + 4), fitz.Point(xa + 170, ya + 6), color=(0.05, 0.05, 0.12), width=1.1)
        out.append(dict(kind="signature", rect_pt=[xa - 10, ya - 20, xa + 220, yb + 20]))
    nu = int(deg.get("underlines", 0))
    cands = [w for w in words if not w["missing"] and w["kind"] == "body" and len(w["text"]) >= 3]
    for w in rng.sample(cands, min(nu, len(cands))):
        y = w["line_baseline"] + 0.18 * w["size"]
        page.draw_line((w["rect"][0], y), (w["rect"][2], y), color=(0.1, 0.1, 0.1), width=0.9)
        out.append(dict(kind="underline", word_id=w["id"], rect_pt=[w["rect"][0], y - 0.5, w["rect"][2], y + 0.5]))
    for _ in range(int(deg.get("margin_marks", 0))):
        ln = rng.choice(lines)
        side = rng.choice(("left", "right"))
        x = rng.uniform(18, 40) if side == "left" else PAGE_W - rng.uniform(18, 40)
        y0, y1 = ln["rect"][1], ln["rect"][3]
        kind = rng.choice(("bar", "check", "cross"))
        col = (0.15, 0.15, 0.15)
        if kind == "bar":
            page.draw_line((x, y0 - 4), (x, y1 + 4), color=col, width=1.3)
        elif kind == "check":
            page.draw_polyline([(x - 6, (y0 + y1) / 2), (x - 2, y1), (x + 7, y0 - 3)], color=col, width=1.2)
        else:
            page.draw_line((x - 5, y0), (x + 5, y1), color=col, width=1.1)
            page.draw_line((x - 5, y1), (x + 5, y0), color=col, width=1.1)
        out.append(dict(kind=f"margin_{kind}", rect_pt=[x - 7, y0 - 4, x + 7, y1 + 4]))
    return out


# ---------------------------------------------------------------------------
# Geometry serialisation
# ---------------------------------------------------------------------------
def _rect_polygon_px(r, n: int = 7) -> np.ndarray:
    """Closed-order polygon (clockwise from top-left) sampled along edges."""
    x0, y0, x1, y1 = r
    top = [(x0 + (x1 - x0) * i / (n - 1), y0) for i in range(n)]
    right = [(x1, y0 + (y1 - y0) * 0.5)]
    bot = [(x1 - (x1 - x0) * i / (n - 1), y1) for i in range(n)]
    left = [(x0, y1 - (y1 - y0) * 0.5)]
    return np.array(top + right + bot + left, float)


def _norm(pts: np.ndarray, W: int, H: int) -> list[list[float]]:
    return [[round(float(x) / W, 6), round(float(y) / H, 6)] for x, y in pts]


def _aabb(pts: np.ndarray, W: int, H: int) -> list[float]:
    return [round(float(pts[:, 0].min()) / W, 6), round(float(pts[:, 1].min()) / H, 6),
            round(float(pts[:, 0].max()) / W, 6), round(float(pts[:, 1].max()) / H, 6)]


def _map_pts(tf: PageTransform, pts: np.ndarray) -> np.ndarray:
    x, y = tf.forward(pts[:, 0], pts[:, 1])
    return np.stack([x, y], axis=1)


def ink_bboxes(idmap: np.ndarray, ntok: int, W: int, H: int) -> list:
    """Per-token ink bbox (normalised) from a word-id map (0 = no word)."""
    ys, xs = np.nonzero(idmap)
    ids = idmap[ys, xs] - 1
    out = [None] * ntok
    if not len(ids):
        return out
    order = np.argsort(ids, kind="stable")
    ids, xs, ys = ids[order], xs[order], ys[order]
    bounds = np.flatnonzero(np.diff(ids)) + 1
    for seg_x, seg_y, seg_i in zip(np.split(xs, bounds), np.split(ys, bounds), np.split(ids, bounds)):
        t = int(seg_i[0])
        if 0 <= t < ntok:
            out[t] = [round(float(seg_x.min()) / W, 6), round(float(seg_y.min()) / H, 6),
                      round(float(seg_x.max() + 1) / W, 6), round(float(seg_y.max() + 1) / H, 6)]
    return out


# ---------------------------------------------------------------------------
# Main entry: generate one page
# ---------------------------------------------------------------------------
def render_gray(page: fitz.Page) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()


def render_idmap(page: fitz.Page) -> np.ndarray:
    old = fitz.TOOLS.show_aa_level()
    fitz.TOOLS.set_aa_level(0)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), colorspace=fitz.csRGB, alpha=False)
    finally:
        fitz.TOOLS.set_aa_level(old.get("graphics", 8) if isinstance(old, dict) else 8)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ids = (r << 8) | (g << 4) | b            # = token id + 1
    ids[(r > 15) | (g > 15) | (b > 15)] = 0  # paper / anything not a token
    return ids


def generate(spec: dict, *, keep_arrays: bool = False) -> dict:
    """Generate one page. Returns {'image': uint8 HxW, 'truth': dict, ...}."""
    spec = copy.deepcopy(spec)
    deg = dict(DEGRADE_KEYS)
    deg.update(spec.get("degrade", {}))
    seed = int(spec["seed"])
    blocks = plan_page(spec)
    black = render_plan(blocks)
    words, notes = extract_words(black[0], blocks)
    lines = group_lines(words, blocks)
    ntok = len(words)

    idmap_clean = render_idmap(black[0])
    draw_rules(black[0], blocks)
    overlays = draw_overlays(black[0], deg, words, lines, seed)
    clean_u8 = render_gray(black[0])
    H, W = clean_u8.shape
    if idmap_clean.shape != clean_u8.shape:
        raise RuntimeError("identity and image renders differ in size")

    img = clean_u8.astype(np.float32) / 255.0
    img = print_side(img, deg, seed)
    tf = PageTransform(W, H, deg, seed)
    img = warp_image(img, tf, fill=1.0)
    idmap = warp_image(idmap_clean, tf, fill=0, nearest=True)
    img = camera_side(img, deg, seed)
    u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    if int(deg.get("jpeg_quality", 0)) > 0:
        u8 = jpeg_roundtrip(u8, int(deg["jpeg_quality"]))

    inks = ink_bboxes(idmap, ntok, W, H)
    ltr_all: list[bool] = []
    for b in blocks:
        ltr_all += ltr_flags(b["tokens"])

    twords = []
    for w in words:
        tw = dict(id=w["id"], reading_index=w["id"], text=w["text"], line=w["line"],
                  visual_index=w["visual_index"], kind=w["kind"], block=w["block"],
                  hemistich=w["hemistich"], ltr=ltr_all[w["id"]], tags=token_tags(w["text"]),
                  fonts=w["fonts"], missing=w["missing"], ink_aabb=inks[w["id"]])
        if not w["missing"]:
            rpx = [v * ZOOM for v in w["rect"]]
            poly = _map_pts(tf, _rect_polygon_px(rpx))
            quad = _map_pts(tf, np.array([[rpx[0], rpx[1]], [rpx[2], rpx[1]],
                                          [rpx[2], rpx[3]], [rpx[0], rpx[3]]], float))
            tw.update(quad=_norm(quad, W, H), polygon=_norm(poly, W, H), aabb=_aabb(poly, W, H),
                      layout_rect_pt=[round(v, 4) for v in w["rect"]],
                      baseline_pt=round(w["line_baseline"], 4),
                      glyph_origin_spread_pt=round(w["baseline_spread"], 3),
                      discontiguous=len(w["segments"]) > 1,
                      segments=[dict(layout_rect_pt=[round(v, 4) for v in sr],
                                     polygon=_norm(_map_pts(tf, _rect_polygon_px([v * ZOOM for v in sr])), W, H),
                                     aabb=_aabb(_map_pts(tf, _rect_polygon_px([v * ZOOM for v in sr])), W, H))
                                for sr in w["segments"]],
                      pdf_extracted_text=w["extracted"])
        twords.append(tw)

    tlines = []
    for ln in lines:
        x0, y0, x1, y1 = [v * ZOOM for v in ln["rect"]]
        by = ln["baseline"] * ZOOM
        n = max(2, int((x1 - x0) / 40) + 1)
        base = np.array([[x1 - (x1 - x0) * i / (n - 1), by] for i in range(n)], float)   # right -> left
        poly = _map_pts(tf, _rect_polygon_px([x0, y0, x1, y1], n=max(3, n)))
        tlines.append(dict(index=ln["index"], kind=ln["kind"], text=ln["text"],
                           word_ids=ln["word_ids"], segments=ln["segments"],
                           baseline=_norm(_map_pts(tf, base), W, H),
                           polygon=_norm(poly, W, H), aabb=_aabb(poly, W, H)))
    tover = []
    for o in overlays:
        rpx = [v * ZOOM for v in o["rect_pt"]]
        tover.append(dict(o, aabb=_aabb(_map_pts(tf, _rect_polygon_px(rpx)), W, H)))
    for b in blocks:
        if b.get("rule"):
            r = b["rule"]
            tover.append(dict(kind="footnote_rule", rect_pt=r,
                              aabb=_aabb(_map_pts(tf, _rect_polygon_px([r[0] * ZOOM, r[1] * ZOOM - 1, r[2] * ZOOM, r[3] * ZOOM + 1])), W, H)))

    fonts = []
    for role, p in (("font", spec["font"]), ("body_font", spec["body_font"])):
        fonts.append(dict(role=role, path=p, sha256=sha256_file(_abs(p)), name=font_status(_abs(p))["name"]))
    truth = dict(
        generator_version=GENERATOR_VERSION, spec=spec, pymupdf=fitz.VersionBind,
        fonts=fonts, page_pt=[PAGE_W, PAGE_H], zoom=ZOOM,
        coords="normalised to the degraded image: x 0..1 left->right, y 0..1 top->bottom",
        image=dict(width=W, height=H, sha256=sha256_array(u8)),
        lines=tlines, words=twords, overlays=tover,
        notes=dict(notes, n_words=ntok, n_missing=sum(1 for w in words if w["missing"]),
                   n_discontiguous=sum(1 for w in words if not w["missing"] and len(w["segments"]) > 1),
                   kashida="MuPDF justifies by widening spaces only; no tatweel is inserted"),
    )
    truth["truth_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in truth.items() if k != "truth_sha256"}).encode()).hexdigest()
    res = dict(image=u8, truth=truth, blocks=blocks)
    if keep_arrays:
        res.update(clean=clean_u8, idmap=idmap, idmap_clean=idmap_clean, transform=tf,
                   doc=black)
    else:
        black.close()
    return res


def save_page(res: dict, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = res["truth"]["spec"]["name"]
    u8 = res["image"]
    png = out_dir / f"{name}.png"
    fitz.Pixmap(fitz.csGRAY, u8.shape[1], u8.shape[0], np.ascontiguousarray(u8).tobytes(), False).save(str(png))
    tj = out_dir / f"{name}.truth.json"
    t = dict(res["truth"], image=dict(res["truth"]["image"], path=_rel(str(png))))
    tj.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    return png, tj


# ---------------------------------------------------------------------------
# Text-error injection (the image is unchanged; only the text given to an
# aligner is corrupted)
# ---------------------------------------------------------------------------
def inject_text_errors(text_or_words, rate: float, seed: int, vocab: list[str] | None = None) -> dict:
    """Corrupt round(rate * n) word positions with substitutions, deletions
    and insertions (roughly equal shares). Deterministic by seed.

    Returns dict(words=[...corrupted...], text=str, ops=[...],
    src_index=[for each output word: source index or None if inserted],
    status=[for each source word: 'kept' | 'sub' | 'del'])."""
    words = text_or_words.split() if isinstance(text_or_words, str) else list(text_or_words)
    rng = random.Random(sub_seed(seed, f"inject:{rate}"))
    vocab = list(vocab) if vocab else sorted({w for p in PROSE for w in p.split()})
    n = len(words)
    k = int(round(rate * n))
    pos = sorted(rng.sample(range(n), min(k, n)))
    ops_at = {p: rng.choice(("sub", "del", "ins")) for p in pos}
    out, src, status, ops = [], [], [], []
    for i, w in enumerate(words):
        op = ops_at.get(i)
        if op == "ins":
            new = rng.choice(vocab)
            out.append(new); src.append(None)
            ops.append(dict(op="ins", before_src=i, new=new))
            op = None
        if op == "sub":
            new = w
            while new == w:
                new = rng.choice(vocab) if rng.random() < 0.6 else _perturb(w, rng)
            out.append(new); src.append(i); status.append("sub")
            ops.append(dict(op="sub", src=i, orig=w, new=new))
        elif op == "del":
            status.append("del")
            ops.append(dict(op="del", src=i, orig=w))
        else:
            out.append(w); src.append(i); status.append("kept")
    return dict(words=out, text=" ".join(out), ops=ops, src_index=src, status=status, rate=rate, seed=seed)


_CONFUSABLE = ["بپتثن", "جچحخ", "دذ", "رزژ", "سش", "صض", "طظ", "عغ", "فق", "کگ", "ی", "هة"]


def _perturb(w: str, rng: random.Random) -> str:
    """Change one letter to a visually confusable one (dot errors)."""
    idx = [i for i, c in enumerate(w) if any(c in g and len(g) > 1 for g in _CONFUSABLE)]
    if not idx:
        return w + "ی"
    i = rng.choice(idx)
    g = next(g for g in _CONFUSABLE if w[i] in g)
    return w[:i] + rng.choice([c for c in g if c != w[i]]) + w[i + 1:]


# ---------------------------------------------------------------------------
# Standard set + manifest
# ---------------------------------------------------------------------------
def standard_specs() -> tuple[list[dict], list[str]]:
    """One clean + one photo page per available font class, plus every
    layout class in the default modern face. Returns (specs, gaps)."""
    specs, gaps = [], []
    for cls in FONT_CANDIDATES:
        if resolve_class_font(cls) is None:
            gaps.append(cls)
            continue
        for preset in ("clean", "photo"):
            specs.append(make_spec(cls, preset=preset, seed=11))
    for lay in LAYOUTS:
        specs.append(make_spec("modern_naskh", layout=lay, preset="clean", seed=23,
                               name=f"layout_{lay}_clean_s23"))
    return specs, gaps


def write_manifest(out_dir: Path = OUT_DIR, path: Path = MANIFEST_PATH) -> dict:
    specs, gaps = standard_specs()
    entries = []
    for s in specs:
        res = generate(s)
        save_page(res, out_dir)
        t = res["truth"]
        entries.append(dict(name=s["name"], spec=s, image_sha256=t["image"]["sha256"],
                            truth_sha256=t["truth_sha256"], fonts=t["fonts"],
                            n_words=t["notes"]["n_words"], n_lines=len(t["lines"])))
        print(f"  {s['name']}: {t['notes']['n_words']} words, {len(t['lines'])} lines")
    man = dict(generator_version=GENERATOR_VERSION, pymupdf=fitz.VersionBind,
               note="Specs, seeds and hashes only. Images regenerate into out/synth/ "
                    "(gitignored). Hashes are valid only for the same font files "
                    "(sha256 recorded) and PyMuPDF version.",
               gaps=gaps, entries=entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def fonts_report() -> list[dict]:
    rows = []
    for cls, cands in FONT_CANDIDATES.items():
        for name, path in cands:
            st = font_status(path)
            rows.append(dict(font_class=cls, font=name, **st))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fonts", help="font availability per class")
    g = sub.add_parser("generate", help="generate one page into out/synth/")
    g.add_argument("--class", dest="font_class", required=True, choices=list(FONT_CANDIDATES))
    g.add_argument("--preset", default="clean", choices=list(DEGRADE_PRESETS))
    g.add_argument("--layout", choices=LAYOUTS)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--text-source", default="embedded", help="'embedded' or 'book:<slug>'")
    g.add_argument("--degrade", default="{}", help="JSON overrides for degradation params")
    sub.add_parser("manifest", help="regenerate the standard set and rewrite tests/data/synth_manifest.json")
    r = sub.add_parser("regen", help="regenerate one manifest entry by name")
    r.add_argument("name")
    inj = sub.add_parser("inject", help="text-error injection on a truth file's line text")
    inj.add_argument("truth")
    inj.add_argument("--rate", type=float, default=0.15)
    inj.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "fonts":
        for row in fonts_report():
            print(f"{row['font_class']:13s} {row['font']:28s} exists={row['exists']!s:5s} "
                  f"persian={row['persian_letters']!s:5s} missing={row.get('missing')!r} {row['path']}")
    elif a.cmd == "generate":
        spec = make_spec(a.font_class, preset=a.preset, seed=a.seed, layout=a.layout,
                         text_source=a.text_source, degrade=json.loads(a.degrade))
        png, tj = save_page(generate(spec))
        print(png); print(tj)
    elif a.cmd == "manifest":
        man = write_manifest()
        print(f"wrote {MANIFEST_PATH} ({len(man['entries'])} entries; gaps={man['gaps']})")
    elif a.cmd == "regen":
        man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        e = next(e for e in man["entries"] if e["name"] == a.name)
        res = generate(e["spec"])
        png, tj = save_page(res)
        ok = res["truth"]["image"]["sha256"] == e["image_sha256"] and res["truth"]["truth_sha256"] == e["truth_sha256"]
        print(png, "hashes match manifest" if ok else "HASH MISMATCH (font files or PyMuPDF changed?)")
    elif a.cmd == "inject":
        t = json.loads(Path(a.truth).read_text(encoding="utf-8"))
        words = [w for ln in t["lines"] for w in ln["text"].split()]
        res = inject_text_errors(words, a.rate, a.seed)
        print(json.dumps(dict(n=len(words), n_ops=len(res["ops"]), ops=res["ops"][:20]), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
