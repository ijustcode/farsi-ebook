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
# Larger invented text pool (text_source="pool"). Every template, filler and
# sentence below was written for this file; none is copied from any book,
# document or page under books/. Templates combine into millions of distinct
# sentences, so a page drawn from the pool almost never repeats a phrase
# longer than a few function words (the small embedded corpus above repeats
# whole paragraphs on one page, which makes text alignment ambiguous in an
# unrealistic way).
# ---------------------------------------------------------------------------
POOL_FILL: dict[str, list[str]] = {
    "org": ["هیئت مدیره", "مدیر عامل", "بازرس قانونی", "مجمع عمومی", "واحد حسابداری", "کمیته فنی",
            "اداره کارگزینی", "دفتر حقوقی", "شورای نظارت", "معاونت اداری", "مسئول انبار",
            "نماینده فروشنده", "کارشناس رسمی دادگستری", "دبیرخانه", "واحد بازرگانی", "مدیر کارخانه",
            "سرپرست کارگاه", "خزانه‌دار", "هیئت داوری", "کمیسیون معاملات", "واحد کنترل کیفیت",
            "اداره تدارکات", "انجمن صنفی", "شورای کارگاه"],
    "person": ["آقای رضایی", "خانم کاظمی", "آقای مهندس صالحی", "دکتر نوروزی", "آقای فرهادی",
               "خانم امینی", "آقای حیدری", "خانم توکلی", "آقای جهانگیری", "آقای ملکی", "خانم یزدانی",
               "آقای شریفی", "خانم دکتر بهرامی", "آقای قنبری", "آقای مهندس سلطانی", "خانم رحیمی",
               "آقای عسگری", "خانم مرادی"],
    "obj": ["گزارش مالی سال گذشته", "صورت‌حساب‌های معوق", "برنامه تعمیرات سالانه", "فهرست اموال منقول",
            "طرح توسعه کارگاه", "قرارداد خرید مواد اولیه", "پرونده استخدام کارکنان جدید",
            "آیین‌نامه انضباطی", "بودجه پیشنهادی سال آینده", "نتایج بازرسی دوره‌ای", "سند انتقال زمین مجاور",
            "صورتجلسه نشست قبلی", "فهرست بدهکاران", "طرح بیمه تکمیلی کارکنان", "نقشه ساختمان اداری",
            "گزارش مصرف سوخت", "پیشنهاد خرید دستگاه بسته‌بندی", "دفتر روزنامه حسابداری",
            "فهرست قیمت‌های تازه", "برنامه نوبت‌کاری تابستان", "گزارش حوادث کارگاه", "پرونده مالیاتی شرکت"],
    "verb": ["را بررسی و تصویب کرد", "را به مجمع عمومی ارائه خواهد داد",
             "را برای اظهار نظر به کمیته فنی ارجاع داد", "را تا پایان ماه آینده تکمیل خواهد کرد",
             "را رد کرد و اصلاح آن را خواستار شد", "را با چند اصلاح جزئی پذیرفت",
             "را به اطلاع همه کارکنان رساند", "را در دستور کار جلسه بعد قرار داد",
             "را برای امضا آماده کرد", "را به دقت بازبینی کرد و نکات مبهم آن را یادداشت کرد",
             "را بایگانی کرد", "را با نسخه پیشین مقایسه کرد"],
    "time": ["در جلسه مورخ {date}", "پس از بحث و تبادل نظر", "در پایان سال مالی", "طی نامه شماره {num}",
             "بنا به پیشنهاد {person}", "با حضور اکثریت اعضا", "در مهلت مقرر", "پیش از برگزاری مجمع",
             "در نخستین فرصت", "پس از دریافت نظر کارشناس", "در غیاب {person}", "روز {weekday} گذشته"],
    "weekday": ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه"],
    "period": ["ماه", "سه ماه", "شش ماه", "سال"],
    "office": ["دبیرخانه مرکزی", "دفتر فنی", "امور مالی", "بایگانی اسناد", "دفتر مدیریت", "واحد حقوقی",
               "انبار مرکزی", "دفتر نمایندگی تهران"],
    "city": ["تبریز", "شیراز", "مشهد", "کرمان", "رشت", "یزد", "همدان", "اراک", "قزوین", "سمنان",
             "اصفهان", "اهواز", "زنجان", "ساری", "بندرعباس"],
    "street": ["بهار", "نسترن", "لاله", "فردوسی", "سعدی", "مولوی", "خیام", "نیما", "پروین", "سهروردی",
               "بوعلی", "شهریار"],
    "reason": ["کمبود قطعات یدکی", "تأخیر در پرداخت", "تعمیرات اساسی", "بارندگی شدید", "قطعی برق",
               "نبود مجوز لازم", "اختلاف در محاسبات", "کمبود نیروی کار"],
    "product": ["پارچه پنبه‌ای", "کاشی و سرامیک", "روغن نباتی", "قطعات خودرو", "کاغذ تحریر",
                "لوله‌های فولادی", "کنسرو ماهی", "رنگ ساختمانی", "شیشه جام", "فرش ماشینی", "صابون",
                "کفش چرمی"],
    "season": ["بهار", "تابستان", "پاییز", "زمستان", "نیمه نخست سال", "سه ماهه سوم"],
    "change": ["افزایش", "کاهش"],
    "machine": ["دستگاه پرس", "کوره ذوب", "دیگ بخار", "نوار نقاله", "دستگاه برش", "ژنراتور برق",
                "پمپ آب", "جرثقیل سقفی"],
    "law": ["ماده {n} آیین‌نامه داخلی", "مصوبه هیئت مدیره", "بند {n} قرارداد اصلی", "مقررات ایمنی کار",
            "دستورالعمل انبارداری"],
    "topic": ["نرخ دستمزد", "ساعت کار زمستانی", "ایمنی کارگاه", "تأمین مواد اولیه", "فروش صادراتی",
              "تعمیر ساختمان", "حمل و نقل کالا", "آموزش کارکنان", "پاداش پایان سال", "خرید زمین"],
    "unit": ["واحد تولید", "انبار", "آزمایشگاه", "بخش اداری", "تعمیرگاه", "واحد فروش"],
    "course": ["ایمنی و آتش‌نشانی", "حسابداری صنعتی", "نگهداری ماشین‌آلات", "کمک‌های اولیه", "انبارداری"],
    "place": ["شهرک صنعتی", "بندر", "دفتر مرکزی", "انبار شماره دو", "اداره ثبت", "بانک ملی شعبه بازار",
              "دادگستری", "گمرک"],
    "vehicle": ["اتوبوس", "قطار", "خودروی شرکت", "کامیون باری", "تاکسی"],
}
POOL_TEMPLATES = [
    "{org} {time} {obj} {verb}.",
    "مقرر شد {obj} حداکثر ظرف {n} روز توسط {org} تهیه و برای امضا به {person} تحویل داده شود.",
    "به موجب این ماده، {org} موظف است هر {period} یک بار {obj} را به {org2} گزارش دهد.",
    "هرگونه تغییر در {obj} تنها با موافقت کتبی {org} و ثبت در دفاتر {office} معتبر است.",
    "{person} ضمن تشکر از {org}، پیشنهاد کرد که {obj} با دقت بیشتری بازنگری شود.",
    "مبلغ {amount} ریال بابت {obj} از محل اعتبارات {office} پرداخت و رسید آن ضمیمه پرونده شد.",
    "نشانی دفتر شرکت در شهر {city}، خیابان {street}، پلاک {n} است و تغییر آن با تصمیم {org} ممکن خواهد بود.",
    "در صورت غیبت {person}، وظایف وی موقتاً به {org} سپرده می‌شود تا تکلیف {obj} روشن شود.",
    "{org} اعلام کرد که کار {machine} به دلیل {reason} تا اطلاع ثانوی متوقف می‌ماند.",
    "صبح روز {weekday} {person} با {vehicle} به {place} رفت و {obj} را با خود برد.",
    "کارکنان {unit} می‌توانند درخواست‌های خود را تا ساعت {hour} به {office} تحویل دهند.",
    "گزارش {org} نشان می‌دهد که تولید {product} در {season} امسال {pct} درصد {change} یافته است.",
    "لطفاً {obj} را پس از امضا در دو نسخه به نشانی {office} ارسال فرمایید.",
    "{person} در پاسخ به پرسش {person2} گفت که {obj} هنوز به دست ما نرسیده است.",
    "تعمیر {machine} کارگاه شماره {n} به علت {reason} {n2} هفته به تأخیر افتاد.",
    "به استناد {law}، {org} حق دارد {unit} را بدون اخطار قبلی بازرسی کند.",
    "یادداشت: {obj} باید پیش از {date} در {office} بایگانی شود.",
    "در این نشست درباره {topic} و {topic2} گفتگو شد و تصمیم نهایی به جلسه آینده موکول گردید.",
    "{person} پیشنهاد داد که {n} نفر از کارکنان {unit} به دوره آموزشی {course} فرستاده شوند.",
    "محموله {product} که از {city} ارسال شده بود روز {weekday} با {n} ساعت تأخیر به {place} رسید.",
    "با توجه به {reason}، {org} ناچار شد سفارش {product} را به نیمه نخست سال بعد موکول کند.",
    "رونوشت این نامه برای آگاهی به {org} و {office} فرستاده می‌شود.",
    "هزینه تعمیر {machine} برابر {amount} ریال برآورد شده و {org} با پرداخت آن موافقت کرده است.",
    "{org2} یادآور شد که رعایت {law} برای همه کارکنان {unit} الزامی است.",
]
POOL_SENTENCES = [
    "باران دیشب تا نزدیک صبح ادامه داشت و آب از زیر در انبار به داخل راه یافت.",
    "نامه شما رسید و از اینکه در این مدت کوتاه پاسخ دادید سپاسگزاریم.",
    "کلید قفسه اسناد نزد خزانه‌دار است و بدون اجازه کتبی به کسی داده نمی‌شود.",
    "ساعت کار دفتر در ماه رمضان از هشت صبح تا دو بعد از ظهر خواهد بود.",
    "چند نفر از کارگران قدیمی هنوز روش دستی بسته‌بندی را به دستگاه تازه ترجیح می‌دهند.",
    "اگر قیمت مواد اولیه همچنان بالا برود، باید در نرخ فروش تجدید نظر کنیم.",
    "سقف سالن شماره سه پیش از آغاز زمستان باید عایق‌کاری شود.",
    "تلفن دفتر چند روز است قطع شده و نامه‌ها با پیک فرستاده می‌شوند.",
    "در بازدید هفته گذشته چند کپسول آتش‌نشانی خالی پیدا شد که بی‌درنگ پر شدند.",
    "از همکاران خواهشمندیم پیش از ترک محل کار چراغ‌ها و بخاری‌ها را خاموش کنند.",
    "نمونه‌های رسیده از کارخانه جدید از نظر رنگ و دوام با سفارش ما همخوانی ندارد.",
    "حساب جاری شرکت در پایان ماه مانده‌ای کمتر از حد انتظار نشان می‌داد.",
    "قرار است دیوار جنوبی محوطه تا پایان تابستان بازسازی شود.",
    "دفترچه‌های راهنمای دستگاه به زبان آلمانی است و هنوز ترجمه نشده است.",
    "کامیون حامل بارها در گردنه به علت برف سنگین دو روز متوقف ماند.",
    "پرداخت حقوق این ماه به سبب تعطیلی بانک‌ها یک روز دیرتر انجام خواهد شد.",
    "نگهبان شب گزارش داد که در ورودی پشتی چند بار باز و بسته شده است.",
    "آزمایش‌های تازه نشان داد که رطوبت انبار برای نگهداری کاغذ مناسب نیست.",
    "به هر یک از کارکنان یک دست لباس کار و یک جفت دستکش تازه داده می‌شود.",
    "برای جلسه آینده فهرست پرسش‌های خود را دست‌کم دو روز زودتر به دبیرخانه بدهید.",
]


def _pool_fill(key: str, rng: random.Random) -> str:
    if key in ("org2",):
        return rng.choice(POOL_FILL["org"])
    if key in ("person2",):
        return rng.choice(POOL_FILL["person"])
    if key in ("topic2",):
        return rng.choice(POOL_FILL["topic"])
    if key in ("n", "n2"):
        return fa(rng.randint(2, 48))
    if key == "num":
        return fa(rng.randint(100, 9999)) + "/" + fa(rng.randint(1, 99))
    if key == "pct":
        return fa(rng.randint(3, 40))
    if key == "hour":
        return fa(rng.randint(8, 16))
    if key == "amount":
        return fa(f"{rng.randint(12, 980) * 1000:,}").replace(",", "٬")
    if key == "date":
        return f"{fa(rng.randint(1335, 1360))}/{fa(rng.randint(1, 12))}/{fa(rng.randint(1, 29))}"
    return POOL_FILL[key][rng.randrange(len(POOL_FILL[key]))]


def pool_sentence(rng: random.Random) -> str:
    """One invented sentence: 9 in 10 from a template, 1 in 10 a fixed sentence."""
    if rng.random() < 0.10:
        return POOL_SENTENCES[rng.randrange(len(POOL_SENTENCES))]
    tpl = POOL_TEMPLATES[rng.randrange(len(POOL_TEMPLATES))]
    out = tpl
    for _ in range(3):   # fillers may contain placeholders ({date}, {person}, {n})
        out = re.sub(r"\{(\w+)\}", lambda m: _pool_fill(m.group(1), rng), out)
    return out


def pool_paragraph(rng: random.Random, lo: int = 1, hi: int = 3) -> str:
    return " ".join(pool_sentence(rng) for _ in range(rng.randint(lo, hi)))


MEMO_TITLES = ["صورتجلسه هیئت مدیره", "آیین‌نامه داخلی کارگاه", "قرارداد همکاری فنی", "گزارش بازرسی سالانه",
               "دستورالعمل انبارداری", "شرح وظایف کارکنان", "تفاهم‌نامه همکاری", "مقررات ایمنی کارگاه"]
MEMO_SUBTITLES = ["شرکت صنایع نمونه‌سازان البرز", "کارخانه ریسندگی آفتاب", "مؤسسه حمل و نقل ستاره شمال",
                  "شرکت تولیدی کاوش", "اتحادیه تعاونی روستایی دشت", "شرکت بازرگانی آینده‌نگر"]
MEMO_SECTIONS = ["بخش اول - تعاریف", "بخش دوم - وظایف و اختیارات", "بخش سوم - امور مالی",
                 "بخش چهارم - مقررات عمومی", "بخش پنجم - تعهدات طرفین", "بخش ششم - موارد پیش‌بینی نشده"]


def file_paragraphs(path: str) -> list[str]:
    """Paragraphs from a UTF-8 text/Markdown file given at run time. The text
    is used to lay out the page only; it is never written to tracked files
    (but it does appear in the generated truth JSON under out/synth/).
    Files of private documents are refused."""
    p = Path(path).expanduser().resolve()
    for slug in PRIVATE_SLUGS:
        if f"/books/{slug}/" in str(p) + "/":
            raise ValueError(f"{p} belongs to a private document and may not be used as a text source")
    paras = []
    for block in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")):
        t = re.sub(r"^[#>*\-\s]+", "", block.strip(), flags=re.M)
        t = re.sub(r"\[\^?\d+\]:?", "", t)
        t = " ".join(t.replace("---", " ").replace("*", "").split())
        if len(t.split()) >= 4:
            paras.append(t)
    if not paras:
        raise ValueError(f"{p}: no paragraphs of >= 4 words")
    return paras


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
    # v2 presets also fix layout / text source / leading (v1 presets do not,
    # so their specs are unchanged).
    for k, v in PRESET_SPEC.get(preset, {}).items():
        if (k == "layout" and layout is not None) or k in over:
            continue
        spec[k] = v
        if k == "layout" and name is None:
            spec["name"] = f"{font_class}_{v}_{preset}_s{seed}"
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
    if src == "pool" or src.startswith("file:"):
        # Larger sources (added after generator v1; embedded output unchanged).
        paras = None if src == "pool" else file_paragraphs(src.split(":", 1)[1])
        fpool = TextPool(paras, rng) if paras else None

        def nxt():
            return fpool.next() if fpool else pool_paragraph(rng)
        if lay in ("articles", "memo"):
            n = rng.randint(1, 40)
            while True:
                yield f"ماده ({fa(n)}) - " + nxt()
                n += 1
        while True:
            yield nxt()
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
    elif lay == "memo":
        # Typewritten office document: centred title (+ subtitle), numbered
        # articles, occasional centred section headings. One type size, as on
        # a typewriter; headings are marked for typed underlining.
        stream = _stream(spec, rng)
        heads = [(MEMO_TITLES[rng.randrange(len(MEMO_TITLES))], True),
                 (MEMO_SUBTITLES[rng.randrange(len(MEMO_SUBTITLES))], False)]
        spool = TextPool(MEMO_SECTIONS, rng)
        first = True
        while y < ybot - size * lead:
            if first or rng.random() < 0.22:
                if not first:
                    heads = [(spool.next(), True)]
                for text, ul in heads:
                    hb = dict(kind="heading", font=cls_font, size=size, leading=1.5, align="center",
                              indent=0.0, underline=ul)
                    toks = text.split()
                    u = _fits(fitz.Rect(x0, y, x1, ybot), hb, toks)
                    if u < 0:
                        break
                    hb.update(rect=[x0, y, x1, y + u + FIT_SLACK], tokens=toks)
                    blocks.append(hb)
                    y += u + gap
                y += gap
            first = False
            # MuPDF mirrors text-align for dir="rtl": "right" gives flush-LEFT
            # lines (what v1's non-justified layouts get), "start" gives the
            # flush-right, ragged-left lines of a Persian typewriter.
            b = body_block(font=cls_font, indent=0.0, al="justify" if spec["justify"] else "start")
            toks = next(stream).split()
            n, u = _fit_prefix(fitz.Rect(x0, y, x1, ybot), b, toks)
            if n == 0:
                break
            b.update(rect=[x0, y, x1, y + u + FIT_SLACK], tokens=toks[:n])
            blocks.append(b)
            y += u + gap * 0.5
            if n < len(toks):
                break
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
                     glyphs=[(c["c"], list(c["bbox"])) for c in ink],   # internal (not in truth)
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
# Pipeline v2: degradations aimed at the hard typewritten-photo profile
# (plan §13 O1/O2, Step 1 results). Every parameter is off by default; a spec
# that enables none of them takes the v1 path, byte-identical to before.
#
# v2 renders the text alone, edits it in pixel space (word spacing, per-glyph
# typewriter texture, filled counters) moving the word-id map with it, then
# composites a separately rendered non-text layer (rules, v1 overlays, typed
# underlines, large stamps and signatures, specks) by multiplication, and
# inserts a per-line baseline warp before the v1 page transform. Every move of
# text pixels is an integer pixel move applied identically to the id map, and
# the layout rectangles are moved by the same amount, so truth stays exact.
# ---------------------------------------------------------------------------
V2_DEFAULTS: dict = {
    # 1. stray ink specks (O1 trigger: just beyond the line START = right end)
    "specks": 0, "specks_start_frac": 0.5, "specks_end_frac": 0.15,
    "specks_size": 1.6, "specks_size_sigma": 0.45, "specks_dist_pt": [1.0, 9.0],
    "specks_ink": [0.05, 0.45],
    # 2. tight word spacing (O2 trigger)
    "tight_frac": 0.0, "tight_to": [0.0, 1.0],
    # 3. large overlays covering several lines
    "stamps_big": 0, "stamp_radius_pt": [55.0, 80.0], "stamp_opacity": 0.9,
    "signatures_big": 0, "signature_lines": [3, 6], "signature_width_pt": [140.0, 260.0],
    "signature_stroke_pt": [1.2, 2.2],
    # 4. uneven baselines (per line, before the v1 page warp)
    "baseline_jitter_px": 0.0, "baseline_slope_deg": 0.0, "baseline_slope_rho": 0.6,
    "baseline_curve_px": 0.0,
    # 5. typewriter texture
    "tw_ink_var": 0.0, "tw_halfstrike": 0.0, "tw_halfstrike_depth": 0.7,
    "tw_jitter_px": 0.0, "tw_key_bias_px": 0.0, "tw_fill_counters": 0.0,
    "typed_underlines": False, "typed_underline_body": 0,
}
DEGRADE_KEYS.update(V2_DEFAULTS)
_V2_SWITCHES = ("specks", "tight_frac", "stamps_big", "signatures_big", "baseline_jitter_px",
                "baseline_slope_deg", "baseline_curve_px", "tw_ink_var", "tw_halfstrike", "tw_jitter_px",
                "tw_key_bias_px", "tw_fill_counters", "typed_underlines", "typed_underline_body")
PIPELINE_V2 = 2

# Hard typewritten-photo profile (plan §13): the v1 "photo" geometry pushed a
# little further (top/mid/bottom slope 0.2/2.4/3.6 deg), heavier type (ink
# spread 2 px), plus every v2 degradation. Aimed at the failure profile
# described in the plan, not fitted to any real page image.
DEGRADE_PRESETS["tw_photo_hard"] = dict(
    warp_top_deg=0.2, warp_mid_deg=2.4, warp_bottom_deg=3.6, warp_bow_px=8.0,
    rotate_deg=0.5, perspective=0.01,
    ink_spread_px=2, ink_dropout=0.15,
    light_gradient=0.4, blur_sigma_px=1.2, noise_sigma=0.035, jpeg_quality=65,
    margin_marks=2,
    specks=45, specks_start_frac=0.5, specks_end_frac=0.15, specks_size=1.8, specks_size_sigma=0.5,
    specks_dist_pt=[1.0, 9.0],
    tight_frac=0.3, tight_to=[0.0, 1.0],
    stamps_big=1, signatures_big=2,
    baseline_jitter_px=3.0, baseline_slope_deg=0.5, baseline_curve_px=3.0,
    tw_ink_var=0.45, tw_halfstrike=0.1, tw_jitter_px=0.7, tw_key_bias_px=0.7, tw_fill_counters=0.4,
    typed_underlines=True, typed_underline_body=2)
# Same family at roughly half strength, on the v1 photo geometry.
DEGRADE_PRESETS["tw_photo_medium"] = dict(
    warp_top_deg=0.1, warp_mid_deg=2.2, warp_bottom_deg=3.3, warp_bow_px=6.0,
    rotate_deg=0.4, perspective=0.008,
    ink_spread_px=2, ink_dropout=0.12,
    light_gradient=0.35, blur_sigma_px=1.1, noise_sigma=0.03, jpeg_quality=70,
    margin_marks=2,
    specks=25, specks_start_frac=0.5, specks_end_frac=0.15, specks_size=1.5, specks_size_sigma=0.45,
    specks_dist_pt=[1.5, 10.0],
    tight_frac=0.12, tight_to=[0.2, 1.2],
    stamps_big=1, signatures_big=1,
    baseline_jitter_px=2.0, baseline_slope_deg=0.3, baseline_curve_px=2.0,
    tw_ink_var=0.3, tw_halfstrike=0.05, tw_jitter_px=0.5, tw_key_bias_px=0.5, tw_fill_counters=0.25,
    typed_underlines=True, typed_underline_body=1)
PRESET_SPEC: dict[str, dict] = {
    "tw_photo_hard": dict(layout="memo", text_source="pool", leading=1.75),
    "tw_photo_medium": dict(layout="memo", text_source="pool", leading=1.75),
}


def v2_enabled(deg: dict) -> bool:
    return any(bool(deg.get(k)) for k in _V2_SWITCHES)


def _dilate_bool(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        o = out.copy()
        o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        out = o
    return out


def _run_lengths(m: np.ndarray) -> np.ndarray:
    """Lengths of True runs along axis 1."""
    p = np.pad(m.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(p, axis=1)
    s = np.nonzero(d == 1)[1]
    e = np.nonzero(d == -1)[1]
    return e - s


def stroke_width_px(idmap: np.ndarray) -> float:
    """Typical stroke thickness of the text ink: min of the median horizontal
    and median vertical ink run lengths."""
    m = idmap > 0
    h = _run_lengths(m)
    v = _run_lengths(m.T)
    if not len(h) or not len(v):
        return 1.0
    return float(min(np.median(h), np.median(v)))


def word_extents_px(idmap: np.ndarray, ntok: int) -> list:
    """Per token ink extent in clean pixels [x0, y0, x1, y1] (x1/y1 exclusive)."""
    return [None if b is None else [int(round(v)) for v in b] for b in ink_bboxes(idmap, ntok, 1, 1)]


def inner_gaps_px(idmap: np.ndarray, ext: list) -> list:
    """Per token: widths of empty-column runs strictly inside its own ink
    x-extent (column projection of that word's ink only), i.e. the gaps a
    gap-based splitter sees inside the word (after non-joining letters)."""
    out = []
    for tid, e in enumerate(ext):
        if e is None:
            out.append([]); continue
        col = (idmap[e[1]:e[3], e[0]:e[2]] == tid + 1).any(0)
        out.append([int(r) for r in _run_lengths(~col[None, :])])
    return out


def _owner_map(idmap: np.ndarray, inkmask: np.ndarray, iters: int = 3) -> np.ndarray:
    """Word id (+1) for every ink pixel: the exact id map, extended into the
    anti-aliased fringe from the nearest id pixel."""
    own = idmap.copy()
    for _ in range(iters):
        need = (own == 0) & inkmask
        if not need.any():
            break
        nb = np.zeros_like(own)
        nb[1:] = np.maximum(nb[1:], own[:-1]); nb[:-1] = np.maximum(nb[:-1], own[1:])
        nb[:, 1:] = np.maximum(nb[:, 1:], own[:, :-1]); nb[:, :-1] = np.maximum(nb[:, :-1], own[:, 1:])
        own = np.where(need, nb, own)
    return own


def plan_tightening(words, lines, ext, ltr, deg, letter_gap, seed):
    """Choose word boundaries to tighten and the horizontal shift of every
    word. Within a line (visual order right->left), closing boundary j by d
    pixels moves every word left of it right by d, so the line start (right
    edge) stays put. Lines with a discontiguous or left-to-right word are not
    touched."""
    rng = random.Random(sub_seed(seed, "tighten"))
    frac = float(deg.get("tight_frac", 0.0))
    lo, hi = deg.get("tight_to", [0.0, 1.0])
    dx = np.zeros(len(words) + 1, np.int64)      # indexed by token id + 1
    recs = []
    for ln in lines:
        ws = sorted((words[i] for i in ln["word_ids"]), key=lambda w: w["visual_index"])
        ok_line = all(not w["missing"] and len(w["segments"]) == 1 and ext[w["id"]] is not None
                      and not ltr[w["id"]] for w in ws)
        cum = 0
        for j in range(len(ws) - 1):
            r, l = ws[j], ws[j + 1]
            er, el = ext[r["id"]], ext[l["id"]]
            gap0 = None if (er is None or el is None) else er[0] - el[2]
            elig = ok_line and r["block"] == l["block"] and gap0 is not None and gap0 > 0
            u, v = rng.random(), rng.random()
            target = None
            if elig and frac > 0 and u < frac:
                target = int(max(0, math.floor((lo + (hi - lo) * v) * letter_gap + 0.5)))
                if target >= gap0:
                    target = None
            d = (gap0 - target) if target is not None else 0
            cum += d
            dx[l["id"] + 1] = cum
            recs.append(dict(line=ln["index"], right_id=r["id"], left_id=l["id"],
                             gap_px_before=gap0, tightened=target is not None,
                             target_px=target, eligible=bool(elig)))
        if ws:
            # words after the last boundary already carry cum; the first has 0
            pass
    return dx, recs


def plan_glyph_texture(words, deg, seed):
    """Per glyph: vertical misalignment (integer px), ink density factor and
    optional half-strike gradient. Key bias is a per-character constant (the
    same letter is always a little high or low), jitter is per strike."""
    rng = np.random.default_rng(sub_seed(seed, "tw_texture"))
    var = float(deg.get("tw_ink_var", 0.0)); hsp = float(deg.get("tw_halfstrike", 0.0))
    depth = float(deg.get("tw_halfstrike_depth", 0.7))
    jit = float(deg.get("tw_jitter_px", 0.0)); kb = float(deg.get("tw_key_bias_px", 0.0))
    bias_cache: dict = {}

    def bias(ch):
        if ch not in bias_cache:
            r = np.random.default_rng(sub_seed(seed, f"key:{ord(ch)}"))
            bias_cache[ch] = float(r.normal(0, kb)) if kb > 0 else 0.0
        return bias_cache[ch]
    out = {}
    for w in words:
        if w["missing"]:
            continue
        gl = []
        for ch, _bb in w["glyphs"]:
            j = float(rng.normal(0, jit)) if jit > 0 else 0.0
            dy = int(np.clip(round(bias(ch) + j), -4, 4))
            dens = 1.0 - var * float(rng.random()) if var > 0 else 1.0
            hs = depth * float(rng.uniform(0.5, 1.0)) if (hsp > 0 and rng.random() < hsp) else 0.0
            hdir = 1 if rng.random() < 0.5 else -1
            gl.append((dy, dens, hs, hdir))
        out[w["id"]] = gl
    return out


def move_text(T: np.ndarray, idmap: np.ndarray, words, dx: np.ndarray, glyph_tex: dict):
    """Apply per-word horizontal shifts and per-glyph vertical shifts/density
    to the text layer T (1 = paper) and the id map. Returns (T', idmap',
    per-word (dy_min, dy_max))."""
    H, W = T.shape
    ink = 1.0 - T
    own = _owner_map(idmap, ink > 0.004)
    ys, xs = np.nonzero(own)
    tid = own[ys, xs] - 1
    dyp = np.zeros(len(ys), np.int64)
    fac = np.ones(len(ys), np.float32)
    dy_rng = {}
    if glyph_tex:
        order = np.argsort(tid, kind="stable")
        cuts = np.flatnonzero(np.diff(tid[order])) + 1
        for grp in np.split(order, cuts):
            if not len(grp):
                continue
            t = int(tid[grp[0]])
            w = words[t]
            gl = glyph_tex.get(t)
            if not gl:
                continue
            gx0 = np.array([bb[0] for _c, bb in w["glyphs"]]) * ZOOM
            gx1 = np.array([bb[2] for _c, bb in w["glyphs"]]) * ZOOM
            cx = xs[grp].astype(np.float64) + 0.5
            mid = (gx0 + gx1) / 2
            inside = (cx[:, None] >= gx0[None, :] - 0.5) & (cx[:, None] < gx1[None, :] + 0.5)
            dist = np.abs(cx[:, None] - mid[None, :])
            dist = dist + np.where(inside.any(1)[:, None], 1e6 * (~inside), 0.0)
            g = np.argmin(dist, axis=1)
            dys = np.array([p[0] for p in gl]); dens = np.array([p[1] for p in gl], np.float32)
            hs = np.array([p[2] for p in gl], np.float32); hd = np.array([p[3] for p in gl])
            wy0, wy1 = w["rect"][1] * ZOOM, w["rect"][3] * ZOOM
            fy = np.clip((ys[grp] + 0.5 - wy0) / max(1.0, wy1 - wy0), 0, 1).astype(np.float32)
            fy = np.where(hd[g] > 0, fy, 1 - fy)
            dyp[grp] = dys[g]
            fac[grp] = dens[g] * (1 - hs[g] * fy)
            dy_rng[t] = (int(dys.min()), int(dys.max()))
    ny = np.clip(ys + dyp, 0, H - 1)
    nx = np.clip(xs + dx[tid + 1], 0, W - 1)
    new_ink = np.where(own == 0, ink, 0).astype(np.float32)
    np.maximum.at(new_ink, (ny, nx), (ink[ys, xs] * fac).astype(np.float32))
    new_id = np.zeros_like(idmap)
    isid = idmap[ys, xs] > 0
    new_id[ny[isid], nx[isid]] = idmap[ys[isid], xs[isid]]
    return (1.0 - new_ink).astype(np.float32), new_id, dy_rng


def shift_word_geometry(words, lines, dx: np.ndarray, dy_rng: dict) -> None:
    """Move layout rectangles by the same pixel shifts (pt = px / ZOOM)."""
    for w in words:
        if w["missing"]:
            continue
        sx = float(dx[w["id"] + 1]) / ZOOM
        lo, hi = dy_rng.get(w["id"], (0, 0))
        ey0, ey1 = min(0, lo) / ZOOM, max(0, hi) / ZOOM

        def mv(r):
            return [r[0] + sx, r[1] + ey0, r[2] + sx, r[3] + ey1]
        w["rect"] = mv(w["rect"])
        w["segments"] = [mv(r) for r in w["segments"]]
        w["main_rect"] = mv(w["main_rect"])
        w["glyphs"] = [(c, [bb[0] + sx, bb[1], bb[2] + sx, bb[3]]) for c, bb in w["glyphs"]]
    for ln in lines:
        ws = [words[i] for i in ln["word_ids"]]
        ln["rect"] = [min(w["rect"][0] for w in ws), min(w["rect"][1] for w in ws),
                      max(w["rect"][2] for w in ws), max(w["rect"][3] for w in ws)]


def fill_counters(T: np.ndarray, idmap: np.ndarray, ext: list, size_px: float, prob: float, seed: int) -> int:
    """Fill enclosed counters (holes in a word's ink) with ink, as clogged
    type does. A hole is background not 4-connected to the border of the
    word's ink box, bounded mostly by that word's own ink. Modifies T and
    idmap in place (filled pixels become ink of that word). Returns count."""
    rng = random.Random(sub_seed(seed, "counters"))
    if prob <= 0:
        return 0
    H, W = T.shape
    max_area = (0.45 * size_px) ** 2
    n = 0
    for tid, e in enumerate(ext):
        if e is None:
            continue
        x0, y0, x1, y1 = max(0, e[0] - 1), max(0, e[1] - 1), min(W, e[2] + 1), min(H, e[3] + 1)
        inkc = T[y0:y1, x0:x1] < 0.55
        bg = ~inkc
        reach = np.zeros_like(bg)
        reach[0, :] = bg[0, :]; reach[-1, :] = bg[-1, :]; reach[:, 0] = bg[:, 0]; reach[:, -1] = bg[:, -1]
        while True:
            nr = reach.copy()
            nr[1:] |= reach[:-1]; nr[:-1] |= reach[1:]; nr[:, 1:] |= reach[:, :-1]; nr[:, :-1] |= reach[:, 1:]
            nr &= bg
            if (nr == reach).all():
                break
            reach = nr
        holes = bg & ~reach
        if not holes.any():
            continue
        # label holes (4-connected), small: plain BFS
        lab = np.zeros(holes.shape, np.int32)
        comps = []
        for sy, sx in zip(*np.nonzero(holes)):
            if lab[sy, sx]:
                continue
            k = len(comps) + 1
            stack = [(sy, sx)]; lab[sy, sx] = k; pix = []
            while stack:
                cy, cx = stack.pop(); pix.append((cy, cx))
                for ny_, nx_ in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny_ < lab.shape[0] and 0 <= nx_ < lab.shape[1] and holes[ny_, nx_] and not lab[ny_, nx_]:
                        lab[ny_, nx_] = k; stack.append((ny_, nx_))
            comps.append(pix)
        idc = idmap[y0:y1, x0:x1]
        for k, pix in enumerate(comps, 1):
            decide = rng.random()
            if len(pix) > max_area:
                continue
            m = lab == k
            ring = _dilate_bool(m, 1) & ~m
            owners = idc[ring]
            if not len(owners) or (owners == tid + 1).mean() < 0.6:
                continue
            if decide >= prob:
                continue
            level = 0.12 + 0.25 * rng.random()
            sub = T[y0:y1, x0:x1]
            sub[m] = np.minimum(sub[m], level)
            idc[m] = tid + 1
            n += 1
    return n


def draw_typed_underlines(page, deg, words, lines, blocks, seed) -> list[dict]:
    """Typed underscores: under memo headings marked for underlining, and
    under a few random body phrases. Not text: recorded as overlays."""
    rng = random.Random(sub_seed(seed, "typed_underlines"))
    targets = []
    if deg.get("typed_underlines"):
        for ln in lines:
            ws = [words[i] for i in ln["word_ids"]]
            if ws and all(w["kind"] == "heading" and blocks[w["block"]].get("underline") for w in ws):
                targets.append((ln, ws, "heading"))
    body = [ln for ln in lines if ln["kind"] == "body" and len(ln["word_ids"]) >= 4]
    for _ in range(int(deg.get("typed_underline_body", 0))):
        if not body:
            break
        ln = rng.choice(body)
        ws = sorted((words[i] for i in ln["word_ids"]), key=lambda w: w["visual_index"])
        k = rng.randint(2, min(4, len(ws)))
        s = rng.randint(0, len(ws) - k)
        targets.append((ln, ws[s:s + k], "body"))
    out = []
    for ln, ws, why in targets:
        x0 = min(w["rect"][0] for w in ws); x1 = max(w["rect"][2] for w in ws)
        size = ws[0]["size"]
        y = ws[0]["line_baseline"] + 0.20 * size
        thick = 0.075 * size
        cell = 0.6 * size
        ext0, ext1 = x0 - 0.15 * cell, x1 + 0.15 * cell
        ncell = max(1, int(math.ceil((ext1 - ext0) / cell)))
        right = ext1
        ymin, ymax = y, y
        for c in range(ncell):
            a, b = right - (c + 1) * cell, right - c * cell
            g = rng.uniform(0.02, 0.07) * size
            if rng.random() < 0.5:
                a += g
            else:
                b -= g
            j = rng.gauss(0, 0.035 * size)
            gray = rng.uniform(0.0, 0.3)
            page.draw_rect(fitz.Rect(a, y + j, b, y + j + thick), color=None, fill=(gray, gray, gray), width=0)
            ymin, ymax = min(ymin, y + j), max(ymax, y + j + thick)
        out.append(dict(kind="typed_underline", text=False, under=why, line=ln["index"],
                        word_ids=[w["id"] for w in ws],
                        rect_pt=[right - ncell * cell, ymin, right, ymax]))
    return out


def _render_layer(draw) -> np.ndarray:
    d = fitz.open()
    p = d.new_page(width=PAGE_W, height=PAGE_H)
    draw(p)
    a = render_gray(p).astype(np.float32) / 255.0
    d.close()
    return a


def _draw_big_stamp(page, c: fitz.Point, r: float, rng: random.Random) -> None:
    col = (0.22, 0.28, 0.62)
    page.draw_circle(c, r, color=col, width=2.4)
    page.draw_circle(c, r * 0.93, color=col, width=1.0)
    page.draw_circle(c, r * 0.60, color=col, width=1.3)
    # illegible ring lettering: short strokes, hooks and dots between the rings
    gap0 = rng.uniform(0, 2 * math.pi)
    n = int(r * 0.9)
    for k in range(n):
        t = gap0 + 2 * math.pi * k / n
        if (k / n) > 0.86:
            continue                      # small gap in the ring text
        rr = r * rng.uniform(0.68, 0.74)
        hgt = r * rng.uniform(0.08, 0.18)
        p0 = c + (math.cos(t) * rr, math.sin(t) * rr)
        p1 = c + (math.cos(t) * (rr + hgt), math.sin(t) * (rr + hgt))
        kind = rng.random()
        if kind < 0.55:
            page.draw_line(p0, p1, color=col, width=rng.uniform(0.7, 1.3))
        elif kind < 0.8:
            t2 = t + rng.uniform(0.03, 0.08)
            p2 = c + (math.cos(t2) * rr, math.sin(t2) * rr)
            page.draw_polyline([p1, p0, p2], color=col, width=0.9)
        else:
            page.draw_circle(p1, 0.6, color=col, fill=col, width=0)
    # emblem and two rows of illegible central lettering
    star = []
    for k in range(10):
        t = -math.pi / 2 + math.pi * k / 5
        rr = r * (0.30 if k % 2 == 0 else 0.13)
        star.append(c + (math.cos(t) * rr, math.sin(t) * rr - r * 0.12))
    page.draw_polyline(star + [star[0]], color=col, width=1.1)
    for row in (0.22, 0.38):
        y = c.y + r * row
        half = r * (0.42 - 0.25 * row)
        x = c.x - half
        while x < c.x + half:
            w = rng.uniform(1.5, 5.0)
            page.draw_line((x, y), (x + w, y - rng.uniform(0, 2.5)), color=col, width=1.0)
            x += w + rng.uniform(1.0, 3.0)


def _draw_big_signature(page, x0: float, y0: float, w: float, h: float, stroke: float, rng: random.Random):
    """Handwriting-like scribble: 2-3 pen strokes, each a prolate cycloid
    (loops where the circular speed exceeds the forward speed) with slowly
    varying loop size, plus a long closing sweep; pen pressure varies along
    the stroke."""
    col = (0.06, 0.06, 0.18)
    xs_all, ys_all = [], []
    nstrokes = rng.randint(2, 3)
    cur = x0
    for s_ in range(nstrokes):
        sw = w * rng.uniform(0.3, 0.55)
        loops = rng.randint(1, 4)
        a_amp = sw / loops * rng.uniform(0.08, 0.3)
        b_amp = h * rng.uniform(0.2, 0.45)
        slant = rng.uniform(-0.2, 0.6)
        yc = y0 + h * rng.uniform(0.35, 0.65)
        ph = rng.uniform(0, 2 * math.pi)
        n = 30 * loops
        mod_a = [rng.uniform(0.4, 1.6) for _ in range(loops + 2)]
        mod_b = [rng.uniform(0.2, 1.2) for _ in range(loops + 2)]
        pts = []
        for i in range(n + 1):
            t = i / n
            k = t * loops
            ia = int(k); fr = k - ia
            ma = mod_a[ia] * (1 - fr) + mod_a[ia + 1] * fr
            mb = mod_b[ia] * (1 - fr) + mod_b[ia + 1] * fr
            ang = 2 * math.pi * k + ph
            x = cur + sw * t + a_amp * ma * math.cos(ang)
            y = yc + b_amp * mb * math.sin(ang) + h * 0.15 * math.sin(math.pi * t)
            pts.append(fitz.Point(x + slant * (yc - y), y))
        width = stroke * rng.uniform(0.8, 1.2)
        for i in range(0, len(pts) - 1, 6):
            seg = pts[i:i + 7]
            page.draw_polyline(seg, color=col, width=max(0.5, width * rng.uniform(0.6, 1.25)))
        xs_all += [p.x for p in pts]; ys_all += [p.y for p in pts]
        cur += sw * rng.uniform(0.55, 0.95)
    # closing sweep under / through the name
    a = fitz.Point(x0 - w * 0.05, y0 + h * rng.uniform(0.7, 1.0))
    b = fitz.Point(x0 + w * rng.uniform(0.85, 1.1), y0 + h * rng.uniform(0.2, 0.6))
    c1 = a + (w * 0.35, h * rng.uniform(0.1, 0.4)); c2 = b - (w * 0.3, -h * rng.uniform(0.0, 0.3))
    page.draw_bezier(a, c1, c2, b, color=col, width=stroke * 0.8)
    xs_all += [a.x, b.x, c1.x, c2.x]; ys_all += [a.y, b.y, c1.y, c2.y]
    m = stroke + 4.0
    return [min(xs_all) - m, min(ys_all) - m, max(xs_all) + m, max(ys_all) + m]


def big_overlays(N: np.ndarray, words, lines, deg, seed, idmap) -> list[dict]:
    """Large stamps and signatures covering several lines, each rendered on
    its own layer (uneven rubber-stamp ink for stamps), composited into N,
    and measured against every word's ink: ``ink_overlap`` = share of the
    word's ink pixels touched by overlay ink (dilated 1 px), ``in_extent`` =
    share inside the overlay's extent (stamp disc / signature box). Measured
    in clean (pre-warp) pixels."""
    rng = random.Random(sub_seed(seed, "big_overlays"))
    H, W = N.shape
    body = [ln for ln in lines if ln["kind"] in ("body", "verse")] or lines
    if not body:
        return []
    bx0 = min(ln["rect"][0] for ln in body); bx1 = max(ln["rect"][2] for ln in body)
    pitch = float(np.median(np.diff([ln["baseline"] for ln in body]))) if len(body) > 1 else 20.0
    totals = np.bincount(idmap.ravel(), minlength=len(words) + 1)
    out = []
    jobs = [("stamp", i) for i in range(int(deg.get("stamps_big", 0)))] + \
           [("signature", i) for i in range(int(deg.get("signatures_big", 0)))]
    for kind, i in jobs:
        if kind == "stamp":
            r = rng.uniform(*deg.get("stamp_radius_pt", [55.0, 80.0]))
            ln = body[min(len(body) - 1, int(len(body) * rng.uniform(0.40, 0.95)))]
            c = fitz.Point(rng.uniform(bx0 + 0.3 * r, bx1 - 0.3 * r), (ln["rect"][1] + ln["rect"][3]) / 2)
            lay = _render_layer(lambda p: _draw_big_stamp(p, c, r, rng))
            tex_rng = np.random.default_rng(sub_seed(seed, f"stamp_tex:{i}"))
            f = _smooth_field(lay.shape, (36, 26), tex_rng)
            tex = np.clip(0.15 + 1.1 * f, 0, 1) ** 0.8
            tex *= (tex_rng.random(lay.shape, dtype=np.float32) > 0.08)
            lay = 1.0 - (1.0 - lay) * tex.astype(np.float32) * float(deg.get("stamp_opacity", 0.9))
            rect = [c.x - r, c.y - r, c.x + r, c.y + r]
            rec = dict(kind="stamp_big", text=False, rect_pt=rect, radius_pt=round(r, 2))
        else:
            nl = rng.randint(*[int(v) for v in deg.get("signature_lines", [3, 6])])
            w = rng.uniform(*deg.get("signature_width_pt", [140.0, 260.0]))
            stroke = rng.uniform(*deg.get("signature_stroke_pt", [1.2, 2.2]))
            k0 = int(len(body) * rng.uniform(0.55, 1.0))
            k0 = min(len(body) - 1, k0)
            ya = body[max(0, k0 - nl + 1)]["rect"][1]
            h = max(pitch * nl, body[k0]["rect"][3] - ya)
            xa = rng.uniform(bx0, max(bx0 + 1, bx0 + (bx1 - bx0) * 0.6 - w * 0.3))
            holder = {}
            lay = _render_layer(lambda p: holder.setdefault("r", _draw_big_signature(p, xa, ya, w, h, stroke, rng)))
            rect = holder["r"]
            rec = dict(kind="signature_big", text=False, rect_pt=rect, lines_spanned=nl)
        N *= lay
        mask = lay < 0.8
        x0 = max(0, int(rect[0] * ZOOM) - 2); x1 = min(W, int(math.ceil(rect[2] * ZOOM)) + 2)
        y0 = max(0, int(rect[1] * ZOOM) - 2); y1 = min(H, int(math.ceil(rect[3] * ZOOM)) + 2)
        md = _dilate_bool(mask[y0:y1, x0:x1], 1)
        idc = idmap[y0:y1, x0:x1]
        hit = np.bincount(idc[md].ravel(), minlength=len(words) + 1)
        if kind == "stamp":
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            disc = (xx + 0.5 - c.x * ZOOM) ** 2 + (yy + 0.5 - c.y * ZOOM) ** 2 <= (r * ZOOM) ** 2
            ext = np.bincount(idc[disc].ravel(), minlength=len(words) + 1)
        else:
            ext = np.bincount(idc.ravel(), minlength=len(words) + 1)
        covers = []
        for t in np.flatnonzero(ext[1:]):
            tot = max(1, int(totals[t + 1]))
            covers.append(dict(word_id=int(t), line=words[t]["line"],
                               ink_overlap=round(float(hit[t + 1]) / tot, 4),
                               in_extent=round(float(ext[t + 1]) / tot, 4)))
        rec["covers"] = covers
        rec["lines_covered"] = sorted({cv["line"] for cv in covers if cv["in_extent"] > 0})
        out.append(rec)
    return out


def draw_specks(N: np.ndarray, idmap: np.ndarray, words, lines, ext, deg, seed, stroke: float) -> list[dict]:
    """Stray ink specks drawn into the non-text layer N (never on text ink).

    Categories: ``line_start`` -- at line height just beyond the START of a
    line (its right end: Persian runs right to left), the O1 trigger;
    ``line_end`` -- just beyond its left end; ``margin`` -- anywhere outside
    the text block. Size: diameter = stroke * lognormal(log(specks_size),
    specks_size_sigma), clipped to 0.6..8 strokes. Shape: union of 1-3
    ellipses, anti-aliased by 3x3 supersampling."""
    rng = np.random.default_rng(sub_seed(seed, "specks"))
    H, W = N.shape
    n = int(deg.get("specks", 0))
    fs, fe = float(deg.get("specks_start_frac", 0.5)), float(deg.get("specks_end_frac", 0.15))
    dlo, dhi = deg.get("specks_dist_pt", [1.0, 9.0])
    ilo, ihi = deg.get("specks_ink", [0.05, 0.45])
    forbid = _dilate_bool(idmap > 0, 3)
    tl = []
    for ln in lines:
        es = [ext[i] for i in ln["word_ids"] if ext[i] is not None]
        if not es:
            continue
        w0 = words[ln["word_ids"][0]]
        tl.append(dict(index=ln["index"], x0=min(e[0] for e in es), x1=max(e[2] for e in es),
                       base=ln["baseline"] * ZOOM, size=w0["size"] * ZOOM))
    if not tl:
        return []
    allx0 = min(t["x0"] for t in tl); allx1 = max(t["x1"] for t in tl)
    ally0 = min(e[1] for e in ext if e is not None); ally1 = max(e[3] for e in ext if e is not None)
    out = []
    for _ in range(n):
        u = rng.random()
        cat = "line_start" if u < fs else ("line_end" if u < fs + fe else "margin")
        d = float(np.clip(stroke * math.exp(math.log(float(deg.get("specks_size", 1.6)))
                                            + float(deg.get("specks_size_sigma", 0.45)) * rng.normal()),
                          0.6 * stroke, 8 * stroke))
        placed = None
        for _try in range(30):
            ln = tl[int(rng.integers(len(tl)))]
            dist = rng.uniform(dlo, dhi) * ZOOM
            if cat == "line_start":
                cx = ln["x1"] + dist + d / 2
                cy = ln["base"] - rng.uniform(0.05, 0.55) * ln["size"]
            elif cat == "line_end":
                cx = ln["x0"] - dist - d / 2
                cy = ln["base"] - rng.uniform(0.05, 0.55) * ln["size"]
            else:
                cx = rng.uniform(0.03 * W, 0.97 * W); cy = rng.uniform(0.03 * H, 0.97 * H)
                if allx0 - 20 < cx < allx1 + 20 and ally0 - 20 < cy < ally1 + 20:
                    continue
            m = int(rng.integers(1, 4))
            ells = []
            for k in range(m):
                ox, oy = (rng.uniform(-d / 3, d / 3), rng.uniform(-d / 3, d / 3)) if k else (0.0, 0.0)
                a = d / 2 * rng.uniform(0.5, 1.0); b = d / 2 * rng.uniform(0.5, 1.0)
                ells.append((cx + ox, cy + oy, max(a, 0.6), max(b, 0.6), rng.uniform(0, math.pi)))
            R = int(math.ceil(d)) + 2
            x0, x1 = int(cx) - R, int(cx) + R + 1
            y0, y1 = int(cy) - R, int(cy) + R + 1
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue
            cov = np.zeros((y1 - y0, x1 - x0), np.float32)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            for sy in (1 / 6, 0.5, 5 / 6):
                for sx in (1 / 6, 0.5, 5 / 6):
                    inside = np.zeros(cov.shape, bool)
                    for ex, ey, a, b, th in ells:
                        X = xx + sx - ex; Y = yy + sy - ey
                        c_, s_ = math.cos(th), math.sin(th)
                        inside |= ((X * c_ + Y * s_) / a) ** 2 + ((-X * s_ + Y * c_) / b) ** 2 <= 1
                    cov += inside / 9.0
            if (forbid[y0:y1, x0:x1] & (cov > 0.05)).any():
                continue
            placed = (cov, x0, y0, x1, y1, ln, cx, cy)
            break
        if placed is None:
            continue
        cov, x0, y0, x1, y1, ln, cx, cy = placed
        g = rng.uniform(ilo, ihi)
        N[y0:y1, x0:x1] *= 1.0 - cov * (1.0 - g)
        sy_, sx_ = np.nonzero(cov > 0.5)
        if not len(sx_):
            sy_, sx_ = np.nonzero(cov > 0)
        bx0, bx1 = x0 + int(sx_.min()), x0 + int(sx_.max()) + 1
        by0, by1 = y0 + int(sy_.min()), y0 + int(sy_.max()) + 1
        rec = dict(kind="speck", text=False, category=cat, o1_trigger=cat == "line_start",
                   diameter_px=round(d, 2), stroke_px=round(stroke, 2), size_strokes=round(d / stroke, 3),
                   area_px=int((cov > 0.5).sum()), ink_level=round(float(g), 3),
                   center_px=[round(cx, 1), round(cy, 1)],
                   rect_pt=[bx0 / ZOOM, by0 / ZOOM, bx1 / ZOOM, by1 / ZOOM])
        if cat != "margin":
            rec.update(line=ln["index"], side="right" if cat == "line_start" else "left",
                       gap_px=int(bx0 - ln["x1"]) if cat == "line_start" else int(ln["x0"] - bx1),
                       at_line_height=bool(ln["base"] - ln["size"] <= cy <= ln["base"] + 0.3 * ln["size"]))
        else:
            rec.update(line=None, side=None)
        out.append(rec)
    return out


class LineWarp:
    """Per-line baseline deformation in clean pixel space (applied before the
    v1 PageTransform): line l gets dy_l(x) = offset_l + tan(slope_l)*(x - cx)
    + curve_l * clip(u, -1.3, 1.3)^2 with u = (x - cx)/half_width. dy is
    constant on a plateau of +-0.15 pitch around each line centre and
    interpolated linearly between plateaus, so it is continuous and, after
    the monotonicity guard (dy/dy >= -0.7, else every deviation is scaled
    down; the scale is recorded), invertible column by column."""

    def __init__(self, width, height, centres, cx, hw, offs, slopes, curves):
        self.W, self.H = float(width), float(height)
        self.cx, self.hw = float(cx), max(1.0, float(hw))
        c = np.asarray(centres, float)
        self.n = len(c)
        pitch = np.diff(c) if self.n > 1 else np.array([60.0])
        core = np.empty(self.n)
        for i in range(self.n):
            nb = [pitch[i - 1]] if i > 0 else []
            nb += [pitch[i]] if i < self.n - 1 else []
            core[i] = 0.15 * min(nb) if nb else 10.0
        self.knots = np.empty(2 * self.n)
        self.knots[0::2] = c - core
        self.knots[1::2] = c + core
        self.o = np.asarray(offs, float); self.t = np.tan(np.radians(np.asarray(slopes, float)))
        self.k = np.asarray(curves, float)
        self.scale = 1.0
        xs = np.linspace(0, self.W, 65)
        G = self._g_all(xs)                    # n x len(xs)
        if self.n > 1:
            ramp = self.knots[2::2] - self.knots[1:-1:2]
            dmin = float(((G[1:] - G[:-1]) / ramp[:, None]).min())
            if dmin < -0.7:
                self.scale = 0.7 / -dmin
        self.o *= self.scale; self.t *= self.scale; self.k *= self.scale
        self.maxabs = float(np.abs(self._g_all(xs)).max()) + 2.0
        self.identity = False

    def _g_all(self, x):
        u = np.clip((np.asarray(x, float) - self.cx) / self.hw, -1.3, 1.3)
        return self.o[:, None] + self.t[:, None] * (np.asarray(x, float) - self.cx)[None, :] + self.k[:, None] * u[None, :] ** 2

    def _g(self, l, x):
        u = np.clip((x - self.cx) / self.hw, -1.3, 1.3)
        return self.o[l] + self.t[l] * (x - self.cx) + self.k[l] * u ** 2

    def _dy_d(self, x, y):
        """dy and d(dy)/dy at (x, y)."""
        j = np.searchsorted(self.knots, y, side="right")   # 0..2n
        la = np.clip((j - 1) // 2, 0, self.n - 1)
        lb = np.clip(j // 2, 0, self.n - 1)
        ramp = (j % 2 == 0) & (j > 0) & (j < 2 * self.n)
        ka = self.knots[np.clip(j - 1, 0, 2 * self.n - 1)]
        kb = self.knots[np.clip(j, 0, 2 * self.n - 1)]
        span = np.maximum(kb - ka, 1e-9)
        tt = np.where(ramp, (y - ka) / span, 0.0)
        ga = self._g(la, x)
        gb = np.where(ramp, self._g(lb, x), ga)
        return ga * (1 - tt) + gb * tt, np.where(ramp, (gb - ga) / span, 0.0)

    def dy(self, x, y):
        return self._dy_d(np.asarray(x, float), np.asarray(y, float))[0]

    def forward(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return x, y + self.dy(x, y)

    def inverse(self, x, y):
        """Solve v + dy(x, v) = y (increasing in v): Newton on the piecewise
        linear function, then bisection for any point not converged."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        v = y - self.dy(x, y)
        for _ in range(6):
            f, d = self._dy_d(x, v)
            v = v - (v + f - y) / (1.0 + d)
        bad = np.abs(v + self.dy(x, v) - y) > 1e-6
        if bad.any():
            xb, yb = x[bad], y[bad]
            lo = yb - self.maxabs; hi = yb + self.maxabs
            for _ in range(40):
                mid = (lo + hi) / 2
                big = mid + self.dy(xb, mid) > yb
                hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
            v = v.copy(); v[bad] = (lo + hi) / 2
        return x, v


class ComposedTransform:
    """first, then second (both with forward/inverse in pixel space)."""

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.identity = bool(getattr(first, "identity", False) and getattr(second, "identity", False))

    def forward(self, x, y):
        return self.second.forward(*self.first.forward(x, y))

    def inverse(self, x, y):
        return self.first.inverse(*self.second.inverse(x, y))


def make_line_warp(lines, words, deg, seed, W, H):
    """Seeded per-line baseline parameters (None if all zero)."""
    js, ss, cs = (float(deg.get("baseline_jitter_px", 0.0)), float(deg.get("baseline_slope_deg", 0.0)),
                  float(deg.get("baseline_curve_px", 0.0)))
    if not (js or ss or cs) or not lines:
        return None, []
    rng = np.random.default_rng(sub_seed(seed, "baselines"))
    rho = float(deg.get("baseline_slope_rho", 0.6))
    cen = [(ln["rect"][1] + ln["rect"][3]) / 2 * ZOOM for ln in lines]
    order = np.argsort(cen, kind="stable")
    offs, slopes, curves = [], [], []
    s_prev = 0.0
    for _ in lines:
        offs.append(float(rng.normal(0, js)) if js else 0.0)
        s = rho * s_prev + math.sqrt(max(0.0, 1 - rho * rho)) * float(rng.normal(0, ss)) if ss else 0.0
        s_prev = s
        slopes.append(s)
        curves.append(float(rng.normal(0, cs)) if cs else 0.0)
    x0 = min(ln["rect"][0] for ln in lines) * ZOOM; x1 = max(ln["rect"][2] for ln in lines) * ZOOM
    c_sorted = [cen[i] for i in order]
    # de-duplicate equal centres (should not happen: lines are distinct rows)
    for i in range(1, len(c_sorted)):
        if c_sorted[i] <= c_sorted[i - 1] + 1:
            c_sorted[i] = c_sorted[i - 1] + 1
    lw = LineWarp(W, H, c_sorted, (x0 + x1) / 2, (x1 - x0) / 2,
                  [offs[i] for i in order], [slopes[i] for i in order], [curves[i] for i in order])
    params = []
    for rank, i in enumerate(order):
        params.append(dict(line=lines[i]["index"], offset_px=round(offs[i] * lw.scale, 3),
                           extra_slope_deg=round(math.degrees(math.atan(math.tan(math.radians(slopes[i])) * lw.scale)), 4),
                           curve_px=round(curves[i] * lw.scale, 3)))
    params.sort(key=lambda p: p["line"])
    return lw, params


def _line_slope_deg(baseline_norm, W, H) -> float:
    p = np.array(baseline_norm, float) * [W, H]
    if len(p) < 2 or np.ptp(p[:, 0]) < 1:
        return 0.0
    k = np.polyfit(p[:, 0], p[:, 1], 1)[0]
    return round(math.degrees(math.atan(k)), 3)


def generate_v2(spec: dict, deg: dict, *, keep_arrays: bool = False) -> dict:
    seed = int(spec["seed"])
    blocks = plan_page(spec)
    black = render_plan(blocks)
    words, notes = extract_words(black[0], blocks)
    lines = group_lines(words, blocks)
    ntok = len(words)
    ltr_all: list[bool] = []
    for b in blocks:
        ltr_all += ltr_flags(b["tokens"])

    idmap0 = render_idmap(black[0])
    T = render_gray(black[0]).astype(np.float32) / 255.0        # text only
    H, W = T.shape
    if idmap0.shape != T.shape:
        raise RuntimeError("identity and image renders differ in size")
    stroke = stroke_width_px(idmap0)
    ext0 = word_extents_px(idmap0, ntok)
    inner0 = inner_gaps_px(idmap0, ext0)
    allg = [g for gs in inner0 for g in gs]
    letter_gap = float(np.median(allg)) if allg else stroke

    # 2 + 5: word spacing and per-glyph texture, one pixel move
    dx, gap_recs = plan_tightening(words, lines, ext0, ltr_all, deg, letter_gap, seed)
    tex = None
    if any(deg.get(k) for k in ("tw_ink_var", "tw_halfstrike", "tw_jitter_px", "tw_key_bias_px")):
        tex = plan_glyph_texture(words, deg, seed)
    if dx.any() or tex:
        T, idmap1, dy_rng = move_text(T, idmap0, words, dx, tex or {})
        shift_word_geometry(words, lines, dx, dy_rng)
    else:
        idmap1 = idmap0.copy()
    ext1 = word_extents_px(idmap1, ntok)
    n_filled = fill_counters(T, idmap1, ext1, words[0]["size"] * ZOOM if words else 30.0,
                             float(deg.get("tw_fill_counters", 0.0)), seed)
    inner1 = inner_gaps_px(idmap1, ext1)

    # non-text layer
    def draw_nontext(p):
        draw_rules(p, blocks)
        holder["v1"] = draw_overlays(p, deg, words, lines, seed)
        holder["tu"] = draw_typed_underlines(p, deg, words, lines, blocks, seed)
    holder: dict = {}
    N = _render_layer(draw_nontext)
    overlays = holder["v1"] + holder["tu"]
    bigs = big_overlays(N, words, lines, deg, seed, idmap1)
    specks = draw_specks(N, idmap1, words, lines, ext1, deg, seed, stroke)
    overlays += bigs + specks

    clean_u8 = np.clip(np.rint(T * N * 255.0), 0, 255).astype(np.uint8)
    img = clean_u8.astype(np.float32) / 255.0
    img = print_side(img, deg, seed)
    lw, lw_params = make_line_warp(lines, words, deg, seed, W, H)
    ptf = PageTransform(W, H, deg, seed)
    tf = ComposedTransform(lw, ptf) if lw is not None else ptf
    img = warp_image(img, tf, fill=1.0)
    idmap = warp_image(idmap1, tf, fill=0, nearest=True)
    img = camera_side(img, deg, seed)
    u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    if int(deg.get("jpeg_quality", 0)) > 0:
        u8 = jpeg_roundtrip(u8, int(deg["jpeg_quality"]))

    truth = build_truth(spec, blocks, words, lines, overlays, tf, idmap, u8, notes)
    # ---- v2 truth additions ----
    truth["pipeline"] = PIPELINE_V2
    for o in truth["overlays"]:
        o.setdefault("text", False)
    inks_img = [None if b is None else [b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in ink_bboxes(idmap, ntok, W, H)]
    word_gaps = []
    for g in gap_recs:
        r, l = g["right_id"], g["left_id"]
        er, el = ext1[r], ext1[l]
        gap = None if (er is None or el is None) else er[0] - el[2]
        mi = max(inner1[r] + inner1[l] + [0])
        ir, il = inks_img[r], inks_img[l]
        word_gaps.append(dict(g, gap_px=gap, max_inner_gap_px=mi,
                              o2_trigger=bool(gap is not None and gap <= mi),
                              gap_px_image=None if (ir is None or il is None) else round(ir[0] - il[2], 2)))
    truth["word_gaps"] = word_gaps
    covered: dict = {}
    for oi, o in enumerate(truth["overlays"]):
        for cv in o.get("covers", []):
            covered.setdefault(cv["word_id"], []).append(dict(overlay=oi, kind=o["kind"], ink_overlap=cv["ink_overlap"],
                                                              in_extent=cv["in_extent"]))
    for tw in truth["words"]:
        tw["covered_by"] = covered.get(tw["id"], [])
        tw["inner_gaps_px"] = inner1[tw["id"]]
    for tl in truth["lines"]:
        tl["slope_deg"] = _line_slope_deg(tl["baseline"], W, H)
    truth["line_warp"] = lw_params
    truth["line_warp_scale"] = round(lw.scale, 4) if lw is not None else None
    sl = [tl["slope_deg"] for tl in truth["lines"]]
    k3 = max(1, len(sl) // 3)
    truth["notes"].update(
        stroke_px=round(stroke, 2), letter_gap_px=round(letter_gap, 2),
        n_specks=len(specks), n_specks_o1=sum(1 for s in specks if s["o1_trigger"]),
        n_boundaries=len(word_gaps), n_tightened=sum(1 for g in word_gaps if g["tightened"]),
        n_o2_trigger=sum(1 for g in word_gaps if g["o2_trigger"]),
        n_counters_filled=n_filled,
        n_typed_underlines=sum(1 for o in overlays if o["kind"] == "typed_underline"),
        n_big_overlays=len(bigs),
        n_words_covered=sum(1 for v in covered.values() if any(c["in_extent"] > 0 for c in v)),
        slope_deg_top_mid_bottom=[round(float(np.median(sl[:k3])), 3) if sl else None,
                                  round(float(np.median(sl[len(sl) // 2 - k3 // 2:len(sl) // 2 + k3 // 2 + 1])), 3) if sl else None,
                                  round(float(np.median(sl[-k3:])), 3) if sl else None],
        slope_deg_range=[min(sl), max(sl)] if sl else None,
    )
    seal_truth(truth)
    res = dict(image=u8, truth=truth, blocks=blocks)
    if keep_arrays:
        res.update(clean=clean_u8, idmap=idmap, idmap_clean=idmap1, idmap_layout=idmap0, transform=tf,
                   doc=black, text_layer=T, nontext_layer=N)
    else:
        black.close()
    return res


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
    if v2_enabled(deg):
        return generate_v2(spec, deg, keep_arrays=keep_arrays)
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

    truth = build_truth(spec, blocks, words, lines, overlays, tf, idmap, u8, notes)
    seal_truth(truth)
    res = dict(image=u8, truth=truth, blocks=blocks)
    if keep_arrays:
        res.update(clean=clean_u8, idmap=idmap, idmap_clean=idmap_clean, transform=tf,
                   doc=black)
    else:
        black.close()
    return res


def seal_truth(truth: dict) -> None:
    truth["truth_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in truth.items() if k != "truth_sha256"}).encode()).hexdigest()


def build_truth(spec, blocks, words, lines, overlays, tf, idmap, u8, notes) -> dict:
    """Truth dict shared by both pipelines (unchanged from generator v1)."""
    H, W = u8.shape
    ntok = len(words)
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
    return dict(
        generator_version=GENERATOR_VERSION, spec=spec, pymupdf=fitz.VersionBind,
        fonts=fonts, page_pt=[PAGE_W, PAGE_H], zoom=ZOOM,
        coords="normalised to the degraded image: x 0..1 left->right, y 0..1 top->bottom",
        image=dict(width=W, height=H, sha256=sha256_array(u8)),
        lines=tlines, words=twords, overlays=tover,
        notes=dict(notes, n_words=ntok, n_missing=sum(1 for w in words if w["missing"]),
                   n_discontiguous=sum(1 for w in words if not w["missing"] and len(w["segments"]) > 1),
                   kashida="MuPDF justifies by widening spaces only; no tatweel is inserted"),
    )


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
    # pipeline v2 presets (added after v1; the entries above are unchanged)
    if resolve_class_font("typewriter") is not None:
        for preset in ("tw_photo_medium", "tw_photo_hard"):
            specs.append(make_spec("typewriter", preset=preset, seed=11))
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
    g.add_argument("--layout", choices=LAYOUTS + ("memo",))
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--text-source", default=None,
                   help="'embedded', 'pool' (large invented pool), 'book:<slug>' or 'file:<path>' (default: the preset's, else embedded)")
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
        kw = {} if a.text_source is None else dict(text_source=a.text_source)
        spec = make_spec(a.font_class, preset=a.preset, seed=a.seed, layout=a.layout,
                         degrade=json.loads(a.degrade), **kw)
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
