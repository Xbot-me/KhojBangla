"""
Generates a synthetic 2-column Bangla test page so we can validate the
preprocessing + segmentation pipeline without needing a real scanned newspaper.
Adds slight rotation + noise to simulate an old scan.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import random

FONT_PATH = "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf"

SAMPLE_TEXT = [
    "বাংলাদেশের স্বাধীনতা যুদ্ধ ১৯৭১ সালে সংঘটিত হয়েছিল।",
    "এই যুদ্ধে লক্ষ লক্ষ মানুষ প্রাণ হারিয়েছিলেন।",
    "পুরাতন সংবাদপত্রের সংরক্ষণ ইতিহাস চর্চার জন্য গুরুত্বপূর্ণ।",
    "ডিজিটাল আর্কাইভ গবেষকদের জন্য অত্যন্ত সহায়ক হতে পারে।",
    "সঠিক তথ্য সংরক্ষণ ও যাচাই অত্যন্ত জরুরি বিষয়।",
    "প্রযুক্তির মাধ্যমে পুরনো নথি ডিজিটাইজ করা সম্ভব।",
    "চট্টগ্রাম বন্দর থেকে বাণিজ্যিক জাহাজ চলাচল অব্যাহত আছে।",
    "শিক্ষা ব্যবস্থার উন্নতির জন্য নতুন পরিকল্পনা নেওয়া হয়েছে।",
]


def make_column(width: int, height: int, font: ImageFont.FreeTypeFont, lines: int) -> Image.Image:
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    y = 10
    for i in range(lines):
        text = random.choice(SAMPLE_TEXT)
        draw.text((10, y), text, font=font, fill=0)
        y += 40
    return img


def build_test_page(path: str):
    font = ImageFont.truetype(FONT_PATH, 24)
    page_w, page_h = 1000, 1400
    page = Image.new("L", (page_w, page_h), color=255)

    col_w = 460
    gutter = 40
    col1 = make_column(col_w, page_h - 40, font, lines=28)
    col2 = make_column(col_w, page_h - 40, font, lines=28)

    page.paste(col1, (20, 20))
    page.paste(col2, (20 + col_w + gutter, 20))

    arr = np.array(page)

    # simulate slight scan rotation
    (h, w) = arr.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 1.3, 1.0)
    arr = cv2.warpAffine(arr, matrix, (w, h), borderValue=255)

    # simulate scan noise / aging
    noise = np.random.normal(0, 8, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, arr)
    print(f"Wrote synthetic test page to {path}")


if __name__ == "__main__":
    build_test_page("/home/claude/bangla-ocr/samples/synthetic_page.png")
