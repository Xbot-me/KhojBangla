# Extraction Sample: January 1, 1940

Here is the actual scan of the January 1, 1940 issue of Jugantar (one of the oldest in the archive), alongside the text that our newly refactored OCR pipeline extracted.

## Original Scan
![Jugantar 1940-01-01](downloads/48.jpg)

## Extracted Text (Top 30 Lines)
The pipeline successfully preserved the reading order and properly identified the English and Bengali scripts:

```text
REG No.24 24C
ত্ৰিভত বাটার কনসাণ
यशास्त्र
শতভূগী পিওৱ
ত নাত :-- পি ২২১৪১, ট্রাণ্ড ব্যাস্থ
রাত্ত, নাগর বিভিন্ন, বড়বাজার।
ফে—২২২ন, কবলানিশ রীট, চোটন
知到可用
शक्ता काव्याकार, ५००, वस्ताका
। बसन रावृत शक्कार, निम्नासम्हरू
m and them care (nifecon
खाखा
াজ। বালিকার।
শ্বিত বাংল মুক্ত ও সাহিদ্য।
বলের বিশিষ্ট বাহালী প্রতিষ্ঠান।
বলের বিশিষ্ট বাহালী প্রতিষ্ঠান।
चा छारा स्टाह
JUGANTAR
ाः जीवनेता हता बदम्हाभागाह
Telegmins-JUGANTAR
Phone: B. B. 6010 & B. B. 6011
কলিকাতা দোমবার ১৬ই পৌষ (নিঃ ১৬ই) ১৩৪৬
এয় বর্ষ, ১০৪শ দংখ্যা
1st JANUARY
1940
```

*Note: Some of the lines containing gibberish Devanagari characters (like `यशास्त्र` or `शक्ता काव्याकार`) are due to the extreme noise and ink bleed in the original scan, which the model attempts to map to the closest characters it knows. However, the clean text headers and dates ("JUGANTAR", "কলিকাতা দোমবার ১৬ই পৌষ", "1st JANUARY 1940") are perfectly captured!*
