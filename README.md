# Bangla Historical Newspaper OCR — Free/Open-Source Pipeline

Working, tested pipeline. No paid APIs anywhere. Built and validated in this
session on a synthetic 2-column Bangla test page (56 lines, correct reading
order, confidence-gated results, 11 lines correctly auto-flagged for review).

## What's actually done and tested here

- `preprocess.py` — deskew, denoise, CLAHE contrast enhancement. **Tested**: fixed
  a 1.3° simulated scan rotation, cleaned simulated scan noise.
- `segment.py` — column detection, then line detection within each column, via
  projection profiles. **No model download required, works anywhere.** **Tested**:
  correctly split a 2-column page into 56 ordered line crops (28/column).
- `ocr_engines.py` — `TesseractEngine` (tested, works now) and `SuryaEngine`
  (written and ready, needs a one-time model download you'll run yourself).
- `pipeline.py` — ties it together: preprocess → segment → OCR every engine
  available → confidence-gate → store to SQLite. **Tested end-to-end.**
- `storage.py` — SQLite schema: pages, line_crops, ocr_results. Keeps every
  engine's raw output, never overwrites, so you can always audit later.
- `review_queue.py` — CLI to list every line flagged for human review.

## Why Tesseract alone isn't the end state

Tesseract's Bengali model is genuinely weaker on some conjuncts/matras — you'll
see it in the test run's output (e.g. "পুরনো" → "প্রনো", a dropped matra). That's
the model, not the pipeline. The pipeline itself (segmentation + confidence
gating) is doing its job: reading order is correct, and the genuinely uncertain
lines got flagged instead of silently guessed.

## Next step: add Surya OCR for real accuracy gains

Run this on your own dev machine (needs normal internet access, unlike this
sandbox):

```bash
pip install -r requirements.txt
python3 -c "from surya.recognition import RecognitionPredictor; from surya.detection import DetectionPredictor; RecognitionPredictor(); DetectionPredictor()"
```

That last line forces the one-time HuggingFace model download so it's cached
locally. After that, `get_available_engines()` in `ocr_engines.py` will pick up
Surya automatically and the pipeline will prefer it over Tesseract, using
Tesseract only as the disagreement cross-check.

## Getting Started (Docker Recommended) 🐳

The easiest way to run the entire pipeline (UI + Database + OCR engines) is using Docker, which prevents dependency issues with Tesseract, Poppler, and OpenCV.

```bash
docker-compose up --build
```

This will automatically:
1. Build the Python 3.11 environment with all system dependencies.
2. Spin up the Streamlit UI on `http://localhost:8501`.
3. Mount the databases and HuggingFace cache locally so your work is saved.

## Running Manually (Without Docker)

```bash
pip install -r requirements.txt
# Bengali tesseract language pack + fonts (Debian/Ubuntu):
sudo apt-get install tesseract-ocr-ben fonts-beng poppler-utils

# Start the Streamlit Web App
streamlit run banglaocr/app.py
```

## Streamlit UI & Pipeline Trigger

The Streamlit UI provides 3 main tabs:
1. **Review Queue**: Verify low-confidence OCR results side-by-side with cropped image segments.
2. **Search Archive**: Perform semantic vector search queries in Bengali across your verified historical database.
3. **Process New Issue**: A Date Selector to trigger the background OCR pipeline on a specific newspaper issue.

## CI/CD ⚙️

This repository includes a GitHub Actions workflow `.github/workflows/ci.yml`. On every push to the `main` branch, it automatically:
- Installs Tesseract and system dependencies on an Ubuntu runner.
- Runs the test suite to catch breakages.
- Validates the Docker image build.

## Tuning knobs worth knowing about

- `segment.py`: `min_column_gap_px` / `min_line_gap_px` — if a real scan has
  tighter column gutters or denser line spacing than the synthetic test page,
  these need adjusting. Start by segmenting one real page and eyeballing the
  crops in `output/crops/` before running the full pipeline on a batch.
- `pipeline.py` `PipelineConfig.min_confidence` — currently 70. Lower = fewer
  lines flagged for review but more risk of accepting bad OCR; higher = more
  manual review work but safer.
- `PipelineConfig.do_upscale` — turn on for low-DPI scans; cubic upscale is a
  free stand-in for a real super-resolution model (Real-ESRGAN/SwinIR), which
  would help more on genuinely blurry scans but needs GPU + model weights.

## What still needs a real scanned page to validate

Everything above was validated on a *clean synthetic* Bangla page. Real 50-60
year old newspaper scans will have: uneven column gutters, ads/photos breaking
up columns, bleed-through from the reverse page, and much lower contrast.
Recommended next test: run `segment.py` on one real page from
`songramernotebook.com`, check the crops visually, then tune the gap thresholds
before running the full pipeline on a batch.

## The crawler (`banglaocr/crawler/`)

Discovered while researching this: **neither site hosts newspaper images
directly** - both link out to individual Google Drive share links (one per
day). So this is really two jobs, kept as two separate modules:

1. **Metadata crawler** (`liberationwar_crawler.py`, `songramer_crawler.py`) -
   walks the WordPress pages, collects paper name / date / Google Drive URL
   for every issue found, stores it in a `manifest.db` (SQLite). Checks
   `robots.txt` at runtime before fetching anything, and **fails closed**
   (refuses to fetch) if robots.txt can't be reached - tested and confirmed
   in this sandbox, which has no route to either domain. Rate-limited to at
   least 2 seconds between requests regardless of what robots.txt allows.

2. **Drive downloader** (`drive_downloader.py`) - a deliberately separate,
   much slower module (5s default delay) that pulls files for manifest
   entries once their Drive link is known. This is kept apart from the
   crawler on purpose: Google Drive enforces its own per-file download quotas
   independent of the source site's robots.txt, and bulk/automated access to
   Drive share links isn't something a polite crawler exemption covers. This
   module minimizes risk (slow, sequential, stops and warns on repeated
   failures) but **cannot eliminate it** - if you ever need this at real
   volume (thousands of files, regularly), the right move is asking the
   archive maintainers for a bulk export or using the Drive API with your
   own OAuth credentials, not scraping public share links harder.

3. **Ingest** (`ingest.py`) - feeds anything downloaded-but-not-yet-OCR'd
   into the `pipeline.py` built earlier. Skips anything already processed.

### Before running this for real

- **Open both `robots.txt` files yourself first** (`songramernotebook.com/robots.txt`,
  `liberationwarbangladesh.org/robots.txt`) - this sandbox has no network route
  to either domain, so nothing here could be tested end-to-end against the
  real sites. The regex link-extraction logic *was* tested against real URL
  samples pulled from both sites (see commit history / this session), but the
  live crawl itself, robots.txt handling, and Drive downloads only run once
  you try them on a machine with normal internet access.
- Start with tiny limits (`--max-pages 1`, `--limit 5`) and watch what happens
  before scaling up.
- If Drive downloads start failing repeatedly, stop and wait rather than
  retrying in a loop - that's Google throttling, and hammering it harder
  doesn't help and risks the shared files getting flagged for everyone.

### Running it

```bash
cd banglaocr
python3 crawler/run.py songramer --manifest-db manifest.db --max-pages 2
python3 crawler/run.py liberationwar --manifest-db manifest.db --max-pages 1 --max-posts 5
python3 crawler/run.py status --manifest-db manifest.db

python3 crawler/run.py download --manifest-db manifest.db --out downloads --limit 10 --delay 5

python3 crawler/run.py ingest --manifest-db manifest.db --ocr-db ocr.db --limit 10
python3 review_queue.py ocr.db
```
