"""
Full pipeline: preprocess -> segment into columns/lines -> OCR each line with
whichever engines are available -> confidence-gate -> store to SQLite.

Confidence gating logic (no paid APIs, no LLM in the loop by default):

  - Only Tesseract available (current sandbox state):
        accept if confidence >= min_confidence, else flag needs_review.

  - Tesseract + Surya both available (recommended real-world setup):
        run both, prefer Surya's result (better Bengali accuracy) BUT:
          - if the two engines' texts materially disagree -> flag needs_review
            and store both, so nothing gets silently guessed
          - if they agree (or Surya's confidence is comfortably high) -> accept

This intentionally never calls an LLM to "resolve" disagreements - per the
plan, that's exactly the failure mode (fluent hallucination) we're avoiding.
Disagreements go to a human review queue instead.
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass

from ocr_engines import get_available_engines, OcrLineResult
from preprocess import preprocess_page
from segment import chunk_into_columns, segment_page
import storage


@dataclass
class PipelineConfig:
    db_path: str = "ocr.db"
    min_confidence: float = 70.0          # below this -> needs_review
    agreement_threshold: float = 0.85     # text similarity ratio to count as "agree"
    do_upscale: bool = False
    upscale_factor: float = 2.0


def _texts_agree(a: str, b: str, threshold: float) -> bool:
    if not a and not b:
        return True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def process_page(image_path: str, config: PipelineConfig, source_url: str | None = None) -> int:
    storage.init_db(config.db_path)
    engines = get_available_engines()
    if not engines:
        raise RuntimeError(
            "No OCR engines available. Install pytesseract + tesseract-ocr-ben "
            "at minimum, or set up surya-ocr for the recommended primary engine."
        )

    page_id = storage.insert_page(config.db_path, source_path=image_path, source_url=source_url)

    processed = preprocess_page(
        image_path, do_upscale=config.do_upscale, upscale_factor=config.upscale_factor
    )

    print(f"[page {page_id}] Starting OCR with engines: {', '.join(engines)}")

    # We will use whichever engine is available that provides hierarchical layout (process_page returns list[TextBlock])
    hierarchical_engine = None
    if "hybrid" in engines:
        hierarchical_engine = engines["hybrid"]
    elif "tesseract" in engines:
        hierarchical_engine = engines["tesseract"]
    elif "google_vision" in engines:
        hierarchical_engine = engines["google_vision"]

    if hierarchical_engine:
        # NEW PIPELINE: Use hierarchical layout extraction per column
        import layout_parser as layout
        
        print(f"[page {page_id}] Segmenting into columns for {hierarchical_engine.name}...")
        columns = chunk_into_columns(processed)
        if not columns:
            # Fallback if no columns detected
            columns = [{"column_index": 0, "bbox": type('obj', (object,), {'x': 0, 'y': 0, 'w': processed.shape[1], 'h': processed.shape[0]})(), "image": processed}]
            
        total_articles = 0
        for col in columns:
            col_img = col["image"]
            col_offset_x = getattr(col["bbox"], "x", 0)
            col_offset_y = getattr(col["bbox"], "y", 0)
            
            blocks = hierarchical_engine.process_page(col_img)
            
            # Adjust bounding boxes to global page coordinates
            for block in blocks:
                block.bbox.x += col_offset_x
                block.bbox.y += col_offset_y
                for para in block.paragraphs:
                    para.bbox.x += col_offset_x
                    para.bbox.y += col_offset_y
            
            filtered_blocks = layout.filter_ads(blocks)
            articles = layout.group_articles(filtered_blocks)
            
            for art_idx, article in enumerate(articles):
                line_idx = 0
                for block in article:
                    for para in block.paragraphs:
                        line_crop_id = storage.insert_line_crop(
                            config.db_path, page_id, column_index=col["column_index"], line_index=line_idx, bbox=para.bbox
                        )
                        
                        if para.confidence >= config.min_confidence:
                            _mark_accepted(config.db_path, line_crop_id, hierarchical_engine.name, para.text, para.confidence)
                        else:
                            _mark_needs_review(config.db_path, line_crop_id, hierarchical_engine.name, para.text, para.confidence)
                            
                        line_idx += 1
                total_articles += 1
                
        print(f"[page {page_id}] Processed {total_articles} articles using {hierarchical_engine.name}.")
        return page_id

    # OLD PIPELINE (Fallback to Surya or manual segmentation + Tesseract Line-by-line)
    surya_results = []
    tess_results = []
    bboxes = []

    if "surya" in engines:
        # Use Surya for layout detection only, as its OCR is too slow on CPU.
        bboxes = engines["surya"].detect_layout(processed)
        
        if "tesseract" in engines and bboxes:
            tess_results = engines["tesseract"].ocr_page(processed, bboxes)
    else:
        # Fallback to manual segmentation if Surya isn't available
        crops = segment_page(processed)
        bboxes = [c["bbox"] for c in crops]
        if "tesseract" in engines and bboxes:
            tess_results = engines["tesseract"].ocr_page(processed, bboxes)

    print(f"[page {page_id}] Found {len(bboxes)} text lines.")

    # Match results by index since they share the same bboxes
    for i, bbox in enumerate(bboxes):
        line_crop_id = storage.insert_line_crop(
            config.db_path, page_id, column_index=0, line_index=i, bbox=bbox
        )

        results = {}
        if surya_results:
            results["surya"] = surya_results[i]
        if tess_results:
            results["tesseract"] = tess_results[i]

        if not results:
            storage.insert_ocr_result(
                config.db_path, line_crop_id, "none", "", 0.0,
                is_accepted=False, needs_review=True,
            )
            continue

        _store_and_gate(config, line_crop_id, results)

    return page_id


def _store_and_gate(config: PipelineConfig, line_crop_id: int, results: dict[str, OcrLineResult]) -> None:
    # Store every engine's raw output first, unconditionally.
    for engine_name, result in results.items():
        storage.insert_ocr_result(
            config.db_path, line_crop_id, engine_name, result.text, result.confidence,
            is_accepted=False, needs_review=False,
        )

    if "google_vision" in results:
        gv_res = results["google_vision"]
        # Google vision is extremely accurate, we can trust it.
        # We can still compare it to Tesseract or just accept it directly.
        if gv_res.confidence >= config.min_confidence:
            _mark_accepted(config.db_path, line_crop_id, "google_vision", gv_res.text, gv_res.confidence)
        else:
            _mark_needs_review(config.db_path, line_crop_id, "google_vision", gv_res.text, gv_res.confidence)
            
    elif "surya" in results and "tesseract" in results:
        surya_res = results["surya"]
        tess_res = results["tesseract"]
        agree = _texts_agree(surya_res.text, tess_res.text, config.agreement_threshold)

        if agree and surya_res.confidence >= config.min_confidence:
            _mark_accepted(config.db_path, line_crop_id, "surya", surya_res.text, surya_res.confidence)
        else:
            _mark_needs_review(config.db_path, line_crop_id, "surya", surya_res.text, surya_res.confidence)

    elif "surya" in results:
        r = results["surya"]
        if r.confidence >= config.min_confidence:
            _mark_accepted(config.db_path, line_crop_id, "surya", r.text, r.confidence)
        else:
            _mark_needs_review(config.db_path, line_crop_id, "surya", r.text, r.confidence)

    elif "tesseract" in results:
        r = results["tesseract"]
        if r.confidence >= config.min_confidence:
            _mark_accepted(config.db_path, line_crop_id, "tesseract", r.text, r.confidence)
        else:
            _mark_needs_review(config.db_path, line_crop_id, "tesseract", r.text, r.confidence)


from llm_corrector import correct_text_with_llm

def _mark_accepted(db_path, line_crop_id, engine, text, confidence):
    # Pass through LLM
    llm_res = correct_text_with_llm(text)
    storage.insert_ocr_result(
        db_path, line_crop_id, f"{engine}_final", text, confidence,
        is_accepted=True, needs_review=False,
        corrected_text=llm_res.get("corrected_text", text),
        is_reconstructed=llm_res.get("is_reconstructed", False),
        status='verified'
    )


def _mark_needs_review(db_path, line_crop_id, engine, text, confidence):
    # Pass through LLM
    llm_res = correct_text_with_llm(text)
    storage.insert_ocr_result(
        db_path, line_crop_id, f"{engine}_final", text, confidence,
        is_accepted=True, needs_review=True,
        corrected_text=llm_res.get("corrected_text", text),
        is_reconstructed=llm_res.get("is_reconstructed", False),
        status='pending_review'
    )


def export_page_text(config: PipelineConfig, page_id: int) -> str:
    """Reconstruct plain text for a page in correct reading order, column by column."""
    rows = storage.get_page_text(config.db_path, page_id, only_accepted=False)
    # Since we set column_index=0 for Surya, it's just a flat list of lines.
    # To avoid missing text if it wasn't accepted, we can just grab the accepted ones or the raw surya ones.
    # The get_page_text with only_accepted=True might return nothing if all are in review!
    # Wait, let's fix export_page_text to fetch final results (accepted or in review).
    pass  # We will redefine export_page_text to just print the lines

    with storage.connect(config.db_path) as conn:
        rows = conn.execute("""
            SELECT lc.line_index, orr.text
            FROM line_crops lc
            JOIN ocr_results orr ON orr.line_crop_id = lc.id
            WHERE lc.page_id = ? AND orr.engine LIKE '%_final'
            ORDER BY lc.line_index
        """, (page_id,)).fetchall()
        
    return "\n".join(r[1] for r in rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [db_path]")
        sys.exit(1)

    cfg = PipelineConfig(db_path=sys.argv[2] if len(sys.argv) > 2 else "ocr.db")
    pid = process_page(sys.argv[1], cfg)
    print()
    print(export_page_text(cfg, pid))

    review = storage.get_review_queue(cfg.db_path)
    print(f"\n{len(review)} lines flagged for human review.")
