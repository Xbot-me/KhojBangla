"""
OCR engine wrappers. Each engine exposes the same interface:

    ocr_page(image: np.ndarray, bboxes: list[BoundingBox] = None) -> list[OcrLineResult]

so the pipeline can run one or more engines. Surya can generate bboxes if None is passed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from segment import BoundingBox

@dataclass
class OcrLineResult:
    text: str
    confidence: float  # 0-100
    engine: str
    bbox: BoundingBox


class TesseractHierarchicalEngine:
    name = "tesseract"

    def __init__(self, lang: str = "ben"):
        import pytesseract
        import shutil
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract binary not found in PATH")
        self._pytesseract = pytesseract
        self.lang = lang

    def process_page(self, image: np.ndarray) -> list[TextBlock]:
        from PIL import Image
        import pandas as pd
        
        pil_img = Image.fromarray(image)
        data = self._pytesseract.image_to_data(
            pil_img, 
            lang=self.lang, 
            output_type=self._pytesseract.Output.DATAFRAME
        )
        
        # Drop nan texts
        data = data.dropna(subset=['text'])
        # Drop empty strings
        data = data[data['text'].astype(str).str.strip() != '']
        
        blocks = []
        
        # Group by block_num
        for block_id, block_data in data.groupby('block_num'):
            # Calculate block bbox from its words
            b_xmin = block_data['left'].min()
            b_ymin = block_data['top'].min()
            b_xmax = (block_data['left'] + block_data['width']).max()
            b_ymax = (block_data['top'] + block_data['height']).max()
            block_bbox = BoundingBox(x=b_xmin, y=b_ymin, w=b_xmax-b_xmin, h=b_ymax-b_ymin)
            
            paragraphs = []
            for par_id, par_data in block_data.groupby('par_num'):
                # Calculate paragraph bbox from its words
                p_xmin = par_data['left'].min()
                p_ymin = par_data['top'].min()
                p_xmax = (par_data['left'] + par_data['width']).max()
                p_ymax = (par_data['top'] + par_data['height']).max()
                par_bbox = BoundingBox(x=p_xmin, y=p_ymin, w=p_xmax-p_xmin, h=p_ymax-p_ymin)
                
                # In Tesseract, confidence is per word
                words = []
                confidences = []
                for _, row in par_data.iterrows():
                    w_bbox = BoundingBox(x=row['left'], y=row['top'], w=row['width'], h=row['height'])
                    text = str(row['text']).strip()
                    conf = float(row['conf'])
                    words.append(TextWord(text=text, bbox=w_bbox))
                    confidences.append(conf)
                    
                if not words:
                    continue
                    
                para_text = " ".join(w.text for w in words)
                para_conf = sum(confidences) / len(confidences) if confidences else 0.0
                
                paragraphs.append(TextParagraph(
                    text=para_text, 
                    confidence=para_conf, 
                    bbox=par_bbox, 
                    words=words
                ))
                
            if paragraphs:
                blocks.append(TextBlock(block_id=int(block_id), bbox=block_bbox, paragraphs=paragraphs))
                
        return blocks


class SuryaEngine:
    name = "surya"

    def __init__(self, langs: list[str] | None = None):
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            import surya.settings
            if langs:
                surya.settings.LANGUAGES = langs
        except ImportError as e:
            raise ImportError("surya-ocr is not installed.") from e

        self._foundation_predictor = FoundationPredictor()
        self._recognition_predictor = RecognitionPredictor(self._foundation_predictor)
        self._detection_predictor = DetectionPredictor()

    def detect_layout(self, image: np.ndarray) -> list[BoundingBox]:
        from PIL import Image
        pil_img = Image.fromarray(image).convert("RGB")
        # detect layout returns a DetectionResult object
        result = self._detection_predictor([pil_img])[0]
        
        bboxes = []
        for poly in result.bboxes:
            # poly.bbox is [x1, y1, x2, y2]
            box = BoundingBox(
                x=poly.bbox[0],
                y=poly.bbox[1],
                w=poly.bbox[2] - poly.bbox[0],
                h=poly.bbox[3] - poly.bbox[1]
            )
            bboxes.append(box)
        return bboxes

    def ocr_polygons(self, image: np.ndarray, polygons: list[list[list[int]]]) -> list[tuple[str, float]]:
        from PIL import Image
        pil_img = Image.fromarray(image).convert("RGB")
        # polygons is list of polygons. We pass it as [[poly1, poly2, ...]] for the 1 image.
        res = self._recognition_predictor([pil_img], polygons=[polygons])
        
        results = []
        for text_line in res[0].text_lines:
            text = getattr(text_line, 'text', '')
            confidence = getattr(text_line, 'confidence', 0.0)
            results.append((text, confidence))
        return results

    def ocr_page(self, image: np.ndarray, bboxes: list[BoundingBox] = None) -> list[OcrLineResult]:
        from PIL import Image
        pil_img = Image.fromarray(image).convert("RGB")
        
        if bboxes is None:
            # Detect layout
            det_res = self._detection_predictor(images=[pil_img])[0]
            surya_bboxes = det_res.bboxes
        else:
            surya_bboxes = [[[b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h]] for b in bboxes]
            
        if not surya_bboxes:
            return []

        # Recognize text
        # If passing our own bboxes, it expects List[List[List[int]]]
        predictions = self._recognition_predictor(
            images=[pil_img],
            task_names=None,
            bboxes=[surya_bboxes] if bboxes is not None else None,
            det_predictor=self._detection_predictor if bboxes is None else None
        )
        
        page_result = predictions[0]
        lines = getattr(page_result, "text_lines", [])
        
        results = []
        for line in lines:
            poly = line.polygon # List of [x,y] points
            xmin = int(min(p[0] for p in poly))
            ymin = int(min(p[1] for p in poly))
            xmax = int(max(p[0] for p in poly))
            ymax = int(max(p[1] for p in poly))
            box = BoundingBox(x=xmin, y=ymin, w=xmax-xmin, h=ymax-ymin)
            
            conf = getattr(line, "confidence", 0.0) * 100
            results.append(OcrLineResult(text=line.text, confidence=conf, engine=self.name, bbox=box))
            
        return results

class HybridEngine:
    name = "hybrid"
    
    def __init__(self):
        self.tesseract = TesseractHierarchicalEngine(lang="ben")
        self.surya = SuryaEngine(["bn"])
        
    def process_page(self, image: np.ndarray) -> list[TextBlock]:
        # 1. Get layout from Tesseract
        blocks = self.tesseract.process_page(image)
        
        # 2. Reread text using SuryaOCR for all paragraphs in the page at once
        polygons = []
        paras = []
        for block in blocks:
            for para in block.paragraphs:
                # convert bbox to polygon [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x, y, w, h = para.bbox.x, para.bbox.y, para.bbox.w, para.bbox.h
                polygons.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                paras.append(para)
                
        if polygons:
            try:
                results = self.surya.ocr_polygons(image, polygons)
                for i, para in enumerate(paras):
                    if i < len(results):
                        para.text = results[i][0]
                        para.confidence = results[i][1]
            except Exception as e:
                print(f"[warn] Surya OCR failed in hybrid engine: {e}")
                    
        return blocks


class PaddleEngine:
    name = "paddle"

    def __init__(self, lang: str = "bn"):
        import os
        # Set HOME to workspace to avoid sandbox issues with ~/.paddleocr and ~/.paddlex
        os.environ["HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "home"))
        os.makedirs(os.environ["HOME"], exist_ok=True)
        
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError("paddleocr is not installed.") from e

        # redirect stdout/stderr to suppress verbose paddleocr logs during init
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def ocr_page(self, image: np.ndarray, bboxes: list[BoundingBox] = None) -> list[OcrLineResult]:
        if bboxes is None:
            raise ValueError("PaddleEngine requires bboxes provided by a layout detector")
            
        results = []
        for box in bboxes:
            crop = box.crop(image)
            # ocr.ocr returns a list of lines for the image
            res = self.ocr.ocr(crop, cls=True)
            
            # res is a list of results, one for each image.
            # res[0] is the result for our single crop image.
            if not res or not res[0]:
                results.append(OcrLineResult(text="", confidence=0.0, engine=self.name, bbox=box))
                continue
                
            line_texts = []
            min_conf = 100.0
            for line_res in res[0]:
                _, (text, conf) = line_res
                line_texts.append(text)
                min_conf = min(min_conf, float(conf) * 100)
                
            results.append(OcrLineResult(
                text=" ".join(line_texts),
                confidence=min_conf,
                engine=self.name,
                bbox=box
            ))
                
        return results


@dataclass
class TextWord:
    text: str
    bbox: BoundingBox

@dataclass
class TextParagraph:
    text: str
    confidence: float
    bbox: BoundingBox
    words: list[TextWord]

@dataclass
class TextBlock:
    block_id: int
    bbox: BoundingBox
    paragraphs: list[TextParagraph]

class GoogleVisionEngine:
    name = "google_vision"

    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.environ.get("VISION_API_KEY")
        if not self.api_key:
            raise ValueError("VISION_API_KEY environment variable not set")

    def process_page(self, image: np.ndarray) -> list[TextBlock]:
        import cv2
        import base64
        import requests

        # encode to jpg
        _, buffer = cv2.imencode('.jpg', image)
        content = base64.b64encode(buffer).decode('utf-8')

        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"

        payload = {
            "requests": [
                {
                    "image": {
                        "content": content
                    },
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION"
                        }
                    ],
                    "imageContext": {
                        "languageHints": ["bn"]
                    }
                }
            ]
        }

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"[error] Google Vision API failed: {response.status_code} {response.text}")
            return []

        res = response.json()
        annotations = res.get('responses', [{}])[0].get('fullTextAnnotation')
        if not annotations:
            return []

        blocks = []
        for p_idx, page in enumerate(annotations.get('pages', [])):
            for b_idx, block in enumerate(page.get('blocks', [])):
                block_box = self._extract_bbox(block.get('boundingBox'))
                
                paragraphs = []
                for para in block.get('paragraphs', []):
                    para_box = self._extract_bbox(para.get('boundingBox'))
                    para_conf = para.get('confidence', 0.0) * 100
                    
                    words = []
                    for word in para.get('words', []):
                        word_box = self._extract_bbox(word.get('boundingBox'))
                        # join symbols to form word text
                        word_text = "".join(
                            symbol.get('text', '') for symbol in word.get('symbols', [])
                        )
                        # Vision also sometimes puts spaces in the property.detectedBreak
                        # but typically joining words with space is fine at the paragraph level.
                        words.append(TextWord(text=word_text, bbox=word_box))
                        
                    para_text = " ".join(w.text for w in words)
                    paragraphs.append(TextParagraph(text=para_text, confidence=para_conf, bbox=para_box, words=words))
                    
                blocks.append(TextBlock(block_id=b_idx, bbox=block_box, paragraphs=paragraphs))

        return blocks

    def _extract_bbox(self, bounding_box_dict) -> BoundingBox:
        if not bounding_box_dict or 'vertices' not in bounding_box_dict:
            return BoundingBox(0, 0, 0, 0)
        vertices = bounding_box_dict['vertices']
        xs = [v.get('x', 0) for v in vertices]
        ys = [v.get('y', 0) for v in vertices]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return BoundingBox(x=xmin, y=ymin, w=xmax-xmin, h=ymax-ymin)


def get_available_engines() -> dict[str, object]:
    engines = {}
    try:
        engines["google_vision"] = GoogleVisionEngine()
    except Exception as e:
        print(f"[info] Google Vision engine unavailable: {e}")
        
    try:
        engines["tesseract"] = TesseractHierarchicalEngine()
    except Exception as e:
        print(f"[warn] Tesseract engine unavailable: {e}")

    try:
        engines["surya"] = SuryaEngine(["bn"])
    except Exception as e:
        print(f"[info] Surya engine not ready yet ({e}).")
        
    try:
        engines["paddle"] = PaddleEngine()
    except Exception as e:
        print(f"[info] Paddle engine unavailable ({e})")

    try:
        if "tesseract" in engines and "surya" in engines:
            engines["hybrid"] = HybridEngine()
    except Exception as e:
        print(f"[info] Hybrid engine unavailable ({e})")

    return engines
