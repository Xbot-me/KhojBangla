import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List
from PIL import Image
import numpy as np

class TrOCREngine:
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        """
        Initialize the TrOCR engine for batch inference.
        Uses microsoft/trocr-base-printed as a placeholder until fine-tuned historical weights are provided.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TrOCR model '{model_name}' on {self.device}...")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def recognize_lines(self, line_images: List[np.ndarray], batch_size: int = 16) -> List[str]:
        """
        Run TrOCR on a list of cropped line images in batches to maximize GPU utilization.
        """
        if not line_images:
            return []
            
        # Convert OpenCV numpy arrays to PIL Images (RGB)
        pil_images = []
        for img in line_images:
            if len(img.shape) == 2:
                # convert grayscale to RGB
                img = np.stack((img,)*3, axis=-1)
            pil_images.append(Image.fromarray(img).convert("RGB"))
            
        results = []
        
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i + batch_size]
                
                # Preprocess batch
                pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
                
                # Generate text
                generated_ids = self.model.generate(pixel_values)
                
                # Decode text
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                results.extend(generated_text)
                
        return results
