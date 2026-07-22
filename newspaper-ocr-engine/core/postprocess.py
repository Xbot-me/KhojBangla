import os
from typing import List
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class LLMCorrector:
    def __init__(self, model_path: str = "weights/bangla_llm_q4.gguf"):
        """
        Initialize llama.cpp with the quantized GGUF model for fast local inference.
        """
        self.llm = None
        if Llama is not None and os.path.exists(model_path):
            print(f"Loading local LLM from {model_path}...")
            # Use n_gpu_layers=-1 to offload all layers to GPU if available
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=2048,
                verbose=False
            )
        else:
            # print("Warning: LLM model not found or llama_cpp not installed. Post-processing will be skipped.")
            pass
            
    def correct_texts(self, raw_texts: List[str]) -> List[str]:
        """
        Correct a batch of raw OCR text lines using the LLM.
        """
        if self.llm is None:
            return raw_texts
            
        system_prompt = (
            "You are an OCR correction engine. Fix spelling errors and missing ligatures "
            "in the following historical Bengali text. Do not add new information or summarize. "
            "Return only the corrected text."
        )
        
        corrected_texts = []
        for text in raw_texts:
            if not text.strip():
                corrected_texts.append(text)
                continue
                
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{text}\n<|assistant|>\n"
            
            try:
                # Force deterministic output with temperature 0.1
                response = self.llm(
                    prompt,
                    max_tokens=256,
                    temperature=0.1,
                    stop=["<|user|>", "<|system|>"],
                    echo=False
                )
                corrected = response['choices'][0]['text'].strip()
                corrected_texts.append(corrected)
            except Exception as e:
                print(f"LLM correction failed for text '{text}': {e}")
                corrected_texts.append(text)
                
        return corrected_texts
