import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults to llama3 but can be overridden
DEFAULT_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are an expert in the Bengali language and historical archives.
You will be given OCR output from a historical Bengali newspaper.
The OCR may contain misspelled words, random characters, or missing text due to faded ink or damage.
Your task is to correct the text and reconstruct missing sections where appropriate.
Do NOT change the original historical meaning. Do NOT translate to English.
Output ONLY a valid JSON object with the following keys:
- "corrected_text": The corrected and reconstructed Bengali text.
- "is_reconstructed": A boolean (true or false) indicating if you had to heavily reconstruct missing or garbled words.

Input OCR Text:
{text}
"""

def correct_text_with_llm(text: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Sends the OCR text to a local Ollama LLM for correction.
    Returns a dictionary with 'corrected_text' and 'is_reconstructed'.
    """
    if not text or not text.strip():
        return {"corrected_text": text, "is_reconstructed": False}

    prompt = PROMPT_TEMPLATE.format(text=text)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        response_text = data.get("response", "{}")
        
        # Parse the JSON response
        result = json.loads(response_text)
        return {
            "corrected_text": result.get("corrected_text", text),
            "is_reconstructed": result.get("is_reconstructed", False)
        }
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama request failed: {e}. Returning original text.")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Ollama JSON response: {e}. Returning original text.")
    except Exception as e:
        logger.warning(f"Unexpected error in LLM correction: {e}. Returning original text.")
        
    return {"corrected_text": text, "is_reconstructed": False}

if __name__ == "__main__":
    # Test
    sample = "বাংলার কৃষক কু__ উপকৃত হউক"
    res = correct_text_with_llm(sample)
    print(f"Original: {sample}")
    print(f"Corrected: {res}")
