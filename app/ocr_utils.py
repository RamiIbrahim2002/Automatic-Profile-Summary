from PIL import Image
import pytesseract
import requests
from .config import HF_API_URL, HF_HEADERS
import logging

logger = logging.getLogger(__name__)

def run_ocr(image: Image.Image) -> str:
    if image.mode != 'L':
        image = image.convert('L')
    image = image.point(lambda x: 0 if x < 140 else 255)
    return pytesseract.image_to_string(image, config="--oem 3 --psm 6").strip()

def query_hf_ocr(image_data: bytes) -> str:
    try:
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, data=image_data)
        resp.raise_for_status()
        return resp.json().get("text", "")
    except Exception as e:
        logger.error(f"HuggingFace OCR error: {e}")
        return ""