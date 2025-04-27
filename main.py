import os
import json
import logging
import requests
import io

from json.decoder import JSONDecodeError
from typing import List

from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import pdf2image
import pytesseract

# Load environment variables
load_dotenv()

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI and Hugging Face setup
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-base-stage1"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\mehdi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Pydantic models
class ProfileForm(BaseModel):
    name: str
    experience: str
    education: str
    skills: str
    contact: str

class ProfileSummary(BaseModel):
    summary: str
    reasoning: str
    tags: List[str]
    seo_keywords: List[str]

# OCR using Tesseract
def run_ocr(image):
    try:
        if image.mode != 'L':
            image = image.convert('L')
        image = image.point(lambda x: 0 if x < 140 else 255)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR with Tesseract failed: {str(e)}")
        return ""

# OCR fallback using Hugging Face
def query_hf_ocr(image_data: bytes) -> str:
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, data=image_data)
        if response.status_code == 200:
            data = response.json()
            return data.get('text', '')
        else:
            logger.error(f"HuggingFace OCR error: {response.status_code} {response.text}")
            return ""
    except Exception as e:
        logger.error(f"HuggingFace OCR request failed: {str(e)}")
        return ""

# Parse CV to structured form
def parse_cv_to_form(extracted_text: str) -> dict:
    prompt = f"""
    Extract the following fields from this CV text (return JSON ONLY):
    - name: Full name (first + last)
    - experience: Work experience (concise, max 200 words)
    - education: List of degrees/certifications with format: "Degree at Institution (Year)"
    - skills: Technical/soft skills (comma-separated)
    - contact: Email/phone (whichever is available)

    Rules:
    - For education: Return as a string with each entry on a new line
    - Omit fields if not found
    - Keep professional tone
    - Return raw text (no markdown)
    - English output preferred

    CV Text:
    {extracted_text}
    """

    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        education = result.get("education", "")
        if isinstance(education, list):
            education = "\n".join(education)

        return {
            "name": result.get("name", ""),
            "experience": result.get("experience", ""),
            "education": education,
            "skills": result.get("skills", ""),
            "contact": result.get("contact", "")
        }
    except Exception as e:
        logger.error(f"CV parsing failed: {str(e)}")
        return {
            "name": "",
            "experience": "",
            "education": "",
            "skills": "",
            "contact": ""
        }

# Main form page
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Process uploaded CV
@app.post("/process-cv/")
async def process_cv(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        if file.filename.lower().endswith('.pdf'):
            images = pdf2image.convert_from_bytes(contents, dpi=400, fmt='png', thread_count=4)
            extracted_text = ""
            for i, image in enumerate(images):
                page_text = run_ocr(image)
                extracted_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
        else:
            image = Image.open(io.BytesIO(contents))
            extracted_text = run_ocr(image)

        if not extracted_text.strip():
            logger.warning("Tesseract failed, trying Hugging Face OCR...")
            extracted_text = query_hf_ocr(contents)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

        return parse_cv_to_form(extracted_text)

    except Exception as e:
        logger.error(f"CV processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Generate profile summary with SEO keywords
@app.post("/generate", response_model=ProfileSummary)
async def generate(
    name: str = Form(...),
    experience: str = Form(...),
    education: str = Form(...),
    skills: str = Form(...),
    contact: str = Form(...),
):
    prompt = (
        f"Voici un profil rempli :\n"
        f"- Nom : {name}\n"
        f"- Expérience : {experience}\n"
        f"- Éducation : {education}\n"
        f"- Compétences : {skills}\n"
        f"- Coordonnées : {contact}\n\n"
        "Generate a professional profile summary. "
        "Return a strict JSON with these fields: summary, reasoning, tags, seo_keywords.\n"
        "Rules for tags:\n"
        "- Must be real LinkedIn SEO terms\n"
        "- Focus on real-world skills, industries, job titles\n"
        "- Avoid generic terms like 'teamwork', 'motivated'\n"
        "- Be specific (e.g., 'Cloud Security', 'Data Engineering', 'Product Management')\n"
        "- Use 5 to 10 tags max\n"
        "- English only.\n"
    )

    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "You are a professional career coach. Always answer with VALID JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        if not all(k in data for k in ["summary", "reasoning", "tags", "seo_keywords"]):
            raise ValueError("Missing required fields in the model output")

        return ProfileSummary(**data)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=502, detail="Invalid JSON response from OpenAI")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
