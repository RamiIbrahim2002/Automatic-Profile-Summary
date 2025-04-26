import os
import json
import logging
from json.decoder import JSONDecodeError
from typing import List

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI

from fastapi import UploadFile, File
from PIL import Image
import io
import pdf2image
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load model once at startup
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def run_ocr(image):
    try:
        # Convert to RGB if needed (but keep as RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply contrast enhancement while maintaining RGB
        enhanced = image.copy()
        for channel in range(3):  # Apply to each RGB channel
            enhanced.putchannel(
                image.getchannel(channel).point(lambda x: 0 if x < 140 else 255),
                channel
            )
        
        # Process with TrOCR
        pixel_values = processor(enhanced, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=2048,
                num_beams=4,
                early_stopping=True
            )
        
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return ""

def parse_cv_to_form(extracted_text: str) -> dict:
    """
    Uses OpenAI to extract structured data from OCR text
    Returns dict with keys: name, experience, education, skills, contact
    """
    prompt = f"""
    Extract the following fields from this CV text (return JSON ONLY):
    - name: Full name (first + last)
    - experience: Work experience (concise, max 200 words)
    - education: Degrees/certifications (institution names + degrees)
    - skills: Technical/soft skills (comma-separated)
    - contact: Email/phone (whichever is available)

    Rules:
    - Omit fields if not found
    - Keep professional tone
    - Return raw text (no markdown)
    - English output preferred

    CV Text:
    {extracted_text}
    """
    logger.info(f"extracted_text: {extracted_text}")
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "name": result.get("name", ""),
            "experience": result.get("experience", ""),
            "education": result.get("education", ""),
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

# ——— Configuration du logger ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Modèles Pydantic ———
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

# ——— Initialisation FastAPI & OpenAI ———
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add this at the start of your application
def verify_pdf2image():
    try:
        from pdf2image.exceptions import PDFInfoNotInstalledError
        try:
            pdf2image.convert_from_bytes(b'%PDF', dpi=100)
            logger.info("pdf2image is working correctly")
        except PDFInfoNotInstalledError:
            logger.error("poppler-utils not installed! Install with:")
            logger.error("Windows: conda install -c conda-forge poppler")
            logger.error("Mac: brew install poppler")
            logger.error("Linux: sudo apt-get install poppler-utils")
            raise
    except Exception as e:
        logger.error(f"PDF processing verification failed: {str(e)}")
        raise

# Call this during startup
verify_pdf2image()

# ——— Endpoint formulaire ———
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-cv/")
async def process_cv(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        if file.filename.lower().endswith('.pdf'):
            images = pdf2image.convert_from_bytes(
                contents,
                dpi=300,
                fmt='png',  # Ensure proper format
                thread_count=4  # Faster processing
            )
            extracted_text = ""
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}")
                page_text = run_ocr(image)
                extracted_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
        else:
            image = Image.open(io.BytesIO(contents))
            extracted_text = run_ocr(image)
        
        logger.info(f"Extracted text length: {len(extracted_text)} chars")
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted")
            
        return parse_cv_to_form(extracted_text)
        
    except Exception as e:
        logger.error(f"CV processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ——— Endpoint de génération ———
@app.post("/generate", response_model=ProfileSummary)
async def generate(
    name: str = Form(...),
    experience: str = Form(...),
    education: str = Form(...),
    skills: str = Form(...),
    contact: str = Form(...),
):
    # Construction du prompt
    prompt = (
        f"Voici un profil rempli :\n"
        f"- Nom : {name}\n"
        f"- Expérience : {experience}\n"
        f"- Éducation : {education}\n"
        f"- Compétences : {skills}\n"
        f"- Coordonnées : {contact}\n\n"
        "Follow these rules for tags:\n"
        "1. **Must be real LinkedIn SEO terms** (check against trending LinkedIn profiles)\n"
        "2. **Prioritize skills recruiters search for** (e.g., 'Python Developer' not just 'Python')\n"
        "3. **Mix of:**\n"
        "   - **Job titles** (e.g., 'Digital Marketing Specialist')\n"
        "   - **Technical skills** (e.g., 'Google Analytics Certified')\n"
        "   - **Industry keywords** (e.g., 'FinTech' if in finance)\n"
        "4. **Avoid generic terms** (e.g., 'Hardworking')\n"
        "5. **English preferred** (unless local market requires otherwise)\n"
        "6. **5-10 tags max**\n\n"
        
        "Example of good LinkedIn tags:\n"
        "- 'AI Engineer'\n"
        "- 'Cloud Architecture'\n"
        "- 'Data Visualization Expert'\n"
        "- 'Agile Project Management'\n"
        "- 'SEO & Content Marketing'\n\n"
    )

    # Définition du JSON Schema
    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Résumé clair et concis du profil"
            },
            "reasoning": {
                "type": "string",
                "description": "Raisonnement expliquant pourquoi le résumé couvre tout"
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": "^[A-Z][a-zA-Z0-9& ]+$",  # Enforces Title Case (e.g., "Machine Learning")
                    "description": "Must be a real LinkedIn SEO term (e.g., 'Frontend Developer')"
                },
                "minItems": 5,
                "maxItems": 10
            }
        },
        "required": ["summary", "reasoning", "tags"],
        "additionalProperties": False
    }

    try:
        # Appel à l’API OpenAI avec Structured Outputs
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "Tu es un assistant qui génère des résumés de profil."},
                {"role": "user",   "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "profile_summary",
                    "schema": schema
                }
            }
        )

        raw = response.choices[0].message.content
        #logger.info(f"[LLM] Raw response: {raw!r}")

        data = json.loads(raw)
        result = ProfileSummary(**data)
        logger.info(f"[LLM] Parsed ProfileSummary: {result}")

        return result

    except JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e} — content was: {raw!r}")
        raise HTTPException(status_code=502, detail="Invalid JSON from LLM")

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Pour lancer :
# uvicorn main:app --reload --port 8000
