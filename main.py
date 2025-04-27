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
import pytesseract
from dotenv import load_dotenv

load_dotenv()

# Configure Tesseract path (uncomment and adjust for your system if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\mehdi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac

def run_ocr(image):
    try:
        # Convert to grayscale for better OCR results
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply basic preprocessing
        # 1. Thresholding to binarize the image
        image = image.point(lambda x: 0 if x < 140 else 255)
        
        # Use pytesseract to extract text
        custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return text.strip()
    
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
    logger.info(f"extracted_text: {extracted_text}")
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Handle education formatting
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

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# FastAPI & OpenAI initialization
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Form endpoint
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
                dpi=400,
                fmt='png',
                thread_count=4
            )
            extracted_text = ""
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}")
                # Save image for debugging (optional)
                image.save(f"page_{i+1}.png")
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

# Generation endpoint
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
        "Follow these rules for tags:\n"
        "1. Must be real LinkedIn SEO terms\n"
        "2. Prioritize skills recruiters search for\n"
        "3. Mix of job titles, technical skills, and industry keywords\n"
        "4. Avoid generic terms\n"
        "5. English preferred\n"
        "6. 5-10 tags max\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "reasoning": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 10
            }
        },
        "required": ["summary", "reasoning", "tags"],
        "additionalProperties": False
    }

    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "Tu es un assistant qui génère des résumés de profil."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object", "schema": schema}
        )

        data = json.loads(response.choices[0].message.content)
        return ProfileSummary(**data)

    except JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        raise HTTPException(status_code=502, detail="Invalid JSON from LLM")
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)