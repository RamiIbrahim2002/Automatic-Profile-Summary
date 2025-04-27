import os
import json
import logging
import requests

from json.decoder import JSONDecodeError
from typing import List

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from fastapi import File, UploadFile

from dotenv import load_dotenv
load_dotenv()

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-base-stage1"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

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
    seo_keywords: List[str]

# ——— Initialisation FastAPI & OpenAI ———
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ——— Endpoint Ocr ———
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and extract text using Hugging Face OCR.
    """
    try:
        image_data = await file.read()
        ocr_result = query_hf_ocr(image_data)
        return {"text": ocr_result.get("text", "No text detected")}
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image")
    
# ——— Endpoint formulaire ———
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
# ...existing code...

# Updated prompt for better SEO keyword extraction
    prompt = (
        f"Voici un profil rempli :\n"
        f"- Nom : {name}\n"
        f"- Expérience : {experience}\n"
        f"- Éducation : {education}\n"
        f"- Compétences : {skills}\n"
        f"- Coordonnées : {contact}\n\n"
        "Génère un JSON strict respectant le schéma donné, incluant un résumé, un raisonnement, et des mots-clés optimisés pour le SEO. "
        "Les mots-clés doivent être spécifiques, puissants et optimisés pour les recherches sur LinkedIn, Upwork, ou Google. "
        "Utilise des combinaisons précises comme 'Full Stack JavaScript Developer', 'React.js Frontend Engineer', ou 'Certified AWS Solutions Architect'. "
        "Priorise les mots-clés qui incluent des technologies, certifications, industries, et rôles recherchés."
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
                "items": {"type": "string"},
                "description": "Suggestions de tags pour valoriser le profil"
            },
            "seo_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mots-clés optimisés pour le SEO, basés sur le profil"
            }
        },
        "required": ["summary", "reasoning", "tags", "seo_keywords"],
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

def validate_summary(profile_data: dict, summary: str, tags: List[str]) -> dict:
    """
    Validates the generated summary based on key questions.
    Returns a dictionary with True/False for each validation question.
    """
    validation_results = {
        "mentions_key_sections": "experience" in summary.lower() and "education" in summary.lower() and "skills" in summary.lower(),
        "no_factual_errors": all(value.lower() in summary.lower() for value in profile_data.values()),
        "logical_flow": summary.count(".") > 3,  # Example: Check if there are enough sentences for logical flow
        "emphasizes_career_points": len(summary.split()) > 30,  # Example: Ensure the summary is detailed enough
        "accurate_tags": all(tag.lower() in summary.lower() for tag in tags),
    }
    return validation_results

def validate_summary(profile_data: dict, summary: str, tags: List[str]) -> dict:
    """
    Valide le résumé généré en répondant à des questions clés.
    Retourne un dictionnaire avec True/False pour chaque question de validation.
    """
    validation_results = {
        # Vérifie si les sections clés (expérience, éducation, compétences) sont mentionnées
        "mentions_key_sections": (
            "experience" in summary.lower() and 
            "education" in summary.lower() and 
            "skills" in summary.lower()
        ),
        # Vérifie qu'il n'y a pas d'erreurs factuelles par rapport aux données du profil
        "no_factual_errors": all(
            value.lower() in summary.lower() for value in profile_data.values()
        ),
        # Vérifie si le résumé a un flux logique (ex. : assez de phrases pour être clair)
        "logical_flow": summary.count(".") > 3,
        # Vérifie si les points de carrière importants sont bien mis en avant
        "emphasizes_career_points": len(summary.split()) > 30,
        # Vérifie si les tags suggérés sont pertinents et bien alignés avec le résumé
        "accurate_tags": all(
            tag.lower() in summary.lower() for tag in tags
        ),
    }
    return validation_results

def correct_summary(profile_data: dict, summary: str, tags: List[str], validation_results: dict) -> dict:
    """
    Dynamically corrects the summary and tags based on validation results using fallback logic.
    Returns the improved summary and tags.
    """

    # Fallback for missing key sections
    if not validation_results["mentions_key_sections"]:
        missing_sections = []
        if "experience" not in summary.lower():
            missing_sections.append(f"experience in {profile_data['experience']}")
        if "education" not in summary.lower():
            missing_sections.append(f"education in {profile_data['education']}")
        if "skills" not in summary.lower():
            missing_sections.append(f"skills such as {profile_data['skills']}")
        if missing_sections:
            summary += " This profile includes " + ", ".join(missing_sections) + "."

    # Fallback for factual errors
    if not validation_results["no_factual_errors"]:
        summary = f"Based on the provided details: {profile_data['experience']}, {profile_data['education']}, and {profile_data['skills']}. " + summary

    # Fallback for logical flow
    if not validation_results["logical_flow"]:
        summary = reorganize_summary(summary)

    # Fallback for emphasizing career points
    if not validation_results["emphasizes_career_points"]:
        summary += " This summary highlights the most important career achievements and skills."

    # Fallback for inaccurate tags
    if not validation_results["accurate_tags"]:
        tags = generate_tags(profile_data)

    return {"summary": summary, "tags": tags}

def reorganize_summary(summary: str) -> str:
    """
    Reorganizes the summary to ensure logical flow.
    """
    # Example logic: Split sentences and reorder them
    sentences = summary.split(". ")
    if len(sentences) > 1:
        # Example: Move the first sentence to the end for better flow
        reordered = sentences[1:] + [sentences[0]]
        return ". ".join(reordered)
    return summary

def generate_tags(profile_data: dict) -> List[str]:
    """
    Generates tags dynamically based on profile data.
    """
    tags = []
    if "skills" in profile_data:
        tags.append(f"{profile_data['skills']} Expert")
    if "experience" in profile_data:
        tags.append(f"{profile_data['experience']} Specialist")
    if "education" in profile_data:
        tags.append(f"{profile_data['education']} Professional")
    return tags

def query_hf_ocr(image_data: bytes):
    """
    Sends an image to the Hugging Face OCR model and returns the extracted text.
    """
    response = requests.post(HF_API_URL, headers=HF_HEADERS, files={"inputs": image_data})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error with Hugging Face API")
    return response.json()


# Pour lancer :
# uvicorn main:app --reload --port 8000
