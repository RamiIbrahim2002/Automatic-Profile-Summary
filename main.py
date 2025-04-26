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

from dotenv import load_dotenv
load_dotenv()

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
    prompt = (
        f"Voici un profil rempli :\n"
        f"- Nom : {name}\n"
        f"- Expérience : {experience}\n"
        f"- Éducation : {education}\n"
        f"- Compétences : {skills}\n"
        f"- Coordonnées : {contact}\n\n"
        "Génère un JSON strict respectant le schéma donné."
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
