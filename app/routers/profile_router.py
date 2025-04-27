from fastapi import APIRouter, Form, HTTPException
from json.decoder import JSONDecodeError
import json, os
from typing import List

from ..schemas import ProfileSummary, ValidationItem, ValidationResponse, SUMMARY_SCHEMA, VALIDATION_SCHEMA
from ..config import openai, logger

router = APIRouter()

# Helper to generate summary
def _call_generate(name, experience, education, skills, contact) -> dict:
    logger.info("Calling generate summary API")
    prompt = (
        f"Profil:\n- Nom : {name}\n- Expérience : {experience}\n"
        f"- Éducation : {education}\n- Compétences : {skills}\n- Contact : {contact}\n\n"
        "Return JSON with summary, reasoning, tags, seo_keywords."
    )
    resp = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role":"system","content":"You are a career coach; output VALID JSON only."},
            {"role":"user","content":prompt}
        ],
        response_format={"type":"json_schema","json_schema":{"name":"profile_summary","schema":SUMMARY_SCHEMA}}
    )
    result = json.loads(resp.choices[0].message.content)
    logger.debug(f"Generated summary: {result}")
    return result

# Helper to validate summary
def _call_validate(summary_data: dict) -> List[ValidationItem]:
    logger.info("Calling validate summary API")
    questions = [
        "Does the summary accurately reflect the experience?",
        "Is the list of tags relevant and specific?",
        "Are the seo_keywords appropriate?"
    ]
    prompt = (
        f"Validate this profile summary JSON against the questions below and return as JSON schema:\n"
        f"ProfileSummary:\n{json.dumps(summary_data)}\nQuestions:\n"
    )
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    resp = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role":"system","content":"You are a validation assistant. Answer with JSON matching the provided schema."},
            {"role":"user","content":prompt}
        ],
        response_format={"type":"json_schema","json_schema":{"name":"validation","schema":VALIDATION_SCHEMA}}
    )
    raw = resp.choices[0].message.content
    logger.debug(f"Validation raw output: {raw}")
    data = json.loads(raw)
    items = [ValidationItem(**item) for item in data.get("results", [])]
    logger.info(f"Parsed validation items: {items}")
    return items

@router.post("/generate", response_model=ProfileSummary)
async def generate(
    name: str = Form(...),
    experience: str = Form(...),
    education: str = Form(...),
    skills: str = Form(...),
    contact: str = Form(...),
):
    # Generate and validate, with up to 2 retries
    summary_data = _call_generate(name, experience, education, skills, contact)
    for attempt in range(2):
        validation_items = _call_validate(summary_data)
        logger.info(f"Validation attempt {attempt+1}: {[item.valid for item in validation_items]}")
        if all(item.valid for item in validation_items):
            return ProfileSummary(**summary_data)
        logger.warning("Validation failed, regenerating summary...")
        summary_data = _call_generate(name, experience, education, skills, contact)
    # Return last result even if validation fails
    logger.warning("Returning summary after retries, validation still failed.")
    return ProfileSummary(**summary_data)
