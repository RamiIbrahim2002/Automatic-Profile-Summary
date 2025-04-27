import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import pdf2image
import json
from json.decoder import JSONDecodeError
from PIL import Image

from ..ocr_utils import run_ocr, query_hf_ocr
from ..schemas import ProfileForm, CV_SCHEMA
from ..config import openai

router = APIRouter()

@router.post("/process-cv/", response_model=ProfileForm)
async def process_cv(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.lower().endswith(".pdf"):
        pages = pdf2image.convert_from_bytes(contents, dpi=400, fmt="png", thread_count=4)
        text = "\n\n".join(f"--- PAGE {i+1} ---\n{run_ocr(img)}" for i, img in enumerate(pages))
    else:
        img = Image.open(io.BytesIO(contents))
        text = run_ocr(img)

    if not text.strip():
        text = query_hf_ocr(contents)
    if not text.strip():
        raise HTTPException(400, "No text extracted.")

    try:
        resp = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[{"role":"user","content":f"Extract CV fields as JSON:\n{text}"}],
            response_format={"type":"json_schema","json_schema":{"name":"cv_form","schema":CV_SCHEMA}}
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        edu = data.get("education", [])
        data["education"] = "\n".join(edu) if isinstance(edu, list) else edu
        return ProfileForm(**data)
    except JSONDecodeError:
        raise HTTPException(502, "Invalid JSON from LLM")
    except Exception as e:
        raise HTTPException(500, str(e))
