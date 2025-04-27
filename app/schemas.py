from pydantic import BaseModel
from typing import List

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

class ValidationItem(BaseModel):
    question: str
    valid: bool

class ValidationResponse(BaseModel):
    results: List[ValidationItem]

# JSON Schemas
CV_SCHEMA = {
    "type": "object",
    "properties": {
        "name":       {"type": "string"},
        "experience": {"type": "string"},
        "education":  {"type": "array", "items": {"type": "string"}},
        "skills":     {"type": "string"},
        "contact":    {"type": "string"},
    },
    "required": ["name", "experience", "skills"],
    "additionalProperties": False
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary":      {"type": "string"},
        "reasoning":    {"type": "string"},
        "tags":         {"type": "array", "items": {"type": "string"}},
        "seo_keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["summary", "reasoning", "tags", "seo_keywords"],
    "additionalProperties": False
}

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "valid":    {"type": "boolean"}
                },
                "required": ["question", "valid"],
                "additionalProperties": False
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}
