# FastAPI Career Profile Generator

A modular FastAPI application that:

- Parses uploaded CV files (PDF or image) into structured data using Tesseract OCR with Hugging Face fallback.
- Generates a professional profile summary, reasoning, relevant tags, and SEO keywords via OpenAI’s structured JSON-Schema outputs.
- Automatically validates the generated summary against a set of questions and retries if validation fails.

## Pipeline
![pipline](pipeline.png)


## Features

- **CV Parsing**: Extract name, experience, education, skills, and contact information from PDF or image resumes.
- **Profile Generation**: Create a clear summary, reasoning, tags, and SEO keywords using OpenAI’s preview client with JSON-Schema-based prompts.
- **Automated Validation**: Ask the model to answer three yes/no questions about the generated summary; automatically retry generation up to 2 times if any check fails.
- **Modular Codebase**: Clean separation of concerns across configuration, schemas, OCR utilities, and routers.
- **Structured Responses**: All LLM outputs conform to explicit JSON Schemas and are validated by Pydantic models.

## Requirements

- Python 3.10+
- Tesseract OCR installed (for local OCR)
- A Hugging Face API token (for OCR fallback)
- OpenAI preview client (`openai>=0.27.0`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fastapi-career-app.git
   cd fastapi-career-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux/macOS
   .\.venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=gpt-4o-mini        # or another supported model
   HF_API_TOKEN=your_huggingface_token
   ```

5. Ensure Tesseract is accessible (update the path in `app/ocr_utils.py` if needed).

## Project Structure

```
fastapi_career_app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app & template route
│   ├── config.py            # Env loading & client setup
│   ├── schemas.py           # Pydantic models & JSON Schemas
│   ├── ocr_utils.py         # Tesseract + HF OCR helpers
│   └── routers/
│       ├── __init__.py
│       ├── cv_router.py     # /process-cv/ endpoint
│       └── profile_router.py# /generate endpoint with auto-validation
├── templates/
│   └── index.html           # Jinja2 form template
├── static/
│   └── style.css            # Custom styles
├── requirements.txt
└── README.md
```

## Usage

Start the server from the project root:
```bash
uvicorn app.main:app --reload --port 8000
```

- **GET /**
  - Renders the main form (upload CV or fill in profile fields).
  - Returns an HTML form.

- **POST /process-cv/**
  - Accepts a file (`multipart/form-data`).
  - Returns a JSON object matching the `ProfileForm` schema.

- **POST /generate**
  - Accepts form fields: `name`, `experience`, `education`, `skills`, `contact`.
  - Generates and validates a profile summary. Retries up to 2 times if validation fails.
  - Returns a JSON object matching the `ProfileSummary` schema.

## Example

```bash
# Generate summary from form data:
curl -X POST http://127.0.0.1:8000/generate \
  -F name="Jane Doe" \
  -F experience="5 years in software engineering" \
  -F education="BSc Computer Science" \
  -F skills="Python, FastAPI, Docker" \
  -F contact="jane@example.com"
```

Response:
```json
{
  "summary": "...",
  "reasoning": "...",
  "tags": ["Backend Development", "API Design", "Cloud Engineering"],
  "seo_keywords": ["FastAPI developer", "Python backend engineer"]
}
```

## Logging

- `INFO` logs for each call to generation and validation.
- `DEBUG` logs for raw and parsed LLM outputs.
- `WARNING` when validation fails and regeneration occurs.


