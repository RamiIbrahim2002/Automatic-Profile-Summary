import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-base-stage1"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}