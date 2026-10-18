from openai import OpenAI

from app.config import get_settings

settings = get_settings()

openai = OpenAI(api_key=settings.openai_api_key)