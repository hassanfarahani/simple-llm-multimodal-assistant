# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    openai_api_key: str
    hf_token: str
    database_url: str
    database_url_host: str
    use_paid_models: bool = False                       # ← the switch
    ollama_base_url: str = "http://host.docker.internal:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()