from pydantic_settings import BaseSettings, SettingsConfigDict

class config(BaseSettings):
    openai_api_key: str
    google_api_key: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

config = config()