from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SECRET_KEY: str = "dev-secret-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h for POC
    DB_PATH: str = "./interviewiq.db"
    CHROMA_PATH: str = "./chroma_db"
    LLM_PROVIDER: str = "groq"
    WHISPER_MODEL: str = "base"


settings = Settings()
