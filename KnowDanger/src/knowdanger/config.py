from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    openai_api_key: str | None = None
    p_threshold: float = 0.80
    class Config: env_prefix = "KNOWDANGER_"
settings = Settings()  # import and use
