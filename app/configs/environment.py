from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

@lru_cache
def get_env_filename():
    return ".env"

class EnvironmentSettings(BaseSettings):
    APP_NAME: str
    ENVIRONMENT : str
    HUGGINGFACE_ACCESS_TOKEN : str
    APP_PORT : int 
    CONTAINER_PORT : int

    model_config = SettingsConfigDict(
        env_file=get_env_filename(),
        env_file_encoding="utf-8",
    )

@lru_cache
def get_environment_variables():
    return EnvironmentSettings()
