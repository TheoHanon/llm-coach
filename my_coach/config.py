from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_small : str = "mistral-small-latest"
    model_large : str = "mistral-medium-latest"
    model_provider : str = "mistralai"

    temperature_small : float = 0.95
    top_p_small : float = 0.95

    temperature_large : float = 0.95
    top_p_large : float = 0.95

    save_path : str = "training_plan.csv"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",          # <- key line to avoid the error
        case_sensitive=False,
    )


settings = Settings()