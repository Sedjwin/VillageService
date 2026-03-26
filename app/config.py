from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./data/village.db"
    aigateway_url: str = "http://localhost:8001"
    aigateway_token: str = "change-me"
    dashboard_url: str = "http://localhost:8000"
    usermanager_url: str = "http://localhost:8005"
    default_tick_rate: int = 60
    agentmanager_url: str = "http://localhost:8003"
    # Empty string = use AIGateway token's default model (recommended)
    # Set to a specific model ID via admin panel or .env once you have a dedicated token
    world_agent_model: str = ""
    agent_brain_model: str = ""
    conversation_model: str = ""

    model_config = {"env_file": ".env"}


settings = Settings()
