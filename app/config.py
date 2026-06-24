"""Application configuration via pydantic-settings."""
import secrets
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Citta Engine configuration loaded from environment variables."""
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./citta.db"
    REDIS_URL: str = ""
    # Auth -- generate random secret on first run if not set
    SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    # Engine limits
    DEFAULT_TIMEOUT_MS: int = 5000
    MAX_INPUT_CHARS: int = 1_000_000
    RATE_LIMIT_PER_MINUTE: int = 100
    # Billing defaults (USD)
    BILLING_API_CALL_COST: float = 0.001
    BILLING_PACKAGE_SUBSCRIPTION_COST: float = 5.0
    # Admin seed
    ADMIN_EMAIL: str = "admin@localhost"
    ADMIN_PASSWORD: str = ""
    # Rust engine toggle -- OFF by default.
    # Rust covers 6/7 condition types (no numeric_comparison, no CJK).
    # Enable only for high-throughput English text-pattern workloads.
    RUST_ENABLED: bool = False
    # LLM (optional)
    LLM_API_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"
    # CORS origins for production
    CORS_ORIGINS: str = "*"
    # Dev mode -- skip auth for local development.
    # Default OFF in production.  Set CITTA_DEV_MODE=true to enable.
    DEV_MODE: bool = False

    model_config = {"env_prefix": "CITTA_", "case_sensitive": False}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.SECRET_KEY:
            self.SECRET_KEY = secrets.token_urlsafe(32)
        if not self.ADMIN_PASSWORD:
            self.ADMIN_PASSWORD = secrets.token_urlsafe(16)


settings = Settings()
