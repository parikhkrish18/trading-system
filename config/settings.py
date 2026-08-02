"""
Single source of truth for environment configuration.

Everything else in the repo imports `settings` from here rather than
calling os.environ directly, so there's exactly one place that knows how
config is loaded and validated.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Broker ---
    alpaca_paper_api_key: str = Field(default="", alias="ALPACA_PAPER_API_KEY")
    alpaca_paper_secret_key: str = Field(default="", alias="ALPACA_PAPER_SECRET_KEY")
    alpaca_paper_base_url: str = Field(default="https://paper-api.alpaca.markets", alias="ALPACA_PAPER_BASE_URL")

    alpaca_live_api_key: str = Field(default="", alias="ALPACA_LIVE_API_KEY")
    alpaca_live_secret_key: str = Field(default="", alias="ALPACA_LIVE_SECRET_KEY")
    alpaca_live_base_url: str = Field(default="https://api.alpaca.markets", alias="ALPACA_LIVE_BASE_URL")

    trading_mode: str = Field(default="paper", alias="TRADING_MODE")  # "paper" | "live"
    broker: str = Field(default="ibkr", alias="BROKER")  # "ibkr" | "alpaca"

    # --- IBKR (TWS / IB Gateway socket — no REST keys) ---
    ibkr_host: str = Field(default="127.0.0.1", alias="IBKR_HOST")
    ibkr_client_id: int = Field(default=1, alias="IBKR_CLIENT_ID")
    ibkr_live_client_id: int = Field(default=2, alias="IBKR_LIVE_CLIENT_ID")
    ibkr_paper_port: int = Field(default=7497, alias="IBKR_PAPER_PORT")  # TWS paper
    ibkr_live_port: int = Field(default=7496, alias="IBKR_LIVE_PORT")  # TWS live

    # --- Data vendors ---
    polygon_api_key: str = Field(default="", alias="POLYGON_API_KEY")
    fundamentals_news_api_key: str = Field(default="", alias="FUNDAMENTALS_NEWS_API_KEY")
    # Free key: https://fred.stlouisfed.org/docs/api/api_key.html — used for
    # CPI/jobs release dates in data/ingest/macro_calendar.py (BLS's own site
    # blocks automated requests, see that module's docstring).
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")

    # --- LLM (sentiment scoring) ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # --- DB ---
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="trading", alias="DB_NAME")
    db_user: str = Field(default="trading", alias="DB_USER")
    db_password: str = Field(default="change_me_locally", alias="DB_PASSWORD")

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")

    # --- Alerts ---
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")

    # --- Trade approval (Telegram) ---
    # Used by execution/approval_loop.py to send trade proposals to a human
    # and read back approve/reject replies. Blank = approval transport
    # unconfigured; the loop's Telegram step is a stub either way today.
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # --- Risk limits ---
    max_drawdown_pct: float = Field(default=0.15, alias="MAX_DRAWDOWN_PCT")
    max_single_position_pct: float = Field(default=0.25, alias="MAX_SINGLE_POSITION_PCT")
    # Lower than max_single_position_pct deliberately: a long position can
    # only ever lose 100% of what's put in, but a short's loss is structurally
    # uncapped (the underlying can keep rising) — size shorts more
    # conservatively by default to reflect that asymmetry.
    max_short_position_pct: float = Field(default=0.15, alias="MAX_SHORT_POSITION_PCT")
    max_correlated_exposure_pct: float = Field(default=0.50, alias="MAX_CORRELATED_EXPOSURE_PCT")

    @property
    def db_url(self) -> str:
        return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def is_live(self) -> bool:
        if self.trading_mode not in ("paper", "live"):
            raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {self.trading_mode!r}")
        return self.trading_mode == "live"


settings = Settings()
