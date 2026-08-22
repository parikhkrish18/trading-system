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
    # Outside regular trading hours, market orders aren't accepted at all —
    # only limit orders with extended_hours=True. When this is on,
    # execution/broker_alpaca.py switches order types automatically outside
    # RTH; when off, orders submitted outside RTH just queue as normal DAY
    # market orders until the next open (the old, pre-extended-hours
    # behavior). IBKR support for this isn't implemented yet — see
    # execution/broker_ibkr.py. Default off: human approval adds latency,
    # and extended-hours limit orders against thin quotes interact badly
    # with a gate that can take minutes — re-enable deliberately via env.
    allow_extended_hours_trading: bool = Field(default=False, alias="ALLOW_EXTENDED_HOURS_TRADING")

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

    # --- Feature set ---
    # Which feature set the dashboard-triggered pipeline runs on. CLI
    # invocations still pass --feature-set-id explicitly; this only feeds
    # the manual-trigger endpoints in monitoring/dashboard/server.py.
    feature_set_id: str = Field(default="v4", alias="FEATURE_SET_ID")

    # --- Dashboard ---
    # Bearer token protecting mutating dashboard endpoints (POST
    # /api/tests/run). Blank is fine on localhost; required when the
    # dashboard binds a non-loopback interface.
    dashboard_api_token: str = Field(default="", alias="DASHBOARD_API_TOKEN")
    # Where the dashboard listens. Loopback by default so it isn't reachable
    # from the network by accident; the Docker image sets 0.0.0.0 explicitly
    # (containers need it to publish the port).
    dashboard_host: str = Field(default="127.0.0.1", alias="DASHBOARD_HOST")

    # --- Alerts ---
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")

    # --- Trade approval (Telegram) ---
    # Used by execution/approval_gate.py to send trade proposals to a human
    # and read back approve/reject replies. Blank = approval transport
    # unconfigured; in telegram mode that fails closed (all proposals
    # rejected) rather than trading unattended.
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    # "telegram" = every open/close needs a human reply on the phone;
    # "auto" = approve everything without asking (the old unattended
    # behavior, kept as a deliberate escape hatch).
    approval_mode: str = Field(default="telegram", alias="APPROVAL_MODE")  # "telegram" | "auto"
    # How long to wait for replies before giving up. Must stay under 3600 —
    # the hourly contradiction monitor shares the one Telegram bot, and a
    # poll that outlives the hour would collide with the next cycle.
    approval_timeout_s: int = Field(default=900, alias="APPROVAL_TIMEOUT_S")
    # What happens to a *close* proposal nobody answered in time. Opens
    # always fail closed (rejected). "reject" keeps closes symmetric;
    # "approve" lets risk-reducing exits proceed unattended on timeout.
    approval_timeout_close_action: str = Field(
        default="reject", alias="APPROVAL_TIMEOUT_CLOSE_ACTION"
    )  # "reject" | "approve"

    # --- Data freshness ---
    # The weekly cycle refuses to trade if the newest price or feature row
    # is older than this many days (scripts/run_weekly_cycle.py). Guards
    # against every ingest job failing silently — or vendors returning
    # empty responses — and the cycle then trading on last week's data.
    max_data_staleness_days: int = Field(default=3, alias="MAX_DATA_STALENESS_DAYS")

    # --- Risk limits ---
    max_drawdown_pct: float = Field(default=0.15, alias="MAX_DRAWDOWN_PCT")
    # Conservative defaults: these cap sizing in the diversified strategy
    # (risk.sizing.select_trades) AND act as circuit-breaker thresholds.
    # The concentrated 2-trade strategy legitimately deploys up to ~70% in
    # one name — when running STRATEGY_MODE=concentrated, raise these via
    # env (MAX_SINGLE_POSITION_PCT=0.80, MAX_CORRELATED_EXPOSURE_PCT=0.95)
    # so the breakers sit above the strategy's intended sizes instead of
    # tripping on normal operation.
    max_single_position_pct: float = Field(default=0.25, alias="MAX_SINGLE_POSITION_PCT")
    # Lower than max_single_position_pct deliberately: a long position can
    # only ever lose 100% of what's put in, but a short's loss is structurally
    # uncapped (the underlying can keep rising) — size shorts more
    # conservatively by default to reflect that asymmetry. Only used by the
    # diversified-book path (risk.sizing.select_trades), not the
    # concentrated strategy.
    max_short_position_pct: float = Field(default=0.15, alias="MAX_SHORT_POSITION_PCT")
    max_correlated_exposure_pct: float = Field(default=0.50, alias="MAX_CORRELATED_EXPOSURE_PCT")

    # --- Forecast horizon ---
    # How many trading days ahead the model predicts (and therefore how
    # long a position is meant to be held). 20 ≈ a calendar month: the
    # swing-trade posture — multi-week holds targeting 3-10% moves, where
    # the fixed round-trip cost floor eats proportionally less of the
    # expected move than it does at 5 days. Used as the default by
    # models/train.py and models/screener.py; scripts/compare_horizons.py
    # measures whether a given value actually earns its keep.
    target_horizon_days: int = Field(default=20, alias="TARGET_HORIZON_DAYS")

    # --- Hold rules (execution/hold_rules.py) ---
    # With multi-week holds (TARGET_HORIZON_DAYS above), a position must NOT
    # be closed just because something else scored marginally higher on
    # Monday. A held position is closed only when a real exit condition
    # fires; these settings define "real".
    #
    # Consecutive weekly cycles a position can miss the shortlist before it
    # is proposed for closing. 2 means: slip out once, you're held; still
    # out the following week, the close goes to the human.
    hold_max_missed_cycles: int = Field(default=2, alias="HOLD_MAX_MISSED_CYCLES")
    # Unrealized-loss fraction at which a close is proposed (stop loss).
    hold_stop_loss_pct: float = Field(default=0.08, alias="HOLD_STOP_LOSS_PCT")
    # Unrealized-gain fraction at which a close is proposed — the top of the
    # 3-10% move band the swing horizon targets.
    hold_take_profit_pct: float = Field(default=0.10, alias="HOLD_TAKE_PROFIT_PCT")

    # --- Strategy selection ---
    # "diversified" (default) = top-k book sized by risk.sizing.select_trades
    # under the conservative caps above. "concentrated" = the 2-trade
    # high-conviction split below (needs the env cap overrides to breathe).
    strategy_mode: str = Field(default="diversified", alias="STRATEGY_MODE")  # "diversified" | "concentrated"
    # How many names the diversified book holds at most.
    screener_top_k: int = Field(default=10, alias="SCREENER_TOP_K")
    # When true, the diversified screener scales its shortlist up
    # proportionally until the book is ~100% allocated instead of leaving
    # the unused remainder in cash (risk.sizing.scale_to_full_deployment).
    # The per-position caps above still bind — if they're reached first, the
    # cappable maximum is deployed and the shortfall is logged. Off by
    # default: being fully invested is a decision about risk appetite, not a
    # default anyone should inherit by accident.
    full_deployment: bool = Field(default=False, alias="FULL_DEPLOYMENT")

    # --- Concentrated 2-trade strategy (models.screener.select_concentrated_trades) ---
    # Split between the two highest-conviction picks is weighted by relative
    # confidence, bounded so the dominant leg can't swallow the whole
    # deployment: max_concentrated_position_pct caps it, and
    # min_concentrated_position_pct (= 1 - max) floors the other leg.
    max_concentrated_position_pct: float = Field(default=0.70, alias="MAX_CONCENTRATED_POSITION_PCT")
    min_concentrated_position_pct: float = Field(default=0.30, alias="MIN_CONCENTRATED_POSITION_PCT")

    @property
    def db_url(self) -> str:
        return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def is_live(self) -> bool:
        if self.trading_mode not in ("paper", "live"):
            raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {self.trading_mode!r}")
        return self.trading_mode == "live"


settings = Settings()
