"""
Single source of truth for environment configuration.

Everything else in the repo imports `settings` from here rather than
calling os.environ directly, so there's exactly one place that knows how
config is loaded and validated.
"""
from __future__ import annotations

from urllib.parse import quote_plus

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

    # --- Dashboard (password-only login page; see monitoring/dashboard/server.py) ---
    # One shared password gates the whole dashboard — the page itself and
    # every /api read alike. No username, no separate operator token for the
    # state-changing endpoints (e.g. POST /api/tests/run): enter this once
    # at /login and the session cookie it sets covers everything until it's
    # cleared (or the password changes, which invalidates every outstanding
    # cookie at once — see _session_token). No default on purpose: with this
    # empty the dashboard only serves on loopback.
    dashboard_password: str = Field(default="", alias="DASHBOARD_PASSWORD")

    # --- Feature set ---
    # Which feature set scripts/init_database.py builds when no
    # --feature-set-id is passed explicitly. Other CLI entry points
    # (scripts/run_weekly_cycle.py etc.) always pass --feature-set-id
    # themselves rather than reading this default.
    feature_set_id: str = Field(default="v4", alias="FEATURE_SET_ID")

    # --- Dashboard ---
    # Where the dashboard listens. Loopback by default so it isn't reachable
    # from the network by accident; the Docker image sets 0.0.0.0 explicitly
    # (containers need it to publish the port).
    dashboard_host: str = Field(default="127.0.0.1", alias="DASHBOARD_HOST")
    # Which port it binds. The alias is PORT, not DASHBOARD_PORT, because
    # that is the variable every PaaS (Railway included) injects into the
    # container and then routes the public domain to — binding anything else
    # there means the platform health-checks a dead port and the deploy is
    # marked failed. Unset locally, so 8501 stays the local default.
    dashboard_port: int = Field(default=8501, alias="PORT")

    # --- Alerts ---
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")

    # --- Trade approval / notifications (Telegram) ---
    # Used by execution/approval_gate.py. In "auto" mode (the default) it is
    # a notification channel only: proposals execute immediately and
    # Telegram receives a message once a batch has actually been acted on
    # (see send_followup and the post-trade outcome messages in
    # execution/trading_loop.py and execution/contradiction_monitor.py).
    # Blank credentials just mean those notifications are logged instead of
    # sent — auto mode never blocks or fails closed on a missing bot.
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    # "auto" = proposals execute immediately (no human reply required);
    # Telegram, if configured, gets a post-trade notification, never a
    # question. "telegram" = the old pre-trade human gate — every open/close
    # needs an "approve"/"reject" reply on the phone before it executes, and
    # blank credentials fail the whole batch closed rather than trading
    # unattended. Kept as an opt-in for anyone who wants the gate back.
    approval_mode: str = Field(default="auto", alias="APPROVAL_MODE")  # "auto" | "telegram"
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

    # --- Prediction target ---
    # What the model is trained to predict.
    #
    # "absolute" (the original) = each stock's raw forward return. That
    # number is dominated by whatever the whole market did, so the model
    # spent its capacity learning market drift and was then credited with
    # the drift as if it were skill: 94.6% of its trades were longs in a
    # rising market, and it still lost to buy-and-hold at every horizon.
    #
    # "relative" (default) = the cross-sectional excess return: a stock's
    # forward return minus the equal-weight mean forward return of the
    # universe on the SAME date. The market term cancels, leaving only the
    # part a stock-picker could have added — which is exactly what
    # excess_return grades. Features are z-scored per date to match, so the
    # model ranks stocks against same-day peers instead of absolute levels.
    #
    # Both modes are kept switchable so the comparison can be re-run; the
    # evaluation harness always measures money in absolute returns
    # regardless of which label the model was trained on.
    target_mode: str = Field(default="relative", alias="TARGET_MODE")  # "absolute" | "relative"

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

    # --- Per-pick exit levels (execution/exit_levels.py) ---
    # The two settings above are the fallback, used when a stock's
    # volatility can't be measured. Normally each pick gets its own levels,
    # derived from what the model predicted for it and how far that
    # particular stock normally moves — one pair of numbers cannot be right
    # for both a utility and a biotech.
    #
    # Take profit is the predicted move itself, bounded: never below what a
    # round trip costs (closing into a guaranteed loss), never above this
    # many horizon-sigmas (a target the stock has no history of reaching).
    exit_take_profit_max_sigmas: float = Field(default=2.0, alias="EXIT_TAKE_PROFIT_MAX_SIGMAS")
    exit_min_take_profit_pct: float = Field(default=0.03, alias="EXIT_MIN_TAKE_PROFIT_PCT")
    # Stop loss in horizon-sigmas. 1.5 is deliberately wider than one
    # sigma: at one sigma roughly a third of positions are stopped out by
    # ordinary wandering before a multi-week thesis has had time to be
    # right or wrong.
    exit_stop_loss_sigmas: float = Field(default=1.5, alias="EXIT_STOP_LOSS_SIGMAS")
    # Bounds, because volatility is estimated from a short window and can
    # be badly wrong right after a gap. Without them one quiet month would
    # set a 1% stop that closes on the first ordinary day.
    exit_min_stop_loss_pct: float = Field(default=0.05, alias="EXIT_MIN_STOP_LOSS_PCT")
    exit_max_stop_loss_pct: float = Field(default=0.20, alias="EXIT_MAX_STOP_LOSS_PCT")

    # --- Between-cycle emergency brake (execution/contradiction_monitor.py) ---
    # How far a held position must move against itself, over the monitor's
    # 5-day window, before it proposes closing mid-week.
    #
    # This is an emergency brake, not a second opinion. The weekly rules
    # above decide when a position has run its course; this exists only so a
    # collapse on a Tuesday isn't discovered the following Monday. It was
    # 0.04, which is roughly one week's ordinary movement for a typical S&P
    # 500 name — so it fired on noise, hourly, against a thesis that needs
    # ~20 trading days to play out, reintroducing exactly the churn that
    # HOLD_MAX_MISSED_CYCLES exists to prevent. 0.11 is about three standard
    # deviations of weekly movement: rare enough to mean something broke.
    #
    # Lower it only with evidence that real failures are being missed, not
    # because it has been quiet — quiet is the intended state.
    contradiction_momentum_pct: float = Field(default=0.11, alias="CONTRADICTION_MOMENTUM_PCT")

    # --- Direction ---
    # Whether the screener may propose short candidates at all.
    #
    # Off by default on the evidence: across the 10-fold walk-forward, short
    # trades paid -1.069% per trade at a 41.6% win rate, against +0.225% and
    # 52.1% for longs. Shorts were only 5% of trades and still dragged the
    # book down. When false, short candidates are dropped before sizing (see
    # models/screener.py) rather than sized to zero, so they never reach the
    # approval gate at all. The whole short code path — sizing caps,
    # shortability checks, the short leg of the concentrated split — is left
    # intact so this can be turned back on the day shorts demonstrate a
    # positive excess return over the benchmark.
    allow_shorts: bool = Field(default=False, alias="ALLOW_SHORTS")

    # --- Long/short ranking preference (only matters once ALLOW_SHORTS=true) ---
    # A small handicap applied to a short candidate's ranking score relative
    # to longs when the screener decides which candidates make the shortlist
    # — ties and close calls go to the long side. 0.0 = no preference, longs
    # and shorts compete purely on conviction. 1.0 effectively excludes every
    # short (use ALLOW_SHORTS=false for that instead — it's the tested,
    # evidenced way to turn shorts off entirely, and doesn't leave a stray
    # knob implying shorts are "on" when nothing can ever clear it).
    #
    # This only changes SELECTION ORDER — which candidates make the cut. A
    # short that is selected is still sized on its true conviction, under
    # the existing MAX_SHORT_POSITION_PCT cap below (already more
    # conservative than the long cap, for the structurally-uncapped-loss
    # reason noted there).
    short_ranking_penalty: float = Field(default=0.15, alias="SHORT_RANKING_PENALTY")
    # A short is exempted from the handicap above — ranked on raw conviction
    # like a long — when its own derived stop-loss (execution/exit_levels.py:
    # how far *this* stock has to move against the position, based on its
    # own volatility, before the loss is capped) is at or below this
    # fraction. This is what "little downside risk" means operationally
    # here: a calm stock the model is confident about, not a guess about the
    # trade's odds. A short with unmeasurable volatility (falls back to the
    # global HOLD_STOP_LOSS_PCT default, currently wider than this
    # threshold) does not qualify — unknown risk is handicapped, not waved
    # through.
    short_low_risk_stop_loss_pct: float = Field(default=0.06, alias="SHORT_LOW_RISK_STOP_LOSS_PCT")

    # --- Strategy selection ---
    # "diversified" (default) = top-k book sized by risk.sizing.select_trades
    # under the conservative caps above. "concentrated" = the small
    # high-conviction book below (needs the env cap overrides to breathe).
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

    # --- Concentrated strategy (models.screener.select_concentrated_trades) ---
    # A small, high-conviction book: never more than max_concentrated_positions
    # names held at once, never fewer than min_concentrated_positions UNLESS
    # fewer than that many actually clear the confidence bar that cycle — the
    # minimum is a target the screen tries to reach, never a reason to force
    # a trade with no real edge (see DEFAULT_MIN_ABS_RETURN in
    # models/screener.py). execution/contradiction_monitor.py's mid-week
    # reactivation targets this same range: if an emergency close drops the
    # book below the max, it re-screens immediately to top back up rather
    # than waiting for the next weekly cycle.
    min_concentrated_positions: int = Field(default=2, alias="MIN_CONCENTRATED_POSITIONS")
    max_concentrated_positions: int = Field(default=3, alias="MAX_CONCENTRATED_POSITIONS")
    # Capital is split across however many names are actually held (between
    # the min and max above), weighted by relative conviction — the stronger
    # the predicted move, the bigger that leg — bounded two ways so no single
    # pick swallows the whole book and no also-ran pick gets squeezed to a
    # token sliver:
    #   - max_concentrated_position_pct: hard ceiling on any one leg,
    #     regardless of how many names are held.
    #   - min_concentrated_leg_floor_fraction: every leg is guaranteed at
    #     least this fraction of what an EQUAL split would have given it
    #     (e.g. 0.6 with 3 legs = at least 0.6 * 1/3 = 20% each). Expressed
    #     as a fraction of the equal share, not an absolute percentage, so
    #     it stays feasible however many names end up held (2 or 3) instead
    #     of being tuned for one specific count.
    max_concentrated_position_pct: float = Field(default=0.70, alias="MAX_CONCENTRATED_POSITION_PCT")
    min_concentrated_leg_floor_fraction: float = Field(default=0.6, alias="MIN_CONCENTRATED_LEG_FLOOR_FRACTION")

    @property
    def db_url(self) -> str:
        # User and password are percent-encoded because they are credentials,
        # not URL syntax. A managed Postgres generates the password for you —
        # Railway's can contain '@', '/', ':' or '#' — and interpolating one
        # of those raw silently reshapes the URL, so the driver reports an
        # unreachable host rather than a bad password. quote_plus with an
        # empty safe set escapes every reserved character.
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return f"postgresql+psycopg2://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def is_live(self) -> bool:
        if self.trading_mode not in ("paper", "live"):
            raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {self.trading_mode!r}")
        return self.trading_mode == "live"


settings = Settings()
