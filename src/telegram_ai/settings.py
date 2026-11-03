import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(default="", repr=False)
    database_url: str = field(default="", repr=False)
    database_url_unpooled: str = field(default="", repr=False)
    llm_api_key: str = field(default="", repr=False)
    webhook_secret: str = field(default="", repr=False)
    allowed_user_ids: frozenset[int] = frozenset()
    webhook_base_url: str = ""
    llm_api_base: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemma-4-31b-it:free"
    llm_free_only: bool = True
    port: int = 10000
    workers: int = 2
    min_request_interval: float = 5
    max_pending_per_user: int = 3
    max_pending_total: int = 30
    history_turns: int = 12
    history_chars: int = 24000
    max_input_chars: int = 8000
    max_output_tokens: int = 1024
    max_output_chars: int = 12000
    model_timeout: float = 60
    telegram_timeout: float = 15
    lease_seconds: float = 180
    max_generation_attempts: int = 2
    max_send_attempts: int = 4

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        def value(key: str, default: str = "") -> str:
            return env.get(key, "").strip() or default

        ids = frozenset(
            int(item.strip()) for item in value("ALLOWED_USER_IDS").split(",") if item.strip()
        )
        if any(item <= 0 for item in ids):
            raise ValueError("ALLOWED_USER_IDS: нужны положительные Telegram user ID")
        flag = value("LLM_FREE_ONLY", "true").lower()
        if flag not in {"true", "false"}:
            raise ValueError("LLM_FREE_ONLY: допустимы true и false")
        settings = cls(
            bot_token=value("BOT_TOKEN"),
            database_url=value("DATABASE_URL"),
            database_url_unpooled=value("DATABASE_URL_UNPOOLED"),
            llm_api_key=value("LLM_API_KEY"),
            webhook_secret=value("WEBHOOK_SECRET"),
            allowed_user_ids=ids,
            webhook_base_url=value("WEBHOOK_BASE_URL", value("RENDER_EXTERNAL_URL")).rstrip("/"),
            llm_api_base=value("LLM_API_BASE", cls.llm_api_base).rstrip("/"),
            llm_model=value("LLM_MODEL", cls.llm_model),
            llm_free_only=flag == "true",
            port=int(value("PORT", "10000")),
            workers=int(value("WORKERS", "2")),
            min_request_interval=float(value("MIN_REQUEST_INTERVAL", "5")),
            max_pending_per_user=int(value("MAX_PENDING_PER_USER", "3")),
            max_pending_total=int(value("MAX_PENDING_TOTAL", "30")),
            history_turns=int(value("HISTORY_TURNS", "12")),
            history_chars=int(value("HISTORY_CHARS", "24000")),
            max_input_chars=int(value("MAX_INPUT_CHARS", "8000")),
            max_output_tokens=int(value("MAX_OUTPUT_TOKENS", "1024")),
            max_output_chars=int(value("MAX_OUTPUT_CHARS", "12000")),
            model_timeout=float(value("MODEL_TIMEOUT", "60")),
        )
        if not 1 <= settings.workers <= 4:
            raise ValueError("WORKERS: допустимо от 1 до 4")
        for name in (
            "port",
            "max_pending_per_user",
            "max_pending_total",
            "history_turns",
            "history_chars",
            "max_input_chars",
            "max_output_tokens",
            "max_output_chars",
            "model_timeout",
        ):
            if getattr(settings, name) <= 0:
                raise ValueError(f"{name}: значение должно быть положительным")
        if settings.min_request_interval < 0:
            raise ValueError("MIN_REQUEST_INTERVAL: значение должно быть неотрицательным")
        if settings.max_input_chars > settings.history_chars:
            raise ValueError("MAX_INPUT_CHARS не может превышать HISTORY_CHARS")
        if settings.model_timeout > 120:
            raise ValueError("MODEL_TIMEOUT не может превышать 120 секунд")
        if settings.llm_free_only and (
            settings.llm_api_base != cls.llm_api_base or not settings.llm_model.endswith(":free")
        ):
            raise ValueError("Бесплатный режим требует OpenRouter и модель с суффиксом :free")
        return settings

    def validate_runtime(self) -> None:
        for name in ("bot_token", "database_url", "llm_api_key", "webhook_secret"):
            if not getattr(self, name):
                raise ValueError(f"Не задана переменная {name.upper()}")
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", self.webhook_secret):
            raise ValueError("WEBHOOK_SECRET: нужны 32–256 букв, цифр, знаков _ или -")
        parsed = urlsplit(self.webhook_base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.query:
            raise ValueError("WEBHOOK_BASE_URL: нужен публичный HTTPS URL")
