import math
import os
import tomllib
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]


class ModelSettings(BaseModel):
    id: str
    base_url: str
    temperature: float = 0
    max_output_tokens: int = Field(default=1024, gt=0)
    context_window: int = Field(default=131072, gt=0)
    timeout_seconds: float = Field(default=90, gt=0)
    input_price: float = 0
    output_price: float = 0
    cache_price: float = 0
    reasoning_effort: str = "none"
    provider: str | None = None


class ContextSettings(BaseModel):
    max_input_tokens: int = Field(default=8192, gt=0)
    keep_recent_turns: int = Field(default=3, ge=0)
    strategy: str = "tiered"
    tokenizer: str
    tokenizer_revision: str
    estimate_margin: float = Field(default=1.3, ge=1)
    summary_tokens: int = Field(default=512, gt=0)


class Settings(BaseModel):
    root: Path = ROOT
    model: ModelSettings
    context: ContextSettings
    agent: dict[str, Any]
    tools: dict[str, Any]
    retrieval: dict[str, Any]
    pi: dict[str, Any]
    evaluation: dict[str, Any]

    @model_validator(mode="after")
    def check_budget(self) -> "Settings":
        if self.context.max_input_tokens + self.model.max_output_tokens > self.model.context_window:
            raise ValueError("Input budget + output reserve exceed the model context window")

        if self.context.summary_tokens > self.model.max_output_tokens:
            raise ValueError("context.summary_tokens must not exceed model.max_output_tokens")

        if self.context.strategy not in {"trim", "tiered"}:
            raise ValueError("context.strategy must be trim or tiered")

        return self

    @model_validator(mode="after")
    def check_limits(self) -> "Settings":
        for key in ("max_steps", "max_llm_calls"):
            value = self.agent.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"agent.{key} must be a positive integer")

        for key in ("turn_timeout_seconds", "tool_timeout_seconds"):
            value = self.agent.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"agent.{key} must be a finite positive number")

        retries = self.agent.get("api_max_retries")
        if type(retries) is not int or retries < 0:
            raise ValueError("agent.api_max_retries must be a nonnegative integer")
        return self

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def runtime(self) -> Path:
        return self.data / "runtime"

    @property
    def sessions(self) -> Path:
        return self.data / "06_sessions"

    @property
    def checks(self) -> Path:
        return self.data / "07_checks"

    @property
    def database_url(self) -> str:
        return os.environ.get("LAB3_DATABASE_URL", "")

    @property
    def admin_url(self) -> str:
        return os.environ.get("LAB3_DATABASE_ADMIN_URL", "")

    @property
    def tool_url(self) -> str:
        return f"http://{self.tools['host']}:{self.tools['port']}"

    def prompts(self) -> dict:
        return tomllib.loads((self.root / "resources/prompts.toml").read_text(encoding="utf-8"))


def load_settings(path: Path | None = None) -> Settings:
    config_path = (path or ROOT / "resources/config.toml").resolve()
    root = config_path.parent.parent
    load_dotenv(root / ".env", override=False)
    values = tomllib.loads(config_path.read_text(encoding="utf-8"))

    if os.environ.get("LAB3_MODEL"):
        values["model"]["id"] = os.environ["LAB3_MODEL"]

    if os.environ.get("LAB3_MODEL_BASE_URL"):
        values["model"]["base_url"] = os.environ["LAB3_MODEL_BASE_URL"]

    return Settings(root=root, **values)


def refresh_model(settings: Settings) -> dict:
    # Тарифы берём для выбранного API ID, чтобы смена модели не оставила цену GLM.
    response = httpx.get("https://polza.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    models = response.json().get("data", [])
    selected = next((item for item in models if item["id"] == settings.model.id), None)

    if selected is None:
        raise ValueError(f"Model {settings.model.id} is absent from the current Polza catalog")

    provider = selected["top_provider"]
    settings.model.provider = provider["name"]

    if "tools" not in provider.get("supported_parameters", []):
        raise ValueError("Selected model endpoint does not advertise native tool calling")

    prices = provider["pricing"]

    if prices.get("currency") != "RUB":
        raise ValueError("Expected RUB pricing; configure currency conversion explicitly")

    settings.model.input_price = float(prices["prompt_per_million"])
    settings.model.output_price = float(prices["completion_per_million"])
    settings.model.cache_price = float(
        prices.get("input_cache_read_per_million", prices["prompt_per_million"])
    )
    settings.model.context_window = provider.get("context_length") or settings.model.context_window
    settings.check_budget()

    return selected
