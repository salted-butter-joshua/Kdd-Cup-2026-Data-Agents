from __future__ import annotations

from dataclasses import dataclass, field
from data_agent_baseline.agents.model import PROVIDER_DEFAULTS, normalize_provider
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


def _default_gold_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "output"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)
    gold_path: Path = field(default_factory=_default_gold_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    provider: str = "deepseek"
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 24
    temperature: float = 0.0
    strip_think: bool = True
    reasoning_split: bool | None = None


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 300
    empty_retries: int = 0


@dataclass(frozen=True, slots=True)
class EvalConfig:
    enabled: bool = True
    lambda_extra: float = 0.1


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _infer_gold_path(input_root: Path, default_gold: Path) -> Path:
    if input_root.name == "input":
        return input_root.parent / "output"
    return default_gold


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})
    eval_payload = payload.get("eval", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
        gold_path=_path_value(
            dataset_payload.get("gold_path"),
            _infer_gold_path(
                _path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
                dataset_defaults.gold_path,
            ),
        ),
    )
    provider = normalize_provider(agent_payload.get("provider", agent_defaults.provider))
    provider_defaults = PROVIDER_DEFAULTS[provider]
    raw_api_base = agent_payload.get("api_base")
    if raw_api_base is None or not str(raw_api_base).strip():
        api_base = str(provider_defaults["api_base"])
    else:
        api_base = str(raw_api_base).strip()

    reasoning_split_raw = agent_payload.get("reasoning_split", agent_defaults.reasoning_split)
    if reasoning_split_raw is None:
        reasoning_split: bool | None = None
    else:
        reasoning_split = bool(reasoning_split_raw)

    agent_config = AgentConfig(
        provider=provider,
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=api_base,
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        strip_think=bool(agent_payload.get("strip_think", agent_defaults.strip_think)),
        reasoning_split=reasoning_split,
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
        empty_retries=int(run_payload.get("empty_retries", run_defaults.empty_retries)),
    )
    eval_defaults = EvalConfig()
    eval_config = EvalConfig(
        enabled=bool(eval_payload.get("enabled", eval_defaults.enabled)),
        lambda_extra=float(eval_payload.get("lambda_extra", eval_defaults.lambda_extra)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config, eval=eval_config)
