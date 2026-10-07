from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI

# Hard per-request bounds. The SDK defaults (10 min timeout, 2 retries with
# backoff) let one throttled call eat the whole per-task time budget.
_DEFAULT_REQUEST_TIMEOUT = max(
    1.0, float(os.environ.get("DATA_AGENT_LLM_TIMEOUT", "60") or "60")
)
_DEFAULT_MAX_RETRIES = max(0, int(os.environ.get("DATA_AGENT_LLM_MAX_RETRIES", "1") or "1"))


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


# Thinking / chain-of-thought wrappers some providers embed in `content`.
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>"
    r"|<thinking\b[^>]*>.*?</thinking\s*>"
    r"|<reason(?:ing)?\b[^>]*>.*?</reason(?:ing)?\s*>"
    r"|◁think▷.*?◁/think▷",
    flags=re.IGNORECASE | re.DOTALL,
)
# Unclosed leading think block (model cut off before close tag).
_THINK_UNCLOSED_RE = re.compile(
    r"^(?:<think\b[^>]*>|<thinking\b[^>]*>|<reason(?:ing)?\b[^>]*>|◁think▷).*?"
    r"(?=```|(?:\{\s*\"thought\")|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_think_content(text: str) -> str:
    """Remove provider thinking wrappers; keep the final answer body (e.g. JSON)."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()


def _message_text_content(message: Any) -> str:
    """Prefer final answer `content`; fall back to reasoning fields if content is empty.

    vLLM Qwen thinking mode often sets ``content=null`` and puts tokens in
    ``reasoning`` / ``reasoning_content``. Callers still strip think wrappers.
    """
    content = getattr(message, "content", None)
    text = _coerce_text(content)
    if text:
        return text
    for attr in ("reasoning_content", "reasoning"):
        text = _coerce_text(getattr(message, attr, None))
        if text:
            return text
    return ""


def _coerce_text(content: Any) -> str:
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in {None, "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "reasoning_split": False,
        "allow_empty_api_key": False,
    },
    "minimax": {
        "api_base": "https://api.minimax.io/v1",
        # Separate thinking into reasoning_* fields so `content` stays parseable JSON.
        "reasoning_split": True,
        "allow_empty_api_key": False,
    },
    "zhipu": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        # GLM thinking is optional; keep it off so ReAct JSON stays in `content`.
        "reasoning_split": False,
        "allow_empty_api_key": False,
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "reasoning_split": False,
        "allow_empty_api_key": False,
    },
    # vLLM / SGLang / Ollama-compatible / any local OpenAI /v1 server.
    "local": {
        "api_base": "http://localhost:8090/v1",
        "reasoning_split": False,
        "allow_empty_api_key": True,
        "disable_thinking": True,
        "max_tokens": 2048,
    },
}


def normalize_provider(provider: str | None) -> str:
    name = (provider or "deepseek").strip().lower()
    if name in {"qwen", "vllm", "sglang", "ollama"}:
        name = "local"
    if name not in PROVIDER_DEFAULTS:
        supported = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ValueError(f"Unsupported agent.provider={provider!r}. Supported: {supported}")
    return name


def normalize_api_base(api_base: str) -> str:
    """Root URL for the OpenAI SDK (…/v1), not the /chat/completions path."""
    base = (api_base or "").strip().rstrip("/")
    low = base.casefold()
    if low.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    return base


def _is_loopback_base(api_base: str) -> bool:
    low = (api_base or "").casefold()
    return any(token in low for token in ("localhost", "127.0.0.1", "[::1]", "0.0.0.0"))


def resolve_api_key(*, api_key: str, provider: str, api_base: str) -> str:
    key = (api_key or "").strip()
    if key:
        return key
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    if bool(defaults.get("allow_empty_api_key")) or _is_loopback_base(api_base):
        # openai.OpenAI requires a non-empty string even when the server ignores it.
        return "EMPTY"
    return ""


class OpenAIModelAdapter:
    """OpenAI-compatible chat adapter with provider-specific think handling."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        provider: str = "deepseek",
        strip_think: bool = True,
        reasoning_split: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = normalize_provider(provider)
        self.model = model
        self.api_base = normalize_api_base(api_base)
        self.api_key = resolve_api_key(
            api_key=api_key, provider=self.provider, api_base=self.api_base
        )
        self.temperature = temperature
        self.strip_think = strip_think
        defaults = PROVIDER_DEFAULTS[self.provider]
        if reasoning_split is None:
            self.reasoning_split = bool(defaults.get("reasoning_split", False))
        else:
            self.reasoning_split = reasoning_split
        self.extra_body = dict(extra_body or {})
        if request_timeout is not None:
            self.request_timeout = float(request_timeout)
        elif self.provider == "local":
            self.request_timeout = max(_DEFAULT_REQUEST_TIMEOUT, 180.0)
        else:
            self.request_timeout = _DEFAULT_REQUEST_TIMEOUT
        self.max_retries = int(max_retries) if max_retries is not None else _DEFAULT_MAX_RETRIES
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        else:
            raw_max = defaults.get("max_tokens")
            self.max_tokens = int(raw_max) if raw_max else None

    def _request_kwargs(self) -> dict[str, Any]:
        extra = dict(self.extra_body)
        if self.provider == "minimax" and self.reasoning_split:
            # MiniMax: keep thinking out of `content` for ReAct JSON parsing.
            extra.setdefault("reasoning_split", True)
        if self.provider == "zhipu":
            # Zhipu GLM: thinking off by default. When reasoning_split is true,
            # enable it; the trace lands in `reasoning_content`, not `content`.
            thinking_type = "enabled" if self.reasoning_split else "disabled"
            extra.setdefault("thinking", {"type": thinking_type})
        if self.provider == "local" and PROVIDER_DEFAULTS["local"].get("disable_thinking"):
            extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return extra

    def _openai_client(self) -> OpenAI:
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }
        if self.provider == "local":
            # This vLLM/uvicorn front-end returns empty HTTP 502 for default
            # httpx keepalive / IPv6 sockets. Force IPv4 + Connection: close.
            import httpx

            kwargs["http_client"] = httpx.Client(
                timeout=self.request_timeout,
                trust_env=False,
                limits=httpx.Limits(max_keepalive_connections=0, max_connections=8),
                headers={"Connection": "close", "Accept-Encoding": "identity"},
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            )
        return OpenAI(**kwargs)

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = self._openai_client()

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            create_kwargs["max_tokens"] = self.max_tokens
        extra = self._request_kwargs()
        if extra:
            create_kwargs["extra_body"] = extra

        try:
            from data_agent_baseline.run.progress import mark as _mark

            _mark(
                "llm_request_start",
                provider=self.provider,
                model=self.model,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
                message_count=len(messages),
            )
            response = client.chat.completions.create(**create_kwargs)
            _mark("llm_request_done", provider=self.provider, model=self.model)
        except APIError as exc:
            from data_agent_baseline.run.progress import mark as _mark

            _mark("llm_request_failed", provider=self.provider, error=str(exc)[:300])
            raise RuntimeError(f"Model request failed: {exc}") from exc
        except Exception as exc:
            from data_agent_baseline.run.progress import mark as _mark

            _mark(
                "llm_request_failed",
                provider=self.provider,
                error_type=type(exc).__name__,
                error=str(exc)[:300] or type(exc).__name__,
            )
            raise
        finally:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = _message_text_content(choices[0].message)
        if not content:
            raise RuntimeError(
                "Model response missing text content "
                "(thinking-only replies are not usable for ReAct JSON)."
            )
        if self.strip_think:
            content = strip_think_content(content)
            if not content:
                raise RuntimeError(
                    "Model response was empty after stripping think/reasoning wrappers."
                )
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
