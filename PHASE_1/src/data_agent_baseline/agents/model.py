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
    """Prefer final answer `content`; never treat reasoning fields as the reply."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    # Some SDKs return multimodal list parts.
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
        joined = "".join(parts).strip()
        if joined:
            return joined
    return ""


PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "reasoning_split": False,
    },
    "minimax": {
        "api_base": "https://api.minimax.io/v1",
        # Separate thinking into reasoning_* fields so `content` stays parseable JSON.
        "reasoning_split": True,
    },
    "zhipu": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        # GLM thinking is optional; keep it off so ReAct JSON stays in `content`.
        "reasoning_split": False,
    },
}


def normalize_provider(provider: str | None) -> str:
    name = (provider or "deepseek").strip().lower()
    if name not in PROVIDER_DEFAULTS:
        supported = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ValueError(f"Unsupported agent.provider={provider!r}. Supported: {supported}")
    return name


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
    ) -> None:
        self.provider = normalize_provider(provider)
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.strip_think = strip_think
        defaults = PROVIDER_DEFAULTS[self.provider]
        if reasoning_split is None:
            self.reasoning_split = bool(defaults.get("reasoning_split", False))
        else:
            self.reasoning_split = reasoning_split
        self.extra_body = dict(extra_body or {})
        self.request_timeout = (
            float(request_timeout) if request_timeout is not None else _DEFAULT_REQUEST_TIMEOUT
        )
        self.max_retries = int(max_retries) if max_retries is not None else _DEFAULT_MAX_RETRIES

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
        return extra

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": self.temperature,
        }
        extra = self._request_kwargs()
        if extra:
            create_kwargs["extra_body"] = extra

        try:
            response = client.chat.completions.create(**create_kwargs)
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

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
