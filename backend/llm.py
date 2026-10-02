"""One tiny LLM seam: complete(system, user) -> str. Tests swap it via set_llm()."""
import os
from typing import Callable, Optional

_override: Optional[Callable[[str, str], str]] = None
_model = None


def set_llm(fn: Optional[Callable[[str, str], str]]):
    global _override
    _override = fn


def _build():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=800, timeout=30)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0, timeout=30)
    raise RuntimeError(f"Unknown LLM_PROVIDER {provider}")


def complete(system: str, user: str) -> str:
    global _model
    if _override:
        return _override(system, user)
    if _model is None:
        _model = _build()
    content = _model.invoke([("system", system), ("human", user)]).content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content
