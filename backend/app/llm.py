"""A small OpenAI-compatible chat call. `LLM_BASE_URL` and `LLM_MODEL` choose the provider, `LLM_API_KEY` the account.

Used by the safety filter's LLM check and the generator. Every failure is an `LLMError`, so callers can tell "the
model said no" from "the model could not answer". Retries are opt-in and bounded by a deadline: a live show can wait
a second or two on a 429, not longer (the bridge line covers that, docs/plans/live-loop.md T9).
"""
import os
import time
from dataclasses import dataclass
from typing import Callable

import httpx


# Overloaded or rate-limited: the same request can succeed a moment later. 29% of Exp 19's attempts were 429 or 503.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MIN_ATTEMPT_S = 0.5  # an attempt with less time than this left cannot finish, so it is not started


def strip_fence(reply: str) -> str:
    """The text of a reply that the model wrapped in a markdown code fence."""
    reply = reply.strip()
    if reply.startswith("```"):
        reply = reply.strip("`").removeprefix("json").strip()
    return reply


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable or status in RETRY_STATUS


@dataclass
class Client:
    base_url: str
    api_key: str
    model: str
    timeout: float = 8.0
    transport: httpx.BaseTransport | None = None  # tests pass a fake transport; production leaves it None
    retries: int = 0
    backoff_s: float = 0.5
    deadline_s: float | None = None  # wall time across all attempts, each attempt's timeout included
    sleep: Callable[[float], None] = time.sleep
    # Quota bookkeeping: a 429 still counts against the provider's limit, so every request is an attempt.
    attempts: int = 0
    retried: int = 0
    failed: int = 0

    def chat(self, system: str, user: str, *, json_reply: bool = False, temperature: float = 0.0) -> str:
        start = time.monotonic()
        for attempt in range(self.retries + 1):
            self.attempts += 1
            left = None if self.deadline_s is None else self.deadline_s - (time.monotonic() - start)
            try:
                return self._once(system, user, json_reply, temperature,
                                  self.timeout if left is None else max(min(self.timeout, left), 0.1))
            except LLMError as e:
                wait = self.backoff_s * 2 ** attempt
                late = (self.deadline_s is not None
                        and time.monotonic() - start + wait + MIN_ATTEMPT_S > self.deadline_s)
                if not e.retryable or attempt == self.retries or late:
                    self.failed += 1
                    raise
                self.retried += 1
                self.sleep(wait)

    def _once(self, system: str, user: str, json_reply: bool, temperature: float, timeout: float) -> str:
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_reply:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(timeout=timeout, transport=self.transport) as http:
                resp = http.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"http {e.response.status_code}", e.response.status_code) from e
        except httpx.HTTPError as e:
            raise LLMError(f"transport: {type(e).__name__}", retryable=True) from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError("malformed response") from e
        # A provider safety block arrives as a normal 200 with no usable text.
        if choice.get("finish_reason") == "content_filter":
            raise LLMError("blocked by provider")
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty reply")
        return text


def from_env(timeout: float = 8.0, **options) -> Client:
    """Raises KeyError when LLM_BASE_URL or LLM_MODEL is unset; the key may be empty (local Ollama)."""
    return Client(os.environ["LLM_BASE_URL"], os.environ.get("LLM_API_KEY", ""), os.environ["LLM_MODEL"], timeout,
                  **options)
