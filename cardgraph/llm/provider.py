"""Model access: providers, structured output, caching, cost accounting.

Three providers, chosen by `CARDGRAPH_LLM` or auto-detected:

  anthropic   The SDK, with ANTHROPIC_API_KEY. Structured output is done with a
              forced tool call rather than "please return JSON", because a
              forced tool call is validated by the API against your schema and
              prose-wrapped JSON is not.
  claude-cli  Shells out to the `claude` binary. Reuses an existing Claude Code
              login, so the repo runs for anyone who already has the CLI
              without provisioning a separate key. Slower per call and has no
              tool-forcing, so JSON gets extracted and repaired.
  stub        Replays recorded responses from a cassette directory. This is what
              makes the analysis tests runnable offline and deterministic in CI.

Everything above the provider talks to `LLM`, which adds the parts you always
end up needing and always regret bolting on later: a content-addressed disk
cache, bounded retries with backoff, JSON repair, schema validation, and a
ledger so `analyze` can print what a run cost instead of surprising you.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

DEFAULT_MODEL = os.environ.get("CARDGRAPH_MODEL", "claude-sonnet-4-6")

# Published per-million-token prices are a moving target and belong in config,
# not in a constant that silently goes stale. The ledger reports token counts
# unconditionally and dollars only when a price is configured.
def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write cache data without leaving a parseable-looking partial file."""
    directory = os.path.dirname(path) or "."
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".card-scraper-cache-", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


PRICES: dict[str, tuple[float, float]] = {}
if os.environ.get("CARDGRAPH_PRICE_IN") and os.environ.get("CARDGRAPH_PRICE_OUT"):
    PRICES[DEFAULT_MODEL] = (float(os.environ["CARDGRAPH_PRICE_IN"]),
                             float(os.environ["CARDGRAPH_PRICE_OUT"]))


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """No provider is configured. Callers degrade to deterministic analysis
    rather than failing the run."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    calls: int = 0
    cache_hits: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    def summary(self) -> str:
        s = (f"{self.calls} call(s), {self.cache_hits} cached, "
             f"{self.input_tokens:,} in / {self.output_tokens:,} out")
        if self.cost_usd:
            s += f", ${self.cost_usd:.4f}"
        return s


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    parsed: Any = None
    from_cache: bool = False
    provider: str = ""


class Provider(ABC):
    name = "base"

    @abstractmethod
    def complete(self, system: str, user: str, *, schema: dict | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        ...

    @property
    def available(self) -> bool:
        return True


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None

    @property
    def available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMUnavailable("pip install anthropic") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, *, schema: dict | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        client = self._client_or_raise()
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        if schema is not None:
            # Forced tool call: the API validates the shape for us, which is
            # strictly better than asking for JSON and hoping.
            kwargs["tools"] = [{
                "name": "emit",
                "description": "Emit the structured result.",
                "input_schema": schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit"}

        msg = client.messages.create(**kwargs)
        u = Usage(
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
            cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            calls=1,
        )
        price = PRICES.get(self.model)
        if price:
            u.cost_usd = (u.input_tokens * price[0] + u.output_tokens * price[1]) / 1e6

        parsed, text = None, ""
        for block in msg.content:
            if getattr(block, "type", "") == "tool_use":
                parsed = block.input
                text = json.dumps(parsed)
            elif getattr(block, "type", "") == "text":
                text += block.text
        return LLMResponse(text=text, usage=u, parsed=parsed, provider=self.name)


class ClaudeCLIProvider(Provider):
    """Shell out to the `claude` binary in headless mode.

    No tool-forcing available here, so structured output is requested in the
    prompt and repaired on the way back. Slower than the SDK; the point is that
    it needs no API key, which lowers the barrier for anyone who already has
    Claude Code installed.
    """

    name = "claude-cli"

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 300,
                 binary: str = "claude"):
        self.model = model
        self.timeout = timeout
        self.binary = binary

    @property
    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def complete(self, system: str, user: str, *, schema: dict | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        if not self.available:
            raise LLMUnavailable(f"{self.binary!r} not found on PATH")
        prompt = user
        if schema is not None:
            prompt = (f"{user}\n\n---\nReturn ONLY a JSON object matching this "
                      f"JSON Schema. No prose, no markdown fences, no "
                      f"explanation.\n{json.dumps(schema)}")
        cmd = [self.binary, "-p", prompt, "--output-format", "json"]
        if system:
            cmd += ["--append-system-prompt", system]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude CLI timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise LLMError(f"claude CLI failed ({proc.returncode}): "
                           f"{proc.stderr[-400:]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON: "
                           f"{proc.stdout[:300]!r}") from exc
        if payload.get("is_error"):
            raise LLMError(f"claude CLI error: {payload.get('result')!r}")

        text = payload.get("result") or ""
        raw = payload.get("usage") or {}
        u = Usage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cache_read_tokens=raw.get("cache_read_input_tokens", 0) or 0,
            cost_usd=payload.get("total_cost_usd"),
            calls=1,
        )
        return LLMResponse(text=text, usage=u, provider=self.name)


class StubProvider(Provider):
    """Replay recorded responses. Keyed by a hash of (system, user, schema), so
    a prompt edit misses the cassette loudly instead of silently replaying a
    stale answer for a question you no longer ask."""

    name = "stub"

    def __init__(self, cassette_dir: str, record_with: Provider | None = None):
        self.dir = cassette_dir
        self.record_with = record_with
        os.makedirs(cassette_dir, exist_ok=True)

    @staticmethod
    def key(system: str, user: str, schema: dict | None) -> str:
        basis = json.dumps([system, user, schema], sort_keys=True)
        return hashlib.sha256(basis.encode()).hexdigest()[:24]

    def complete(self, system: str, user: str, *, schema: dict | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0) -> LLMResponse:
        k = self.key(system, user, schema)
        path = os.path.join(self.dir, f"{k}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                return LLMResponse(text=d["text"], usage=Usage(**d.get("usage", {})),
                                   parsed=d.get("parsed"), from_cache=True,
                                   provider=self.name)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A killed recording process can leave a truncated cassette.
                # Treat it as a cache miss so a configured live provider can
                # repair it instead of failing every future analysis.
                try:
                    os.remove(path)
                except OSError:
                    pass
        if self.record_with is None:
            raise LLMUnavailable(
                f"no cassette for key {k}. Re-record with "
                f"CARDGRAPH_RECORD=1 and a live provider.")
        resp = self.record_with.complete(system, user, schema=schema,
                                         max_tokens=max_tokens,
                                         temperature=temperature)
        _write_json_atomic(path, {
            "text": resp.text, "parsed": resp.parsed,
            "usage": asdict(resp.usage),
            "_prompt_preview": user[:400],
        })
        return resp


# --------------------------------------------------------------------------
# JSON extraction and repair
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model output that may be wrapped in prose or
    fences. Tries, in order: the whole string, fenced blocks, and the widest
    balanced brace/bracket span."""
    text = (text or "").strip()
    if not text:
        raise LLMError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in _FENCE.finditer(text):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # trailing-comma repair, the single most common malformation
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
    raise LLMError(f"no JSON found in response: {text[:200]!r}")


def validate(value: Any, schema: dict) -> list[str]:
    """Minimal structural check against the subset of JSON Schema we emit.

    Deliberately not a full validator: the failure we care about is a model
    returning the wrong *shape* (a list where an object belongs, a missing
    required key), and a 60-line checker catches that without adding a
    dependency the rest of the project does not need.
    """
    errs: list[str] = []

    def walk(val: Any, sch: dict, path: str) -> None:
        t = sch.get("type")
        if t == "object":
            if not isinstance(val, dict):
                errs.append(f"{path}: expected object, got {type(val).__name__}")
                return
            for req in sch.get("required", []):
                if req not in val:
                    errs.append(f"{path}.{req}: required field missing")
            for key, sub in (sch.get("properties") or {}).items():
                if key in val:
                    walk(val[key], sub, f"{path}.{key}")
        elif t == "array":
            if not isinstance(val, list):
                errs.append(f"{path}: expected array, got {type(val).__name__}")
                return
            item = sch.get("items")
            if isinstance(item, dict):
                for i, v in enumerate(val):
                    walk(v, item, f"{path}[{i}]")
        elif t == "string":
            if not isinstance(val, str):
                errs.append(f"{path}: expected string")
            elif sch.get("enum") and val not in sch["enum"]:
                errs.append(f"{path}: {val!r} not in {sch['enum']}")
        elif t == "number":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errs.append(f"{path}: expected number")
        elif t == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                errs.append(f"{path}: expected integer")
        elif t == "boolean":
            if not isinstance(val, bool):
                errs.append(f"{path}: expected boolean")

    walk(value, schema, "$")
    return errs


# --------------------------------------------------------------------------
# The facade
# --------------------------------------------------------------------------

_ITEM_PATH = re.compile(r"^\$\.([A-Za-z_][\w]*)\[(\d+)\]")


def _salvage(value: Any, schema: dict, key: str,
             errs: list[str]) -> tuple[Any, list[str]]:
    """Drop invalid elements of `value[key]` and re-validate.

    Returns (possibly-trimmed value, remaining errors). If any error lies
    outside that array the response is structurally wrong rather than partially
    wrong, and nothing is salvaged.
    """
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        return value, errs

    bad: set[int] = set()
    for e in errs:
        m = _ITEM_PATH.match(e)
        if not m or m.group(1) != key:
            return value, errs      # error outside the salvageable array
        bad.add(int(m.group(2)))

    trimmed = dict(value)
    trimmed[key] = [v for i, v in enumerate(value[key]) if i not in bad]
    return trimmed, validate(trimmed, schema)


def autodetect(model: str = DEFAULT_MODEL) -> Provider | None:
    choice = os.environ.get("CARDGRAPH_LLM", "").strip().lower()
    candidates: list[Provider]
    if choice == "anthropic":
        candidates = [AnthropicProvider(model)]
    elif choice in ("claude-cli", "cli"):
        candidates = [ClaudeCLIProvider(model)]
    elif choice in ("none", "off"):
        return None
    else:
        candidates = [AnthropicProvider(model), ClaudeCLIProvider(model)]
    for p in candidates:
        if p.available:
            return p
    return None


class LLM:
    """Cache + retry + validation wrapper. This is what analyzers use."""

    def __init__(self, provider: Provider | None = None,
                 cache_dir: str = "data/llm_cache", max_retries: int = 3,
                 model: str = DEFAULT_MODEL):
        self.provider = provider if provider is not None else autodetect(model)
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.ledger = Usage()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        return self.provider.name if self.provider else "none"

    def _cache_path(self, system: str, user: str, schema: dict | None) -> str:
        basis = json.dumps([self.provider_name, system, user, schema],
                           sort_keys=True)
        h = hashlib.sha256(basis.encode()).hexdigest()[:32]
        return os.path.join(self.cache_dir, f"{h}.json")

    def json(self, system: str, user: str, schema: dict, *,
             max_tokens: int = 4096, temperature: float = 0.0,
             salvage_key: str | None = None) -> Any:
        """Structured call. Returns the parsed value or raises LLMError.

        On a schema violation we retry with the validation errors appended to
        the prompt. Telling the model exactly which field was wrong fixes it far
        more often than resampling at the same temperature does.

        `salvage_key` names a top-level array whose bad elements may be dropped
        instead of failing the whole response. Without it, one invalid item in a
        list of eight throws away seven good ones you have already paid for, and
        then spends another call regenerating them. With it, the bad element is
        discarded and the rest is returned. Only used where partial results are
        genuinely useful — a list of findings is; a chain with a hole in the
        middle is not.
        """
        if not self.provider:
            raise LLMUnavailable("no LLM provider configured")

        if self.cache_dir:
            cp = self._cache_path(system, user, schema)
            if os.path.exists(cp):
                try:
                    with open(cp, encoding="utf-8") as fh:
                        payload = json.load(fh)
                    parsed = payload["parsed"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # Cache corruption must never make evidence analysis
                    # impossible; discard only the bad artifact and recompute.
                    try:
                        os.remove(cp)
                    except OSError:
                        pass
                else:
                    self.ledger.cache_hits += 1
                    return parsed

        last_err: str = ""
        prompt = user
        for attempt in range(self.max_retries):
            try:
                resp = self.provider.complete(
                    system, prompt, schema=schema, max_tokens=max_tokens,
                    temperature=temperature if attempt == 0 else 0.2)
            except LLMError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
                continue

            self.ledger.add(resp.usage)
            value = resp.parsed if resp.parsed is not None else None
            if value is None:
                try:
                    value = extract_json(resp.text)
                except LLMError as exc:
                    last_err = str(exc)
                    prompt = (f"{user}\n\nYour previous reply could not be "
                              f"parsed as JSON ({last_err}). Return only the "
                              f"JSON object.")
                    continue

            errs = validate(value, schema)
            if errs and salvage_key:
                value, errs = _salvage(value, schema, salvage_key, errs)
            if not errs:
                if self.cache_dir:
                    _write_json_atomic(self._cache_path(system, user, schema), {
                        "parsed": value, "usage": asdict(resp.usage),
                    })
                return value
            last_err = "; ".join(errs[:6])
            prompt = (f"{user}\n\nYour previous reply failed validation:\n"
                      f"{last_err}\nReturn corrected JSON only.")

        raise LLMError(f"schema validation failed after {self.max_retries} "
                       f"attempts: {last_err}")

    def text(self, system: str, user: str, *, max_tokens: int = 2048,
             temperature: float = 0.0) -> str:
        if not self.provider:
            raise LLMUnavailable("no LLM provider configured")
        resp = self.provider.complete(system, user, max_tokens=max_tokens,
                                      temperature=temperature)
        self.ledger.add(resp.usage)
        return resp.text
