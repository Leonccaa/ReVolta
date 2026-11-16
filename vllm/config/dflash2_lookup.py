# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 lookup input provenance and versioned, immutable worker decisions."""

import os
from collections.abc import Mapping
from dataclasses import fields
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    StrictBool,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_core import ArgsKwargs

from vllm import envs
from vllm.config.utils import config, is_init_field, normalize_value
from vllm.logger import init_logger

logger = init_logger(__name__)
LOOKUP_ENV_PREFIX = "VLLM_DFLASH2_LOOKUP_"
SYSTEM_FIELDS = frozenset(
    {
        "target_model_config",
        "target_parallel_config",
        "draft_model_config",
        "draft_parallel_config",
    }
)


def _supported_policy(value: Any) -> Any:
    if value is not None and value != "longest":
        raise ValueError(
            f"dflash2_lookup policy {value!r} is not provided in this release; "
            "use 'longest'"
        )
    return value


@config
class DFlash2LookupExpandConfig:
    """Reserved expansion sentinel for the first native-width implementation."""

    enabled: StrictBool = False
    """Expansion is reserved and must remain false, including when lookup is off."""

    @model_validator(mode="wrap")
    @classmethod
    def _capture_explicit_keys(
        cls, data: Any, handler: ModelWrapValidatorHandler
    ) -> "DFlash2LookupExpandConfig":
        if isinstance(data, cls):
            return handler(data)
        raw = raw_speculative_input(cls, data)
        result = handler(data)
        result.__dict__["_explicit_keys"] = tuple(sorted(raw))
        return result

    def __post_init__(self) -> None:
        if self.enabled:
            raise ValueError("dflash2_lookup.expand.enabled=true is not supported")


@config
class DFlash2LookupConfig:
    """Native-width GPU lookup policy. Enabling requires explicit policy and bounds."""

    enabled: StrictBool = False
    """Enable native-width GPU head fusion."""
    policy: Literal["longest"] | None = None
    """Search policy, required when enabled until release defaults are selected."""
    nmin: StrictInt | None = Field(default=None, ge=1)
    """Minimum suffix length, required when enabled."""
    nmax: StrictInt | None = Field(default=None, ge=1)
    """Maximum suffix length, required when enabled."""
    search_window: StrictInt = Field(default=65536, ge=1)
    """Maximum distance searched in committed token history."""
    gate: Literal["first_token"] = "first_token"
    """Require the first lookup token to match the neural draft."""
    collect_stats: StrictBool = True
    """Collect lookup counters when the engine's existing statistics switch permits."""
    expand: DFlash2LookupExpandConfig = Field(default_factory=DFlash2LookupExpandConfig)
    """Reserved expansion configuration, supporting only enabled=false."""
    max_requests: StrictInt = Field(default=3, ge=1, le=3)
    """Maximum live decode requests per lookup batch.
    Set to 1 for single-request admission."""

    @field_validator("policy", mode="before")
    @classmethod
    def _validate_policy(cls, value: Any) -> Any:
        return _supported_policy(value)

    @model_validator(mode="wrap")
    @classmethod
    def _capture_explicit_keys(
        cls, data: Any, handler: ModelWrapValidatorHandler
    ) -> "DFlash2LookupConfig":
        if isinstance(data, cls):
            return handler(data)
        raw = raw_speculative_input(cls, data)
        result = handler(data)
        result.__dict__["_explicit_keys"] = tuple(sorted(raw))
        return result

    def __post_init__(self) -> None:
        if self.nmin is not None and self.nmax is not None and self.nmin > self.nmax:
            raise ValueError("dflash2_lookup requires nmin <= nmax")
        if self.enabled and None in (self.policy, self.nmin, self.nmax):
            raise ValueError(
                "enabled dflash2_lookup requires explicit policy, nmin and nmax"
            )


class LookupSource(BaseModel):
    """Captured before dataclass defaults, prompt normalization and engine injection."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always"
    )
    root: Literal["absent", "null", "object"]
    keys: tuple[str, ...]
    expand_keys: tuple[str, ...]
    ngram_assist: bool | None
    prompt_fields: tuple[Literal["prompt_lookup_min", "prompt_lookup_max"], ...]
    environment: tuple[tuple[str, str], ...]
    system_fields: tuple[str, ...] = ()


class ResolvedDFlash2Lookup(BaseModel):
    """Trusted transport record. Worker environments never reinterpret this record."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always"
    )
    schema_version: Literal[1] = 1
    admission_version: Literal[1, 2] = 1
    max_requests: StrictInt = Field(default=1, ge=1, le=3)
    mode: Literal["off", "gpu_head", "legacy_host", "legacy_gpu"]
    policy: Literal["longest"] | None = None
    nmin: StrictInt | None = None
    nmax: StrictInt | None = None
    search_window: StrictInt | None = None
    gate: Literal["first_token"] = "first_token"
    collect_stats: StrictBool = False
    draft_tokens: StrictInt | None
    verify_tokens: StrictInt
    # Values already normalized exactly as envs.compile_factors does.
    env_factors: tuple[tuple[str, StrictBool | StrictInt | str], ...]
    source: LookupSource

    @field_validator("policy", mode="before")
    @classmethod
    def _validate_policy(cls, value: Any) -> Any:
        return _supported_policy(value)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_version(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported resolved lookup schema version")
        return value

    @field_validator("admission_version", mode="before")
    @classmethod
    def _validate_admission_version(cls, value: Any) -> Any:
        if type(value) is not int or value not in (1, 2):
            raise ValueError("Unsupported resolved lookup admission version")
        return value

    @model_validator(mode="after")
    def _validate_admission(self) -> "ResolvedDFlash2Lookup":
        expected = 2 if self.max_requests > 1 else 1
        if self.admission_version != expected:
            raise ValueError("Inconsistent lookup admission version and max_requests")
        return self

    @model_validator(mode="after")
    def _validate_factors(self) -> "ResolvedDFlash2Lookup":
        factors = dict(self.env_factors)
        required = {
            name
            for name in envs.environment_variables
            if name.startswith(LOOKUP_ENV_PREFIX)
        }
        if (
            len(factors) != len(self.env_factors)
            or not required <= factors.keys()
            or any(not name.startswith(LOOKUP_ENV_PREFIX) for name in factors)
        ):
            raise ValueError("Invalid resolved lookup environment factors")
        return self

    def graph_options(self) -> tuple[Any, ...]:
        """Versioned forward identity, retaining the single-request cache key."""
        return (
            self.schema_version,
            self.mode,
            self.policy,
            self.nmin,
            self.nmax,
            self.search_window,
            self.gate,
            self.draft_tokens,
            self.verify_tokens,
            *((self.max_requests,) if self.max_requests > 1 else ()),
            self.admission_version,
        )


def raw_speculative_input(cls: type, data: Any) -> dict[str, Any]:
    if isinstance(data, ArgsKwargs):
        names = [field.name for field in fields(cls) if is_init_field(cls, field.name)]
        return dict(zip(names, data.args)) | dict(data.kwargs or {})
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{cls.__name__} input must be a mapping or constructor arguments"
        )
    return dict(data)


def capture_lookup_source(
    raw: dict[str, Any],
    *,
    system_fields: tuple[str, ...] = (),
    environment: tuple[tuple[str, str], ...] | None = None,
) -> LookupSource:
    lookup = raw.get("dflash2_lookup")
    expand = (
        lookup.expand
        if isinstance(lookup, DFlash2LookupConfig)
        else (lookup.get("expand") if isinstance(lookup, dict) else None)
    )
    return LookupSource(
        root="absent"
        if "dflash2_lookup" not in raw
        else "null"
        if lookup is None
        else "object",
        keys=_explicit_keys(lookup),
        expand_keys=_explicit_keys(expand),
        ngram_assist=raw.get("ngram_assist"),
        prompt_fields=tuple(
            name
            for name in ("prompt_lookup_min", "prompt_lookup_max")
            if raw.get(name) is not None
        ),
        environment=environment
        if environment is not None
        else tuple(
            sorted(
                (name, value)
                for name, value in os.environ.items()
                if name.startswith(LOOKUP_ENV_PREFIX)
            )
        ),
        system_fields=system_fields,
    )


def validate_lookup_input(value: Any) -> DFlash2LookupConfig | None:
    """Validate mutable nested instances through the same schema as raw input."""
    original = value
    if isinstance(value, DFlash2LookupConfig):
        value = TypeAdapter(DFlash2LookupConfig).dump_python(value, warnings=False)
    validated = TypeAdapter(DFlash2LookupConfig | None).validate_python(value)
    if isinstance(original, DFlash2LookupConfig) and validated is not None:
        validated.__dict__["_explicit_keys"] = original.__dict__.get(
            "_explicit_keys", ()
        )
        if isinstance(original.expand, DFlash2LookupExpandConfig):
            validated.expand.__dict__["_explicit_keys"] = original.expand.__dict__.get(
                "_explicit_keys", ()
            )
    return validated


def _explicit_keys(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(sorted(value))
    if isinstance(value, (DFlash2LookupConfig, DFlash2LookupExpandConfig)):
        return value.__dict__["_explicit_keys"]
    return ()


def lookup_env_factors() -> tuple[tuple[str, Any], ...]:
    factors = {}
    for name, getter in envs.environment_variables.items():
        if name.startswith(LOOKUP_ENV_PREFIX):
            try:
                factors[name] = normalize_value(getter())
            except Exception:
                # Match compile_factors' treatment of malformed legacy envs.
                continue
    for name, value in os.environ.items():
        if name.startswith(LOOKUP_ENV_PREFIX) and name not in factors:
            factors[name] = normalize_value(value)
    return tuple(sorted(factors.items()))


def validate_lookup_conflicts(
    lookup: DFlash2LookupConfig | None, source: LookupSource
) -> None:
    if lookup is None or not lookup.enabled:
        return
    if source.ngram_assist or source.prompt_fields or source.environment:
        raise ValueError(
            "dflash2_lookup.enabled conflicts with explicit legacy sources "
            f"ngram_assist={source.ngram_assist}, prompt={source.prompt_fields}, "
            f"env={tuple(name for name, _ in source.environment)}"
        )


def resolve_lookup(
    spec: Any, source: LookupSource, env_factors: tuple[tuple[str, Any], ...]
) -> ResolvedDFlash2Lookup:
    from vllm.config.speculative import _get_dflash2_checkpoint_draft_tokens

    lookup = spec.dflash2_lookup
    hf_config = getattr(spec.draft_model_config, "hf_config", None)
    draft_tokens = (
        _get_dflash2_checkpoint_draft_tokens(hf_config)
        if spec.method == "dflash"
        else None
    )
    mode = "off"
    if spec.ngram_assist:
        mode = (
            "legacy_gpu"
            if draft_tokens is not None and spec.num_speculative_tokens > draft_tokens
            else "legacy_host"
        )
    if lookup is not None:
        max_model_len = getattr(spec.target_model_config, "max_model_len", None)
        if max_model_len is not None and any(
            bound is not None and bound > max_model_len
            for bound in (lookup.nmin, lookup.nmax)
        ):
            raise ValueError("dflash2_lookup nmin/nmax must be <= max_model_len")
        if lookup.enabled:
            from vllm.platforms import current_platform

            if spec.method != "dflash" or draft_tokens is None:
                raise ValueError(
                    "dflash2_lookup requires method='dflash' "
                    "and DFlash2 selector capability"
                )
            if spec.num_speculative_tokens != 7 or draft_tokens != 7:
                raise ValueError(
                    "dflash2_lookup requires native K=D=7; expansion is not supported"
                )
            if not current_platform.is_cuda():
                raise ValueError("dflash2_lookup requires CUDA")
            mode = "gpu_head"
    if (
        mode != "gpu_head"
        and source.root != "absent"
        and (source.ngram_assist or source.prompt_fields or source.environment)
    ):
        logger.warning(
            "dflash2_lookup source=%s keys=%s is disabled. Legacy inputs "
            "select effective mode=%s",
            source.root,
            source.keys,
            mode,
        )
    nmin = spec.prompt_lookup_min if spec.ngram_assist else None
    nmax = spec.prompt_lookup_max if spec.ngram_assist else None
    if mode == "gpu_head":
        assert lookup is not None
        nmin, nmax = lookup.nmin, lookup.nmax
    max_requests = lookup.max_requests if mode == "gpu_head" and lookup else 1
    return ResolvedDFlash2Lookup(
        mode=mode,
        max_requests=max_requests,
        admission_version=2 if max_requests > 1 else 1,
        policy=lookup.policy if mode == "gpu_head" else None,
        nmin=nmin,
        nmax=nmax,
        search_window=lookup.search_window if mode == "gpu_head" else None,
        collect_stats=bool(
            mode == "gpu_head" and lookup is not None and lookup.collect_stats
        ),
        draft_tokens=draft_tokens,
        verify_tokens=spec.num_speculative_tokens,
        env_factors=env_factors,
        source=source,
    )
