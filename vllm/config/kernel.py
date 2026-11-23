# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config, get_hash_factors, hash_factors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@config
class IrOpPriorityConfig:
    """
    Configuration for vLLM IR op priority for dispatching/lowering during the
    forward pass. Each member is a list of strings, which will be installed
    in worker init via vllm.ir.ops.<op_name>.set_default().
    A single comma-separated string is accepted as well,

    If specified manually, platform defaults will be appended to the lists.
    See KernelConfig.set_platform_defaults().
    """

    rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.rms_norm"""

    fused_add_rms_norm: list[str] = Field(default_factory=list)
    """Priority list for vllm.ir.ops.fused_add_rms_norm"""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.

        Also, manually add IR op impl UUIDs to make sure they affect the compile cache.
        """
        factors = get_hash_factors(self, set())

        # Implementations are hidden from Dynamo,
        # so they don't show up in the traced files list.
        from vllm.ir.op import IrOp

        assert "_impls" not in factors
        factors["_impls"] = {
            name: {
                provider: IrOp.registry[name].impls[provider].uuid() for provider in p
            }
            for name, p in asdict(self).items()  # type: ignore[call-overload]
        }

        return hash_factors(factors)

    @field_validator("*", mode="before")
    @classmethod
    def _to_list_str(cls, value: str | list[str]):
        if isinstance(value, str):
            value = value.replace(" ", "").split(",")

        assert all(isinstance(v, str) for v in value)
        return value

    def _iter_op_priorities(self):
        """
        Yield (IrOp, priority_list) for each field, after importing platform
        kernels and validating each entry.
        """
        from vllm.ir.op import IrOp
        from vllm.platforms import current_platform

        current_platform.import_ir_kernels()

        for field in fields(self):  # type: ignore[arg-type]
            op_priority = getattr(self, field.name)
            assert op_priority is not None, (
                f"IR op priority for {field.name} must be set"
            )
            logger.debug("Setting IR op priority for %s to %s", field.name, op_priority)
            yield IrOp.registry[field.name], op_priority

    def set_default(self) -> None:
        """
        Permanently set the IR op priority for all op members.
        """
        for ir_op, op_priority in self._iter_op_priorities():
            ir_op.set_default(op_priority)

    @contextlib.contextmanager
    def set_priority(self):
        """
        Context manager to set the IR op priority for all op members.
        It also imports IR kernel implementations for the current platform
        to ensure all implementations are made available.
        """
        with contextlib.ExitStack() as stack:
            for ir_op, op_priority in self._iter_op_priorities():
                stack.enter_context(ir_op.set_priority(op_priority))
            yield

    @classmethod
    def with_default(
        cls, default: list[str], /, **kwargs: list[str]
    ) -> "IrOpPriorityConfig":
        """
        A helper to create an IrOpPriorityConfig where fields not specified in kwargs
        use the given default list.
        """
        for field in fields(cls):  # type: ignore[arg-type]
            if field.name not in kwargs:
                kwargs[field.name] = list(default)

        return cls(**kwargs)


MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_b12x",
    "marlin",
    "sm70_skinny",
    "humming",
    "triton_unfused",
    "aiter",
    "emulation",
]

LinearBackend = Literal[
    "auto",
    "turbomind",
    "cutlass",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "flashinfer_cudnn",
    "marlin",
    "triton",
    "deep_gemm",
    "torch",
    "aiter",
    "machete",
    "fbgemm",
    "conch",
    "exllama",
    "emulation",
]


@config
class Sm70NvFp4Config:
    """Per-engine weight-only NVFP4 policy; None retains legacy auto selection.

    Resolve before loading layers. Native capability and local layout checks
    remain with the linear kernels. Explicit fields override deprecated envs.
    """

    dense_qpn2: bool = True
    """Allow native QPN2 with FP16 dense prefill on supported Turing workers."""
    qpn2: bool | None = None
    """Enable QPN2 small-M kernels; auto follows the qualified draft workload."""
    prefill: bool | None = None
    """Enable the existing bounded QPN2 prefill dispatcher."""
    shared_weight: bool | None = None
    """Share packed codes with TurboMind instead of retaining a second copy."""
    shared_scales: bool | None = None
    """Allow compact scales when batch layouts and native ABI permit them."""
    prefill_min_m: int | None = None
    """User override for the retained prefill threshold, normally 1024 rows."""
    qualified: bool = Field(default=False, init=False)
    """Whether the draft/state contract has passed the retained quality gate."""
    resolved: bool = Field(default=False, init=False)
    """Prevent reparsing process environment when a config is reused."""

    def resolve(self, *, qualified: bool) -> None:
        from vllm import envs

        if self.resolved:
            return
        self.qualified = qualified
        defaults = {
            "qpn2": ("VLLM_SM70_NVFP4_QPN2", qualified),
            "prefill": ("VLLM_SM70_NVFP4_QPN2_PREFILL", qualified),
            "shared_weight": ("VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT", True),
            "shared_scales": ("VLLM_SM70_NVFP4_QPN2_SHARED_SCALES", True),
            "prefill_min_m": ("VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M", 256),
        }
        for field, (name, default) in defaults.items():
            if envs.is_set(name):
                logger.warning_once(
                    "%s is deprecated; use kernel_config.sm70_nvfp4.%s. "
                    "Explicit configuration takes precedence.",
                    name,
                    field,
                )
            if getattr(self, field) is None:
                setattr(
                    self, field, getattr(envs, name) if envs.is_set(name) else default
                )
        self.resolved = True


@config
class Sm70AwqConfig:
    """Per-engine AWQ policy; native support is checked by the linear kernel."""

    enabled: bool | None = None
    """Use TurboMind AWQ; auto preserves the legacy backend preference."""
    prefill_exact_dense: bool | None = None
    """Use bounded exact-dense prefill for quality-qualified projections."""
    fused_silu: bool | None = None
    """Experimental fused gate/up epilogue; auto remains disabled."""
    resolved: bool = Field(default=False, init=False)
    """Whether the compatibility adapter has resolved this engine's policy."""

    def resolve(self) -> None:
        from vllm import envs

        if self.resolved:
            return
        for field, name in {
            "enabled": "VLLM_SM70_AWQ_TURBOMIND",
            "prefill_exact_dense": "VLLM_SM70_AWQ_PREFILL_EXACT_DENSE",
            "fused_silu": "VLLM_SM70_AWQ_MLP_ENGINE",
        }.items():
            if envs.is_set(name):
                logger.warning_once(
                    "%s is deprecated for dense linear layers; use "
                    "kernel_config.sm70_awq.%s. Explicit configuration wins.",
                    name,
                    field,
                )
            if getattr(self, field) is None:
                value = getattr(envs, name)
                if field == "enabled":
                    value = envs.use_sm70_turbomind(value)
                setattr(self, field, value)
        self.resolved = True


@config
class Sm70Fp8Config:
    """Per-engine serialized block-FP8 variants; native tuning stays unchanged.

    None preserves the legacy default and override precedence. Resolution is
    idempotent and never writes process environment. Shared aliases still serve
    unmigrated online and compressed-tensors FP8 loaders.
    """

    enabled: bool | None = None
    """Use TurboMind; auto retains the shared legacy backend preference."""
    block_qpn8: bool = True
    """Use native weight-only block QPN8 when its kernel capabilities match."""
    dequant_fallback: bool | None = None
    """Keep the legacy dense dequantization route available when requested."""
    qpn8: bool | None = None
    """Use the retained projection-qualified QPN8 weight layout."""
    qpn8_pp2_tp4: bool | None = None
    """Use the measured serialized PP2/TP4 QPN8 pipeline variant."""
    qpn8_shared_gate: bool | None = None
    """Enable the measured non-fused shared-expert QPN8 variant."""
    prescaled_decode: bool | None = None
    """Use reversible UE8M0 scale shifts in the qualified M1 decode lane."""
    prescaled_shared_gate: bool | None = None
    """Allow the qualified shared-expert M1 prescaled variant."""
    legacy_prefill_fast_selector: bool = Field(default=True, init=False)
    """Mirror native tuning selection until its host API accepts configuration."""
    prefill_prescaled: bool | None = None
    """Prepare the retained exact-8K pre-scaled projection variant."""
    prefill_exact_dense: bool | None = None
    """Use the bounded exact-dense prefill workspace."""
    prefill_visible_dense_mm: bool | None = None
    """Diagnostic visible dense MM for existing AsyncTP experiments."""
    legacy_grouped_bmm_decode: bool = Field(default=True, init=False)
    """Mirror the native shared flag; its host API does not accept config yet."""
    gated_silu: bool | None = None
    """Prepare the existing fused gate/up epilogue."""
    explicit_enables: tuple[str, ...] = Field(default=(), init=False)
    """Retain the legacy error for an explicit route with missing native ops."""
    resolved: bool = Field(default=False, init=False)
    """Whether compatibility values have been frozen for this engine."""
    force_marlin: bool = Field(default=False, init=False)
    """Retained legacy backend rollback, resolved alongside enabled."""

    def resolve(self) -> None:
        from vllm import envs

        if self.resolved:
            return
        aliases = {
            "enabled": "VLLM_SM70_FP8_TURBOMIND",
            "dequant_fallback": "VLLM_SM70_FP8_DEQUANT_FALLBACK",
            "qpn8": "VLLM_SM70_FP8_QPN8",
            "qpn8_pp2_tp4": "VLLM_SM70_FP8_QPN8_PP2_TP4",
            "qpn8_shared_gate": "VLLM_SM70_FP8_QPN8_PP2_TP4_SHARED_GATE",
            "prescaled_decode": "VLLM_SM70_FP8_PRESCALED_M1_DECODE",
            "prescaled_shared_gate": "VLLM_SM70_FP8_PRESCALED_M1_SHARED_GATE",
            "prefill_prescaled": "VLLM_SM70_FP8_PREFILL_PRESCALED",
            "prefill_exact_dense": "VLLM_SM70_FP8_PREFILL_EXACT_DENSE",
            "prefill_visible_dense_mm": "VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM",
            "gated_silu": "VLLM_SM70_FP8_DENSE_GATED_SILU",
        }
        explicit = []
        generic_is_auto = self.qpn8 is None
        specific_is_auto = self.qpn8_pp2_tp4 is None
        generic_override = envs.is_set(aliases["qpn8"])
        specific_override = envs.is_set(aliases["qpn8_pp2_tp4"])
        for field, name in aliases.items():
            if envs.is_set(name):
                logger.warning_once(
                    "%s is deprecated for serialized FP8 linear layers; use "
                    "kernel_config.sm70_fp8.%s. Explicit configuration wins.",
                    name,
                    field,
                )
            value = getattr(self, field)
            if value is None:
                value = getattr(envs, name)
                if field == "enabled":
                    value = envs.use_sm70_turbomind(value)
                setattr(self, field, value)
                if envs.is_set(name) and value:
                    explicit.append(field)
            elif value:
                explicit.append(field)
        if specific_is_auto:
            # An explicit generic rollback wins over a specific legacy enable.
            if generic_override and not self.qpn8:
                self.qpn8_pp2_tp4 = False
            elif (
                not specific_override
                and generic_override
                or not generic_is_auto
                and not specific_override
            ):
                self.qpn8_pp2_tp4 = self.qpn8
        self.explicit_enables = tuple(explicit)
        self.force_marlin = envs.force_sm70_marlin()
        self.legacy_grouped_bmm_decode = envs.VLLM_SM70_FP8_GROUPED_BMM_DECODE
        self.legacy_prefill_fast_selector = envs.VLLM_SM70_FP8_PREFILL_FAST_SELECTOR
        self.resolved = True


@config
class Sm70GgufConfig:
    """Operation-level policy for native GGUF storage on Volta."""

    enabled: bool = True
    """Admit the packaged native extension when the operator supports the format."""

    prefill_min_m: int = 8
    """Use dequantization plus tensor-core FP16 GEMM from this token count."""

    @field_validator("prefill_min_m")
    @classmethod
    def _positive_prefill_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("GGUF prefill_min_m must be positive")
        return value


@config
class Sm70SparseConfig:
    """Per-engine sparse attention policy; individual operators guard layouts."""

    indexer_decode_cublas: bool = True
    """Share paged index keys between query heads and rows when eligible."""
    decode_bmm: bool = True
    """Gather packed FP8 keys for eligible FP16 sparse decode matmuls."""
    prefill_bmm: bool = True
    """Use bounded batched matmuls for eligible FP16 sparse prefill."""
    active: bool = Field(default=False, init=False)
    """Whether the engine's metadata describes sparse indexed attention."""
    reason: str | None = Field(default=None, init=False)
    """Startup capability rejection; individual calls also check tensor layouts."""


@config
class LongContextKVConfig:
    """Active Host KV long-context mode for compressed-sparse (QSA) models.

    When enabled, the main QSA K/V history is budgeted in a host memory pool
    (the authoritative copy) with a GPU hot-page pool; recurrent state, drafter
    windows and the compressed index stay on the GPU. Startup checks that each
    pool holds at least one ``max_model_len`` request, and the scheduler admits
    or grows a request only when the GPU state segment, the GPU history
    segment, the host history pool and the hot-page pool all have room. A
    request that does not fit waits (or preempts under the existing rules); it
    is never rejected for lack of pool memory. Disabled, nothing here has any
    effect.
    """

    enabled: bool = False
    """Turn on long-context mode; every other field is inert while False."""
    latest_only: bool = True
    """Keep each observed chain's latest, resend and structural branch points."""
    enable_junction_evidence: bool = True
    """Require successor evidence for junction retention; False uses legacy policy."""
    superseded_keep_ssd: bool = False
    """Archive superseded checkpoints on SSD before releasing their RAM."""
    host_store_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """Unified host budget across MAIN, STATE and DRAFT, summed over workers.
    Includes ZERO_PAGE and build reservations. Mutually exclusive with the
    legacy host_pool_gib/host_pool_tokens and connector cpu_bytes_to_use."""
    host_pool_gib: float | None = Field(default=None, gt=0)
    """Host history pool for the main QSA K/V, in GiB summed over all workers
    (like ``kv_offloading_size``). Exactly one of this and
    ``host_pool_tokens`` is required when enabled."""
    host_pool_tokens: int | None = Field(default=None, gt=0)
    """Host history pool sized in history tokens (rounded down to whole KV
    blocks) instead of GiB."""
    hot_pool_gib: (
        Annotated[float, Field(gt=0, allow_inf_nan=False)] | Literal["auto"] | None
    ) = None
    """Guaranteed minimum hot slots, converted from a per-GPU GiB budget
    covering payload, slot metadata,
    shared workspace and write staging. Per-history-block metadata is charged
    separately to the history segment. At most one of this and
    ``hot_pool_ratio`` may be set. ``auto`` uses the remaining KV memory after
    state, demand-sized history and prefill staging. With neither set, use
    ``auto`` (a change from the previous implicit ratio sizing). Auto guarantees
    the per-step verify working set. The actual hot region may be larger than
    either minimum; static host allocation allocates a static pool. A minimum that
    cannot coexist
    with the active working set is a startup error. The host pool must still
    be sized explicitly."""
    hot_pool_ratio: float | None = Field(default=None, gt=0, le=1)
    """Guaranteed minimum hot slots, using the existing conversion from a
    fraction of the host history pool's rank-local tokens. Mutually exclusive
    with hot_pool_gib; the actual hot region may be larger."""
    prewarm_hot_pool: bool = True
    """Warm the hot-page pool from the last prefill step's selection."""
    prefill_staging_gib: float = Field(default=0.0, ge=0)
    """GPU memory per GPU reserved for append-prefill history staging."""
    state_blocks: int | None = Field(default=None, gt=0)
    """GPU state-segment block IDs (recurrent states, PLE states, drafter
    windows, checkpoints). None reserves ``max_num_seqs`` worst-case requests
    plus the pool's null block."""
    main_kv_placement: Literal["host", "gpu"] = "host"
    """Where the main QSA K/V history lives. ``host``: one page per history
    block ID in a pinned host store on every worker, read through the GPU
    hot-page pool. ``gpu``: in the GPU history segment next to the
    compressed index (the same-plan reference arm); the host and hot pools
    then only budget admission."""
    draft_main_placement: Literal["gpu", "host"] = "host"
    """Where an MTP draft's main K/V lives when it has the target's sharded
    geometry (``qsa_dcp_index_shard`` with DCP). ``host`` (default): one more
    layer of the host store, so each history ID takes fewer GPU bytes and the
    context concurrency limit rises, at the price of draft forwards reading
    host pages. ``gpu``: in the GPU history segment, next to its index pages:
    fastest decode and the layout earlier releases ran by default. ``host``
    needs ``main_kv_placement="host"``; otherwise, or for a draft that does
    not share the target's geometry, the draft stays on the GPU and a warning
    is logged."""
    staged_writethrough: bool = True
    """Stage eager prefill writes on device and submit merged same-stream D2H
    copies. Decode/capture uses word-vectorized UVA. False retains the legacy
    writer. Byte-preserving and excluded from compile and SSD identities."""
    prefill_gather: Literal["passes", "union"] = "union"
    """Mixed steps with ``main_kv_placement="host"``. ``union`` (default,
    selection-union gather): fill the union of the step's selections into
    hot slots once per window of whole eight-row groups and run the GPU arm's
    attention call unchanged on a page-size-4 table, so the result does not
    depend on the window split. ``passes``: remap the selection in query-row
    passes that fit the hot slots (R_cap rows each). Decode steps are
    unaffected."""
    prefill_insert_age: Literal["old", "recent"] = "old"
    """``union`` only: hot groups that only prefill rows want are inserted
    as old (first evicted) or recent."""
    value_arbitration: bool = True
    """Host placement with the shared GPU region only: move the boundary
    between the GPU prefix-index cache and the hot pages by bounded value
    feedback. False keeps the fixed default split. Scheduler-only; it never
    affects correctness or admission and is excluded from the compile hash."""
    value_arbiter_settings: dict[str, bool | float | int | str] = Field(
        default_factory=dict
    )
    """Overrides of the arbitration window, dead band, quantum and the
    placeholder cost weights (``bw_uva``, ``bw_h2d``, ``w_async``,
    ``w_ttft``) for measurement runs. ``busy_time`` defaults to true and
    uses the union of dispatch-to-output busy spans; false restores the
    historical step-to-step wall clock. Excluded from the compile hash."""

    def validate_settings(self) -> None:
        """Reject contradictory sizes; called only when the mode is enabled."""
        if not self.enabled:
            return
        if (
            sum(
                x is not None
                for x in (
                    self.host_store_gib,
                    self.host_pool_gib,
                    self.host_pool_tokens,
                )
            )
            != 1
        ):
            raise ValueError(
                "kernel_config.long_context_kv needs exactly one of host_store_gib, "
                "host_pool_gib and host_pool_tokens when enabled."
            )
        if self.hot_pool_gib is not None and self.hot_pool_ratio is not None:
            raise ValueError(
                "kernel_config.long_context_kv accepts at most one of "
                "hot_pool_gib and hot_pool_ratio."
            )


@config
class TopologyAllReduceConfig:
    """Experimental measured single-node CUDA IPC all-reduce selection."""

    enabled: bool = False
    """Qualify and select routes before first model compilation and capture."""

    precision: Literal["fp16", "fp32"] = "fp16"
    """Use FP16 sums/transfers by default; opt into FP32 sums and partial
    transfers for accuracy at higher latency. Input and output stay FP16."""

    cache_dir: str | None = None
    """Selection cache directory; None uses PyTorch's default extension root."""


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    topology_all_reduce: TopologyAllReduceConfig = Field(
        default_factory=TopologyAllReduceConfig
    )
    """Opt-in topology-aware peer reduction with startup measured selection."""

    ir_op_priority: IrOpPriorityConfig = Field(default_factory=IrOpPriorityConfig)
    """
    vLLM IR op priority for dispatching/lowering during the forward pass.
    Platform defaults appended automatically during VllmConfig.__post_init__.
    """

    enable_flashinfer_autotune: bool = None  # type: ignore[assignment]
    """If True, run FlashInfer autotuning during kernel warmup."""

    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware
    - "triton": Use Triton-based fused MoE kernels
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
    - "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels
    - "cutlass": Use vLLM CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)
    - "flashinfer_b12x": Use FlashInfer CuteDSL fused MoE for SM12x
      (RTX Pro 6000 / DGX Spark)
    - "marlin": Use Marlin kernels (weight-only quantization)
    - "sm70_skinny": Use the skinny QPN kernels for NVFP4 and MXFP4 on SM70/SM75
      (weight-only quantization)
    - "humming": Use Humming Mixed Precision kernels
    - "triton_unfused": Use Triton unfused MoE kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "emulation": use BF16/FP16 GEMM, dequantizing weights and
                   running QDQ on activations.
    """

    linear_backend: LinearBackend = "auto"
    """Backend for quantized linear layer GEMM kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware
    - "turbomind": SM70 compressed-tensors NVFP4 weight-only kernels
    - "cutlass": Use CUTLASS-based kernels
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels
    - "flashinfer_trtllm": Use FlashInfer with TensorRT-LLM kernels
    - "flashinfer_cudnn": Use FlashInfer with cuDNN kernels
    - "marlin": Use Marlin kernels
    - "triton": Use Triton-based kernels
    - "deep_gemm": Use DeepGEMM kernels
    - "torch": Use PyTorch native scaled_mm kernels
    - "aiter": Use AMD AITer kernels (ROCm only)
    - "machete": Use Machete kernels (mixed-precision)
    - "fbgemm": Use FBGEMM kernels
    - "conch": Use Conch mixed-precision kernels
    - "exllama": Use Exllama mixed-precision kernels
    - "emulation": Use slow dequant-to-BF16 emulation (for testing only)"""

    sm70_rmsnorm_gated_exact: bool | None = None
    """Native gated norm; auto follows the Flash-Next model quality boundary."""

    sm70_dflash2_fp16_m8: bool = False
    """Opt in to packed FP16 M8 DFlash2 draft projections on SM70 TP4."""

    def resolve_sm70_rmsnorm_gated(self, *, qualified: bool) -> None:
        if self.sm70_rmsnorm_gated_exact is not None:
            return
        import os

        from vllm import envs

        name = "VLLM_SM70_RMSNORM_GATED_EXACT"
        self.sm70_rmsnorm_gated_exact = (
            bool(envs.environment_variables[name]())
            if name in os.environ
            else qualified
        )

    sm70_nvfp4: Sm70NvFp4Config = Field(default_factory=Sm70NvFp4Config)
    """SM70 compressed-tensors NVFP4 policy, resolved per engine."""

    sm70_awq: Sm70AwqConfig = Field(default_factory=Sm70AwqConfig)
    """SM70 dense AWQ policy, resolved per engine."""

    sm70_fp8: Sm70Fp8Config = Field(default_factory=Sm70Fp8Config)
    """SM70 serialized block-FP8 variant policy, resolved per engine."""

    sm70_gguf: Sm70GgufConfig = Field(default_factory=Sm70GgufConfig)
    """Native GGUF admission and Volta tensor-core prefill policy."""

    sm70_sparse: Sm70SparseConfig = Field(default_factory=Sm70SparseConfig)
    """SM70 sparse attention policy; admission uses actual tensor capabilities."""

    sm70_skinny_moe: bool = True
    """Admit compatible NVFP4/MXFP4 skinny MoE kernels on SM70/SM75."""

    sm70_skinny_moe_applicable: bool = Field(default=False, init=False)
    """Whether a loaded MoE family consults the skinny kernel policy."""

    moe_kernel_selections: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed MoE capability decisions, excluded from compilation hashing."""

    fused_fp16_aux_gemv: bool = True
    """Fuse compatible auxiliary projections already using exact FP16 GEMV."""

    fused_fp16_aux_gemv_applicable: bool = Field(default=False, init=False)
    """Whether loaded auxiliary projections admit the exact GEMV fusion."""

    linear_kernel_selections: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed selector decisions for loaded local layouts; diagnostic only."""

    qsa_auto_e4m3: bool = True
    """Default eligible QSA auto caches to planar per-token-head E4M3."""
    qsa_auto_e4m3_active: bool = Field(default=False, init=False)
    """Whether automatic QSA E4M3 storage was selected."""
    qsa_auto_e4m3_reason: str | None = Field(default=None, init=False)
    """Startup reason when automatic QSA E4M3 storage cannot be selected."""

    sm70_qsa_shared_key: bool = False
    """Share FP16 indexer keys across M2..8 queries of one request on SM70
    (1CatAI #1127; opt-in here, also VLLM_SM70_QSA_SHARED_KEY=1)."""

    sm70_qsa_fused_topk: bool = False
    """Use native fused scoring and exact Top-512 for replicated SM70 QSA
    decode and verification batches."""

    sm70_qsa_paged_fp16_h12: bool = False
    """Use the native FP16 paged QSA reader for eligible TP4/DCP2 target
    decode on SM70. Its reduction and probability rounding differ."""

    sm70_qsa_paged_fp16_h6: bool | None = None
    """Use the native FP16 paged QSA reader in replicated MTP decode graphs
    on SM70. When unset, enable it in the SM70 TP4/DCP2 NVFP4 profile with
    async MTP and FP16 GPU KV. Its reduction and probability rounding differ."""

    sm70_qsa_device_history: bool = False
    """Use native SM70 QSA GPU-cache reads for eligible TP4/DCP2 decode
    and verification batches. This changes the local attention arithmetic."""

    sm70_qsa_input_batch: bool = False
    """Pack and fuse the two checkpoint-FP16 QSA input projections in the
    SM70 TP4 NVFP4 target MTP decode graph."""

    sm70_gdn_input_v154: bool = False
    """Use the v154 packed GDN inputs for physical M3/M4 during SM70 target
    FULL graph capture. Other rows keep their existing path."""

    qsa_dcp_index_shard: bool = False
    """Shard the target QSA compressed indexer keys over the DCP group.

    Group g is owned by DCP rank ``g % dcp`` at local column ``g // dcp``. Each
    rank scores only its own groups; local top-k candidates are merged across
    the group into the unchanged global selection. Requires DCP > 1; the MTP
    draft indexer stays replicated.
    """

    long_context_kv: LongContextKVConfig = Field(default_factory=LongContextKVConfig)
    """Active Host KV long-context mode and its pool sizes (off by default)."""

    qsa_cross_group_hit_check: bool = False
    """Report a prefix hit only when every KV cache group is complete at it.

    Checks the final hit of the GPU prefix cache and of the offloading
    connector against one per-group rule (full attention: every block; Mamba
    align: the state at the hit; sliding window: its window), turns an
    incomplete hit into a miss, trims extra full-attention blocks, and checks
    the connector's load plan against the same hit. Also lets an owner of
    storage outside these pools register residency probes. Scheduler-only;
    off keeps the existing lookup results unchanged.
    """

    ple_disk_cascade: bool | None = None
    """Allow resident FP8 PLE tiers to spill to mapped checkpoint storage.

    Unset resolves to True upstream, or False in the ReVolta SM70 QSA TP4
    profile. Explicit True/False always wins over the profile.
    """
    ple_disk_cascade_source: str = Field(default="explicit", init=False)
    """Diagnostic origin of the resolved cascade choice; excluded from hashes."""
    ple_disk_release_pages: bool = False
    """Release file-backed PLE mappings after gathers to reduce resident RAM."""
    ple_disk_cascade_active: bool = Field(default=False, init=False)
    """Resolved FP8 storage, dtype and pipeline capability admission."""
    ple_disk_cascade_reason: str | None = Field(default=None, init=False)
    """Startup reason when the disk cascade cannot serve this configuration."""

    sm70_greedy_verify: bool = False
    """Verify greedy MTP drafts from TP-local target argmax pairs instead of
    gathering full-vocabulary logits for the rejection sampler."""

    sm70_sparse_target_sampling: bool | Literal["device_top64"] = False
    """Compact target sampling for SM70 MRv2 with uniform MTP3-5,
    greedy drafts and target temperature=1, top_k=20, top_p=0.95.
    False keeps the existing fallback; True uses guarded compact sampling
    with dense reference sampling for ambiguous cutoffs. "device_top64"
    applies the rank-20 score cutoff, including ties, within the existing
    FP32 Top64 on device, then applies top-p. Support is deliberately bounded
    at 64 even if unseen vocabulary entries tie at the cutoff. FP32 top-p
    CDF rounding near the cutoff can change the retained boundary token
    versus dense sampling; this mode does not fall back for these cutoffs.
    Unsupported batches keep the existing fallback in either compact mode.
    """
    sm70_nvlink_hc_active: bool = Field(default=False, init=False)
    """Resolved TP hardware admission, which changes the compiled HC graph."""

    sm70_hc_ll_async: bool = False
    """Allow the usual SM70 NVLink HC LL admission under async scheduling."""

    sm70_hc_moe_dual_partial: bool = False
    """Pass separate routed and shared expert outputs into NVLink HC on
    Flash-Next decode. Independent of HCX admission. HCX accumulates both
    partials in FP32, so its dual-input path can change numerical results.
    """

    sm70_hcx: bool = False
    """Use local-schedule HCX at Flash-Next attention-to-MLP and MoE-to-next-
    attention HyperConnection boundaries in uniform decode graphs of at most
    eight rows. Requires admitted NVLink HC LL. Changes FP16 rounding."""
    sm70_top1x: bool = False
    """Exchange TP4 greedy top-1 candidates over the admitted NVLink HC LL peers
    in two hops instead of an all-gather (1CatAI v1.5.4). Selects the same token
    as the all-gather for finite logits."""
    sm70_hcx_active: bool = Field(default=False, init=False)
    """HCX operator/weight preparation changed the target compiler graph."""

    ple_result_transport: Literal["auto", "cuda", "mapped"] = "auto"
    """Select CPU PLE result transport by local operator/resource capability."""
    ple_result_transports: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed per-layer result transport and small pinned-buffer sizes."""

    ple_input_prepare: bool = False
    """Fuse qualified SM70 PLE context gathering and query-boundary staging."""
    ple_input_preparations: dict[str, Any] = Field(
        default_factory=dict, init=False, repr=False
    )
    """Observed PLE input operator selection and fallback reasons."""

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    @field_validator("linear_backend", mode="before")
    @classmethod
    def _normalize_linear_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        ignored_factors = {
            "topology_all_reduce",  # Include enabled math policy below, not cache path.
            "enable_flashinfer_autotune",
            "ir_op_priority",  # handled separately below
            "linear_kernel_selections",
            "moe_kernel_selections",
            "sm70_skinny_moe_applicable",
            "fused_fp16_aux_gemv_applicable",
            "ple_disk_cascade_reason",
            "ple_disk_cascade_source",
            "ple_result_transports",
            "ple_input_prepare",  # Input staging is outside the compiled model.
            "ple_input_preparations",
            "qsa_auto_e4m3_reason",
            # Sampler-only switches; they do not change the compiled model.
            "sm70_greedy_verify",
            "sm70_top1x",
            "sm70_sparse_target_sampling",
            "sm70_hc_ll_async",  # Compiled model uses sm70_nvlink_hc_active.
            # Scheduler-only prefix-hit check; no compiled graph changes.
            "qsa_cross_group_hit_check",
        }
        if not self.sm70_qsa_fused_topk:
            ignored_factors.add("sm70_qsa_fused_topk")
        if not self.sm70_qsa_paged_fp16_h6:
            ignored_factors.add("sm70_qsa_paged_fp16_h6")
        if not self.sm70_skinny_moe_applicable:
            ignored_factors.add("sm70_skinny_moe")
        if not self.sm70_dflash2_fp16_m8:
            ignored_factors.add("sm70_dflash2_fp16_m8")
        if not self.sm70_nvlink_hc_active:
            ignored_factors.add("sm70_nvlink_hc_active")
        if not self.sm70_hc_moe_dual_partial:
            ignored_factors.add("sm70_hc_moe_dual_partial")
        ignored_factors.add("sm70_hcx")  # Only resolved preparation changes the graph.
        if not self.sm70_hcx_active:
            ignored_factors.add("sm70_hcx_active")
        if not self.fused_fp16_aux_gemv_applicable:
            ignored_factors.add("fused_fp16_aux_gemv")
        if not self.qsa_auto_e4m3_active:
            ignored_factors.update({"qsa_auto_e4m3", "qsa_auto_e4m3_active"})
        if not self.ple_disk_cascade_active:
            ignored_factors.update(
                {
                    "ple_disk_cascade",
                    "ple_disk_release_pages",
                    "ple_disk_cascade_active",
                }
            )
        if not self.sm70_awq.resolved:
            # An unused format must not perturb another format's graph cache.
            ignored_factors.add("sm70_awq")
        if not self.sm70_fp8.resolved:
            ignored_factors.add("sm70_fp8")
        if not self.sm70_sparse.active:
            ignored_factors.add("sm70_sparse")
        if not self.long_context_kv.enabled:
            # An unused mode must not perturb existing graph caches.
            ignored_factors.add("long_context_kv")
        factors = get_hash_factors(self, ignored_factors)
        if self.topology_all_reduce.enabled:
            factors["topology_all_reduce"] = dict(
                enabled=True, precision=self.topology_all_reduce.precision
            )
        if "long_context_kv" in factors:
            # Scheduler-only arbitration knobs keep the previous identity.
            lc_factor = factors["long_context_kv"]
            assert isinstance(lc_factor, tuple)
            name, items = lc_factor
            factors["long_context_kv"] = (
                name,
                tuple(
                    item
                    for item in items
                    if item[0]
                    not in (
                        "value_arbitration",
                        "value_arbiter_settings",
                        "latest_only",
                        "superseded_keep_ssd",
                        "enable_junction_evidence",
                        "staged_writethrough",
                    )
                    and not (
                        item[0] == "host_store_gib"
                        and self.long_context_kv.host_store_gib is None
                    )
                ),
            )
        factors["ir_op_priority"] = self.ir_op_priority.compute_hash()
        return hash_factors(factors)

    @field_validator("enable_flashinfer_autotune", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        if value is None:
            return value
        return handler(value)

    def set_platform_defaults(self, vllm_config: "VllmConfig") -> None:
        """Set platform-specific defaults for the kernel config."""
        import torch

        from vllm.platforms import current_platform

        model = vllm_config.model_config
        text = getattr(model, "hf_text_config", None)
        self.sm70_sparse.active = bool(getattr(text, "index_head_dim", None))
        self.sm70_sparse.reason = (
            "no indexed sparse-attention metadata"
            if not self.sm70_sparse.active
            else (
                "requires CUDA compute capability 7.x"
                if not current_platform.is_cuda()
                or not current_platform.is_device_capability_family(70)
                else ("requires FP16 queries" if model.dtype != torch.float16 else None)
            )
        )

        platform_op_priority = current_platform.get_default_ir_op_priority(vllm_config)
        logger.debug(
            "Setting platform-specific IR op priority defaults: %s, user-defined: %s",
            platform_op_priority,
            self.ir_op_priority,
        )
        for op_name, op_priority in asdict(platform_op_priority).items():
            current_op_priority: list[str] = getattr(self.ir_op_priority, op_name)
            if current_op_priority is None:
                setattr(self.ir_op_priority, op_name, op_priority)
            else:
                # Append platform-specific priorities
                # Must be idempotent because vllm_config.set_platform_defaults() may be
                # called multiple times (due to VllmConfig.__post_init__ manual call).
                unique_op_priority = [
                    op for op in op_priority if op not in current_op_priority
                ]
                current_op_priority.extend(unique_op_priority)

        logger.info(
            "Final IR op priority after setting platform defaults: %s",
            self.ir_op_priority,
        )
