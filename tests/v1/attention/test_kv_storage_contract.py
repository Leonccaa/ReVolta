# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical owner contracts: real 816-slot pages, bytes and borrowed views."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.attention.kv_cache_view", reason="v1 storage API absent")

from vllm.v1.attention.kv_cache_view import (
    KVCacheView,
    KVPageLayout,
    KVPhysicalPage,
    bind_attention_cache,
    bind_planar_kv_cache,
)
from vllm.v1.attention.kv_codecs import (
    BF16,
    FP8_E4M3,
    FP16,
    INT8,
    INT8_PER_TOKEN_HEAD,
    resolve_kv_codec,
)
from vllm.v1.attention.ops.int8_kv import reference_int8_kv_write
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache


def spec(dtype=torch.int8, mode=KVQuantMode.INT8_PLANAR):
    return FullAttentionSpec(
        block_size=816, num_kv_heads=1, head_size=256, dtype=dtype, kv_quant_mode=mode
    )


def packed(raw, member=0, padded=0):
    return bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=816,
        kernel_block_size=16,
        num_heads=1,
        head_size=256,
        page_size_bytes=424320 + padded,
        member_index=member,
        member_count=2,
    )


def test_identity_and_geometry():
    assert resolve_kv_codec("int8") is INT8
    assert INT8.cache_identity != INT8_PER_TOKEN_HEAD.cache_identity
    assert INT8.rounding == "nearest_even"
    assert INT8_PER_TOKEN_HEAD.rounding == "truncate"
    layout = KVPageLayout(16, 1, 256, 256)
    assert (
        layout.value_offset,
        layout.key_scale_offset,
        layout.value_scale_offset,
        layout.page_size_bytes,
    ) == (4096, 8192, 8256, 8320)
    assert spec().page_size_bytes == 424320
    assert spec(torch.uint8, KVQuantMode.FP8_PER_TENSOR).page_size_bytes == 417792
    assert spec(torch.float16, KVQuantMode.NONE).page_size_bytes == 835584


@pytest.mark.parametrize("member", [0, 1])
@pytest.mark.parametrize("padding", [0, 816])  # 16 bytes per kernel page
def test_packed_view_zero_and_offload(member, padding):
    page = (424320 + padding) * 2
    raw = torch.full((3 * page,), 73, dtype=torch.int8)
    carrier = packed(raw, member, padding)
    view = KVCacheView.bind(carrier, INT8, 256)
    assert view.key.stride() == (16640 + padding * 2 // 51, 256, 256, 1)
    assert view.key_scale.stride() == ((16640 + padding * 2 // 51) // 4, 1, 1)
    assert view.subpage_mapping(4).owner_page_span == 4
    # Block zeroer uses physical base, even for member 1.
    base, size = KVPhysicalPage.zeroer_segment(carrier, 0, 51)
    assert base == raw.data_ptr() and size == page
    pages = KVPhysicalPage(424320 + padding, 2, member).raw_pages(carrier, 3)
    saved = pages[1].clone()  # CPU/SSD transfer is opaque bytes, includes padding
    pages[1].zero_()
    assert torch.all(pages[0] == 73) and torch.all(pages[2] == 73)
    pages[1].copy_(saved)
    assert torch.all(raw == 73)
    # Binding/restoring never initializes scales or writes payload.
    rebound = KVCacheView.bind(packed(raw, member, padding), INT8, 256)
    assert rebound.key.data_ptr() == view.key.data_ptr()
    assert rebound.key_scale.data_ptr() == view.key_scale.data_ptr()
    assert torch.all(raw == 73)


@pytest.mark.parametrize("codec", [FP16, BF16, FP8_E4M3])
@pytest.mark.parametrize("members", [1, 2])
def test_legacy_view_is_bitwise_identical(codec, members):
    item = torch.empty((), dtype=codec.storage_dtype).element_size()
    page = 816 * 512 * item
    raw = torch.arange(3 * page * members, dtype=torch.int64).to(torch.uint8)
    shape = (153, 2, 16, 1, 256)
    for member in range(members):
        cache = bind_attention_cache(
            raw,
            shape=shape,
            dtype=codec.storage_dtype,
            page_size_bytes=page,
            scheduler_block_size=816,
            kernel_block_size=16,
            member_index=member,
            member_count=members,
        )
        old = raw.view(codec.storage_dtype).as_strided(
            shape, (8192 * members, 4096, 256, 256, 1), 8192 * member
        )
        assert cache.stride() == old.stride()
        assert cache.storage_offset() == old.storage_offset()
        view = KVCacheView.bind(cache, codec, 256)
        for actual, expected in zip((view.key, view.value), old.unbind(1)):
            assert torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        assert torch.equal(
            raw, torch.arange(raw.numel(), dtype=torch.int64).to(torch.uint8)
        )


def test_exact_rne_and_nonfinite_rows():
    x = torch.tensor(
        [
            [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.0],
            [0.0] * 8,
            [float("inf")] + [1.0] * 7,
            [float("nan")] + [1.0] * 7,
        ],
        dtype=torch.float32,
    )
    q, s = INT8.encode(x)
    assert q[0].tolist() == [127, 0, 2, 2, 0, -2, -2, 0]
    assert s[0].item() == 1.0
    assert s[1].view(torch.int32).item() == torch.tensor(1e-6).view(torch.int32).item()
    assert torch.isnan(s[2:]).all() and (q[2:] == 0).all()
    # Independent scalar RN32 division oracle at FP16 extremes and tiny values.
    rows = torch.cat(
        (
            torch.linspace(-65504, 65504, 256).half()[None],
            torch.linspace(-1e-5, 1e-5, 256).half()[None],
        )
    )
    q, s = INT8.encode(rows)
    for row in range(2):
        vals = rows[row].tolist()
        scale = torch.tensor(max(max(abs(v) for v in vals) / 127, 1e-6)).item()
        expected = [
            max(-128, min(127, round(torch.tensor(v / scale).item()))) for v in vals
        ]
        assert q[row].tolist() == expected and s[row].item() == scale


def test_row_writer_boundaries_batch_invariance_and_peer_isolation():
    page = 848640
    raw = torch.full((3 * page,), 77, dtype=torch.int8)
    other = raw.clone()
    slots = torch.tensor([15, 16, 815, 816, 1631, 1632, -1, 2448])
    torch.manual_seed(5)
    k = torch.randn(10, 1, 256, dtype=torch.float16)
    v = k.flip(-1).contiguous()
    a = KVCacheView.bind(packed(raw, 1), INT8, 256)
    b = KVCacheView.bind(packed(other, 1), INT8, 256)
    reference_int8_kv_write(k, v, a, slots)
    for row, slot in enumerate(slots):
        reference_int8_kv_write(k[row : row + 1], v[row : row + 1], b, slot[None])
    assert torch.equal(raw, other)
    # Prefix reuse restores complete bytes, no codec-dependent slicing.
    restore = raw.clone()
    raw.zero_()
    raw.copy_(restore)
    assert torch.equal(raw, other)
    for page_idx in range(153):
        start = page_idx * 16640
        assert torch.all(raw[start : start + 8320] == 77)  # member 0 untouched
    with pytest.raises(ValueError, match="unique"):
        reference_int8_kv_write(k, v, a, torch.tensor([1, 1]))


def test_real_runner_allocates_mixed_target_and_draft_by_spec():
    class Backend:
        @staticmethod
        def get_kv_cache_shape(n, b, h, d, cache_dtype_str):
            return n, 2, b, h, d

        @staticmethod
        def get_kv_cache_stride_order():
            return tuple(range(5))

        @staticmethod
        def get_kv_cache_block_dim(*a, **kw):
            return 0

    target = spec()
    draft = replace(
        spec(torch.float16, KVQuantMode.NONE), block_size=1632, dcp_sharded=False
    )
    raw = torch.full((3 * 848640,), 33, dtype=torch.int8)
    draft_raw = torch.full((3 * draft.page_size_bytes,), 29, dtype=torch.int8)
    groups = [
        SimpleNamespace(
            kv_cache_group_id=0,
            kv_cache_spec=target,
            backend=Backend,
            layer_names=["a", "b"],
        ),
        SimpleNamespace(
            kv_cache_group_id=1,
            kv_cache_spec=draft,
            backend=Backend,
            layer_names=["draft"],
        ),
    ]
    out = _reshape_kv_cache(
        groups,
        {"a": raw, "b": raw, "draft": draft_raw},
        "int8",
        [16, 16],
        {},
        {"a": (0, 2), "b": (1, 2)},
    )
    assert out["a"].shape == (153, 2, 16, 1, 260)
    assert out["b"].storage_offset() == 8320
    assert out["draft"].dtype == torch.float16
    assert out["draft"].shape == (306, 2, 16, 1, 256)
    assert torch.all(raw == 33) and torch.all(draft_raw == 29)


def test_invalid_padding_and_bounds():
    with pytest.raises(ValueError, match="divide"):
        packed(torch.empty(3 * 848640, dtype=torch.int8), padded=16)
    with pytest.raises(ValueError, match="smaller"):
        KVPageLayout(16, 1, 256, 256).scheduler_page_stride(8192)
    with pytest.raises(ValueError, match="allocation"):
        KVPhysicalPage(848640).raw_pages(torch.empty(100, dtype=torch.int8), 3)


def test_physical_padding_keeps_mixed_codec_token_grids():
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size

    a = replace(spec(), block_size=16)
    b = replace(spec(torch.float16, KVQuantMode.NONE), block_size=16)
    result = unify_kv_cache_spec_page_size({"int8": a, "fp16": b})
    assert all(s.block_size == 16 for s in result.values())
    assert {s.page_size_bytes for s in result.values()} == {16384}
    raw = torch.full((3 * 16384,), 17, dtype=torch.int8)
    cache = bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=16,
        kernel_block_size=16,
        num_heads=1,
        head_size=256,
        page_size_bytes=result["int8"].page_size_bytes,
    )
    view = KVCacheView.bind(cache, INT8, 256)
    assert view.key.stride(0) == 16384
    assert torch.all(raw == 17)


def test_actual_zeroer_canonical_offload_and_prefix_int8():
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.worker.utils import KVBlockZeroer

    class Backend:
        @staticmethod
        def get_kv_cache_block_dim(*args, **kwargs):
            return 0

    raw = torch.full((3 * 848640,), 57, dtype=torch.int8)
    views = {name: packed(raw, i) for i, name in enumerate(["a", "b"])}
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec(),
        backend=Backend,
        layer_names=["a", "b"],
    )
    zeroer = KVBlockZeroer(torch.device("cpu"), pin_memory=False)
    zeroer.init_meta(
        [group],
        [16],
        "int8",
        set(),
        {name: SimpleNamespace(kv_cache=t) for name, t in views.items()},
    )
    addrs, strides, sizes = (t.tolist() for t in zeroer._meta[:3])
    assert addrs == [raw.data_ptr()]
    assert strides == sizes == [848640 // 4]
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[KVCacheGroupSpec(["a", "b"], spec())],
        kv_cache_tensors=[
            KVCacheTensor(
                size=raw.numel(), shared_by=["a", "b"], packed_members=["a", "b"]
            )
        ],
    )
    offload_spec = Mock()
    offload_spec.kv_cache_config = config
    offload_spec.get_handlers.return_value = []
    worker = OffloadingConnectorWorker(offload_spec)
    worker.register_kv_caches(views)
    canonical = offload_spec.get_handlers.call_args[0][0]
    assert len(canonical.tensors) == 1
    assert canonical.tensors[0].tensor.shape == (3, 848640)
    assert len(canonical.group_data_refs[0]) == 1
    assert canonical.group_data_refs[0][0].page_size_bytes == 848640


def test_owner_dtype_spelling_is_independent_of_target():
    assert spec().layout_cache_dtype("fp8_e4m3") == "int8"
    assert spec(torch.float16, KVQuantMode.NONE).layout_cache_dtype("int8") == "auto"


def test_prefix_reuses_int8_blocks_and_retains_dcp_grid():
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    init_none_hash(sha256)
    target = replace(spec(), block_size=16)
    draft = replace(target, dcp_sharded=False)
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], target),
            KVCacheGroupSpec(["draft"], draft),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=128,
        hash_block_size=16,
        dcp_world_size=2,
        enable_caching=True,
    )
    assert [m.block_size for m in manager.coordinator.single_type_managers] == [32, 16]
    first = make_request("first", list(range(64)) + [91] * 5, 16, sha256)
    assert manager.allocate_slots(first, 69) is not None
    second = make_request("second", list(range(64)) + [92] * 5, 16, sha256)
    blocks, tokens = manager.get_computed_blocks(second)
    assert tokens == 64 and [len(g) for g in blocks.blocks] == [2, 4]
    manager.free(first)


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "int8"])
def test_platform_preserves_1632_grid(monkeypatch, dtype):
    from tests.platforms.test_hybrid_dcp_block_alignment import (
        _LooseBackend,
        _vllm_config,
    )
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms.interface import Platform
    from vllm.v1.kv_cache_interface import MambaSpec

    config, model_cls = _vllm_config(1632, 2)
    config.model_config.get_head_size = lambda: 256
    config.cache_config.cache_dtype = dtype
    # 835584 is the byte-width-compatible state-page budget at the fixed grid.
    model_cls.get_mamba_specs_from_config = classmethod(
        lambda cls, cfg: (
            MambaSpec(shapes=((417792,),), dtypes=(torch.float16,), block_size=-1),
        )
    )
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *a, **kw: (model_cls, "Fake")
    )
    Platform._align_hybrid_block_size(config, _LooseBackend)
    assert config.cache_config.block_size == 1632
    assert config.cache_config.mamba_block_size == 1632
    assert config.cache_config.mamba_page_size_padded == (
        848640 if dtype == "int8" else None
    )


def test_ssd_frame_budget_and_persistent_namespace():
    from vllm.v1.attention.kv_cache_view import kv_storage_metadata
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    owner = KVPhysicalPage(424320, 2, 1)
    assert owner.page_size_bytes % 512 == 256
    assert owner.io_frame_bytes() == 851968
    assert owner.io_frame_bytes(header_bytes=64) == 851968
    tensor = KVCacheTensor(
        size=3 * 848640, shared_by=["a", "b"], packed_members=["a", "b"]
    )
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[KVCacheGroupSpec(["a", "b"], spec())],
        kv_cache_tensors=[tensor],
    )
    meta = kv_storage_metadata(config, "int8")
    assert meta["owners"]["b"]["codec"] == ["int8", 1, "planar_scales"]
    assert meta["tensors"][0]["packed_members"] == ["a", "b"]
    assert meta["tensors"][0]["bytes_per_block"] == 848640
    old = replace(spec(torch.uint8, KVQuantMode.FP8_PER_TENSOR))
    config.kv_cache_groups = [KVCacheGroupSpec(["a", "b"], old)]
    assert kv_storage_metadata(config, "fp8_e4m3") != meta


def test_offload_copies_interior_padding_for_single_int8_member():
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    # Padding is spread over 51 kernel pages, so the last scale plane lies
    # beyond real_page_size_bytes. A prefix slice would lose valid scale bytes.
    padded = replace(spec(), page_size_padded=425136)
    raw = torch.full((3 * padded.page_size_bytes,), 61, dtype=torch.int8)
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=816,
        kernel_block_size=16,
        num_heads=1,
        head_size=256,
        page_size_bytes=padded.page_size_bytes,
    )
    view = KVCacheView.bind(carrier, INT8, 256)
    view.value_scale[50].fill_(123.5)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[KVCacheGroupSpec(["a"], padded)],
        kv_cache_tensors=[KVCacheTensor(size=raw.numel(), shared_by=["a"])],
    )
    offload_spec = Mock()
    offload_spec.kv_cache_config = config
    offload_spec.get_handlers.return_value = []
    OffloadingConnectorWorker(offload_spec).register_kv_caches({"a": carrier})
    canonical = offload_spec.get_handlers.call_args[0][0]
    ref = canonical.group_data_refs[0][0]
    assert ref.page_size_bytes == padded.page_size_bytes
    pages = canonical.tensors[0].tensor
    saved = pages[0, : ref.page_size_bytes].clone()
    pages[0].zero_()
    pages[0, : ref.page_size_bytes].copy_(saved)
    assert torch.all(view.value_scale[50] == 123.5)


def test_mixed_fp16_virtual_padding_is_part_of_offload_span():
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    padded = replace(spec(torch.float16, KVQuantMode.NONE), page_size_padded=848640)
    raw = torch.zeros(3 * padded.page_size_bytes, dtype=torch.int8)
    carrier = bind_attention_cache(
        raw,
        shape=(153, 2, 16, 1, 256),
        dtype=torch.float16,
        page_size_bytes=padded.page_size_bytes,
        scheduler_block_size=816,
        kernel_block_size=16,
    )
    view = KVCacheView.bind(carrier, FP16, 256)
    view.value[50].fill_(123.5)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[KVCacheGroupSpec(["a"], padded)],
        kv_cache_tensors=[KVCacheTensor(size=raw.numel(), shared_by=["a"])],
    )
    offload_spec = Mock()
    offload_spec.kv_cache_config = config
    offload_spec.get_handlers.return_value = []
    OffloadingConnectorWorker(offload_spec).register_kv_caches({"a": carrier})
    canonical = offload_spec.get_handlers.call_args[0][0]
    ref = canonical.group_data_refs[0][0]
    assert ref.page_size_bytes == padded.page_size_bytes
    pages = canonical.tensors[0].tensor
    saved = pages[0, : ref.page_size_bytes].clone()
    pages[0].zero_()
    pages[0, : ref.page_size_bytes].copy_(saved)
    assert torch.all(view.value[50] == 123.5)


def test_legacy_triton_writer_does_not_reinterpret_planar_int8_as_fp8():
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        _is_supported_kv_cache_dtype,
    )

    assert not _is_supported_kv_cache_dtype("int8")
    assert _is_supported_kv_cache_dtype("auto")
    assert _is_supported_kv_cache_dtype("fp8_e4m3")


@pytest.mark.parametrize("configured_dtype", ["fp8_e4m3", "int8"])
def test_int8_draft_requires_ssd_framing_even_with_a_legacy_target(
    monkeypatch, configured_dtype
):
    from vllm.v1.attention.kv_cache_view import has_planar_kv_cache
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    # Test the tiering policy without constructing a shared-memory/IO backend.
    def parent_init(self, *args):
        self.extra_config = {"secondary_tiers": [{"type": "fs"}]}
        self.partition_by_group = False

    monkeypatch.setattr(CPUOffloadingSpec, "__init__", parent_init)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=configured_dtype),
        compute_hash=lambda: "frozen-config",
    )
    cache = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[
            KVCacheGroupSpec(["draft"], replace(spec(), dcp_sharded=False))
        ],
        kv_cache_tensors=[KVCacheTensor(size=3 * 424320, shared_by=["draft"])],
    )
    assert has_planar_kv_cache(cache)
    with pytest.raises(ValueError, match="aligned FS framing"):
        TieringOffloadingSpec(config, cache)
    # Ordinary FP8-only tiering keeps its old namespace policy.
    cache.kv_cache_groups = [
        KVCacheGroupSpec(["target"], spec(torch.uint8, KVQuantMode.FP8_PER_TENSOR))
    ]
    assert not has_planar_kv_cache(cache)
    assert TieringOffloadingSpec(config, cache).persistent_layout is None


def test_int8_v1_kernel_geometry_is_part_of_persistent_identity():
    from vllm.v1.attention.kv_cache_view import kv_storage_metadata
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    raw = torch.empty(3 * 424320, dtype=torch.int8)
    # 48 divides 816, but differs from v1's frozen 16-token file layout.
    with pytest.raises(ValueError, match="16-token"):
        bind_planar_kv_cache(
            raw,
            num_blocks=3,
            scheduler_block_size=816,
            kernel_block_size=48,
            num_heads=1,
            head_size=256,
            page_size_bytes=424320,
        )
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=[KVCacheGroupSpec(["a"], spec())],
        kv_cache_tensors=[KVCacheTensor(size=raw.numel(), shared_by=["a"])],
    )
    kernel = kv_storage_metadata(config, "int8")["owners"]["a"]["kernel_page"]
    assert kernel == {
        "tokens": 16,
        "alignment": 16,
        "region_offsets": [0, 4096, 8192, 8256],
        "region_bytes": 8320,
        "member_stride": 8320,
    }


def test_legacy_inline_int8_keeps_its_original_transfer_span():
    # Inline scale bytes are not physical padding. Preserve the old canonical
    # ref policy for int8_per_token_head instead of redefining that format.
    carrier = torch.empty(153, 2, 16, 1, 260, dtype=torch.int8)
    owner = KVPhysicalPage(424320)
    assert owner.transfer_size(carrier, 3, 417792, planar=False) == 417792
