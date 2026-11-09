# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-group prefix-hit contract on the GPU prefix cache
(``kernel_config.qsa_cross_group_hit_check``).

A hit of ``L`` tokens is reported only when every KV cache group is complete
at ``L``: full attention holds blocks ``[0, L)``, a Mamba ``align`` group the
state ending at ``L``, a sliding-window group its window before ``L``. With
the flag off the coordinator's answer is returned unchanged.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_hit_contract import (
    GroupHitRule,
    HitNeed,
    KVHitContract,
    cross_group_hit_check_enabled,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BS = 16
SW_BLOCKS = 3  # cdiv(sliding_window - 1, BS) for sliding_window = 3 * BS


@pytest.fixture(autouse=True)
def _hash_fn():
    init_none_hash(sha256)


def _full(head_size: int = 1) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BS, num_kv_heads=1, head_size=head_size, dtype=torch.float32
    )


def _mamba() -> MambaSpec:
    return MambaSpec(
        block_size=BS,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _window() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=BS,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=SW_BLOCKS * BS,
    )


def _ring() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def _manager(specs: list[KVCacheSpec], check: bool) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=512,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        hash_block_size=BS,
        enable_caching=True,
        cross_group_hit_check=check,
    )


def _request(num_blocks: int, req_id: str = "r") -> Request:
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_blocks * BS + 1)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _cache(manager: KVCacheManager, request: Request, present: dict[int, set[int]]):
    """Publish block ``i`` of group ``g`` for every ``i in present[g]``."""
    pool = manager.block_pool
    for group_id, indices in present.items():
        for index in sorted(indices):
            (block,) = pool.get_new_blocks(1)
            block.block_hash = make_block_hash_with_group_id(
                request.block_hashes[index], group_id
            )
            pool.cached_block_hash_to_block.insert(block.block_hash, block)


def _hit(manager: KVCacheManager, request: Request):
    blocks, num_tokens = manager.get_computed_blocks(request)
    return [list(b) for b in blocks.blocks], num_tokens


def _oracle(specs: list[KVCacheSpec], present: dict[int, set[int]], limit: int) -> int:
    """Largest L (in blocks, <= limit) at which every group is complete."""
    for end in range(limit, 0, -1):
        ok = True
        for group_id, spec in enumerate(specs):
            have = present.get(group_id, set())
            if isinstance(spec, CircularBufferSpec):
                continue
            if isinstance(spec, MambaSpec):
                need = {end - 1}
            elif isinstance(spec, SlidingWindowSpec):
                need = set(range(max(0, end - SW_BLOCKS), end))
            else:
                need = set(range(end))
            ok &= need <= have
        if ok:
            return end
    return 0


# --- rules --------------------------------------------------------------------


def test_required_blocks_per_need():
    prefix = GroupHitRule(0, HitNeed.PREFIX, 16)
    state = GroupHitRule(1, HitNeed.STATE, 16)
    window = GroupHitRule(2, HitNeed.WINDOW, 16, window_tokens=48)
    assert list(prefix.required_blocks(80)) == [0, 1, 2, 3, 4]
    assert list(state.required_blocks(80)) == [4]
    assert list(window.required_blocks(80)) == [2, 3, 4]
    assert list(window.required_blocks(32)) == [0, 1]
    # Blocks served by an earlier (local) hit are excluded.
    assert list(prefix.required_blocks(80, start_tokens=48)) == [3, 4]
    assert list(window.required_blocks(80, start_tokens=64)) == [4]
    assert list(GroupHitRule(3, HitNeed.NONE, 16).required_blocks(80)) == []
    # A hit inside a block needs that block; a window then counts one more.
    assert list(prefix.required_blocks(72)) == [0, 1, 2, 3, 4]
    assert list(window.required_blocks(72)) == [1, 2, 3, 4]


@pytest.mark.parametrize("block_size", [4, 12, 16, 32])
@pytest.mark.parametrize("window", [1, 2, 16, 17, 32, 45, 65])
def test_window_blocks_match_token_interval(block_size, window):
    rule = GroupHitRule(0, HitNeed.WINDOW, block_size, window_tokens=window - 1)
    for end in range(1, 3 * block_size + window):
        # A token interval oracle, independent of the block endpoint formula.
        required = {
            token // block_size for token in range(max(0, end - window + 1), end)
        }
        # A partial current page is retained even for a one-token window.
        if end % block_size:
            required.add(end // block_size)
        assert set(rule.required_blocks(end)) == required


@pytest.mark.parametrize("probe_group", [0, 7])
@pytest.mark.parametrize("cap_blocks", [1, 2])
def test_prefix_probe_cap_preserves_eagle_state_lookup(probe_group, cap_blocks):
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], _full()),
            KVCacheGroupSpec(["state"], _mamba()),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=4096,
        hash_block_size=BS,
        enable_caching=True,
        use_eagle=True,
        cross_group_hit_check=True,
    )
    request = _request(4)
    _cache(manager, request, {0: {0, 1, 2, 3}, 1: {cap_blocks - 1}})
    before = manager.get_computed_blocks(request)[1]
    assert before == cap_blocks * BS
    manager.hit_contract.register_probe(
        probe_group,
        lambda g, i, h, bid: i < cap_blocks,
        block_size=BS,
    )
    assert manager.get_computed_blocks(request)[1] == before
    manager.hit_contract.register_probe(probe_group, lambda g, i, h, bid: False)
    assert manager.get_computed_blocks(request)[1] == 0


@pytest.mark.parametrize("group_id", [0, 1])
def test_probe_exception_is_a_miss(group_id):
    manager = _manager([_full(), _mamba()], check=True)
    request = _request(3)
    _cache(manager, request, {0: {0, 1, 2}, 1: {1}})

    def failed_probe(*args):
        raise OSError("host read failed")

    manager.hit_contract.register_probe(group_id, failed_probe)
    assert manager.get_computed_blocks(request)[1] == 0


def test_probe_only_group_requires_hash_grid():
    manager = _manager([_full(), _mamba()], check=True)
    with pytest.raises(ValueError, match="multiple of hash_block_size"):
        manager.hit_contract.register_probe(7, lambda *args: True, block_size=BS + 1)
    assert 7 not in manager.hit_contract.rules


@pytest.mark.parametrize("eagle", [False, True])
def test_probe_partial_last_page_keeps_earlier_state(eagle):
    """regression coverage. A 96-token cap also depends on the second 64-token host
    page."""
    manager = _manager([_full(), _mamba()], check=True)
    manager = KVCacheManager(
        manager.kv_cache_config,
        max_model_len=4096,
        hash_block_size=BS,
        enable_caching=True,
        use_eagle=eagle,
        cross_group_hit_check=True,
    )
    request = _request(6)
    _cache(manager, request, {0: set(range(6)), 1: {3, 4}})
    assert manager.get_computed_blocks(request)[1] == 80
    manager.hit_contract.register_probe(7, lambda g, i, h, b: i == 0, block_size=64)
    for _ in range(1024):
        assert manager.get_computed_blocks(request)[1] == 64
    manager.hit_contract.register_probe(7, lambda *args: False)
    assert manager.get_computed_blocks(request)[1] == 0


@pytest.mark.parametrize("block_size", [16, 32, 64, 96])
@pytest.mark.parametrize("start", [0, 16, 64])
def test_prefix_cap_checks_partial_page_against_token_oracle(block_size, start):
    contract = KVHitContract([GroupHitRule(7, HitNeed.PREFIX, block_size)], path="test")
    for end in range(start + 1, 193):
        for missing in range(193 // block_size):
            checked: list[int] = []

            def probe(
                group, index, block_hash, block_id, checked=checked, missing=missing
            ):
                checked.append(index)
                return index != missing

            contract.register_probe(7, probe)
            actual = contract.probe_cap(lambda rule, index: b"hash", start, end)
            required = {token // block_size for token in range(start, end)}
            expected = missing * block_size if missing in required else end
            assert actual == expected
            assert set(checked) <= required


def test_flag_reads_only_a_real_true():
    def cfg(value):
        return SimpleNamespace(kernel_config=SimpleNamespace(**value))

    assert not cross_group_hit_check_enabled(SimpleNamespace())
    assert not cross_group_hit_check_enabled(cfg({}))
    assert not cross_group_hit_check_enabled(cfg({"qsa_cross_group_hit_check": 1}))
    assert cross_group_hit_check_enabled(cfg({"qsa_cross_group_hit_check": True}))


def test_rules_follow_the_managers():
    manager = _manager([_full(), _ring(), _mamba(), _window()], check=True)
    assert manager.hit_contract is not None
    rules = manager.hit_contract.rules
    assert [rules[g].need for g in range(4)] == [
        HitNeed.PREFIX,
        HitNeed.NONE,
        HitNeed.STATE,
        HitNeed.WINDOW,
    ]
    assert rules[3].window_tokens == _window().sliding_window - 1
    assert _manager([_full(), _mamba()], check=False).hit_contract is None


# --- the coordinator already finds the common hit: no false rejection --------

SPEC_SETS = {
    "full+mamba": [_full(), _mamba()],
    "full+ring+2mamba+window": [_full(), _ring(), _mamba(), _mamba(), _window()],
    "full+window+window": [_full(), _window(), _window()],
}


@pytest.mark.parametrize("name", list(SPEC_SETS))
@pytest.mark.parametrize("seed", range(40))
def test_contract_agrees_with_coordinator(name: str, seed: int):
    """Random cache contents (blocks evicted per group independently): the
    reported hit is the oracle's common complete prefix, and the contract
    returns exactly the flag-off answer."""
    specs = SPEC_SETS[name]
    rng = random.Random(seed)
    num_blocks = 12
    present: dict[int, set[int]] = {}
    for group_id, spec in enumerate(specs):
        density = 0.95 if isinstance(spec, FullAttentionSpec) else 0.55
        present[group_id] = {i for i in range(num_blocks) if rng.random() < density}

    on = _manager(specs, check=True)
    off = _manager(specs, check=False)
    request = _request(num_blocks)
    for manager in (on, off):
        _cache(manager, request, present)
    blocks_on, hit_on = _hit(on, request)
    blocks_off, hit_off = _hit(off, request)

    assert hit_on == hit_off == _oracle(specs, present, num_blocks) * BS
    ids = [[b.block_id for b in g] for g in blocks_on]
    assert ids == [[b.block_id for b in g] for g in blocks_off]
    assert on.hit_contract is not None
    assert on.hit_contract.stats.rejected_hits == 0


# --- incomplete coordinator answers become misses -----------------------------


def _forced(manager: KVCacheManager, blocks, hit_tokens: int):
    manager.coordinator.find_longest_cache_hit = (  # type: ignore[method-assign]
        lambda block_hashes, max_len: (tuple(list(b) for b in blocks), hit_tokens)
    )


def _real(manager: KVCacheManager, n: int) -> list[KVCacheBlock]:
    blocks = manager.block_pool.get_new_blocks(n)
    for i, block in enumerate(blocks):
        block.block_hash = make_block_hash_with_group_id(bytes([i]), 0)
    return blocks


@pytest.mark.parametrize(
    "case",
    [
        "full_short",  # full attention misses its last block
        "full_null",  # a null block inside the full-attention prefix
        "state_missing",  # no Mamba state at the hit
        "window_hole",  # a hole in the sliding window before the hit
        "off_grid",  # hit not on the Mamba state grid
    ],
)
def test_incomplete_answer_is_a_miss(case: str):
    specs = [_full(), _mamba(), _window()]
    request = _request(8)
    for check in (False, True):
        manager = _manager(specs, check=check)
        null = manager.block_pool.null_block
        full = _real(manager, 4)
        state = [null, null, null, *_real(manager, 1)]
        window = [null, *_real(manager, 3)]
        hit = 4 * BS
        if case == "full_short":
            full = full[:3]
        elif case == "full_null":
            full[1] = null
        elif case == "state_missing":
            state[3] = null
        elif case == "window_hole":
            window[2] = null
        else:
            hit = 4 * BS - 1
        _forced(manager, (full, state, window), hit)
        blocks, num_tokens = _hit(manager, request)
        if check:
            assert num_tokens == 0 and blocks == [[], [], []]
            assert manager.hit_contract is not None
            assert manager.hit_contract.stats.rejected_hits == 1
        else:
            # Without the contract the incomplete answer is passed through.
            assert num_tokens == hit


def test_second_full_attention_spec_is_trimmed_to_the_hit():
    """Two full-attention specs (e.g. a split main KV / index pair) and a
    Mamba state that limits the hit. The coordinator truncates only its first
    full-attention spec group; the second keeps blocks past the hit."""
    specs = [_full(head_size=1), _full(head_size=2), _mamba()]
    request = _request(10)
    present = {0: set(range(10)), 1: set(range(10)), 2: {3}}
    for check in (False, True):
        manager = _manager(specs, check=check)
        _cache(manager, request, present)
        blocks, num_tokens = _hit(manager, request)
        assert num_tokens == 4 * BS
        assert len(blocks[0]) == 4
        if check:
            assert len(blocks[1]) == 4
        else:
            assert len(blocks[1]) == 10  # existing behaviour, see report


# --- residency probes (Active Host KV history pool hook) ----------------------


def test_prefix_probe_caps_hit_to_common_prefix():
    specs = [_full(), _mamba(), _window()]
    request = _request(10)
    present = {0: set(range(10)), 1: {1, 3, 6}, 2: set(range(10))}
    manager = _manager(specs, check=True)
    _cache(manager, request, present)
    assert _hit(manager, request)[1] == 7 * BS
    contract = manager.hit_contract
    assert contract is not None

    calls = []
    evicted = request.block_hashes[5]

    def probe(group_id, index, block_hash, block_id):
        calls.append((group_id, index, block_id is not None))
        return block_hash != evicted

    contract.register_probe(0, probe)
    blocks, num_tokens = _hit(manager, request)
    # Block 5 is gone from the probed store: the largest common hit below it
    # is the Mamba state ending at block 4.
    assert num_tokens == 4 * BS
    assert len(blocks[0]) == 4
    assert contract.stats.probe_capped == 1
    assert contract.stats.rejected_hits == 0
    # The cap pass passes no GPU block ID; the final check does.
    assert (0, 5, False) in calls
    assert (0, 3, True) in calls


def test_state_probe_failure_is_a_miss():
    specs = [_full(), _mamba()]
    request = _request(10)
    manager = _manager(specs, check=True)
    _cache(manager, request, {0: set(range(10)), 1: {3, 6}})
    contract = manager.hit_contract
    assert contract is not None
    contract.register_probe(1, lambda g, i, h, b: i != 6)
    assert _hit(manager, request)[1] == 0
    assert contract.stats.rejected_hits == 1


def test_probe_registration_rules():
    manager = _manager([_full(), _ring()], check=True)
    contract = manager.hit_contract
    assert contract is not None
    with pytest.raises(ValueError):
        contract.register_probe(1, lambda g, i, h, b: True)  # scratch ring
    with pytest.raises(ValueError):
        contract.register_probe(7, lambda g, i, h, b: True)  # unknown, no size
    contract.register_probe(7, lambda g, i, h, b: True, block_size=BS)
    assert contract.rules[7].need is HitNeed.PREFIX
