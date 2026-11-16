# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SM70_GDN_CPU_SPEC_QSL must not change any GDN spec metadata value."""

import random

import pytest
import torch

from vllm.v1.worker.gpu import attn_utils


def _batch(rng: random.Random, num_reqs: int, k: int):
    lens, drafts = [], []
    for _ in range(num_reqs):
        kind = rng.choice(["spec", "prefill", "decode", "empty"])
        if kind == "spec":
            lens.append(k + 1)
            drafts.append(rng.randint(0, k))
        elif kind == "prefill":
            lens.append(rng.randint(2, 4000))
            drafts.append(-1)
        elif kind == "decode":
            lens.append(1)
            drafts.append(-1)
        else:
            lens.append(0)
            drafts.append(-1)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0, out=qsl[1:])
    return qsl, torch.tensor(drafts, dtype=torch.int32)


def _run(monkeypatch, flag, qsl, drafts, k, legacy):
    monkeypatch.setattr(attn_utils, "_GDN_CPU_SPEC_QUERY_START_LOC", flag)
    return attn_utils.compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=drafts,
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl.clone(),
        num_spec_state_tokens=k,
        legacy_mixed_decode_routing=legacy,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_cpu_offsets_match_gpu_path(monkeypatch, legacy):
    rng = random.Random(1234)
    checked = 0
    for _ in range(400):
        k = rng.choice([3, 4, 7])
        qsl, drafts = _batch(rng, rng.randint(1, 8), k)
        ref = _run(monkeypatch, False, qsl, drafts, k, legacy)
        new = _run(monkeypatch, True, qsl, drafts, k, legacy)
        assert (ref is None) == (new is None)
        if ref is None:
            continue
        for field in ref.__dataclass_fields__:
            a, b = getattr(ref, field), getattr(new, field)
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                assert (a is None) == (b is None), field
                if a is not None:
                    assert a.dtype == b.dtype, field
                    assert torch.equal(a, b), field
            else:
                assert a == b, field
        checked += 1
    assert checked > 100


def test_check_mode_counts_and_matches(monkeypatch, tmp_path):
    flag = tmp_path / "check.on"
    flag.touch()
    monkeypatch.setattr(attn_utils, "_GDN_CPU_SPEC_QSL_CHECK_FILE", str(flag))
    monkeypatch.setattr(
        attn_utils, "_GDN_CPU_SPEC_QSL_CHECK_STATE", {"checked": 0, "mismatched": 0}
    )
    rng = random.Random(7)
    for _ in range(200):
        k = 7
        qsl, drafts = _batch(rng, rng.randint(2, 8), k)
        _run(monkeypatch, True, qsl, drafts, k, False)
    state = attn_utils._GDN_CPU_SPEC_QSL_CHECK_STATE
    assert state["checked"] > 50
    assert state["mismatched"] == 0


def test_check_mode_reports_a_mismatch(monkeypatch):
    monkeypatch.setattr(
        attn_utils, "_GDN_CPU_SPEC_QSL_CHECK_STATE", {"checked": 0, "mismatched": 0}
    )
    lens = torch.tensor([8, 100, 8], dtype=torch.int32)
    masks = torch.tensor([True, False, True])
    good_spec = torch.tensor([0, 8, 16], dtype=torch.int32)
    bad_non_spec = torch.tensor([0, 99], dtype=torch.int32)
    attn_utils._check_cpu_spec_qsl(lens, masks, good_spec, bad_non_spec)
    assert attn_utils._GDN_CPU_SPEC_QSL_CHECK_STATE == {"checked": 1, "mismatched": 1}
