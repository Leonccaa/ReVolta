# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json

import pytest

import vllm.revolta_profile as profile
from vllm.config import DeviceConfig, KernelConfig, VllmConfig


@pytest.mark.parametrize(
    "explicit,enabled,expected,source",
    [
        (None, True, False, "profile"),
        (True, True, True, "explicit"),
        (False, True, False, "explicit"),
        (None, False, True, "upstream_default"),
    ],
)
def test_cascade_profile_preserves_explicit_choices(
    explicit, enabled, expected, source
):
    kernel = KernelConfig(
        **({} if explicit is None else {"ple_disk_cascade": explicit})
    )
    profile.apply_revolta_kernel_profile(kernel, enabled=enabled)
    assert kernel.ple_disk_cascade is expected
    audit = json.loads(profile.profile_audit(enabled=enabled, kernel_config=kernel))
    assert len(audit["values"]) == 45
    assert "ple_disk_cascade" not in audit["values"]
    assert audit["kernel_config"]["ple_disk_cascade"] == {
        "value": expected,
        "source": source,
    }
    clone = copy.deepcopy(kernel)
    profile.apply_revolta_kernel_profile(clone, enabled=enabled)
    assert (
        json.loads(profile.profile_audit(enabled=enabled, kernel_config=clone)) == audit
    )


def test_cascade_provenance_does_not_change_kernel_hash():
    automatic = KernelConfig()
    profile.apply_revolta_kernel_profile(automatic, enabled=True)
    explicit = KernelConfig(ple_disk_cascade=False)
    assert automatic.compute_hash() == explicit.compute_hash()


def test_vllmconfig_applies_kernel_defaults_at_profile_entry(monkeypatch):
    monkeypatch.setattr(profile, "apply_revolta_profile", lambda *a, **k: True)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    assert config.kernel_config.ple_disk_cascade is False
    assert config.kernel_config.ple_disk_cascade_source == "profile"


@pytest.mark.parametrize("value", [False, True])
def test_programmatic_choice_before_profile_is_explicit(value):
    kernel = KernelConfig()
    kernel.ple_disk_cascade = value
    profile.apply_revolta_kernel_profile(kernel, enabled=True)
    assert kernel.ple_disk_cascade is value
    assert kernel.ple_disk_cascade_source == "explicit"


def test_changed_resolved_profile_value_is_explicit():
    kernel = KernelConfig()
    profile.apply_revolta_kernel_profile(kernel, enabled=True)
    kernel.ple_disk_cascade = True
    profile.apply_revolta_kernel_profile(kernel, enabled=True)
    assert kernel.ple_disk_cascade is True
    assert kernel.ple_disk_cascade_source == "explicit"
