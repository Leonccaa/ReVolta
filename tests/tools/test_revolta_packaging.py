# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for distribution identity and lossless debug splitting."""

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import regex as re

ROOT = Path(os.getenv("REVOLTA_SOURCE_ROOT") or Path(__file__).resolve().parents[2])


def load_setup_function(name, **scope):
    tree = ast.parse((ROOT / "setup.py").read_text())
    nodes: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert nodes, f"setup.py is missing {name}"
    exec(compile(ast.Module(nodes, type_ignores=[]), "setup.py", "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("override", [None, "0.1.0+local.20261018.g0123456789"])
def test_release_version_and_generated_stamp_agree(monkeypatch, override):
    from setuptools_scm import get_version

    monkeypatch.delenv("SETUPTOOLS_SCM_PRETEND_VERSION", raising=False)
    monkeypatch.delenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REVOLTA", raising=False)
    monkeypatch.delenv("VLLM_VERSION_OVERRIDE", raising=False)
    if override:
        monkeypatch.setenv("VLLM_VERSION_OVERRIDE", override)
    stamps = []

    def stamp(**kwargs):
        stamps.append(os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION"))
        # Use real setuptools-scm normalization without writing to the checkout.
        kwargs.pop("write_to", None)
        return get_version(**kwargs)

    fn = load_setup_function(
        "get_vllm_version",
        os=os,
        sys=sys,
        RELEASE_VERSION="0.1.0",
        get_version=stamp,
        envs=SimpleNamespace(VLLM_TARGET_DEVICE="empty"),
        _no_device=lambda: True,
    )
    actual = fn()
    assert actual == (override or "0.1.0+empty")
    assert stamps[-1] == actual


def test_distribution_name_and_command():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "revolta"
    assert project["scripts"]["vllm"] == "vllm.entrypoints.cli.main:main"


def test_docker_explicit_release_version_is_exact():
    source = (ROOT / "docker/Dockerfile").read_text()
    assert 'export VLLM_VERSION_OVERRIDE="${SETUPTOOLS_SCM_PRETEND_VERSION}"' in source


def test_sm70_image_skips_unsupported_cubin_prefetch():
    source = (ROOT / "docker/Dockerfile").read_text()
    assert "Skipping unsupported FlashInfer cubin prefetch for SM70" in source


def test_metadata_prefers_revolta():
    from vllm import version

    with patch(
        "vllm.version.distribution", return_value=object(), create=True
    ) as lookup:
        assert version.get_distribution_name() == "revolta"
        lookup.assert_called_once_with("revolta")


def test_metric_compatibility_survives_version_reset():
    from vllm import version

    with patch("vllm.version.__version_tuple__", (0, 1, 0)):
        assert version._prev_minor_version_was("1.4")
        assert not version._prev_minor_version_was("0.0")
        assert version._prev_minor_version() == "1.4"


def test_gguf_target_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_BUILD_GGUF", raising=False)
    monkeypatch.delenv("CMAKE_ARGS", raising=False)
    fn = load_setup_function("build_gguf", os=os, re=re)
    assert not fn()
    monkeypatch.setenv("VLLM_BUILD_GGUF", "ON")
    assert fn()
    monkeypatch.setenv("CMAKE_ARGS", "-DVLLM_BUILD_GGUF:BOOL=OFF")
    assert not fn()
    source = (ROOT / "CMakeLists.txt").read_text()
    assert (
        'option(VLLM_BUILD_GGUF "Build the unsupported GGUF extension" OFF)' in source
    )


def test_gguf_missing_does_not_break_normal_imports():
    # Fail even if a base image happens to contain the optional extension.
    code = """
import sys
class MissingGGUF:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm._C_gguf":
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingGGUF())
import vllm
import vllm.entrypoints.cli.main
from vllm.model_executor.layers.quantization.gguf_native import native_available
assert not native_available()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_split_debug_preserves_sections_and_record(tmp_path):
    from wheel.wheelfile import WheelFile

    path = ROOT / "tools/wheel_debug.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("wheel_debug", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "example.c"
    source.write_text('const char *message = "hello"; int answer(void) {return 42;}\n')
    library = tmp_path / "original.so"
    subprocess.run(["cc", "-g", "-shared", "-fPIC", source, "-o", library], check=True)
    # Include a fatbin-shaped section so all five required sections are exercised.
    fatbin = tmp_path / "fatbin"
    fatbin.write_bytes(b"cuda section fixture")
    subprocess.run(
        ["objcopy", "--add-section", f".nv_fatbin={fatbin}", library], check=True
    )
    unpacked = tmp_path / "unpacked"
    (unpacked / "vllm").mkdir(parents=True)
    executable = unpacked / "vllm/vllm-rs"
    executable.write_bytes(b"executable fixture")
    executable.chmod(0o755)
    import shutil

    shutil.copy2(library, unpacked / "vllm/example.so")
    metadata = unpacked / "revolta-0.1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: revolta\nVersion: 0.1.0\n"
    )
    (metadata / "WHEEL").write_text(
        "Wheel-Version: 1.0\nTag: cp312-cp312-linux_x86_64\n"
    )
    wheel = tmp_path / "revolta-0.1.0-cp312-cp312-linux_x86_64.whl"
    with WheelFile(wheel, "w") as output:
        output.write_files(unpacked)
    report = module.strip_wheel(wheel, tmp_path / "debug")
    assert all(report["libraries"]["vllm/example.so"].values())
    # WheelFile validates every member against the regenerated RECORD.
    with WheelFile(wheel) as result:
        assert (result.getinfo("vllm/vllm-rs").external_attr >> 16) & 0o111
        result.extractall(tmp_path / "stripped")
    stripped = tmp_path / "stripped/vllm/example.so"
    module.verify(library, stripped)
    subprocess.run(["objcopy", "--remove-section", ".nv_fatbin", stripped], check=True)
    with pytest.raises(ValueError, match="Runtime sections changed"):
        module.verify(library, stripped)
