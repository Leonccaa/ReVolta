# Building ReVolta

This page explains how to build the ReVolta wheel and Docker image from a source checkout. Most users do not need it: install the PyPI wheel or pull the release image as described in [Getting started](docs/getting-started.md).

## Requirements

- Linux x86-64 with Docker BuildKit. No GPU is needed to compile.
- Memory for the native compile. In the 0.1.0 build, 10 parallel compile jobs used about 21 GB of RAM. Four jobs with one NVCC thread per job is a conservative start on a 32 GB host.

`docker/Dockerfile` pins the toolchain: CUDA 12.8.1, Python 3.12, Ubuntu 22.04 and PyTorch 2.10.0 for CUDA 12.8. It builds for SM70 only (`torch_cuda_arch_list=7.0`, the default). CUDA 13 cannot compile SM70. The Dockerfile is the only tested build environment.

## Package name and version

The distribution is `revolta`; the Python package and the CLI remain `vllm`.

`get_vllm_version()` in `setup.py` sets the version from the first of these that is present:

1. `VLLM_VERSION_OVERRIDE`, used exactly as given, with no suffix.
2. `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REVOLTA` or `SETUPTOOLS_SCM_PRETEND_VERSION`.
3. The fallback version in `get_vllm_version()` (`0.2.0`).

Versions from 2 and 3 get a local suffix (`+…`, or `.…` if the version already has a local part):

- A CUDA build that compiles the extensions itself appends `cu128`, because CUDA 12.8 differs from the main CUDA version `VLLM_MAIN_CUDA_VERSION` (13.0). An `sdist` build omits it.
- A build with precompiled extensions appends `precompiled`, unless `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1`.

Git tags are not used: the repository carries tags inherited from vLLM, so `setuptools-scm` never chooses the version. Always pass the version you want.

In the Docker build, `--build-arg SETUPTOOLS_SCM_PRETEND_VERSION=<version>` sets both `SETUPTOOLS_SCM_PRETEND_VERSION` and `VLLM_VERSION_OVERRIDE`, so the wheel metadata and `vllm --version` report exactly that version. `--build-arg VLLM_BUILD_COMMIT=<commit>` records the commit in `vllm/_version.py` and in the image labels. For a private build, use a PEP 440 local version, for example `0.2.0+local.20261127.g15c5d32a87`.

## Build the wheel and the image

Build from a clean checkout. The build reads no Git metadata when the version and commit are passed as below, so a linked worktree or an exported source tree also works; the `build` stage still bind-mounts `.git`, so a `.git` entry must exist in the build context. In an exported tree, which has no Git history, set `commit` to the known commit ID instead of running `git rev-parse`.

```bash
version=0.2.0
commit=$(git rev-parse HEAD)
args=(-f docker/Dockerfile
  --build-arg torch_cuda_arch_list=7.0
  --build-arg max_jobs=4 --build-arg nvcc_threads=1
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="$version"
  --build-arg VLLM_BUILD_COMMIT="$commit")
docker build "${args[@]}" --target build -t "local/revolta-build:$version" .
container=$(docker create "local/revolta-build:$version")
mkdir -p artifacts
docker cp "$container:/workspace/dist" artifacts/
docker cp "$container:/workspace/debug" artifacts/
docker cp "$container:/workspace/unstripped" artifacts/
docker rm "$container"
docker build "${args[@]}" --target vllm-openai \
  --build-arg VLLM_IMAGE_TAG="local/revolta:$version" \
  -t "local/revolta:$version" .
```

The `build` stage produces:

| Directory | Contents |
| --- | --- |
| `dist/` | The runtime wheel, with debug information split off |
| `debug/` | `<wheel>.debug.tar.xz` with the `*.so.debug` files, and `<wheel>.sections.json` |
| `unstripped/` | The same wheel before splitting |

The `vllm-openai` target installs `dist/` into the runtime image, with `vllm serve` as the entry point. `vllm-openai-nonroot` is the same image running as the `vllm` user.

Flash-V100 and FlashQLA are bundled into the ReVolta wheel. DeepEP is skipped on SM70. The build fails if the compressed wheel is larger than `VLLM_MAX_SIZE_MB` (95 MiB by default, below the 100 MB PyPI limit); `--build-arg RUN_WHEEL_CHECK=false` skips this check.

GGUF is unsupported. The CMake option `VLLM_BUILD_GGUF` is `OFF` by default, so `_C_gguf` is not built and release wheels cannot run GGUF dense or MoE models or dequantize through `_C_gguf`. For development only, set `VLLM_BUILD_GGUF=ON` or pass `CMAKE_ARGS=-DVLLM_BUILD_GGUF=ON`. GGUF modules load that extension lazily; normal imports, AWQ and NVFP4 do not need it.

## Debug symbols

The `build` stage saves the unstripped wheel, then runs `tools/wheel_debug.py strip`. For each shared library, the tool runs `objcopy --only-keep-debug`, `strip --strip-debug` and `objcopy --add-gnu-debuglink`, and keeps `.symtab`. Before it replaces the wheel, it checks that `.text`, `.nv_fatbin`, `.rodata`, `.dynsym` and `.dynstr` are unchanged, and it regenerates the wheel's `RECORD` hashes. The JSON manifest records each library's section hashes, with `null` for a section the library does not have (host-only libraries can lack `.nv_fatbin`).

To use the symbols, extract the debug archive next to the installed libraries and move each `*.so.debug` into a `.debug/` subdirectory next to its `.so`, or point GDB's debug-file search path at them. Debug symbols are not part of the runtime image.

To check one library pair from the same build (needs GNU binutils):

```bash
uv venv --python 3.12
.venv/bin/python tools/wheel_debug.py verify original.so stripped.so
```

The check shows that the runtime sections are byte-identical. It does not replace GPU service, output and performance tests.

## See also

- [Getting started](docs/getting-started.md): install the release wheel or image.
- [Contributing](CONTRIBUTING.md): pull requests and required checks.
- [Support and known limits](docs/known-limits.md): what the SM70 build covers.
