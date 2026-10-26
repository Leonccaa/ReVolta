# Building ReVolta

The distribution is `revolta`; Python imports and the CLI remain `vllm`.
The release version is `RELEASE_VERSION` in `setup.py` (initially `0.1.0`).
Set `VLLM_VERSION_OVERRIDE` for an exact wheel and runtime version, or pass
`SETUPTOOLS_SCM_PRETEND_VERSION` to the Docker build. Inherited Git tags do not
select the ReVolta version. Use a PEP 440 local version for private deployments,
for example `0.1.3+deployment.20261025.g0123456789`.

## SM70 source build

Build from a clean checkout with Docker BuildKit. No GPU is needed to compile.
Use CUDA 12.8; CUDA 13 cannot compile SM70. Limit parallelism to available memory:
10 native compile jobs have used approximately 21 GB of RAM. Four jobs and one
NVCC thread per job are a conservative starting point on a 32 GB host.

```bash
version=0.1.3
docker build --target build -f docker/Dockerfile \
  --build-arg torch_cuda_arch_list=7.0 \
  --build-arg max_jobs=4 --build-arg nvcc_threads=1 \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="$version" \
  -t "local/revolta-build:$version" .
container=$(docker create "local/revolta-build:$version")
mkdir -p artifacts
docker cp "$container:/workspace/dist" artifacts/
docker cp "$container:/workspace/debug" artifacts/
docker cp "$container:/workspace/unstripped" artifacts/
docker rm "$container"
docker build --target vllm-openai -f docker/Dockerfile \
  --build-arg torch_cuda_arch_list=7.0 \
  --build-arg max_jobs=4 --build-arg nvcc_threads=1 \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="$version" \
  -t "local/revolta:$version" .
```

Flash-V100 and FlashQLA are bundled into the ReVolta wheel. DeepEP is skipped
on SM70. GGUF is unsupported: `_C_gguf` is disabled by the default CMake option
`VLLM_BUILD_GGUF=OFF`. For development only, set `VLLM_BUILD_GGUF=ON` or pass
`CMAKE_ARGS=-DVLLM_BUILD_GGUF=ON` to build it. GGUF modules load that extension
lazily; normal imports, AWQ and NVFP4 do not require it. Native GGUF dense/MoE
execution and dequantization through `_C_gguf` are unavailable in release wheels;
remaining legacy GGUF code is outside the supported release scope.

## Debug symbols and verification

The Docker wheel stage saves an unstripped wheel, then runs
`tools/wheel_debug.py strip`. It uses `objcopy --only-keep-debug`,
`strip --strip-debug`, and `objcopy --add-gnu-debuglink`, retaining `.symtab`.
It verifies `.text`, `.nv_fatbin`, `.rodata`, `.dynsym`, and `.dynstr` before
replacing the wheel, and regenerates wheel `RECORD` hashes. The JSON manifest
records missing sections explicitly (host libraries can lack `.nv_fatbin`).
The default compressed wheel limit is 95 MiB, below 100 decimal MB.

Extract the debug archive beside the installed libraries, then move each
`*.so.debug` into a `.debug/` subdirectory next to its corresponding `.so`, or
configure GDB's debug-file search path. Debug packages are separate from the
runtime image. For an independent same-build pair check:

```bash
uv venv --python 3.12
.venv/bin/python tools/wheel_debug.py verify original.so stripped.so
```

The section check proves preservation of runtime section bytes. GPU service,
output-equivalence and performance tests still require a separate GPU window.
