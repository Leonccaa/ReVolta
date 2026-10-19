// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// RN division / RNE encoding contract: 1CatAI/1Cat-vLLM #1184.
// Separate translation unit keeps the legacy writer code generation unchanged.
#include "torch_utils.h"
#include "dispatch_utils.h"
#include "../cuda_compat.h"

namespace {
struct RowLayout {
  int64_t block, token, head;
  __device__ int64_t offset(int64_t b, int64_t t, int64_t h) const {
    return b * block + t * token + h * head;
  }
};

template <typename T>
__global__ void cache_int8_kernel(const T* key, const T* value, int8_t* kc,
                                  int8_t* vc, float* ks, float* vs,
                                  const int64_t* slots, int64_t rows,
                                  int64_t pages, int heads, int dim, int tokens,
                                  RowLayout ki, RowLayout vi, RowLayout ko,
                                  RowLayout vo, RowLayout kso, RowLayout vso) {
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= rows) return;
  const int64_t slot = slots[row];
  if (slot < 0 || slot >= pages * tokens) return;
  const int64_t page = slot / tokens, token = slot % tokens;
  const T* src = blockIdx.z == 0 ? key : value;
  RowLayout input = blockIdx.z == 0 ? ki : vi;
  const int64_t in = input.offset(row, 0, head);
  float maximum = 0.f;
  int invalid = 0;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float x = static_cast<float>(src[in + d]);
    invalid |= !isfinite(x);
    maximum = fmaxf(maximum, fabsf(x));
  }
  __shared__ float maxima[128];
  __shared__ int bad[128];
  maxima[threadIdx.x] = maximum;
  bad[threadIdx.x] = invalid;
  __syncthreads();
  for (int s = 64; s; s >>= 1) {
    if (threadIdx.x < s) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + s]);
      bad[threadIdx.x] |= bad[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float scale = bad[0] ? __int_as_float(0x7fc00000)
                             : fmaxf(__fdiv_rn(maxima[0], 127.f), 1e-6f);
  int8_t* dst = blockIdx.z == 0 ? kc : vc;
  RowLayout output = blockIdx.z == 0 ? ko : vo;
  RowLayout scales = blockIdx.z == 0 ? kso : vso;
  float* sp = blockIdx.z == 0 ? ks : vs;
  if (threadIdx.x == 0) sp[scales.offset(page, token, head)] = scale;
  const int64_t out = output.offset(page, token, head);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float x = static_cast<float>(src[in + d]);
    const float q = bad[0] ? 0.f : __fdiv_rn(x, scale);
    dst[out + d] =
        static_cast<int8_t>(__float2int_rn(fminf(fmaxf(q, -128.f), 127.f)));
  }
}

RowLayout layout(const torch::stable::Tensor& tensor) {
  return {tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}
}  // namespace

void reshape_and_cache_flash_int8(torch::stable::Tensor& key,
                                  torch::stable::Tensor& value,
                                  torch::stable::Tensor& key_cache,
                                  torch::stable::Tensor& value_cache,
                                  torch::stable::Tensor& slot_mapping,
                                  torch::stable::Tensor& k_scale_cache,
                                  torch::stable::Tensor& v_scale_cache) {
  using ST = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(key.dim() == 3 && value.sizes().equals(key.sizes()));
  STD_TORCH_CHECK(key.scalar_type() == value.scalar_type());
  STD_TORCH_CHECK(key_cache.dim() == 4 &&
                  value_cache.sizes().equals(key_cache.sizes()));
  STD_TORCH_CHECK(key_cache.scalar_type() == ST::Char &&
                  value_cache.scalar_type() == ST::Char);
  STD_TORCH_CHECK(k_scale_cache.scalar_type() == ST::Float &&
                  v_scale_cache.scalar_type() == ST::Float);
  STD_TORCH_CHECK(k_scale_cache.dim() == 3 &&
                  v_scale_cache.sizes().equals(k_scale_cache.sizes()));
  for (int i = 0; i < 3; ++i) {
    STD_TORCH_CHECK(k_scale_cache.size(i) == key_cache.size(i));
  }
  STD_TORCH_CHECK(key.size(1) == key_cache.size(2) &&
                  key.size(2) == key_cache.size(3));
  STD_TORCH_CHECK(key.size(2) > 0 && key.size(1) > 0 && key_cache.size(1) > 0);
  STD_TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1 &&
                  key_cache.stride(3) == 1 && value_cache.stride(3) == 1);
  STD_TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.stride(0) == 1 &&
                  slot_mapping.scalar_type() == ST::Long &&
                  slot_mapping.size(0) <= key.size(0));
  for (const auto* t : {&key, &value, &key_cache, &value_cache, &slot_mapping,
                        &k_scale_cache, &v_scale_cache}) {
    STD_TORCH_CHECK(t->device().is_cuda() &&
                    t->get_device_index() == key.get_device_index());
  }
  if (slot_mapping.size(0) == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const dim3 grid(slot_mapping.size(0), key.size(1), 2);
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(key.scalar_type(), "cache_int8", [&] {
    cache_int8_kernel<scalar_t><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(key.data_ptr()),
        reinterpret_cast<const scalar_t*>(value.data_ptr()),
        reinterpret_cast<int8_t*>(key_cache.data_ptr()),
        reinterpret_cast<int8_t*>(value_cache.data_ptr()),
        reinterpret_cast<float*>(k_scale_cache.data_ptr()),
        reinterpret_cast<float*>(v_scale_cache.data_ptr()),
        slot_mapping.const_data_ptr<int64_t>(), slot_mapping.size(0),
        key_cache.size(0), key.size(1), key.size(2), key_cache.size(1),
        RowLayout{key.stride(0), 0, key.stride(1)},
        RowLayout{value.stride(0), 0, value.stride(1)}, layout(key_cache),
        layout(value_cache), layout(k_scale_cache), layout(v_scale_cache));
  });
}
