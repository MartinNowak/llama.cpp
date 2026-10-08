#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// device buffer for attention KV tensors over a virtual memory management
// reservation: stable addresses, granularity-aligned tensors, fully backed.
// Reachable through ggml_backend_reg_get_proc_address as
// "ggml_backend_cuda_paged_kv_buffer_type"; NULL where the device carries no
// VMM support or the build excludes it.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_paged_kv_buffer_type(int device);

// the same reservation with nothing backed at allocation: the owner declares
// the buffer-relative byte intervals it requires through
// ggml_backend_cuda_paged_kv_require, which commits every unit they touch
// and, where reclaim is set, releases every backed unit outside them after
// the device quiesces. Reachable as "ggml_backend_cuda_paged_kv_sparse_buffer_type"
// and "ggml_backend_cuda_paged_kv_require".
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_paged_kv_sparse_buffer_type(int device);
GGML_BACKEND_API bool ggml_backend_cuda_paged_kv_require(ggml_backend_buffer_t buffer, const size_t * offsets, const size_t * sizes, size_t n_ranges, bool reclaim);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
