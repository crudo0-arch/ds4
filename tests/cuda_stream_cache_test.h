#ifndef DS4_CUDA_STREAM_CACHE_TEST_H
#define DS4_CUDA_STREAM_CACHE_TEST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_cuda_stream_cache_test_snapshot {
    uint32_t budget;
    uint64_t expert_bytes;
    uint64_t clock;
    uint32_t slot_count;
    uint32_t map_count;
    uint32_t prefill_id_count;
    uint32_t prefill_slot_count;
    int selected_valid;
    int selected_logical_tier;
    uint64_t selected_stage_bytes;
    int prefetch_active;
    int prefetch_owner_tier;
    uint32_t prefetch_slot_count;
} ds4_cuda_stream_cache_test_snapshot;

void ds4_cuda_test_stream_cache_set_device(int device);
void ds4_cuda_test_stream_cache_set_tier_device(int tier, int device);
void ds4_cuda_test_stream_cache_seed_current(uint32_t budget,
        uint64_t expert_bytes, uint64_t clock, uint64_t gate_key,
        int prefetch_active);
int ds4_cuda_test_stream_cache_snapshot_current(
        ds4_cuda_stream_cache_test_snapshot *out);
void ds4_cuda_test_stream_cache_teardown_current(void);
void ds4_cuda_test_stream_cache_release_current(void);
int ds4_cuda_test_stream_prefetch_owner_tier(void);
int ds4_cuda_test_stream_prefetch_worker_state_tier(int prefetch_state_tier);
void ds4_cuda_test_stream_prefetch_cleanup_all(void);

#ifdef __cplusplus
}
#endif

#endif
