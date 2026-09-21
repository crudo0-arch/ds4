#include "cuda_stream_cache_test.h"
#include <assert.h>
#include <stdio.h>

static ds4_cuda_stream_cache_test_snapshot snapshot_current(void) {
    ds4_cuda_stream_cache_test_snapshot s;
    assert(ds4_cuda_test_stream_cache_snapshot_current(&s));
    return s;
}

int main(void) {
    /* Device ids deliberately differ from tier ids. This catches routing that
     * returns the CUDA device ordinal directly or always selects tier zero. */
    ds4_cuda_test_stream_cache_set_tier_device(0, 7);
    ds4_cuda_test_stream_cache_set_tier_device(1, 3);

    ds4_cuda_test_stream_cache_set_device(7);
    ds4_cuda_test_stream_cache_teardown_current();
    ds4_cuda_test_stream_cache_seed_current(11, 101, 1001, 0x1000, 1);
    assert(ds4_cuda_test_stream_prefetch_owner_tier() == 0);

    /* Releasing a never-initialized tier 1 must not follow its zero-filled
     * selected-cache logical_tier back to tier 0 and clear tier 0. */
    ds4_cuda_test_stream_cache_set_device(3);
    ds4_cuda_test_stream_cache_release_current();
    ds4_cuda_test_stream_cache_set_device(7);
    ds4_cuda_stream_cache_test_snapshot untouched = snapshot_current();
    assert(untouched.budget == 11 && untouched.expert_bytes == 101);
    assert(untouched.slot_count == 1 && untouched.map_count == 1);
    /* Worker lookup must remain tier 0 even when its thread-local device
     * would route ambient state to tier 1. */
    ds4_cuda_test_stream_cache_set_device(3);
    assert(ds4_cuda_test_stream_prefetch_worker_state_tier(0) == 0);

    ds4_cuda_test_stream_cache_set_device(3);
    ds4_cuda_test_stream_cache_teardown_current();
    ds4_cuda_test_stream_cache_seed_current(22, 202, 2002, 0x2000, 0);
    assert(ds4_cuda_test_stream_prefetch_owner_tier() == 1);
    ds4_cuda_test_stream_cache_set_device(7);
    assert(ds4_cuda_test_stream_prefetch_worker_state_tier(1) == 1);

    ds4_cuda_test_stream_cache_set_device(3);
    ds4_cuda_stream_cache_test_snapshot b = snapshot_current();
    assert(b.budget == 22 && b.expert_bytes == 202 && b.clock == 2002);
    assert(b.slot_count == 1 && b.map_count == 1);
    assert(b.prefill_id_count == 1 && b.prefill_slot_count == 1);
    assert(b.selected_valid == 1 && b.selected_logical_tier == 1);
    assert(b.selected_stage_bytes == 209 && b.prefetch_active == 0);
    assert(b.prefetch_owner_tier == 1 && b.prefetch_slot_count == 1);

    ds4_cuda_test_stream_cache_set_device(7);
    ds4_cuda_stream_cache_test_snapshot a = snapshot_current();
    assert(a.budget == 11 && a.expert_bytes == 101 && a.clock == 1001);
    assert(a.slot_count == 1 && a.map_count == 1);
    assert(a.prefill_id_count == 1 && a.prefill_slot_count == 1);
    assert(a.selected_valid == 1 && a.selected_logical_tier == 0);
    assert(a.selected_stage_bytes == 108 && a.prefetch_active == 1);
    assert(a.prefetch_owner_tier == 0 && a.prefetch_slot_count == 1);
    assert(ds4_cuda_test_stream_prefetch_owner_tier() == 0);

    ds4_cuda_test_stream_cache_teardown_current();
    a = snapshot_current();
    assert(a.budget == 0 && a.expert_bytes == 0 && a.clock == 1);
    assert(a.slot_count == 0 && a.map_count == 0);
    assert(a.prefill_id_count == 0 && a.prefill_slot_count == 0);
    assert(a.selected_valid == 0 && a.selected_logical_tier == -1);
    assert(a.selected_stage_bytes == 0 && a.prefetch_active == 0);

    ds4_cuda_test_stream_cache_set_device(3);
    b = snapshot_current();
    assert(b.budget == 22 && b.expert_bytes == 202 && b.clock == 2002);
    assert(b.slot_count == 1 && b.map_count == 1);
    assert(b.selected_valid == 1 && b.selected_logical_tier == 1);
    assert(b.selected_stage_bytes == 209 && b.prefetch_active == 0);
    ds4_cuda_test_stream_cache_teardown_current();

    /* atexit cleanup must visit every configured tier, not only the tier
     * selected by the caller's current device. */
    ds4_cuda_test_stream_cache_set_device(7);
    ds4_cuda_test_stream_cache_seed_current(11, 101, 1001, 0x3000, 0);
    ds4_cuda_test_stream_cache_set_device(3);
    ds4_cuda_test_stream_cache_seed_current(22, 202, 2002, 0x4000, 0);
    ds4_cuda_test_stream_cache_set_device(7);
    ds4_cuda_test_stream_prefetch_cleanup_all();
    a = snapshot_current();
    assert(a.prefetch_slot_count == 0);
    ds4_cuda_test_stream_cache_set_device(3);
    b = snapshot_current();
    assert(b.prefetch_slot_count == 0);

    puts("cuda stream cache production routing and ownership: OK");
    return 0;
}
