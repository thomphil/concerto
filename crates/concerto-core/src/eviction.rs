use crate::types::*;

/// Determine which models to evict from a GPU to free enough space.
///
/// Returns the list of model IDs to evict, ordered by eviction priority.
/// Returns None if evicting all models still wouldn't free enough space.
pub fn select_evictions(
    gpu: &GpuState,
    space_needed: bytesize::ByteSize,
    headroom: bytesize::ByteSize,
    policy: EvictionPolicy,
) -> Option<Vec<ModelId>> {
    let total_needed = bytesize::ByteSize::b(space_needed.as_u64() + headroom.as_u64());

    // Already have space — no eviction needed
    if gpu.memory_available >= total_needed {
        return Some(vec![]);
    }

    // Sort candidates by eviction priority
    let mut candidates: Vec<&LoadedModel> = gpu.loaded_models.iter().collect();
    sort_by_eviction_priority(&mut candidates, policy);

    // Greedily evict until we have enough space
    let mut freed: u64 = 0;
    let mut to_evict = Vec::new();
    let deficit = total_needed.as_u64().saturating_sub(gpu.memory_available.as_u64());

    for candidate in candidates {
        if freed >= deficit {
            break;
        }
        freed += candidate.vram_usage.as_u64();
        to_evict.push(candidate.model_id.clone());
    }

    if freed >= deficit {
        Some(to_evict)
    } else {
        // Even evicting everything isn't enough
        None
    }
}

fn sort_by_eviction_priority(candidates: &mut [&LoadedModel], policy: EvictionPolicy) {
    match policy {
        EvictionPolicy::Lru => {
            // Oldest last_request_at first (most stale → evict first)
            candidates.sort_by(|a, b| a.last_request_at.cmp(&b.last_request_at));
        }
        EvictionPolicy::Lfu => {
            // Fewest requests first
            candidates.sort_by(|a, b| a.request_count.cmp(&b.request_count));
        }
        EvictionPolicy::SizeWeightedLru => {
            // Score = staleness * size (prefer evicting large stale models)
            // Lower score = more worth evicting
            candidates.sort_by(|a, b| {
                let now = chrono::Utc::now();
                let a_staleness = (now - a.last_request_at).num_seconds().max(1) as f64;
                let b_staleness = (now - b.last_request_at).num_seconds().max(1) as f64;
                let a_score = a_staleness * a.vram_usage.as_u64() as f64;
                let b_score = b_staleness * b.vram_usage.as_u64() as f64;
                // Higher score = more eviction-worthy, so reverse order
                b_score
                    .partial_cmp(&a_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::test_helpers::GpuStateBuilder;
    use bytesize::ByteSize;
    use chrono::{Duration, Utc};

    #[test]
    fn no_eviction_needed_when_space_available() {
        let gpu = GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model("model-a", 8, 8001)
            .build();

        let result = select_evictions(
            &gpu,
            ByteSize::gb(8),
            ByteSize::gb(1),
            EvictionPolicy::Lru,
        );

        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn evicts_lru_model_when_full() {
        let old = Utc::now() - Duration::hours(2);
        let recent = Utc::now() - Duration::minutes(5);

        let gpu = GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model_last_used("old-model", 8, 8001, old)
            .with_model_last_used("recent-model", 8, 8002, recent)
            .build();

        let result = select_evictions(
            &gpu,
            ByteSize::gb(10),
            ByteSize::gb(1),
            EvictionPolicy::Lru,
        );

        let evicted = result.expect("should have eviction plan");
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], ModelId("old-model".into()));
    }

    #[test]
    fn evicts_multiple_models_if_needed() {
        let oldest = Utc::now() - Duration::hours(3);
        let old = Utc::now() - Duration::hours(2);
        let recent = Utc::now() - Duration::minutes(5);

        let gpu = GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model_last_used("oldest", 8, 8001, oldest)
            .with_model_last_used("old", 8, 8002, old)
            .with_model_last_used("recent", 8, 8003, recent)
            .build();

        let result = select_evictions(
            &gpu,
            ByteSize::gb(20),
            ByteSize::gb(1),
            EvictionPolicy::Lru,
        );

        let evicted = result.expect("should have eviction plan");
        assert_eq!(evicted.len(), 3);
        // Should be ordered oldest first
        assert_eq!(evicted[0], ModelId("oldest".into()));
        assert_eq!(evicted[1], ModelId("old".into()));
    }

    #[test]
    fn returns_none_when_impossible() {
        let gpu = GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model("model-a", 8, 8001)
            .build();

        // Requesting more than the GPU's total memory
        let result = select_evictions(
            &gpu,
            ByteSize::gb(30),
            ByteSize::gb(1),
            EvictionPolicy::Lru,
        );

        assert!(result.is_none());
    }

    #[test]
    fn lfu_evicts_least_used_first() {
        let gpu = GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model("popular", 8, 8001)
            .with_model("unpopular", 8, 8002)
            .build();

        // Manually set request counts (builder defaults to 0, so we need to construct directly)
        let mut gpu = gpu;
        gpu.loaded_models[0].request_count = 1000;
        gpu.loaded_models[1].request_count = 5;

        let result = select_evictions(
            &gpu,
            ByteSize::gb(10),
            ByteSize::gb(1),
            EvictionPolicy::Lfu,
        );

        let evicted = result.expect("should have eviction plan");
        assert_eq!(evicted[0], ModelId("unpopular".into()));
    }
}
