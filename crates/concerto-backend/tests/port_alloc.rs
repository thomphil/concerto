//! Unit tests for the [`PortAllocator`].

use concerto_backend::PortAllocator;

#[test]
fn allocate_returns_unique_ports() {
    let allocator = PortAllocator::with_range(9000..9005);
    let a = allocator.allocate().unwrap();
    let b = allocator.allocate().unwrap();
    let c = allocator.allocate().unwrap();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
    assert!((9000..9005).contains(&a));
    assert!((9000..9005).contains(&b));
    assert!((9000..9005).contains(&c));
}

#[test]
fn released_port_can_be_reallocated() {
    let allocator = PortAllocator::with_range(9000..9002);
    let a = allocator.allocate().unwrap();
    let b = allocator.allocate().unwrap();
    assert_eq!(allocator.in_use_count(), 2);
    assert!(
        allocator.allocate().is_none(),
        "range is exhausted, allocate must return None"
    );

    allocator.release(a);
    assert_eq!(allocator.in_use_count(), 1);

    let reused = allocator
        .allocate()
        .expect("a freshly released port should be available");
    assert_eq!(reused, a);
    assert_eq!(allocator.in_use_count(), 2);
    // `b` is still held.
    let _ = b;
}

#[test]
fn exhausted_allocator_returns_none() {
    let allocator = PortAllocator::with_range(9100..9102);
    assert!(allocator.allocate().is_some());
    assert!(allocator.allocate().is_some());
    assert!(allocator.allocate().is_none());
    assert!(allocator.allocate().is_none());
}

#[test]
fn release_of_unallocated_port_is_noop() {
    let allocator = PortAllocator::with_range(9200..9205);
    allocator.release(9201);
    assert_eq!(allocator.in_use_count(), 0);
    // Should still allocate cleanly afterwards.
    assert!(allocator.allocate().is_some());
}

#[test]
fn default_range_is_used_when_unspecified() {
    let allocator = PortAllocator::new();
    let port = allocator.allocate().unwrap();
    assert!((8100..9000).contains(&port));
}
