//! `ThreadAllocCount` counts the calling thread's allocation calls only.

use ojas_test_support::{thread_allocs, ThreadAllocCount};

#[global_allocator]
static ALLOC: ThreadAllocCount = ThreadAllocCount;

#[test]
fn another_threads_allocations_are_not_counted() {
    let before = thread_allocs();
    let other = std::thread::spawn(|| {
        let mut sink = Vec::new();
        for i in 0..100usize {
            sink.push(vec![i; 8]);
        }
        sink.len()
    });
    assert_eq!(other.join().unwrap(), 100);
    // `spawn` and `join` allocate on this thread, but nothing like the
    // 100+ allocations the other thread made.
    let mine = thread_allocs() - before;
    assert!(mine < 50, "this thread counted {mine} allocations");

    let before = thread_allocs();
    let v = vec![1u32; 64];
    let after = thread_allocs();
    assert_eq!(after - before, 1, "one `vec!` is one allocation");
    drop(v);
    assert_eq!(thread_allocs(), after, "a free is not counted");
}
