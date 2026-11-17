//! The no-GPU skip rule shared by every test that opens a device
//! (`common::open_or_skip`). Platform-neutral, so the skip branch is
//! exercised on a machine that does have Metal and on Linux.

mod common;

use common::open_or_skip;

#[derive(Debug)]
struct NoDevice;

#[test]
fn an_opened_device_is_used_whether_or_not_the_skip_is_allowed() {
    assert_eq!(open_or_skip::<_, NoDevice>("dev", Ok(7), false), Some(7));
    assert_eq!(open_or_skip::<_, NoDevice>("dev", Ok(7), true), Some(7));
}

#[test]
fn a_device_that_did_not_open_skips_when_allowed() {
    assert_eq!(open_or_skip::<u8, _>("dev", Err(NoDevice), true), None);
}

#[test]
#[should_panic(expected = "OJAS_ALLOW_NO_GPU")]
fn a_device_that_did_not_open_fails_loudly_when_not_allowed() {
    let _ = open_or_skip::<u8, _>("dev", Err(NoDevice), false);
}
