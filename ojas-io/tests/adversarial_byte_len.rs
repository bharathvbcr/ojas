//! Adversarial: sub-byte dtypes still multiply by bits in u64, so a tensor
//! whose byte count fits u64 is refused when its bit count does not -- the
//! case the whole-byte fix removed, left in place for F4 / F6.

use ojas_io::{SafeTensorsWriter, StDtype, TensorSpec};

#[test]
fn sub_byte_dtype_with_representable_byte_size_is_not_refused_for_its_bit_count() {
    for dtype in [StDtype::F4, StDtype::F6E2M3, StDtype::F6E3M2] {
        // A multiple of 8 elements keeps the size whole; the first multiple
        // whose bit count passes u64 still has a byte count far below it.
        let numel = (u64::MAX / dtype.bits() + 1) / 8 * 8;
        let shape = [numel];
        let bytes = u64::try_from(u128::from(numel) * u128::from(dtype.bits()) / 8)
            .expect("representable by construction");
        let spec = [TensorSpec {
            name: "t",
            dtype,
            shape: &shape,
        }];
        let w = SafeTensorsWriter::new(Vec::new(), &spec, &[])
            .unwrap_or_else(|e| panic!("{dtype:?} {numel}: refused {bytes}-byte size: {e}"));
        assert_eq!(w.remaining(), bytes);
    }
}
