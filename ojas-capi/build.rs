// Gate allocator-aware results on exactly the answer the `gusset` crate
// compiled with. `gusset` declares `links = "gusset"` and publishes its probe
// result as `DEP_GUSSET_ALLOCATOR_API`; this is the pattern adopters copy.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(gusset_allocator_api)");
    println!("cargo::rerun-if-env-changed=DEP_GUSSET_ALLOCATOR_API");
    if std::env::var("DEP_GUSSET_ALLOCATOR_API").as_deref() == Ok("1") {
        println!("cargo::rustc-cfg=gusset_allocator_api");
    }
}
