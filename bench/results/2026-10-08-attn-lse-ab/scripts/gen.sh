#!/bin/bash
# Assemble each tree's attn_ab example: a backend macro, then the tree shim.
set -eu
D=/Users/bharath/Code/research/ojas/target-attn-ab
for be in metal wgpu; do
  case $be in
    metal) ctor='ojas_metal::MetalBackend::new($b).unwrap()' ;;
    wgpu) ctor='ojas_wgpu::WgpuBackend::open($b).unwrap()' ;;
  esac
  for t in old new; do
    out=$D/ab/$t/ojas-$be/examples/attn_ab_$be.rs
    {
      echo 'macro_rules! BACKEND_MAIN {'
      echo '    ($b:expr, $a:ty) => {{'
      echo "        let be = $ctor;"
      echo "        ab::run::<_, \$a>(&be, \"$be\");"
      echo '    }};'
      echo '}'
      echo
      cat "$D/shim_$t.rs"
    } >"$out"
    echo "$out"
  done
done
