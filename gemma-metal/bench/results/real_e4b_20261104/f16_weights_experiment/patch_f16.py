import pathlib
p = pathlib.Path("/private/tmp/claude-501/-Users-bharath-Code-research-ojas/5fb0daf7-60f3-4eaa-b3c2-af0627b7fb33/scratchpad/bf16tree/gemma-metal/src/kernels.rs")
s = p.read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
rep('''            crate::quant::QuantScheme::Bf16 => {
                Err(Error::Metal("HotQuantBanks::gemv: Bf16 unsupported".into()))
            }''', '''            crate::quant::QuantScheme::Bf16 => {
                // Scratch experiment: Bf16 banks here hold fp16 weights (`F16W=1`).
                let p = gpu.rt.pipeline("gemv_f16w").map_err(map_metal)?;
                let (rows, cols) = (self.rows, self.cols);
                let n_tg = (rows as usize).div_ceil(8);
                dispatch_2d_tg(&gpu.rt, &p, n_tg, 1, 64, |bnd| {
                    set_gpu_buf(bnd, &self.packed, 0);
                    set_gpu_buf(bnd, x, 1);
                    set_gpu_buf(bnd, y, 2);
                    set_u32(bnd, rows, 3);
                    set_u32(bnd, cols, 4);
                    set_u32(bnd, u32::from(x_is_bf16), 5);
                })
                .map_err(map_metal)
            }''')
rep('''pub fn upload_quant_hot(gpu: &GemmaGpu, w: &QuantMatrix) -> Result<HotQuantBanks> {
''', '''fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let h = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let r = if rem > halfway || (rem == halfway && (h & 1) == 1) { h + 1 } else { h };
        return sign | r as u16;
    }
    let h = man >> 13;
    let rem = man & 0x1fff;
    let mut out = ((e as u32) << 10) | h;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        out += 1;
    }
    sign | out as u16
}

pub fn upload_quant_hot(gpu: &GemmaGpu, w: &QuantMatrix) -> Result<HotQuantBanks> {
    if std::env::var("F16W").map(|v| v == "1").unwrap_or(false)
        && matches!(w.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
        && w.rows != 262_144
        && w.cols % 8 == 0
    {
        let f = w.dequant_f32()?;
        let mut bytes = Vec::with_capacity(f.len() * 2);
        for v in &f {
            bytes.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
        }
        let packed = gpu.rt.alloc_buffer_hot(bytes.len()).map_err(map_metal)?;
        packed.write_bytes(&bytes);
        let scales = gpu.rt.alloc_buffer_hot(4).map_err(map_metal)?;
        let zeros = gpu.rt.alloc_buffer_hot(4).map_err(map_metal)?;
        return Ok(HotQuantBanks {
            scheme: crate::quant::QuantScheme::Bf16,
            layout: HotGemvLayout::RowMajor,
            rows: w.rows as u32,
            cols: w.cols as u32,
            group_size: w.scheme.group_size().unwrap_or(64) as u32,
            packed,
            scales,
            zeros,
        });
    }
''')
p.write_text(s)
print("patched")
