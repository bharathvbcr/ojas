"""Scratch-tree experiment: run D18's persistent gate->down kernel on the real Gemma4 graph.

MK_DUAL=1 lets it run when the dual-norm post-FFN norm is present: the kernel adds down(mid)
into a zeroed `down` scratch instead of `x`, then the normal post-FFN tail applies the norm and
residual (and writes bf16(x) for the per-layer gate when the fused tail is on).
MK_NOSYNC=1 drops the per-layer host synchronize/readback: the barrier counters are reset on
the GPU (`scale_f32_inplace(.., 0.0)` over their u32 bits) and `fail` is not checked.
"""
import pathlib

p = pathlib.Path(__file__).parent / "mktree/gemma-metal/src/gpu_model.rs"
s = p.read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)


rep("""                ) && !(use_gemma4_dual_norm && has_post_ff);
                if use_fuse_gate_down {""", """                ) && (std::env::var("MK_DUAL").is_ok() || !(use_gemma4_dual_norm && has_post_ff));
                if use_fuse_gate_down && use_gemma4_dual_norm && has_post_ff {
                    let w = layer.post_ff_norm.as_ref().expect("has_post_ff");
                    scale_f32_inplace(gpu, &self.down, 0.0, hidden)?;
                    gpu.barrier()?;
                    self.dispatch_fuse_gate_down_q4_ref(li, &x_bf16, &self.down)?;
                    gpu.barrier()?;
                    if emit_tail {
                        rms_norm_residual_add_emit_bf16(
                            gpu,
                            &self.down,
                            w,
                            &self.x,
                            None,
                            &gpu.act_bf16_scratch(hidden as usize)?,
                            1,
                            hidden,
                            eps,
                            1.0,
                        )?;
                    } else {
                        rms_norm_residual_add_f32_scaled(gpu, &self.down, w, &self.x, 1, hidden, eps, 1.0)?;
                    }
                } else if use_fuse_gate_down {""")
rep("""                            self.dispatch_fuse_gate_down_q4_ref(li, &x_bf16)?;""",
    """                            self.dispatch_fuse_gate_down_q4_ref(li, &x_bf16, &self.x)?;""")
rep("""    fn dispatch_fuse_gate_down_q4_ref(&self, li: usize, x_bf16: &GpuBuffer) -> Result<()> {
        let scratch = self.fuse_gate_down.as_ref().ok_or_else(|| {
            Error::Metal("fuse_gate_down scratch missing (ensure first)".into())
        })?;
        scratch.deps.write_u32(&[0, 0]);
        scratch.fail.write_u32(&[0]);""", """    fn dispatch_fuse_gate_down_q4_ref(&self, li: usize, x_bf16: &GpuBuffer, x_out: &GpuBuffer) -> Result<()> {
        let scratch = self.fuse_gate_down.as_ref().ok_or_else(|| {
            Error::Metal("fuse_gate_down scratch missing (ensure first)".into())
        })?;
        let nosync = std::env::var("MK_NOSYNC").is_ok();
        if nosync {
            scale_f32_inplace(&self.model.gpu, &scratch.deps, 0.0, 2)?;
            scale_f32_inplace(&self.model.gpu, &scratch.fail, 0.0, 1)?;
            self.model.gpu.barrier()?;
        } else {
            scratch.deps.write_u32(&[0, 0]);
            scratch.fail.write_u32(&[0]);
        }""")
rep("""            x_bf16,
            &self.mid,
            &self.x,
            &scratch.deps,
            &scratch.fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )?;
        self.model.gpu.synchronize()?;""", """            x_bf16,
            &self.mid,
            x_out,
            &scratch.deps,
            &scratch.fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )?;
        if nosync {
            return Ok(());
        }
        self.model.gpu.synchronize()?;""")
rep("""    emit_bf16_tail_enabled, mlp_gelu_tanh_bf16, mlp_gelu_tanh_bf16_up_at, ple_lookup,
    ple_lookup_q4_mlx,""", """    emit_bf16_tail_enabled, mlp_gelu_tanh_bf16, mlp_gelu_tanh_bf16_up_at, ple_lookup,
    ple_lookup_q4_mlx, rms_norm_residual_add_emit_bf16, rms_norm_residual_add_f32_scaled,""")
p.write_text(s)
print("patched")
