#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import json
import time

def run_experiment(mixer: str, seq_len: int, batch_size: int, steps: int = 1000):
    """Runs the Metal native training engine for a given config."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting Run: Mixer={mixer.upper()} | Context={seq_len} | Batch={batch_size}")
    print(f"{'='*60}")
    
    # We will pass the mixer type via an environment variable or flag. 
    # For now, we assume the Rust binary accepts `--mixer` and `--seq-len`
    cmd = [
        "cargo", "run", "--release", "--bin", "train", "--",
        "--preset", "arch02-128m",
        "--mixer", mixer,
        "--seq-len", str(seq_len),
        "--batch", str(batch_size),
        "--total-steps", str(steps),
        "--warmdown-steps", str(int(steps * 0.1)),
    ]
    
    # For context length scaling, we adjust batch size to keep token budget roughly constant
    # or to fit in the 64GB M5 Pro unified memory.
    
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    
    start_t = time.time()
    
    # Log the output to a specific file
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{mixer}_ctx{seq_len}.log")
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, 
                env=env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Stream output to console and file
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
                f.flush()
                
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ Error: Run failed with exit code {process.returncode}")
                return False
                
    except KeyboardInterrupt:
        print("\n⚠️ Run interrupted by user.")
        process.kill()
        return False
        
    duration = time.time() - start_t
    print(f"✅ Run completed in {duration:.1f} seconds. Log saved to {log_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the Context-Length Crossover Experiment")
    parser.add_argument("--steps", type=int, default=1500, help="Number of training steps per run")
    args = parser.parse_args()
    
    # 9-arm experiment matrix: 3 mixers x 3 context lengths
    mixers = ["attention", "mingru", "mamba2"]
    
    # Context length -> Batch size (to fit in M5 Pro 64GB VRAM and normalize token volume)
    # 512 * 16 = 8192 tokens/step
    # 2048 * 4 = 8192 tokens/step
    # 8192 * 1 = 8192 tokens/step
    contexts = {
        512: 16,
        2048: 4,
        8192: 1
    }
    
    print("Starting 9-arm Metal-Native Context Crossover Matrix on M5 Pro...")
    
    results = {}
    for seq_len, batch_size in contexts.items():
        for mixer in mixers:
            success = run_experiment(mixer, seq_len, batch_size, steps=args.steps)
            results[f"{mixer}_ctx{seq_len}"] = "SUCCESS" if success else "FAILED"
            
            if not success:
                print("Aborting matrix due to failure.")
                sys.exit(1)
                
    print("\n🎉 All runs completed successfully!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
