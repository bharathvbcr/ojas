"""Reference oracle for ojas-data's window permutation (sampler.rs).

Reimplements SplitMix64 and the 4-round Feistel cycle-walk from the
documented algorithm, independently of the Rust code, and prints window
starts for the pinned cases in ojas-data/tests/sampler.rs.
"""

M = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
EPOCH_STEP = 0xD1B54A32D192ED03


def splitmix(state):
    state = (state + GAMMA) & M
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return state, z ^ (z >> 31)


def keys(seed, epoch):
    state = seed ^ ((epoch * EPOCH_STEP) & M)
    out = []
    for _ in range(4):
        state, k = splitmix(state)
        out.append(k)
    return out


def start(seed, windows, seq_len, epoch, ordinal):
    bits = (windows - 1).bit_length()
    h = max((bits + 1) // 2, 1)
    mask = (1 << h) - 1
    ks = keys(seed, epoch)
    v = ordinal
    while True:
        left = (v >> h) & mask
        right = v & mask
        for k in ks:
            _, f = splitmix(right ^ k)
            left, right = right, left ^ (f & mask)
        v = (left << h) | right
        if v < windows:
            return v * seq_len


# Matches the test: bin of 1001 tokens, seq_len 10 -> W = 100.
for seed in (42, 0xDEAD_BEEF):
    for epoch in (0, 1, 7):
        print(seed, epoch, [start(seed, 100, 10, epoch, o) for o in range(8)])
