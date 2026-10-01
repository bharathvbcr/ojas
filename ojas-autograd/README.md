# ojas-autograd

`ojas-autograd` provides a reverse-mode automatic differentiation tape alongside an IEEE-754 `f64` central-difference numerical gradient checker.

---

## Tape Graph Architecture

```mermaid
flowchart TD
    subgraph TapeForward["Dynamic Tape Construction (Forward)"]
        V1["Var 1 (Input x)"] --> Op1["Tape::linear(x, W)"]
        V2["Var 2 (Weight W)"] --> Op1
        Op1 --> V3["Var 3 (Hidden)"]
        V3 --> Op2["Tape::rms_norm(Hidden, weight)"]
        Op2 --> V4["Var 4 (Loss)"]
    end

    subgraph TapeBackward["Reverse-Mode Traversal (Backward)"]
        Seed["Seed: dLoss/dVar4 = 1.0"] --> AdjOp2["Adjoint Op2 (RMSNorm Grad)"]
        AdjOp2 --> AdjOp1["Adjoint Op1 (Linear Grad)"]
        AdjOp1 --> GradW["Accumulate dLoss/dW"]
        AdjOp1 --> GradX["Accumulate dLoss/dx"]
    end

    TapeForward --> TapeBackward
```

---

## Numerical Gradcheck

`ojas-autograd` verifies analytic backward passes against numerical finite differences calculated in double precision (`f64`):

```mermaid
flowchart LR
    Analytical["Analytic Gradient (ojas-cpu Backward)\nf32 Vector"] --> Matcher{"gradients_match()\natol=1e-3, rtol=1e-2"}
    Numeric["central_diff() on f64 Forward\nSlope = [f(x+h) - f(x-h)] / (2h)"] --> Matcher
    Matcher -->|Passes within tolerance| Green["Verified Analytic Gradient"]
    Matcher -->|Exceeds tolerance| Err["Fail Fast with Detailed Error"]
```

### Gradcheck Invariants
1. **Finite Differences in f64:** The numerical oracle evaluates strictly in double precision (`f64`), preventing single-precision subtraction cancelation artifacts.
2. **Strict Step Size:** `h` must be finite and $> 0.0$.
3. **No NaN Masking:** Parity checks compare absolute and relative error on every single scalar coordinate. Any coordinate producing `NaN` or exceeding threshold returns a descriptive failure string.
