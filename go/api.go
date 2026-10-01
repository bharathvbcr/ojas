// Package ojas is the in-process training and inference API.
//
// One gusset handle runs every step. The worker count defaults to 1 and
// changes with SetPoolSize before the first call. WithDerivedPoolSize uses
// runtime.GOMAXPROCS(0), capped at gusset.MaxPoolSize; the default stays 1
// because the tests in this package open that pool. Load sends a relative
// path (at most 4 KiB). Larger step and generate payloads go through a
// gusset Buffer. Those buffers share a 64 MiB live budget
// (SetBufferBudget) so queued callers cannot each hold a 1 GiB buffer.
// Weights are not sent. WithMemoryGovernor is off until called.
//
// Link the umbrella archive before `go test`. The gusset module's default
// linker path is its own checkout, which does not contain ojas_engine_init,
// so the test uses pkg-config. -a is required because Go's build cache does
// not track libgusset.a. On Linux use PKG_CONFIG_PATH="$PWD/linux".
//
//	cargo build -p ojas-gusset-engine
//	cd go && PKG_CONFIG_PATH="$PWD" go test -a -tags gusset_pkgconfig -count=1 -timeout 10m
package ojas

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
)

// SetModelRoot is the directory relative load paths are resolved under.
func SetModelRoot(ctx context.Context, dir string) error {
	if ctx == nil {
		return errors.New("ojas: nil context")
	}
	return setRoot(dir)
}

// Device selectors for LoadOn.
//
// DeviceCPU and DeviceCPUParallel compute on the CPU. DeviceCPUParallel
// splits large linears across the threads argument (1..=MaxCPUThreads).
// DeviceMetal opens the Metal device at load and returns a model id whose
// Step and GenerateGreedy run on that device, reading back only the values
// they return; Generate with caller logits takes the argmax on the host.
// With no Metal device the load fails with the device error, never a CPU
// session. DeviceWgpu does the same with a wgpu device: Step and
// GenerateGreedy run on it and read back only what they return, and with no
// wgpu adapter the load fails with a "wgpu:" error, never a CPU session.
// threads is ignored for DeviceMetal and DeviceWgpu.
const (
	DeviceCPU         uint32 = deviceCPU
	DeviceCPUParallel uint32 = deviceCPUParallel
	DeviceMetal       uint32 = deviceMetal
	DeviceWgpu        uint32 = deviceWgpu
)

// MaxCPUThreads is the largest threads value DeviceCPUParallel accepts.
// Larger values are refused, not clamped.
const MaxCPUThreads uint32 = 256

// In-band error kinds. When a gusset error message contains one of these
// prefixes, errors.Is matches the corresponding sentinel. ojas-capi emits
// E_CAPACITY, E_NONFINITE and E_DEVICE_LOST, choosing the kind from the typed
// Rust error (ojas-capi/src/lib.rs kind_of), never from message text. Nothing
// in Rust emits E_BUSY today.
//
//	ojas:E_CAPACITY:
//	ojas:E_DEVICE_LOST:
//	ojas:E_BUSY:
//	ojas:E_NONFINITE:
var (
	ErrCapacity   = errors.New("ojas: capacity")
	ErrDeviceLost = errors.New("ojas: device lost")
	ErrBusy       = errors.New("ojas: busy")
	ErrNonFinite  = errors.New("ojas: non-finite")
)

// Load checks a relative safetensors path and returns a CPU model id.
//
// Load reads the file header and counts its tensors. It does not read the
// weights, and later calls do not use them.
func Load(ctx context.Context, path string) (uint64, error) {
	return LoadOn(ctx, DeviceCPU, 1, path)
}

// LoadOn is Load with a device selector. threads is used only by
// DeviceCPUParallel. A device that cannot be opened is the load's error;
// see the device constants.
func LoadOn(ctx context.Context, device, threads uint32, path string) (uint64, error) {
	if ctx == nil {
		return 0, errors.New("ojas: nil context")
	}
	if len(path) > inlineLimit || 12+len(path) > inlineLimit {
		return 0, errors.New("path exceeds 4096 bytes")
	}
	payload := make([]byte, 12+len(path))
	copy(payload[:4], []byte("OJDV"))
	binary.LittleEndian.PutUint32(payload[4:8], device)
	binary.LittleEndian.PutUint32(payload[8:12], threads)
	copy(payload[12:], path)
	out, err := callEngine(ctx, opLoad, payload)
	if err != nil {
		return 0, err
	}
	if len(out) < 8 {
		return 0, errors.New("load: short result")
	}
	return binary.LittleEndian.Uint64(out[:8]), nil
}

// StepRequest is one training step.
//
// Set Logits and Targets for mean cross-entropy, or Tokens and TokenTargets
// for a linear map of the u16 tokens followed by that loss. An empty request
// is the header-only form and returns a shape error.
type StepRequest struct {
	Batch uint32
	Seq   uint32
	Step  uint32
	Lr    float32

	Logits  []float32
	Targets []uint32

	Tokens       []uint16
	TokenTargets []uint16
}

// StepStats is the inline f32 trio from a step.
type StepStats struct {
	Loss     float32
	GradNorm float32
	Lr       float32
}

// Step computes the loss, gradient norm, and learning rate for the payload on
// the model's device. It needs a live model id but does not read or update
// weights.
// Rust errors are returned as text.
func Step(ctx context.Context, id uint64, req StepRequest) (StepStats, error) {
	payload, err := encodeStep(id, req)
	if err != nil {
		return StepStats{}, err
	}
	out, err := callEngine(ctx, opStep, payload)
	if err != nil {
		return StepStats{}, err
	}
	if len(out) != 12 {
		return StepStats{}, errors.New("step: short result")
	}
	return StepStats{
		Loss:     math.Float32frombits(binary.LittleEndian.Uint32(out[0:4])),
		GradNorm: math.Float32frombits(binary.LittleEndian.Uint32(out[4:8])),
		Lr:       math.Float32frombits(binary.LittleEndian.Uint32(out[8:12])),
	}, nil
}

// Generate returns argmax of logits. A non-finite logit is an error.
func Generate(ctx context.Context, id uint64, logits []float32) (uint32, error) {
	return generate(ctx, id, genLogits, logits, nil)
}

// GenerateGreedy returns one token from a fixed two-token demonstration model
// (vocabulary 0 and 1). It does not use the loaded file. A prompt id of 2 or
// more is an out-of-range error.
func GenerateGreedy(ctx context.Context, id uint64, prompt []uint32) (uint32, error) {
	return generate(ctx, id, genGreedy, nil, prompt)
}

// Free drops a session. An unknown id, including a second free, is an error.
func Free(ctx context.Context, id uint64) error {
	var payload [8]byte
	binary.LittleEndian.PutUint64(payload[:], id)
	_, err := callEngine(ctx, opFree, payload[:])
	return err
}

// Close joins the worker pool. It returns when that join finishes or ctx
// ends, whichever comes first. A cancelled or expired context does not
// open a second pool: if the join is still running, later calls wait for
// it or refuse. If gusset reports that workers are still running after
// its join budget (30s), Close returns that error and this process will
// not open another pool on top of those workers.
func Close(ctx context.Context) error {
	if ctx == nil {
		return errors.New("ojas: nil context")
	}
	return closeHandle(ctx)
}

func encodeStep(id uint64, req StepRequest) ([]byte, error) {
	if _, err := mulRows(req.Batch, req.Seq); err != nil {
		return nil, err
	}
	logitFields := len(req.Logits) > 0 || len(req.Targets) > 0
	tokenFields := len(req.Tokens) > 0 || len(req.TokenTargets) > 0
	if logitFields && tokenFields {
		return nil, errors.New("step payload has both logits and tokens: set Logits/Targets or Tokens/TokenTargets")
	}
	mode := stepHeader
	classes := uint32(0)
	switch {
	case logitFields:
		mode = stepLogits
		rows, err := mulRows(req.Batch, req.Seq)
		if err != nil {
			return nil, err
		}
		if err := fitUint32(len(req.Logits)); err != nil {
			return nil, err
		}
		if err := fitUint32(len(req.Targets)); err != nil {
			return nil, err
		}
		if rows <= 0 || len(req.Targets) == 0 || len(req.Logits)%len(req.Targets) != 0 {
			classes = 0
		} else {
			classes = uint32(len(req.Logits) / len(req.Targets))
		}
	case tokenFields:
		mode = stepTokens
		if len(req.TokenTargets) > 0 {
			var max uint16
			for _, t := range req.TokenTargets {
				if t > max {
					max = t
				}
			}
			classes = uint32(max) + 1
			if classes < 2 {
				classes = 2
			}
		}
	}
	// The Rust parser owns shape checks. Classes of 0 still go out so a
	// bad request is a Rust error string, not a Go invention.
	buf := make([]byte, 0, 8+4*6)
	var word [8]byte
	binary.LittleEndian.PutUint64(word[:], id)
	buf = append(buf, word[:]...)
	putU32 := func(v uint32) {
		binary.LittleEndian.PutUint32(word[:4], v)
		buf = append(buf, word[:4]...)
	}
	putU32(mode)
	putU32(req.Batch)
	putU32(req.Seq)
	if mode == stepHeader {
		putU32(req.Step)
		return buf, nil
	}
	putU32(classes)
	putU32(req.Step)
	binary.LittleEndian.PutUint32(word[:4], math.Float32bits(req.Lr))
	buf = append(buf, word[:4]...)
	if mode == stepLogits {
		for _, v := range req.Logits {
			binary.LittleEndian.PutUint32(word[:4], math.Float32bits(v))
			buf = append(buf, word[:4]...)
		}
		for _, v := range req.Targets {
			binary.LittleEndian.PutUint32(word[:4], v)
			buf = append(buf, word[:4]...)
		}
		return buf, nil
	}
	for _, v := range req.Tokens {
		binary.LittleEndian.PutUint16(word[:2], v)
		buf = append(buf, word[:2]...)
	}
	for _, v := range req.TokenTargets {
		binary.LittleEndian.PutUint16(word[:2], v)
		buf = append(buf, word[:2]...)
	}
	return buf, nil
}

func generate(ctx context.Context, id uint64, mode uint32, logits []float32, prompt []uint32) (uint32, error) {
	if err := fitUint32(len(logits)); err != nil {
		return 0, err
	}
	if err := fitUint32(len(prompt)); err != nil {
		return 0, err
	}
	var word [8]byte
	buf := make([]byte, 0, 16+len(logits)*4+len(prompt)*4)
	binary.LittleEndian.PutUint64(word[:], id)
	buf = append(buf, word[:]...)
	binary.LittleEndian.PutUint32(word[:4], mode)
	buf = append(buf, word[:4]...)
	if mode == genLogits {
		binary.LittleEndian.PutUint32(word[:4], uint32(len(logits)))
		buf = append(buf, word[:4]...)
		for _, v := range logits {
			binary.LittleEndian.PutUint32(word[:4], math.Float32bits(v))
			buf = append(buf, word[:4]...)
		}
	} else {
		binary.LittleEndian.PutUint32(word[:4], uint32(len(prompt)))
		buf = append(buf, word[:4]...)
		for _, v := range prompt {
			binary.LittleEndian.PutUint32(word[:4], v)
			buf = append(buf, word[:4]...)
		}
	}
	out, err := callEngine(ctx, opGenerate, buf)
	if err != nil {
		return 0, err
	}
	if len(out) != 4 {
		return 0, errors.New("generate: short result")
	}
	return binary.LittleEndian.Uint32(out), nil
}

func poisonHandle(ctx context.Context) error {
	_, err := callEngine(ctx, opPanic, nil)
	return err
}

func mulRows(batch, seq uint32) (int, error) {
	if batch != 0 && uint64(seq) > math.MaxUint64/uint64(batch) {
		return 0, errors.New("ojas: batch*seq overflows")
	}
	n := uint64(batch) * uint64(seq)
	if n > uint64(math.MaxInt) {
		return 0, errors.New("ojas: batch*seq overflows")
	}
	return int(n), nil
}

func fitUint32(n int) error {
	if n < 0 || uint64(n) > math.MaxUint32 {
		return errors.New("ojas: length exceeds uint32")
	}
	return nil
}
