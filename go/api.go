// Package ojas is the in-process training and inference API.
//
// A model id names a nanolab GPT held on one device by the Rust engine
// (ojas-capi). LoadModel reads one from a safetensors file, NewModel
// initialises one, and Resume restores one with its trainer from a
// checkpoint directory. OpenTrainer attaches a trainer over a token bin;
// TrainStep and TrainStepTokens run optimizer steps on the device;
// SaveCheckpoint writes the checkpoint directory. LoadTokenizer, Tokenize and
// Detokenize use the GPT-2 byte-level BPE; GenerateIDs samples continuations
// on the device. Every path is relative to SetModelRoot's directory.
//
// One gusset handle runs every call. The worker count defaults to 1 and
// changes with SetPoolSize before the first call. WithDerivedPoolSize uses
// runtime.GOMAXPROCS(0), capped at gusset.MaxPoolSize; the default stays 1
// because the tests in this package open that pool. Payloads up to 4 KiB go
// inline; larger ones go through a gusset Buffer. Those buffers share a
// 64 MiB live budget (SetBufferBudget) so queued callers cannot each hold a
// 1 GiB buffer. Weights are not sent. WithMemoryGovernor is off until called.
//
// Concurrency: calls on different ids overlap (with a pool larger than 1).
// Two calls on one id never overlap: the second is refused with ErrBusy
// while the first runs. With a pool of 1, calls run one at a time and
// ErrBusy cannot occur.
//
// Cancellation: a cancelled or expired ctx returns its error at once
// (errors.Is(err, context.Canceled) or context.DeadlineExceeded); gusset
// detaches from the job. The engine stops the job at its next op, and a
// TrainStep stopped that way commits nothing: the next TrainStep is the same
// step. A cancel that lands after the step's last cancellable op (its
// optimizer has started) does not stop it, and that step commits even though
// the caller saw context.Canceled; the next result's Step says which
// happened. Until the detached job stops, a pool of 1 queues the next call
// behind it, and a larger pool returns ErrBusy for the same id.
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
	"fmt"
	"math"
)

// SetModelRoot is the directory every relative path is resolved under.
func SetModelRoot(ctx context.Context, dir string) error {
	if ctx == nil {
		return errors.New("ojas: nil context")
	}
	return setRoot(dir)
}

// Device selectors for LoadOptions.Device.
//
// DeviceCPU and DeviceCPUParallel compute on the CPU. DeviceCPUParallel
// splits large linears across Threads (1..=MaxCPUThreads). DeviceMetal
// opens the Metal device at load and keeps the model there; DeviceWgpu does
// the same with a wgpu device. With no such device the load fails with the
// device's error ("metal:" or "wgpu:"), never a CPU session. Generate with
// caller logits takes the argmax on the host.
//
// On Metal and wgpu a non-finite value (ErrNonFinite) or a lost device
// (ErrDeviceLost) is detected on the device and reported when the call
// synchronizes, so it may come from an earlier op of the same call. Every
// call synchronizes before it returns, so it never comes from an earlier
// call.
const (
	DeviceCPU         uint32 = deviceCPU
	DeviceCPUParallel uint32 = deviceCPUParallel
	DeviceMetal       uint32 = deviceMetal
	DeviceWgpu        uint32 = deviceWgpu
)

// MaxCPUThreads is the largest Threads DeviceCPUParallel accepts. Larger
// values are refused, not clamped.
const MaxCPUThreads uint32 = 256

// In-band error kinds. When an engine message starts with one of these
// prefixes, errors.Is matches the corresponding sentinel. ojas-capi chooses
// the kind from the typed Rust error (ojas-capi/src/lib.rs kind_of) or from
// the session lock, never from message text, and puts it first; a prefix
// anywhere else in a message (a user path, say) selects nothing.
//
//	ojas:E_CAPACITY:    a byte budget, the process memory ceiling, the
//	                    64-session table or a context length
//	ojas:E_DEVICE_LOST: the device was lost
//	ojas:E_BUSY:        another call holds this model id
//	ojas:E_NONFINITE:   a NaN or infinity in a loss, gradient or logit
//	ojas:E_POISONED:    the trainer was left partly updated by a failed
//	                    optimizer step, or a call panicked holding the model
var (
	ErrCapacity   = errors.New("ojas: capacity")
	ErrDeviceLost = errors.New("ojas: device lost")
	ErrBusy       = errors.New("ojas: busy")
	ErrNonFinite  = errors.New("ojas: non-finite")
	// ErrPoisoned is the model's state, not the gusset handle's
	// (gusset.ErrPoisoned). Every later TrainStep, SaveCheckpoint and
	// GenerateIDs on the id refuses with it; Resume from a checkpoint.
	ErrPoisoned = errors.New("ojas: model poisoned")
)

// Numerics selects a CPU session's arithmetic contract.
type Numerics uint32

const (
	// NumericsDefault keeps the backend's default.
	NumericsDefault Numerics = 0
	// NumericsExact is ascending-index f32 without fused multiply-add;
	// results are bitwise reproducible. CPU only.
	NumericsExact Numerics = 1
	// NumericsFast lets the CPU reorder reductions. CPU only.
	NumericsFast Numerics = 2
)

// LoadOptions says where a new model id computes and what it may hold.
type LoadOptions struct {
	// Device is one of the Device constants. The zero value is DeviceCPU.
	Device uint32
	// Threads is read only for DeviceCPUParallel.
	Threads uint32
	// BudgetBytes caps what the model, its trainer and every call on it may
	// allocate. 0 means 1 GiB. Every model's budget draws from the
	// process-wide memory ceiling (SetMemoryCeiling, 1 GiB by default), so a
	// BudgetBytes above the ceiling is ErrCapacity at load, and all open
	// models together never pass it. A 124M training session needs several
	// GiB: raise the ceiling first.
	BudgetBytes uint64
	// Numerics is CPU only; Metal and wgpu refuse anything but the default.
	Numerics Numerics
}

func (o LoadOptions) put(r *record) *record {
	r.u32(tagDevice, o.Device)
	if o.Device == DeviceCPUParallel {
		r.u32(tagThreads, o.Threads)
	}
	if o.BudgetBytes != 0 {
		r.u64(tagBudget, o.BudgetBytes)
	}
	if o.Numerics != NumericsDefault {
		r.u32(tagNumerics, uint32(o.Numerics))
	}
	return r
}

// LoadModel reads a nanolab safetensors file (nanolab state_dict names, the
// spec in __metadata__["ojas.spec"]) and returns a model id on opts.Device.
// The path is relative to the model root, at most 4 KiB, and opened once
// without following a final symlink. Every tensor is checked against the
// spec before any is kept, and the weights are uploaded to the device.
func LoadModel(ctx context.Context, path string, opts LoadOptions) (uint64, error) {
	if ctx == nil {
		return 0, errors.New("ojas: nil context")
	}
	if len(path) > inlineLimit {
		return 0, errors.New("path exceeds 4096 bytes")
	}
	return session(ctx, opLoad, opts.put(new(record)).str(tagPath, path).bytes(nil))
}

// Inspect counts the tensors in a safetensors file from its header alone.
// No model id is created.
func Inspect(ctx context.Context, path string) (uint32, error) {
	if len(path) > inlineLimit {
		return 0, errors.New("path exceeds 4096 bytes")
	}
	out, err := callEngine(ctx, opInspect, []byte(path))
	if err != nil {
		return 0, err
	}
	if len(out) != 4 {
		return 0, errors.New("inspect: short result")
	}
	return binary.LittleEndian.Uint32(out), nil
}

// ModelSpec is a nanolab GPT shape. The embedding is tied, and QK-norm, the
// per-head gate and the value residual are always on.
type ModelSpec struct {
	Vocab, NEmbd, NLayer, NHead, NKVHead, HeadDim, Hidden, MaxSeq uint32
	RopeBase, RMSEps                                              float64
}

// NewModel returns a model id holding a fresh nanolab init of spec: each
// parameter is drawn from its own counter RNG keyed by seed and its name.
func NewModel(ctx context.Context, spec ModelSpec, seed uint64, opts LoadOptions) (uint64, error) {
	r := opts.put(new(record))
	for _, f := range []struct{ tag, v uint32 }{
		{tagVocab, spec.Vocab}, {tagNEmbd, spec.NEmbd}, {tagNLayer, spec.NLayer},
		{tagNHead, spec.NHead}, {tagNKVHead, spec.NKVHead}, {tagHeadDim, spec.HeadDim},
		{tagHidden, spec.Hidden}, {tagMaxSeq, spec.MaxSeq},
	} {
		r.u32(f.tag, f.v)
	}
	r.f64(tagRopeBase, spec.RopeBase).f64(tagRMSEps, spec.RMSEps).u64(tagSeed, seed)
	return session(ctx, opNew, r.bytes(nil))
}

// ScheduleKind is the learning-rate schedule's shape.
type ScheduleKind uint32

const (
	// ScheduleCosine warms up linearly, then decays by cosine.
	ScheduleCosine ScheduleKind = 0
	// ScheduleWSD warms up, holds, then decays over the last DecayFrac.
	ScheduleWSD ScheduleKind = 1
)

// Schedule scales every group's peak learning rate per step.
type Schedule struct {
	Kind          ScheduleKind
	Warmup, Total uint64
	// DecayFrac is read only for ScheduleWSD.
	DecayFrac float64
}

// BinFormat is a token bin's layout.
type BinFormat uint32

const (
	// BinHeaderless is little-endian uint16 tokens and nothing else.
	BinHeaderless BinFormat = 0
	// BinFineWeb is the FineWeb 256 x int32 header, then uint16 tokens.
	BinFineWeb BinFormat = 1
)

// NonFinitePolicy is what a non-finite loss or gradient does to the data
// cursor. Either way the step returns ErrNonFinite and commits nothing else.
type NonFinitePolicy uint32

const (
	// OnNonFiniteAbort keeps the cursor: the next step retries the batches.
	OnNonFiniteAbort NonFinitePolicy = 0
	// OnNonFiniteSkipBatch moves the cursor past them.
	OnNonFiniteSkipBatch NonFinitePolicy = 1
)

// TrainConfig is a trainer's data and optimizer setup. Every field is sent;
// a zero learning rate trains with a zero learning rate. NanolabTrainConfig
// fills nanolab's optimizer defaults.
type TrainConfig struct {
	// TokenBin is a path under the model root.
	TokenBin  string
	BinFormat BinFormat
	// Batch rows of Seq tokens per micro-batch, Accum micro-batches a step.
	Batch, Seq, Accum uint32
	// Seed keys the sampler's per-epoch permutation of windows.
	Seed     uint64
	Schedule Schedule
	// MatrixLR is the Muon peak for hidden matrices; AdamLR the AdamW peak
	// for the embedding and every vector.
	MatrixLR, AdamLR float64
	GradClip         float32
	OnNonFinite      NonFinitePolicy
	// TokenizerHash names the tokenizer that made the bin. It is saved with
	// every checkpoint and Resume refuses a different one.
	TokenizerHash [32]byte
}

// NanolabTrainConfig is cfg with nanolab's optimizer defaults: Muon 0.025,
// AdamW 6e-4, clip 1.0, abort on a non-finite value.
func NanolabTrainConfig(tokenBin string, batch, seq, accum uint32, seed uint64, schedule Schedule) TrainConfig {
	return TrainConfig{
		TokenBin: tokenBin, BinFormat: BinHeaderless,
		Batch: batch, Seq: seq, Accum: accum, Seed: seed, Schedule: schedule,
		MatrixLR: 0.025, AdamLR: 6e-4, GradClip: 1.0, OnNonFinite: OnNonFiniteAbort,
	}
}

func (c TrainConfig) put(r *record) *record {
	r.str(tagTokenBin, c.TokenBin).u32(tagBinFormat, uint32(c.BinFormat))
	r.u32(tagBatch, c.Batch).u32(tagSeq, c.Seq).u32(tagAccum, c.Accum).u64(tagDataSeed, c.Seed)
	r.u32(tagSchedule, uint32(c.Schedule.Kind)).u64(tagWarmup, c.Schedule.Warmup).u64(tagTotal, c.Schedule.Total)
	if c.Schedule.Kind == ScheduleWSD {
		r.f64(tagDecayFrac, c.Schedule.DecayFrac)
	}
	r.f64(tagMatrixLR, c.MatrixLR).f64(tagAdamLR, c.AdamLR).f32(tagGradClip, c.GradClip)
	r.u32(tagOnNonFinite, uint32(c.OnNonFinite))
	if c.TokenizerHash != ([32]byte{}) {
		r.raw(tagTokenizerHash, c.TokenizerHash[:])
	}
	return r
}

// OpenTrainer attaches a trainer to id at step 0 over cfg.TokenBin. The
// model's weights become the trainer's; a model that already trains is
// refused. A refusal changes nothing.
func OpenTrainer(ctx context.Context, id uint64, cfg TrainConfig) error {
	payload := cfg.put(new(record)).bytes(binary.LittleEndian.AppendUint64(nil, id))
	_, err := callEngine(ctx, opTrainOpen, payload)
	return err
}

// StepResult is what one optimizer step did.
type StepResult struct {
	// Loss is the mean of the step's micro-batch losses.
	Loss float32
	// GradNorm is the global gradient norm before clipping.
	GradNorm float32
	// MatrixLR and AdamLR are the learning rates this step applied.
	MatrixLR, AdamLR float64
	// Step is the count of completed steps after this one.
	Step uint64
	// Tokens is the number of input tokens in the step.
	Tokens uint64
}

// TrainStep runs one step on the trainer's next Accum sampled micro-batches.
// It reads back the loss and nothing else. A step that fails before its
// optimizer (ErrNonFinite, ErrCapacity, ErrDeviceLost, a cancellation)
// commits nothing; one that fails in its optimizer leaves the model
// ErrPoisoned.
func TrainStep(ctx context.Context, id uint64) (StepResult, error) {
	payload := binary.LittleEndian.AppendUint64(nil, id)
	payload = binary.LittleEndian.AppendUint32(payload, stepSampled)
	return trainStep(ctx, payload)
}

// TrainStepTokens runs one step on caller micro-batches: x[i] and y[i] are
// row-major [rows, seq] input and target ids, seq the trainer's Seq. The
// sampler's cursor does not move.
func TrainStepTokens(ctx context.Context, id uint64, seq uint32, x, y [][]uint32) (StepResult, error) {
	payload, err := encodeTokens(id, seq, x, y)
	if err != nil {
		return StepResult{}, err
	}
	return trainStep(ctx, payload)
}

func encodeTokens(id uint64, seq uint32, x, y [][]uint32) ([]byte, error) {
	if seq == 0 {
		return nil, errors.New("ojas: seq is 0")
	}
	if len(x) == 0 || len(x) != len(y) {
		return nil, fmt.Errorf("ojas: %d input and %d target micro-batches", len(x), len(y))
	}
	if err := fitUint32(len(x)); err != nil {
		return nil, err
	}
	total := 0
	for i := range x {
		if len(x[i]) != len(y[i]) || len(x[i]) == 0 || len(x[i])%int(seq) != 0 {
			return nil, fmt.Errorf("ojas: micro-batch %d has %d inputs and %d targets; each must be a non-zero multiple of seq %d", i, len(x[i]), len(y[i]), seq)
		}
		if err := fitUint32(len(x[i]) / int(seq)); err != nil {
			return nil, err
		}
		total += 2 * len(x[i])
	}
	payload := make([]byte, 0, 20+8*len(x)+4*total)
	payload = binary.LittleEndian.AppendUint64(payload, id)
	payload = binary.LittleEndian.AppendUint32(payload, stepTokens)
	payload = binary.LittleEndian.AppendUint32(payload, uint32(len(x)))
	payload = binary.LittleEndian.AppendUint32(payload, seq)
	for i := range x {
		payload = binary.LittleEndian.AppendUint32(payload, uint32(len(x[i])/int(seq)))
		for _, v := range x[i] {
			payload = binary.LittleEndian.AppendUint32(payload, v)
		}
		for _, v := range y[i] {
			payload = binary.LittleEndian.AppendUint32(payload, v)
		}
	}
	return payload, nil
}

func trainStep(ctx context.Context, payload []byte) (StepResult, error) {
	out, err := callEngine(ctx, opTrainStep, payload)
	if err != nil {
		return StepResult{}, err
	}
	if len(out) != 40 {
		return StepResult{}, errors.New("train step: short result")
	}
	le := binary.LittleEndian
	return StepResult{
		Loss:     math.Float32frombits(le.Uint32(out[0:4])),
		GradNorm: math.Float32frombits(le.Uint32(out[4:8])),
		MatrixLR: math.Float64frombits(le.Uint64(out[8:16])),
		AdamLR:   math.Float64frombits(le.Uint64(out[16:24])),
		Step:     le.Uint64(out[24:32]),
		Tokens:   le.Uint64(out[32:40]),
	}, nil
}

// SaveCheckpoint writes id's trainer to the checkpoint directory dir under
// the model root: model.safetensors (loadable by LoadModel and torch),
// optim.safetensors and state.ojck. dir may not exist yet; its parent must.
// The directory is written beside dir and swapped in whole, so an error
// leaves any earlier checkpoint as it was. A poisoned trainer is refused.
func SaveCheckpoint(ctx context.Context, id uint64, dir string) error {
	if len(dir) > inlineLimit {
		return errors.New("path exceeds 4096 bytes")
	}
	payload := append(binary.LittleEndian.AppendUint64(nil, id), dir...)
	_, err := callEngine(ctx, opSave, payload)
	return err
}

// Resume returns a new model id with the trainer saved in dir, continuing
// the same run on opts.Device. cfg must equal the saved TrainConfig field
// for field, TokenizerHash included; this is how the run is identified, so
// Resume takes it rather than trusting the directory. A refusal creates
// nothing.
func Resume(ctx context.Context, dir string, opts LoadOptions, cfg TrainConfig) (uint64, error) {
	if len(dir) > inlineLimit {
		return 0, errors.New("path exceeds 4096 bytes")
	}
	r := cfg.put(opts.put(new(record))).str(tagPath, dir)
	return session(ctx, opResume, r.bytes(nil))
}

// LoadTokenizer gives id a GPT-2 byte-level BPE from a Hugging Face
// vocab.json and merges.txt under the model root (each at most 32 MiB).
func LoadTokenizer(ctx context.Context, id uint64, vocabJSON, mergesTXT string) error {
	r := new(record).str(tagVocabJSON, vocabJSON).str(tagMergesTXT, mergesTXT)
	_, err := callEngine(ctx, opTokenizer, r.bytes(binary.LittleEndian.AppendUint64(nil, id)))
	return err
}

// Tokenize encodes text with id's tokenizer (GPT-2 ordinary encoding:
// special tokens are plain text).
func Tokenize(ctx context.Context, id uint64, text string) ([]uint32, error) {
	payload := binary.LittleEndian.AppendUint64(nil, id)
	payload = binary.LittleEndian.AppendUint32(payload, tokenizeText)
	out, err := callEngine(ctx, opTokenize, append(payload, text...))
	if err != nil {
		return nil, err
	}
	return decodeIDs(out)
}

// Detokenize decodes ids with id's tokenizer. Ids that do not form UTF-8
// are an error.
func Detokenize(ctx context.Context, id uint64, ids []uint32) (string, error) {
	payload := binary.LittleEndian.AppendUint64(nil, id)
	payload = binary.LittleEndian.AppendUint32(payload, tokenizeIDs)
	for _, v := range ids {
		payload = binary.LittleEndian.AppendUint32(payload, v)
	}
	out, err := callEngine(ctx, opTokenize, payload)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// SampleOptions controls GenerateIDs. Temperature 0 is greedy. TopK 0 and
// TopP 0 are off; otherwise TopK >= 1 and TopP in (0, 1].
type SampleOptions struct {
	Temperature  float32
	TopK         uint32
	TopP         float32
	Seed         uint64
	MaxNewTokens uint32
	// Stop ends generation right after one of these ids is emitted; that id
	// is the last returned.
	Stop []uint32
}

// GenerateIDs continues prompt with id's current weights (its trainer's,
// once one is open) on its device and returns the new ids. The prompt and
// every new id but the last must fit the model's context (MaxSeq);
// otherwise it is ErrCapacity before anything runs. Each forward reads back
// one logit row.
func GenerateIDs(ctx context.Context, id uint64, prompt []uint32, opts SampleOptions) ([]uint32, error) {
	r := new(record).f32(tagTemperature, opts.Temperature)
	if opts.TopK != 0 {
		r.u32(tagTopK, opts.TopK)
	}
	if opts.TopP != 0 {
		r.f32(tagTopP, opts.TopP)
	}
	r.u64(tagSeed, opts.Seed).u32(tagMaxNew, opts.MaxNewTokens)
	if len(opts.Stop) > 0 {
		r.u32s(tagStop, opts.Stop)
	}
	r.u32s(tagPrompt, prompt)
	out, err := callEngine(ctx, opSample, r.bytes(binary.LittleEndian.AppendUint64(nil, id)))
	if err != nil {
		return nil, err
	}
	return decodeIDs(out)
}

// GenerateGreedy is GenerateIDs with temperature 0 and one new token.
func GenerateGreedy(ctx context.Context, id uint64, prompt []uint32) (uint32, error) {
	ids, err := GenerateIDs(ctx, id, prompt, SampleOptions{MaxNewTokens: 1})
	if err != nil {
		return 0, err
	}
	if len(ids) != 1 {
		return 0, errors.New("generate: short result")
	}
	return ids[0], nil
}

// Generate returns argmax of caller logits on the host. It needs a live id
// but reads no model. A non-finite logit is ErrNonFinite.
func Generate(ctx context.Context, id uint64, logits []float32) (uint32, error) {
	if err := fitUint32(len(logits)); err != nil {
		return 0, err
	}
	buf := make([]byte, 0, 16+len(logits)*4)
	buf = binary.LittleEndian.AppendUint64(buf, id)
	buf = binary.LittleEndian.AppendUint32(buf, genLogits)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(logits)))
	for _, v := range logits {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
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

// DefaultMemoryCeiling is the process-wide memory ceiling until
// SetMemoryCeiling changes it: 1 GiB.
const DefaultMemoryCeiling uint64 = 1 << 30

// SetMemoryCeiling replaces the process-wide byte ceiling every model's
// budget (LoadOptions.BudgetBytes) draws from. All open models together
// never account more than the ceiling; a charge past it is ErrCapacity.
//
// It is refused for 0, and while any model is open or still being built or
// released, because a ceiling cannot change under live models without
// splitting their accounting: Free every id (or Close) first. A refusal
// changes nothing. The ceiling outlives Close; it changes only through this
// call.
func SetMemoryCeiling(ctx context.Context, bytes uint64) error {
	_, err := callEngine(ctx, opSetCeiling, binary.LittleEndian.AppendUint64(nil, bytes))
	return err
}

// Free drops a model id. An unknown id, including a second free, is an
// error. A call already running on the id finishes first.
func Free(ctx context.Context, id uint64) error {
	_, err := callEngine(ctx, opFree, binary.LittleEndian.AppendUint64(nil, id))
	return err
}

// Close joins the worker pool. It returns when that join finishes or ctx
// ends, whichever comes first. A cancelled or expired context does not
// open a second pool: if the join is still running, later calls wait for
// it or refuse. If gusset reports that workers are still running after
// its join budget (30s), Close returns that error and this process will
// not open another pool on top of those workers. Close drops every model.
func Close(ctx context.Context) error {
	if ctx == nil {
		return errors.New("ojas: nil context")
	}
	return closeHandle(ctx)
}

// session runs an op that creates a model id and returns it.
func session(ctx context.Context, op uint32, payload []byte) (uint64, error) {
	out, err := callEngine(ctx, op, payload)
	if err != nil {
		return 0, err
	}
	if len(out) != 12 {
		return 0, errors.New("load: short result")
	}
	return binary.LittleEndian.Uint64(out[:8]), nil
}

func decodeIDs(out []byte) ([]uint32, error) {
	if len(out)%4 != 0 {
		return nil, errors.New("ojas: result is not whole uint32 ids")
	}
	ids := make([]uint32, len(out)/4)
	for i := range ids {
		ids[i] = binary.LittleEndian.Uint32(out[4*i:])
	}
	return ids, nil
}

func poisonHandle(ctx context.Context) error {
	_, err := callEngine(ctx, opPanic, nil)
	return err
}

func fitUint32(n int) error {
	if n < 0 || uint64(n) > math.MaxUint32 {
		return errors.New("ojas: length exceeds uint32")
	}
	return nil
}
