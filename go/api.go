// Package ojas is the in-process training and inference API.
//
// One gusset handle, pool size 1, runs every step. Load sends a relative
// path (at most 4 KiB). Larger step and generate payloads go through a
// gusset Buffer. Weights are not sent.
//
// Link the umbrella archive before `go test`. The gusset module's default
// linker path is its own checkout, which does not contain ojas_engine_init,
// so the test uses pkg-config:
//
//	cargo build -p ojas-gusset-engine
//	cd go && PKG_CONFIG_PATH="$PWD" go test -tags gusset_pkgconfig -count=1 -timeout 10m
package ojas

import (
	"encoding/binary"
	"errors"
	"math"
)

// SetModelRoot is the directory relative load paths are resolved under.
func SetModelRoot(dir string) error {
	mu.Lock()
	defer mu.Unlock()
	return setRoot(dir)
}

// Load opens a relative safetensors path and returns a model id.
func Load(path string) (uint64, error) {
	mu.Lock()
	defer mu.Unlock()
	if len(path) > inlineLimit {
		return 0, errors.New("path exceeds 4096 bytes")
	}
	out, err := callLocked(opLoad, []byte(path))
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

// Step runs one step on a loaded session. Rust errors are returned as text.
func Step(id uint64, req StepRequest) (StepStats, error) {
	mu.Lock()
	defer mu.Unlock()
	payload, err := encodeStep(id, req)
	if err != nil {
		return StepStats{}, err
	}
	out, err := callLocked(opStep, payload)
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
func Generate(id uint64, logits []float32) (uint32, error) {
	mu.Lock()
	defer mu.Unlock()
	return generate(id, genLogits, logits, nil)
}

// GenerateGreedy asks the tiny ojas-infer model for one token.
func GenerateGreedy(id uint64, prompt []uint32) (uint32, error) {
	mu.Lock()
	defer mu.Unlock()
	return generate(id, genGreedy, nil, prompt)
}

// Free drops a session. An unknown id, including a second free, is an error.
func Free(id uint64) error {
	mu.Lock()
	defer mu.Unlock()
	var payload [8]byte
	binary.LittleEndian.PutUint64(payload[:], id)
	_, err := callLocked(opFree, payload[:])
	return err
}

// Close joins the worker pool. With no call still running it returns nil.
func Close() error {
	mu.Lock()
	defer mu.Unlock()
	return closeHandle()
}

func encodeStep(id uint64, req StepRequest) ([]byte, error) {
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
		rows := int(req.Batch) * int(req.Seq)
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

func generate(id uint64, mode uint32, logits []float32, prompt []uint32) (uint32, error) {
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
	out, err := callLocked(opGenerate, buf)
	if err != nil {
		return 0, err
	}
	if len(out) != 4 {
		return 0, errors.New("generate: short result")
	}
	return binary.LittleEndian.Uint32(out), nil
}

func poisonHandle() error {
	mu.Lock()
	defer mu.Unlock()
	_, err := callLocked(opPanic, nil)
	return err
}
