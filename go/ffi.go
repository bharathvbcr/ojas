package ojas

/*
#cgo noescape ojas_engine_init
#cgo nocallback ojas_engine_init
#cgo noescape ojas_engine_reset
#cgo nocallback ojas_engine_reset
#cgo noescape ojas_set_model_root
#cgo nocallback ojas_set_model_root
#cgo noescape ojas_take_last_error
#cgo nocallback ojas_take_last_error
#include <stddef.h>
int ojas_engine_init(void);
void ojas_engine_reset(void);
int ojas_set_model_root(const unsigned char *ptr, size_t len);
size_t ojas_take_last_error(unsigned char *dst, size_t cap);
*/
import "C"

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/bharathvbcr/gusset"
)

const (
	opLoad     uint32 = 1
	opStep     uint32 = 2
	opGenerate uint32 = 3
	opFree     uint32 = 4
	opPanic    uint32 = 5

	stepHeader uint32 = 0
	stepLogits uint32 = 1
	stepTokens uint32 = 2
	genLogits  uint32 = 1
	genGreedy  uint32 = 2

	deviceCPU         uint32 = 0
	deviceCPUParallel uint32 = 1
	deviceMetal       uint32 = 2
	deviceWgpu        uint32 = 3

	inlineLimit = 4096
	maxPoolSize = 1024
)

var (
	handleMu sync.Mutex
	handle   *gusset.Handle
	poisoned atomic.Bool
	poolSize atomic.Uint32
)

func init() {
	poolSize.Store(1)
}

// SetPoolSize chooses the gusset worker count used by the next Open.
// The size is refused outside 1..=1024 and after the handle already exists.
func SetPoolSize(n int) error {
	if n < 1 || n > maxPoolSize {
		return errors.New("ojas: pool size is outside 1..=1024")
	}
	handleMu.Lock()
	defer handleMu.Unlock()
	if handle != nil {
		return errors.New("ojas: pool size is fixed after the handle is open")
	}
	poolSize.Store(uint32(n))
	return nil
}

// Stats forwards the gusset allocator counters.
func Stats() gusset.AllocStats {
	return gusset.Stats()
}

// AdviseMemoryLimit forwards the call to gusset. The returned value is the
// limit gusset installed.
func AdviseMemoryLimit(total int64) int64 {
	return gusset.AdviseMemoryLimit(total)
}

func engineInit() error {
	if C.ojas_engine_init() == 0 {
		return nil
	}
	return errors.New(lastCError())
}

func engineReset() {
	C.ojas_engine_reset()
}

func lastCError() string {
	scratch := make([]byte, 512)
	full := int(C.ojas_take_last_error((*C.uchar)(unsafe.Pointer(&scratch[0])), C.size_t(len(scratch))))
	runtime.KeepAlive(scratch)
	if full <= 0 {
		return "ojas engine call failed"
	}
	if full > len(scratch) {
		scratch = make([]byte, full)
		full = int(C.ojas_take_last_error((*C.uchar)(unsafe.Pointer(&scratch[0])), C.size_t(len(scratch))))
		runtime.KeepAlive(scratch)
		if full > len(scratch) {
			full = len(scratch)
		}
	}
	return string(scratch[:full])
}

func setRoot(dir string) error {
	b := []byte(dir)
	var rc C.int
	if len(b) > 0 {
		rc = C.ojas_set_model_root((*C.uchar)(unsafe.Pointer(&b[0])), C.size_t(len(b)))
	} else {
		rc = C.ojas_set_model_root(nil, 0)
	}
	runtime.KeepAlive(b)
	if rc == 0 {
		return nil
	}
	return errors.New(lastCError())
}

func callEngine(ctx context.Context, opcode uint32, payload []byte) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("ojas: nil context")
	}
	if poisoned.Load() {
		return nil, gusset.ErrPoisoned
	}
	h, err := ensureHandle()
	if err != nil {
		return nil, err
	}
	ctx = gusset.ContextWithOpcode(ctx, opcode)
	var out []byte
	if len(payload) <= inlineLimit {
		out, err = h.Call(ctx, payload)
	} else {
		out, err = callBuffer(h, ctx, payload)
	}
	if errors.Is(err, gusset.ErrPanic) || errors.Is(err, gusset.ErrPoisoned) {
		poisoned.Store(true)
		engineReset()
	}
	return out, err
}

func callBuffer(h *gusset.Handle, ctx context.Context, payload []byte) ([]byte, error) {
	buf, err := h.NewBuffer(len(payload))
	if err != nil {
		return nil, err
	}
	defer buf.Free()
	dst := buf.Bytes()
	if len(dst) < len(payload) {
		return nil, errors.New("ojas: buffer view is short")
	}
	copy(dst, payload)
	out, err := h.CallBuffer(ctx, buf)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	view := out.Bytes()
	copied := make([]byte, len(view))
	copy(copied, view)
	runtime.KeepAlive(out)
	return copied, nil
}

func ensureHandle() (*gusset.Handle, error) {
	handleMu.Lock()
	defer handleMu.Unlock()
	if handle != nil {
		return handle, nil
	}
	if err := engineInit(); err != nil {
		return nil, err
	}
	n := int(poolSize.Load())
	h, err := gusset.Open(gusset.WithPoolSize(n))
	if err != nil {
		return nil, err
	}
	handle = h
	return h, nil
}

func closeHandle() error {
	handleMu.Lock()
	defer handleMu.Unlock()
	poisoned.Store(false)
	h := handle
	handle = nil
	if h == nil {
		return nil
	}
	return h.Close()
}
