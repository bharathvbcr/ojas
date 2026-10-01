package ojas

/*
#cgo noescape ojas_engine_init
#cgo nocallback ojas_engine_init
#cgo noescape ojas_engine_reset
#cgo nocallback ojas_engine_reset
#cgo noescape ojas_set_model_root
#cgo nocallback ojas_set_model_root
#cgo noescape ojas_last_error_len
#cgo nocallback ojas_last_error_len
#cgo noescape ojas_copy_last_error
#cgo nocallback ojas_copy_last_error
#include <stddef.h>
int ojas_engine_init(void);
void ojas_engine_reset(void);
int ojas_set_model_root(const unsigned char *ptr, size_t len);
size_t ojas_last_error_len(void);
size_t ojas_copy_last_error(unsigned char *dst, size_t cap);
*/
import "C"

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"time"
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

	inlineLimit = 4096
)

var (
	mu       sync.Mutex
	handle   *gusset.Handle
	poisoned bool
)

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
	n := int(C.ojas_last_error_len())
	if n == 0 {
		return "ojas engine call failed"
	}
	buf := make([]byte, n)
	written := C.ojas_copy_last_error((*C.uchar)(unsafe.Pointer(&buf[0])), C.size_t(n))
	runtime.KeepAlive(buf)
	return string(buf[:int(written)])
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

func callLocked(opcode uint32, payload []byte) ([]byte, error) {
	if poisoned {
		return nil, gusset.ErrPoisoned
	}
	if err := ensureHandle(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ctx = gusset.ContextWithOpcode(ctx, opcode)
	var (
		out []byte
		err error
	)
	if len(payload) <= inlineLimit {
		out, err = handle.Call(ctx, payload)
	} else {
		out, err = callBuffer(ctx, payload)
	}
	if errors.Is(err, gusset.ErrPanic) || errors.Is(err, gusset.ErrPoisoned) {
		poisoned = true
		engineReset()
	}
	return out, err
}

func callBuffer(ctx context.Context, payload []byte) ([]byte, error) {
	buf, err := handle.NewBuffer(len(payload))
	if err != nil {
		return nil, err
	}
	defer buf.Free()
	dst := buf.Bytes()
	if len(dst) < len(payload) {
		return nil, errors.New("ojas: buffer view is short")
	}
	copy(dst, payload)
	out, err := handle.CallBuffer(ctx, buf)
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

func ensureHandle() error {
	if handle != nil {
		return nil
	}
	if err := engineInit(); err != nil {
		return err
	}
	h, err := gusset.Open(gusset.WithPoolSize(1))
	if err != nil {
		return err
	}
	handle = h
	return nil
}

func closeHandle() error {
	poisoned = false
	h := handle
	handle = nil
	if h == nil {
		return nil
	}
	return h.Close()
}
