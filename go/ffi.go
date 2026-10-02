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
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/bharathvbcr/gusset"
)

// Opcodes of ojas-capi (ojas-capi/src/engine.rs). Opcode 15 sets the
// process memory ceiling (SetMemoryCeiling); 16 reads the host profile
// (SystemProfile). Opcode 2 was the
// payload-only Step; it is retired and not reused.
const (
	opLoad          uint32 = 1
	opGenerate      uint32 = 3
	opFree          uint32 = 4
	opPanic         uint32 = 5
	opNew           uint32 = 6
	opTrainOpen     uint32 = 7
	opTrainStep     uint32 = 8
	opSave          uint32 = 9
	opResume        uint32 = 10
	opTokenizer     uint32 = 11
	opTokenize      uint32 = 12
	opSample        uint32 = 13
	opInspect       uint32 = 14
	opSetCeiling    uint32 = 15
	opSystemProfile uint32 = 16
	genLogits       uint32 = 1
	stepSampled     uint32 = 0
	stepTokens      uint32 = 1
	tokenizeText    uint32 = 1
	tokenizeIDs     uint32 = 2

	deviceCPU         uint32 = 0
	deviceCPUParallel uint32 = 1
	deviceMetal       uint32 = 2
	deviceWgpu        uint32 = 3

	inlineLimit = 4096

	// defaultBufferBudget is the production live-NewBuffer cap. One gusset
	// buffer may be 1 GiB, and NewBuffer runs before the pool semaphore, so
	// an unlimited budget lets every queued caller hold that much. 64 MiB
	// is shared across those callers. 0 means unlimited; tests pass 0 or a
	// tiny cap through SetBufferBudget before the handle opens.
	defaultBufferBudget int64 = 64 << 20

	// memoryGovernorInterval bounds how often an opted-in governor calls
	// AdviseMemoryLimit. It is not once per step.
	memoryGovernorInterval = time.Second

	prefixCapacity   = "ojas:E_CAPACITY:"
	prefixDeviceLost = "ojas:E_DEVICE_LOST:"
	prefixBusy       = "ojas:E_BUSY:"
	prefixNonFinite  = "ojas:E_NONFINITE:"
	prefixPoisoned   = "ojas:E_POISONED:"
)

// Option-record tags (ojas-capi/src/wire.rs, mod tag). One numbering for
// every record.
const (
	tagPath     uint32 = 1
	tagDevice   uint32 = 2
	tagThreads  uint32 = 3
	tagBudget   uint32 = 4
	tagNumerics uint32 = 5
	tagSeed     uint32 = 6

	tagVocab    uint32 = 10
	tagNEmbd    uint32 = 11
	tagNLayer   uint32 = 12
	tagNHead    uint32 = 13
	tagNKVHead  uint32 = 14
	tagHeadDim  uint32 = 15
	tagHidden   uint32 = 16
	tagMaxSeq   uint32 = 17
	tagRopeBase uint32 = 18
	tagRMSEps   uint32 = 19

	tagTokenBin      uint32 = 20
	tagBinFormat     uint32 = 21
	tagBatch         uint32 = 22
	tagSeq           uint32 = 23
	tagAccum         uint32 = 24
	tagDataSeed      uint32 = 25
	tagSchedule      uint32 = 26
	tagWarmup        uint32 = 27
	tagTotal         uint32 = 28
	tagDecayFrac     uint32 = 29
	tagMatrixLR      uint32 = 30
	tagAdamLR        uint32 = 31
	tagGradClip      uint32 = 32
	tagOnNonFinite   uint32 = 33
	tagTokenizerHash uint32 = 34

	tagVocabJSON uint32 = 40
	tagMergesTXT uint32 = 41

	tagTemperature uint32 = 50
	tagTopK        uint32 = 51
	tagTopP        uint32 = 52
	tagMaxNew      uint32 = 53
	tagStop        uint32 = 54
	tagPrompt      uint32 = 55
)

// record writes one option record: count u32, then tag u32, len u32, value
// per field, little-endian. Rust refuses an unknown or repeated tag.
type record struct {
	count uint32
	body  []byte
}

func (r *record) raw(tag uint32, value []byte) *record {
	r.count++
	r.body = binary.LittleEndian.AppendUint32(r.body, tag)
	r.body = binary.LittleEndian.AppendUint32(r.body, uint32(len(value)))
	r.body = append(r.body, value...)
	return r
}

func (r *record) u32(tag, v uint32) *record {
	return r.raw(tag, binary.LittleEndian.AppendUint32(nil, v))
}

func (r *record) u64(tag uint32, v uint64) *record {
	return r.raw(tag, binary.LittleEndian.AppendUint64(nil, v))
}

func (r *record) f32(tag uint32, v float32) *record {
	return r.u32(tag, math.Float32bits(v))
}

func (r *record) f64(tag uint32, v float64) *record {
	return r.u64(tag, math.Float64bits(v))
}

func (r *record) str(tag uint32, v string) *record {
	return r.raw(tag, []byte(v))
}

func (r *record) u32s(tag uint32, v []uint32) *record {
	b := make([]byte, 0, 4*len(v))
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, x)
	}
	return r.raw(tag, b)
}

// bytes appends the record to prefix.
func (r *record) bytes(prefix []byte) []byte {
	out := binary.LittleEndian.AppendUint32(prefix, r.count)
	return append(out, r.body...)
}

var (
	handleMu sync.Mutex
	handle   *gusset.Handle
	poisoned atomic.Bool
	poolSize atomic.Uint32
	// bufferBudget is applied on the next Open. Negative values are refused
	// by SetBufferBudget. Zero is unlimited.
	bufferBudget atomic.Int64
	closeMu      sync.Mutex
	calls        callGate

	// closeJoining is set for the whole gusset Close, including after Close's
	// context returns early. ensureHandle refuses a new pool while it is set.
	closeJoining atomic.Bool
	joinDone     chan struct{}
	joinErr      error

	// closeFailed is set when gusset Close reports workers still running.
	// The pool stays allocated until those workers exit, so a later Open
	// would be a second pool on the same engine.
	closeFailed atomic.Pointer[closeFailure]

	// memoryGovernorTotal is 0 until WithMemoryGovernor. Refresh uses that
	// captured total; it does not read a new machine size.
	memoryGovernorTotal atomic.Int64
	memoryGovernorLast  atomic.Int64

	errHandleClosed = errors.New("ojas: handle is closed")
	errCloseJoining = errors.New("ojas: close is still joining the worker pool")
)

type closeFailure struct {
	err error
}

// callGate counts calls that have entered and not yet returned.
// Close sets closed, waits until the count is zero, then drops the handle.
// The mutex covers that count only. It is not held across the gusset call,
// so two sessions still overlap.
type callGate struct {
	mu     sync.Mutex
	closed bool
	n      int
	idle   sync.Cond
}

func (g *callGate) enter() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if err := stuckCloseError(); err != nil {
		return err
	}
	// Closed covers the join too. Callers already treat "handle is closed"
	// as the answer while Close is dropping the pool.
	if g.closed || closeJoining.Load() {
		return errHandleClosed
	}
	g.n++
	return nil
}

func (g *callGate) leave() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.n--
	if g.n == 0 {
		g.idle.Broadcast()
	}
}

// drain marks the gate closed and waits until in-flight calls leave, or
// until ctx ends. An already-ended context returns before closed is set.
// A deadline that lands during the wait returns with closed already set;
// the caller reopens when it is aborting the close before dropping the handle.
func (g *callGate) drain(ctx context.Context) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	g.closed = true
	if g.n == 0 {
		return nil
	}
	stopped := make(chan struct{})
	defer close(stopped)
	go func() {
		select {
		case <-ctx.Done():
			g.mu.Lock()
			g.idle.Broadcast()
			g.mu.Unlock()
		case <-stopped:
		}
	}()
	for g.n > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		g.idle.Wait()
	}
	return ctx.Err()
}

func (g *callGate) reopen() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closed = false
}

func init() {
	poolSize.Store(1)
	bufferBudget.Store(defaultBufferBudget)
	calls.idle.L = &calls.mu
}

// SetPoolSize chooses the gusset worker count used by the next Open.
// The size is refused outside 1..=gusset.MaxPoolSize and after the handle
// already exists. The default is 1.
func SetPoolSize(n int) error {
	if n < 1 || n > gusset.MaxPoolSize {
		return fmt.Errorf("ojas: pool size is outside 1..=%d", gusset.MaxPoolSize)
	}
	handleMu.Lock()
	defer handleMu.Unlock()
	if handle != nil {
		return errors.New("ojas: pool size is fixed after the handle is open")
	}
	poolSize.Store(uint32(n))
	return nil
}

// WithDerivedPoolSize sets the next Open's worker count from
// runtime.GOMAXPROCS(0), at least 1 and at most gusset.MaxPoolSize.
//
// It does not change the default of 1. Call it before the first engine
// call, the same way as SetPoolSize.
func WithDerivedPoolSize() error {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	if n > gusset.MaxPoolSize {
		n = gusset.MaxPoolSize
	}
	return SetPoolSize(n)
}

// SetBufferBudget sets the live NewBuffer cap used by the next Open.
// bytes == 0 leaves the budget unlimited. A negative value is refused.
// The production default, applied when this is not called, is 64 MiB.
func SetBufferBudget(bytes int64) error {
	if bytes < 0 {
		return errors.New("ojas: buffer budget is negative")
	}
	handleMu.Lock()
	defer handleMu.Unlock()
	if handle != nil {
		return errors.New("ojas: buffer budget is fixed after the handle is open")
	}
	bufferBudget.Store(bytes)
	return nil
}

// Stats forwards the gusset allocator counters.
func Stats() gusset.AllocStats {
	return gusset.Stats()
}

// AdviseMemoryLimit forwards total to gusset. The returned value is the
// previous Go runtime memory limit (what debug.SetMemoryLimit returns),
// not the limit just installed. This overwrites GOMEMLIMIT. The governor
// is the supported caller; do not call this from a request path unless
// that overwrite is what you want.
func AdviseMemoryLimit(total int64) int64 {
	return gusset.AdviseMemoryLimit(total)
}

// WithMemoryGovernor opts in to overwriting GOMEMLIMIT. total is captured
// once (a later call replaces that capture; nothing re-reads the machine).
// AdviseMemoryLimit runs immediately, then at most once per second from
// engine calls. It is not called otherwise, and Open does not call it.
// total must be positive.
func WithMemoryGovernor(total int64) error {
	if total <= 0 {
		return errors.New("ojas: memory governor total must be positive")
	}
	memoryGovernorTotal.Store(total)
	AdviseMemoryLimit(total)
	memoryGovernorLast.Store(time.Now().UnixNano())
	return nil
}

func refreshMemoryGovernor() {
	total := memoryGovernorTotal.Load()
	if total <= 0 {
		return
	}
	now := time.Now().UnixNano()
	last := memoryGovernorLast.Load()
	if last != 0 && now-last < int64(memoryGovernorInterval) {
		return
	}
	if !memoryGovernorLast.CompareAndSwap(last, now) {
		return
	}
	AdviseMemoryLimit(total)
}

func engineInit() error {
	if C.ojas_engine_init() == 0 {
		return nil
	}
	return annotateEngineError(errors.New(lastCError()))
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
	return annotateEngineError(errors.New(lastCError()))
}

func callEngine(ctx context.Context, opcode uint32, payload []byte) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("ojas: nil context")
	}
	if err := calls.enter(); err != nil {
		return nil, err
	}
	defer calls.leave()
	refreshMemoryGovernor()
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
	return out, annotateEngineError(err)
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
	if err := poolBlock(); err != nil {
		return nil, err
	}
	handleMu.Lock()
	defer handleMu.Unlock()
	if err := poolBlock(); err != nil {
		return nil, err
	}
	if handle != nil {
		return handle, nil
	}
	if err := engineInit(); err != nil {
		return nil, err
	}
	n := int(poolSize.Load())
	h, err := gusset.Open(gusset.WithPoolSize(n), gusset.WithBufferBudget(bufferBudget.Load()))
	if err != nil {
		return nil, err
	}
	handle = h
	return h, nil
}

func closeHandle(ctx context.Context) error {
	closeMu.Lock()
	if err := stuckCloseError(); err != nil {
		closeMu.Unlock()
		return err
	}
	if closeJoining.Load() {
		done := joinDone
		closeMu.Unlock()
		return waitForJoin(ctx, done)
	}
	if err := ctx.Err(); err != nil {
		closeMu.Unlock()
		return err
	}
	// Wait out calls that already entered. A call that arrives now gets
	// errHandleClosed instead of sharing the handle Close is about to drop.
	if err := calls.drain(ctx); err != nil {
		calls.reopen()
		closeMu.Unlock()
		return err
	}
	// Drop every Rust session while no call is inside, so a raced Free is
	// not required to release the 64-session cap.
	engineReset()
	handleMu.Lock()
	poisoned.Store(false)
	h := handle
	handle = nil
	handleMu.Unlock()
	if h == nil {
		calls.reopen()
		closeMu.Unlock()
		return nil
	}

	done := make(chan struct{})
	joinDone = done
	closeJoining.Store(true)
	closeMu.Unlock()

	errc := make(chan error, 1)
	go func() {
		errc <- h.Close()
	}()
	select {
	case err := <-errc:
		return completeJoin(done, err)
	case <-ctx.Done():
		// The join keeps running. completeJoin refuses a second pool until
		// it finishes, and never reopens if workers are still inside it.
		go func() {
			completeJoin(done, <-errc)
		}()
		return ctx.Err()
	}
}

func waitForJoin(ctx context.Context, done chan struct{}) error {
	if done == nil {
		return errCloseJoining
	}
	select {
	case <-done:
		closeMu.Lock()
		defer closeMu.Unlock()
		if err := stuckCloseError(); err != nil {
			return err
		}
		return joinErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

func completeJoin(done chan struct{}, err error) error {
	closeMu.Lock()
	defer closeMu.Unlock()
	// A second completion is a no-op. The deadline path and the inline path
	// cannot both win the select, but a test can call this once.
	select {
	case <-done:
		if err := stuckCloseError(); err != nil {
			return err
		}
		return joinErr
	default:
	}
	err = releaseAfterClose(err)
	joinErr = err
	closeJoining.Store(false)
	close(done)
	return err
}

// releaseAfterClose reopens the gate only when the pool has actually gone.
// A "workers still running" error means gusset kept the pool alive, so the
// gate stays closed and a later Open is refused.
func releaseAfterClose(err error) error {
	if closeLeftWorkersRunning(err) {
		closeFailed.Store(&closeFailure{err: err})
		return err
	}
	// Clear the joining flag before the gate opens, so a caller cannot
	// observe an open gate and still be told a join is in progress.
	closeJoining.Store(false)
	calls.reopen()
	return err
}

func closeLeftWorkersRunning(err error) bool {
	return err != nil && strings.Contains(err.Error(), "workers still running")
}

func stuckCloseError() error {
	if f := closeFailed.Load(); f != nil && f.err != nil {
		return f.err
	}
	return nil
}

func poolBlock() error {
	if err := stuckCloseError(); err != nil {
		return err
	}
	if closeJoining.Load() {
		return errCloseJoining
	}
	return nil
}

// inBandSentinel is the sentinel for an engine message that starts with an
// in-band kind. Rust puts the kind first and only where a typed error made
// it (ojas-capi/src/lib.rs kind_of), so a prefix anywhere else, such as
// inside a user path echoed in the message, selects nothing (finding F10).
func inBandSentinel(msg string) error {
	switch {
	case strings.HasPrefix(msg, prefixCapacity):
		return ErrCapacity
	case strings.HasPrefix(msg, prefixDeviceLost):
		return ErrDeviceLost
	case strings.HasPrefix(msg, prefixBusy):
		return ErrBusy
	case strings.HasPrefix(msg, prefixNonFinite):
		return ErrNonFinite
	case strings.HasPrefix(msg, prefixPoisoned):
		return ErrPoisoned
	default:
		return nil
	}
}

// engineMessage is the engine's own text: a gusset.Error's Msg, which is the
// string the Rust handler returned, or the text of an error made from
// ojas_take_last_error. gusset's Error() adds a "gusset error [n]:" head.
func engineMessage(err error) string {
	var ge *gusset.Error
	if errors.As(err, &ge) {
		return ge.Msg
	}
	return err.Error()
}

// annotateEngineError maps an in-band ojas:E_* prefix onto a sentinel.
// The original error stays in the chain, so its text and errors.Is targets
// are still visible. A message with no prefix is returned unchanged.
func annotateEngineError(err error) error {
	if err == nil {
		return nil
	}
	sentinel := inBandSentinel(engineMessage(err))
	if sentinel == nil {
		return err
	}
	return fmt.Errorf("%w: %w", sentinel, err)
}
