package ojas

import (
	"errors"
	"strings"
	"testing"

	"github.com/bharathvbcr/gusset"
)

// ft-757cb6b2 E7: a device wait that ran out of its budget is retryable and
// is not a lost device. The Rust side maps ojas_core::OjasError::WaitTimeout
// to the "ojas:E_TIMEOUT:" kind (ojas-capi/src/lib.rs kind_of); this is the
// decoder's half.
//
// A real timeout cannot be reached through the C ABI from a Go test: it needs
// a GPU backlog longer than tessl's event-wait budget, and the budget is set
// by TESSL_EVENT_WAIT_MS inside the engine process, with 1 ms as its floor.
// So the narrowest seam is tested: the kind prefix an engine error carries
// selects ErrTimeout, and only it. The Rust test
// error_kinds_follow_the_ojas_error_variant covers the other half (variant
// to kind), and ojas-metal's wait_timeout_tests cover making the timeout
// happen on a real device.
func TestATimedOutDeviceWaitIsErrTimeoutAndNotDeviceLost(t *testing.T) {
	msgs := []string{
		"ojas:E_TIMEOUT: train_step: backend Metal: device wait timed out after 30000 ms (retryable): waited",
		"ojas:E_TIMEOUT: sync",
	}
	for _, msg := range msgs {
		err := annotateEngineError(&gusset.Error{Code: 1, Msg: msg})
		if !errors.Is(err, ErrTimeout) {
			t.Errorf("%q: errors.Is(ErrTimeout) = false: %v", msg, err)
		}
		for _, s := range kindSentinels() {
			if s != ErrTimeout && errors.Is(err, s) {
				t.Errorf("%q also selects %v", msg, s)
			}
		}
		if errors.Is(err, ErrDeviceLost) {
			t.Errorf("%q: a timeout is retryable, not a lost device", msg)
		}
		if !strings.Contains(err.Error(), msg) {
			t.Errorf("%q: message dropped: %v", msg, err)
		}
	}
	// The kind selects a sentinel only at the start of the message: a user
	// path that spells it selects nothing.
	echoed := annotateEngineError(errors.New("missing file: /data/ojas:E_TIMEOUT: x"))
	if errors.Is(echoed, ErrTimeout) {
		t.Errorf("an echoed path selected ErrTimeout: %v", echoed)
	}
	if ErrTimeout.Error() == ErrDeviceLost.Error() {
		t.Error("ErrTimeout and ErrDeviceLost read the same")
	}
}
