// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

//go:build windows

package ui

import (
	"errors"
	"os"
	"time"
	"unsafe"

	"github.com/erikgeiser/coninput"
	"golang.org/x/sys/windows"
)

// winConsole is the process console for the glyph measurement (GL-03): the
// standard input and output handles, read and written as console records.
type winConsole struct {
	in  windows.Handle
	out windows.Handle
	w   *os.File
}

var procWriteConsoleInputW = windows.NewLazySystemDLL("kernel32.dll").NewProc("WriteConsoleInputW")

func openProbeConsole() probeConsole {
	return &winConsole{in: windows.Handle(os.Stdin.Fd()), out: windows.Handle(os.Stdout.Fd()), w: os.Stdout}
}

// Modes reads both console modes; an error (redirected input or output, no
// console) gives the fallback set at once (GLP S3).
func (c *winConsole) Modes() (uint32, uint32, error) {
	var in, out uint32
	if err := windows.GetConsoleMode(c.in, &in); err != nil {
		return 0, 0, errors.New("input: " + err.Error())
	}
	if err := windows.GetConsoleMode(c.out, &out); err != nil {
		return 0, 0, errors.New("output: " + err.Error())
	}
	return in, out, nil
}

func (c *winConsole) SetIn(mode uint32) error  { return windows.SetConsoleMode(c.in, mode) }
func (c *winConsole) SetOut(mode uint32) error { return windows.SetConsoleMode(c.out, mode) }
func (c *winConsole) Write(b []byte) (int, error) {
	return c.w.Write(b)
}

// Read waits on the input handle until the deadline and reads the records
// that arrived (with VT input on, the cursor reply arrives as key records
// with virtual key 0 carrying its characters; G009, G010).
func (c *winConsole) Read(deadline time.Time) ([]ConRecord, error) {
	d := time.Until(deadline)
	if d <= 0 {
		return nil, nil
	}
	ms := (d + time.Millisecond - 1) / time.Millisecond // a part of a millisecond waits one
	ev, err := windows.WaitForSingleObject(c.in, uint32(ms))
	if err != nil || ev != windows.WAIT_OBJECT_0 {
		return nil, err
	}
	recs, err := coninput.ReadNConsoleInputs(c.in, 64)
	out := make([]ConRecord, 0, len(recs))
	for _, r := range recs {
		cr := ConRecord{Raw: r}
		if k, ok := r.Unwrap().(coninput.KeyEventRecord); ok {
			cr.Key, cr.Down, cr.VK, cr.Char = true, k.KeyDown, uint16(k.VirtualKeyCode), k.Char
		}
		out = append(out, cr)
	}
	return out, err
}

// Forward writes the kept key records back to the console input
// (WriteConsoleInputW), after the modes are restored: the UI library reads
// them as the user's keys (GLP step 5; S8-LA-08).
func (c *winConsole) Forward(recs []ConRecord) error {
	raw := make([]coninput.InputRecord, 0, len(recs))
	for _, r := range recs {
		if ir, ok := r.Raw.(coninput.InputRecord); ok {
			raw = append(raw, ir)
		}
	}
	if len(raw) == 0 {
		return nil
	}
	var n uint32
	r1, _, e := procWriteConsoleInputW.Call(uintptr(c.in), uintptr(unsafe.Pointer(&raw[0])), uintptr(len(raw)), uintptr(unsafe.Pointer(&n)))
	if r1 == 0 {
		return e
	}
	return nil
}
