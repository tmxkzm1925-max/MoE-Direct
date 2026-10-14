//go:build windows

package core

import (
	"os"
	"path/filepath"
	"testing"
)

// TestOSIOMeterSmall reads a 4 MiB scratch file through the real meter (overlapped unbuffered
// reads, then the synchronous probe reads) with a 50 ms window: the reads complete and are
// counted. It times nothing it asserts on; the speed check's arithmetic is TestS8LA38's.
func TestOSIOMeterSmall(t *testing.T) {
	m := osIOMeter{}
	dir := t.TempDir()
	p := filepath.Join(dir, "experts.bin")
	if err := m.WriteScratch(p, 4*1048576); err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(p); err != nil || st.Size() != 4*1048576 {
		t.Fatalf("scratch %v %v", st, err)
	}
	phys := m.PhysicalSectorBytes(p)
	if phys <= 0 {
		phys = 4096
	}
	offs := []int64{0, 1048576, 2097152, 3145728}
	for _, qd := range []int{1, 2, 4, 8} {
		r := m.RunPoint(p, qd, 1048576, offs, 50, phys)
		if !r.OK || r.QD != qd || r.Reads <= 0 || r.Bytes != int64(r.Reads)*1048576 || r.Ticks <= 0 {
			t.Errorf("QD%d: %+v", qd, r)
		}
	}
	pr, err := m.RandomRead(p, 65536, 16)
	if err != nil || pr.Samples != 16 || pr.Bytes != 16*65536 || pr.MiBps <= 0 {
		t.Errorf("probe %+v %v", pr, err)
	}
}
