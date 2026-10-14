package selftest

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

const entryName = "MoE-Direct-launcher-selftest.ps1"

// TestS8HX12 (S8-HX-12; KNOWN row 194; K194): the Go self-test entry starts while other
// processes' command lines only hold its path, and is refused as a duplicate, matched by its
// own -File argument, while one entry runs. Real child processes in both cases. The entry is
// a byte copy in a temporary folder, so a self-test run of the module itself (which runs this
// test) is a different path and does not count as a duplicate here.
func TestS8HX12(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the self-test entry and its guard are Windows PowerShell")
	}
	ps, err := exec.LookPath("powershell.exe")
	if err != nil {
		t.Fatal(err)
	}
	src, err := filepath.Abs(filepath.Join("..", "..", entryName))
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	entry := filepath.Join(dir, entryName)
	other := filepath.Join(dir, "other-script.ps1")
	if err := os.WriteFile(entry, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(other, []byte("param([string] $Path)\r\nStart-Sleep -Seconds 120\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Case 1: three processes whose command lines hold the entry's path, none running it.
	holders := map[string]*exec.Cmd{
		// a shell whose command text says «powershell -NoProfile -File <entry>» (row 194's case)
		"powershell -Command text": exec.Command(ps, "-NoProfile", "-Command",
			"Start-Sleep -Seconds 120 # powershell -NoProfile -File "+entry),
		// another script run with the entry's path as its argument
		"powershell -File other-script <entry>": exec.Command(ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", other, entry),
	}
	shell := exec.Command(os.Getenv("ComSpec"))
	shell.SysProcAttr = &syscall.SysProcAttr{CmdLine: `cmd.exe /d /s /c "rem ` + entry + ` & set /p x="`}
	shellIn, err := shell.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	holders["cmd shell text"] = shell
	for name, c := range holders {
		if err := c.Start(); err != nil {
			t.Fatalf("holder %s: %v", name, err)
		}
		c := c
		t.Cleanup(func() { c.Process.Kill(); c.Wait() })
	}
	t.Cleanup(func() { shellIn.Close() })
	time.Sleep(500 * time.Millisecond)

	out1, code1 := runEntry(t, ps, entry, "-GuardOnly")
	t.Logf("case 1 (path held by %d processes): exit %d: %s", len(holders), code1, out1)
	if code1 != 0 || !strings.Contains(out1, "MoE-Direct-launcher-selftest: guard: start") {
		t.Fatalf("case 1: want exit 0 and the start line, got exit %d: %s", code1, out1)
	}

	// Case 2: one entry runs (guard passed, holding); a second start is refused.
	running := exec.Command(ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", entry, "-GuardOnly", "-HoldSeconds", "120")
	rout, err := running.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := running.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { running.Process.Kill(); running.Wait() })
	waitLine(t, rout, "guard: start", 60*time.Second)

	out2, code2 := runEntry(t, ps, entry, "-GuardOnly")
	t.Logf("case 2 (entry running as pid %d): exit %d: %s", running.Process.Pid, code2, out2)
	want := fmt.Sprintf("MoE-Direct-launcher-selftest: REFUSED duplicate: another process runs this entry: pid %d (-File %s)",
		running.Process.Pid, entry)
	if code2 != 2 || strings.TrimSpace(out2) != want {
		t.Fatalf("case 2: want exit 2 and\n%s\ngot exit %d:\n%s", want, code2, out2)
	}
	for name, c := range holders {
		if strings.Contains(out2, fmt.Sprintf("pid %d ", c.Process.Pid)) {
			t.Fatalf("case 2: the path-holding process %q (pid %d) was matched", name, c.Process.Pid)
		}
		if c.ProcessState != nil {
			t.Fatalf("holder %q ended before the checks: %v", name, c.ProcessState)
		}
	}
	t.Logf("S8-HX-12: case 1 start (exit 0) while %d processes held the path; case 2 refused (exit 2) as a duplicate of pid %d, matched by its -File argument",
		len(holders), running.Process.Pid)
}

// runEntry starts the entry once and returns its output and exit code.
func runEntry(t *testing.T, ps, entry string, args ...string) (string, int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, ps, append([]string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", entry}, args...)...)
	out, err := c.CombinedOutput()
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			t.Fatalf("entry %v: %v", args, err)
		}
	}
	return strings.TrimSpace(string(out)), c.ProcessState.ExitCode()
}

// waitLine reads r until a line contains want, or fails after d.
func waitLine(t *testing.T, r io.Reader, want string, d time.Duration) {
	t.Helper()
	got := make(chan bool, 1)
	go func() {
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			if strings.Contains(sc.Text(), want) {
				got <- true
				io.Copy(io.Discard, r)
				return
			}
		}
		got <- false
	}()
	select {
	case ok := <-got:
		if !ok {
			t.Fatalf("the running entry ended before %q", want)
		}
	case <-time.After(d):
		t.Fatalf("no %q from the running entry within %s", want, d)
	}
}
