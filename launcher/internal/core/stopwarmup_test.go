package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeServerExe is the fixture-only name of the stand-in server: the fixture catalog's
// server_exe (fixture_test.go), so the launcher spawns it as it would spawn the engine. It is
// never the name of a real engine process (GL-FX1: a copy named llama-server.exe showed as a
// live engine to the seconds layer's lock probes and to any check for the user's server).
const fakeServerExe = "moe-fixture-server.exe"

// TestMain lets the test binary stand in for the server: a copy of it named fakeServerExe runs
// the fake server below instead of the tests (the helper-process pattern; FIX2-WIRE).
func TestMain(m *testing.M) {
	if strings.EqualFold(filepath.Base(os.Args[0]), fakeServerExe) {
		fakeServerMain(os.Args[1:])
		return
	}
	os.Exit(m.Run())
}

// fakeServerMain serves /health at once and holds every chat request (the warmup) until the
// client goes away, so a stop arrives while the warmup runs.
//
// GL-08: a file fake_server_mode.txt beside the copy changes that, by the words it holds:
// «events» appends one line per start, request and console break to fake_server_events.log
// beside it (the fake child's event log); «answer» answers a chat request at once; «break_exit0»
// exits 0 on a console break and «ignore_break» logs the break and keeps serving; «exit_early»
// exits 1 before it listens; «die» lets POST /die end the server with code 3. GL-07:
// «slow_health» answers /health 503 «Loading model» for the first 1.2 s. GL-11: «smoke» serves
// the check run's endpoints (/v1/models, «ping», an 80-chunk stream that logs the engine's cancel
// and release lines to stderr when it is dropped) and writes the engine's raw seal line to
// stderr after the listening line; «no_seal» leaves the seal line out. GL-21: «metrics» writes the
// engine's metrics header to MOE_DIRECT_METRICS before it listens, as the engine does at the
// seal (OBS:272, :418-421): prefetch on with the K and N of the env when both are set and
// MOE_NO_PREFETCH is not 1, else off with «unset» (PB:1307). GL-15: «slots» serves the slot API
// as llama-server does with --slot-save-path (GET /slots; POST /slots/0?action=save writes a
// GGSQ state file under that path, restore reads it back, erase answers), and writes one stderr
// line «[slots] METHOD URI BODY» per request (the request log; stderr is outside the bundle
// root). Without the file the behaviour is the one above.
func fakeServerMain(args []string) {
	dir := "."
	if exe, err := os.Executable(); err == nil {
		dir = filepath.Dir(exe)
	}
	modeB, _ := os.ReadFile(filepath.Join(dir, "fake_server_mode.txt"))
	mode := string(modeB)
	has := func(w string) bool { return strings.Contains(mode, w) }
	event := func(s string) {
		if !has("events") {
			return
		}
		if f, err := os.OpenFile(filepath.Join(dir, "fake_server_events.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
			f.WriteString(s + "\n")
			f.Close()
		}
	}
	event("start pid=" + strconv.Itoa(os.Getpid()) + " argv=" + strings.Join(args, " "))
	if has("exit_early") {
		event("exit 1 before listening")
		os.Exit(1)
	}
	if has("break") {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt) // CTRL_BREAK_EVENT arrives as os.Interrupt
		go func() {
			for range sig {
				event("ctrl_break")
				if has("break_exit0") {
					os.Exit(0)
				}
			}
		}()
	}
	host, port, slotDir := "127.0.0.1", "", ""
	for i := 0; i+1 < len(args); i++ {
		switch args[i] {
		case "--host":
			host = args[i+1]
		case "--port":
			port = args[i+1]
		case "--slot-save-path":
			slotDir = args[i+1]
		}
	}
	if has("metrics") { // GL-21
		fakeMetricsHeader(os.Getenv(EnvMetrics))
	}
	var errMu sync.Mutex
	errLine := func(s string) { // one complete stderr line per write
		errMu.Lock()
		os.Stderr.WriteString(s + "\n")
		errMu.Unlock()
	}
	errLine("main: server is listening on http://" + host + ":" + port)
	if has("smoke") && !has("no_seal") {
		errLine("[moe-direct] sealed: all=12 host=4 nonhost=8 slots=6/6 moe_layers=2")
	}
	mux := http.NewServeMux()
	var task atomic.Int64
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"object":"list","data":[{"id":"fixture"}]}`)
	})
	listening := time.Now()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		event(r.Method + " /health")
		if has("slow_health") && time.Since(listening) < 1200*time.Millisecond { // GL-07: a load that takes a while
			w.WriteHeader(http.StatusServiceUnavailable)
			io.WriteString(w, `{"error":{"code":503,"message":"Loading model"}}`)
			return
		}
		io.WriteString(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		event(r.Method + " /v1/chat/completions")
		if has("smoke") {
			b, _ := io.ReadAll(r.Body)
			id := task.Add(1)
			if !strings.Contains(string(b), `"stream":true`) {
				io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`)
				return
			}
			fl := w.(http.Flusher)
			for i := 1; i <= 80; i++ {
				select {
				case <-r.Context().Done():
					errLine(fmt.Sprintf("srv          stop: cancel task, id_task = %d", id))
					errLine(fmt.Sprintf("slot      release: id  0 | task %d | stop processing: n_tokens = %d, truncated = 0", id, i))
					return
				default:
				}
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"%d \"}}]}\n\n", i)
				fl.Flush()
				time.Sleep(10 * time.Millisecond)
			}
			io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		if has("answer") {
			io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
			return
		}
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Minute):
		}
	})
	if has("slots") { // GL-15
		mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
			errLine("[slots] " + r.Method + " " + r.URL.RequestURI())
			io.WriteString(w, `[{"id":0,"n_ctx":4096,"speculative":false,"is_processing":false,"id_task":7,"n_prompt_tokens":200}]`)
		})
		mux.HandleFunc("/slots/0", func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			errLine(strings.TrimSpace("[slots] " + r.Method + " " + r.URL.RequestURI() + " " + string(b)))
			var req struct {
				Filename string `json:"filename"`
			}
			_ = json.Unmarshal(b, &req)
			path := filepath.Join(slotDir, req.Filename)
			switch r.URL.Query().Get("action") {
			case "save":
				data := []byte("GGSQ\x02\x00\x00\x00fixture slot state")
				if slotDir == "" || os.WriteFile(path, data, 0o644) != nil {
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				fmt.Fprintf(w, `{"id_slot":0,"filename":"%s","n_saved":128,"n_written":%d}`, req.Filename, len(data))
			case "restore":
				st, err := os.Stat(path)
				if err != nil {
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				fmt.Fprintf(w, `{"id_slot":0,"filename":"%s","n_restored":128,"n_read":%d}`, req.Filename, st.Size())
			case "erase":
				io.WriteString(w, `{"id_slot":0,"n_erased":0}`)
			}
		})
	}
	mux.HandleFunc("/die", func(w http.ResponseWriter, r *http.Request) {
		if has("die") {
			event("die: exit 3")
			os.Exit(3)
		}
	})
	err := http.ListenAndServe(net.JoinHostPort(host, port), mux)
	if err != nil {
		event("listen failed: " + err.Error())
	}
	os.Exit(0)
}

// A stop during the post-ready warmup (FIX2-CORE landing section 4 item 5, a core repair under
// the FIX2-WIRE brief's clause): Stop's teardown owns the end, and the start reports that end as
// its final event (not ready, the stop's event attached) instead of a second teardown.
func TestStopDuringWarmup(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("owned children are Windows-only")
	}
	f := newFixture(t, true)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(f.root, fakeServerExe), b, 0o755); err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	f.l.os.portOwners = func(int) ([]int, error) { // the listener is the child while the port answers
		c, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 200*time.Millisecond)
		if err != nil {
			return []int{}, nil
		}
		c.Close()
		f.l.mu.Lock()
		s := f.l.sess
		f.l.mu.Unlock()
		if s == nil {
			return []int{}, nil
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.child == nil {
			return []int{}, nil
		}
		return []int{s.child.PID()}, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	ch, err := f.l.Start(ctx, Request{ModelPath: f.modelPath, Settings: map[string]string{"port": strconv.Itoa(port)}})
	if err != nil {
		t.Fatal(err)
	}
	var final StartEvent
	stopped := false
	var stopFinal StopEvent
	for ev := range ch {
		if ev.Step == StepWarmup && ev.State == StepRunning && !stopped {
			stopped = true
			time.Sleep(300 * time.Millisecond) // the warmup request is in flight
			sch, err := f.l.Stop()
			if err != nil {
				t.Fatalf("stop during the warmup: %v", err)
			}
			go func() {
				for e := range sch {
					if e.Final {
						stopFinal = e
					}
				}
			}()
		}
		if ev.Final {
			final = ev
		}
	}
	if !stopped {
		t.Fatalf("the warmup never ran: final %+v", final)
	}
	if final.Ready || final.StopInfo == nil || final.Err == nil || !strings.Contains(final.Err.Error(), "stopped during the warmup") ||
		final.Teardown == nil || final.Warmup == nil {
		t.Fatalf("final %+v", final)
	}
	<-f.l.Done()
	if evs := f.l.StopEvents(); len(evs) != 1 || evs[0].Status != final.Status || !evs[0].WasReady {
		t.Errorf("one stop event, from the stop: %+v", evs)
	}
	time.Sleep(200 * time.Millisecond)
	if stopFinal.Final && stopFinal.Status != final.Status {
		t.Errorf("stop %s, start %s", stopFinal.Status, final.Status)
	}
	if !final.Teardown.ChildGone || !final.Teardown.ListenerGone || final.Teardown.TaskkillUsed {
		t.Errorf("teardown %+v", final.Teardown)
	}
}

// fakeMetricsHeader writes the engine's header line (the field set of a real header, X-VQD
// checks :106) and one more line after it, as the engine's metrics file holds once it serves.
func fakeMetricsHeader(path string) {
	if path == "" {
		return
	}
	k, n := os.Getenv(EnvPrefetchK), os.Getenv(EnvPrefetchN)
	on := k != "" && n != "" && os.Getenv(EnvNoPrefetch) != "1"
	h := map[string]any{"type": "header", "invocation": "", "session": "", "leg": "", "pid": os.Getpid(), "seq": 0,
		"ts": "2026-10-12T12:00:00.000Z", "arch": "testarch", "qd_effective": 4, "io_path": "persistent", "mode": "bin",
		"prefetch_on": on, "prefetch_k": 0, "prefetch_n": 0, "prefetch_off_reason": "unset"}
	if on {
		kv, _ := strconv.Atoi(k)
		nv, _ := strconv.Atoi(n)
		h["prefetch_k"], h["prefetch_n"], h["prefetch_off_reason"] = kv, nv, ""
	}
	b, _ := json.Marshal(h)
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, append(b, []byte("\n{\"type\":\"tick\",\"seq\":1}\n")...), 0o644)
}
