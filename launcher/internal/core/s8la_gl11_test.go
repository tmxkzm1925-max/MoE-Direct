package core

// GL-11 (version 8): scenario S8-LA-25 of the plan reviews/_plan_8_STAGE1_v8-plansplit.md
// (d1dbe526f), built to the F-LS edition reviews/_spec_8_F-LS_launcher_go_edition.md (d3a5ab2ed)
// conflict row 18-1 [A-SMK]: the «작동 확인» check after ready is the PowerShell smoke checklist
// (LS:291-299; Invoke-SmokeChecklist PS:12665-12950), items 1 to 7, one log line per item. The
// live parts (off, on, first once, a failure that ends the start with exit 6) start the fake
// server child of stopwarmup_test.go and are Windows-only (s8la_gl11_windows_test.go).

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// la25Server is fakeLlama with the built-in web UI page when page is set.
type la25Server struct {
	*fakeLlama
	page string
}

func (s la25Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		w.Write([]byte(s.page))
		return
	}
	s.fakeLlama.ServeHTTP(w, r)
}

const (
	la25Raw  = "[moe-direct] sealed: all=12 host=4 nonhost=8 slots=6/6 moe_layers=2"
	la25Info = "0.01.234.567 I load_tensors: moe-direct: sealed all=12 host=4 nonhost=8 slots=6/6 moe_layers=2 (matched live=12 host=4 nonhost=8)"
)

func TestS8LA25(t *testing.T) {
	t.Run("item 5 seal attestation", func(t *testing.T) {
		cases := []struct {
			name  string
			lines []string
			ok    bool
			why   string
		}{
			{"raw line", []string{"x", la25Raw, "y"}, true, ""},
			{"legacy info line", []string{la25Info}, true, ""},
			{"info without timestamp", []string{strings.TrimPrefix(la25Info, "0.01.234.567 I ")}, true, ""},
			{"raw and info agree", []string{la25Raw, la25Info}, true, ""},
			{"none", []string{"main: server is listening"}, false, "no seal success line"},
			{"substring is no evidence", []string{"note " + la25Raw}, false, "no seal success line"},
			{"malformed raw", []string{la25Raw + " extra"}, false, "malformed complete seal success line"},
			{"malformed info", []string{"load_tensors: moe-direct: sealed all=x"}, false, "malformed complete seal success line"},
			{"duplicate raw", []string{la25Raw, la25Raw}, false, "duplicate raw attestation"},
			{"duplicate info", []string{la25Info, la25Info}, false, "duplicate info attestation"},
			{"disagree", []string{la25Raw, strings.Replace(la25Info, "slots=6/6", "slots=5/6", 1)}, false, "seal representations disagree on have"},
			{"outside Int64", []string{strings.Replace(la25Raw, "all=12", "all=99999999999999999999", 1)}, false, "outside Int64"},
		}
		for _, c := range cases {
			a := engineSealAttestation(c.lines)
			if a.OK != c.ok || (c.why != "" && !strings.Contains(a.Reason, c.why)) {
				t.Errorf("%s: %+v", c.name, a)
			}
			if c.ok && (a.SlotsHave != 6 || a.SlotsNeed != 6) {
				t.Errorf("%s: slots %d/%d", c.name, a.SlotsHave, a.SlotsNeed)
			}
		}
	})

	t.Run("item 6 loopback-only binding", func(t *testing.T) {
		list := func(a ...string) func(int) ([]string, error) {
			return func(int) ([]string, error) { return a, nil }
		}
		cases := []struct {
			name string
			l    func(int) ([]string, error)
			ok   bool
			why  string
		}{
			{"ipv4 loopback", list("127.0.0.1"), true, ""},
			{"both loopbacks", list("127.0.0.1", "::1"), true, ""},
			{"any address", list("127.0.0.1", "0.0.0.0"), false, "non-loopback listener: 0.0.0.0"},
			{"lan address", list("192.168.0.7"), false, "non-loopback listener: 192.168.0.7"},
			{"ipv6 any", list("::"), false, "non-loopback listener: ::"},
			{"no listener", list(), false, "no listener found"},
			{"table error", func(int) ([]string, error) { return nil, errors.New("GetExtendedTcpTable 87") }, false, "listener enumeration failed: GetExtendedTcpTable 87"},
			{"no table", nil, false, "listener enumeration failed"},
		}
		for _, c := range cases {
			ok, _, why := loopbackOnlyBinding(8093, c.l)
			if ok != c.ok || (c.why != "" && why != c.why && !strings.HasPrefix(why, c.why)) {
				t.Errorf("%s: ok=%v why=%q", c.name, ok, why)
			}
		}
	})

	t.Run("the whole checklist, items 1 to 7", func(t *testing.T) {
		fl := &fakeLlama{release: true}
		srv := httptest.NewServer(la25Server{fakeLlama: fl, page: "<html>webui</html>"})
		defer srv.Close()
		readLog := func() []string { return append([]string{la25Raw}, fl.lines()...) }
		lb := func(int) ([]string, error) { return []string{"127.0.0.1"}, nil }
		x := smokeContext{ManifestSha: strings.Repeat("ab", 32), Port: 8093, Listeners: lb}

		// An API-only bundle: seven items, item 7 skipped, the PowerShell lines.
		r := runSmokeFull(context.Background(), srv.URL, readLog, testSmokeTimings, x)
		if !r.OK || len(r.Items) != 7 || len(r.Failures) != 0 {
			t.Fatalf("api-only pass: %+v", r)
		}
		want := map[int]string{
			5: "[5] verify PASS consumption          PASS (engine seal line x1, slots 6/6; launcher gate manifest abababababab)",
			6: "[6] loopback-only binding            PASS (127.0.0.1)",
			7: "[7] built-in web UI                  SKIP (API-only bundle)",
		}
		prefix := map[int]string{ // PowerShell's lines (PS:12674-12880): the verdict column
			1: "[1] /health 200                      PASS", 2: "[2] /v1/models                       PASS",
			3: "[3] chat completion stream+non-stream PASS", 4: "[4] cancel + slot reclaim            PASS (",
		}
		for i, it := range r.Items {
			if p, ok := prefix[it.N]; ok && !strings.HasPrefix(it.Line(), p) {
				t.Errorf("item %d line %q, want prefix %q", it.N, it.Line(), p)
			}
			if it.N != i+1 {
				t.Errorf("item order %d: %+v", i, it)
			}
			if w, ok := want[it.N]; ok && it.Line() != w {
				t.Errorf("item %d line %q, want %q", it.N, it.Line(), w)
			}
			if !strings.HasPrefix(it.Line(), "[") || !strings.Contains(it.Line(), " PASS") && !strings.Contains(it.Line(), " SKIP") {
				t.Errorf("item %d: %q", it.N, it.Line())
			}
		}

		// A bundle with the web UI: the page and a completion through the same origin.
		x.WebUI = true
		r = runSmokeFull(context.Background(), srv.URL, readLog, testSmokeTimings, x)
		if !r.OK || r.Items[6].Skipped || r.Items[6].Line() != "[7] built-in web UI load + send      PASS" {
			t.Errorf("webui pass: %+v", r.Items[6])
		}
		blank := httptest.NewServer(la25Server{fakeLlama: fl})
		defer blank.Close()
		r = runSmokeFull(context.Background(), blank.URL, readLog, testSmokeTimings, x)
		if r.OK || len(r.Failures) != 1 || r.Failures[0] != "7:webui" || !strings.HasSuffix(r.Items[6].Line(), "FAIL (load=false send=true)") {
			t.Errorf("webui empty page: %v %+v", r.Failures, r.Items[6])
		}
		x.WebUI = false

		// Item 5 needs the launcher's own gate manifest besides the engine's line.
		y := x
		y.ManifestSha = ""
		r = runSmokeFull(context.Background(), srv.URL, readLog, testSmokeTimings, y)
		if r.OK || len(r.Failures) != 1 || r.Failures[0] != "5:verify-consume" || r.Items[4].Detail != "no launcher gate manifest" {
			t.Errorf("no manifest: %v %+v", r.Failures, r.Items[4])
		}
		// No seal line in the server's stderr.
		r = runSmokeFull(context.Background(), srv.URL, fl.lines, testSmokeTimings, x)
		if r.OK || r.Failures[0] != "5:verify-consume" || r.Items[4].Line() != "[5] verify PASS consumption          FAIL (no seal success line)" {
			t.Errorf("no seal: %v %+v", r.Failures, r.Items[4])
		}
		// A listener on every address.
		y = x
		y.Listeners = func(int) ([]string, error) { return []string{"0.0.0.0"}, nil }
		r = runSmokeFull(context.Background(), srv.URL, readLog, testSmokeTimings, y)
		if r.OK || len(r.Failures) != 1 || r.Failures[0] != "6:loopback" || r.Items[5].Detail != "non-loopback listener: 0.0.0.0" {
			t.Errorf("any address: %v %+v", r.Failures, r.Items[5])
		}
		// Nothing listening: every item that asks the server fails, item 7 still skips.
		r = runSmokeFull(context.Background(), "http://127.0.0.1:1", func() []string { return nil }, testSmokeTimings,
			smokeContext{Listeners: func(int) ([]string, error) { return nil, nil }})
		if r.OK || strings.Join(r.Failures, ",") != "1:/health,2:/v1/models,3:chat,4:cancel,5:verify-consume,6:loopback" || !r.Items[6].Skipped {
			t.Errorf("down: %v", r.Failures)
		}
		if !strings.HasPrefix(r.Line(), "smoke FAIL: 1:/health") {
			t.Errorf("summary %q", r.Line())
		}
	})

	t.Run("live starts", la25Live)
}
