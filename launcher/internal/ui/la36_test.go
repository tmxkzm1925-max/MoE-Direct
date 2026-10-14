// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package ui

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// notTextFiles are the source files whose literals are no screen text.
var notTextFiles = map[string]string{
	"tr.go":            "the string tables themselves",
	"fake.go":          "the fake backend's data (the backend delivers names, paths and reasons)",
	"glyph.go":         "the glyph probe's record, written to the diagnostic log before the screens",
	"probe_windows.go": "the glyph probe's console calls",
	"probe_other.go":   "the glyph probe's console calls",
	"lang_windows.go":  "reading the display language",
	"lang_other.go":    "reading the display language",
}

// notText are the literals of the screen files that are no words to translate: key names,
// units, numbers' formats, the product name and records. Each is a source literal (checked).
var notText = map[string]string{
	" tok/s":        "unit",
	" tok/s ":       "unit",
	"tok/s":         "unit",
	" · Esc ":       "key name",
	"Enter ":        "key name",
	" · PID %d":     "process id",
	"PID %d":        "process id",
	" · TTFT ":      "measure name",
	"TTFT ~":        "measure name",
	"TTFT ~%.0f s":  "measure name and unit",
	" · ctrl+f":     "key name",
	" · ctrl+x":     "key name",
	"/health ":      "the server's endpoint",
	", %.0f %% CPU": "unit",
	", tab":         "key name",
	"English":       "the language's own name (한국어 is the other)",
	"MoE-Direct":    "the product name",
	"RAM · ":        "measure name",
	"ctrl+f c":      "key names",
	"ctrl+r ":       "key name",
	"tab a":         "key names",
	"tab ↑↓ end e":  "key names",
	"↑↓ tab ⏎":      "key names",
	"L:4177 ① (reply cut is the client platform's)":         "a removed action's record, not shown",
	"L:4265 ② (H5K and the stored key table: next version)": "a removed action's record, not shown",
	"conflict row 6 (1) at c900f9cd0; U-28 (ii); rule 6":    "a removed action's record, not shown",
	"not ported (after the user's decisions)":               "the backend's error value; the screens say r.np()",
	"\x00graph\x00":   "a layout marker",
	"\x00log\x00":     "a layout marker",
	"\x00verdict\x00": "a layout marker",
	"\x00boxes\x00":   "a layout marker",
	"queue depth":     "a match key: the start's sweep step as the wire names it (UX-R14 speedDone)",
}

// wordLike: a literal that reads as words (three letters in a row, not an identifier).
var (
	hasWord   = regexp.MustCompile(`[A-Za-z]{3,}`)
	ident     = regexp.MustCompile(`^[a-z0-9_+.-]+$`)
	hangul    = regexp.MustCompile(`[가-힣]`)
	englishKV = regexp.MustCompile(`^[ -~·…→←↑↓⏎]+$`)
)

// sourceLits reads every string literal of the non-test files of this package and of the
// backend that delivers the screens' words (internal/wire), the imports aside: all of them,
// and the ones of this package's screen files that read as screen words.
func sourceLits(t *testing.T) (all, words map[string]string) {
	t.Helper()
	all, words = map[string]string{}, map[string]string{}
	fs, err := filepath.Glob("*.go")
	wf, err2 := filepath.Glob(filepath.Join("..", "wire", "*.go"))
	if err != nil || err2 != nil || len(fs) == 0 || len(wf) == 0 {
		t.Fatalf("source files: %v %v %v %v", fs, wf, err, err2)
	}
	set := token.NewFileSet()
	for _, f := range append(fs, wf...) {
		if strings.HasSuffix(f, "_test.go") || f == "tr.go" {
			continue
		}
		_, here := notTextFiles[f]
		here = !here && !strings.Contains(f, "wire")
		file, err := parser.ParseFile(set, f, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			if _, ok := n.(*ast.ImportSpec); ok {
				return false
			}
			b, ok := n.(*ast.BasicLit)
			if !ok || b.Kind != token.STRING {
				return true
			}
			s, err := strconv.Unquote(b.Value)
			if err != nil {
				t.Fatalf("%s: %v", set.Position(b.Pos()), err)
			}
			at := set.Position(b.Pos()).String()
			all[s] = at
			if here && hasWord.MatchString(s) && !ident.MatchString(s) {
				words[s] = at
			}
			return true
		})
	}
	return all, words
}

// lookups hears every table lookup while f runs.
func lookups(f func()) (seen map[[2]string]bool) {
	var mu sync.Mutex
	seen = map[[2]string]bool{}
	trSeen = func(lang, key string, found bool) {
		mu.Lock()
		defer mu.Unlock()
		k := [2]string{lang, key}
		seen[k] = seen[k] || found
	}
	defer func() { trSeen = nil }()
	f()
	return seen
}

// TestS8LA36 (plan S8-LA-36; GL-17), the screen side: the log page lists the log folder and
// its files and holds three lists, all, warnings and errors, each with its count, a notice
// shown in «all» only and counted in neither other list (review items 18, 29 (2)); the
// language action switches every screen between English and Korean, the screens drawn
// only in English before included, and both string tables hold the same key set (U-37 (i),
// L:4264): English is the key itself, so the Korean table holds every key the screens ask
// for and no key the source no longer has. The folder and the merge are the wire side
// (internal/wire TestS8LA36).
func TestS8LA36(t *testing.T) {
	ko := table[LangKorean]

	// (1) the key sets: every screen word of the source is a key, every key is a word of the
	// source or of the backend's (a setting's label, a step's name)
	all, words := sourceLits(t)
	var missing, stale, staleNot []string
	for s, at := range words {
		if _, ok := ko[s]; !ok {
			if _, ok := notText[s]; !ok {
				missing = append(missing, fmt.Sprintf("%s %q", at, s))
			}
		}
	}
	for k := range ko {
		if _, ok := all[k]; !ok {
			stale = append(stale, fmt.Sprintf("%q", k))
		}
	}
	for s := range notText {
		if _, ok := all[s]; !ok {
			staleNot = append(staleNot, fmt.Sprintf("%q", s))
		}
	}
	sort.Strings(missing)
	sort.Strings(stale)
	sort.Strings(staleNot)
	if len(missing)+len(stale)+len(staleNot) > 0 {
		t.Fatalf("Korean table lacks %d screen words:\n%s\nholds %d keys the source does not have:\n%s\nnotText holds %d literals the source does not have:\n%s",
			len(missing), strings.Join(missing, "\n"), len(stale), strings.Join(stale, "\n"), len(staleNot), strings.Join(staleNot, "\n"))
	}
	for k, v := range ko {
		if strings.Count(k, "%") != strings.Count(v, "%") {
			t.Fatalf("format verbs differ: %q -> %q", k, v)
		}
	}

	// (2) every screen in both languages: each literal key asked is held; each Korean screen
	// has Korean text and no English sentence of the table left in it
	ids := append(append([]string{}, designIDs...), "S0")
	en, kr := 0, 0
	seen := lookups(func() {
		for _, id := range ids {
			for _, sz := range sizes {
				hs := reach(t, id, LangEnglish, sz[0], sz[1])
				hs.want(id)
				if p := hs.plain(); hangul.MatchString(strings.ReplaceAll(p, "한국어", "")) {
					t.Fatalf("%s %dx%d English shows Korean:\n%s", id, sz[0], sz[1], p)
				}
				en++
				// the language action's effect (keys.go: m.r.lang from langAt), on this same screen
				hs.m.langAt, hs.m.r.lang = 2, LangKorean
				p := hs.plain()
				if !hangul.MatchString(p) {
					t.Fatalf("%s %dx%d drawn only in English:\n%s", id, sz[0], sz[1], p)
				}
				for k, v := range ko {
					if len(k) >= 12 && strings.Contains(k, " ") && englishKV.MatchString(k) && v != k && strings.Contains(p, k) {
						t.Fatalf("%s %dx%d Korean screen shows the English %q:\n%s", id, sz[0], sz[1], k, p)
					}
				}
				kr++
			}
		}
	})
	asked := map[string]bool{}
	var lacks []string
	for k, found := range seen {
		if _, lit := all[k[1]]; !lit || k[1] == "" {
			continue // a value the backend delivers (a name, a reason) is no key
		}
		if k[0] == LangEnglish {
			asked[k[1]] = true
			if _, ok := ko[k[1]]; !ok {
				lacks = append(lacks, "en "+strconv.Quote(k[1]))
			}
		} else if !found {
			lacks = append(lacks, "ko "+strconv.Quote(k[1]))
		}
	}
	sort.Strings(lacks)
	if len(lacks) > 0 {
		t.Fatalf("keys asked but not in the Korean table:\n%s", strings.Join(lacks, "\n"))
	}

	// (3) the language action switches the running screens: auto (English here) → English →
	// Korean → auto, every place following at once
	hs := newH(t, NewFake(), LangEnglish, 120, 30)
	hs.keys("5", "tab", "down", "down", "right", "right")
	if hs.m.r.lang != LangKorean {
		t.Fatalf("language action: %q", hs.m.r.lang)
	}
	for _, place := range []string{"1", "2", "3", "4", "5"} {
		hs.keys(place)
		if p := hs.plain(); !hangul.MatchString(strings.ReplaceAll(p, "한국어", "")) {
			t.Fatalf("place %s after the switch:\n%s", place, p)
		}
	}
	hs.keys("down", "down", "right") // 5 Settings keeps its section; arriving chooses no row
	if hs.m.r.lang != LangEnglish || hs.m.langAt != 0 {
		t.Fatalf("language action back: %q", hs.m.r.lang)
	}
	for _, place := range []string{"1", "2", "3", "4", "5"} {
		hs.keys(place)
		if p := hs.plain(); hangul.MatchString(p) {
			t.Fatalf("place %s after the switch back:\n%s", place, p)
		}
	}

	// (4) the log page: the log folder and its files listed; three lists with counts
	f := NewFake()
	at := f.Logs[len(f.Logs)-1].At
	f.Logs = append(f.Logs,
		LogLine{At: at.Add(time.Minute), Level: LogError, Text: "srv  load failed"},
		LogLine{At: at.Add(2 * time.Minute), Level: LogNotice, Text: "[moe-direct] NOTICE: budget lowered"})
	f.Info.LogDir = filepath.Join(os.TempDir(), "MoE-Direct", "logs")
	f.Info.LogFiles = []string{"server_20261012T090000Z_err.log", "launcher_20261012T085959Z.jsonl"}
	hs = newH(t, f, LangEnglish, 120, 30)
	hs.keys("4")
	warn, errs := 0, 0
	for _, x := range f.Logs {
		switch x.Level {
		case LogWarn:
			warn++
		case LogError:
			errs++
		}
	}
	p := hs.plain()
	mustHave(t, p, fmt.Sprintf("all %d │ warnings %d │ errors %d", len(f.Logs), warn, errs),
		"notice  [moe-direct] mode=virtual", "notice  [moe-direct] NOTICE: budget lowered", "RAM low", "load failed",
		"logs  ", "MoE-Direct", "2 · server_20261012T090000Z_err.log · launcher_")
	hs.keys("tab")
	p = hs.plain()
	mustHave(t, p, "RAM low")
	mustLack(t, p, "notice", "budget lowered", "load failed", "catalog read")
	hs.keys("tab")
	p = hs.plain()
	mustHave(t, p, "load failed")
	mustLack(t, p, "notice", "budget lowered", "RAM low")
	if warn != 1 || errs != 1 || len(f.Logs) != 6 {
		t.Fatalf("the fake's lines changed: %d %d %d", len(f.Logs), warn, errs)
	}
	t.Logf("%d keys in each table; %d English and %d Korean screen renders; %d literal keys asked; lists all %d · warnings %d · errors %d",
		len(ko), en, kr, len(asked), len(f.Logs), warn, errs)
}
