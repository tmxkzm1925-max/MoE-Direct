// Command MoE-Direct-launcher is the Go launcher of MoE-Direct (U-26, [A-BIN]). Since GL-01
// (version 8) this tree is release source: build.ps1 in the module root is its one build
// command, and its name and release version come from internal/version. It grew from the
// draft bundles GOL-DRAFT, GOL-FIX1 and GOL-FIX2 (DRAFT_STATUS.md). The catalog default is
// models.json beside the launcher (GL-14) and the engine root default is the launcher's own
// folder (GL-08).
//
// It probes the console's glyph width (decision ④), then runs the screens of internal/ui over
// internal/core through the adapter internal/wire, and exits with the PowerShell launcher's
// exit-code table (STATUS_EXIT, Start-MoeDirect.ps1:203-221): the code of the last start or
// stop, 0 when nothing was started (U-32 (ii): quitting from the stopped page keeps the last
// stop's code).
//
// GL-08 (F-LS 9.2-9.4, W7 duties 19, 23, 40; conflict row 15): the launcher has two run modes,
// interactive (the screens) and argument-driven (args.go); the draft's scripted-input mode is
// gone from the binary, and there is no repro or collect mode. The screens run only on an
// interactive console with no start or stop given. Off an interactive console (or with
// -non-interactive) no screen is drawn and nothing is asked: one notice line is printed and the
// argument-driven mode runs; without a model that line is the guidance and the run ends
// fail_model_path (3). Conflict row 5: closing the window ends the launcher at once and the job
// object ends the server with it.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"moedirect/launcher/internal/core"
	"moedirect/launcher/internal/ui"
	"moedirect/launcher/internal/version"
	"moedirect/launcher/internal/wire"
)

// defaultRoot is the engine root of a release binary (GL-08): the folder of the launcher
// executable, which is the bundle root where llama-server.exe and expects\ sit (the bundle's
// staging-flat layout, make_bundle.ps1:17-28, with the launcher at the bundle root). -root
// overrides it.
func defaultRoot() string {
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	if r, err := filepath.EvalSymlinks(exe); err == nil {
		exe = r
	}
	return filepath.Dir(exe)
}

// catalogFileName is CATALOG_FILE_NAME (PS:784).
const catalogFileName = "models.json"

// defaultCatalog is the catalog a release binary reads: models.json beside the launcher at the
// bundle root (GL-14: make_bundle.ps1:17-20 puts the launcher and the catalog at the root, and
// Read-Catalog reads <root>\models.json, PS:2288-2291). -catalog names another file.
func defaultCatalog() string {
	exe, err := os.Executable()
	if err != nil {
		return catalogFileName
	}
	return filepath.Join(filepath.Dir(exe), catalogFileName)
}

// exitUsage is the exit code of a bad command line: fail_custom_args (3) in the PowerShell table.
var exitUsage = core.ExitCode(core.StatusFailCustomArgs)

// msgPrefix starts every message line of the launcher (U-26); the final status line keeps its
// fixed form (statusLine, F-LS 11.5).
const msgPrefix = version.LauncherName + ":"

func main() { os.Exit(run()) }

// options are the command line.
type options struct {
	catalog, root, state, modelsRoot string
	lang                             string
	nonInteractive                   bool
	arg                              argStart
}

// allowlistFlags are the argument-driven mode's setting arguments (flag, allowlist key).
var allowlistFlags = []struct{ flag, key string }{{"port", "port"}, {"ctx", "ctx"}, {"threads", "threads"},
	{"budget-mb", "budget_mb"}, {"qd", "qd"}, {"warmup", "warmup"}, {"prefetch", "prefetch"},
	{"prefetch-k", "prefetch_k"}, {"prefetch-n", "prefetch_n"}} // GL-19: the manual pair (F-PF §2.1)

// flagSet is the command line's flag set; set receives the allowlist flags.
func flagSet(o *options, errw io.Writer, set map[string]*string) *flag.FlagSet {
	fs := flag.NewFlagSet(version.LauncherName, flag.ContinueOnError)
	fs.SetOutput(errw)
	fs.StringVar(&o.catalog, "catalog", defaultCatalog(), "catalog file (models.json; default: beside the launcher)")
	fs.StringVar(&o.root, "root", defaultRoot(), "engine bundle root (llama-server.exe and expects\\); default: the launcher's own folder")
	fs.StringVar(&o.state, "state", core.LauncherStateDir(), "launcher state directory (the launcher's logs, presets, history, digest cache and measured store directly under it; the PowerShell launcher's files read only)")
	fs.StringVar(&o.modelsRoot, "models-root", "", "folder to scan for models (default: every fixed drive's models folders)")
	fs.StringVar(&o.lang, "lang", "", "screen language: en or ko (default: en, English)")
	// The argument-driven mode (conflict row 15; W7 duty 23), named after the PowerShell parameters.
	fs.StringVar(&o.arg.model, "model", "", "model file: start it without the screen (the argument-driven mode)")
	fs.BoolVar(&o.arg.stop, "stop", false, "argument-driven mode: stop before start (PowerShell -Action stop): nothing is started, exit 2")
	fs.BoolVar(&o.nonInteractive, "non-interactive", false, "never draw the screen or ask (PowerShell -NonInteractive)")
	fs.IntVar(&o.arg.runSeconds, "run-seconds", 0, "argument-driven mode: serve this long after ready, then stop (0 = stop right after ready)")
	fs.BoolVar(&o.arg.smoke, "smoke", false, "argument-driven mode: run the «작동 확인» check after ready, then stop (a failure exits 6)")
	fs.StringVar(&o.arg.repackMode, "repack-mode", "", "argument-driven mode: packed or virtual")
	for _, k := range allowlistFlags {
		set[k.key] = fs.String(k.flag, "", "argument-driven mode: the "+k.key+" setting (its PowerShell range applies)")
	}
	return fs
}

// flagNames lists the command line's flags, sorted (S8-LA-29: the mode and argument list).
func flagNames() []string {
	var names []string
	flagSet(&options{}, io.Discard, map[string]*string{}).VisitAll(func(f *flag.Flag) { names = append(names, f.Name) })
	sort.Strings(names)
	return names
}

// parse reads the command line. A bad flag is fail_custom_args (3), PowerShell's code for an
// argument it cannot use (the flag package's own exit code 2 would read as cancelled_user).
func parse(args []string, errw io.Writer) (*options, int, bool) {
	o := &options{}
	set := map[string]*string{}
	fs := flagSet(o, errw, set)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil, 0, false
		}
		statusLine(errw, core.StatusFailCustomArgs)
		return nil, exitUsage, false
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(errw, "%s unexpected argument %q\n", msgPrefix, fs.Arg(0))
		statusLine(errw, core.StatusFailCustomArgs)
		return nil, exitUsage, false
	}
	kv := map[string]string{}
	for k, v := range set {
		kv[k] = *v
	}
	o.arg.settings = settingsOf(kv)
	// F-PF §9 C1 (S8-PF-01): K and N are checked on the value as given, so a given -prefetch-k
	// or -prefetch-n reaches the check untrimmed, an empty value included.
	fs.Visit(func(f *flag.Flag) {
		for _, a := range allowlistFlags {
			if a.flag == f.Name && (a.key == "prefetch_k" || a.key == "prefetch_n") {
				o.arg.settings[a.key] = f.Value.String()
			}
		}
	})
	return o, 0, true
}

// argPath decides the run mode (W7 duties 19, 40; conflict row 15): the interactive mode only
// on an interactive console with no start or stop given, else the argument-driven mode. Off an
// interactive console it prints exactly one notice line; without a model that line is the
// guidance and the run ends fail_model_path (3).
func argPath(o *options, term core.TerminalInfo, errw io.Writer) (bool, int, bool) {
	if term.Interactive && o.arg.model == "" && !o.arg.stop {
		return false, 0, true
	}
	if o.arg.model == "" {
		if term.Interactive {
			fmt.Fprintln(errw, msgPrefix+" -stop needs -model <file.gguf>")
		} else {
			fmt.Fprintln(errw, guidance(term.Reason))
		}
		statusLine(errw, core.StatusFailModelPath)
		return true, core.ExitCode(core.StatusFailModelPath), false
	}
	if !term.Interactive {
		fmt.Fprintln(errw, notice(term.Reason))
	}
	return true, 0, true
}

// glyphProbe is GL-03's measurement of the interactive mode; the tests replace it to see that
// the argument-driven path never calls it.
var glyphProbe = ui.Probe

// argGlyphReason is the glyph record's reason on the argument-driven path: «no console: »
// followed by the console detection's reason (F-LS 9.4). With -model on an interactive console
// (a case 9.4 does not name) the run still draws no screen and measures nothing.
func argGlyphReason(term core.TerminalInfo) string {
	if term.Interactive {
		return "no screen: argument-driven mode"
	}
	return "no console: " + term.Reason
}

func run() int {
	o, code, ok := parse(os.Args[1:], os.Stderr)
	if !ok {
		return code
	}
	term := core.DetectTerminal(o.nonInteractive)
	useArgs, code, ok := argPath(o, term, os.Stderr)
	if !ok { // the guidance exit without a model ends here, before the start log opens
		return code
	}
	l, err := core.New(core.Options{CatalogPath: o.catalog, BundleRoot: o.root, StateDir: o.state, ModelsRoot: o.modelsRoot})
	if err != nil {
		fmt.Fprintln(os.Stderr, msgPrefix, err)
		return exitUsage
	}
	if useArgs {
		// F-LS 9.4 (conflict row 7.10; S8-LA-07; GL-FX3): every argument-driven run that opens
		// the start log writes the one glyph record without measuring
		l.StartLog(core.StartLogGlyphProbe, ui.NoProbe(argGlyphReason(term)).LogData())
	}
	if useArgs && o.arg.stop {
		return runStop(l, o.arg, os.Stdout, os.Stderr)
	}
	if useArgs {
		// conflict 18-2: no screen on this path, so the file check starts at once (the
		// interactive screen starts it after its first draw, GL-13); the start waits for it
		l.StartBundleCheck()
		var running atomic.Bool
		running.Store(true)
		closeOnWindowClose(running.Load, func() int { return 0 })
		code := runArgs(l, o.arg, os.Stdout, os.Stderr, interrupts())
		running.Store(false)
		return code
	}

	// the glyph self-measurement runs before the UI library's Run and its
	// result is a start-log record (GL-03; decision ④; GLP steps 1-7)
	probe := glyphProbe(ui.ProbeWait)
	l.StartLog(core.StartLogGlyphProbe, probe.LogData())
	b := wire.New(l, o.catalog, version.LauncherVersion)
	m := ui.New(b, ui.Options{Glyphs: probe.Set, Lang: o.lang})
	closeOnWindowClose(b.Serving, b.ExitCode)
	if _, err := tea.NewProgram(m, tea.WithAltScreen()).Run(); err != nil {
		fmt.Fprintln(os.Stderr, msgPrefix, err)
	}
	return b.Shutdown(3 * time.Minute)
}
