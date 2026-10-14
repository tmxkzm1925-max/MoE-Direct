// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

// Package script is the draft-only scripted-input mode: it drives the same Bubble Tea model the
// console runs, without a console. A script is a list of key presses and waits; the driver runs
// the model's commands the way Bubble Tea does (each off the update loop, their messages fed
// back in order of arrival) and writes the rendered frames, ANSI stripped, to a file. It exists
// so a rehearsal can walk the real binary through its screens without a person.
package script

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

// Step is one script line.
//
//	key <name>                  one key press (Bubble Tea key names: q, ?, 1, enter, ctrl+f, left …)
//	wait <seconds> <regexp>     run until the screen matches, or fail after the timeout
//	pause <milliseconds>        run for a while
//	frame <label>               write the current screen
//	quit <seconds>              run until the program quits, or fail after the timeout
type Step struct {
	Line    int
	Kind    string
	Key     string
	Re      *regexp.Regexp
	Timeout time.Duration
	Label   string
}

// Parse reads a script; blank lines and lines starting with # are skipped.
func Parse(r io.Reader) ([]Step, error) {
	var steps []Step
	sc := bufio.NewScanner(r)
	n := 0
	for sc.Scan() {
		n++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		kind, rest, _ := strings.Cut(line, " ")
		rest = strings.TrimSpace(rest)
		st := Step{Line: n, Kind: kind}
		switch kind {
		case "key":
			if _, ok := keyMsg(rest); !ok {
				return nil, fmt.Errorf("line %d: unknown key %q", n, rest)
			}
			st.Key = rest
		case "wait":
			secs, re, ok := strings.Cut(rest, " ")
			if !ok {
				return nil, fmt.Errorf("line %d: wait <seconds> <regexp>", n)
			}
			s, err := strconv.ParseFloat(secs, 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: %v", n, err)
			}
			if st.Re, err = regexp.Compile(strings.TrimSpace(re)); err != nil {
				return nil, fmt.Errorf("line %d: %v", n, err)
			}
			st.Timeout = time.Duration(s * float64(time.Second))
		case "pause", "quit":
			v, err := strconv.ParseFloat(rest, 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: %v", n, err)
			}
			if kind == "pause" {
				st.Timeout = time.Duration(v * float64(time.Millisecond))
			} else {
				st.Timeout = time.Duration(v * float64(time.Second))
			}
		case "frame":
			st.Label = rest
		default:
			return nil, fmt.Errorf("line %d: unknown step %q", n, kind)
		}
		steps = append(steps, st)
	}
	return steps, sc.Err()
}

var keyNames = func() map[string]tea.KeyMsg {
	m := map[string]tea.KeyMsg{}
	for t := -200; t <= 300; t++ {
		k := tea.Key{Type: tea.KeyType(t)}
		if t == int(tea.KeyRunes) {
			continue
		}
		if s := k.String(); s != "" {
			if _, dup := m[s]; !dup {
				m[s] = tea.KeyMsg(k)
			}
		}
	}
	return m
}()

// keyMsg is the key message whose String() is name.
func keyMsg(name string) (tea.KeyMsg, bool) {
	if k, ok := keyNames[name]; ok {
		return k, true
	}
	if r := []rune(name); len(r) == 1 {
		return tea.KeyMsg(tea.Key{Type: tea.KeyRunes, Runes: r}), true
	}
	return tea.KeyMsg{}, false
}

// settle is how long the driver runs after a key before it writes the frame.
const settle = 400 * time.Millisecond

// Driver runs a model by script.
type Driver struct {
	m      tea.Model
	msgs   chan tea.Msg
	out    io.Writer
	t0     time.Time
	frames int
	quit   bool
	last   string
}

// Run drives m through steps at width × height and writes frames to out. It returns when the
// steps are done or the program quits; a failed wait writes its frame and returns an error.
func Run(m tea.Model, steps []Step, out io.Writer, width, height int) error {
	d := &Driver{m: m, msgs: make(chan tea.Msg, 4096), out: out, t0: time.Now()}
	d.handle(tea.WindowSizeMsg{Width: width, Height: height})
	d.run(m.Init())
	for _, st := range steps {
		if d.quit {
			d.frame(fmt.Sprintf("program quit before line %d", st.Line))
			return nil
		}
		switch st.Kind {
		case "key":
			k, _ := keyMsg(st.Key)
			d.handle(k)
			d.pump(settle, nil)
			d.frame(fmt.Sprintf("line %d · after key %s", st.Line, st.Key))
		case "pause":
			d.pump(st.Timeout, nil)
		case "frame":
			d.frame(fmt.Sprintf("line %d · %s", st.Line, st.Label))
		case "wait":
			ok := d.pump(st.Timeout, func() bool { return st.Re.MatchString(d.text()) })
			if !ok {
				d.frame(fmt.Sprintf("line %d · wait for /%s/ FAILED after %s", st.Line, st.Re, st.Timeout))
				return fmt.Errorf("line %d: the screen did not match /%s/ within %s", st.Line, st.Re, st.Timeout)
			}
			d.frame(fmt.Sprintf("line %d · matched /%s/", st.Line, st.Re))
		case "quit":
			if !d.pump(st.Timeout, func() bool { return d.quit }) {
				d.frame(fmt.Sprintf("line %d · quit FAILED after %s", st.Line, st.Timeout))
				return fmt.Errorf("line %d: the program did not quit within %s", st.Line, st.Timeout)
			}
		}
	}
	if d.quit {
		d.frame("program quit")
	}
	return nil
}

// Quit reports whether the program asked to quit.
func (d *Driver) Quit() bool { return d.quit }

// run starts a command off the update loop, as Bubble Tea does.
func (d *Driver) run(c tea.Cmd) {
	if c == nil {
		return
	}
	go func() {
		if msg := c(); msg != nil {
			d.msgs <- msg
		}
	}()
}

// handle feeds one message to the model.
func (d *Driver) handle(msg tea.Msg) {
	switch x := msg.(type) {
	case tea.QuitMsg:
		d.quit = true
		return
	case tea.BatchMsg:
		for _, c := range x {
			d.run(c)
		}
		return
	}
	m, c := d.m.Update(msg)
	d.m = m
	d.run(c)
}

// pump handles messages for at most dur; with until set it stops as soon as until holds.
func (d *Driver) pump(dur time.Duration, until func() bool) bool {
	deadline := time.After(dur)
	check := time.NewTicker(100 * time.Millisecond)
	defer check.Stop()
	for {
		if until != nil && until() {
			return true
		}
		if d.quit && until == nil {
			return true
		}
		select {
		case msg := <-d.msgs:
			d.handle(msg)
		case <-check.C:
		case <-deadline:
			return until == nil || until()
		}
	}
}

func (d *Driver) text() string {
	d.last = ansi.Strip(d.m.View())
	return d.last
}

func (d *Driver) frame(label string) {
	d.frames++
	fmt.Fprintf(d.out, "=== frame %d · t+%.1fs · %s · %s ===\n%s\n", d.frames, time.Since(d.t0).Seconds(),
		time.Now().Format("15:04:05"), label, strings.TrimRight(d.text(), "\n"))
}
