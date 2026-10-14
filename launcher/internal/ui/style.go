// Draft (GOL-DRAFT): editable, no completion stamp, never release use.
// Layout primitives copied from the round-5 mock (designs/launcher_design/
// proto/go5/main.go, g.go, h.go, m.go), made safe for any window size: every
// repeat count and width is clamped, so a small window shows less and never
// panics (decision ⑤).

package ui

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

var st = lipgloss.NewStyle

func fg(c string) lipgloss.Style { return st().Foreground(lipgloss.Color(c)) }

var (
	dim = fg("8")
	txt = fg("7")
	hi  = fg("15")
	hiB = fg("15").Bold(true)
	cy  = fg("14")
	cyB = fg("14").Bold(true)
	gr  = fg("10")
	grB = fg("10").Bold(true)
	ye  = fg("11")
	yeB = fg("11").Bold(true)
	yeD = fg("3")
	rd  = fg("9")
	rdB = fg("9").Bold(true)
	bl  = fg("12")
)

func rep(s string, n int) string {
	if n <= 0 {
		return ""
	}
	return strings.Repeat(s, n)
}

func clamp01(f float64) float64 {
	if math.IsNaN(f) || f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}

func indent(s string, n int) string {
	p := rep(" ", n)
	return p + strings.ReplaceAll(s, "\n", "\n"+p)
}

func pill(s, bg string) string {
	return st().Background(lipgloss.Color(bg)).Foreground(lipgloss.Color("0")).Bold(true).Render(" " + s + " ")
}

func pad(s string, w int) string {
	if n := lipgloss.Width(s); n < w {
		return s + rep(" ", w-n)
	}
	return s
}

func lpad(s string, w int) string {
	if n := lipgloss.Width(s); n < w {
		return rep(" ", w-n) + s
	}
	return s
}

func lines(xs ...string) string { return strings.Join(xs, "\n") }

func bar(frac float64, w int, s lipgloss.Style) string {
	if w <= 0 {
		return ""
	}
	n := int(math.Round(clamp01(frac) * float64(w)))
	return s.Render(rep("█", n)) + dim.Render(rep("░", w-n))
}

func use(frac float64, w int) string {
	s := gr
	if frac >= 0.9 {
		s = rd
	} else if frac >= 0.75 {
		s = ye
	}
	return bar(frac, w, s)
}

type seg struct {
	frac  float64
	glyph string
	sty   lipgloss.Style
}

// stack draws parts left to right; whatever is left over is free space.
func stack(w int, parts ...seg) string {
	if w <= 0 {
		return ""
	}
	var b strings.Builder
	used := 0
	for _, p := range parts {
		n := int(math.Round(clamp01(p.frac) * float64(w)))
		n = min(n, w-used)
		b.WriteString(p.sty.Render(rep(p.glyph, n)))
		used += n
	}
	b.WriteString(dim.Render(rep("░", w-used)))
	return b.String()
}

// mbar draws parts given in G against a total; the rest is free (·).
func mbar(w int, total float64, parts ...seg) string {
	if w <= 0 {
		return ""
	}
	var b strings.Builder
	used := 0
	for _, p := range parts {
		n := 0
		if total > 0 {
			n = int(p.frac/total*float64(w) + 0.5)
		}
		n = max(0, min(n, w-used))
		b.WriteString(p.sty.Render(rep(p.glyph, n)))
		used += n
	}
	b.WriteString(dim.Render(rep("·", w-used)))
	return b.String()
}

// tall draws values (0..1) as block columns, rows high.
func tall(vals []float64, rows int, s lipgloss.Style) []string {
	g := []rune(" ▁▂▃▄▅▆▇█")
	out := make([]string, max(rows, 0))
	for r := range out {
		var b strings.Builder
		for _, v := range vals {
			lvl := int(math.Round(clamp01(v)*float64(rows*8))) - (rows-1-r)*8
			b.WriteRune(g[min(8, max(0, lvl))])
		}
		out[r] = s.Render(b.String())
	}
	return out
}

// series fits samples to w columns (newest right) and scales them to 0..1.
func series(samples []float64, w int) []float64 {
	if w <= 0 {
		return nil
	}
	out := make([]float64, w)
	if len(samples) == 0 {
		return out
	}
	src := samples
	if len(src) > w {
		src = src[len(src)-w:]
	}
	top := 0.0
	for _, v := range src {
		top = math.Max(top, v)
	}
	off := w - len(src)
	for i, v := range src {
		if top > 0 {
			out[off+i] = 0.05 + 0.9*v/top
		}
	}
	return out
}

var font = map[rune][3]string{
	'0': {"█▀█", "█ █", "▀▀▀"}, '1': {"▀█ ", " █ ", "▀▀▀"}, '2': {"▀▀█", "█▀▀", "▀▀▀"},
	'3': {"▀▀█", " ▀█", "▀▀▀"}, '4': {"█ █", "▀▀█", "  ▀"}, '5': {"█▀▀", "▀▀█", "▀▀▀"},
	'6': {"█▀▀", "█▀█", "▀▀▀"}, '7': {"▀▀█", "  █", "  ▀"}, '8': {"█▀█", "█▀█", "▀▀▀"},
	'9': {"█▀█", "▀▀█", "▀▀▀"}, '.': {" ", " ", "▀"}, ':': {" ", "▀", "▀"},
	'-': {"   ", "▀▀▀", "   "},
}

// big draws a number three rows high; a rune outside the font is skipped.
func big(s string, sty lipgloss.Style) [3]string {
	var r [3]string
	first := true
	for _, ch := range s {
		f, ok := font[ch]
		if !ok {
			continue
		}
		for k := 0; k < 3; k++ {
			if !first {
				r[k] += " "
			}
			r[k] += f[k]
		}
		first = false
	}
	for k := range r {
		r[k] = sty.Render(r[k])
	}
	return r
}

func commas(v int) string {
	s := fmt.Sprint(v)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// cut truncates a rendered line to w cells.
func cut(s string, w int) string {
	if w <= 0 {
		return ""
	}
	if lipgloss.Width(s) <= w {
		return s
	}
	return ansi.Truncate(s, w, "")
}

// num formats a value with the given verb; "—" when it is not known.
func num(v Val, verb string) string {
	if !v.Ok() {
		return "—"
	}
	return fmt.Sprintf(verb, v.V)
}

// tok formats a token count as 9.4k.
func tok(v Val) string {
	if !v.Ok() {
		return "—"
	}
	if v.V >= 1000 {
		return fmt.Sprintf("%.1fk", v.V/1000)
	}
	return fmt.Sprintf("%.0f", v.V)
}

func fmtG(v float64) string { return fmt.Sprintf("%.1f GB", v) }

func dur(d durationLike) string {
	s := int(d.Seconds())
	switch {
	case s < 60:
		return fmt.Sprintf("%d s", s)
	case s < 3600:
		return fmt.Sprintf("%dm %02ds", s/60, s%60)
	}
	return fmt.Sprintf("%dh %02dm", s/3600, (s%3600)/60)
}

type durationLike interface{ Seconds() float64 }
