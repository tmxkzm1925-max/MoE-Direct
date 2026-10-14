package wire

import (
	"testing"

	"moedirect/launcher/internal/core"
)

// TestUXR12TemplateWords (UX-R12, the sub-lead's addendum (1); UX-R10 landing 4, finding 18):
// the wire's two notes of a model outside the catalog read in UX-R10's template form: the
// preset note of its settings «outside the catalog · <arch> template settings» and its setup
// note, the same with the ⏎ hint; no «structure defaults».
func TestUXR12TemplateWords(t *testing.T) {
	b, model := uxr6Backend(t, true)
	v, err := b.ModelSettings(model)
	if err != nil {
		t.Fatal(err)
	}
	if v.Preset != "outside the catalog · gpt-oss template settings" {
		t.Errorf("preset note: %q", v.Preset)
	}
	n := templateNote(core.ModelFile{Arch: "gpt-oss"})
	if n != "outside the catalog · gpt-oss template settings · ⏎ shows its setup plan (the template repack)" {
		t.Errorf("setup note: %q", n)
	}
}
