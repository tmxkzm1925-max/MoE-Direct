package core

// UX-R21 (version 8; the user's acceptance item 32, reviews/_user_8_LAUNCHER-REVIEW_12.md,
// screenshots 185445 and 185529): DeepSeek-V4.1 (architecture deepseek4, no catalog profile
// and no template) is refused with fail_model_path, which the screens read as «the model file
// is missing or cannot be used». The refusal stays (status, exit code); its reason tells the
// cause, and the screens' text (wire modelPathReason) is chosen by these reasons, so this test
// pins each cause's reason as core gives it on a fixture.

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func TestUXR21ModelPathCauses(t *testing.T) {
	f := newFixture(t, true)
	dir := filepath.Join(f.dir, "models", "d")
	ds := filepath.Join(dir, "DeepSeek-V4.1-Flash-MXFP4-engram.gguf")
	writeModel(t, ds, ggufBytes(10, modelKVs("deepseek4", 43, 256, 6, 38), 64))
	junk := filepath.Join(dir, "junk.gguf")
	writeModel(t, junk, []byte("not a gguf file"))
	split := filepath.Join(dir, "cut-00001-of-00002.gguf")
	writeModel(t, split, ggufBytes(10, modelKVs("deepseek4", 43, 256, 6, 38), 64))
	for _, c := range []struct{ what, path, prefix string }{
		{"unsupported structure (DeepSeek-V4.1)", ds, "unsupported GGUF: no catalog profile matches"},
		{"missing file", filepath.Join(dir, "gone.gguf"), "model file not found"},
		{"no path", "", "empty model path"},
		{"unreadable header", junk, "unsupported GGUF: GGUF "},
		{"incomplete shard set", split, "incomplete shard set"},
	} {
		_, err := f.l.Preflight(context.Background(), Request{ModelPath: c.path, NoPreset: true})
		le, ok := AsLaunchError(err)
		if !ok || le.Status != StatusFailModelPath || ExitCode(le.Status) != 3 || !strings.HasPrefix(le.Reason, c.prefix) {
			t.Fatalf("%s: %v (want fail_model_path, exit 3, reason «%s…»)", c.what, err, c.prefix)
		}
		t.Logf("%s: %s · exit %d · %s", c.what, le.Status, ExitCode(le.Status), le.Reason)
	}
}
