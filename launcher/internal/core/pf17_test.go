package core

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// S8-PF-17 (K188; F-PF §10, KNOWN row 188, RRB:90): a derived profile (a GGUF outside the
// catalog whose structure has a template) with the init request is refused: the derived-profile
// writer gives it `prefetch_promotion_hold: true`, a derived profile is never validated (its
// evidence is fixed at unverified), and the hold key refuses init with the existing literal of
// P4 §1.1 (OFF phase4_hold_unresolved); the catalog request ends as OFF
// not_opted_in_evidence_unverified. The hold key never lets init through (F-PF §10.2; the
// S8-PF-17 literal confirmed by Astra 8_19). The catalog rows are unchanged (PLAN:376).
// «Decision and literal per case» (F-PF §10.4).
func TestS8PF17(t *testing.T) {
	rec, models := la31Load(t)
	ctx := context.Background()
	realCatalog := filepath.Join("..", "..", "..", "launcher", "models.json")
	catalogBefore, err := fileSha256(realCatalog)
	if err != nil {
		t.Fatal(err)
	}

	const holdText = "a derived profile holds prefetch_promotion_hold = true by contract (F-PF §10.1)"
	archs := make([]string, 0, len(rec.Derived))
	for a := range rec.Derived {
		archs = append(archs, a)
	}
	sort.Strings(archs)

	t.Run("writer-holds-the-key", func(t *testing.T) {
		// F-PF §10.4 case 1: the derived profile object holds prefetch_promotion_hold: true,
		// with the evidence and activation of P4:89-90 and no tuple.
		for _, arch := range archs {
			tp, why := parseTemplatePlan(la31Plan(t, arch))
			if tp == nil {
				t.Fatalf("%s: %s", arch, why)
			}
			o, err := newDerivedProfile(tp, la31PlanOut, nil)
			if err != nil {
				t.Fatal(err)
			}
			if h, ok := o["prefetch_promotion_hold"].(bool); !ok || !h {
				t.Errorf("%s: prefetch_promotion_hold %v", arch, o["prefetch_promotion_hold"])
			}
			if psString(o["prefetch_evidence"]) != "unverified" || psString(o["prefetch_activation"]) != "off" || o["prefetch"] != nil {
				t.Errorf("%s: prefetch axes %v %v %v", arch, o["prefetch_evidence"], o["prefetch_activation"], o["prefetch"])
			}
			if v := validateDerivedProfile(o); v != "" {
				t.Errorf("%s: the writer's profile fails the validator: %s", arch, v)
			}
		}
	})

	t.Run("validator-requires-the-key", func(t *testing.T) {
		// A derived profile without the hold, or with it false, is not the writer's: refused
		// with its literal; one that claims performance_validated is refused as before.
		tp, _ := parseTemplatePlan(la31Plan(t, "gpt-oss"))
		for name, c := range map[string]struct {
			edit func(jobj)
			want string
		}{
			"hold absent": {func(o jobj) { delete(o, "prefetch_promotion_hold") }, "required key 'prefetch_promotion_hold' missing in the derived profile"},
			"hold false":  {func(o jobj) { o["prefetch_promotion_hold"] = false }, holdText},
			"hold string": {func(o jobj) { o["prefetch_promotion_hold"] = "true" }, holdText},
			"validated": {func(o jobj) {
				g, _ := asObj(o["gates"])
				g["performance_validated"] = true
			}, "a derived profile can never claim performance_validated"},
		} {
			o, _ := newDerivedProfile(tp, la31PlanOut, nil)
			c.edit(o)
			if got := validateDerivedProfile(o); got != c.want {
				t.Errorf("%s: %q, want %q", name, got, c.want)
			}
		}
	})

	t.Run("decision-per-request", func(t *testing.T) {
		// F-PF §10.4 cases 2 and 3 at the resolver, with every other condition met (probe ok,
		// QD 8, the template selection); and the contrast of §10.5: the same row without the
		// hold would start init_v1-unvalidated, so the key is what refuses it.
		tp, _ := parseTemplatePlan(la31Plan(t, "gpt-oss"))
		o, _ := newDerivedProfile(tp, la31PlanOut, nil)
		p := profileFromDerived(o, tp, la31Plan(t, "gpt-oss"))
		for _, c := range []struct{ request, reason, echo string }{
			{"init", "phase4_hold_unresolved", "off(reason=phase4_hold_unresolved)"},
			{"catalog", "not_opted_in_evidence_unverified", "off(reason=not_opted_in_evidence_unverified)"},
			{"", "not_opted_in_evidence_unverified", "off(reason=not_opted_in_evidence_unverified)"},
		} {
			d := resolvePrefetch(p, true, optInOf(c.request), 8, SelectionTemplate, c.request)
			if d.On || d.OffReason != c.reason || d.Echo != c.echo || d.Warning != "" {
				t.Errorf("request %q: %+v", c.request, d)
			}
		}
		if prefetchArm(p, armInit, SelectionTemplate) != armNone {
			t.Errorf("an init arm on a derived profile")
		}
		unheld := profileFromDerived(o, tp, la31Plan(t, "gpt-oss"))
		raw := jobj{}
		for k, v := range o {
			raw[k] = v
		}
		raw["prefetch_promotion_hold"] = false
		unheld.Raw = raw
		if d := resolvePrefetch(unheld, true, armInit, 8, SelectionTemplate, "init"); !d.On || d.Provenance != "init_v1-unvalidated" {
			t.Errorf("contrast without the hold: %+v", d)
		}
	})

	t.Run("start-per-request", func(t *testing.T) {
		// The same two cases through the start: a GGUF outside the catalog, its plan run, its
		// template setup present; init and catalog requests both start with prefetch off, with
		// the literal per case and no init warning.
		f := newFixture(t, false)
		catBefore, _ := fileSha256(f.catalogPath)
		mdir := filepath.Join(f.dir, "models", "tpl")
		model := filepath.Join(mdir, "gpt-oss-tpl.gguf")
		b, err := os.ReadFile(filepath.Join(models, "gpt-oss-tpl.gguf"))
		if err != nil {
			t.Fatal(err)
		}
		writeModel(t, model, b)
		f.l.deriveExec = &la31PlanExec{text: map[string]string{"gpt-oss-tpl.gguf": la31Plan(t, "gpt-oss")}}
		w := rec.Derived["gpt-oss"]
		_, err = f.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true})
		var rn *RepackNeededError
		if !errors.As(err, &rn) {
			t.Fatalf("want the template setup first, got %v", err)
		}
		out := filepath.Join(mdir, "repack")
		writeJSON(t, filepath.Join(out, DerivedExpectFileName), map[string]any{"derived_from": "gpt-oss@1"})
		dsha, _ := fileSha256(filepath.Join(out, DerivedExpectFileName))
		os.WriteFile(filepath.Join(out, "experts.bin"), []byte{0}, 0o644)
		writePackedArtifacts(t, out, w.LockID, dsha)
		for _, c := range []struct{ request, reason string }{
			{"init", "phase4_hold_unresolved"},
			{"catalog", "not_opted_in_evidence_unverified"},
		} {
			res, err := f.l.Preflight(ctx, Request{ModelPath: model, NoPreset: true, Settings: map[string]string{"prefetch": c.request}})
			if err != nil {
				t.Fatalf("request %q: %v", c.request, err)
			}
			pf := res.Config.Prefetch
			if res.Profile == nil || res.Profile.Derived == nil || res.Selection != SelectionTemplate {
				t.Fatalf("request %q: not the derived profile: %+v", c.request, res.Profile)
			}
			if pf.On || pf.OffReason != c.reason || pf.Echo != offEcho(c.reason) || pf.Request != c.request || res.Start.PrefetchOn {
				t.Errorf("request %q: %+v (start prefetch on %v)", c.request, pf, res.Start.PrefetchOn)
			}
			for _, n := range res.Notices {
				if n == "prefetch_init_v1 is an unvalidated starting point (K=n_expert_used, N=min(4,QD-1,floor(QD/2))): no performance claim is made for this pair" {
					t.Errorf("request %q: the init warning was given", c.request)
				}
			}
		}
		// The catalog rows are unchanged: the fixture catalog file, and the catalog as read.
		if catAfter, _ := fileSha256(f.catalogPath); catAfter != catBefore {
			t.Errorf("the fixture catalog changed")
		}
		cat, err := ReadCatalog(f.catalogPath, f.root)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range cat.Profiles {
			if p.Derived != nil || strings.HasPrefix(p.ID, "derived-") {
				t.Errorf("a derived row in the catalog: %s", p.ID)
			}
		}
	})

	if after, err := fileSha256(realCatalog); err != nil || after != catalogBefore {
		t.Errorf("the catalog models.json changed: %v", err)
	}
}
