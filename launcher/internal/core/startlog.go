// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

// StartLogGlyphProbe is the kind of the start-log record of the glyph
// self-measurement (GL-03; decision ④, DECISION_LEDGER.md:4181): the set
// chosen, its reason and the width of each probed glyph.
const StartLogGlyphProbe = "GLYPH_PROBE"

// StartLog writes one record of the launcher's start into the diagnostic log
// (the file whose first record is the LAUNCH header).
func (l *Launcher) StartLog(kind string, data map[string]any) { l.diag.write(kind, data) }
