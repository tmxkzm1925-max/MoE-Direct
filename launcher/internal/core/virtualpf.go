package core

// GL-20: the virtual lift of the manual-prefetch edition (F-PF §7, §8.2-§8.4; L:4199; F-LS 11.7
// and OQ-5 (b)). On a virtual repack QD, the prefetch request and the manual pair are accepted on
// every Go path with the notice of §7.2; a QD below the model's wave width is refused before
// start (§7.3); MOE_DIRECT_QD and MOE_DIRECT_IO_QD_TOTAL carry the same value and the three
// other IO knobs are neither set nor passed (§7.4); an ON candidate shows N beside the N_cap the
// engine will apply (§8.2), with a notice when N_cap is 0 on every layer (§8.3).

import (
	"fmt"
	"strconv"
	"strings"
)

// The IO keys of a virtual repack (MD:2836-3020).
const (
	EnvIOQDTotal            = "MOE_DIRECT_IO_QD_TOTAL"
	EnvIOQDDemandReserve    = "MOE_DIRECT_IO_QD_DEMAND_RESERVE"
	EnvIOQDPrefetchChildMax = "MOE_DIRECT_IO_QD_PREFETCH_CHILD_MAX"
	EnvIOPartParallelism    = "MOE_DIRECT_IO_PART_PARALLELISM"
)

// The engine defaults of the knobs the launcher does not expose (D9; [A-QK]; MD:2849; U-40 (i)).
const (
	EngineDemandReserve   int64 = 4
	EngineChildMax        int64 = 4
	EnginePartParallelism int64 = 3
)

// virtualUnexposedKnobs are neither set by the launcher nor passed from the catalog or the shell
// (§7.4).
var virtualUnexposedKnobs = []string{EnvIOQDDemandReserve, EnvIOQDPrefetchChildMax, EnvIOPartParallelism}

// T18 · NoticeNCapZero (§8.3): a notice, not a refusal.
const NoticeNCapZero = "prefetch admits nothing: N_cap is 0 on every layer at this QD (the run starts; raise QD to let prefetch work)"

// refuseWaveText is T18 RefuseWave (§7.3, «작동 불가»), a text distinct from the notices.
func refuseWaveText(qd, w int64) string {
	return fmt.Sprintf("cannot work: QD %d is below this model's virtual wave width %d", qd, w)
}

// WaveWidths is T16: the virtual wave width per layer of one model, as X-VQD records it.
type WaveWidths struct {
	ProfileID string
	PerLayer  []int64
}

// Max is the start's refusal threshold (MD:2977-2978, :2988).
func (w WaveWidths) Max() int64 {
	var m int64
	for _, v := range w.PerLayer {
		m = max(m, v)
	}
	return m
}

func uniformWidths(id string, layers int, w int64) WaveWidths {
	ws := WaveWidths{ProfileID: id, PerLayer: make([]int64, layers)}
	for i := range ws.PerLayer {
		ws.PerLayer[i] = w
	}
	return ws
}

// virtualWaveWidths is X-VQD's record (End, reviews/_landing_8_X-VQD_v8-xrun.md section 1, commit
// 2f0012a07): W_l = min(P_l, io_part_parallelism 3) per layer, recorded equal to the engine's own
// refusal text on every model run. A profile without a record has no launcher-side wave check
// and no N_cap; the engine still refuses a QD below its width (MD:2995-3001).
var virtualWaveWidths = map[string]WaveWidths{
	"qwen35-122b-nonextn": uniformWidths("qwen35-122b-nonextn", 48, 3), // P_l 3 (gate, up, down)
	"deepseek-v4-flash":   uniformWidths("deepseek-v4-flash", 43, 3),   // P_l 3
	"gpt-oss-120b":        uniformWidths("gpt-oss-120b", 36, 3),        // P_l 6, capped by io_part_parallelism 3
}

// waveWidthsFor is the recorded widths of a profile.
func waveWidthsFor(profileID string) (WaveWidths, bool) {
	w, ok := virtualWaveWidths[profileID]
	return w, ok && len(w.PerLayer) > 0
}

// NCap is T17, the engine's static per-layer cap restated (DSP:2528-2544):
// N_cap(l) = min(N, floor(child_max / W_l), floor((qd_total - reserve) / W_l)), 0 when W_l <= 0
// or N <= 0, and the second term 0 when qd_total <= reserve.
func NCap(n, qdTotal, reserve, childMax, w int64) int64 {
	if w <= 0 || n <= 0 {
		return 0
	}
	c := min(n, childMax/w)
	var c2 int64
	if qdTotal > reserve {
		c2 = (qdTotal - reserve) / w
	}
	return max(0, min(c, c2))
}

// nCapPerLayer is N_cap on every layer with the QD the run passes and the engine defaults (§8.2).
func nCapPerLayer(n, qd int64, w WaveWidths) []int64 {
	out := make([]int64, len(w.PerLayer))
	for i, wl := range w.PerLayer {
		out[i] = NCap(n, qd, EngineDemandReserve, EngineChildMax, wl)
	}
	return out
}

// nCapMin is the smallest N_cap over the layers, the screen's value (U-39 (iii)).
func nCapMin(per []int64) (int64, bool) {
	if len(per) == 0 {
		return 0, false
	}
	m := per[0]
	for _, v := range per[1:] {
		m = min(m, v)
	}
	return m, true
}

// nCapZeroEverywhere is §8.3's case: N_cap 0 on every layer (a notice, not a refusal).
func nCapZeroEverywhere(per []int64) bool {
	for _, v := range per {
		if v != 0 {
			return false
		}
	}
	return len(per) > 0
}

// nCapLogLine is the log's one line with the list per layer (§8.2).
func nCapLogLine(per []int64) string {
	s := make([]string, len(per))
	for i, v := range per {
		s[i] = strconv.FormatInt(v, 10)
	}
	return "prefetch_n_cap_per_layer=" + strings.Join(s, ",")
}

// virtualOffCatalogPolicy is §7.2's condition: the run's QD is not 8, or its prefetch outcome is
// not the row's catalog policy (a manual pair other than the catalog pair, or an init request).
func virtualOffCatalogPolicy(qd int, request string, pair *ManualPair, prof *Profile) bool {
	if qd != VirtualPinnedQD || request == "init" {
		return true
	}
	if pair != nil {
		a := catalogPrefetchAxes(prof)
		return !(a.activation == "catalog-fixed" && a.tupleOK && a.k == pair.K && a.n == pair.N)
	}
	return false
}
