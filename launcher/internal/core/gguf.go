package core

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// splitRegex is SPLIT_REGEX (PS:2555).
var splitRegex = regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.gguf$`)

// shardPaths is Get-ShardPaths (PS:2556-2577): the full sibling set of a split model, or the
// one file.
func shardPaths(modelPath string) ([]string, bool, error) {
	base := filepath.Base(modelPath)
	m := splitRegex.FindStringSubmatch(base)
	if m == nil {
		return []string{modelPath}, false, nil
	}
	dir := filepath.Dir(modelPath)
	cnt, _ := strconv.Atoi(m[3])
	if cnt < 1 {
		return nil, false, fmt.Errorf("split count 0: %s", base)
	}
	var paths []string
	for i := 1; i <= cnt; i++ {
		sib := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", m[1], i, cnt))
		if st, err := os.Stat(sib); err != nil || st.IsDir() {
			return nil, false, fmt.Errorf("incomplete shard set: missing sibling %s (expected %d shards)", sib, cnt)
		}
		paths = append(paths, sib)
	}
	return paths, true, nil
}

// GGUFHeader is the part of a GGUF header the launcher reads (Read-GgufHeader PS:2630-2674).
type GGUFHeader struct {
	Path        string
	Version     uint32
	NTensors    int64
	Meta        map[string]any // int64, float64, bool or string; arrays are skipped
	FileBytes   int64
	SourceIndex int
}

type ggufWant struct {
	exact  []string
	suffix []string
	count  int
}

var ggufSuffixes = []string{".block_count", ".expert_count", ".expert_used_count"}

// wantIdentify is the identify read: 4 keys, 7 when the file is one shard of a split set.
func wantIdentify(split bool) ggufWant {
	w := ggufWant{exact: []string{"general.architecture", "split.count", "split.no", "split.tensors.count"},
		suffix: ggufSuffixes, count: 4}
	if split {
		w.count = 7
	}
	return w
}

// wantLabel is LabelMode (PS:2650), plus general.file_type for the quant column of the list
// (an addition: the PowerShell menu shows no quant; landing, deviation 3).
func wantLabel() ggufWant {
	return ggufWant{exact: []string{"general.architecture", "general.file_type"}, suffix: ggufSuffixes, count: 5}
}

type ggufReader struct {
	r   *bufio.Reader
	err error
}

func (g *ggufReader) u32() uint32 {
	var b [4]byte
	if _, err := io.ReadFull(g.r, b[:]); err != nil {
		panic(err)
	}
	return binary.LittleEndian.Uint32(b[:])
}

func (g *ggufReader) u64() uint64 {
	var b [8]byte
	if _, err := io.ReadFull(g.r, b[:]); err != nil {
		panic(err)
	}
	return binary.LittleEndian.Uint64(b[:])
}

func (g *ggufReader) bytes(n int) []byte {
	b := make([]byte, n)
	if _, err := io.ReadFull(g.r, b); err != nil {
		panic(err)
	}
	return b
}

func (g *ggufReader) skip(n uint64) {
	for n > 0 {
		step := n
		if step > 1<<30 {
			step = 1 << 30
		}
		d, err := g.r.Discard(int(step))
		if err != nil {
			panic(err)
		}
		n -= uint64(d)
	}
}

// str is Read-GgufString (PS:2579-2585): a length above 16 MiB is refused.
func (g *ggufReader) str() string {
	n := g.u64()
	if n > 16777216 {
		panic(errors.New("gguf string length out of range"))
	}
	return string(g.bytes(int(n)))
}

// value is Read-GgufValue (PS:2593-2615).
func (g *ggufReader) value(t uint32, want bool) any {
	switch t {
	case 0:
		return pick(want, int64(g.bytes(1)[0]))
	case 1:
		return pick(want, int64(int8(g.bytes(1)[0])))
	case 2:
		return pick(want, int64(binary.LittleEndian.Uint16(g.bytes(2))))
	case 3:
		return pick(want, int64(int16(binary.LittleEndian.Uint16(g.bytes(2)))))
	case 4:
		return pick(want, int64(g.u32()))
	case 5:
		return pick(want, int64(int32(g.u32())))
	case 6:
		return pick(want, float64(math.Float32frombits(g.u32())))
	case 7:
		return pick(want, g.bytes(1)[0] != 0)
	case 8:
		if want {
			return g.str()
		}
		g.skip(g.u64())
		return nil
	case 9:
		et := g.u32()
		n := g.u64()
		for i := uint64(0); i < n; i++ {
			g.value(et, false)
		}
		return nil
	case 10:
		return pick(want, int64(g.u64()))
	case 11:
		return pick(want, int64(g.u64()))
	case 12:
		return pick(want, math.Float64frombits(g.u64()))
	}
	panic(fmt.Errorf("unknown gguf value type %d", t))
}

func pick(want bool, v any) any {
	if want {
		return v
	}
	return nil
}

// readGGUFHeader is Read-GgufHeader: magic, version, tensor count, then the key/value pairs
// until the wanted number of keys is found.
func readGGUFHeader(path string, w ggufWant) (h *GGUFHeader, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("GGUF header parse failed (%s): %v", path, err)
	}
	defer f.Close()
	defer func() {
		if r := recover(); r != nil {
			h = nil
			err = fmt.Errorf("GGUF header parse failed (%s): %v", path, r)
		}
	}()
	g := &ggufReader{r: bufio.NewReaderSize(f, 1<<20)}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(g.r, magic); err != nil || string(magic) != "GGUF" {
		return nil, fmt.Errorf("GGUF magic mismatch: %s", path)
	}
	h = &GGUFHeader{Path: path, Meta: map[string]any{}}
	h.Version = g.u32()
	h.NTensors = int64(g.u64())
	nKV := g.u64()
	got := 0
	for i := uint64(0); i < nKV; i++ {
		key := g.str()
		t := g.u32()
		want := false
		for _, e := range w.exact {
			if strings.EqualFold(key, e) {
				want = true
				break
			}
		}
		if !want {
			for _, s := range w.suffix {
				if strings.HasSuffix(key, s) {
					want = true
					break
				}
			}
		}
		v := g.value(t, want)
		if want {
			if _, ok := h.Meta[key]; !ok {
				got++
			}
			h.Meta[key] = v
		}
		if got >= w.count {
			break
		}
	}
	if st, err := f.Stat(); err == nil {
		h.FileBytes = st.Size()
	}
	return h, nil
}

// ShardSet is the identified model (Get-ModelShardSet PS:2676-2749).
type ShardSet struct {
	Shards       []*GGUFHeader
	Arch         string
	Meta         map[string]any
	TotalBytes   int64
	TotalTensors int64
	IsSplit      bool
}

func metaLong(v any) (int64, bool) {
	switch x := v.(type) {
	case int64:
		return x, true
	case float64:
		return int64(x), true
	case bool:
		if x {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// ArchLong is Get-ArchMetaLong (PS:2751-2756).
func (s *ShardSet) ArchLong(suffix string) (int64, bool) {
	v, ok := s.Meta[s.Arch+suffix]
	if !ok {
		return 0, false
	}
	return metaLong(v)
}

// readModelShardSet is Get-ModelShardSet.
func readModelShardSet(modelPath string) (*ShardSet, *LaunchError) {
	paths, split, err := shardPaths(modelPath)
	if err != nil {
		return nil, stop(StatusFailModelPath, "%v", err)
	}
	set := &ShardSet{IsSplit: split, Meta: map[string]any{}}
	for i, p := range paths {
		h, err := readGGUFHeader(p, wantIdentify(split))
		if err != nil {
			return nil, stop(StatusFailModelPath, "unsupported GGUF: %v", err)
		}
		h.SourceIndex = i
		set.Shards = append(set.Shards, h)
	}
	for _, h := range set.Shards {
		if v, ok := h.Meta["general.architecture"]; ok {
			a := fmt.Sprint(v)
			if set.Arch == "" {
				set.Arch = a
			} else if a != set.Arch {
				return nil, stop(StatusFailModelPath, "architecture metadata conflicts between shards: %s vs %s", set.Arch, a)
			}
		}
	}
	if set.Arch == "" {
		return nil, stop(StatusFailModelPath, "unsupported GGUF: general.architecture absent in every shard")
	}
	for _, h := range set.Shards {
		set.TotalTensors += h.NTensors
	}
	for _, h := range set.Shards {
		if v, ok := h.Meta["split.count"]; ok {
			if sc, _ := metaLong(v); sc != int64(len(set.Shards)) {
				return nil, stop(StatusFailModelPath, "incomplete shard set: split.count(%d) != discovered shards(%d)", sc, len(set.Shards))
			}
		}
		if v, ok := h.Meta["split.no"]; ok {
			if sn, _ := metaLong(v); sn != int64(h.SourceIndex) {
				return nil, stop(StatusFailModelPath, "incomplete shard set: split.no(%d) != source_index(%d)", sn, h.SourceIndex)
			}
		}
		if v, ok := h.Meta["split.tensors.count"]; ok {
			if stc, _ := metaLong(v); stc != set.TotalTensors {
				return nil, stop(StatusFailModelPath, "incomplete shard set: split.tensors.count(%d) != summed tensors(%d)", stc, set.TotalTensors)
			}
		}
	}
	for _, h := range set.Shards {
		for k, v := range h.Meta {
			if _, ok := set.Meta[k]; !ok {
				set.Meta[k] = v
			}
		}
		set.TotalBytes += h.FileBytes
	}
	return set, nil
}

// fileTypeNames is llama_ftype of the engine tree (include/llama.h:117-160), for the quant
// column; general.file_type carries this number.
var fileTypeNames = map[int64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
	11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
	17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
	23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
	29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
	38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}

// QuantName names a general.file_type value; "" when unknown or absent.
func QuantName(meta map[string]any) string {
	v, ok := meta["general.file_type"]
	if !ok {
		return ""
	}
	n, ok := metaLong(v)
	if !ok {
		return ""
	}
	if s, ok := fileTypeNames[n]; ok {
		return s
	}
	return fmt.Sprintf("file_type %d", n)
}
