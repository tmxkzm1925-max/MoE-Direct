// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

package core

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strconv"
)

// ModelShape is what the need path reads from a model's GGUF files (GL-05; FIT section 5):
// the metadata the engine tables use and the expert tensors that set the slot stride. Only
// the file's content enters it: neither the file name nor general.name is used (S8-LA-12).
type ModelShape struct {
	Arch    string
	NLayer  int64 // <arch>.block_count: the executed layers L
	NEmbd   int64 // <arch>.embedding_length
	NExpert int64 // <arch>.expert_count
	NVocab  int64 // the token list's length, else <arch>.vocab_size
	// Identity is the model's content identity for the residual key (S8-LA-12: the file name
	// and general.name do not enter): a hash of the architecture's numbers and the expert table.
	Identity string
	// Experts are the expert tensors of each MoE layer (gate, up and down; FIT section 5).
	Experts map[int64][]ExpertPart
}

// ExpertPart is one expert tensor of one layer: its type, dims and absolute file offset.
type ExpertPart struct {
	Name      string
	Type      uint32
	Dims      []int64
	AbsOffset int64
	RelOffset int64 // within the data section: the same in a renamed copy
}

// MoELayers is the number of layers holding expert tensors.
func (s *ModelShape) MoELayers() int64 { return int64(len(s.Experts)) }

// ggmlBlock is a ggml type's block: values per block and bytes per block (ggml.c type traits).
var ggmlBlock = map[uint32][2]int64{
	0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20}, 6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 9: {32, 36},
	10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176}, 14: {256, 210}, 15: {256, 292},
	16: {256, 66}, 17: {256, 74}, 18: {256, 98}, 19: {256, 50}, 20: {32, 18}, 21: {256, 110},
	22: {256, 82}, 23: {256, 136}, 24: {1, 1}, 25: {1, 2}, 26: {1, 4}, 27: {1, 8}, 28: {1, 8},
	29: {256, 56}, 30: {1, 2}, 34: {256, 54}, 35: {256, 66}, 39: {32, 17},
}

var expertTensorRe = regexp.MustCompile(`^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$`)

// readModelShape reads every shard of a model: the metadata, then each tensor's info, the
// data section starting at the file's alignment (general.alignment, default 32).
func readModelShape(modelPath string) (*ModelShape, error) {
	paths, _, err := shardPaths(modelPath)
	if err != nil {
		return nil, err
	}
	s := &ModelShape{Experts: map[int64][]ExpertPart{}}
	meta := map[string]int64{}
	for _, p := range paths {
		if err := readShapeFile(p, s, meta); err != nil {
			return nil, err
		}
	}
	if s.Arch == "" {
		return nil, fmt.Errorf("GGUF has no general.architecture: %s", modelPath)
	}
	defer func() { s.Identity = s.identity(meta) }()
	s.NLayer, s.NEmbd, s.NExpert = meta[s.Arch+".block_count"], meta[s.Arch+".embedding_length"], meta[s.Arch+".expert_count"]
	// the next-token prediction layers are in the file but not executed: the engine's pool and
	// tables leave them out (Qwen: 49 blocks, L = 48 and a stride of 1,497 pages, FIT section 5)
	if n := meta[s.Arch+".nextn_predict_layers"]; n > 0 && n < s.NLayer {
		s.NLayer -= n
		for l := range s.Experts {
			if l >= s.NLayer {
				delete(s.Experts, l)
			}
		}
	}
	s.NVocab = meta["tokenizer.ggml.tokens#len"]
	if s.NVocab == 0 {
		s.NVocab = meta[s.Arch+".vocab_size"]
	}
	return s, nil
}

func readShapeFile(path string, s *ModelShape, meta map[string]int64) (err error) {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("GGUF read failed (%s): %v", path, err)
	}
	defer f.Close()
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("GGUF read failed (%s): %v", path, r)
		}
	}()
	cr := &countReader{r: f}
	g := &ggufReader{r: bufio.NewReaderSize(cr, 1<<20)}
	magic := g.bytes(4)
	if string(magic) != "GGUF" {
		return fmt.Errorf("GGUF magic mismatch: %s", path)
	}
	_ = g.u32()
	nT := g.u64()
	nKV := g.u64()
	align := int64(32)
	for i := uint64(0); i < nKV; i++ {
		key := g.str()
		t := g.u32()
		switch {
		case t == 9:
			et := g.u32()
			n := g.u64()
			meta[key+"#len"] = int64(n)
			for j := uint64(0); j < n; j++ {
				g.value(et, false)
			}
		case key == "general.architecture" && t == 8:
			s.Arch = g.str()
		default:
			v := g.value(t, true)
			if n, ok := metaLong(v); ok {
				if _, has := meta[key]; !has {
					meta[key] = n
				}
				if key == "general.alignment" && n > 0 {
					align = n
				}
			}
		}
	}
	type info struct {
		name string
		t    uint32
		dims []int64
		off  int64
	}
	infos := make([]info, 0, nT)
	for i := uint64(0); i < nT; i++ {
		name := g.str()
		nd := g.u32()
		if nd > 8 {
			panic(errors.New("tensor dims out of range"))
		}
		dims := make([]int64, nd)
		for d := range dims {
			dims[d] = int64(g.u64())
		}
		t := g.u32()
		infos = append(infos, info{name, t, dims, int64(g.u64())})
	}
	pos := cr.n - int64(g.r.Buffered())
	data := (pos + align - 1) / align * align
	for _, in := range infos {
		m := expertTensorRe.FindStringSubmatch(in.name)
		if m == nil {
			continue
		}
		l, _ := strconv.ParseInt(m[1], 10, 64)
		s.Experts[l] = append(s.Experts[l], ExpertPart{Name: in.name, Type: in.t, Dims: in.dims, AbsOffset: data + in.off, RelOffset: in.off})
	}
	return nil
}

type countReader struct {
	r io.Reader
	n int64
}

func (c *countReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// SlotStride is S of FIT section 5 (T_pool): per MoE layer, the sum over its expert parts of
// align_up(h + s, A), where s is one expert's slice, h the part's offset within a page of A
// when s is a whole number of pages (else 0); S is the largest layer's sum. ok is false when a
// part has a type or shape the table does not know.
func (s *ModelShape) SlotStride(pageBytes int64) (int64, bool) {
	if pageBytes <= 0 || len(s.Experts) == 0 {
		return 0, false
	}
	alignUp := func(v int64) int64 { return (v + pageBytes - 1) / pageBytes * pageBytes }
	var best int64
	for _, parts := range s.Experts {
		var sum int64
		for _, p := range parts {
			b, ok := ggmlBlock[p.Type]
			if !ok || len(p.Dims) < 2 || p.Dims[0]%b[0] != 0 {
				return 0, false
			}
			slice := p.Dims[0] / b[0] * b[1]
			for _, d := range p.Dims[1 : len(p.Dims)-1] {
				slice *= d
			}
			h := int64(0)
			if slice%pageBytes == 0 {
				h = p.AbsOffset % pageBytes
			}
			sum += alignUp(h + slice)
		}
		best = max(best, sum)
	}
	return best, best > 0
}

// String is a short account of the shape for the diagnostic log.
func (s *ModelShape) String() string {
	return fmt.Sprintf("%s L=%d embd=%d experts=%d vocab=%d moe_layers=%d", s.Arch, s.NLayer, s.NEmbd, s.NExpert, s.NVocab,
		s.MoELayers())
}

// identity hashes the numeric metadata (strings such as general.name are left out) and the
// expert table's types, dims and offsets within the data section, so a copy under another file
// name and general.name has the same identity.
func (s *ModelShape) identity(meta map[string]int64) string {
	keys := make([]string, 0, len(meta))
	for k := range meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	fmt.Fprintf(h, "%s|", s.Arch)
	for _, k := range keys {
		fmt.Fprintf(h, "%s=%d|", k, meta[k])
	}
	ls := make([]int64, 0, len(s.Experts))
	for l := range s.Experts {
		ls = append(ls, l)
	}
	sort.Slice(ls, func(i, j int) bool { return ls[i] < ls[j] })
	for _, l := range ls {
		for _, p := range s.Experts[l] {
			fmt.Fprintf(h, "%s:%d:%v:%d|", p.Name, p.Type, p.Dims, p.RelOffset)
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:32]
}
