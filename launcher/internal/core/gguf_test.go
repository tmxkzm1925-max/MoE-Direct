package core

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGGUFHeaderAndShardSet(t *testing.T) {
	dir := t.TempDir()
	one := filepath.Join(dir, "single.gguf")
	os.WriteFile(one, ggufBytes(7, modelKVs("qwen35moe", 49, 256, 8, 15), 100), 0o644)
	set, le := readModelShardSet(one)
	if le != nil {
		t.Fatal(le)
	}
	if set.Arch != "qwen35moe" || set.IsSplit || len(set.Shards) != 1 || set.TotalTensors != 7 {
		t.Fatalf("set %+v", set)
	}
	if n, _ := set.ArchLong(".block_count"); n != 49 {
		t.Errorf("block_count %d", n)
	}
	h, err := readGGUFHeader(one, wantLabel())
	if err != nil || QuantName(h.Meta) != "Q4_K_M" {
		t.Errorf("label read: %v quant %q", err, QuantName(h.Meta))
	}

	// A split set: split.count / split.no / split.tensors.count cross-checked.
	split := func(i, cnt, no, tensors, total int) string {
		p := filepath.Join(dir, "m-0000"+string(rune('0'+i))+"-of-0000"+string(rune('0'+cnt))+".gguf")
		kvs := append(modelKVs("gpt-oss", 36, 128, 4, 38), kv{"split.count", 4, cnt}, kv{"split.no", 4, no}, kv{"split.tensors.count", 4, total})
		os.WriteFile(p, ggufBytes(uint64(tensors), kvs, 10), 0o644)
		return p
	}
	p1 := split(1, 2, 0, 3, 5)
	split(2, 2, 1, 2, 5)
	set, le = readModelShardSet(p1)
	if le != nil {
		t.Fatal(le)
	}
	if !set.IsSplit || len(set.Shards) != 2 || set.TotalTensors != 5 {
		t.Fatalf("split set %+v", set)
	}
	split(2, 2, 0, 2, 5) // wrong split.no in shard 2
	_, le = readModelShardSet(p1)
	wantStatus(t, le, StatusFailModelPath, "split.no(0) != source_index(1)")

	p3 := filepath.Join(dir, "x-00001-of-00003.gguf")
	os.WriteFile(p3, ggufBytes(1, modelKVs("gpt-oss", 1, 1, 1, 1), 0), 0o644)
	_, le = readModelShardSet(p3)
	wantStatus(t, le, StatusFailModelPath, "incomplete shard set: missing sibling")

	bad := filepath.Join(dir, "bad.gguf")
	os.WriteFile(bad, []byte("GGUX0000"), 0o644)
	_, le = readModelShardSet(bad)
	wantStatus(t, le, StatusFailModelPath, "GGUF magic mismatch")
}
