package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The shard digest cache (Get-ModelShardSha256Set PS:2771-2835). The PowerShell launcher's
// cache file is read, never written; a digest this package computes goes into its own cache,
// in the same format, and both are consulted. Both sit directly under the state root (F-LS
// 8.3; GL-FX2), so the launcher's own cache has a name of its own (sub-lead ruling, GL-FX2):
// the PowerShell file under its name stays PowerShell's.

const (
	shardCacheFile    = "model_shards.cache.json"          // KV_SHARD_CACHE_FILE (PS:1059), PowerShell's
	ownShardCacheFile = "launcher_model_shards.cache.json" // this launcher's (GL-FX2)
	shardCacheVersion = 2                                  // KV_SHARD_CACHE_VERSION (PS:1060)
)

func shardCacheKey(id FileIdentity) string {
	return fmt.Sprintf("v2|%s|%d|%d", id.ID, id.Size, id.MTime)
}

// readShardCache reads one cache file into key -> sha (lower case); anything unreadable is
// an empty cache, as in PowerShell.
func readShardCache(path string) map[string]string {
	out := map[string]string{}
	v, err := readJSONFileStrict(path)
	if err != nil {
		return out
	}
	o, _ := asObj(v)
	if ver, ok := jNonNegInt(o["cache_version"]); !ok || ver != shardCacheVersion {
		return out
	}
	es, _ := jArr(o["entries"])
	for _, e := range es {
		eo, _ := asObj(e)
		k, _ := eo["key"].(string)
		s, _ := eo["sha256"].(string)
		if k != "" && isSha256Hex(s) {
			out[k] = strings.ToLower(s)
		}
	}
	return out
}

func (l *Launcher) psShardCachePath() string { return filepath.Join(l.opt.StateDir, shardCacheFile) }
func (l *Launcher) ownShardCachePath() string {
	return filepath.Join(l.opt.StateDir, ownShardCacheFile)
}

// ShardDigests are the source digests of a shard set, in shard order. Known[i] is false when
// shard i is in neither cache and hashing was not allowed.
type ShardDigests struct {
	Sha256   []string
	Known    []bool
	Complete bool
	Hashed   int // shards hashed by this call
}

// shardDigests looks every shard up in the two caches; with hash set it hashes the misses
// (and records them in the launcher's own cache), as Get-ModelShardSha256Set does at identify.
// With hash unset (the model list) nothing is hashed.
func (l *Launcher) shardDigests(ctx context.Context, paths []string, hash bool) (ShardDigests, error) {
	ps := readShardCache(l.psShardCachePath())
	own := readShardCache(l.ownShardCachePath())
	d := ShardDigests{Sha256: make([]string, len(paths)), Known: make([]bool, len(paths)), Complete: true}
	dirty := false
	for i, p := range paths {
		id, idErr := l.os.identity(p)
		key := ""
		if idErr == nil {
			key = shardCacheKey(id)
			if s, ok := ps[key]; ok {
				d.Sha256[i], d.Known[i] = s, true
				continue
			}
			if s, ok := own[key]; ok {
				d.Sha256[i], d.Known[i] = s, true
				continue
			}
		} else {
			l.diag.write("IDENTIFY_SHARD_IDENTITY_UNAVAILABLE", map[string]any{"path": p, "reason": idErr.Error()})
		}
		if !hash {
			d.Complete = false
			continue
		}
		s, err := hashFileCtx(ctx, p)
		if err != nil {
			l.diag.write("IDENTIFY_SHARD_HASH_FAILED", map[string]any{"path": p, "reason": err.Error()})
			return d, err
		}
		d.Sha256[i], d.Known[i] = s, true
		d.Hashed++
		if key != "" {
			own[key] = s
			dirty = true
		}
	}
	if dirty {
		if err := writeShardCache(l.ownShardCachePath(), own); err != nil {
			l.diag.write("IDENTIFY_SHARD_CACHE_FAILED", map[string]any{"path": l.ownShardCachePath(), "reason": err.Error()})
		}
	}
	return d, nil
}

func writeShardCache(path string, m map[string]string) error {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	type entry struct {
		Key    string `json:"key"`
		Sha256 string `json:"sha256"`
	}
	obj := struct {
		CacheVersion int     `json:"cache_version"`
		Entries      []entry `json:"entries"`
	}{CacheVersion: shardCacheVersion}
	for _, k := range keys {
		obj.Entries = append(obj.Entries, entry{k, m[k]})
	}
	b, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(path, b)
}

type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c ctxReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

// hashFileCtx is Get-FileSha256Lower with cancellation.
func hashFileCtx(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.CopyBuffer(h, ctxReader{ctx, f}, make([]byte, 4<<20)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
