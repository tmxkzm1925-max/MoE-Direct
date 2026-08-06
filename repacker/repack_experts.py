# -*- coding: utf-8 -*-
"""repack-v2: GGUF 전문가 텐서 재팩 도구 (REPACK_V2_DESIGN.md §2/§3/§5 동결 사양의 구현).

목적: MoE GGUF(gpt-oss / Mistral / Qwen / K2.6 등 다모델)의 routed 전문가(expert) 텐서를
(layer,expert) 단위 연속·섹터 정렬 레코드로 재배치한 사본(experts.bin)+매니페스트(manifest.json,
schema_version "2.0")를 생성하고, 소스와 출력을 디스크에서 각각 다시 읽어 SHA-256 전수 비교
(독립 2패스)로 무손실을 증명한다. "이동만, 변환 없음"(D3 무손실 계약) — 양자화·타입 변환 없음.

v1(SCHEMA v1.1, gpt-oss 전용) 대비 v2 확장(설계 §2):
  ① 멀티-shard 입력(split 패턴 발견·형제 전수·shard별 헤더·전역 유일·source_index/abs_offset)
  ② 동적 routed 스키마(arch 정확 키 매칭·정규식 후보·moe_layers 0시작·연속 가정 금지·
     층별 파트 집합 동일성·expert_axis==마지막&&ne[last]==n_expert·inner order 동결)
  ③ type-trait nbytes 표(버전고정)·표 밖=중단·ne0%bv==0·gap 검사 소스 정렬 패딩만 허용·
     읽기는 이론 nbytes만
  ④ 층별 record layout(payload[l]/stride[l]/record_base prefix-sum/slot_stride_max)
  ⑤ reference lock(--profile 필수·저장소 동결 expect 카탈로그·코드 내 EXPECT_CATALOG 대조·
     미등록/불일치/비엄격 JSON=RepackAbort·DEFAULT_OUT_DIR 폐지)
  ⑥ I/O 층 단위 버퍼링 + preflight 가용 RAM 검사(가용 < 피크 층 expert bytes + 2GiB = 중단)

레코드 내부 순서(동결): weights (gate→up→down | gate_up→down) → bias 동순.
  - gpt-oss(separate 3w+3b): gate.w → up.w → down.w → gate.b → up.b → down.b
    (= v1 RECORD_ORDER 정확 재현 — ⓪-a 바이트 동일성의 전제).
  - Mistral(fused 2-part, bias 無): gate_up.w → down.w.
레코드 순서: moe_layer 오름차순 × expert 오름차순. 레코드 시작 offset은 항상 해당 층의
정렬 stride[l] 배수(A=max(4096, 논리 섹터, 물리 섹터)) — 남는 자리는 0x00 패딩.

사용법:
    python repack_experts.py --plan --profile gpt-oss-120b --model <gguf> --out <dir>
    python repack_experts.py --selftest
    python repack_experts.py --profile gpt-oss-120b --model <gguf> --out <dir> [--force]
    python repack_experts.py --verify-only --profile gpt-oss-120b --model <gguf> --out <dir>
      (재팩 없이 기존 산출물 1회 전체 재검증 → verify_report.json 에 새 레코드 append.
       구 report(manifest_sha256 필드 없음)로 만들어진 산출물의 무재팩 승급 경로.)

요건: Python 3.11 · 표준 라이브러리만(ctypes 포함) · Windows 전용(섹터 질의·RAM 질의는
ctypes로 IOCTL_STORAGE_QUERY_PROPERTY/GetDiskFreeSpaceW/GlobalMemoryStatusEx 직접 호출).

설계 해석 지점(완료 보고서 "해석 지점"에도 명기):
  - repack_log.jsonl 은 --out 이 아니라 이 스크립트 디렉토리(bench/repack/)에 append-only 누적
    (plan/selftest/repack 전 실행을 아우르는 도구 실행 이력. --plan 은 쓰기 0이라 기록 안 함).
  - type-trait 표는 b10057 ggml-common.h(디스크 실측: D:\\moe-tools\\llama.cpp-src-b10057\\
    ggml\\src\\ggml-common.h)에서 블록 정의 전항 대조로 도출(selftest 관문 _selftest_traits).
"""
import argparse
import ctypes
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from ctypes import wintypes
from datetime import datetime, timezone

SCRIPT_VERSION = '2.0.0'
SCHEMA_VERSION = '2.0'
EXPECT_SCHEMA_VERSION = '1.0'

# §0 routed 정규식(설계 동결): _shexp(shared)·ffn_gate_inp(router)·exp_probs_b(gating bias)는
# 비매치(상주 유지). fused(gate_up)·separate(gate/up/down) 공통. weight/bias 양쪽.
TENSOR_NAME_RE = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.(weight|bias)$')

# 레코드 내부 순서 결정용 kind 우선순위(동결). gate_up 과 up 은 fused/separate 배타이므로
# 우선순위 충돌 없음. weights 는 이 우선순위로, bias 는 동순으로 정렬한다.
KIND_PRIORITY = {'gate': 0, 'up': 1, 'gate_up': 1, 'down': 2}

# §2-3 type-trait nbytes 표(버전고정, (block_values, block_bytes)). b10057 ggml-common.h 전항
# 대조 완료(_selftest_traits 가 필드 분해로 재도출해 이 표와 일치 검증):
#   F32:1/4  F16:1/2(ggml_half=2)  MXFP4:32/17(e1+qs16, QK_MXFP4=32)
#   Q3_K:256/110(hmask32+qs64+scales12+d2)  Q4_K:256/144(dm4+scales12+qs128)
#   Q5_K:256/176(dm4+scales12+qh32+qs128)   Q6_K:256/210(ql128+qh64+scales16+d2)
#   Q8_0:32/34(d2+qs32)
QUANT_TRAITS = {
    'F32':   (1, 4),
    'F16':   (1, 2),
    'MXFP4': (32, 17),
    'Q3_K':  (256, 110),
    'Q4_K':  (256, 144),
    'Q5_K':  (256, 176),
    'Q6_K':  (256, 210),
    'Q8_0':  (32, 34),
}

# 코드 내 동결 expect 카탈로그(설계 §2-5·부록A): {profile_id: {sha256: 승인 digest, scope}}.
# --plan/본실행은 --profile 로 지정한 id 의 expect 파일(bench/repack/expects/<id>.expect.json)을
# 로드해 sha256 이 이 승인 digest 와, routed_scope 가 이 카탈로그의 scope 와 정확히 일치할 때만
# 사용한다(미등록 id·digest 불일치·scope 불일치·비엄격 JSON=RepackAbort). 자동 생성·우회 옵션
# 없음. 신규 모델 추가 = expect 파일 + 이 카탈로그 갱신 커밋(교차 심의 대상).
EXPECT_CATALOG = {
    'gpt-oss-120b': {'sha256': 'be127b3e27454eb369eeca253decf7eab1ac4964849a731d352ffe72d5be828e', 'scope': 'all'},
    'kimi-k2.6-ram-447gb': {'sha256': 'fee9902ca2ed77b4b0c06be49ceb4463fe0ff1d5d42750ed43cae82f162681f6', 'scope': 'all'},
    # 사다리 ①(26-07-24 등재): 원천=bench_results/g5/gguf_map_mistral4_s{1,2}.json(2-shard 실측
    # 합산 — routed 72·expert 70,212,648,960B=이론 산술 정확 일치·layer19 경계 분산).
    'mistral-small-4': {'sha256': '00a9f67ae8a5844b1ebbf2d82f8f2d869478d953df010f02d8715ad6c7865d7c', 'scope': 'all'},
    # 사다리 ②(26-07-24 등재): 원천=bench_results/g5/gguf_map_qwen122_s{1,2}.json(2-shard 실측 —
    # ★fused 추정 반증: separate 3-part·49층 전층 MoE·routed 147=3×49·Q4_K 120/Q6_K 24/Q8_0 3
    # 혼용·expert 72,779,563,008B·layer25 경계 분산).
    'qwen35-122b': {'sha256': '982639f39e15f04716415a65b909f5d761884880d1ddab87c04d9734e55bc15f', 'scope': 'all'},
    # 부록A(26-07-25 등재): qwen35-122b 와 동일 원본에서 NextN(마지막 1층) 제외 — routed
    # 147→144(3개 제외)·expert_bytes_total 72,779,563,008→70,212,648,960B.
    'qwen35-122b-nonextn': {'sha256': '882fdac06cf7d75c5d9944f6eb373a204667186cb96929dff605e5f7f9f9a800', 'scope': 'execution'},
    # 노트북 티어(26-07-28 등재): 원천=bench/laptop_kit/qwen35b_map.json(노트북 실기 덤프·단일
    # shard 22,016,023,168B·733텐서 전량). 40층 전층 MoE·separate 3-part·bias 無·routed 120=3×40·
    # Q4_K 80(gate/up)/Q5_K 40(down)·expert 19,461,570,560B. ★MTP/NextN 텐서·nextn_predict_layers
    # KV 모두 부재 → scope='all' 이 곧 실행 집합(nonextn 분리 불요).
    'qwen35-35b': {'sha256': '616eb096c272ecdc299166afea9c47ebf84f93ae10363c462c84c6c358262d83', 'scope': 'all'},
    # 397B(26-07-29 등재): 원천=bench_results/g5/gguf_map_qwen397_s{1..6}.json(6-shard 실측 합산 —
    # HF 공표 SHA 6/6 대조 PASS 원본). 60층 전층 MoE(per_layer 0..59)·separate 3-part·routed
    # 180=3×60·expert 233,538,846,720B·★n_expert=512(카탈로그 첫 512E — 기존 실증 상한 384E·재팩
    # verify 전수가 게이트)·top-10. shard1=메타 전용(텐서 0·data_start=raw header_end 10,943,537·
    # file_bytes 10,943,552 는 15B trailing padding 포함 — 양자 상이가 정상). ★MTP/NextN
    # 텐서·KV 모두 부재(dense 목록·55 KV 전수 확인) → scope='all'(nonextn 분리 불요·35B 동형).
    # ★26-07-29 정정: shard1(메타 전용·텐서 0)의 data_start 를 upstream 정합값 10,943,537 로
    # 수리(구 등재값은 무조건 패딩한 10,943,552 — gguf.cpp:756 참조. 실기동 seal fail-close
    # 발화 후 수리). expect 재해시 → sha 00ad762c… → 21b698ca….
    'qwen35-397b': {'sha256': '21b698ca0267724ceebbeb4255585dcb24eef9ad7d9d3bad6480833ef3d77f14', 'scope': 'all'},
    # MiniMax M2.7(26-07-29 등재): 원천=bench_results/g5/gguf_map_minimax_m27.json(4-shard 헤더
    # 실측·HF 공표 SHA 4/4 대조 PASS·총 809텐서). 62층 전층 MoE·256E·top-8·separate 3-part·
    # routed 186=3×62(Q4_K 156/Q6_K 30)·expert 135,725,580,288B(원본의 98.1% — 최고 비중)·
    # 카탈로그 첫 minimax-m2 arch(gating_func=2 sigmoid·exp_probs_b.bias 층별 존재=K2.6 동형·
    # 재팩 무관). ★MTP/NextN 텐서·KV·shexp 모두 부재(전수 확인) → scope='all'.
    'minimax-m27': {'sha256': '8e58ea0a629dff055b5c95221322d8ba9fada4680588ec85618b827321a9234a', 'scope': 'all'},
    # DeepSeek-V4-Flash(26-08-01 등재): 원천=bench_results/g5/gguf_map_dsv4flash.json(단일 shard
    # 실측 156,378,344,992B — HF 공표 LFS SHA b43b3c3a… 전체 대조 PASS·revision ed48c7a2…·1328
    # 텐서 오프셋 재구성 폐합 0불일치·tail padding 0). 43층 전층 MoE(선두 dense 無)·256E·top-6·
    # separate 3-part·routed 129=3×43 전량 MXFP4(★카탈로그 첫 deepseek4 arch — QAT 네이티브라
    # 비트 동일=무손실·MXFP4 타입은 gpt-oss 기검증·separate 3-part는 qwen/minimax 기검증)·
    # expert 147,169,738,752B(파일의 94.1%)·shexp 129텐서 Q8_0 1.07GiB=트렁크 상주 재팩 무관.
    # ★MTP/NextN 텐서·KV 모두 부재(전수 확인) → scope='all'(35B/397B 동형·nonextn 분리 불요).
    'deepseek-v4-flash': {'sha256': 'b2be25b3d3739cdbf97aca583fc993070e8a98a75f1d4b1d7714eb6b6af1ab9b', 'scope': 'all'},
}

EXPECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expects')

# ---------------------------------------------------------------------------
# OPEN_ARCH A축 상수(OPEN_ARCH_DESIGN.md v0.2 §1 — experimental·비공개 feature gate)
#
# 유도는 "공통 정규식 자동발견"이 아니라 **arch별 폐합 규칙**이다(§1 r1 확정표). template_id 는
# GGUF arch 정확값이다(내부 adapter 명과 구분 — gpt-oss 는 `openai_moe` 가 아니라 `gpt-oss`).
#   gpt-oss   : 전 층(0..block_count-1) gate/up/down × weight+**bias** · separate 고정
#   qwen35moe : 실행 MoE 층 separate 3 weight · routed bias 없음 ·
#               `{arch}.nextn_predict_layers>0` 이면 기본 scope=execution(마지막 N층 정확 제외)
#   deepseek2 : `leading_dense_block_count..block_count-1` separate 3 weight · bias 없음 ·
#               mixed-quant 는 텐서별 동결 QUANT_TRAITS 검사로만
# 규칙 개정 = version 증가(= derived_from 문자열 변경 = 소비자 재승인 대상).
#
# ★M5 원자 활성화 토큰(OPEN_ARCH_DESIGN.md v0.2 §4 · OPENARCH_B_SPEC_DRAFT v0.2 §2-5 D2).
# 리패커·엔진·런처가 **같은 문자열**을 지녀야 번들이 조립된다(셋 중 하나라도 부재/불일치
# = 조립 실패 — 버전 무관·조립기 상시 관문). 다른 두 축의 실물: 엔진 `ggml-moe-direct.cpp` OPEN_ARCH_TEMPLATE_ABI_STR ·
# 런처 `Start-MoeDirect.ps1` $script:OPEN_ARCH_TEMPLATE_ABI.
# ★이 상수는 **선언뿐**이다 — 이 파일의 어떤 동작도 참조하지 않는다(대조 주체는 M5 조립기
# make_bundle.ps1). ARCH_TEMPLATES 행에는 넣지 않는다(B스펙 r2 소비자 분리: catalog 행은
# Gate G 전용 · ABI 필드는 M5 조립기 전용).
OPEN_ARCH_TEMPLATE_ABI = 'open-arch-template/1'

ARCH_TEMPLATES = {
    'gpt-oss':   {'version': '1', 'weight_kinds': ('gate', 'up', 'down'),
                  'bias': 'required',  'layer_rule': 'all',           'nextn': 'forbidden'},
    'qwen35moe': {'version': '1', 'weight_kinds': ('gate', 'up', 'down'),
                  'bias': 'forbidden', 'layer_rule': 'all',           'nextn': 'execution-default'},
    'deepseek2': {'version': '1', 'weight_kinds': ('gate', 'up', 'down'),
                  'bias': 'forbidden', 'layer_rule': 'leading-dense', 'nextn': 'forbidden'},
}

# derived expect 파일명(§1: 저장 위치=<repack-output>\derived.expect.json — ★번들 expects_dir 금지).
DERIVED_EXPECT_FILENAME = 'derived.expect.json'

# inventory_sha256 입력 포맷(동결 — B축 엔진이 live arch·층 공식으로 독립 재생성해 대조한다):
#   sha256( b'MOE-INVENTORY-V1\n' + ''.join(sorted(rows)) )
#   row = '<tensor-name>\t<type>\t<d0>,<d1>,..\t<layer>\t<scope>\n'   (utf-8, 행 문자열 사전순)
# 텐서명은 전역 유일이므로 정렬은 전순서다. 요약 수치가 아니라 **실 집합 결속**이 목적.
INVENTORY_DIGEST_HEADER = b'MOE-INVENTORY-V1\n'

# routed 도 shared-expert 도 아닌 "expert 처럼 보이는" 텐서 = 미분류(fail-close) 판정용.
# ★D6 수리(26-08-02): router 계열(`ffn_gate_inp`)도 expert-like 로 본다 — 개명·신설 명명이
# 아래 허용표 밖으로 새면 조용히 통과하지 않고 미분류로 걸려야 한다(예: ffn_gate_inp2.weight).
_EXPERT_LIKE_RE = re.compile(r'exps|shexp|expert|exp_probs|gate_inp', re.I)
# 동결 비-routed expert 관련 텐서 허용표(§0 상주 유지 대상). ★**실측 관측형만** 등재한다 —
# 실물 헤더 전수 대조(26-08-02 · bench_results/g5/gguf_map_{dsv4flash,k26,minimax_m27,
# mistral4_s1,s2,qwen122_s1,s2,qwen397_s2..s6}.json + bench/laptop_kit/qwen35b_map.json)에서
# 관측된 전체 집합은 다음 5형태다(★26-08-03 정정: gpt-oss 실물 20B[24층]·120B[36층] 헤더
# 전수에서 ffn_gate_inp.bias[n_expert] F32 관측 — 구 "gpt-oss 는 이 계열 0종" 문구 반증.
# M5 preflight fail-close 가 설계 의도대로 적발·심의 후 등재[사용자 승인 26-08-03]):
#   ffn_{gate,up,down}_shexp.weight · ffn_gate_inp.weight · ffn_gate_inp_shexp.weight ·
#   exp_probs_b.bias · ffn_gate_inp.bias
# ★D6 수리(26-08-02): 구 표는 미관측인 `ffn_gate_up_shexp.*` 와 모든 shexp `.bias` 까지 허용해
# `blk.0.ffn_gate_shexp.bias` 같은 새 명명이 known resident 로 조용히 통과했다(조용한 통과 1건
# 실재). 미관측 형태는 등재하지 않는다 — 실물에서 나오면 미분류로 중단시켜 심의 대상으로 올린다.
_NONROUTED_EXPERT_RE = re.compile(
    r'^blk\.\d+\.('
    r'ffn_(gate|up|down)_shexp\.weight'
    r'|ffn_gate_inp_shexp\.weight'
    r'|ffn_gate_inp\.weight'
    r'|ffn_gate_inp\.bias'
    r'|exp_probs_b\.bias'
    r')$')
# NextN/MTP 표식 텐서(qwen35moe 122B 실물: blk.48.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight).
_MTP_MARKER_RE = re.compile(r'^blk\.(\d+)\.nextn\.')

# 엔진 live trace 게이트와 동일 규약(1차 소스: ggml-moe-direct.cpp:3779-3787 —
# `n_expert>255 && ggml_moe_trace_enabled()` = 기동 중단). 리패커도 같은 env 로 판정한다.
TRACE_ENV_VAR = 'MOE_P1_TRACE'

# split 파일 패턴(llama.cpp gguf-split): <base>-NNNNN-of-MMMMM.gguf
SPLIT_RE = re.compile(r'^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<cnt>\d{5})\.gguf$')

RAM_HEADROOM_BYTES = 2 * (1 << 30)   # §2-6: 가용 < 피크 + 2GiB = 중단
FREE_DISK_HEADROOM = 1 << 30         # 승계: bin + 1GiB 여유


class RepackAbort(Exception):
    """검증된 전제조건 위반 등 '정상적으로 중단해야 하는' 상황(버그가 아님)."""
    pass


# ---------------------------------------------------------------------------
# GGUF 헤더 파싱 (bench/gguf_map.py 함수 이식 — r/rstr/rval_skip_strings 및
# data_start/abs_offset 재구성 로직. 원본 검증됨.)
# ---------------------------------------------------------------------------
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR_FMT = {T_U8: 'B', T_I8: 'b', T_U16: 'H', T_I16: 'h', T_U32: 'I', T_I32: 'i',
              T_F32: 'f', T_BOOL: '?', T_U64: 'Q', T_I64: 'q', T_F64: 'd'}
GGML_TYPE = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
             10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
             16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
             22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M',
             30: 'BF16', 39: 'MXFP4'}


def r(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))


def rstr(f):
    (n,) = r(f, '<Q')
    return f.read(n).decode('utf-8', errors='replace')


def rval(f, t):
    if t == T_STR:
        return rstr(f)
    if t == T_ARR:
        (et,) = r(f, '<I')
        (cnt,) = r(f, '<Q')
        if et == T_STR:
            return ['<%d strings>' % cnt, [rstr(f) for _ in range(min(cnt, 4))]][0] if cnt > 4 else [rstr(f) for _ in range(cnt)]
        fmt = SCALAR_FMT[et]
        vals = list(r(f, '<%d%s' % (cnt, fmt)))
        return vals if cnt <= 8 else '<%d x %s>' % (cnt, fmt)
    return r(f, '<' + SCALAR_FMT[t])[0]


def rval_skip_strings(f, t):
    # arrays of strings can be huge (tokenizer vocab) - stream past, keep count only
    if t == T_ARR:
        (et,) = r(f, '<I')
        (cnt,) = r(f, '<Q')
        if et == T_STR:
            for _ in range(cnt):
                (n,) = r(f, '<Q')
                f.seek(n, 1)
            return '<%d strings>' % cnt
        fmt = SCALAR_FMT[et]
        sz = struct.calcsize(fmt)
        if cnt > 64:
            f.seek(cnt * sz, 1)
            return '<%d x %s>' % (cnt, fmt)
        return list(r(f, '<%d%s' % (cnt, fmt)))
    return rval(f, t)


def parse_gguf_header(path):
    """단일 GGUF shard 헤더만 읽는다(쓰기 없음). 각 텐서에 rel_offset·abs_offset·gap 을 부여.
    gap = 이 shard 의 rel_offset 오름차순에서 '다음 텐서까지의 간격'(마지막 텐서는 데이터
    영역 끝까지). v2 는 텐서 bytes 를 gap 이 아니라 type·dims 이론값으로 정하고, gap 은
    §2-3 offset-gap 검사(소스 정렬 패딩만 허용)에만 쓴다."""
    fsize = os.path.getsize(path)
    meta = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise RepackAbort('GGUF magic mismatch: %r (%s)' % (magic, path))
        (ver,) = r(f, '<I')
        (n_tensors,) = r(f, '<Q')
        (n_kv,) = r(f, '<Q')
        for _ in range(n_kv):
            k = rstr(f)
            (t,) = r(f, '<I')
            meta[k] = rval_skip_strings(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = rstr(f)
            (nd,) = r(f, '<I')
            dims = list(r(f, '<%dQ' % nd))
            (ttype,) = r(f, '<I')
            (off,) = r(f, '<Q')
            tensors.append({'name': name, 'dims': dims,
                             'type': GGML_TYPE.get(ttype, 'type_%d' % ttype), 'rel_offset': off})
        header_end = f.tell()
    align = int(meta.get('general.alignment', 32))
    # ★upstream 정합(26-07-29 실결함 수리 — 397B smoke 에서 seal fail-close 발화).
    # 1차 소스: D:\moe-tools\llama.cpp-src-b10057\ggml\src\gguf.cpp:756
    #   `if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment)))`
    # → 텐서 0개 shard 는 데이터 정렬 seek 자체를 건너뛰므로 ctx->offset(= gguf_get_data_offset
    #   이 소비자에게 주는 live 값)이 raw header_end 그대로다. 무조건 패딩하면 메타 전용
    #   shard 에서만 어긋난다(397B shard1: raw 10,943,537 vs 패딩 10,943,552 = 15B 차 →
    #   엔진 seal 거부 "manifest v2 validation failed: sources[i].data_start != live"
    #   — 26-07-30 배포 표면 영어화 후 문구, 1차 소스 ggml-moe-direct.cpp:987·1093).
    #   텐서 보유 shard 는 양측 동일.
    data_start = header_end if n_tensors == 0 else (header_end + align - 1) // align * align
    by_off = sorted(tensors, key=lambda t: t['rel_offset'])
    data_region = fsize - data_start
    for i, t in enumerate(by_off):
        nxt = by_off[i + 1]['rel_offset'] if i + 1 < len(by_off) else data_region
        t['gap'] = nxt - t['rel_offset']
        t['abs_offset'] = data_start + t['rel_offset']
    return {'path': path, 'file_bytes': fsize, 'gguf_version': ver, 'alignment': align,
            'data_start': data_start, 'data_region': data_region, 'meta': meta, 'tensors': by_off}


# ---------------------------------------------------------------------------
# §2-1 멀티-shard 로더
# ---------------------------------------------------------------------------
def discover_shard_paths(model_path):
    """split 패턴(-NNNNN-of-MMMMM.gguf) 감지 → 형제 전수 발견. 비매치=단일 shard 1개.
    형제 개수 != MMMMM = 중단, 형제 파일 부재 = 중단."""
    base = os.path.basename(model_path)
    m = SPLIT_RE.match(base)
    if not m:
        return [model_path]
    d = os.path.dirname(model_path)
    prefix, cnt = m.group('base'), int(m.group('cnt'))
    if cnt < 1:
        raise RepackAbort('split count 0: %s' % base)
    paths = []
    for i in range(1, cnt + 1):
        sib = os.path.join(d, '%s-%05d-of-%05d.gguf' % (prefix, i, cnt))
        if not os.path.exists(sib):
            raise RepackAbort('split sibling shard missing: %s (%d expected in total)' % (sib, cnt))
        paths.append(sib)
    if len(paths) != cnt:
        raise RepackAbort('split sibling count mismatch: found=%d MMMMM=%d' % (len(paths), cnt))
    return paths


def load_model_shards(model_path):
    """멀티-shard 입력을 로드한다: 형제 전수 발견 → shard별 독립 헤더 파싱 → source_index 부여
    → 전 shard 텐서명 전역 유일 assert → arch 메타 충돌 assert → split KV(있으면) 대조."""
    paths = discover_shard_paths(model_path)
    shards = []
    for idx, p in enumerate(paths):
        h = parse_gguf_header(p)
        h['source_index'] = idx
        shards.append(h)

    # 전역 텐서명 유일성
    seen = {}
    for h in shards:
        for t in h['tensors']:
            if t['name'] in seen:
                raise RepackAbort('duplicate tensor name across shards: %s (shard %d, %d)' % (t['name'], seen[t['name']], h['source_index']))
            seen[t['name']] = h['source_index']

    # arch 메타 shard 간 충돌 assert (present 한 shard 들만 비교)
    archs = [(h['source_index'], h['meta']['general.architecture'])
             for h in shards if 'general.architecture' in h['meta']]
    if not archs:
        raise RepackAbort('general.architecture metadata missing (absent from every shard)')
    arch0 = archs[0][1]
    for si, a in archs:
        if a != arch0:
            raise RepackAbort('arch metadata conflict between shards: shard%d=%r vs shard%d=%r'
                               % (archs[0][0], arch0, si, a))

    # split KV 대조(존재할 때만 — 미검증: Mistral 실물 split KV. ① --plan 게이트에서 확인)
    # split.tensors.count 는 b10057 규약상 shard별 수가 아니라 모델 전체 텐서 수를 모든 shard 에 기록.
    total_tensors = sum(len(h['tensors']) for h in shards)
    split_notes = []
    for h in shards:
        mk = h['meta']
        if 'split.count' in mk:
            sc = int(mk['split.count'])
            if sc != len(shards):
                raise RepackAbort('split.count(%d) != number of shards found(%d) [shard %d]'
                                   % (sc, len(shards), h['source_index']))
        if 'split.no' in mk:
            sn = int(mk['split.no'])
            if sn != h['source_index']:
                raise RepackAbort('split.no(%d) != source_index(%d)' % (sn, h['source_index']))
        if 'split.tensors.count' in mk:
            stc = int(mk['split.tensors.count'])
            if stc != total_tensors:
                raise RepackAbort('split.tensors.count(%d) != total tensor count across shards(%d) [shard %d]'
                                   % (stc, total_tensors, h['source_index']))
        split_notes.append({'source_index': h['source_index'],
                             'split_kv_present': 'split.count' in mk})

    # 병합 메타(first-wins). arch 는 위에서 충돌 검증 완료.
    merged_meta = {}
    for h in shards:
        for k, v in h['meta'].items():
            merged_meta.setdefault(k, v)
    return {'shards': shards, 'arch': arch0, 'meta': merged_meta,
            'is_split': len(shards) > 1, 'split_notes': split_notes,
            'model_path': model_path}


# ---------------------------------------------------------------------------
# §2-3 type-trait — 이론 bytes / per-expert slice
# ---------------------------------------------------------------------------
def _prod(seq):
    p = 1
    for x in seq:
        p *= x
    return p


def _ceil_to(value, align):
    return ((value + align - 1) // align) * align


def per_expert_slice_bytes(ttype, dims):
    """§2-3: per-expert slice = ne0/bv × bb × ∏ne[1..last-1]. 표 밖 타입·ne0%bv!=0 = 중단.
    dims 는 ggml ne 순서(dims[0]=ne0, dims[-1]=expert 축)."""
    if ttype not in QUANT_TRAITS:
        raise RepackAbort('type not in the type-trait table (fail-closed): %s dims=%r - extending the table needs a new review' % (ttype, dims))
    bv, bb = QUANT_TRAITS[ttype]
    if len(dims) < 2:
        raise RepackAbort('routed tensor dims has too few axes (expert axis required): %s dims=%r' % (ttype, dims))
    ne0 = dims[0]
    if ne0 % bv != 0:
        raise RepackAbort('ne0(%d) %% block_values(%d) != 0: type=%s dims=%r' % (ne0, bv, ttype, dims))
    middle = _prod(dims[1:-1])   # ne[1..last-1]
    return (ne0 // bv) * bb * middle


def theory_tensor_bytes(ttype, dims):
    """텐서 이론 온디스크 bytes = per-expert slice × n_expert(=dims[-1])."""
    return per_expert_slice_bytes(ttype, dims) * dims[-1]


# ---------------------------------------------------------------------------
# §2-2 동적 routed 스키마 결정 + §2-4 층별 record layout
# ---------------------------------------------------------------------------
def _kind_suffix(name):
    m = TENSOR_NAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), '%s.%s' % (m.group(2), m.group(3))


def make_record_order(part_names):
    """part_names(집합, 'kind.suffix') → 레코드 내부 순서 리스트(weights 우선순위순 → bias 동순)."""
    weights = sorted([p for p in part_names if p.endswith('.weight')],
                     key=lambda p: KIND_PRIORITY[p.rsplit('.', 1)[0]])
    biases = sorted([p for p in part_names if p.endswith('.bias')],
                    key=lambda p: KIND_PRIORITY[p.rsplit('.', 1)[0]])
    return weights + biases


def _read_nextn_predict_layers(model, arch):
    """부록A: {arch}.nextn_predict_layers KV 를 shard 별로 읽어 값 충돌을 검사한다(정확 키 매칭).
    반환: KV 존재 시 정수 값, 전 shard 부재 시 None. 값 충돌(멀티-shard)=RepackAbort."""
    key = '%s.nextn_predict_layers' % arch
    vals = {h['source_index']: int(h['meta'][key]) for h in model['shards'] if key in h['meta']}
    if not vals:
        return None
    distinct = set(vals.values())
    if len(distinct) > 1:
        raise RepackAbort('%s value conflict between shards: %r' % (key, vals))
    return next(iter(distinct))


def build_layout(model, scope='all'):
    """설계 §2-2·§2-4·부록A: 병합 shard 에서 routed 스키마·층별 layout 을 재도출한다.

    arch 정확 키 매칭(last-suffix 금지)으로 n_expert/n_layer/n_expert_used 획득 →
    §0 정규식으로 routed 후보 수집 → moe_layers(0시작·연속 가정 없이 매치 존재 층) →
    scope='execution' 이면 {arch}.nextn_predict_layers 기준 마지막 N블록 routed 제외(층별
    파트 집합 동일성 검사보다 먼저) → 층별 파트 집합 동일성 assert →
    전 routed expert_axis==마지막 && ne[last]==n_expert 강제 → type-trait 이론 bytes·offset-gap
    검사(소스 정렬 패딩만 허용) → per-expert part_bytes·payload[l]."""
    arch = model['arch']
    meta = model['meta']

    def _need(key):
        if key not in meta:
            raise RepackAbort('exact metadata key missing (fail-closed): %s' % key)
        return int(meta[key])

    n_expert = _need('%s.expert_count' % arch)
    n_layer = _need('%s.block_count' % arch)
    n_expert_used = _need('%s.expert_used_count' % arch)
    if n_expert >= 0xFFFF:
        raise RepackAbort('n_expert(%d) >= 0xFFFF - violates the uint16 NA sentinel headroom (design section 4-4 limit)' % n_expert)

    # routed 후보 수집(전 shard) — source_index 부착
    by_layer = {}
    for h in model['shards']:
        for t in h['tensors']:
            ks = _kind_suffix(t['name'])
            if ks is None:
                continue
            layer, kind = ks
            tt = dict(t)
            tt['source_index'] = h['source_index']
            tt['source_alignment'] = h['alignment']
            by_layer.setdefault(layer, {})
            if kind in by_layer[layer]:
                raise RepackAbort('layer %d part %s is duplicated (across all shards)' % (layer, kind))
            by_layer[layer][kind] = tt

    if not by_layer:
        raise RepackAbort('no routed tensor found (regex %s matched nothing)' % TENSOR_NAME_RE.pattern)

    moe_layers = sorted(by_layer)                 # 0시작·연속 가정 없음
    if max(moe_layers) >= n_layer:
        raise RepackAbort('moe layer index(%d) exceeds block_count(%d)' % (max(moe_layers), n_layer))

    # 부록A: scope='execution' — 마지막 N블록(layer >= block_count-N) routed 제외. 층별 파트
    # 집합 동일성 검사보다 먼저 수행(제외층의 schema 가 달라도 무관하도록).
    if scope == 'execution':
        n_layer_nextn = _read_nextn_predict_layers(model, arch)
        if not n_layer_nextn:
            raise RepackAbort('--scope execution was requested but %s.nextn_predict_layers is absent or 0 - the request is meaningless (fail-closed)' % arch)
        if n_layer_nextn >= n_layer:
            raise RepackAbort('nextn_predict_layers(%d) >= block_count(%d) - out of range' % (n_layer_nextn, n_layer))
        cutoff = n_layer - n_layer_nextn
        for l in [l for l in moe_layers if l >= cutoff]:
            del by_layer[l]
        moe_layers = sorted(by_layer)
        if not moe_layers:
            raise RepackAbort('no routed layer remains after the execution-scope exclusion - the repack would be meaningless (fail-closed)')

    # 층별 파트 집합 동일성(fused {gate_up,down} vs separate {gate,up,down}·bias 유무 혼재=중단)
    ref_parts = set(by_layer[moe_layers[0]].keys())
    for l in moe_layers:
        s = set(by_layer[l].keys())
        if s != ref_parts:
            raise RepackAbort('per-layer part set mismatch (conservative contract): layer%d=%s vs layer%d=%s'
                               % (moe_layers[0], sorted(ref_parts), l, sorted(s)))

    # 스키마 검증(weight kinds 는 separate/fused 둘 중 하나·bias 는 weight 집합과 동일 또는 없음)
    weight_kinds = {p.rsplit('.', 1)[0] for p in ref_parts if p.endswith('.weight')}
    bias_kinds = {p.rsplit('.', 1)[0] for p in ref_parts if p.endswith('.bias')}
    if weight_kinds == {'gate', 'up', 'down'}:
        schema = 'separate'
    elif weight_kinds == {'gate_up', 'down'}:
        schema = 'fused'
    else:
        raise RepackAbort('unknown weight schema (conservative contract): %s - only separate{gate,up,down} and fused{gate_up,down} are allowed'
                           % sorted(weight_kinds))
    if bias_kinds and bias_kinds != weight_kinds:
        raise RepackAbort('bias part set does not match the weight set (partial bias unsupported, conservative contract): bias=%s weight=%s'
                           % (sorted(bias_kinds), sorted(weight_kinds)))
    has_bias = bool(bias_kinds)
    record_order = make_record_order(ref_parts)

    # 층별 gap 검사용: shard 별 rel_offset 오름차순은 parse_gguf_header 가 이미 부여(t['gap']).
    # 전 routed 텐서 expert_axis·ne[last]·이론 bytes·gap 검사 + per-expert part_bytes 산출.
    layers = []          # 층별 record layout(오름차순)
    n_routed = 0
    used_types = set()
    for l in moe_layers:
        parts = []
        part_offset = 0
        for kind in record_order:
            t = by_layer[l][kind]
            dims = t['dims']
            # expert_axis == 마지막 축 && ne[last] == n_expert
            if dims[-1] != n_expert:
                raise RepackAbort('expert_axis violation (ne[last]!=n_expert): %s dims=%r n_expert=%d'
                                   % (t['name'], dims, n_expert))
            expert_axis = len(dims) - 1
            slice_bytes = per_expert_slice_bytes(t['type'], dims)   # 표 밖·ne0%bv 검사 포함
            theory = slice_bytes * n_expert
            # offset-gap 검사: 0 <= gap - 이론 <= align_up(이론, source_alignment) - 이론
            gap = t['gap']
            aligned = _ceil_to(theory, t['source_alignment'])
            if not (theory <= gap <= aligned):
                raise RepackAbort('offset-gap violation (exceeds source alignment padding; possible misattribution): %s type=%s dims=%r theory=%d gap=%d limit=%d source_align=%d'
                                   % (t['name'], t['type'], dims, theory, gap, aligned, t['source_alignment']))
            used_types.add(t['type'])
            n_routed += 1
            parts.append({'name': kind, 'source_tensor': t['name'], 'source_index': t['source_index'],
                          'type': t['type'], 'dims': list(dims), 'expert_axis': expert_axis,
                          'part_offset': part_offset, 'part_bytes': slice_bytes,
                          'abs_offset': t['abs_offset'], 'theory_bytes': theory})
            part_offset += slice_bytes
        payload = part_offset
        layers.append({'layer': l, 'payload_bytes': payload, 'parts': parts})

    return {'arch': arch, 'n_layer': n_layer, 'n_expert': n_expert, 'n_expert_used': n_expert_used,
            'moe_layers': moe_layers, 'schema': schema, 'has_bias': has_bias,
            'record_order': record_order, 'layers': layers, 'n_routed': n_routed,
            'used_types': sorted(used_types), 'scope': scope}


def compute_record_layout(layout, A):
    """§2-4: stride[l]=ceil(payload[l]/A)×A·record_base[l]=moe_layers 오름차순 prefix-sum·
    slot_stride_max=max(stride)·bin_bytes=최종 base. records[]=moe_layer×expert 열거."""
    n_expert = layout['n_expert']
    base = 0
    strides = []
    for L in layout['layers']:
        stride = _ceil_to(L['payload_bytes'], A)
        L['stride_bytes'] = stride
        L['record_base'] = base
        strides.append(stride)
        base += n_expert * stride
    bin_bytes = base
    slot_stride_max = max(strides) if strides else 0
    n_records = len(layout['layers']) * n_expert
    records = []
    for L in layout['layers']:
        for e in range(n_expert):
            records.append({'layer': L['layer'], 'expert': e,
                            'offset': L['record_base'] + e * L['stride_bytes'],
                            'payload_bytes': L['payload_bytes']})
    return {'A': A, 'bin_bytes': bin_bytes, 'slot_stride_max': slot_stride_max,
            'n_records': n_records, 'records': records}


# ---------------------------------------------------------------------------
# §2-5 reference lock — expect 카탈로그 로드·대조
# ---------------------------------------------------------------------------
class _DuplicateManifestKey(Exception):
    """object_pairs_hook 이 중첩 객체 내 중복 키를 감지했을 때의 표식(JSON 봉인 승계)."""


def _reject_duplicate_keys(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise _DuplicateManifestKey(k)
        seen.add(k)
    return dict(pairs)


class _NonStandardJSONConstant(Exception):
    """parse_constant 훅이 NaN/Infinity/-Infinity 를 만났을 때의 표식(RFC 봉인 승계)."""


def _reject_nonstandard_constant(value):
    raise _NonStandardJSONConstant(value)


def _strict_json_load_bytes(raw):
    """엄격 JSON 로드: 중복 키 거부·비표준 상수 거부. (bytes → 객체)"""
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=_reject_nonstandard_constant)


def load_expect_profile(profile_id):
    """--profile 로 지정된 id 의 expect 파일을 로드·재해시·카탈로그 대조한다(설계 §2-5·부록A).
    미등록 id·digest 불일치·scope 불일치·비엄격 JSON·profile_id 내부 불일치 = RepackAbort. 반환:
    (expect_dict, sha256_hex)."""
    if profile_id not in EXPECT_CATALOG:
        raise RepackAbort('unregistered profile id (not in the catalog, fail-closed): %r - registered: %s'
                           % (profile_id, sorted(EXPECT_CATALOG)))
    catalog_entry = EXPECT_CATALOG[profile_id]
    path = os.path.join(EXPECTS_DIR, '%s.expect.json' % profile_id)
    if not os.path.exists(path):
        raise RepackAbort('expect file missing: %s' % path)
    raw = open(path, 'rb').read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != catalog_entry['sha256']:
        raise RepackAbort('expect digest mismatch (tampered or unapproved): %s\n  measured=%s\n  approved=%s'
                           % (path, digest, catalog_entry['sha256']))
    try:
        expect = _strict_json_load_bytes(raw)
    except _DuplicateManifestKey as e:
        raise RepackAbort('expect is not strict JSON (duplicate key): %s' % e)
    except _NonStandardJSONConstant as e:
        raise RepackAbort('expect is not strict JSON (non-standard constant): %s' % e)
    except Exception as e:
        raise RepackAbort('expect JSON is corrupt: %r' % e)
    if not isinstance(expect, dict):
        raise RepackAbort('the top level of expect is not an object')
    if expect.get('profile_id') != profile_id:
        raise RepackAbort('profile_id inside expect(%r) != requested --profile(%r) [three-way identity violated]'
                           % (expect.get('profile_id'), profile_id))
    if expect.get('expect_schema_version') != EXPECT_SCHEMA_VERSION:
        raise RepackAbort('expect_schema_version(%r) != %r' % (expect.get('expect_schema_version'), EXPECT_SCHEMA_VERSION))
    expect_scope = expect.get('routed_scope', 'all')
    if expect_scope != catalog_entry['scope']:
        raise RepackAbort('expect routed_scope(%r) != scope registered in the catalog(%r) [%s]'
                           % (expect_scope, catalog_entry['scope'], profile_id))
    return expect, digest


def cross_check_expect(model, layout, plan, expect, scope='all'):
    """재도출 layout/plan 을 expect 기대치와 대조(설계 §2-5·부록A). 불일치=RepackAbort(fail-closed)."""
    problems = []
    expect_scope = expect.get('routed_scope', 'all')
    if scope != expect_scope:
        problems.append('--scope %r != expect routed_scope %r [four-way match violated]' % (scope, expect_scope))
    if layout['arch'] != expect.get('arch'):
        problems.append('arch %r != expect %r' % (layout['arch'], expect.get('arch')))
    if layout['n_layer'] != expect.get('n_layer'):
        problems.append('n_layer %d != expect %r' % (layout['n_layer'], expect.get('n_layer')))
    if layout['n_expert'] != expect.get('n_expert'):
        problems.append('n_expert %d != expect %r' % (layout['n_expert'], expect.get('n_expert')))
    if layout['n_expert_used'] != expect.get('n_expert_used'):
        problems.append('n_expert_used %d != expect %r' % (layout['n_expert_used'], expect.get('n_expert_used')))
    if layout['n_routed'] != expect.get('routed_tensors'):
        problems.append('routed_tensors %d != expect %r' % (layout['n_routed'], expect.get('routed_tensors')))
    expert_bytes_total = layout['n_expert'] * sum(L['payload_bytes'] for L in layout['layers'])
    if expert_bytes_total != expect.get('expert_bytes_total'):
        problems.append('expert_bytes_total %d != expect %r' % (expert_bytes_total, expect.get('expert_bytes_total')))
    exp_sources = expect.get('sources', [])
    if len(exp_sources) != len(model['shards']):
        problems.append('sources count %d != expect %d' % (len(model['shards']), len(exp_sources)))
    else:
        for i, (h, es) in enumerate(zip(model['shards'], exp_sources)):
            if h['file_bytes'] != es.get('file_bytes'):
                problems.append('shard%d file_bytes %d != expect %r' % (i, h['file_bytes'], es.get('file_bytes')))
            if h['data_start'] != es.get('data_start'):
                problems.append('shard%d data_start %d != expect %r' % (i, h['data_start'], es.get('data_start')))
    if problems:
        raise RepackAbort('expect values do not match (fail-closed) - aborting immediately:\n  ' + '\n  '.join(problems))
    return {'expert_bytes_total': expert_bytes_total}


# ---------------------------------------------------------------------------
# OPEN_ARCH A축 — 아키 템플릿 유도(OPEN_ARCH_DESIGN.md v0.2 §1 · experimental)
#
# 기본 CLI 동작은 완전 불변(카탈로그 전용). 이 경로는 비공개 feature gate
# `--experimental-arch-template` 로만 진입하며, 릴리스 활성화는 M5 원자 관문 뒤다.
# 등록 경로의 expect 로드·대조 로직(load_expect_profile / cross_check_expect)은 무변경 —
# 유도 결과는 "현장 생성 derived expect" 로 **같은 대조기**에 투입될 뿐이다.
#
# fail-close 전건 코드표(§1 목록)는 아래 TEMPLATE_FAIL_CODES 단일 상수가 **유일 정본**이다.
# ---------------------------------------------------------------------------

# ★fail-close 사유 코드 동결표(단일 정본 · 23종 — arch-unsupported 포함).
# 계약: ①_abort_template 이 이 표 밖 코드를 거부한다(오타·미등재 코드가 조용히 나가지 못함)
#       ②selftest 는 이 표의 **전 코드가 각각 정확한 `[template:<code>]` 문자열로 거부됨**을
#         표본으로 증명하고 커버리지(23/23)를 관문화한다 — 양성 대조는 거부 집계와 분리한다.
TEMPLATE_FAIL_CODES = (
    'arch-unsupported',          # 미지원 arch(템플릿 부재) — 재팩 비스코프 유지
    'kv-missing',                # arch 구조 KV 부재
    'kv-type',                   # KV 형 불일치(정수 아님)
    'kv-shard-conflict',         # 같은 KV 가 shard 간 값 충돌
    'kv-range',                  # block_count 등 정의역 붕괴(층집합이 성립 불가한 퇴화 입력 가드)
    'shard-naming',              # 비표준 shard naming
    'tensor-name-duplicate',     # tensor-name 중복(정상 경로에선 load_model_shards 가 선행 차단)
    'layer-set-mismatch',        # arch 공식 층집합 != 실제
    'part-missing',              # part 누락
    'part-extra',                # part 추가
    'part-schema',               # fused-separate 불일치(혼재 포함)
    'part-bias',                 # bias 유무/부분 bias 위반
    'expert-like-unclassified',  # 미분류 expert-like tensor
    'expert-axis-not-last',      # expert axis 비말단
    'dims-last-not-n-expert',    # dims[-1] != n_expert
    'quant-off-table',           # 표 밖 quant
    'arithmetic-closure',        # 산술 불폐합(블록 정합·독립 2산식 불일치)
    'nextn-range',               # NextN 범위 오류(음수·>=block_count·요청 불가 scope)
    'leading-dense-range',       # leading-dense 범위 오류
    'mtp-ambiguous',             # 모호 MTP(표식 텐서 ↔ KV 불일치)
    'n-expert-range',            # n_expert <= 0 or >= 0xFFFF
    'n-expert-used-range',       # n_expert_used <= 0 or > n_expert or >= 0xFFFF
    'trace-gate',                # live trace 활성 ∧ n_expert > 255
)


def _abort_template(code, msg):
    if code not in TEMPLATE_FAIL_CODES:
        raise RepackAbort('internal: template fail-close code %r is not in the frozen '
                          'TEMPLATE_FAIL_CODES table - %s' % (code, msg))
    raise RepackAbort('[template:%s] %s' % (code, msg))


def arch_template_for(arch):
    """arch → 동결 템플릿. 미등재 arch = 명시 거부(미지원 arch 재팩은 비스코프)."""
    tpl = ARCH_TEMPLATES.get(arch)
    if tpl is None:
        _abort_template('arch-unsupported',
                        'architecture %r has no frozen arch template - repacking an unsupported '
                        'architecture is out of scope (fail-closed). supported: %s'
                        % (arch, sorted(ARCH_TEMPLATES)))
    return tpl


def _trace_gate_active():
    """엔진 live trace 활성 여부(env). ★엔진과 **정확히 동일** 판정 — 값이 정확히 '1' 일 때만
    활성이다(D1 수리 26-08-02). 1차 소스 직접 확인:
      D:\\moe-tools\\llama.cpp-k26-s1s4\\ggml\\src\\ggml-moe-trace.cpp:265-268
        `bool local_env_is_1(const char*n){const char*v=std::getenv(n); return v && std::strcmp(v,"1")==0;}`
        (:668 `g_enabled_cached.store(local_env_is_1("MOE_P1_TRACE"))` → ggml_moe_trace_enabled)
      D:\\moe-tools\\llama.cpp-k26-s1s4\\ggml\\src\\ggml-moe-direct.cpp:3783
        `if (model.n_expert > 255 && ggml_moe_trace_enabled()) { ... return -1; }`
    구 구현은 비어있지 않은 '0' 외 전부 활성으로 봐서 예컨대 MOE_P1_TRACE=2 에서 리패커는
    거부·엔진은 통과로 판정이 갈렸다(공백 trim 도 엔진엔 없다)."""
    return os.environ.get(TRACE_ENV_VAR) == '1'


def _template_kv(model, key, required):
    """구조 KV 를 shard 별로 읽어 부재/형/충돌을 각각 별도 사유로 거부한다(정확 키 매칭).
    반환: 정수 값 또는 (required=False 이고 전 shard 부재면) None."""
    vals = {}
    for h in model['shards']:
        if key in h['meta']:
            vals[h['source_index']] = h['meta'][key]
    if not vals:
        if required:
            _abort_template('kv-missing', 'required structural metadata key is absent from every shard: %s' % key)
        return None
    for si, v in sorted(vals.items()):
        if isinstance(v, bool) or not isinstance(v, int):
            _abort_template('kv-type', 'structural metadata %s on shard %d is %s(%r) - an integer is required'
                            % (key, si, type(v).__name__, v))
    distinct = sorted(set(vals.values()))
    if len(distinct) > 1:
        _abort_template('kv-shard-conflict', 'structural metadata %s conflicts between shards: %r' % (key, vals))
    return distinct[0]


def _check_shard_naming(model):
    """비표준 shard naming 거부: 다중 shard 는 llama.cpp gguf-split 규약
    (<base>-NNNNN-of-MMMMM.gguf·인덱스 1시작·MMMMM=실제 개수)만 허용, 단일은 .gguf 만."""
    n = len(model['shards'])
    for h in model['shards']:
        base = os.path.basename(h['path'])
        m = SPLIT_RE.match(base)
        if n == 1:
            if not base.lower().endswith('.gguf'):
                _abort_template('shard-naming', 'non-standard shard file name (a .gguf extension is required): %s' % base)
            if m is not None and int(m.group('cnt')) != 1:
                _abort_template('shard-naming', 'split naming declares %s parts but only one shard was discovered: %s'
                                % (m.group('cnt'), base))
        else:
            if m is None:
                _abort_template('shard-naming', 'non-standard split shard file name (expected <base>-NNNNN-of-MMMMM.gguf): %s' % base)
            if int(m.group('idx')) != h['source_index'] + 1 or int(m.group('cnt')) != n:
                _abort_template('shard-naming', 'split shard index/count mismatch: %s (source_index=%d, shards=%d)'
                                % (base, h['source_index'], n))


def _inventory_digest(rows):
    """정렬 inventory digest(동결 포맷 — INVENTORY_DIGEST_HEADER 주석 참조)."""
    payload = INVENTORY_DIGEST_HEADER + ''.join(sorted(rows)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def derive_arch_template(model, requested_scope=None):
    """§1: arch별 폐합 규칙으로 routed 인벤토리를 **독립 유도**한다(정규식 자동발견 아님).
    전건 fail-close 는 위 코드표 그대로 각각 명시 사유로 거부한다.
    requested_scope=None 이면 템플릿 기본 scope(qwen35moe·nextn>0 = execution)를 쓴다."""
    arch = model['arch']
    tpl = arch_template_for(arch)

    # (1) 비표준 shard naming
    _check_shard_naming(model)

    # (2) tensor-name 전역 유일(정상 경로에선 load_model_shards 가 선행 차단 — 방어 이중화)
    owner = {}
    for h in model['shards']:
        for t in h['tensors']:
            if t['name'] in owner:
                _abort_template('tensor-name-duplicate', 'tensor name %r appears more than once (shard %d and %d)'
                                % (t['name'], owner[t['name']], h['source_index']))
            owner[t['name']] = h['source_index']

    # (3) 구조 KV(부재/형/충돌 각각 별도 사유)
    n_layer = _template_kv(model, '%s.block_count' % arch, required=True)
    n_expert = _template_kv(model, '%s.expert_count' % arch, required=True)
    n_expert_used = _template_kv(model, '%s.expert_used_count' % arch, required=True)
    nextn = _template_kv(model, '%s.nextn_predict_layers' % arch, required=False)
    leading_dense = None
    if tpl['layer_rule'] == 'leading-dense':
        leading_dense = _template_kv(model, '%s.leading_dense_block_count' % arch, required=True)

    # (4) 정의역
    if n_layer <= 0:
        _abort_template('kv-range', '%s.block_count(%d) must be positive' % (arch, n_layer))
    if n_expert <= 0 or n_expert >= 0xFFFF:
        _abort_template('n-expert-range', '%s.expert_count(%d) is out of range (0 < n_expert < 0xFFFF)' % (arch, n_expert))
    if n_expert_used <= 0 or n_expert_used > n_expert or n_expert_used >= 0xFFFF:
        _abort_template('n-expert-used-range', '%s.expert_used_count(%d) is out of range (0 < n_expert_used <= n_expert(%d) < 0xFFFF)'
                        % (arch, n_expert_used, n_expert))

    # (5) live trace 게이트(엔진 규약 동형 — uint8 expert wire 상한)
    if n_expert > 255 and _trace_gate_active():
        _abort_template('trace-gate', 'live trace is enabled (%s is set) and n_expert(%d) > 255 - the uint8 expert wire '
                                      'cannot represent it (fail-closed, same gate as the engine)' % (TRACE_ENV_VAR, n_expert))

    # (6) NextN/leading-dense 범위 + scope 결정
    if tpl['nextn'] == 'execution-default':
        if nextn is not None and nextn < 0:
            _abort_template('nextn-range', '%s.nextn_predict_layers(%d) is negative' % (arch, nextn))
        if nextn and nextn >= n_layer:
            _abort_template('nextn-range', '%s.nextn_predict_layers(%d) >= block_count(%d)' % (arch, nextn, n_layer))
        default_scope = 'execution' if (nextn or 0) > 0 else 'all'
    else:
        if nextn:
            _abort_template('mtp-ambiguous', 'the %s template does not model NextN but %s.nextn_predict_layers=%d is present'
                            % (arch, arch, nextn))
        default_scope = 'all'
    scope = requested_scope or default_scope
    if scope not in ('all', 'execution'):
        _abort_template('nextn-range', 'unknown routed scope %r (expected all|execution)' % (scope,))
    if scope == 'execution':
        if tpl['nextn'] != 'execution-default':
            _abort_template('nextn-range', 'the %s template does not support --scope execution (no NextN clause)' % arch)
        if not nextn:
            _abort_template('nextn-range', 'scope=execution was requested but %s.nextn_predict_layers is absent or 0' % arch)

    first_layer = 0
    if tpl['layer_rule'] == 'leading-dense':
        if leading_dense < 0 or leading_dense >= n_layer:
            _abort_template('leading-dense-range', '%s.leading_dense_block_count(%d) is out of range (0 <= x < block_count(%d))'
                            % (arch, leading_dense, n_layer))
        first_layer = leading_dense
    last_excl = n_layer - (nextn if scope == 'execution' else 0)
    expected_layers = list(range(first_layer, last_excl))
    if not expected_layers:
        _abort_template('nextn-range' if scope == 'execution' else 'layer-set-mismatch',
                        'the template layer set is empty (first=%d, end_exclusive=%d, block_count=%d, scope=%s)'
                        % (first_layer, last_excl, n_layer, scope))

    # (7) 모호 MTP — NextN 표식 텐서와 KV 선언의 정합
    mtp_layers = sorted({int(m.group(1)) for name in owner for m in [_MTP_MARKER_RE.match(name)] if m})
    if mtp_layers:
        if not nextn:
            _abort_template('mtp-ambiguous', 'NextN/MTP marker tensors exist on layers %r but %s.nextn_predict_layers '
                                             'is absent or 0' % (mtp_layers, arch))
        tail = list(range(n_layer - nextn, n_layer))
        if mtp_layers != tail:
            _abort_template('mtp-ambiguous', 'NextN/MTP marker layers %r != the tail declared by nextn_predict_layers=%d (%r)'
                            % (mtp_layers, nextn, tail))
    elif nextn:
        _abort_template('mtp-ambiguous', '%s.nextn_predict_layers=%d is declared but no NextN/MTP marker tensor '
                                         '(blk.<n>.nextn.*) exists' % (arch, nextn))

    # (8) 미분류 expert-like tensor
    for name in sorted(owner):
        if TENSOR_NAME_RE.match(name) or _NONROUTED_EXPERT_RE.match(name) or _MTP_MARKER_RE.match(name):
            continue
        if _EXPERT_LIKE_RE.search(name):
            _abort_template('expert-like-unclassified', 'tensor %r looks expert-related but is neither a routed tensor '
                                                        'nor a known resident (shared-expert/router/gating-bias) tensor' % name)

    # (9) arch 공식 층집합 == 실제
    routed_by_layer = {}
    for h in model['shards']:
        for t in h['tensors']:
            ks = _kind_suffix(t['name'])
            if ks is None:
                continue
            layer, kind = ks
            tt = dict(t)
            tt['source_index'] = h['source_index']
            tt['source_alignment'] = h['alignment']
            routed_by_layer.setdefault(layer, {})[kind] = tt
    actual_layers = set(routed_by_layer)
    expected_set = set(expected_layers)
    allowed_extra = set(range(last_excl, n_layer)) if scope == 'execution' else set()
    missing_layers = sorted(expected_set - actual_layers)
    unexpected_layers = sorted(actual_layers - expected_set - allowed_extra)
    if missing_layers or unexpected_layers:
        _abort_template('layer-set-mismatch', 'the routed layer set does not match the %s template formula '
                                              '(expected %d..%d, scope=%s): missing=%r unexpected=%r'
                        % (arch, expected_layers[0], expected_layers[-1], scope, missing_layers, unexpected_layers))

    # (10) 층별 part 집합(누락/추가/fused-separate/부분 bias 각각 별도 사유)
    expected_w = {'%s.weight' % k for k in tpl['weight_kinds']}
    expected_b = {'%s.bias' % k for k in tpl['weight_kinds']} if tpl['bias'] == 'required' else set()
    for l in expected_layers:
        have = set(routed_by_layer[l])
        have_w = {p for p in have if p.endswith('.weight')}
        have_b = {p for p in have if p.endswith('.bias')}
        if have_w != expected_w:
            lack, extra = expected_w - have_w, have_w - expected_w
            if extra and not lack:
                _abort_template('part-extra', 'layer %d has unexpected routed weight parts %s (template expects exactly %s)'
                                % (l, sorted(extra), sorted(expected_w)))
            if lack and not extra:
                _abort_template('part-missing', 'layer %d is missing routed weight parts %s (template expects exactly %s)'
                                % (l, sorted(lack), sorted(expected_w)))
            _abort_template('part-schema', 'layer %d routed weight schema mismatch (fused/separate): have=%s template=%s'
                            % (l, sorted(have_w), sorted(expected_w)))
        if have_b != expected_b:
            if expected_b:
                _abort_template('part-bias', 'layer %d routed bias set %s != template requirement %s (partial or absent bias)'
                                % (l, sorted(have_b), sorted(expected_b)))
            _abort_template('part-bias', 'layer %d has routed bias parts %s but the %s template forbids routed bias'
                            % (l, sorted(have_b), arch))

    # (11) 텐서별 축·타입·산술 + 인벤토리 행
    record_order = make_record_order(expected_w | expected_b)
    rows = []
    inventory = []
    expert_bytes_total = 0
    for l in expected_layers:
        for part in record_order:
            t = routed_by_layer[l][part]
            dims = list(t['dims'])
            if len(dims) < 2:
                _abort_template('expert-axis-not-last', '%s has too few axes for an expert axis: dims=%r' % (t['name'], dims))
            if dims[-1] != n_expert:
                if n_expert in dims[:-1]:
                    _abort_template('expert-axis-not-last', '%s carries the expert axis at position %d, not last: dims=%r n_expert=%d'
                                    % (t['name'], dims.index(n_expert), dims, n_expert))
                _abort_template('dims-last-not-n-expert', '%s dims[-1](%d) != n_expert(%d): dims=%r'
                                % (t['name'], dims[-1], n_expert, dims))
            if t['type'] not in QUANT_TRAITS:
                _abort_template('quant-off-table', '%s type %s is not in the frozen type-trait table (per-tensor check)'
                                % (t['name'], t['type']))
            bv = QUANT_TRAITS[t['type']][0]
            if dims[0] % bv != 0:
                _abort_template('arithmetic-closure', '%s ne0(%d) %% block_values(%d) != 0 for type %s - the per-expert '
                                                      'slice does not close' % (t['name'], dims[0], bv, t['type']))
            slice_bytes = per_expert_slice_bytes(t['type'], dims)
            expert_bytes_total += slice_bytes * n_expert
            rows.append('%s\t%s\t%s\t%d\t%s\n'
                        % (t['name'], t['type'], ','.join(str(d) for d in dims), l, scope))
            inventory.append({'name': t['name'], 'type': t['type'], 'dims': dims, 'layer': l, 'scope': scope})

    return {
        'template_id': arch, 'template_version': tpl['version'],
        'derived_from': '%s@%s' % (arch, tpl['version']),
        'scope': scope, 'default_scope': default_scope,
        'arch': arch, 'n_layer': n_layer, 'n_expert': n_expert, 'n_expert_used': n_expert_used,
        'nextn_predict_layers': nextn, 'leading_dense_block_count': leading_dense,
        'layers': expected_layers, 'record_order': record_order,
        'routed_tensors': len(rows), 'expert_bytes_total': expert_bytes_total,
        'inventory': inventory, 'inventory_sha256': _inventory_digest(rows),
    }


def build_derived_expect(model, layout, derivation):
    """유도 결과 → derived expect(등록 expect 와 **의미 동일 스키마** + derived_from·inventory_sha256).
    ★독립 2산식 폐합: 템플릿 유도(공식)와 build_layout(정규식) 결과가 어긋나면 산술 불폐합으로 거부.
    반환: (expect dict, 파일에 쓸 바이트, 그 바이트의 sha256)."""
    if list(layout['moe_layers']) != list(derivation['layers']):
        _abort_template('arithmetic-closure', 'template layer set %r != regex-derived moe_layers %r'
                        % (derivation['layers'], list(layout['moe_layers'])))
    if list(layout['record_order']) != list(derivation['record_order']):
        _abort_template('arithmetic-closure', 'template record order %r != regex-derived %r'
                        % (derivation['record_order'], layout['record_order']))
    for field, tval, lval in (('n_layer', derivation['n_layer'], layout['n_layer']),
                              ('n_expert', derivation['n_expert'], layout['n_expert']),
                              ('n_expert_used', derivation['n_expert_used'], layout['n_expert_used']),
                              ('routed_tensors', derivation['routed_tensors'], layout['n_routed']),
                              ('scope', derivation['scope'], layout.get('scope', 'all'))):
        if tval != lval:
            _abort_template('arithmetic-closure', 'template %s(%r) != regex-derived(%r)' % (field, tval, lval))
    layout_total = layout['n_expert'] * sum(L['payload_bytes'] for L in layout['layers'])
    if layout_total != derivation['expert_bytes_total']:
        _abort_template('arithmetic-closure', 'template expert_bytes_total(%d) != regex-derived(%d)'
                        % (derivation['expert_bytes_total'], layout_total))

    expect = {
        'expect_schema_version': EXPECT_SCHEMA_VERSION,
        'derived_from': derivation['derived_from'],
        'template_id': derivation['template_id'],
        'template_version': derivation['template_version'],
        'inventory_sha256': derivation['inventory_sha256'],
        'routed_scope': derivation['scope'],
        'arch': derivation['arch'],
        'n_layer': derivation['n_layer'],
        'n_expert': derivation['n_expert'],
        'n_expert_used': derivation['n_expert_used'],
        'routed_tensors': derivation['routed_tensors'],
        'expert_bytes_total': derivation['expert_bytes_total'],
        'sources': [{'file_bytes': h['file_bytes'], 'data_start': h['data_start']} for h in model['shards']],
    }
    raw = (json.dumps(expect, ensure_ascii=False, indent=1, allow_nan=False) + '\n').encode('utf-8')
    return expect, raw, hashlib.sha256(raw).hexdigest()


def arch_template_lock_id(derivation):
    """template 모드의 reference_lock.profile_id(카탈로그 id 와 형태가 겹치지 않는 접두 사용)."""
    return 'arch-template:%s' % derivation['derived_from']


def write_derived_expect(out_dir, raw):
    """derived expect 를 <repack-output>\\derived.expect.json 에 원자 기록(번들 expects_dir 금지)."""
    path = os.path.join(out_dir, DERIVED_EXPECT_FILENAME)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# §4 섹터 질의 (승계) — 1순위 IOCTL, 2순위 GetDiskFreeSpaceW
# ---------------------------------------------------------------------------
FILE_DEVICE_MASS_STORAGE = 0x0000002D
METHOD_BUFFERED = 0
FILE_ANY_ACCESS = 0


def _ctl_code(device_type, function, method, access):
    return (device_type << 16) | (access << 14) | (function << 2) | method


IOCTL_STORAGE_QUERY_PROPERTY = _ctl_code(FILE_DEVICE_MASS_STORAGE, 0x0500, METHOD_BUFFERED, FILE_ANY_ACCESS)
STORAGE_ACCESS_ALIGNMENT_PROPERTY_ID = 6
PROPERTY_STANDARD_QUERY = 0
GENERIC_READ_UNUSED = 0
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _STORAGE_PROPERTY_QUERY(ctypes.Structure):
    _fields_ = [
        ('PropertyId', wintypes.DWORD),
        ('QueryType', wintypes.DWORD),
        ('AdditionalParameters', ctypes.c_byte * 1),
    ]


class _STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ('Version', wintypes.DWORD),
        ('Size', wintypes.DWORD),
        ('BytesPerCacheLine', wintypes.DWORD),
        ('BytesOffsetForCacheAlignment', wintypes.DWORD),
        ('BytesPerLogicalSector', wintypes.DWORD),
        ('BytesPerPhysicalSector', wintypes.DWORD),
        ('BytesOffsetForSectorAlignment', wintypes.DWORD),
    ]


def _validate_alignment_descriptor(bytes_returned, version, size, logical, physical):
    struct_size = ctypes.sizeof(_STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR)
    if bytes_returned < struct_size:
        raise OSError('IOCTL bytes_returned too small: %d (expected >= %d)' % (bytes_returned, struct_size))
    if version != struct_size:
        raise OSError('descriptor Version(%d) != struct size(%d)' % (version, struct_size))
    if size < struct_size:
        raise OSError('descriptor Size(%d) < struct size(%d)' % (size, struct_size))
    if logical <= 0 or physical <= 0:
        raise OSError('IOCTL returned a non-positive value: logical=%d physical=%d' % (logical, physical))
    if (logical & (logical - 1)) != 0 or (physical & (physical - 1)) != 0:
        raise OSError('IOCTL returned a value that is not a power of two: logical=%d physical=%d' % (logical, physical))
    if physical % logical != 0:
        raise OSError('physical(%d) is not a multiple of logical(%d)' % (physical, logical))


def _query_via_ioctl(drive_root):
    volume_path = r'\\.\%s' % drive_root.rstrip('\\')
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    CreateFileW = kernel32.CreateFileW
    CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                             wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    CreateFileW.restype = wintypes.HANDLE

    handle = CreateFileW(volume_path, GENERIC_READ_UNUSED, FILE_SHARE_READ | FILE_SHARE_WRITE,
                          None, OPEN_EXISTING, 0, None)
    if handle in (None, 0) or handle == INVALID_HANDLE_VALUE:
        err = ctypes.get_last_error()
        raise OSError('CreateFileW(%s) failed, err=%d' % (volume_path, err))
    try:
        query = _STORAGE_PROPERTY_QUERY(PropertyId=STORAGE_ACCESS_ALIGNMENT_PROPERTY_ID,
                                         QueryType=PROPERTY_STANDARD_QUERY)
        outbuf = _STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR()
        bytes_returned = wintypes.DWORD(0)
        DeviceIoControl = kernel32.DeviceIoControl
        DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                     wintypes.LPVOID]
        DeviceIoControl.restype = wintypes.BOOL
        ok = DeviceIoControl(handle, IOCTL_STORAGE_QUERY_PROPERTY,
                              ctypes.byref(query), ctypes.sizeof(query),
                              ctypes.byref(outbuf), ctypes.sizeof(outbuf),
                              ctypes.byref(bytes_returned), None)
        if not ok:
            err = ctypes.get_last_error()
            raise OSError('DeviceIoControl(IOCTL_STORAGE_QUERY_PROPERTY) failed, err=%d' % err)
        _validate_alignment_descriptor(bytes_returned.value, outbuf.Version, outbuf.Size,
                                        outbuf.BytesPerLogicalSector, outbuf.BytesPerPhysicalSector)
        return outbuf.BytesPerLogicalSector, outbuf.BytesPerPhysicalSector
    finally:
        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL
        CloseHandle(handle)


def _query_via_getdiskfreespace(drive_root):
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    GetDiskFreeSpaceW = kernel32.GetDiskFreeSpaceW
    GetDiskFreeSpaceW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD),
                                   ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                   ctypes.POINTER(wintypes.DWORD)]
    GetDiskFreeSpaceW.restype = wintypes.BOOL
    root = drive_root.rstrip('\\') + '\\'
    spc, bps, nfc, tnc = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
    ok = GetDiskFreeSpaceW(root, ctypes.byref(spc), ctypes.byref(bps), ctypes.byref(nfc), ctypes.byref(tnc))
    if not ok:
        err = ctypes.get_last_error()
        raise OSError('GetDiskFreeSpaceW(%s) failed, err=%d' % (root, err))
    return bps.value


def query_sector_alignment(drive_root):
    try:
        logical, physical = _query_via_ioctl(drive_root)
        return {'method': 'IOCTL_STORAGE_QUERY_PROPERTY', 'logical': int(logical),
                'physical': int(physical), 'fallback_reason': None}
    except Exception as e_ioctl:
        try:
            logical = _query_via_getdiskfreespace(drive_root)
            return {'method': 'GetDiskFreeSpaceW', 'logical': int(logical), 'physical': 4096,
                    'fallback_reason': 'primary IOCTL failed: %r' % e_ioctl}
        except Exception as e_fallback:
            return {'method': 'default_4096', 'logical': 4096, 'physical': 4096,
                    'fallback_reason': 'primary IOCTL failed: %r; secondary GetDiskFreeSpaceW also failed: %r'
                                        % (e_ioctl, e_fallback)}


def _existing_ancestor(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def _drive_root(path):
    anc = _existing_ancestor(path)
    drive, _ = os.path.splitdrive(anc)
    return drive if drive else anc


def query_sector_alignment_for_path(path):
    return query_sector_alignment(_drive_root(path))


def resolve_alignment(out_dir, allow_default_align):
    """정렬 질의 → A=max(4096, logical, physical). default_4096 강등은 CLI 에선 중단
    (allow_default_align=False), selftest 만 허용(True)."""
    align_info = query_sector_alignment_for_path(out_dir)
    if align_info['method'] == 'default_4096' and not allow_default_align:
        raise RepackAbort('both the primary and secondary sector-alignment queries failed - production cannot fall back to the 4096 default: %s'
                           % align_info['fallback_reason'])
    A = max(4096, align_info['logical'], align_info['physical'])
    return A, align_info


# ---------------------------------------------------------------------------
# §2-6 가용 RAM 질의 (GlobalMemoryStatusEx)
# ---------------------------------------------------------------------------
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', wintypes.DWORD),
        ('dwMemoryLoad', wintypes.DWORD),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]


def available_ram_bytes():
    """가용 물리 RAM(ullAvailPhys) 질의. 반환: (가용 bytes 또는 None, 오류문자열 또는 None)."""
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        GlobalMemoryStatusEx = kernel32.GlobalMemoryStatusEx
        GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
        GlobalMemoryStatusEx.restype = wintypes.BOOL
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not GlobalMemoryStatusEx(ctypes.byref(st)):
            err = ctypes.get_last_error()
            return None, 'GlobalMemoryStatusEx failed, err=%d' % err
        return int(st.ullAvailPhys), None
    except Exception as e:
        return None, 'available-RAM query raised: %r' % e


def peak_layer_expert_bytes(layout):
    """§2-6 피크 = 최대 층 expert bytes(= max_l payload[l] × n_expert). 층 단위 버퍼링 피크."""
    if not layout['layers']:
        return 0
    return layout['n_expert'] * max(L['payload_bytes'] for L in layout['layers'])


# ---------------------------------------------------------------------------
# 엄격 타입 비교 (승계)
# ---------------------------------------------------------------------------
def _typed_eq(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual.keys()) == set(expected.keys())
                and all(_typed_eq(actual[k], expected[k]) for k in expected))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_typed_eq(a, e) for a, e in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def _type_mismatch(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual.keys()) != set(expected.keys()):
            return True
        return any(_type_mismatch(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return not isinstance(actual, list) or any(
            _type_mismatch(a, e) for a, e in zip(actual, expected))
    return type(actual) is not type(expected)


# ---------------------------------------------------------------------------
# 매니페스트 v2 구성 (§3)
# ---------------------------------------------------------------------------
def build_manifest(model, layout, plan, A, align_info, profile_id, expect_sha256):
    sources = []
    for h in model['shards']:
        sources.append({'index': h['source_index'], 'path': h['path'], 'bytes': h['file_bytes'],
                        'mtime': os.path.getmtime(h['path']), 'gguf_version': h['gguf_version'],
                        'alignment': h['alignment'], 'data_start': h['data_start']})

    manifest_layers = []
    for L in layout['layers']:
        parts = [{'name': p['name'], 'source_tensor': p['source_tensor'], 'source_index': p['source_index'],
                  'type': p['type'], 'dims': list(p['dims']), 'expert_axis': p['expert_axis'],
                  'part_offset': p['part_offset'], 'part_bytes': p['part_bytes']} for p in L['parts']]
        manifest_layers.append({'layer': L['layer'], 'payload_bytes': L['payload_bytes'],
                                 'stride_bytes': L['stride_bytes'], 'record_base': L['record_base'],
                                 'parts': parts})

    source_tensors = []
    for L in layout['layers']:
        for p in L['parts']:
            source_tensors.append({'name': p['source_tensor'], 'source_index': p['source_index'],
                                    'abs_offset': p['abs_offset'], 'bytes': p['part_bytes'] * layout['n_expert'],
                                    'type': p['type'], 'dims': list(p['dims'])})

    quant_traits = {}
    for tt in layout['used_types']:
        bv, bb = QUANT_TRAITS[tt]
        quant_traits[tt] = {'block_values': bv, 'block_bytes': bb}

    expert_payload_total = layout['n_expert'] * sum(L['payload_bytes'] for L in layout['layers'])

    model_dict = {'arch': layout['arch'], 'n_layer': layout['n_layer'], 'n_expert': layout['n_expert'],
                  'n_expert_used': layout['n_expert_used'], 'moe_layers': list(layout['moe_layers'])}
    if layout.get('scope', 'all') != 'all':
        model_dict['routed_scope'] = layout['scope']   # 부록A: 부재=all(기존 4모델 산출물 불변)

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'model': model_dict,
        'sources': sources,
        'layout': {
            'align_bytes': A,
            'align_query': {'method': align_info['method'], 'logical': align_info['logical'],
                             'physical': align_info['physical'], 'fallback_reason': align_info['fallback_reason']},
            'slot_stride_max': plan['slot_stride_max'],
            'layers': manifest_layers,
        },
        'records': plan['records'],
        'source_tensors': source_tensors,
        'quant_traits': quant_traits,
        'totals': {'expert_payload_total': expert_payload_total, 'bin_file_bytes': plan['bin_bytes'],
                   'n_records': plan['n_records']},
        'reference_lock': {'profile_id': profile_id, 'expect_sha256': expect_sha256},
        'tool': {'version': SCRIPT_VERSION, 'ts': datetime.now(timezone.utc).isoformat(),
                  'cmdline': ' '.join(sys.argv)},
    }
    return manifest


# ---------------------------------------------------------------------------
# 요약 출력 (--plan / 본실행 preflight 공유)
# ---------------------------------------------------------------------------
def _print_plan_summary(model, layout, plan, A, align_info, out_dir, profile_id, expect_sha256, expect_totals,
                        derived=None):
    print('profile=%s expect_sha256=%s' % (profile_id, expect_sha256))
    if derived is not None:
        d = derived['derivation']
        print('[EXPERIMENTAL arch-template] derived_from=%s routed_scope=%s (template default=%s) inventory_sha256=%s'
              % (d['derived_from'], d['scope'], d['default_scope'], d['inventory_sha256']))
        print('[EXPERIMENTAL arch-template] template layers=%d..%d (%d) routed_tensors=%d nextn=%r leading_dense=%r'
              % (d['layers'][0], d['layers'][-1], len(d['layers']), d['routed_tensors'],
                 d['nextn_predict_layers'], d['leading_dense_block_count']))
    print('arch=%s n_layer=%d n_expert=%d n_expert_used=%d schema=%s bias=%s'
          % (layout['arch'], layout['n_layer'], layout['n_expert'], layout['n_expert_used'],
             layout['schema'], layout['has_bias']))
    print('moe_layers: %d entries [%d..%d] (starts_at_0=%s contiguous=%s)'
          % (len(layout['moe_layers']), layout['moe_layers'][0], layout['moe_layers'][-1],
             layout['moe_layers'][0] == 0,
             layout['moe_layers'] == list(range(layout['moe_layers'][0], layout['moe_layers'][-1] + 1))))
    print('shards=%d (split=%s)' % (len(model['shards']), model['is_split']))
    for h in model['shards']:
        print('  shard[%d]: %s bytes=%d gguf_v=%d align=%d data_start=%d'
              % (h['source_index'], os.path.basename(h['path']), h['file_bytes'],
                 h['gguf_version'], h['alignment'], h['data_start']))
    print('routed tensors=%d  used_types=%s' % (layout['n_routed'], layout['used_types']))
    print('expert_payload_total(=expert_bytes)=%d' % expect_totals['expert_bytes_total'])
    print('alignment query: method=%s logical=%d physical=%d fallback_reason=%s'
          % (align_info['method'], align_info['logical'], align_info['physical'], align_info['fallback_reason']))
    strides = [L['stride_bytes'] for L in layout['layers']]
    payloads = [L['payload_bytes'] for L in layout['layers']]
    uniform = len(set(strides)) == 1
    print('output alignment A=%d, stride[l] %s (min=%d max=%d), slot_stride_max=%d'
          % (A, 'uniform' if uniform else 'varies per layer', min(strides), max(strides), plan['slot_stride_max']))
    print('payload[l] min=%d max=%d, records=%d' % (min(payloads), max(payloads), plan['n_records']))
    print('experts.bin expected size=%d B (%.3f GB)' % (plan['bin_bytes'], plan['bin_bytes'] / 1e9))

    peak = peak_layer_expert_bytes(layout)
    avail, _ram_err = available_ram_bytes()
    ram_need = peak + RAM_HEADROOM_BYTES
    if avail is None:
        print('preflight RAM: peak layer expert bytes=%d (%.2f GiB) - available-RAM query failed (warning)'
              % (peak, peak / (1 << 30)))
    else:
        print('preflight RAM: peak layer expert bytes=%d (%.2f GiB), required=%.2f GiB (peak+2GiB), available=%.2f GiB - %s'
              % (peak, peak / (1 << 30), ram_need / (1 << 30), avail / (1 << 30),
                 'sufficient' if avail >= ram_need else 'insufficient (a real run would abort)'))

    du = shutil.disk_usage(_existing_ancestor(out_dir))
    required = plan['bin_bytes'] + FREE_DISK_HEADROOM
    print('free space on the output volume=%d B (%.2f GB) - required=%.2f GB (bin+1GiB) - %s'
          % (du.free, du.free / 1e9, required / 1e9, 'sufficient' if du.free >= required else 'insufficient (a real run would abort)'))

    L0 = layout['layers'][0]
    print('--- moe layer %d record parts (name, source_tensor, src, type, dims, part_offset, part_bytes) ---'
          % L0['layer'])
    for p in L0['parts']:
        print('  %-12s <- %-32s [s%d] type=%-6s dims=%-18r off=%-12d peb=%d'
              % (p['name'], p['source_tensor'], p['source_index'], p['type'], p['dims'],
                 p['part_offset'], p['part_bytes']))


# ---------------------------------------------------------------------------
# 재팩 실행 + 매니페스트(§3) + 독립 2패스 검증(§5)
# ---------------------------------------------------------------------------
def _prepare(model_path, out_dir, profile_id, allow_default_align, enforce_reference, scope='all',
             arch_template=False):
    """공통 준비: shard 로드 → layout 재도출 → (참조 락) expect 로드·대조 → A/plan 산출.
    arch_template=True(비공개 gate) 면 카탈로그 대신 §1 템플릿으로 derived expect 를 현장 유도해
    같은 대조기(cross_check_expect)에 투입한다. 반환 8번째=derived(템플릿 모드에서만 non-None)."""
    model = load_model_shards(model_path)
    derived = None
    if arch_template:
        derivation = derive_arch_template(model, requested_scope=scope)
        scope = derivation['scope']          # 템플릿 기본 scope(qwen35moe nextn>0 = execution)
    else:
        scope = scope or 'all'
    layout = build_layout(model, scope=scope)
    A, align_info = resolve_alignment(out_dir, allow_default_align)
    plan = compute_record_layout(layout, A)
    if arch_template:
        expect, expect_raw, expect_sha256 = build_derived_expect(model, layout, derivation)
        expect_totals = cross_check_expect(model, layout, plan, expect, scope=scope)
        derived = {'derivation': derivation, 'expect': expect, 'raw': expect_raw,
                   'sha256': expect_sha256, 'lock_id': arch_template_lock_id(derivation)}
    elif enforce_reference:
        expect, expect_sha256 = load_expect_profile(profile_id)
        expect_totals = cross_check_expect(model, layout, plan, expect, scope=scope)
    else:
        # selftest 면제 경로(설계 §2-5): 합성 자체 expect·카탈로그 비경유. reference_lock 은
        # 명시적 selftest 표식으로 채운다(카탈로그 대조 없음).
        expect_sha256 = 'selftest-exempt'
        expect_totals = {'expert_bytes_total': layout['n_expert'] * sum(L['payload_bytes'] for L in layout['layers'])}
    return model, layout, A, align_info, plan, expect_sha256, expect_totals, derived


def do_repack(model_path, out_dir, profile_id, force=False, run_verify=True,
              enforce_reference=True, allow_default_align=False, scope='all', arch_template=False):
    model, layout, A, align_info, plan, expect_sha256, expect_totals, derived = _prepare(
        model_path, out_dir, profile_id, allow_default_align, enforce_reference, scope=scope,
        arch_template=arch_template)
    if derived is not None:
        profile_id = derived['lock_id']      # reference_lock 은 template 표식으로(카탈로그 id 아님)

    print('=== real-run preflight: same summary as --plan ===')
    _print_plan_summary(model, layout, plan, A, align_info, out_dir,
                        profile_id if (enforce_reference or derived) else '(selftest-exempt)',
                        expect_sha256, expect_totals, derived=derived)

    bin_bytes = plan['bin_bytes']
    n_expert = layout['n_expert']
    bin_path = os.path.join(out_dir, 'experts.bin')
    manifest_path = os.path.join(out_dir, 'manifest.json')
    partial_marker = bin_path + '.partial'

    if (os.path.exists(bin_path) or os.path.exists(manifest_path)) and not force:
        raise RepackAbort('output already exists(%s) - use --force to overwrite' % out_dir)
    if os.path.exists(partial_marker):
        print('[warning] found a .partial marker from a previous run - treating the artifact as incomplete; this run will overwrite it')

    # §2-6 preflight 가용 RAM 검사(피크 층 expert bytes + 2GiB). 프로덕션은 질의 실패도 fail-closed.
    peak = peak_layer_expert_bytes(layout)
    avail, ram_err = available_ram_bytes()
    if avail is None:
        if enforce_reference:
            raise RepackAbort('available-RAM query failed - production preflight is fail-closed: %s' % ram_err)
    elif avail < peak + RAM_HEADROOM_BYTES:
        raise RepackAbort('not enough available RAM (per-layer buffering peak + 2GiB): peak=%d B(%.2f GiB) required=%.2f GiB available=%.2f GiB'
                           % (peak, peak / (1 << 30), (peak + RAM_HEADROOM_BYTES) / (1 << 30), avail / (1 << 30)))

    # 여유 용량 검사(bin+1GiB), preflight 통과 후 디렉토리 생성
    free = shutil.disk_usage(_existing_ancestor(out_dir)).free
    required = bin_bytes + FREE_DISK_HEADROOM
    if free < required:
        raise RepackAbort('not enough free space: required %d B (bin+1GiB, %.2f GB), available %d B (%.2f GB)'
                           % (required, required / 1e9, free, free / 1e9))
    os.makedirs(out_dir, exist_ok=True)

    # §1: derived expect 는 재팩 산출물과 같은 디렉토리에 원자 기록(★번들 expects_dir 금지).
    if derived is not None:
        dpath = write_derived_expect(out_dir, derived['raw'])
        print('[EXPERIMENTAL arch-template] wrote %s (sha256=%s)' % (dpath, derived['sha256']))

    open(partial_marker, 'w').close()
    t0 = time.time()
    cum_mb = 0.0

    # 전 shard 파일 핸들(멀티-shard 레코드는 여러 shard 에서 슬라이스)
    shard_files = {}
    try:
        for h in model['shards']:
            shard_files[h['source_index']] = open(h['path'], 'rb')
        with open(bin_path, 'wb') as fout:
            for L in layout['layers']:
                # §2-6 층 단위 버퍼링: 이 층의 각 파트 텐서를 이론 nbytes 만큼 통읽기
                bufs = {}
                for p in L['parts']:
                    fs = shard_files[p['source_index']]
                    fs.seek(p['abs_offset'])
                    data = fs.read(p['theory_bytes'])
                    if len(data) != p['theory_bytes']:
                        raise RepackAbort('short read from source: layer=%d part=%s expected=%d actual=%d'
                                           % (L['layer'], p['name'], p['theory_bytes'], len(data)))
                    bufs[p['name']] = data
                stride = L['stride_bytes']
                payload_bytes = L['payload_bytes']
                for e in range(n_expert):
                    record_offset = L['record_base'] + e * stride
                    assert fout.tell() == record_offset, ('fout.tell()(%d) != record_offset(%d)'
                                                            % (fout.tell(), record_offset))
                    written = 0
                    for p in L['parts']:
                        peb = p['part_bytes']
                        chunk = bufs[p['name']][e * peb:(e + 1) * peb]
                        if len(chunk) != peb:
                            raise RepackAbort('slice size mismatch: layer=%d expert=%d part=%s expected=%d actual=%d'
                                               % (L['layer'], e, p['name'], peb, len(chunk)))
                        fout.write(chunk)
                        written += peb
                    if written != payload_bytes:
                        raise RepackAbort('record payload sum mismatch: layer=%d expert=%d expected=%d actual=%d'
                                           % (L['layer'], e, payload_bytes, written))
                    pad = stride - payload_bytes
                    if pad:
                        fout.write(b'\x00' * pad)
                elapsed = time.time() - t0
                mb = sum(p['theory_bytes'] for p in L['parts']) / 1e6
                cum_mb += mb
                rate = cum_mb / elapsed if elapsed > 0 else 0.0
                print('[repack] moe layer %d (%d/%d) done - %.1f MB (this layer), cumulative %.1f MB/%.1fs, average %.1f MB/s'
                      % (L['layer'], layout['layers'].index(L) + 1, len(layout['layers']), mb, cum_mb, elapsed, rate))
            assert fout.tell() == bin_bytes, 'final fout.tell()(%d) != bin_bytes(%d)' % (fout.tell(), bin_bytes)
    except Exception:
        traceback.print_exc()
        for fh in shard_files.values():
            try:
                fh.close()
            except Exception:
                pass
        raise
    for fh in shard_files.values():
        fh.close()

    # UI-5: 마지막 층 완료 후 manifest 봉인 → verify 전수 구간이 이어진다. 각 구간 시작을 알린다
    # (stdout print 전용 — 산출물·결정론 무변).
    lock_enforced = enforce_reference or (derived is not None)
    print('[manifest] sealing manifest.json (%d records, reference_lock=%s)...'
          % (plan['n_records'], profile_id if lock_enforced else 'selftest-exempt'))
    manifest = build_manifest(model, layout, plan, A, align_info,
                              profile_id if lock_enforced else 'selftest-exempt', expect_sha256)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, allow_nan=False)

    verify_result = None
    if run_verify:
        verify_result = verify_repack(model_path, out_dir,
                                      profile_id=profile_id, enforce_reference=lock_enforced,
                                      allow_default_align=allow_default_align,
                                      arch_template=arch_template)
        with open(os.path.join(out_dir, 'verify_report.json'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(verify_result, ensure_ascii=False) + '\n')
        if verify_result['pass'] and os.path.exists(partial_marker):
            os.remove(partial_marker)
        if verify_result['pass']:
            print('[verify] %d/%d PASS' % (verify_result['pairs_pass'], verify_result['pairs_total']))
        else:
            print('[verify] FAIL - %d/%d, %d problems, %d failures, %d padding anomalies'
                  % (verify_result['pairs_pass'], verify_result['pairs_total'],
                     len(verify_result['problems']), len(verify_result['failures']),
                     len(verify_result['padding_failures'])))

    _append_repack_log({'ts': datetime.now(timezone.utc).isoformat(), 'mode': 'repack',
                         'model': model_path, 'out': out_dir, 'profile': profile_id,
                         'elapsed_s': time.time() - t0, 'bin_bytes': bin_bytes,
                         'n_records': plan['n_records'],
                         'verify_pass': verify_result['pass'] if verify_result else None})

    return layout, manifest, verify_result


# ---------------------------------------------------------------------------
# 매니페스트 내부 교차 불변식(§3 8항) — 로드된 manifest 자체 일관성(가드·problems 수집)
# ---------------------------------------------------------------------------
def _check_manifest_invariants(manifest, problems):
    """§3 8항 교차 불변식을 로드된 manifest 자체에 대해 강제(독립 재구성 대조와 별개의
    내부 일관성 방어). 크래시 금지 — 구조 손상은 problems 수집."""
    try:
        model = manifest.get('model') if isinstance(manifest.get('model'), dict) else {}
        m_layout = manifest.get('layout') if isinstance(manifest.get('layout'), dict) else {}
        layers = m_layout.get('layers') if isinstance(m_layout.get('layers'), list) else []
        records = manifest.get('records') if isinstance(manifest.get('records'), list) else []
        source_tensors = manifest.get('source_tensors') if isinstance(manifest.get('source_tensors'), list) else []
        quant_traits = manifest.get('quant_traits') if isinstance(manifest.get('quant_traits'), dict) else {}
        n_expert = model.get('n_expert')
        moe_layers = model.get('moe_layers')

        # 불변식 2: layers[].layer 유일·오름차순·model.moe_layers 와 타입 포함 전항 동일
        layer_ids = [L.get('layer') for L in layers if isinstance(L, dict)]
        if len(layer_ids) != len(set(layer_ids)):
            problems.append('invariant 2: duplicate layers[].layer')
        if layer_ids != sorted(x for x in layer_ids if isinstance(x, int)):
            problems.append('invariant 2: layers[].layer is not ascending')
        if not _typed_eq(layer_ids, moe_layers if isinstance(moe_layers, list) else None):
            problems.append('invariant 2: layers[].layer != model.moe_layers (type included)')

        # 불변식 3: parts 연쇄(part_offset[0]==0·연속·마지막 끝==payload_bytes·name/source_tensor 유일)
        for L in layers:
            if not isinstance(L, dict):
                problems.append('invariant 3: layer entry is not an object'); continue
            parts = L.get('parts') if isinstance(L.get('parts'), list) else []
            names = [p.get('name') for p in parts if isinstance(p, dict)]
            stensors = [p.get('source_tensor') for p in parts if isinstance(p, dict)]
            if len(names) != len(set(names)):
                problems.append('invariant 3: layer %r duplicate part name' % L.get('layer'))
            if len(stensors) != len(set(stensors)):
                problems.append('invariant 3: layer %r duplicate source_tensor' % L.get('layer'))
            off = 0
            for p in parts:
                if not isinstance(p, dict):
                    problems.append('invariant 3: part entry is not an object'); break
                if p.get('part_offset') != off:
                    problems.append('invariant 3: layer %r part %r part_offset(%r)!=running total(%d)'
                                    % (L.get('layer'), p.get('name'), p.get('part_offset'), off))
                    break
                pb = p.get('part_bytes')
                if not isinstance(pb, int) or isinstance(pb, bool):
                    problems.append('invariant 3: layer %r part %r part_bytes is not an integer' % (L.get('layer'), p.get('name')))
                    break
                off += pb
            else:
                if off != L.get('payload_bytes'):
                    problems.append('invariant 3: layer %r sum(parts)(%d)!=payload_bytes(%r)'
                                    % (L.get('layer'), off, L.get('payload_bytes')))

        # 불변식 4: base 연쇄(record_base[0]==0·다음==이전+n_expert×stride·slot_stride_max==max(stride))
        base = 0
        strides = []
        for L in layers:
            if not isinstance(L, dict):
                continue
            if L.get('record_base') != base:
                problems.append('invariant 4: layer %r record_base(%r)!=running total(%d)' % (L.get('layer'), L.get('record_base'), base))
                break
            st = L.get('stride_bytes')
            pl = L.get('payload_bytes')
            if not isinstance(st, int) or isinstance(st, bool) or not isinstance(pl, int) or isinstance(pl, bool):
                problems.append('invariant 4: layer %r stride/payload is not an integer' % L.get('layer'))
                break
            if st < pl:
                problems.append('invariant 4: layer %r stride(%d)<payload(%d)' % (L.get('layer'), st, pl))
            strides.append(st)
            if not isinstance(n_expert, int) or isinstance(n_expert, bool):
                problems.append('invariant 4: model.n_expert is not an integer'); break
            base += n_expert * st
        if strides and m_layout.get('slot_stride_max') != max(strides):
            problems.append('invariant 4: slot_stride_max(%r)!=max(stride)(%d)' % (m_layout.get('slot_stride_max'), max(strides)))

        # 불변식 5: records[] = 비권위 witness — layers[]-도출 offset 과 일치 확인
        # (offset==record_base[l]+e×stride[l], payload==payload_bytes[l])
        layer_by_id = {L.get('layer'): L for L in layers if isinstance(L, dict)}
        idx = 0
        witness_ok = True
        for L in layers:
            if not isinstance(L, dict):
                continue
            for e in range(n_expert if isinstance(n_expert, int) and not isinstance(n_expert, bool) else 0):
                if idx >= len(records):
                    problems.append('invariant 5: too few records (idx=%d)' % idx); witness_ok = False; break
                rec = records[idx]
                if not isinstance(rec, dict):
                    problems.append('invariant 5: records[%d] is not an object' % idx); witness_ok = False; break
                want_off = L.get('record_base') + e * L.get('stride_bytes')
                if rec.get('layer') != L.get('layer') or rec.get('expert') != e \
                        or rec.get('offset') != want_off or rec.get('payload_bytes') != L.get('payload_bytes'):
                    problems.append('invariant 5: records[%d] != value derived from layers (layer=%r expert=%d off=%r)'
                                    % (idx, rec.get('layer'), e, rec.get('offset')))
                    witness_ok = False
                    break
                idx += 1
            if not witness_ok:
                break
        if witness_ok and idx != len(records):
            problems.append('invariant 5: too many records (expected %d, actual %d)' % (idx, len(records)))

        # 불변식 6: source_tensors[] ↔ flattened parts 1:1(name·source_index·type·dims·bytes==part_bytes×n_expert)
        flat = []
        for L in layers:
            if not isinstance(L, dict):
                continue
            for p in (L.get('parts') if isinstance(L.get('parts'), list) else []):
                if isinstance(p, dict):
                    flat.append(p)
        if len(flat) != len(source_tensors):
            problems.append('invariant 6: source_tensors count(%d)!=flattened parts(%d)' % (len(source_tensors), len(flat)))
        else:
            for i, (st, p) in enumerate(zip(source_tensors, flat)):
                if not isinstance(st, dict):
                    problems.append('invariant 6: source_tensors[%d] is not an object' % i); continue
                if st.get('name') != p.get('source_tensor'):
                    problems.append('invariant 6: source_tensors[%d].name!=part.source_tensor' % i)
                if st.get('source_index') != p.get('source_index'):
                    problems.append('invariant 6: source_tensors[%d].source_index mismatch' % i)
                if st.get('type') != p.get('type'):
                    problems.append('invariant 6: source_tensors[%d].type mismatch' % i)
                if not _typed_eq(st.get('dims'), p.get('dims')):
                    problems.append('invariant 6: source_tensors[%d].dims mismatch' % i)
                if isinstance(p.get('part_bytes'), int) and not isinstance(p.get('part_bytes'), bool) \
                        and isinstance(n_expert, int) and not isinstance(n_expert, bool):
                    if st.get('bytes') != p.get('part_bytes') * n_expert:
                        problems.append('invariant 6: source_tensors[%d].bytes!=part_bytes*n_expert' % i)

        # 불변식 7: quant_traits == 동결 표(사용 subset)·산술 권위 아님(항상 표에서 재계산)
        for tt, tr in quant_traits.items():
            if tt not in QUANT_TRAITS:
                problems.append('invariant 7: quant_traits type not in the table %r' % tt); continue
            bv, bb = QUANT_TRAITS[tt]
            expected_tr = {'block_values': bv, 'block_bytes': bb}
            if not _typed_eq(tr, expected_tr):
                note = ' - type mismatch' if _type_mismatch(tr, expected_tr) else ''
                problems.append('invariant 7: quant_traits[%s] != frozen table(%d/%d)%s' % (tt, bv, bb, note))
    except Exception as e:
        problems.append('structure corrupt during the invariant check: %r' % e)


# ---------------------------------------------------------------------------
# 독립 2패스 검증 (§5) — 전량 독립 재구성 대조 + per-layer SHA
# ---------------------------------------------------------------------------
def verify_repack(model_path, out_dir, profile_id=None, enforce_reference=True, allow_default_align=False,
                  arch_template=False):
    """§5: 검증 패스가 소스 GGUF(전 shard)를 자체 재파싱해 기대 layout/records/parts/
    source_tensors/스칼라/reference_lock 을 전량 재구성, 디스크 manifest.json 과 엄격 타입
    대조(부분 검사 금지) + records×parts SHA-256 쌍대 재비교(층별 stride) + 패딩 제로 +
    §3 8항 내부 불변식 + reference_lock 독립 재로드·재해시·카탈로그 대조."""
    manifest_path = os.path.join(out_dir, 'manifest.json')
    bin_path = os.path.join(out_dir, 'experts.bin')
    problems = []
    # ★cache key 결속: 이 verify 가 실제로 읽어 검증한 manifest.json **파일 바이트**의 SHA-256.
    # report 레코드에 실어 소비자(C++ seal)가 "이 report 가 이 manifest 를 검증한 것인가"를
    # 독립 대조한다(타 재팩 report 재사용·verify 이후 manifest 교체 차단). 로드 실패 시 None.
    manifest_sha256 = None

    def _fail():
        return {'pairs_total': 0, 'pairs_pass': 0, 'expected_pairs': 0, 'pass': False,
                'failures': [], 'padding_failures': [], 'problems': problems,
                'reference_lock': None, 'manifest_sha256': manifest_sha256,
                'checked_at': datetime.now(timezone.utc).isoformat()}

    # manifest 엄격 로드
    try:
        raw = open(manifest_path, 'rb').read()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = _strict_json_load_bytes(raw)
    except _DuplicateManifestKey as e:
        problems.append('manifest duplicate key: %s' % e); return _fail()
    except _NonStandardJSONConstant as e:
        problems.append('manifest non-standard JSON constant: %s' % e); return _fail()
    except Exception as e:
        problems.append('cannot load manifest.json (corrupt JSON etc.): %r' % e); return _fail()
    if not isinstance(manifest, dict):
        problems.append('the top level of manifest.json is not a JSON object: %r' % type(manifest).__name__); return _fail()

    # 부록A: manifest 자체가 기록한 routed_scope 로 독립 재구성을 구동(부재=all).
    scope = (manifest.get('model') if isinstance(manifest.get('model'), dict) else {}).get('routed_scope', 'all')

    # 소스 전 shard 독립 재파싱 + layout 재도출(호출자 layout 재사용 금지)
    try:
        model = load_model_shards(model_path)
        layout = build_layout(model, scope=scope)
    except RepackAbort as e:
        problems.append('verify pass: independent re-parse of the source failed: %r' % e); return _fail()

    # 정렬 재질의 독립 재도출(§4 사다리)
    try:
        A_expected, align_info = resolve_alignment(out_dir, allow_default_align)
    except RepackAbort as e:
        problems.append('verify pass: alignment re-query failed: %r' % e); return _fail()
    plan_expected = compute_record_layout(layout, A_expected)

    n_expert = layout['n_expert']
    expected_pairs = plan_expected['n_records'] * len(layout['record_order'])
    expected_bin_bytes = plan_expected['bin_bytes']

    # reference_lock 독립 재확인(§3 불변식 8): manifest.reference_lock 3자 동일성·재해시.
    ref = manifest.get('reference_lock') if isinstance(manifest.get('reference_lock'), dict) else {}
    ref_profile = ref.get('profile_id')
    ref_sha = ref.get('expect_sha256')
    reference_lock_out = {'profile_id': ref_profile, 'expect_sha256': ref_sha}
    if arch_template:
        # OPEN_ARCH A축: 카탈로그가 아니라 **소스에서 템플릿을 독립 재유도**해 대조한다
        # (호출자 유도 결과 재사용 금지 — 재유도 → 디스크 derived.expect.json 재해시 → 바이트
        #  동일성 → 같은 cross_check_expect). 등록 경로 로직은 손대지 않는다.
        try:
            derivation_re = derive_arch_template(model, requested_scope=scope)
            _expect_re, raw_re, sha_re = build_derived_expect(model, layout, derivation_re)
        except RepackAbort as e:
            problems.append('independent re-derivation of the arch template failed: %r' % e)
        else:
            lock_expected = arch_template_lock_id(derivation_re)
            if ref_profile != lock_expected:
                problems.append('reference_lock.profile_id(%r) != independently re-derived template lock(%r)'
                                % (ref_profile, lock_expected))
            if profile_id is not None and ref_profile != profile_id:
                problems.append('reference_lock.profile_id(%r) != the lock id used by this run(%r)'
                                % (ref_profile, profile_id))
            dpath = os.path.join(out_dir, DERIVED_EXPECT_FILENAME)
            try:
                raw_disk = open(dpath, 'rb').read()
            except Exception as e:
                problems.append('cannot read the derived expect (%s): %r' % (dpath, e))
            else:
                sha_disk = hashlib.sha256(raw_disk).hexdigest()
                if ref_sha != sha_disk:
                    problems.append('reference_lock.expect_sha256(%r) != re-hash of %s(%r)'
                                    % (ref_sha, DERIVED_EXPECT_FILENAME, sha_disk))
                if raw_disk != raw_re:
                    problems.append('%s bytes != independent re-derivation (on-disk sha=%s, re-derived sha=%s)'
                                    % (DERIVED_EXPECT_FILENAME, sha_disk, sha_re))
                try:
                    expect_disk = _strict_json_load_bytes(raw_disk)
                    cross_check_expect(model, layout, plan_expected, expect_disk, scope=scope)
                except (_DuplicateManifestKey, _NonStandardJSONConstant) as e:
                    problems.append('derived expect is not strict JSON: %r' % e)
                except RepackAbort as e:
                    problems.append('independent re-check of the derived expect failed: %r' % e)
                except Exception as e:
                    problems.append('derived expect JSON is corrupt: %r' % e)
    elif enforce_reference:
        # 요청 profile 과 manifest 의 profile 일치
        if profile_id is not None and ref_profile != profile_id:
            problems.append('reference_lock.profile_id(%r) != requested --profile(%r)' % (ref_profile, profile_id))
        try:
            expect, expect_sha256 = load_expect_profile(ref_profile if isinstance(ref_profile, str) else '')
            if ref_sha != expect_sha256:
                problems.append('reference_lock.expect_sha256(%r) != independent re-hash(%r)' % (ref_sha, expect_sha256))
            cross_check_expect(model, layout, plan_expected, expect, scope=scope)
        except RepackAbort as e:
            problems.append('independent re-check of reference_lock failed: %r' % e)
    else:
        # selftest 면제 경로: 카탈로그 비경유. reference_lock 표식만 확인.
        if ref_profile != 'selftest-exempt' or ref_sha != 'selftest-exempt':
            problems.append('selftest reference_lock marker mismatch: %r/%r' % (ref_profile, ref_sha))

    # ---- 스칼라·구조 전항 독립 재구성 대조 ----
    m_model = manifest.get('model') if isinstance(manifest.get('model'), dict) else {}
    m_layout = manifest.get('layout') if isinstance(manifest.get('layout'), dict) else {}
    m_totals = manifest.get('totals') if isinstance(manifest.get('totals'), dict) else {}
    m_sources = manifest.get('sources') if isinstance(manifest.get('sources'), list) else []
    expert_payload_total = n_expert * sum(L['payload_bytes'] for L in layout['layers'])

    scalar_checks = [
        ('schema_version', manifest.get('schema_version'), SCHEMA_VERSION),
        ('model.arch', m_model.get('arch'), layout['arch']),
        ('model.n_layer', m_model.get('n_layer'), layout['n_layer']),
        ('model.n_expert', m_model.get('n_expert'), n_expert),
        ('model.n_expert_used', m_model.get('n_expert_used'), layout['n_expert_used']),
        ('model.moe_layers', m_model.get('moe_layers'), list(layout['moe_layers'])),
        ('layout.align_bytes', m_layout.get('align_bytes'), A_expected),
        ('layout.slot_stride_max', m_layout.get('slot_stride_max'), plan_expected['slot_stride_max']),
        ('totals.expert_payload_total', m_totals.get('expert_payload_total'), expert_payload_total),
        ('totals.bin_file_bytes', m_totals.get('bin_file_bytes'), expected_bin_bytes),
        ('totals.n_records', m_totals.get('n_records'), plan_expected['n_records']),
    ]
    for field, actual, expected in scalar_checks:
        if not _typed_eq(actual, expected):
            note = ' - type mismatch (actual=%s expected=%s)' % (type(actual).__name__, type(expected).__name__) \
                if _type_mismatch(actual, expected) else ''
            problems.append('manifest.%s(%r) != independently re-derived(%r)%s' % (field, actual, expected, note))

    # sources[] 전항(index·bytes·gguf_version·alignment·data_start; path/mtime 는 존재·타입만)
    if len(m_sources) != len(model['shards']):
        problems.append('sources count %d != independent re-parse %d' % (len(m_sources), len(model['shards'])))
    else:
        for i, (ms, h) in enumerate(zip(m_sources, model['shards'])):
            if not isinstance(ms, dict):
                problems.append('sources[%d] is not an object' % i); continue
            for field, actual, expected in [
                ('index', ms.get('index'), h['source_index']),
                ('bytes', ms.get('bytes'), h['file_bytes']),
                ('gguf_version', ms.get('gguf_version'), h['gguf_version']),
                ('alignment', ms.get('alignment'), h['alignment']),
                ('data_start', ms.get('data_start'), h['data_start']),
            ]:
                if not _typed_eq(actual, expected):
                    problems.append('sources[%d].%s(%r) != re-parse(%r)' % (i, field, actual, expected))

    # layout.layers[] 전항 독립 재구성 대조
    m_layers = m_layout.get('layers') if isinstance(m_layout.get('layers'), list) else []
    if len(m_layers) != len(layout['layers']):
        problems.append('layout.layers count %d != re-derived %d' % (len(m_layers), len(layout['layers'])))
    else:
        for i, (ml, L) in enumerate(zip(m_layers, layout['layers'])):
            if not isinstance(ml, dict):
                problems.append('layout.layers[%d] is not an object' % i); continue
            for field, actual, expected in [
                ('layer', ml.get('layer'), L['layer']),
                ('payload_bytes', ml.get('payload_bytes'), L['payload_bytes']),
                ('stride_bytes', ml.get('stride_bytes'), L['stride_bytes']),
                ('record_base', ml.get('record_base'), L['record_base']),
            ]:
                if not _typed_eq(actual, expected):
                    note = ' - type mismatch' if _type_mismatch(actual, expected) else ''
                    problems.append('layout.layers[%d].%s(%r)!=re-derived(%r)%s' % (i, field, actual, expected, note))
            m_parts = ml.get('parts') if isinstance(ml.get('parts'), list) else []
            if len(m_parts) != len(L['parts']):
                problems.append('layout.layers[%d].parts count %d != %d' % (i, len(m_parts), len(L['parts'])))
                continue
            for j, (mp, p) in enumerate(zip(m_parts, L['parts'])):
                if not isinstance(mp, dict):
                    problems.append('layout.layers[%d].parts[%d] is not an object' % (i, j)); continue
                want = [('name', mp.get('name'), p['name']),
                        ('source_tensor', mp.get('source_tensor'), p['source_tensor']),
                        ('source_index', mp.get('source_index'), p['source_index']),
                        ('type', mp.get('type'), p['type']),
                        ('dims', mp.get('dims'), list(p['dims'])),
                        ('expert_axis', mp.get('expert_axis'), p['expert_axis']),
                        ('part_offset', mp.get('part_offset'), p['part_offset']),
                        ('part_bytes', mp.get('part_bytes'), p['part_bytes'])]
                for field, actual, expected in want:
                    if not _typed_eq(actual, expected):
                        note = ' - type mismatch' if _type_mismatch(actual, expected) else ''
                        problems.append('layout.layers[%d].parts[%d].%s(%r)!=re-derived(%r)%s'
                                        % (i, j, field, actual, expected, note))

    # records[] 독립 재생성 전항 대조(비권위 witness)
    expected_records = plan_expected['records']
    records = manifest.get('records') if isinstance(manifest.get('records'), list) else None
    if records is None:
        problems.append('records missing or not a list')
    elif len(records) != len(expected_records):
        problems.append('records count mismatch: actual=%d expected=%d' % (len(records), len(expected_records)))
    else:
        for i, (rec, exp) in enumerate(zip(records, expected_records)):
            if not isinstance(rec, dict):
                problems.append('records[%d] is not an object' % i); continue
            actual = (rec.get('layer'), rec.get('expert'), rec.get('offset'), rec.get('payload_bytes'))
            wanted = (exp['layer'], exp['expert'], exp['offset'], exp['payload_bytes'])
            if not _typed_eq(list(actual), list(wanted)):
                note = ' - type mismatch' if any(_type_mismatch(a, w) for a, w in zip(actual, wanted)) else ''
                problems.append('records[%d] mismatch: actual=%r expected=%r%s' % (i, actual, wanted, note))

    # source_tensors[] 독립 재생성 전항 대조
    expected_source_tensors = []
    for L in layout['layers']:
        for p in L['parts']:
            expected_source_tensors.append((p['source_tensor'], p['source_index'], p['abs_offset'],
                                            p['part_bytes'] * n_expert, p['type'], list(p['dims'])))
    m_stensors = manifest.get('source_tensors') if isinstance(manifest.get('source_tensors'), list) else None
    if m_stensors is None:
        problems.append('source_tensors missing or not a list')
    elif len(m_stensors) != len(expected_source_tensors):
        problems.append('source_tensors count mismatch: actual=%d expected=%d' % (len(m_stensors), len(expected_source_tensors)))
    else:
        for i, (st, exp) in enumerate(zip(m_stensors, expected_source_tensors)):
            if not isinstance(st, dict):
                problems.append('source_tensors[%d] is not an object' % i); continue
            actual = (st.get('name'), st.get('source_index'), st.get('abs_offset'), st.get('bytes'),
                      st.get('type'), st.get('dims'))
            if not _typed_eq(list(actual), list(exp)):
                note = ' - type mismatch' if any(_type_mismatch(a, w) for a, w in zip(actual, exp)) else ''
                problems.append('source_tensors[%d](%s) mismatch: actual=%r expected=%r%s' % (i, exp[0], actual, exp, note))

    # quant_traits 독립 재생성 대조
    expected_traits = {tt: {'block_values': QUANT_TRAITS[tt][0], 'block_bytes': QUANT_TRAITS[tt][1]}
                       for tt in layout['used_types']}
    m_traits = manifest.get('quant_traits') if isinstance(manifest.get('quant_traits'), dict) else {}
    if set(m_traits.keys()) != set(expected_traits.keys()):
        problems.append('quant_traits key set mismatch: actual=%s expected=%s' % (sorted(m_traits), sorted(expected_traits)))
    else:
        for tt, exp in expected_traits.items():
            if not _typed_eq(m_traits.get(tt), exp):
                problems.append('quant_traits[%s] mismatch: actual=%r expected=%r' % (tt, m_traits.get(tt), exp))

    # bin 실제 크기(totals 우회 방지)
    actual_bin_bytes = os.path.getsize(bin_path)
    if actual_bin_bytes != expected_bin_bytes:
        problems.append('experts.bin actual size(%d) != independently computed n_records*stride(%d)'
                        % (actual_bin_bytes, expected_bin_bytes))

    # §3 8항 내부 불변식(로드 manifest 자체 일관성)
    _check_manifest_invariants(manifest, problems)

    # ---- per-layer SHA-256 쌍대(소스 slice ↔ bin slice) + 패딩 제로 ----
    failures = []
    padding_failures = []
    pairs_total = 0
    pairs_pass = 0
    # UI-5: 이 구간은 수만 쌍을 재읽기·재해시하므로 수 분간 무출력이 되기 쉽다(실기동 stall 오인).
    # 진행 출력은 stdout print 뿐 — 산출물(verify_report.json·manifest.json·repack_log.jsonl)과
    # 결정론에는 어떤 영향도 주지 않는다. 총량의 5%마다(최소 1000쌍 간격) 1줄이라 최대 20줄이고,
    # 픽스처처럼 작은 세트에서는 주기 출력이 아예 나오지 않는다.
    _v_step = max(1000, expected_pairs // 20)
    _v_next = _v_step
    if not problems:
        print('[verify] checking %s record pairs (re-reading source shards and experts.bin)...'
              % format(expected_pairs, ','))
        shard_files = {}
        try:
            for h in model['shards']:
                shard_files[h['source_index']] = open(h['path'], 'rb')
            with open(bin_path, 'rb') as fout:
                for L in layout['layers']:
                    stride = L['stride_bytes']
                    payload_bytes = L['payload_bytes']
                    for e in range(n_expert):
                        off = L['record_base'] + e * stride
                        cursor = off
                        for p in L['parts']:
                            peb = p['part_bytes']
                            fs = shard_files[p['source_index']]
                            src_off = p['abs_offset'] + e * peb
                            fs.seek(src_off)
                            src_bytes = fs.read(peb)
                            fout.seek(cursor)
                            out_bytes = fout.read(peb)
                            h_src = hashlib.sha256(src_bytes).hexdigest()
                            h_out = hashlib.sha256(out_bytes).hexdigest()
                            pairs_total += 1
                            if h_src == h_out and len(src_bytes) == peb and len(out_bytes) == peb:
                                pairs_pass += 1
                            else:
                                failures.append({'layer': L['layer'], 'expert': e, 'part': p['name'],
                                                 'src_offset': src_off, 'out_offset': cursor,
                                                 'src_sha256': h_src, 'out_sha256': h_out})
                            cursor += peb
                        pad_len = stride - payload_bytes
                        if pad_len:
                            fout.seek(off + payload_bytes)
                            pad_bytes = fout.read(pad_len)
                            if pad_bytes != b'\x00' * pad_len:
                                padding_failures.append({'layer': L['layer'], 'expert': e,
                                                         'offset': off + payload_bytes, 'pad_len': pad_len})
                        if pairs_total >= _v_next:
                            pct = (pairs_total * 100.0 / expected_pairs) if expected_pairs else 100.0
                            print('[verify] %s/%s pairs (%.0f%%)'
                                  % (format(pairs_total, ','), format(expected_pairs, ','), pct))
                            _v_next += _v_step
        finally:
            for fh in shard_files.values():
                try:
                    fh.close()
                except Exception:
                    pass

    passed = (not problems) and (not failures) and (not padding_failures) \
        and pairs_total == expected_pairs and pairs_pass == expected_pairs

    return {'pairs_total': pairs_total, 'pairs_pass': pairs_pass, 'expected_pairs': expected_pairs,
            'pass': passed, 'failures': failures, 'padding_failures': padding_failures,
            'problems': problems, 'reference_lock': reference_lock_out,
            'manifest_sha256': manifest_sha256,
            'checked_at': datetime.now(timezone.utc).isoformat()}


# RC-1: append-only 블랙박스의 기록 내용은 그대로 두고 '어디에' 쌓을지만 호출자가 정할 수 있게 한다.
# 미지정 = 종전대로 이 스크립트 옆(bench 직접 실행 하위 호환). launcher 는 번들 밖 경로를 넘긴다 —
# 번들 안에 파일이 생기면 다음 기동의 SHA manifest 게이트가 자기 번들을 거부하기 때문(실사고 2회).
_REPACK_LOG_PATH = None


def _set_repack_log_path(path):
    global _REPACK_LOG_PATH
    _REPACK_LOG_PATH = path or None


def _append_repack_log(entry):
    log_path = _REPACK_LOG_PATH
    if log_path:
        parent = os.path.dirname(os.path.abspath(log_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
    else:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repack_log.jsonl')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# --plan / 본실행 커맨드
# ---------------------------------------------------------------------------
def cmd_plan(args):
    arch_template = bool(getattr(args, 'arch_template', False))
    print('=== --plan: GGUF header analysis (0 bytes written) ===')
    print('profile: %s' % ('(EXPERIMENTAL arch-template: derived on the spot)' if arch_template else args.profile))
    print('model: %s' % args.model)
    print('out (planned target): %s' % args.out)
    model, layout, A, align_info, plan, expect_sha256, expect_totals, derived = _prepare(
        args.model, args.out, args.profile, allow_default_align=False, enforce_reference=True,
        scope=args.scope, arch_template=arch_template)
    _print_plan_summary(model, layout, plan, A, align_info, args.out,
                        derived['lock_id'] if derived else args.profile,
                        expect_sha256, expect_totals, derived=derived)
    if derived is not None:
        # --plan 은 0바이트 계약이므로 파일로 쓰지 않고 본문만 보여준다(실기록은 본실행에서).
        print('--- derived expect (%s, not written in --plan) ---' % DERIVED_EXPECT_FILENAME)
        print(derived['raw'].decode('utf-8').rstrip('\n'))
    print('--plan done (0 bytes written, no GPU used)')


def cmd_repack(args):
    arch_template = bool(getattr(args, 'arch_template', False))
    print('=== real run: --profile=%s --model=%s --out=%s force=%s arch_template=%s ==='
          % (args.profile, args.model, args.out, args.force, arch_template))
    layout, manifest, verify_result = do_repack(args.model, args.out, args.profile,
                                                 force=args.force, run_verify=True, scope=args.scope,
                                                 arch_template=arch_template)
    if verify_result and verify_result['pass']:
        print('=== real run done: PASS %d/%d ===' % (verify_result['pairs_pass'], verify_result['pairs_total']))
    else:
        print('=== real run verification FAILED - do not proceed ===')
        sys.exit(1)


# ---------------------------------------------------------------------------
# --verify-only (하위호환 승급 경로) — 재팩 없이 기존 산출물 1회 전체 재검증
# ---------------------------------------------------------------------------
def do_verify_only(model_path, out_dir, profile_id, enforce_reference=True, allow_default_align=False,
                   arch_template=False):
    """기존 산출물(source GGUF + experts.bin + manifest.json)을 **재팩 없이** 기존 verify 경로
    (verify_repack)로 1회 전체 재검증하고 verify_report.json 에 새 레코드를 append 한다.
    manifest_sha256 이 없는 구 report 로 생성된 산출물을 재팩 없이 승급시키는 경로다.
    ★산출물 바이트는 건드리지 않는다 — experts.bin/manifest.json 불변, .partial 표식도
    생성·제거하지 않는다(쓰기는 verify_report.json append 1줄뿐. scope 는 manifest.model.
    routed_scope 를 verify_repack 이 스스로 읽어 구동하므로 --scope 를 받지 않는다).
    실패 시에도 실패 레코드를 append 한다(재팩 경로 관행 그대로 — 장부 누락 금지)."""
    for name in ('experts.bin', 'manifest.json'):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            raise RepackAbort('--verify-only: artifact missing - %s (nothing to re-verify)' % p)
    t0 = time.time()
    verify_result = verify_repack(model_path, out_dir, profile_id=profile_id,
                                  enforce_reference=enforce_reference,
                                  allow_default_align=allow_default_align,
                                  arch_template=arch_template)
    with open(os.path.join(out_dir, 'verify_report.json'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(verify_result, ensure_ascii=False) + '\n')
    _append_repack_log({'ts': datetime.now(timezone.utc).isoformat(), 'mode': 'verify-only',
                         'model': model_path, 'out': out_dir, 'profile': profile_id,
                         'elapsed_s': time.time() - t0,
                         'manifest_sha256': verify_result['manifest_sha256'],
                         'verify_pass': verify_result['pass']})
    return verify_result


def cmd_verify_only(args):
    arch_template = bool(getattr(args, 'arch_template', False))
    print('=== --verify-only: re-verifying the existing artifact without repacking --profile=%s --model=%s --out=%s '
          'arch_template=%s ===' % (args.profile, args.model, args.out, arch_template))
    verify_result = do_verify_only(args.model, args.out, args.profile, arch_template=arch_template)
    if verify_result['pass']:
        print('[verify-only] PASS %d/%d - manifest_sha256=%s'
              % (verify_result['pairs_pass'], verify_result['pairs_total'],
                 verify_result['manifest_sha256']))
        print('=== --verify-only done: appended a new verify_report record (artifact bytes unchanged) ===')
    else:
        print('[verify-only] FAIL - %d/%d, %d problems, %d failures, %d padding anomalies'
              % (verify_result['pairs_pass'], verify_result['pairs_total'],
                 len(verify_result['problems']), len(verify_result['failures']),
                 len(verify_result['padding_failures'])))
        print('=== --verify-only verification FAILED - do not proceed ===')
        sys.exit(1)


# ---------------------------------------------------------------------------
# --selftest: 합성 GGUF 생성 → 재팩 → 독립 2패스 검증 (+ v1 네거티브 회귀 + §5 신설 ①~⑨)
# ---------------------------------------------------------------------------
def write_synthetic_gguf(shard_paths, *, arch='moetest', n_expert=4, n_expert_used=2,
                         block_count=None, moe_layers=(0, 1), schema='separate', bias=True,
                         hidden=8, layer_quant=None, alignment=32, seed=1234,
                         shard_of=None, gap_after=None, off_table_layer=None,
                         axis_violation=False, meta_expert_count=None, nextn_kv=None,
                         schema_by_layer=None, omit_kv=(), extra_kv_by_shard=None,
                         extra_tensors=(), drop_tensors=(), rename_tensors=None,
                         duplicate_tensor=None, allow_block_misalign=False):
    """합성 미니 GGUF 생성(단일/멀티 shard). routed 6/2텐서 명명·전문가 축 최외곽 유지.
    실 모델 불필요(바이트 이동만 검증하므로 quant 내용은 난수). 텐서는 shard 별로 rel_offset
    0부터 이론 nbytes 만큼 연속 배치(gap=이론 → §2-3 검사 통과). 옵션으로 gap 삽입·표 밖 타입·
    expert_axis 위반·멀티-shard 분산·bias 유무·fused/separate·층별 quant·nextn_predict_layers KV
    (부록A execution scope 테스트 — int=shard0 단일값, dict={source_index: value}=shard별 값·
    충돌 유발용)·schema_by_layer({layer: 'separate'|'fused'}=schema 전역값을 해당 층만 오버라이드
    — 부록A8 tail-only schema 이질 테스트)를 구성한다.

    ★OPEN_ARCH A축 변이 옵션(합성 픽스처 전용·기본값은 기존 동작 그대로):
      omit_kv=(전체 KV 키,)             : 해당 KV 를 쓰지 않음(구조 KV 부재 픽스처)
      extra_kv_by_shard={si: [(k,t,v)]} : shard 별 추가 KV(형 불일치·shard 간 값 충돌 픽스처)
      extra_tensors=({'name','dims','type'},) : 텐서 추가(part 추가·MTP 표식 픽스처)
      drop_tensors=(name,)              : 텐서 삭제(part 누락 픽스처)
      rename_tensors={old: new}         : 텐서 개명(미분류 expert-like 픽스처)
      duplicate_tensor=name             : 같은 이름 텐서 2회 기록(tensor-name 중복 픽스처)
      allow_block_misalign=True         : ne0%block_values!=0 합성 허용(산술 불폐합 픽스처)"""
    moe_layers = list(moe_layers)
    if block_count is None:
        block_count = max(moe_layers) + 1
    if layer_quant is None:
        layer_quant = {l: 'F32' for l in moe_layers}
    if shard_of is None:
        shard_of = {}
    schema_by_layer = schema_by_layer or {}
    n_shards = len(shard_paths)
    rng = random.Random(seed)

    def _weight_kinds(l):
        return ['gate', 'up', 'down'] if schema_by_layer.get(l, schema) == 'separate' else ['gate_up', 'down']

    def _nbytes(ttype, dims):
        if ttype in QUANT_TRAITS:
            bv, bb = QUANT_TRAITS[ttype]
            if dims[0] % bv != 0:
                if not allow_block_misalign:
                    raise AssertionError('synthetic dims[0](%d)%%bv(%d)!=0 type=%s' % (dims[0], bv, ttype))
                return -(-dims[0] // bv) * bb * _prod(dims[1:-1]) * dims[-1]   # 올림(픽스처 전용)
            return (dims[0] // bv) * bb * _prod(dims[1:-1]) * dims[-1]
        return 4 * _prod(dims)   # 표 밖 타입은 F32 크기로(도구는 타입 조회 시점에 중단)

    TYPE_CODE = {'F32': 0, 'F16': 1, 'Q3_K': 11, 'Q4_K': 12, 'Q5_K': 13, 'Q6_K': 14,
                 'Q8_0': 8, 'MXFP4': 39, 'Q2_K': 10, 'BF16': 30}

    # 텐서 목록 구성(shard 별)
    per_shard = {i: [] for i in range(n_shards)}
    for l in moe_layers:
        wt = layer_quant[l]
        weight_kinds = _weight_kinds(l)
        for ki, kind in enumerate(weight_kinds):
            name = 'blk.%d.ffn_%s_exps.weight' % (l, kind)
            use_type = wt
            dims = [hidden, hidden, n_expert]
            if off_table_layer == l:
                use_type = 'Q2_K'    # 표 밖(도구 중단 검증)
            if axis_violation and l == moe_layers[0] and ki == 0:
                dims = [n_expert, hidden, hidden]  # expert 축을 마지막이 아니게(중단 검증)
            code = TYPE_CODE[use_type]
            nbytes = _nbytes(use_type, dims)
            per_shard[shard_of.get(name, 0)].append(
                {'name': name, 'dims': dims, 'code': code, 'nbytes': nbytes})
        if bias:
            for kind in weight_kinds:
                name = 'blk.%d.ffn_%s_exps.bias' % (l, kind)
                dims = [hidden, n_expert]
                nbytes = _nbytes('F32', dims)
                per_shard[shard_of.get(name, 0)].append(
                    {'name': name, 'dims': dims, 'code': TYPE_CODE['F32'], 'nbytes': nbytes})

    # ★OPEN_ARCH 변이 옵션(삭제 → 개명 → 중복 → 추가 순 — 배치 전에 적용)
    if drop_tensors:
        _drop = set(drop_tensors)
        for i in range(n_shards):
            per_shard[i] = [t for t in per_shard[i] if t['name'] not in _drop]
    if rename_tensors:
        for i in range(n_shards):
            for t in per_shard[i]:
                if t['name'] in rename_tensors:
                    t['name'] = rename_tensors[t['name']]
    if duplicate_tensor:
        for i in range(n_shards):
            per_shard[i].extend([dict(t) for t in per_shard[i] if t['name'] == duplicate_tensor])
    for spec in (extra_tensors or ()):
        _dims = list(spec['dims'])
        _type = spec.get('type', 'F32')
        per_shard[shard_of.get(spec['name'], 0)].append(
            {'name': spec['name'], 'dims': _dims, 'code': TYPE_CODE[_type], 'nbytes': _nbytes(_type, _dims)})

    # 배치(shard 별 rel_offset 0부터 연속. gap_after 지정 시 해당 텐서 직후에 초과 gap 삽입)
    for i in range(n_shards):
        running = 0
        for t in per_shard[i]:
            t['rel_offset'] = running
            post = 0
            if gap_after is not None and t['name'] == gap_after:
                theory = t['nbytes']
                post = (_ceil_to(theory, alignment) - theory) + alignment  # 정렬 패딩 초과 보장
            t['post_gap'] = post
            t['payload'] = rng.randbytes(t['nbytes'])
            running += t['nbytes'] + post

    def _write_shard(path, tensors, is_first, source_index, total_tensors):
        kv = [('general.architecture', T_STR, arch),
              ('general.alignment', T_U32, alignment)]
        if is_first:
            kv += [('general.name', T_STR, 'moe-repack-selftest'),
                   ('%s.block_count' % arch, T_U32, block_count),
                   ('%s.expert_count' % arch, T_U32, (meta_expert_count if meta_expert_count is not None else n_expert)),
                   ('%s.expert_used_count' % arch, T_U32, n_expert_used)]
        if n_shards > 1:
            kv += [('split.no', T_U16, source_index),
                   ('split.count', T_U16, n_shards),
                   ('split.tensors.count', T_I32, total_tensors)]
        if isinstance(nextn_kv, dict):
            if source_index in nextn_kv:
                kv.append(('%s.nextn_predict_layers' % arch, T_U32, nextn_kv[source_index]))
        elif nextn_kv is not None and is_first:
            kv.append(('%s.nextn_predict_layers' % arch, T_U32, nextn_kv))
        # ★삭제를 먼저 적용한 뒤 추가한다(같은 키를 다른 형으로 바꿔 넣는 픽스처가 성립하도록).
        if omit_kv:
            kv = [item for item in kv if item[0] not in set(omit_kv)]
        for extra in (extra_kv_by_shard or {}).get(source_index, []):
            kv.append(tuple(extra))
        with open(path, 'wb') as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', 3))
            f.write(struct.pack('<Q', len(tensors)))
            f.write(struct.pack('<Q', len(kv)))
            for key, tcode, val in kv:
                kb = key.encode('utf-8')
                f.write(struct.pack('<Q', len(kb))); f.write(kb)
                f.write(struct.pack('<I', tcode))
                if tcode == T_STR:
                    vb = val.encode('utf-8'); f.write(struct.pack('<Q', len(vb))); f.write(vb)
                elif tcode == T_U32:
                    f.write(struct.pack('<I', val))
                elif tcode == T_U16:
                    f.write(struct.pack('<H', val))
                elif tcode == T_I32:
                    f.write(struct.pack('<i', val))
                elif tcode == T_F32:
                    f.write(struct.pack('<f', val))
                else:
                    raise AssertionError('selftest kv unsupported type %r' % tcode)
            for t in tensors:
                nb = t['name'].encode('utf-8')
                f.write(struct.pack('<Q', len(nb))); f.write(nb)
                f.write(struct.pack('<I', len(t['dims'])))
                f.write(struct.pack('<%dQ' % len(t['dims']), *t['dims']))
                f.write(struct.pack('<I', t['code']))
                f.write(struct.pack('<Q', t['rel_offset']))
            header_end = f.tell()
            # ★물리 파일은 생산자 관행대로 정렬 경계까지 패딩한다(실 397B shard1 도 file_bytes
            # 가 패딩값 10,943,552 이고 raw header_end 는 10,943,537 — 15B 의 트레일링 패딩이
            # 파일에 실재한다). 텐서 0개일 때 판독기가 그 패딩을 건너뛰지 '않는' 것이 결함의
            # 본질이므로, 합성물도 패딩 바이트는 그대로 두고 parse 쪽 산식만 upstream 을 따른다.
            data_start = (header_end + alignment - 1) // alignment * alignment
            f.write(b'\x00' * (data_start - header_end))
            for t in tensors:
                assert f.tell() == data_start + t['rel_offset']
                f.write(t['payload'])
                if t['post_gap']:
                    f.write(b'\x00' * t['post_gap'])

    total_tensors = sum(len(per_shard[i]) for i in range(n_shards))
    for i, path in enumerate(shard_paths):
        _write_shard(path, per_shard[i], is_first=(i == 0), source_index=i, total_tensors=total_tensors)

    # 기대 per_expert payload(정상 케이스 참고용 — moe_layers[0] 기준. schema_by_layer 로 층별
    # schema 가 갈린 합성물은 이 값이 다른 층에는 적용되지 않음에 유의).
    weight_kinds0 = _weight_kinds(moe_layers[0])
    per_expert = 0
    for kind in weight_kinds0:
        per_expert += _nbytes(layer_quant[moe_layers[0]], [hidden, hidden, n_expert]) // n_expert
    if bias:
        for kind in weight_kinds0:
            per_expert += _nbytes('F32', [hidden, n_expert]) // n_expert
    return {'per_expert_bytes_expected': per_expert, 'moe_layers': moe_layers, 'n_expert': n_expert}


def _load_manifest_disk(out_dir):
    with open(os.path.join(out_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_manifest_disk(out_dir, manifest):
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, allow_nan=False)


def _selftest_traits():
    """§2-3 관문: type-trait 표를 b10057 ggml-common.h 블록 정의 필드 분해로 재도출해 QUANT_TRAITS
    와 정확 일치 검증(ggml_half=2·QK_K=256·QK8_0=QK_MXFP4=32)."""
    QK_K, HALF = 256, 2
    derived = {
        'F32':   (1, 4),
        'F16':   (1, HALF),
        'MXFP4': (32, 1 + 32 // 2),                                  # e + qs[QK_MXFP4/2]
        'Q3_K':  (QK_K, QK_K // 8 + QK_K // 4 + 12 + HALF),          # hmask+qs+scales[12]+d
        'Q4_K':  (QK_K, HALF * 2 + 12 + QK_K // 2),                  # dm(=d+dmin)+scales[12]+qs
        'Q5_K':  (QK_K, HALF * 2 + 12 + QK_K // 8 + QK_K // 2),      # dm+scales[12]+qh+qs
        'Q6_K':  (QK_K, QK_K // 2 + QK_K // 4 + QK_K // 16 + HALF),  # ql+qh+scales+d
        'Q8_0':  (32, HALF + 32),                                    # d + qs[QK8_0]
    }
    return derived == QUANT_TRAITS, derived


def _run_negative_case(model_path, scratch, tag, corrupt_fn):
    """네거티브 컨트롤 공용 러너: 깨끗한 재팩(내장 verify PASS 기준선) → corrupt_fn 적용 →
    독립 verify_repack 재호출 → 반드시 FAIL 이어야 통과(selftest 면제 경로)."""
    out_dir = os.path.join(scratch, 'neg_%s' % tag)
    layout, manifest, verify_result = do_repack(model_path, out_dir, profile_id=None, force=False,
                                                 run_verify=True, enforce_reference=False, allow_default_align=True)
    if not (verify_result and verify_result['pass']):
        return False, '기준 재팩 자체가 오염 전 PASS 못함 — 테스트 무효(%r)' % (verify_result['problems'] if verify_result else None)
    corrupt_fn(out_dir, layout)
    neg = verify_repack(model_path, out_dir, profile_id=None, enforce_reference=False, allow_default_align=True)
    ok = neg['pass'] is False
    detail = ('pairs=%d/%d(기대 %d) problems=%r failures=%d 패딩실패=%d'
              % (neg['pairs_pass'], neg['pairs_total'], neg['expected_pairs'],
                 neg['problems'], len(neg['failures']), len(neg['padding_failures'])))
    return ok, detail


def _expect_abort(fn):
    """fn() 이 RepackAbort 를 던지면 (True, 메시지). 아니면 (False, ...)."""
    try:
        fn()
        return False, '(RepackAbort 미발생 — 심각)'
    except RepackAbort as e:
        return True, str(e)[:200]
    except Exception as e:
        return False, '(다른 예외: %r)' % e


def cmd_selftest():
    print('=== --selftest 시작 ===')
    checks = []
    scratch = tempfile.mkdtemp(prefix='repack_selftest_')
    model_path = os.path.join(scratch, 'synthetic.gguf')
    out_dir = os.path.join(scratch, 'out')

    try:
        # ---- type-trait 표 관문(§2-3 ggml-common.h 대조) ----
        ok_tr, derived = _selftest_traits()
        checks.append(('type-trait 표가 b10057 ggml-common.h 필드 분해 재도출과 일치(§2-3 관문)', ok_tr))
        print('[selftest] type-trait 표 대조: %s' % ('OK' if ok_tr else 'FAIL %r' % (derived,)))

        # ---- 기본 separate+bias 단일 shard(= gpt-oss 형상 축소) ----
        meta_info = write_synthetic_gguf([model_path], n_expert=4, moe_layers=(0, 1),
                                         schema='separate', bias=True, hidden=8,
                                         alignment=32, seed=1234)
        print('[selftest] 합성 GGUF(separate+bias) 생성: n_expert=%d moe_layers=%r'
              % (meta_info['n_expert'], meta_info['moe_layers']))

        model = load_model_shards(model_path)
        layout = build_layout(model)
        ok_layout = (layout['schema'] == 'separate' and layout['has_bias']
                     and layout['record_order'] == ['gate.weight', 'up.weight', 'down.weight',
                                                     'gate.bias', 'up.bias', 'down.bias']
                     and layout['moe_layers'] == [0, 1])
        checks.append(('레이아웃 재도출(separate+bias·record_order·moe_layers) 정합', ok_layout))
        print('[selftest] 레이아웃 재도출: %s (order=%r)' % ('OK' if ok_layout else 'FAIL', layout['record_order']))

        layout2, manifest, verify_result = do_repack(model_path, out_dir, profile_id=None, force=False,
                                                       run_verify=True, enforce_reference=False, allow_default_align=True)
        pairs_expected = len(layout['moe_layers']) * layout['n_expert'] * len(layout['record_order'])
        ok_verify = (verify_result is not None and verify_result['pass']
                     and verify_result['pairs_total'] == pairs_expected
                     and verify_result['pairs_pass'] == pairs_expected)
        checks.append(('재팩+독립 2패스 검증 PASS (%d/%d, 기대 %d)'
                        % (verify_result['pairs_pass'], verify_result['pairs_total'], pairs_expected), ok_verify))
        print('[selftest] verify: %d/%d -> %s' % (verify_result['pairs_pass'], verify_result['pairs_total'],
                                                   'PASS' if ok_verify else 'FAIL(%r)' % verify_result['problems']))

        # ---- ★cache key 결속 ①: report.manifest_sha256 == 디스크 manifest.json 바이트 SHA-256 ----
        manifest_disk_sha = hashlib.sha256(open(os.path.join(out_dir, 'manifest.json'), 'rb').read()).hexdigest()
        ok_mfsha = (verify_result.get('manifest_sha256') == manifest_disk_sha)
        checks.append(('재팩 report.manifest_sha256 == 디스크 manifest.json 바이트 SHA-256(소비자 결속 키)', ok_mfsha))
        print('[selftest] manifest_sha256: report=%r disk=%r -> %s'
              % (verify_result.get('manifest_sha256'), manifest_disk_sha, 'OK' if ok_mfsha else 'FAIL'))

        # ---- ★cache key 결속 ②: --verify-only 경로(재팩 없이 재검증·report append·산출물 불변) ----
        vo_report_path = os.path.join(out_dir, 'verify_report.json')
        vo_bin_path = os.path.join(out_dir, 'experts.bin')
        vo_mf_path = os.path.join(out_dir, 'manifest.json')
        vo_lines_before = len(open(vo_report_path, 'rb').read().splitlines())
        vo_bin_before = hashlib.sha256(open(vo_bin_path, 'rb').read()).hexdigest()
        vo_mf_before = hashlib.sha256(open(vo_mf_path, 'rb').read()).hexdigest()
        vo_result = do_verify_only(model_path, out_dir, profile_id=None,
                                    enforce_reference=False, allow_default_align=True)
        vo_lines_after = open(vo_report_path, 'r', encoding='utf-8').read().splitlines()
        vo_last = json.loads(vo_lines_after[-1]) if vo_lines_after else None
        ok_vo = (vo_result['pass']
                 and vo_result['pairs_pass'] == pairs_expected
                 and vo_result['manifest_sha256'] == vo_mf_before
                 and len(vo_lines_after) == vo_lines_before + 1
                 and vo_last is not None and vo_last.get('manifest_sha256') == vo_mf_before
                 and vo_last.get('pass') is True
                 and hashlib.sha256(open(vo_bin_path, 'rb').read()).hexdigest() == vo_bin_before
                 and hashlib.sha256(open(vo_mf_path, 'rb').read()).hexdigest() == vo_mf_before
                 and not os.path.exists(vo_bin_path + '.partial'))
        checks.append(('--verify-only: 재팩 없이 재검증 PASS + report 1줄 append + 산출물 바이트 불변', ok_vo))
        print('[selftest] --verify-only: pass=%r pairs=%d/%d report %d->%d줄 bin/manifest 불변=%s'
              % (vo_result['pass'], vo_result['pairs_pass'], vo_result['pairs_total'],
                 vo_lines_before, len(vo_lines_after),
                 hashlib.sha256(open(vo_bin_path, 'rb').read()).hexdigest() == vo_bin_before))

        # ---- ★cache key 결속 ③: manifest 사후 변경 → verify-only 재검증이 새 해시를 기록 ----
        # (소비자 게이트는 report.manifest_sha256 != 실제 manifest 해시를 거부한다. 여기서는
        #  생산자 측 계약만 확인: 같은 산출물이라도 manifest 바이트가 바뀌면 키가 따라 바뀐다.)
        vo2_dir = os.path.join(scratch, 'out_vo2')
        shutil.copytree(out_dir, vo2_dir)
        with open(os.path.join(vo2_dir, 'manifest.json'), 'ab') as f:
            f.write(b' ')   # 의미 불변(후행 공백) · 바이트만 변경 → 해시 변경
        vo2_mf_sha = hashlib.sha256(open(os.path.join(vo2_dir, 'manifest.json'), 'rb').read()).hexdigest()
        vo2_result = do_verify_only(model_path, vo2_dir, profile_id=None,
                                     enforce_reference=False, allow_default_align=True)
        ok_vo2 = (vo2_result['pass'] and vo2_result['manifest_sha256'] == vo2_mf_sha
                  and vo2_mf_sha != vo_mf_before)
        checks.append(('--verify-only: manifest 바이트 변경 시 report.manifest_sha256 이 새 바이트를 따름', ok_vo2))
        print('[selftest] --verify-only(manifest 후행 공백 1B): pass=%r sha %s -> %s'
              % (vo2_result['pass'], vo_mf_before[:12], (vo2_result['manifest_sha256'] or '')[:12]))

        # ===================== v1 네거티브 회귀(v1.1~v1.6 전종, v2 스키마로 재지정) =====================
        def _flip_byte(path, offset):
            with open(path, 'r+b') as f:
                f.seek(offset); b = f.read(1); f.seek(offset); f.write(bytes([b[0] ^ 0xFF]))

        def neg_payload_first_byte(o, lo):
            _flip_byte(os.path.join(o, 'experts.bin'), 0)

        def neg_payload_late_middle_byte(o, lo):
            m = _load_manifest_disk(o); rec = m['records'][-1]
            _flip_byte(os.path.join(o, 'experts.bin'), rec['offset'] + rec['payload_bytes'] // 2)

        def neg_padding_byte(o, lo):
            m = _load_manifest_disk(o); rec = m['records'][-1]
            L = [x for x in m['layout']['layers'] if x['layer'] == rec['layer']][0]
            pad_len = L['stride_bytes'] - rec['payload_bytes']
            assert pad_len > 0, '패딩 변조 전제 위반(pad_len=0)'
            _flip_byte(os.path.join(o, 'experts.bin'), rec['offset'] + rec['payload_bytes'])

        def neg_delete_record(o, lo):
            m = _load_manifest_disk(o); m['records'].pop(); _save_manifest_disk(o, m)

        def neg_change_offset(o, lo):
            m = _load_manifest_disk(o)
            m['records'][0]['offset'] += m['layout']['layers'][0]['stride_bytes']; _save_manifest_disk(o, m)

        def neg_empty_records(o, lo):
            m = _load_manifest_disk(o); m['records'] = []; _save_manifest_disk(o, m)

        def neg_bin_truncate(o, lo):
            m = _load_manifest_disk(o); stride = m['layout']['layers'][0]['stride_bytes']
            bp = os.path.join(o, 'experts.bin')
            with open(bp, 'r+b') as f:
                f.truncate(os.path.getsize(bp) - stride // 2)

        def neg_bin_extra_tail(o, lo):
            with open(os.path.join(o, 'experts.bin'), 'ab') as f:
                f.write(b'\x00' * 64)

        def neg_duplicate_record(o, lo):
            m = _load_manifest_disk(o); m['records'][-1] = dict(m['records'][0]); _save_manifest_disk(o, m)

        def neg_swap_records(o, lo):
            m = _load_manifest_disk(o)
            m['records'][0], m['records'][1] = m['records'][1], m['records'][0]; _save_manifest_disk(o, m)

        def neg_payload_bytes_as_stride(o, lo):
            m = _load_manifest_disk(o); rec = m['records'][-1]
            L = [x for x in m['layout']['layers'] if x['layer'] == rec['layer']][0]
            assert rec['payload_bytes'] < L['stride_bytes'], '패딩 우회 전제 위반(pad_len=0)'
            m['records'][-1]['payload_bytes'] = L['stride_bytes']; _save_manifest_disk(o, m)

        def neg_parts_redistribute(o, lo):
            m = _load_manifest_disk(o); parts = m['layout']['layers'][0]['parts']
            parts[0]['part_bytes'] -= 4; parts[1]['part_bytes'] += 4; _save_manifest_disk(o, m)

        def neg_extra_tail_and_totals(o, lo):
            bp = os.path.join(o, 'experts.bin')
            with open(bp, 'ab') as f:
                f.write(b'\x00' * 64)
            m = _load_manifest_disk(o); m['totals']['bin_file_bytes'] = os.path.getsize(bp); _save_manifest_disk(o, m)

        def neg_align_bytes_one(o, lo):
            m = _load_manifest_disk(o); m['layout']['align_bytes'] = 1
            for L in m['layout']['layers']:
                L['stride_bytes'] = L['payload_bytes']
            _save_manifest_disk(o, m)

        def neg_n_expert(o, lo):
            m = _load_manifest_disk(o); m['model']['n_expert'] = m['model']['n_expert'] - 1; _save_manifest_disk(o, m)

        def neg_delete_offset_key(o, lo):
            m = _load_manifest_disk(o); del m['records'][0]['offset']; _save_manifest_disk(o, m)

        def neg_offset_bool(o, lo):
            m = _load_manifest_disk(o); m['records'][0]['offset'] = False; _save_manifest_disk(o, m)

        def neg_n_layer_float(o, lo):
            m = _load_manifest_disk(o); m['model']['n_layer'] = float(m['model']['n_layer']); _save_manifest_disk(o, m)

        def neg_raw_duplicate_key(o, lo):
            path = os.path.join(o, 'manifest.json')
            text = open(path, 'r', encoding='utf-8').read()
            records_pos = text.index('"records"')
            target = '"offset": 0'
            rel = text.index(target, records_pos)
            text = text[:rel] + '"offset": false, "offset": 0' + text[rel + len(target):]
            open(path, 'w', encoding='utf-8').write(text)

        def neg_parts_dims_type(o, lo):
            m = _load_manifest_disk(o)
            m['layout']['layers'][0]['parts'][0]['dims'][0] = float(m['layout']['layers'][0]['parts'][0]['dims'][0])
            _save_manifest_disk(o, m)

        def neg_quant_traits_block_values_float(o, lo):
            m = _load_manifest_disk(o)
            tt = next(iter(m['quant_traits']))
            m['quant_traits'][tt]['block_values'] = float(m['quant_traits'][tt]['block_values'])
            _save_manifest_disk(o, m)

        def neg_mtime_nan(o, lo):
            path = os.path.join(o, 'manifest.json')
            text = open(path, 'r', encoding='utf-8').read()
            key = '"mtime":'
            vs = text.index(key) + len(key); ve = text.index(',', vs)
            text = text[:vs] + ' NaN' + text[ve:]
            open(path, 'w', encoding='utf-8').write(text)

        negative_cases = [
            ('payload 변조(첫 레코드 첫 바이트)', 'payload_first', neg_payload_first_byte),
            ('payload 변조(후반 레코드 중간 바이트)', 'payload_late_mid', neg_payload_late_middle_byte),
            ('패딩 변조', 'padding', neg_padding_byte),
            ('records 항목 삭제', 'delete', neg_delete_record),
            ('records offset 변경', 'offset', neg_change_offset),
            ('records=[] (빈 records가 PASS되면 안 됨)', 'empty', neg_empty_records),
            ('bin truncate', 'bin_truncate', neg_bin_truncate),
            ('bin extra-tail', 'bin_extra_tail', neg_bin_extra_tail),
            ('records 중복 대체', 'dup_record', neg_duplicate_record),
            ('records 순서 교환', 'swap_records', neg_swap_records),
            ('payload_bytes=stride 변조(패딩 우회)', 'payload_as_stride', neg_payload_bytes_as_stride),
            ('parts part_bytes 재분배(합 유지)', 'parts_redistribute', neg_parts_redistribute),
            ('extra-tail+totals 동시 변조', 'extra_tail_and_totals', neg_extra_tail_and_totals),
            ('align_bytes=1(+stride 동시 변조)', 'align_one', neg_align_bytes_one),
            ('model.n_expert 변조', 'n_expert', neg_n_expert),
            ('키 삭제(records[0].offset 제거)', 'delete_key', neg_delete_offset_key),
            ('records[0].offset=false(bool→int 우회)', 'offset_bool', neg_offset_bool, True),
            ('model.n_layer 실수화(int→float 우회)', 'n_layer_float', neg_n_layer_float, True),
            # ★영어화 추종(26-07-30): require_note 는 verify 사유의 부분문자열 매처다
            #   (원='manifest 중복 키' / 'manifest 비표준 JSON 상수').
            ('raw JSON 중복 키 삽입', 'raw_dup_key', neg_raw_duplicate_key, 'manifest duplicate key'),
            ('parts[0].dims[0] 중첩 리스트 내부 실수화', 'parts_dims_type', neg_parts_dims_type, True),
            ('raw JSON sources[0].mtime: NaN', 'mtime_nan', neg_mtime_nan, 'manifest non-standard JSON constant'),
            ('quant_traits.block_values 실수화(int→float 우회)', 'quant_traits_float',
             neg_quant_traits_block_values_float, True),
        ]
        for name, tag, fn, *rest in negative_cases:
            require_note = rest[0] if rest else None
            ok, detail = _run_negative_case(model_path, scratch, tag, fn)
            if require_note:
                # ★영어화 추종(26-07-30): verify 사유 문자열이 영어로 바뀌었으므로
                #   부분문자열 매처도 같은 변경 단위에서 따라간다(원 사유='타입 불일치').
                note = 'type mismatch' if require_note is True else require_note
                if note not in detail:
                    ok = False
            checks.append(('네거티브 — %s' % name, ok))
            print('[selftest] 네거티브[%s]: %s' % (name, 'PASS' if ok else 'FAIL(%s)' % detail))

        # ===================== §5 신설 ①~⑨ =====================
        # ① 2-shard 분산(경계 텐서 타 shard — Mistral layer19 재현): layer1 의 down.weight 를 shard1 로.
        s1a = os.path.join(scratch, 'split-00001-of-00002.gguf')
        s1b = os.path.join(scratch, 'split-00002-of-00002.gguf')
        shard_of = {'blk.1.ffn_down_exps.weight': 1}
        write_synthetic_gguf([s1a, s1b], n_expert=4, moe_layers=(0, 1), schema='separate', bias=True,
                             hidden=8, alignment=32, seed=222, shard_of=shard_of)
        out1 = os.path.join(scratch, 'out_split')
        _, _, vr1 = do_repack(s1a, out1, profile_id=None, force=False, run_verify=True,
                              enforce_reference=False, allow_default_align=True)
        mdl1 = load_model_shards(s1a); lay1 = build_layout(mdl1)
        # 경계 텐서가 실제로 타 shard 에 있는지 확인
        src_idx = {p['source_tensor']: p['source_index'] for L in lay1['layers'] for p in L['parts']}
        ok1 = (vr1 and vr1['pass'] and mdl1['is_split'] and len(mdl1['shards']) == 2
               and src_idx.get('blk.1.ffn_down_exps.weight') == 1
               and src_idx.get('blk.1.ffn_gate_exps.weight') == 0
               and src_idx.get('blk.0.ffn_down_exps.weight') == 0)
        checks.append(('§5① 2-shard 분산(경계 텐서 타 shard) 재팩+검증 PASS', ok1))
        print('[selftest] §5① 2-shard: %s (split=%s shards=%d)' % ('PASS' if ok1 else 'FAIL',
              mdl1.get('is_split'), len(mdl1['shards'])))

        # ①-b 메타 전용 첫 shard(텐서 0개 — 397B shard1 재현). upstream gguf.cpp:756 은
        #     n_tensors==0 이면 데이터 정렬 seek 을 건너뛰므로 live data_start = raw header_end
        #     이고, 물리 파일에는 생산자가 넣은 트레일링 정렬 패딩이 남는다. 도구가 무조건
        #     패딩하던 구산식은 여기서만 어긋나 소비자 seal 을 fail-close 시켰다(26-07-29 실결함).
        s1c = os.path.join(scratch, 'metafirst-00001-of-00002.gguf')
        s1d = os.path.join(scratch, 'metafirst-00002-of-00002.gguf')
        _names_all = ['blk.%d.ffn_%s_exps.%s' % (l, k, s)
                      for l in (0, 1) for s in ('weight', 'bias') for k in ('gate', 'up', 'down')]
        write_synthetic_gguf([s1c, s1d], n_expert=4, moe_layers=(0, 1), schema='separate', bias=True,
                             hidden=8, alignment=32, seed=224,
                             shard_of={n: 1 for n in _names_all})
        h1c = parse_gguf_header(s1c)
        # raw header_end 독립 재측정(파서와 같은 산식을 쓰지 않고 파일에서 직접 되짚는다)
        with open(s1c, 'rb') as _f:
            assert _f.read(4) == b'GGUF'
            (_v,) = r(_f, '<I'); (_nt,) = r(_f, '<Q'); (_nkv,) = r(_f, '<Q')
            for _ in range(_nkv):
                rstr(_f); (_tc,) = r(_f, '<I'); rval_skip_strings(_f, _tc)
            _raw_end = _f.tell()
        _align1b = h1c['alignment']
        _padded1b = (_raw_end + _align1b - 1) // _align1b * _align1b
        out1b = os.path.join(scratch, 'out_metafirst')
        _, _, vr1b = do_repack(s1c, out1b, profile_id=None, force=False, run_verify=True,
                               enforce_reference=False, allow_default_align=True)
        mdl1b = load_model_shards(s1c)
        mf1b = _load_manifest_disk(out1b)
        src0 = mf1b['sources'][0] if isinstance(mf1b.get('sources'), list) and mf1b['sources'] else {}
        ok1b = (_nt == 0                                  # 첫 shard 가 실제로 텐서 0개
                and _padded1b != _raw_end                 # 픽스처 비공허성(패딩이 실제로 존재)
                and h1c['file_bytes'] == _padded1b        # 물리 파일엔 트레일링 패딩이 남아 있음
                and h1c['data_start'] == _raw_end         # 파서가 upstream(무패딩) 을 따름
                and src0.get('data_start') == _raw_end    # manifest sources[0] 도 raw
                and src0.get('bytes') == _padded1b
                and len(mdl1b['shards']) == 2
                and len(mdl1b['shards'][0]['tensors']) == 0
                and len(mdl1b['shards'][1]['tensors']) == 12
                and vr1b and vr1b['pass'])
        checks.append(('§5①-b 메타 전용 첫 shard(텐서 0·data_start=raw header_end·upstream '
                       'gguf.cpp:756 정합) 재팩+검증 PASS', ok1b))
        print('[selftest] §5①-b meta-first: %s (n_tensors0=%d raw=%d padded=%d parsed=%d '
              'manifest_src0=%r file_bytes=%d)'
              % ('PASS' if ok1b else 'FAIL', _nt, _raw_end, _padded1b, h1c['data_start'],
                 src0.get('data_start'), h1c['file_bytes']))

        # ② fused 2-part(gate_up, down·bias 無 — Mistral 스키마)
        s2 = os.path.join(scratch, 'fused.gguf')
        write_synthetic_gguf([s2], n_expert=4, moe_layers=(0, 1), schema='fused', bias=False,
                             hidden=8, alignment=32, seed=333)
        out2 = os.path.join(scratch, 'out_fused')
        _, _, vr2 = do_repack(s2, out2, profile_id=None, force=False, run_verify=True,
                              enforce_reference=False, allow_default_align=True)
        lay2 = build_layout(load_model_shards(s2))
        ok2 = (vr2 and vr2['pass'] and lay2['schema'] == 'fused' and not lay2['has_bias']
               and lay2['record_order'] == ['gate_up.weight', 'down.weight'])
        checks.append(('§5② fused 2-part(gate_up→down·bias無) 재팩+검증 PASS', ok2))
        print('[selftest] §5② fused: %s (order=%r)' % ('PASS' if ok2 else 'FAIL', lay2['record_order']))

        # ③ 층별 혼합 quant(Q4_K / Q6_K — stride 층별 상이). hidden=256(256-block 정합).
        s3 = os.path.join(scratch, 'mixq.gguf')
        write_synthetic_gguf([s3], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=256, alignment=32, seed=444,
                             layer_quant={0: 'Q4_K', 1: 'Q6_K'})
        out3 = os.path.join(scratch, 'out_mixq')
        _, _, vr3 = do_repack(s3, out3, profile_id=None, force=False, run_verify=True,
                              enforce_reference=False, allow_default_align=True)
        lay3 = build_layout(load_model_shards(s3)); plan3 = compute_record_layout(lay3, 4096)
        strides3 = [L['stride_bytes'] for L in lay3['layers']]
        ok3 = (vr3 and vr3['pass'] and lay3['layers'][0]['parts'][0]['type'] == 'Q4_K'
               and lay3['layers'][1]['parts'][0]['type'] == 'Q6_K'
               and lay3['layers'][0]['payload_bytes'] != lay3['layers'][1]['payload_bytes']
               and strides3 == [110592, 163840]
               and lay3['layers'][1]['record_base'] == 442368
               and plan3['bin_bytes'] == 1097728)
        checks.append(('§5③ 층별 혼합 quant(Q4_K/Q6_K·payload 층별 상이) 재팩+검증 PASS', ok3))
        print('[selftest] §5③ mixq: %s (payload L0=%d L1=%d)' % ('PASS' if ok3 else 'FAIL',
              lay3['layers'][0]['payload_bytes'], lay3['layers'][1]['payload_bytes']))

        # ④ n_expert=384(uint16 경계·레코드 수·offset 산술)
        s4 = os.path.join(scratch, 'e384.gguf')
        write_synthetic_gguf([s4], n_expert=384, n_expert_used=8, moe_layers=(0, 1), schema='separate',
                             bias=False, hidden=32, alignment=32, seed=555)
        out4 = os.path.join(scratch, 'out_e384')
        _, _, vr4 = do_repack(s4, out4, profile_id=None, force=False, run_verify=True,
                              enforce_reference=False, allow_default_align=True)
        lay4 = build_layout(load_model_shards(s4)); plan4 = compute_record_layout(lay4, 4096)
        ok4 = (vr4 and vr4['pass'] and lay4['n_expert'] == 384
               and plan4['n_records'] == 2 * 384)
        checks.append(('§5④ n_expert=384(레코드 수 %d·offset 산술) 재팩+검증 PASS' % plan4['n_records'], ok4))
        print('[selftest] §5④ e384: %s (n_records=%d)' % ('PASS' if ok4 else 'FAIL', plan4['n_records']))

        # ⑤ leading dense(moe_layers 부분집합 — 0시작 가정 부재): block_count=4, moe=[1,2,3]
        s5 = os.path.join(scratch, 'lead.gguf')
        write_synthetic_gguf([s5], n_expert=4, moe_layers=(1, 2, 3), block_count=4, schema='separate',
                             bias=True, hidden=8, alignment=32, seed=666)
        out5 = os.path.join(scratch, 'out_lead')
        _, _, vr5 = do_repack(s5, out5, profile_id=None, force=False, run_verify=True,
                              enforce_reference=False, allow_default_align=True)
        lay5 = build_layout(load_model_shards(s5))
        ok5 = (vr5 and vr5['pass'] and lay5['moe_layers'] == [1, 2, 3] and lay5['n_layer'] == 4)
        checks.append(('§5⑤ leading dense(moe_layers=[1,2,3]·n_layer=4·0시작 아님) 재팩+검증 PASS', ok5))
        print('[selftest] §5⑤ leading dense: %s (moe_layers=%r n_layer=%d)'
              % ('PASS' if ok5 else 'FAIL', lay5['moe_layers'], lay5['n_layer']))

        # ⑥ 표 밖 타입 중단
        s6 = os.path.join(scratch, 'offtable.gguf')
        write_synthetic_gguf([s6], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=8, alignment=32, seed=777, off_table_layer=1)
        ok6, msg6 = _expect_abort(lambda: build_layout(load_model_shards(s6)))
        checks.append(('§5⑥ 표 밖 타입 중단(RepackAbort)', ok6))
        print('[selftest] §5⑥ off-table: %s (%s)' % ('PASS' if ok6 else 'FAIL', msg6))

        # ⑦ expert_axis 위반 중단(ne[last]!=n_expert)
        s7 = os.path.join(scratch, 'axis.gguf')
        write_synthetic_gguf([s7], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=8, alignment=32, seed=888, axis_violation=True)
        ok7, msg7 = _expect_abort(lambda: build_layout(load_model_shards(s7)))
        checks.append(('§5⑦ expert_axis 위반 중단(RepackAbort)', ok7 and 'expert_axis' in msg7))
        print('[selftest] §5⑦ axis-violation: %s (%s)' % ('PASS' if (ok7 and 'expert_axis' in msg7) else 'FAIL', msg7))

        # ⑧ gap 초과(정렬 패딩 초과분) 중단
        s8 = os.path.join(scratch, 'gap.gguf')
        write_synthetic_gguf([s8], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=8, alignment=32, seed=999, gap_after='blk.0.ffn_gate_exps.weight')
        ok8, msg8 = _expect_abort(lambda: build_layout(load_model_shards(s8)))
        checks.append(('§5⑧ gap 초과(정렬 패딩 초과) 중단(RepackAbort)', ok8 and 'gap' in msg8))
        print('[selftest] §5⑧ gap-overflow: %s (%s)' % ('PASS' if (ok8 and 'gap' in msg8) else 'FAIL', msg8))

        # ⑨ profile 카탈로그 게이트(미등록 id·digest 불일치·expect 불일치 각 중단)
        # ⑨-a 미등록 id
        ok9a, m9a = _expect_abort(lambda: load_expect_profile('no-such-profile'))
        # ⑨-b digest 불일치(EXPECT_CATALOG 를 일시 오염)
        real_dir = EXPECTS_DIR
        ok9b = False; m9b = ''
        try:
            saved = EXPECT_CATALOG['gpt-oss-120b']
            EXPECT_CATALOG['gpt-oss-120b'] = {'sha256': '0' * 64, 'scope': saved['scope']}
            ok9b, m9b = _expect_abort(lambda: load_expect_profile('gpt-oss-120b'))
        finally:
            EXPECT_CATALOG['gpt-oss-120b'] = saved
        # ⑨-c expect 기대치 불일치(합성 모델을 실제 카탈로그 profile 로 --plan → cross_check_expect 중단)
        #     실 카탈로그 expect(gpt-oss)는 file_bytes 등이 합성과 다르므로 cross_check_expect 가 중단.
        ok9c, m9c = _expect_abort(lambda: _prepare(model_path, out_dir, 'gpt-oss-120b',
                                                    allow_default_align=True, enforce_reference=True))
        ok9 = ok9a and ok9b and ok9c
        checks.append(('§5⑨ profile 카탈로그 게이트(미등록·digest불일치·expect불일치 각 중단)', ok9))
        print('[selftest] §5⑨ catalog gate: %s\n    미등록=%s\n    digest불일치=%s\n    expect불일치=%s'
              % ('PASS' if ok9 else 'FAIL', m9a, m9b, m9c))

        # ===================== 부록A: routed_scope(all|execution) 7종 =====================
        # A1: KV 부재+scope=all=현행 회귀(manifest.model에 routed_scope 미기록·기존 산출물 불변)
        sA1 = os.path.join(scratch, 'scope_a1.gguf')
        write_synthetic_gguf([sA1], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2001)
        outA1 = os.path.join(scratch, 'out_scope_a1')
        _, manifestA1, vrA1 = do_repack(sA1, outA1, profile_id=None, force=False, run_verify=True,
                                        enforce_reference=False, allow_default_align=True, scope='all')
        layA1 = build_layout(load_model_shards(sA1))
        okA1 = (vrA1 and vrA1['pass'] and layA1['moe_layers'] == [0, 1, 2, 3]
                and 'routed_scope' not in manifestA1['model'])
        checks.append(('부록A-1: KV 부재+scope=all — 현행 회귀(manifest.model에 routed_scope 미기록)', okA1))
        print('[selftest] 부록A-1: %s' % ('PASS' if okA1 else 'FAIL'))

        # A2: N=1 제외 정상(마지막 1층 routed 제외·moe_layers·bytes 산술)
        sA2 = os.path.join(scratch, 'scope_a2.gguf')
        write_synthetic_gguf([sA2], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2002,
                             nextn_kv=1)
        outA2 = os.path.join(scratch, 'out_scope_a2')
        _, manifestA2, vrA2 = do_repack(sA2, outA2, profile_id=None, force=False, run_verify=True,
                                        enforce_reference=False, allow_default_align=True, scope='execution')
        layA2 = build_layout(load_model_shards(sA2), scope='execution')
        planA2 = compute_record_layout(layA2, 4096)
        layA2_all = build_layout(load_model_shards(sA2), scope='all')
        planA2_all = compute_record_layout(layA2_all, 4096)
        okA2 = (vrA2 and vrA2['pass'] and layA2['moe_layers'] == [0, 1, 2]
                and manifestA2['model'].get('routed_scope') == 'execution'
                and manifestA2['model']['moe_layers'] == [0, 1, 2]
                and planA2['n_records'] == 3 * layA2['n_expert']
                and planA2['bin_bytes'] * 4 == planA2_all['bin_bytes'] * 3)
        checks.append(('부록A-2: N=1 제외 정상(moe_layers=[0,1,2]·bytes=all의 3/4) 재팩+검증 PASS', okA2))
        print('[selftest] 부록A-2: %s (moe_layers=%r)' % ('PASS' if okA2 else 'FAIL', layA2['moe_layers']))

        # A3: N=2 tail 전량 제외
        sA3 = os.path.join(scratch, 'scope_a3.gguf')
        write_synthetic_gguf([sA3], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2003,
                             nextn_kv=2)
        outA3 = os.path.join(scratch, 'out_scope_a3')
        _, manifestA3, vrA3 = do_repack(sA3, outA3, profile_id=None, force=False, run_verify=True,
                                        enforce_reference=False, allow_default_align=True, scope='execution')
        layA3 = build_layout(load_model_shards(sA3), scope='execution')
        planA3 = compute_record_layout(layA3, 4096)
        layA3_all = build_layout(load_model_shards(sA3), scope='all')
        planA3_all = compute_record_layout(layA3_all, 4096)
        okA3 = (vrA3 and vrA3['pass'] and layA3['moe_layers'] == [0, 1]
                and manifestA3['model'].get('routed_scope') == 'execution'
                and planA3['n_records'] == 2 * layA3['n_expert']
                and planA3['bin_bytes'] * 4 == planA3_all['bin_bytes'] * 2)
        checks.append(('부록A-3: N=2 tail 전량 제외(moe_layers=[0,1]·bytes=all의 2/4) 재팩+검증 PASS', okA3))
        print('[selftest] 부록A-3: %s (moe_layers=%r)' % ('PASS' if okA3 else 'FAIL', layA3['moe_layers']))

        # A4: trunk-only(tail 텐서 부재)에서 execution 정상(제외 대상이 이미 non-routed)
        sA4 = os.path.join(scratch, 'scope_a4.gguf')
        write_synthetic_gguf([sA4], n_expert=4, moe_layers=(0, 1), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2004,
                             nextn_kv=1)
        outA4 = os.path.join(scratch, 'out_scope_a4')
        _, manifestA4, vrA4 = do_repack(sA4, outA4, profile_id=None, force=False, run_verify=True,
                                        enforce_reference=False, allow_default_align=True, scope='execution')
        layA4 = build_layout(load_model_shards(sA4), scope='execution')
        okA4 = (vrA4 and vrA4['pass'] and layA4['moe_layers'] == [0, 1]
                and manifestA4['model'].get('routed_scope') == 'execution')
        checks.append(('부록A-4: trunk-only(제외 대상 이미 non-routed) execution 정상', okA4))
        print('[selftest] 부록A-4: %s (moe_layers=%r)' % ('PASS' if okA4 else 'FAIL', layA4['moe_layers']))

        # A5: KV 부재/0+execution=RepackAbort · 제외 후 routed 0=RepackAbort
        sA5a = os.path.join(scratch, 'scope_a5_absent.gguf')
        write_synthetic_gguf([sA5a], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2005)
        okA5a, msgA5a = _expect_abort(lambda: build_layout(load_model_shards(sA5a), scope='execution'))

        sA5b = os.path.join(scratch, 'scope_a5_zero.gguf')
        write_synthetic_gguf([sA5b], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2006,
                             nextn_kv=0)
        okA5b, msgA5b = _expect_abort(lambda: build_layout(load_model_shards(sA5b), scope='execution'))

        sA5c = os.path.join(scratch, 'scope_a5_allexcl.gguf')
        write_synthetic_gguf([sA5c], n_expert=4, moe_layers=(2, 3), block_count=5,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2007,
                             nextn_kv=3)
        okA5c, msgA5c = _expect_abort(lambda: build_layout(load_model_shards(sA5c), scope='execution'))

        okA5 = okA5a and okA5b and okA5c
        checks.append(('부록A-5: KV 부재/0=RepackAbort · 제외 후 routed 0=RepackAbort', okA5))
        print('[selftest] 부록A-5: 부재=%s 0=%s 전량제외=%s'
              % ('PASS' if okA5a else 'FAIL(%s)' % msgA5a, 'PASS' if okA5b else 'FAIL(%s)' % msgA5b,
                 'PASS' if okA5c else 'FAIL(%s)' % msgA5c))

        # A6: shard 간 nextn_predict_layers 값 충돌 = RepackAbort
        sA6a = os.path.join(scratch, 'scope_a6-00001-of-00002.gguf')
        sA6b = os.path.join(scratch, 'scope_a6-00002-of-00002.gguf')
        write_synthetic_gguf([sA6a, sA6b], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2008,
                             nextn_kv={0: 1, 1: 2})
        okA6, msgA6 = _expect_abort(lambda: build_layout(load_model_shards(sA6a), scope='execution'))
        # ★영어화 추종(26-07-30): RepackAbort 사유가 영어로 바뀌어 매처도 동반 수정(원='충돌').
        checks.append(('부록A-6: shard 간 nextn_predict_layers 값 충돌 -> RepackAbort',
                       okA6 and 'value conflict between shards' in msgA6))
        print('[selftest] 부록A-6: %s (%s)'
              % ('PASS' if (okA6 and 'value conflict between shards' in msgA6) else 'FAIL', msgA6))

        # A7: expect routed_scope 불일치(CLI scope != expect routed_scope) = RepackAbort
        sA7 = os.path.join(scratch, 'scope_a7.gguf')
        write_synthetic_gguf([sA7], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', bias=True, hidden=8, alignment=32, seed=2009)
        mdlA7 = load_model_shards(sA7)
        layA7 = build_layout(mdlA7, scope='all')
        planA7 = compute_record_layout(layA7, 4096)
        expectA7 = {
            'arch': layA7['arch'], 'n_layer': layA7['n_layer'], 'n_expert': layA7['n_expert'],
            'n_expert_used': layA7['n_expert_used'], 'routed_tensors': layA7['n_routed'],
            'expert_bytes_total': layA7['n_expert'] * sum(L['payload_bytes'] for L in layA7['layers']),
            'sources': [{'file_bytes': h['file_bytes'], 'data_start': h['data_start']} for h in mdlA7['shards']],
            'routed_scope': 'execution',   # CLI 는 all 로 호출 -> 불일치
        }
        okA7, msgA7 = _expect_abort(lambda: cross_check_expect(mdlA7, layA7, planA7, expectA7, scope='all'))
        checks.append(('부록A-7: CLI scope != expect routed_scope -> RepackAbort', okA7 and 'routed_scope' in msgA7))
        print('[selftest] 부록A-7: %s (%s)' % ('PASS' if (okA7 and 'routed_scope' in msgA7) else 'FAIL', msgA7))

        # A8(Codex 교차 [MODIFY] 반영): 제외될 tail 층만 trunk와 다른 routed part 집합(trunk=
        # separate·tail=fused) — "제외가 층별 schema 검사보다 먼저"라는 구현 순서를 실제로 행사
        # (A1~A7 은 전 층 schema 동일이라 순서 무관이었음 — 그 증거 누락이 차단 사유였음).
        # scope=execution: tail이 schema 검사 전에 빠져 재팩+검증 PASS. scope=all: 혼재 그대로
        # 걸려 RepackAbort(기존 계약).
        sA8 = os.path.join(scratch, 'scope_a8.gguf')
        write_synthetic_gguf([sA8], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4,
                             schema='separate', schema_by_layer={3: 'fused'}, bias=True,
                             hidden=8, alignment=32, seed=2010, nextn_kv=1)
        outA8 = os.path.join(scratch, 'out_scope_a8')
        _, manifestA8, vrA8 = do_repack(sA8, outA8, profile_id=None, force=False, run_verify=True,
                                        enforce_reference=False, allow_default_align=True, scope='execution')
        layA8 = build_layout(load_model_shards(sA8), scope='execution')
        okA8_exec = (vrA8 and vrA8['pass'] and layA8['moe_layers'] == [0, 1, 2]
                     and layA8['schema'] == 'separate'
                     and manifestA8['model'].get('routed_scope') == 'execution')
        okA8_all, msgA8_all = _expect_abort(lambda: build_layout(load_model_shards(sA8), scope='all'))
        # ★영어화 추종(26-07-30): 원 매처='파트 집합 불일치'.
        okA8 = okA8_exec and okA8_all and 'part set mismatch' in msgA8_all
        checks.append(('부록A-8: tail-only schema 이질(trunk=separate·tail=fused) — '
                        'execution=제외 후 PASS(순서 행사) · all=파트 집합 불일치 RepackAbort', okA8))
        print('[selftest] 부록A-8: execution=%s(moe_layers=%r schema=%s) all=%s(%s)'
              % ('PASS' if okA8_exec else 'FAIL', layA8['moe_layers'], layA8['schema'],
                 'PASS' if okA8_all else 'FAIL', msgA8_all))

        # ---- argparse 실호출 조합 ----
        contract = _check_argparse_contract(os.path.abspath(__file__), model_path, os.path.join(scratch, 'out_argp'))
        for name, ok, extra in contract:
            checks.append(('argparse: %s' % name, ok))
            print('[selftest] argparse [%s] -> %s (%s)' % (name, 'OK' if ok else 'FAIL', extra))

        # ================= OPEN_ARCH A축(OPEN_ARCH_DESIGN.md v0.2 §1) 회귀 4분리 =================
        # ⓐ 등록 경로 9 expect 전부 불변 / ⓑ 템플릿 semantic replay / ⓒ inventory_sha256 결정론 /
        # ⓓ 음성 mutant / (+ⓔ fail-close 전건표 · ⓕ 기본 CLI 불변)
        # ★실물 GGUF 헤더 대조(등록 6 expect·5 실모델)는 이 selftest 밖에서 실행한다 — selftest 는
        #   D:\ 실물 부재 환경에서도 도는 이식 가능 관문이어야 하므로 합성 픽스처로만 폐합한다.
        def _tpl_fixture(fname, **kw):
            p = os.path.join(scratch, fname)
            write_synthetic_gguf([p], **kw)
            return p

        GPTOSS = dict(arch='gpt-oss', n_expert=4, n_expert_used=2, moe_layers=(0, 1, 2), block_count=3,
                      schema='separate', bias=True, hidden=8, alignment=32, seed=3101)
        QWEN_NEXTN = dict(arch='qwen35moe', n_expert=4, n_expert_used=2, moe_layers=(0, 1, 2, 3), block_count=4,
                          schema='separate', bias=False, hidden=8, alignment=32, seed=3102, nextn_kv=1,
                          extra_tensors=({'name': 'blk.3.nextn.eh_proj.weight', 'dims': [8, 8]},))
        QWEN_PLAIN = dict(arch='qwen35moe', n_expert=4, n_expert_used=2, moe_layers=(0, 1, 2), block_count=3,
                          schema='separate', bias=False, hidden=8, alignment=32, seed=3103)
        DEEPSEEK = dict(arch='deepseek2', n_expert=4, n_expert_used=2, moe_layers=(1, 2, 3), block_count=4,
                        schema='separate', bias=False, hidden=8, alignment=32, seed=3104,
                        extra_kv_by_shard={0: [('deepseek2.leading_dense_block_count', T_U32, 1)]})

        # ---- ⓐ 등록 경로 9 expect 전부 불변(파일 SHA == 카탈로그 승인 digest·전량 로드 PASS) ----
        reg_details = []
        ok_reg = (len(EXPECT_CATALOG) == 9)
        if not ok_reg:
            reg_details.append('카탈로그 항목 수=%d(기대 9)' % len(EXPECT_CATALOG))
        for pid in sorted(EXPECT_CATALOG):
            path = os.path.join(EXPECTS_DIR, '%s.expect.json' % pid)
            try:
                disk_sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            except Exception as e:
                ok_reg = False; reg_details.append('%s: 파일 없음/읽기 실패 %r' % (pid, e)); continue
            if disk_sha != EXPECT_CATALOG[pid]['sha256']:
                ok_reg = False; reg_details.append('%s: 파일 SHA != 카탈로그 승인 digest' % pid); continue
            try:
                exp, sha = load_expect_profile(pid)
            except RepackAbort as e:
                ok_reg = False; reg_details.append('%s: load_expect_profile 실패 %s' % (pid, e)); continue
            if sha != disk_sha or exp.get('profile_id') != pid \
                    or exp.get('routed_scope', 'all') != EXPECT_CATALOG[pid]['scope']:
                ok_reg = False; reg_details.append('%s: 3자 동일성 위반' % pid)
        checks.append(('OPEN_ARCH-ⓐ 등록 경로 9 expect 전부 불변(파일 SHA=카탈로그 승인 digest·9/9 로드 PASS)', ok_reg))
        print('[selftest] OPEN_ARCH-ⓐ 등록 expect 9종: %s %s'
              % ('PASS' if ok_reg else 'FAIL', reg_details if reg_details else ''))

        # ---- ⓑ 템플릿 semantic replay(합성 5형상 — 유도 결과가 layout 재도출과 의미 동일) ----
        replay_specs = [
            ('gpt-oss(전층 3W+3B)', 'tpl_gptoss.gguf', GPTOSS, None, 'gpt-oss@1', 'all', 3, 18),
            ('qwen35moe(nextn=1·기본 scope)', 'tpl_qwen_nextn.gguf', QWEN_NEXTN, None, 'qwen35moe@1', 'execution', 3, 9),
            ('qwen35moe(nextn=1·scope=all 명시)', 'tpl_qwen_nextn_all.gguf', QWEN_NEXTN, 'all', 'qwen35moe@1', 'all', 4, 12),
            ('qwen35moe(nextn 부재)', 'tpl_qwen_plain.gguf', QWEN_PLAIN, None, 'qwen35moe@1', 'all', 3, 9),
            ('deepseek2(leading_dense=1)', 'tpl_deepseek.gguf', DEEPSEEK, None, 'deepseek2@1', 'all', 3, 9),
        ]
        ok_replay = True
        replay_notes = []
        replay_digest = {}
        for label, fname, kw, req_scope, want_from, want_scope, want_layers, want_routed in replay_specs:
            try:
                p = _tpl_fixture(fname, **kw)
                mdl = load_model_shards(p)
                drv = derive_arch_template(mdl, requested_scope=req_scope)
                lay = build_layout(mdl, scope=drv['scope'])
                pl = compute_record_layout(lay, 4096)
                exp, raw, sha = build_derived_expect(mdl, lay, drv)
                cross_check_expect(mdl, lay, pl, exp, scope=drv['scope'])   # 등록 경로와 같은 대조기
                good = (drv['derived_from'] == want_from and drv['scope'] == want_scope
                        and len(drv['layers']) == want_layers and drv['routed_tensors'] == want_routed
                        and exp['routed_tensors'] == want_routed
                        and exp['expert_bytes_total'] == lay['n_expert'] * sum(L['payload_bytes'] for L in lay['layers'])
                        and exp['derived_from'] == want_from and exp['inventory_sha256'] == drv['inventory_sha256']
                        and exp['expect_schema_version'] == EXPECT_SCHEMA_VERSION
                        and exp['sources'] == [{'file_bytes': h['file_bytes'], 'data_start': h['data_start']}
                                               for h in mdl['shards']]
                        and sha == hashlib.sha256(raw).hexdigest())
                replay_digest[label] = drv['inventory_sha256']
                if not good:
                    ok_replay = False
                    replay_notes.append('%s: from=%s scope=%s layers=%d routed=%d'
                                        % (label, drv['derived_from'], drv['scope'], len(drv['layers']),
                                           drv['routed_tensors']))
            except RepackAbort as e:
                ok_replay = False
                replay_notes.append('%s: RepackAbort %s' % (label, e))
        checks.append(('OPEN_ARCH-ⓑ 템플릿 semantic replay(합성 5형상: gpt-oss·qwen35moe all/execution/nextn無·deepseek2) '
                       '— 유도 expect 가 layout 재도출과 의미 동일', ok_replay))
        print('[selftest] OPEN_ARCH-ⓑ replay: %s %s' % ('PASS' if ok_replay else 'FAIL', replay_notes if replay_notes else ''))
        for label, dg in replay_digest.items():
            print('    %-38s inventory_sha256=%s' % (label, dg))

        # ---- ⓑ2 템플릿 재팩 E2E(합성 gpt-oss 형상): derived.expect.json 기록 + verify 독립 재유도 ----
        tpl_model = os.path.join(scratch, 'tpl_gptoss.gguf')
        tpl_out = os.path.join(scratch, 'out_tpl_gptoss')
        _, tpl_manifest, tpl_vr = do_repack(tpl_model, tpl_out, profile_id=None, force=False, run_verify=True,
                                            enforce_reference=False, allow_default_align=True, arch_template=True)
        tpl_expect_path = os.path.join(tpl_out, DERIVED_EXPECT_FILENAME)
        tpl_raw = open(tpl_expect_path, 'rb').read() if os.path.exists(tpl_expect_path) else b''
        tpl_expect = json.loads(tpl_raw.decode('utf-8')) if tpl_raw else {}
        tpl_vo = do_verify_only(tpl_model, tpl_out, profile_id=None, allow_default_align=True, arch_template=True)
        ok_tpl_e2e = (tpl_vr and tpl_vr['pass'] and tpl_vo['pass']
                      and os.path.exists(tpl_expect_path)
                      and not os.path.exists(os.path.join(EXPECTS_DIR, DERIVED_EXPECT_FILENAME))
                      and tpl_manifest['reference_lock']['profile_id'] == 'arch-template:gpt-oss@1'
                      and tpl_manifest['reference_lock']['expect_sha256'] == hashlib.sha256(tpl_raw).hexdigest()
                      and tpl_expect.get('derived_from') == 'gpt-oss@1'
                      and len(tpl_expect.get('inventory_sha256', '')) == 64)
        checks.append(('OPEN_ARCH-ⓑ2 템플릿 재팩 E2E — derived.expect.json 을 <out> 에 기록(번들 expects_dir 무변)·'
                       'reference_lock=arch-template·verify/--verify-only 가 독립 재유도로 PASS', ok_tpl_e2e))
        print('[selftest] OPEN_ARCH-ⓑ2 E2E: %s (verify=%s verify_only=%s lock=%r)'
              % ('PASS' if ok_tpl_e2e else 'FAIL', tpl_vr and tpl_vr['pass'], tpl_vo['pass'],
                 tpl_manifest['reference_lock']['profile_id']))

        # ---- ⓑ3 산출물 변조 → 템플릿 verify 가 fail-close(derived expect 재해시·바이트 동일성) ----
        tpl_neg = []
        vo_dir2 = os.path.join(scratch, 'out_tpl_tamper')
        shutil.copytree(tpl_out, vo_dir2)
        with open(os.path.join(vo_dir2, DERIVED_EXPECT_FILENAME), 'ab') as f:
            f.write(b' ')          # 의미 불변·바이트만 변경 → 재해시 불일치여야 한다
        neg1 = verify_repack(tpl_model, vo_dir2, profile_id=None, enforce_reference=True,
                             allow_default_align=True, arch_template=True)
        tpl_neg.append(('derived.expect.json 1바이트 변조', neg1['pass'] is False))
        vo_dir3 = os.path.join(scratch, 'out_tpl_nolock')
        shutil.copytree(tpl_out, vo_dir3)
        _mf3 = _load_manifest_disk(vo_dir3)
        _mf3['reference_lock']['profile_id'] = 'arch-template:gpt-oss@999'
        _save_manifest_disk(vo_dir3, _mf3)
        neg2 = verify_repack(tpl_model, vo_dir3, profile_id=None, enforce_reference=True,
                             allow_default_align=True, arch_template=True)
        tpl_neg.append(('reference_lock 템플릿 버전 위조', neg2['pass'] is False))
        vo_dir4 = os.path.join(scratch, 'out_tpl_noexpect')
        shutil.copytree(tpl_out, vo_dir4)
        os.remove(os.path.join(vo_dir4, DERIVED_EXPECT_FILENAME))
        neg3 = verify_repack(tpl_model, vo_dir4, profile_id=None, enforce_reference=True,
                             allow_default_align=True, arch_template=True)
        tpl_neg.append(('derived.expect.json 삭제', neg3['pass'] is False))
        ok_tpl_neg = all(ok for _, ok in tpl_neg)
        checks.append(('OPEN_ARCH-ⓑ3 템플릿 verify 네거티브 3종(derived expect 변조·lock 위조·expect 삭제) 전부 FAIL 판정', ok_tpl_neg))
        print('[selftest] OPEN_ARCH-ⓑ3 verify 네거티브: %s %r' % ('PASS' if ok_tpl_neg else 'FAIL', tpl_neg))

        # ---- ⓒ inventory_sha256 결정론(같은 입력 2회 동일) + 집합 민감도 ----
        det_a = derive_arch_template(load_model_shards(tpl_model))['inventory_sha256']
        det_b = derive_arch_template(load_model_shards(tpl_model))['inventory_sha256']
        sens_kw = dict(GPTOSS); sens_kw['layer_quant'] = {0: 'F16', 1: 'F32', 2: 'F32'}
        sens_p = _tpl_fixture('tpl_gptoss_f16.gguf', **sens_kw)
        det_c = derive_arch_template(load_model_shards(sens_p))['inventory_sha256']
        ok_det = (det_a == det_b and len(det_a) == 64 and det_a != det_c)
        checks.append(('OPEN_ARCH-ⓒ inventory_sha256 결정론(동일 입력 2회 동일) + 집합 민감도(텐서 1개 type 변경=digest 변경)', ok_det))
        print('[selftest] OPEN_ARCH-ⓒ digest: run1=%s run2=%s variant=%s -> %s'
              % (det_a[:16], det_b[:16], det_c[:16], 'PASS' if ok_det else 'FAIL'))

        # ---- ⓒ-2 실물 router 계열 동반(★26-08-03 신설 — 교차검증 B④ 처방) ----
        # 구 합성 GPTOSS 는 routed 3W+3B 만 만들고 **실제 router 텐서를 생성하지 않았다**. 그래서
        # 실물 gpt-oss(20B/120B)에 전 층 존재하는 `ffn_gate_inp.{weight,bias}` 가 허용표 밖으로
        # 새어 fail-close 하는 것을 selftest 가 잡을 수 없었다(M5 preflight 가 실물에서 처음 적발).
        # 이 픽스처는 실물 형상을 합성에 반영하고, **router 동반이 routed 수·inventory digest·
        # expert_bytes_total 을 바꾸지 않는다**(= trunk 상주·재팩 무관)는 것을 동시에 못 박는다.
        _router_extra = tuple(
            t for L in (0, 1, 2) for t in (
                {'name': 'blk.%d.ffn_gate_inp.weight' % L, 'dims': [8, 4], 'type': 'F32'},
                {'name': 'blk.%d.ffn_gate_inp.bias' % L,   'dims': [4],    'type': 'F32'},
            ))
        rtr_p = _tpl_fixture('tpl_gptoss_router.gguf', **dict(GPTOSS, extra_tensors=_router_extra))
        base_der = derive_arch_template(load_model_shards(tpl_model))
        try:
            rtr_der = derive_arch_template(load_model_shards(rtr_p))
            rtr_err = None
        except RepackAbort as e:
            rtr_der, rtr_err = None, str(e)
        ok_rtr = (rtr_der is not None
                  and rtr_der['routed_tensors'] == base_der['routed_tensors']
                  and rtr_der['inventory_sha256'] == base_der['inventory_sha256']
                  and rtr_der['expert_bytes_total'] == base_der['expert_bytes_total'])
        checks.append(('OPEN_ARCH-ⓒ2 실물 router 동반(전 층 ffn_gate_inp.weight+bias) 유도 성공 + '
                       'routed 수·inventory digest·expert_bytes_total 추가 전과 동일(trunk 상주·재팩 무관)', ok_rtr))
        print('[selftest] OPEN_ARCH-ⓒ2 router 동반: routed %s->%s digest %s->%s %s'
              % (base_der['routed_tensors'], (rtr_der or {}).get('routed_tensors'),
                 base_der['inventory_sha256'][:16], ((rtr_der or {}).get('inventory_sha256') or '(abort)')[:16],
                 'PASS' if ok_rtr else ('FAIL - ' + (rtr_err or 'mismatch'))))

        # ---- ⓓ 음성 mutant(누락·추가·개명·bias 뒤집기·NextN 범위·shard 충돌) ----
        mut_split = [os.path.join(scratch, 'tpl_mut_conflict-%05d-of-00002.gguf' % i) for i in (1, 2)]
        write_synthetic_gguf(mut_split, arch='qwen35moe', n_expert=4, n_expert_used=2, moe_layers=(0, 1, 2),
                             block_count=3, schema='separate', bias=False, hidden=8, alignment=32, seed=3110,
                             shard_of={'blk.2.ffn_down_exps.weight': 1},
                             extra_kv_by_shard={1: [('qwen35moe.expert_count', T_U32, 8)]})
        mutants = [
            ('① 누락(gpt-oss blk.1.ffn_up_exps.weight 삭제)', 'mut_missing.gguf',
             dict(GPTOSS, drop_tensors=('blk.1.ffn_up_exps.weight',)), None, 'part-missing'),
            ('② 추가(separate 층에 gate_up 추가)', 'mut_extra.gguf',
             dict(GPTOSS, extra_tensors=({'name': 'blk.1.ffn_gate_up_exps.weight', 'dims': [8, 8, 4]},)), None, 'part-extra'),
            ('③ 개명(down.weight -> ffn_downx_exps.weight)', 'mut_rename.gguf',
             dict(GPTOSS, rename_tensors={'blk.1.ffn_down_exps.weight': 'blk.1.ffn_downx_exps.weight'}), None,
             'expert-like-unclassified'),
            ('④-a bias 뒤집기(gpt-oss 는 bias 필수인데 부재)', 'mut_bias_off.gguf',
             dict(GPTOSS, bias=False), None, 'part-bias'),
            ('④-b bias 뒤집기(qwen35moe 는 bias 금지인데 존재)', 'mut_bias_on.gguf',
             dict(QWEN_PLAIN, bias=True), None, 'part-bias'),
            ('⑤ NextN 범위(nextn=block_count)', 'mut_nextn.gguf',
             dict(QWEN_NEXTN, nextn_kv=4), None, 'nextn-range'),
            # ★D6 수리 음성 표본 3종(26-08-02) — 허용표를 실측 관측형으로 축소했음을 행사한다.
            # 구 허용표에서는 ⑦⑧ 이 known resident 로 조용히 통과했고, ⑨ 는 _EXPERT_LIKE_RE 가
            # gate_inp 를 몰라 expert-like 로도 잡히지 않았다.
            ('⑦ 미관측 shexp bias(blk.0.ffn_gate_shexp.bias)', 'mut_shexp_bias.gguf',
             dict(GPTOSS, extra_tensors=({'name': 'blk.0.ffn_gate_shexp.bias', 'dims': [8]},)), None,
             'expert-like-unclassified'),
            ('⑧ 미관측 fused shexp(blk.0.ffn_gate_up_shexp.weight)', 'mut_shexp_gateup.gguf',
             dict(GPTOSS, extra_tensors=({'name': 'blk.0.ffn_gate_up_shexp.weight', 'dims': [8, 8]},)), None,
             'expert-like-unclassified'),
            ('⑨ router 개명(ffn_gate_inp.weight -> ffn_gate_inp2.weight)', 'mut_gate_inp.gguf',
             dict(GPTOSS, extra_tensors=({'name': 'blk.0.ffn_gate_inp2.weight', 'dims': [8, 4]},)), None,
             'expert-like-unclassified'),
        ]
        mut_results = []
        covered_codes = set()      # ⓔ 코드표 커버리지와 합집합으로 폐합(ⓓ 가 행사한 코드 포함)
        for label, fname, kw, req_scope, want_code in mutants:
            p = _tpl_fixture(fname, **kw)
            ok_m, msg_m = _expect_abort(lambda _p=p, _s=req_scope: derive_arch_template(load_model_shards(_p),
                                                                                        requested_scope=_s))
            ok_m = ok_m and ('[template:%s]' % want_code) in msg_m
            if ok_m:
                covered_codes.add(want_code)
            mut_results.append((label, ok_m, msg_m))
        ok_m6, msg_m6 = _expect_abort(lambda: derive_arch_template(load_model_shards(mut_split[0])))
        ok_m6 = ok_m6 and '[template:kv-shard-conflict]' in msg_m6
        if ok_m6:
            covered_codes.add('kv-shard-conflict')
        mut_results.append(('⑥ shard 충돌(shard1 이 expert_count 를 다르게 선언)', ok_m6, msg_m6))
        ok_mut = all(ok for _, ok, _ in mut_results)
        checks.append(('OPEN_ARCH-ⓓ 음성 mutant 9종(누락·추가·개명·bias 유무 뒤집기·NextN 범위·shard 충돌·'
                       '미관측 shexp bias/fused shexp·router 개명) — 각각 지정 사유 코드로 거부', ok_mut))
        print('[selftest] OPEN_ARCH-ⓓ mutant: %s' % ('PASS' if ok_mut else 'FAIL'))
        for label, ok_m, msg_m in mut_results:
            print('    [%s] %-46s %s' % ('PASS' if ok_m else 'FAIL', label, msg_m[:110]))

        # ---- ⓔ fail-close 전건표(TEMPLATE_FAIL_CODES 전 코드가 각각 **정확한**
        #      `[template:<code>]` 문자열로 거부되는지. ⓓ 음성 mutant 가 이미 행사한 코드와
        #      합집합으로 폐합하고, 커버리지 23/23 을 관문화한다.
        #      ★양성 대조(trace 비활성 정상 유도·loader 선행 차단)는 ⓔ2 로 분리한다 —
        #        거부 표본 수에 양성 대조를 섞지 않는다.) ----
        shard_naming_path = os.path.join(scratch, 'tpl_named_wrong.bin')
        shutil.copyfile(tpl_model, shard_naming_path)
        fc_cases = [
            ('arch 미지원(템플릿 부재)', 'fc_arch.gguf', dict(GPTOSS, arch='moetest'), None, 'arch-unsupported'),
            ('구조 KV 부재', 'fc_kvmiss.gguf', dict(GPTOSS, omit_kv=('gpt-oss.expert_used_count',)), None, 'kv-missing'),
            ('구조 KV 형 불일치(block_count 를 F32 로)', 'fc_kvtype.gguf',
             dict(GPTOSS, omit_kv=('gpt-oss.block_count',),
                  extra_kv_by_shard={0: [('gpt-oss.block_count', T_F32, 3.0)]}), None, 'kv-type'),
            # ★신설(26-08-02): 구 전건표에서 유일하게 표본이 없던 정의역 붕괴 코드.
            ('구조 KV 정의역 붕괴(block_count=0)', 'fc_kvrange.gguf',
             dict(GPTOSS, omit_kv=('gpt-oss.block_count',),
                  extra_kv_by_shard={0: [('gpt-oss.block_count', T_U32, 0)]}), None, 'kv-range'),
            ('비표준 shard naming(.bin)', shard_naming_path, None, None, 'shard-naming'),
            ('arch 공식 층집합 != 실제', 'fc_layerset.gguf', dict(GPTOSS, block_count=4), None, 'layer-set-mismatch'),
            ('fused-separate 불일치', 'fc_schema.gguf', dict(GPTOSS, schema='fused'), None, 'part-schema'),
            ('부분 bias', 'fc_partbias.gguf', dict(GPTOSS, drop_tensors=('blk.1.ffn_up_exps.bias',)), None, 'part-bias'),
            ('expert axis 비말단', 'fc_axis.gguf', dict(GPTOSS, axis_violation=True), None,
             'expert-axis-not-last'),
            ('dims[-1] != n_expert', 'fc_dims.gguf', dict(GPTOSS, hidden=16, meta_expert_count=8), None,
             'dims-last-not-n-expert'),
            ('표 밖 quant', 'fc_offtable.gguf', dict(GPTOSS, off_table_layer=1), None, 'quant-off-table'),
            ('산술 불폐합(ne0 % block_values != 0)', 'fc_arith.gguf',
             dict(GPTOSS, layer_quant={0: 'Q4_K', 1: 'Q4_K', 2: 'Q4_K'}, allow_block_misalign=True), None,
             'arithmetic-closure'),
            ('NextN 범위(scope=execution 인데 KV 부재)', 'fc_nextn2.gguf', dict(QWEN_PLAIN), 'execution', 'nextn-range'),
            ('leading-dense 범위', 'fc_lead.gguf',
             dict(DEEPSEEK, extra_kv_by_shard={0: [('deepseek2.leading_dense_block_count', T_U32, 4)]}), None,
             'leading-dense-range'),
            ('모호 MTP(표식 텐서만 있고 KV 부재)', 'fc_mtp.gguf',
             dict(QWEN_PLAIN, extra_tensors=({'name': 'blk.2.nextn.eh_proj.weight', 'dims': [8, 8]},)), None,
             'mtp-ambiguous'),
            ('n_expert 범위(0)', 'fc_nexp.gguf', dict(GPTOSS, meta_expert_count=0), None, 'n-expert-range'),
            ('n_expert_used 범위(0)', 'fc_nused.gguf', dict(GPTOSS, n_expert_used=0), None, 'n-expert-used-range'),
        ]
        fc_results = []
        for label, target, kw, req_scope, want in fc_cases:
            p = target if kw is None else _tpl_fixture(target, **kw)
            ok_c, msg_c = _expect_abort(lambda _p=p, _s=req_scope: derive_arch_template(load_model_shards(_p),
                                                                                        requested_scope=_s))
            ok_c = ok_c and ('[template:%s]' % want) in msg_c
            if ok_c:
                covered_codes.add(want)
            fc_results.append((label, ok_c, msg_c))
        # tensor-name-duplicate: 정상 경로에서는 load_model_shards 의 전역 유일성 assert 가 선행
        # 차단하므로(그 자체는 ⓔ2 양성 대조에서 검사) 템플릿 (2) 방어 이중화 계층이 **그 코드로**
        # 거부하는지는 유일성 assert 만 우회한 모델 dict 로 행사한다(selftest 전용 조립 —
        # 프로덕션 경로·검사 순서는 무변경).
        dup_p = _tpl_fixture('fc_dup.gguf', **dict(GPTOSS, duplicate_tensor='blk.0.ffn_gate_exps.weight'))

        def _shards_no_uniqueness_assert(path):
            h = parse_gguf_header(path)
            h['source_index'] = 0
            return {'shards': [h], 'arch': h['meta']['general.architecture'], 'meta': dict(h['meta']),
                    'is_split': False, 'split_notes': [], 'model_path': path}

        ok_dup, msg_dup = _expect_abort(lambda: derive_arch_template(_shards_no_uniqueness_assert(dup_p)))
        ok_dup = ok_dup and '[template:tensor-name-duplicate]' in msg_dup
        if ok_dup:
            covered_codes.add('tensor-name-duplicate')
        fc_results.append(('tensor-name 중복(템플릿 방어 이중화 계층)', ok_dup, msg_dup))
        # trace 게이트(엔진과 동형 — env 로 판정. 검사 후 원상복구)
        trace_p = _tpl_fixture('fc_trace.gguf', **dict(GPTOSS, n_expert=256, seed=3120))
        _saved_trace = os.environ.get(TRACE_ENV_VAR)
        try:
            os.environ[TRACE_ENV_VAR] = '1'
            ok_tr2, msg_tr2 = _expect_abort(lambda: derive_arch_template(load_model_shards(trace_p)))
            ok_tr2 = ok_tr2 and '[template:trace-gate]' in msg_tr2
        finally:
            if _saved_trace is None:
                os.environ.pop(TRACE_ENV_VAR, None)
            else:
                os.environ[TRACE_ENV_VAR] = _saved_trace
        if ok_tr2:
            covered_codes.add('trace-gate')
        fc_results.append(('live trace 활성 ∧ n_expert>255', ok_tr2, msg_tr2))
        missing_codes = sorted(set(TEMPLATE_FAIL_CODES) - covered_codes)
        offtable_codes = sorted(covered_codes - set(TEMPLATE_FAIL_CODES))
        ok_fc = all(ok for _, ok, _ in fc_results) and not missing_codes and not offtable_codes
        checks.append(('OPEN_ARCH-ⓔ fail-close 전건표 %d종 각각 지정 사유로 거부(§1 목록 전항 — '
                       'ⓓ+ⓔ 합집합 커버리지 %d/%d)'
                       % (len(TEMPLATE_FAIL_CODES), len(covered_codes & set(TEMPLATE_FAIL_CODES)),
                          len(TEMPLATE_FAIL_CODES)), ok_fc))
        print('[selftest] OPEN_ARCH-ⓔ fail-close 전건표: %s (ⓔ 거부 표본 %d건 · 코드 커버리지 %d/%d%s)'
              % ('PASS' if ok_fc else 'FAIL', len(fc_results),
                 len(covered_codes & set(TEMPLATE_FAIL_CODES)), len(TEMPLATE_FAIL_CODES),
                 '' if not (missing_codes or offtable_codes)
                 else ' 미행사=%r 표밖=%r' % (missing_codes, offtable_codes)))
        for label, ok_c, msg_c in fc_results:
            print('    [%s] %-42s %s' % ('PASS' if ok_c else 'FAIL', label, msg_c[:100]))

        # ---- ⓔ2 양성 대조(거부 집계와 **분리** — trace-off 정상 유도가 "19종 거부" 집계에
        #      혼입돼 있던 것을 분리한다) ----
        # ★r2 수리(26-08-02): "trace 비활성" 을 **명시적으로 만든다** — 외부 셸이
        # MOE_P1_TRACE=1 인 채 --selftest 를 돌리면(정상 trace 작업 셸에서 실 도달) 앞선 ⓔ
        # trace 검사가 finally 로 그 "1" 을 원상복구해 이 양성 대조가 [template:trace-gate] 로
        # 중단, 64/64 가 환경 의존적으로 깨졌다. 검사 구간만 unset 하고 finally 로 복원한다.
        pos_controls = []
        _saved_trace2 = os.environ.get(TRACE_ENV_VAR)
        try:
            os.environ.pop(TRACE_ENV_VAR, None)
            ok_tr_off, msg_tr_off = True, ''
            try:
                drv_off = derive_arch_template(load_model_shards(trace_p))
                ok_tr_off = (drv_off['n_expert'] == 256 and drv_off['routed_tensors'] == 18)
                msg_tr_off = 'n_expert=%d routed=%d' % (drv_off['n_expert'], drv_off['routed_tensors'])
            except RepackAbort as e:
                ok_tr_off, msg_tr_off = False, 'trace-off 정상 유도 실패: %s' % e
            pos_controls.append(('trace 비활성(명시 unset) 시 같은 256E 모델은 정상 유도', ok_tr_off, msg_tr_off))
            ok_dup_loader, msg_dup_loader = _expect_abort(lambda: load_model_shards(dup_p))
            ok_dup_loader = (ok_dup_loader and 'duplicate tensor name' in msg_dup_loader
                             and '[template:' not in msg_dup_loader)
            pos_controls.append(('중복 텐서: 정상 경로는 loader 가 선행 차단(템플릿 코드 미발화)',
                                 ok_dup_loader, msg_dup_loader))
        finally:
            if _saved_trace2 is None:
                os.environ.pop(TRACE_ENV_VAR, None)
            else:
                os.environ[TRACE_ENV_VAR] = _saved_trace2
        ok_pos = all(ok for _, ok, _ in pos_controls)
        checks.append(('OPEN_ARCH-ⓔ2 양성 대조 2종(trace 비활성 정상 유도 · 중복 텐서 loader 선행 차단) '
                       '— fail-close 거부 집계와 분리', ok_pos))
        print('[selftest] OPEN_ARCH-ⓔ2 양성 대조: %s' % ('PASS' if ok_pos else 'FAIL'))
        for label, ok_p, msg_p in pos_controls:
            print('    [%s] %-42s %s' % ('PASS' if ok_p else 'FAIL', label, msg_p[:100]))

        # ---- ⓔ3 trace 게이트 판정이 엔진과 정확 일치(exact "1") — unset/"0"/"1"/"2" 4표본 고정 ----
        # 1차 소스(26-08-02 직접 확인): ggml-moe-trace.cpp:265-268 `local_env_is_1` =
        # `v && strcmp(v,"1")==0` (:668 이 ggml_moe_trace_enabled 의 캐시 원천) · 게이트 호출부는
        # ggml-moe-direct.cpp:3783 `model.n_expert > 255 && ggml_moe_trace_enabled()`.
        # 즉 "2" 는 엔진에서 OFF 이므로 리패커도 OFF(정상 유도)여야 한다.
        trace_samples = [(None, False), ('0', False), ('1', True), ('2', False)]
        tg_rows = []
        _saved_trace3 = os.environ.get(TRACE_ENV_VAR)
        try:
            for val, want_on in trace_samples:
                if val is None:
                    os.environ.pop(TRACE_ENV_VAR, None)
                else:
                    os.environ[TRACE_ENV_VAR] = val
                got_on = _trace_gate_active()
                if want_on:
                    ok_s, msg_s = _expect_abort(lambda: derive_arch_template(load_model_shards(trace_p)))
                    ok_s = ok_s and '[template:trace-gate]' in msg_s
                else:
                    try:
                        _d = derive_arch_template(load_model_shards(trace_p))
                        ok_s, msg_s = (_d['n_expert'] == 256 and _d['routed_tensors'] == 18), '정상 유도'
                    except RepackAbort as e:
                        ok_s, msg_s = False, 'RepackAbort %s' % e
                tg_rows.append((val, want_on, got_on, (got_on == want_on) and ok_s, msg_s))
        finally:
            if _saved_trace3 is None:
                os.environ.pop(TRACE_ENV_VAR, None)
            else:
                os.environ[TRACE_ENV_VAR] = _saved_trace3
        ok_tg = all(r[3] for r in tg_rows)
        checks.append(('OPEN_ARCH-ⓔ3 trace 게이트가 엔진과 정확 일치(exact "1") — unset/"0"/"1"/"2" '
                       '4표본 고정(D1: 구현은 "2" 를 활성으로 봐 엔진과 갈렸음)', ok_tg))
        print('[selftest] OPEN_ARCH-ⓔ3 trace 게이트 엔진 일치: %s' % ('PASS' if ok_tg else 'FAIL'))
        for val, want_on, got_on, ok_s, msg_s in tg_rows:
            print('    [%s] %s=%-6r want_on=%-5s got_on=%-5s %s'
                  % ('PASS' if ok_s else 'FAIL', TRACE_ENV_VAR, val, want_on, got_on, msg_s[:80]))

        # ---- ⓕ 기본 CLI 불변(gate 없으면 템플릿 경로 미진입·--profile 병용 거부·--help 미노출) ----
        def _cli(args_):
            return subprocess.run([sys.executable, os.path.abspath(__file__)] + args_,
                                  capture_output=True, text=True, encoding='utf-8', timeout=90)
        cli_out = os.path.join(scratch, 'out_tpl_cli')
        c1 = _cli(['--plan', '--profile', 'gpt-oss-120b', '--model', tpl_model, '--out', cli_out])
        ok_c1 = (c1.returncode != 0) and ('[template:' not in (c1.stdout + c1.stderr))
        c2 = _cli(['--plan', '--experimental-arch-template', '--profile', 'gpt-oss-120b',
                   '--model', tpl_model, '--out', cli_out])
        ok_c2 = (c2.returncode != 0) and ('must not be combined' in c2.stderr)
        c3 = _cli(['--help'])
        ok_c3 = (c3.returncode == 0) and ('experimental-arch-template' not in c3.stdout)
        c4 = _cli(['--plan', '--experimental-arch-template', '--model', tpl_model, '--out', cli_out])
        ok_c4 = (c4.returncode == 0 and 'derived_from=gpt-oss@1' in c4.stdout
                 and not os.path.exists(os.path.join(cli_out, DERIVED_EXPECT_FILENAME))
                 and not os.path.exists(os.path.join(cli_out, 'experts.bin')))
        ok_cli = ok_c1 and ok_c2 and ok_c3 and ok_c4
        checks.append(('OPEN_ARCH-ⓕ 기본 CLI 불변(gate 없으면 카탈로그 경로 그대로·--profile 병용 거부·'
                       '--help 미노출·gate+--plan 은 0바이트)', ok_cli))
        print('[selftest] OPEN_ARCH-ⓕ CLI: catalog기본=%s 병용거부=%s help미노출=%s plan0바이트=%s'
              % (ok_c1, ok_c2, ok_c3, ok_c4))

    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        print('[selftest] 임시 파일 정리 완료: %s' % scratch)

    all_ok = all(ok for _, ok in checks)
    print('--- selftest 체크리스트 ---')
    for name, ok in checks:
        print(' [%s] %s' % ('PASS' if ok else 'FAIL', name))
    n_pass = sum(1 for _, ok in checks if ok)
    print('=== selftest 최종 판정: %s (%d/%d) ===' % ('PASS' if all_ok else 'FAIL', n_pass, len(checks)))

    _append_repack_log({'ts': datetime.now(timezone.utc).isoformat(), 'mode': 'selftest',
                         'pass': all_ok, 'n_pass': n_pass, 'n_total': len(checks),
                         'checks': [{'name': n, 'ok': o} for n, o in checks]})
    return all_ok


def _check_argparse_contract(script_path, model_path, out_dir):
    """argparse 실호출 조합: 필수 인자 누락·미등록 인자·selftest 단독·상호배타·유효 --plan(합성→
    카탈로그 미등록/expect 불일치로 정상 중단·쓰기 0)."""
    results = []

    def _run(args_):
        return subprocess.run([sys.executable, script_path] + args_,
                              capture_output=True, text=True, encoding='utf-8', timeout=90)

    try:
        p1 = _run(['--plan'])
        ok1 = p1.returncode != 0
        extra1 = p1.stderr.strip().splitlines()[-1] if p1.stderr.strip() else '(rc=%d)' % p1.returncode
    except Exception as e:
        ok1, extra1 = False, 'subprocess 예외: %r' % e
    results.append(('--plan 단독(필수 인자 누락) -> 에러 종료', ok1, extra1))

    try:
        p2 = _run(['--totally-bogus-flag'])
        ok2 = (p2.returncode != 0) and ('unrecognized' in p2.stderr.lower())
        extra2 = p2.stderr.strip().splitlines()[-1] if p2.stderr.strip() else '(rc=%d)' % p2.returncode
    except Exception as e:
        ok2, extra2 = False, 'subprocess 예외: %r' % e
    results.append(('미등록 인자 거부', ok2, extra2))

    parser = build_parser()
    try:
        ns = parser.parse_args(['--selftest'])
        ok3 = (ns.selftest is True) and (ns.model is None)
        extra3 = 'parsed selftest=%r model=%r' % (ns.selftest, ns.model)
    except SystemExit as e:
        ok3, extra3 = False, '예상치 못한 SystemExit(%r)' % (e.code,)
    results.append(('--selftest 단독 파싱', ok3, extra3))

    try:
        p4 = _run(['--plan', '--selftest', '--profile', 'gpt-oss-120b', '--model', model_path, '--out', out_dir])
        ok4 = p4.returncode != 0
        extra4 = p4.stderr.strip().splitlines()[-1] if p4.stderr.strip() else '(rc=%d)' % p4.returncode
    except Exception as e:
        ok4, extra4 = False, 'subprocess 예외: %r' % e
    results.append(('--plan/--selftest 동시 지정 -> 상호배타 에러', ok4, extra4))

    # 유효 --plan 실호출: 합성 모델을 실 카탈로그 profile 로 지정 → cross_check_expect 불일치로
    # 정상 중단(rc≠0) + 쓰기 0(repack_log 미증가·out 산출물 부재).
    log_path = os.path.join(os.path.dirname(os.path.abspath(script_path)), 'repack_log.jsonl')
    log_before = os.path.getsize(log_path) if os.path.exists(log_path) else None
    try:
        p5 = _run(['--plan', '--profile', 'gpt-oss-120b', '--model', model_path, '--out', out_dir])
        log_after = os.path.getsize(log_path) if os.path.exists(log_path) else None
        wrote_out = os.path.exists(os.path.join(out_dir, 'experts.bin')) or os.path.exists(os.path.join(out_dir, 'manifest.json'))
        ok5 = (p5.returncode != 0) and (log_after == log_before) and (not wrote_out)
        extra5 = 'rc=%d log_before=%r log_after=%r wrote_out=%s' % (p5.returncode, log_before, log_after, wrote_out)
    except Exception as e:
        ok5, extra5 = False, 'subprocess 예외: %r' % e
    results.append(('유효 --plan(합성→expect 불일치로 정상 중단 + 쓰기 0)', ok5, extra5))

    # --profile 누락 시 본실행/plan 거부
    try:
        p6 = _run(['--model', model_path, '--out', out_dir])
        ok6 = p6.returncode != 0 and ('profile' in p6.stderr.lower())
        extra6 = p6.stderr.strip().splitlines()[-1] if p6.stderr.strip() else '(rc=%d)' % p6.returncode
    except Exception as e:
        ok6, extra6 = False, 'subprocess 예외: %r' % e
    results.append(('--profile 누락 시 거부(경로 인자 금지·카탈로그 필수)', ok6, extra6))

    # ★--verify-only: 모드 상호배타(--plan 동시 지정 거부) + --profile 누락 거부
    try:
        p7 = _run(['--plan', '--verify-only', '--profile', 'gpt-oss-120b', '--model', model_path, '--out', out_dir])
        ok7 = p7.returncode != 0
        extra7 = p7.stderr.strip().splitlines()[-1] if p7.stderr.strip() else '(rc=%d)' % p7.returncode
    except Exception as e:
        ok7, extra7 = False, 'subprocess 예외: %r' % e
    results.append(('--plan/--verify-only 동시 지정 -> 상호배타 에러', ok7, extra7))

    try:
        p8 = _run(['--verify-only', '--model', model_path, '--out', out_dir])
        ok8 = p8.returncode != 0 and ('profile' in p8.stderr.lower())
        extra8 = p8.stderr.strip().splitlines()[-1] if p8.stderr.strip() else '(rc=%d)' % p8.returncode
    except Exception as e:
        ok8, extra8 = False, 'subprocess 예외: %r' % e
    results.append(('--verify-only + --profile 누락 -> 거부', ok8, extra8))

    return results


# ---------------------------------------------------------------------------
# argparse — 전 인자를 여기 하나에만 등록(sys.argv 수동 파싱 금지)
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog='repack_experts.py',
        description=('repack-v2: rearranges the routed expert tensors of a MoE GGUF into contiguous, sector-aligned (layer,expert) records (experts.bin + manifest v2) and proves losslessness by verifying every record (REPACK_V2_DESIGN.md).'))
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument('--plan', action='store_true',
                             help='read only the GGUF header and print the plan (writes 0 bytes; for review).')
    mode_group.add_argument('--selftest', action='store_true',
                             help='run the built-in regression tests against synthetic mini GGUFs (no real model needed).')
    mode_group.add_argument('--verify-only', action='store_true', dest='verify_only',
                             help='fully re-verify an existing artifact (experts.bin + manifest.json) once without repacking and append a new record to verify_report.json (--profile/--model/--out are required; artifact bytes are unchanged; scope comes from the manifest, so --scope is ignored).')
    ap.add_argument('--profile', type=str, default=None,
                     help='reference-lock profile id (required for --plan and real runs; not a path; must be an id registered in EXPECT_CATALOG).')
    ap.add_argument('--model', type=str, default=None,
                     help='input GGUF path (required for --plan and real runs; for a split model, point at any shard and the siblings are discovered).')
    ap.add_argument('--out', type=str, default=None,
                     help='output directory (required for --plan and real runs; there is no default path).')
    ap.add_argument('--force', action='store_true',
                     help='overwrite the output (experts.bin/manifest.json) even if it already exists.')
    ap.add_argument('--scope', type=str, default=None, choices=['all', 'execution'],
                     help='routed tensor scope (appendix A). all=every routed layer (default when unset) / execution=execution layers only, excluding the last N blocks given by nextn_predict_layers.')
    # OPEN_ARCH A축(v0.2 §1) 비공개 feature gate — 기본 CLI 동작(catalog-only)은 완전 불변이고,
    # 이 플래그로만 아키 템플릿 유도 경로에 진입한다. 릴리스 활성화는 M5 원자 관문 후이므로
    # --help 에 노출하지 않는다(argparse.SUPPRESS).
    ap.add_argument('--experimental-arch-template', action='store_true', dest='arch_template',
                     help=argparse.SUPPRESS)
    ap.add_argument('--log-path', type=str, default=None, dest='log_path',
                     help='path to append repack_log.jsonl to (RC-1). Unset = next to this script (previous behaviour). Parent directories are created automatically. The log contents and the artifact are unchanged.')
    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)
    _set_repack_log_path(getattr(args, 'log_path', None))

    if args.selftest:
        ok = cmd_selftest()
        sys.exit(0 if ok else 1)

    if not args.model:
        parser.error('--model is required for --plan, --verify-only and real runs.')
    if not args.out:
        parser.error('--out is required for --plan, --verify-only and real runs (there is no default path).')
    if getattr(args, 'arch_template', False):
        # 비공개 gate: 참조 락을 현장 유도 expect 가 대신하므로 카탈로그 id 와 동시 지정 금지
        # (둘 중 무엇이 권위인지 모호해지는 것을 원천 차단 — fail-closed).
        if args.profile:
            parser.error('--profile must not be combined with --experimental-arch-template (the derived expect replaces the catalog reference lock).')
    elif not args.profile:
        parser.error('--profile is required for --plan, --verify-only and real runs (reference lock - not a path, must be an id registered in EXPECT_CATALOG).')

    try:
        if args.plan:
            cmd_plan(args)
        elif args.verify_only:
            cmd_verify_only(args)
        else:
            cmd_repack(args)
    except RepackAbort as e:
        print('[abort] %s' % e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
