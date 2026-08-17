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
    python repack_experts.py --mode virtual --profile <id> --model <gguf> --out <dir>
      (SPEC_REPACK_V3 §4: 가상 재팩 — experts.bin 을 만들지 않고 manifest v3(schema "3.0",
       mode "virtual")+plan_report.json 만 산출한다. 데이터 이동 0·공간 정확히 원본 1.0×.
       --mode bin(기본)은 아래 v2 규약 그대로이며 바이트 단위로 불변이다.)

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
import io
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
#   IQ2_XS:256/74(d2+qs[QK_K/8]×2+scales[QK_K/32])  IQ3_XXS:256/98(d2+qs[3·QK_K/8])
#     — K3(kimi-k3 UD-Q2_K_XL) 등재로 확장. 1차 소스 ggml-common.h:388-393(block_iq2_xs)·
#       :407-410(block_iq3_xxs)·:89(QK_K=256), 등록 blck_size/type_size=ggml.c:815-818·:823-826.
#       실증: K3 19샤드 2,573텐서 32B 정렬 재구성 불일치 0 + gap 기반↔이론 기반 델타 0.
QUANT_TRAITS = {
    'F32':   (1, 4),
    'F16':   (1, 2),
    'MXFP4': (32, 17),
    'Q3_K':  (256, 110),
    'Q4_K':  (256, 144),
    'Q5_K':  (256, 176),
    'Q6_K':  (256, 210),
    'Q8_0':  (32, 34),
    'IQ2_XS':  (256, 74),
    'IQ3_XXS': (256, 98),
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
    # Kimi-K3 UD-Q2_K_XL(26-08-13 등재): 원천=bench_results/g5/gguf_map_k3_s{1..19}.json(19-shard
    # 헤더 실측·총 2,573텐서·32B 정렬 오프셋 재구성 불일치 0·tail padding 은 shard1[메타 전용·
    # 텐서 0]의 26B 뿐 — 397B shard1 15B 와 동형). 93층 중 층 0 은 dense
    # (kimi-k3.leading_dense_block_count=1) → MoE 층 1..92·separate 3-part·bias 無·routed
    # 276=3×92·★n_expert=896(카탈로그 최대 — 구 상한 512E)·★top-16(첫 16-way)·★19-shard(첫
    # 두자리 shard — 구 최다 6)·expert 799,065,243,648B(파일 861,277,858,912B 의 92.78%).
    # ★routed 양자 혼합 IQ2_XS 263/IQ3_XXS 13(층 91=gate/up/down 전량 IQ3_XXS · 층
    # 12,24,25,28,36,37,48,60,72,84=down 만) → ★층별 payload 비균일 3종(9,547,776×81층·
    # 10,579,968×10층·12,644,352×1층) = 카탈로그 첫 비균일 stride 사례(양성 회귀 v3-P5 가 선행
    # 관문). 선결로 QUANT_TRAITS 에 IQ2_XS·IQ3_XXS 를 등재했다(위 표 — 없으면
    # per_expert_slice_bytes 가 fail-close).
    # ★MTP/NextN 텐서·KV 모두 부재(KV 64종·텐서명 2,573건 전수·_MTP_MARKER_RE 0건) →
    # scope='all'(nextn_predict_layers 부재라 --scope execution 은 build_layout 이 거부).
    # ★shexp 276(Q8_0)·ffn_gate_inp 92(F32 [7168,896])·exp_probs_b.bias 92(F32 [896])=동결
    # 허용표 기적재 트렁크 상주·재팩 무관. K3 신설 ffn_routed_{up,down,norm} 92×3 은 expert 축
    # 부재(Q8_0 2D·F32 1D) → _EXPERT_LIKE_RE 미해당·미분류 0.
    'kimi-k3-ud-q2kxl': {'sha256': 'e57fcebb0dde4d12bb779308397a60cbe44b4f984a52dd4e5a3e46c667d9a8a3', 'scope': 'all'},
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
    # ★r1 실 결함 ①(전반부): 받은 문자열을 그대로 보존하면, 상대 경로로 호출한 실행이 상대 경로를
    # 그대로 manifest 에 박는다. manifest 는 source IDENTITY 를 주장하는 문서인데 상대 경로는
    # "그때 그 디렉토리 기준"이라는 조건부 주장이라 재대조 시점에 같은 뜻이 아니다. 여기서 한 번
    # 절대화하면 형제 shard 도 정규화된 dirname 에서 만들어지므로 전 소비자가 같은 철자를 본다.
    model_path = os.path.abspath(model_path)
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


def write_derived_expect(out_dir, raw, filename=DERIVED_EXPECT_FILENAME):
    """derived expect 를 <repack-output>\\derived.expect.json 에 원자 기록(번들 expects_dir 금지).
    ★§Z-④: mode=virtual 은 `filename` 에 `.partial` candidate 이름을 넘겨 **검증 전 선교체를
    없앤다**(구 산출물은 verifier FAIL 시 그대로 남는다)."""
    path = os.path.join(out_dir, filename)
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
def _same_fs_name(a, b):
    """path identity for manifest comparison.

    Compared absolute, normalised and case-folded, because on this filesystem `D:\\x\\y.gguf` and
    `d:/x/y.gguf` name the same bytes - case and separator are spelling, not identity. The stored
    spelling is left as produced so plan stdout stays readable and the launcher keeps parsing a
    case-preserving path; the folding happens here, at comparison time, and nowhere else.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _same_drive_root(a, b):
    """volume identity. Deliberately NOT `_same_fs_name`: a bare drive spec like `C:` means "the
    current directory on C:" to `abspath`, so resolving it would make the comparison depend on
    where the process happens to be running. Only case and a trailing separator are spelling here.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return (os.path.normcase(a.rstrip('\\/')) == os.path.normcase(b.rstrip('\\/')))


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
# ---------------------------------------------------------------------------
# ★본작업 소비 표면 — launcher 의 `--plan` 텍스트 파서 계약 (26-08-08 리드 처분 ⓑ)
#
# 아래 stdout 줄들은 본작업 launcher 가 파싱한다:
#   bench/moe-direct/launcher/Start-MoeDirect.ps1  함수 ConvertFrom-TemplatePlanText
# 정규식·헤더·완료줄은 그 함수의 사본이며, `--selftest` 의 OPEN_ARCH-ⓑ4~ⓑ6 관문이
# **실제 --plan stdout** 으로 이 표를 직접 주장한다. 파서는 keyed 줄이 각각 정확히 1회
# 나타날 것을 요구하고(중복이면 "단일 plan 캡처가 아니다"로 거부), expect 헤더 이후의
# 모든 줄을 derived expect 본문으로 삼으며, 완료 줄이 없으면 잘린 캡처로 본다.
# ⇒ **이 파일의 --plan 출력을 고칠 때 이 표를 함께 보라.** 표를 깨는 수정은 selftest 가
#   즉시 FAIL 로 잡는다(과거엔 아무도 주장하지 않아 사람 검수에만 걸렸고, 파서 정규식이
#   삭제된 줄을 마침 보지 않아 우연히 안 깨졌다 — 우연은 계약이 아니다).
# ---------------------------------------------------------------------------
LAUNCHER_PLAN_KEYED_LINES = (
    ('derive', r'^\[EXPERIMENTAL arch-template\] derived_from=(\S+) routed_scope=(\S+)'
               r' \(template default=(\S+)\) inventory_sha256=([0-9a-f]{64})$'),
    ('tpl',    r'^\[EXPERIMENTAL arch-template\] template layers=(\d+)\.\.(\d+) \((\d+)\) routed_tensors=(\d+) '),
    ('arch',   r'^arch=(\S+) n_layer=(\d+) n_expert=(\d+) n_expert_used=(\d+) schema=(\S+) bias=(\S+)$'),
    ('moe',    r'^moe_layers: (\d+) entries \[(\d+)\.\.(\d+)\]'),
    ('stride', r'^output alignment A=(\d+), stride\[l\] .* \(min=(\d+) max=(\d+)\), slot_stride_max=(\d+)$'),
    ('bytes',  r'^expert_payload_total\(=expert_bytes\)=(\d+)$'),
)
LAUNCHER_PLAN_EXPECT_HEAD = '--- derived expect (%s, not written in --plan) ---' % DERIVED_EXPECT_FILENAME
LAUNCHER_PLAN_DONE_LINE = '--plan done (0 bytes written, no GPU used)'
# cmd_plan 이 자기 주석으로 선언한 "bin 경로 stdout 무변경" 계약의 머리 4줄(순서 포함).
# 파서가 소비하지는 않으나 ⑤ 시공이 실제로 삭제한 줄이 여기 있다(`profile:`).
LAUNCHER_PLAN_BIN_PREAMBLE = (
    '=== --plan: GGUF header analysis (0 bytes written) ===',
    'profile: ',
    'model: ',
    'out (planned target): ',
)


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
    # §2-1 생산 계약: bin 은 항상 schema "2.0" 이고 mode 필드를 갖지 않는다("3.0"+bin 산출 금지).
    # 현 코드에는 이 값을 바꿀 경로가 없으므로 이 가드는 산출 바이트를 바꾸지 않는다(회귀 방지용).
    _guard_manifest_mode(manifest, MODE_BIN)
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
        problems.append('experts.bin actual size(%d) != independently computed per-layer prefix-sum(%d)'
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
# repack v3 (mode=virtual) — SPEC_REPACK_V3.md v1.0 §2/§4 구현
#
# experts.bin 을 **만들지 않는다**. 리패커는 manifest v3(+plan_report)만 산출하고, 소비자가 원본
# GGUF 에서 `(source_index, abs_offset + expert×slice_bytes)` per-part 산개 읽기로 슬롯을 채운다.
# 공간=정확히 원본 1.0×·온보딩=헤더 파싱+산술(데이터 이동 0).
#
# ★mode=bin(기본)은 v2 바이트 규약 그대로다 — 이 절의 어떤 함수도 bin 경로에서 호출되지 않는다.
#   bin 경로와의 유일한 접점: ①build_manifest 직후의 스키마 가드(_guard_manifest_mode — v2
#   manifest 는 schema 2.0·mode 필드 부재여야 한다는 §2-1 계약을 생산 측에서 못 박는 1줄)
#   ②argparse `--mode` 의 기본값 'bin'(미지정 시 종전 경로 그대로).
# ---------------------------------------------------------------------------
SCHEMA_VERSION_V3 = '3.0'
MODE_BIN = 'bin'
MODE_VIRTUAL = 'virtual'
MANIFEST_FILENAME = 'manifest.json'
PLAN_REPORT_FILENAME = 'plan_report.json'
# DF-1(SPEED_LEVER_LEDGER · SPEC 부속 정오 1 기술 귀결): SP-A 에서 원본 GGUF 가 런타임 데이터
# 원천이 되므로 그 무결성이 무손실 사슬에 편입된다. manifest 는 **헤더 영역 digest**(주소 권위
# 영역 = [0, data_start) — magic·KV·텐서 디렉토리·정렬 패딩 전부)를 기록하고, 전파일 SHA-256 은
# 선택 provenance 로 plan_report 에만 둔다(§2-5·§4-3: 권장·비차단). 범위 선택 근거는 보고서의
# "판단 보류" 항목 참조 — 여기 기본값은 사양이 이미 요구하는 sources[] identity + 헤더 digest.
SOURCE_DIGEST_ALGO = 'sha256'
SOURCE_DIGEST_SCOPE_HEADER = 'header'
U64_MAX = (1 << 64) - 1

# --- §Z-③ legacy alignment 결속 (SPEC_IO_METRICS_V3 §7 · 부속 정오 2-ⓘ) -----------------
# D-A2 의 분자는 `legacy_stride_bytes[l] = align_up(legacy_payload_bytes[l], legacy_align_bytes)`
# 이다. virtual 은 출력 볼륨이 없어 `layout.align_bytes`(=source 볼륨 A)가 **과거 v2 output
# 볼륨의 stride 와 같다는 보장이 없다** — 그래서 legacy 축을 별도 필드로 분리해 결속한다.
#   paired  : 그 모델의 실제 v2 재팩 manifest 가 있으면 그 `layout.align_bytes` 를 승계하고,
#             verifier 가 그 파일을 strict 재개방해 identity·SHA·값을 대조한다.
#   unpaired: v2 실적이 없는 신규 모델(K3 등)은 canonical 4096 을 강제하고, 공개 수치는
#             §6-5 재기준화 각주 의무를 진다.
# [[C:repack.legacy-align]] 이 서술은 **사본**이다 — 권위는 SPEC_IO_METRICS_V3 §7 이고, 필드명·
# enum·canonical 값을 바꾸려면 그 사양을 먼저 고친다(사본은 원본에 진다).
LEGACY_ALIGN_SOURCE_PAIRED = 'paired_v2'
LEGACY_ALIGN_SOURCE_CANONICAL = 'canonical_4096'
LEGACY_ALIGN_CANONICAL_BYTES = 4096


class VirtualBinRegression(RepackAbort):
    """§2-4 경계(bracket EOF) 위반 = 'virtual 로는 성립하지 않는 프로파일' 자동 판정.
    RepackAbort 하위이므로 생산은 즉시 중단되고(plan 단계), 메시지가 mode=bin 회귀를 명시한다."""


# --- checked uint64 산술(§2-4: 전 산술 checked uint64·오버플로=생산 중단) -------------------
# Python int 는 임의정밀이라 하드웨어 오버플로가 없다 — 따라서 "checked" 는 **매 연산 결과의
# uint64 정의역 확인**으로 구현한다(소비자 C++ 측 uint64 와 같은 정의역을 생산 측에서 강제).
def _u64(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepackAbort('checked uint64: %s is not an integer (%r)' % (what, value))
    if value < 0 or value > U64_MAX:
        raise RepackAbort('checked uint64 domain violation: %s=%d (0 <= x <= 2**64-1)' % (what, value))
    return value


def _u64_add(a, b, what):
    return _u64(_u64(a, what + '.lhs') + _u64(b, what + '.rhs'), what)


def _u64_mul(a, b, what):
    return _u64(_u64(a, what + '.lhs') * _u64(b, what + '.rhs'), what)


def _u64_align_up(value, align, what):
    _u64(value, what)
    if isinstance(align, bool) or not isinstance(align, int) or align <= 0:
        raise RepackAbort('checked uint64: alignment for %s is not a positive integer (%r)' % (what, align))
    return _u64(((value + align - 1) // align) * align, what + '.align_up')


def _guard_manifest_mode(manifest, expected_mode):
    """§2-1 생산 측 계약 강제: bin=schema 2.0 ∧ mode 필드 부재 / virtual=schema 3.0 ∧
    mode 'virtual'. `"3.0"+mode:"bin"` 조합은 **산출 금지**(여기서 차단)."""
    sv = manifest.get('schema_version')
    has_mode = 'mode' in manifest
    if expected_mode == MODE_BIN:
        if sv != SCHEMA_VERSION or has_mode:
            raise RepackAbort('mode=bin must produce schema_version %r with no mode field (v2 byte contract): '
                              'schema_version=%r mode_present=%s' % (SCHEMA_VERSION, sv, has_mode))
    elif expected_mode == MODE_VIRTUAL:
        if sv != SCHEMA_VERSION_V3 or manifest.get('mode') != MODE_VIRTUAL:
            raise RepackAbort('mode=virtual must produce schema_version %r with mode %r: schema_version=%r mode=%r'
                              % (SCHEMA_VERSION_V3, MODE_VIRTUAL, sv, manifest.get('mode')))
    else:
        raise RepackAbort('unknown production mode %r (expected %r|%r)' % (expected_mode, MODE_BIN, MODE_VIRTUAL))


def resolve_alignment_for_sources(model, allow_default_align):
    """§2-4 A 출처: virtual 엔 출력 볼륨이 없다 — 질의 대상은 **source shard 가 놓인 볼륨(들)**.
    A = max(4096, 전 source 볼륨의 runtime 섹터 질의 결과). 각 질의 결과를 layout.align_query[]
    에 provenance 로 기록해 소비 시점의 재질의 대조를 가능하게 한다 — ★대조 범위는 **주소 관련
    4필드(`source_index`·`method`·`logical`·`physical`)와 drive root** 이고 `fallback_reason` 은
    비교하지 않는다(r8 M2 로 축소된 계약. 정오 2 ⓙ·§4-3 verifier 머리 주석과 같은 문면을 쓴다 —
    한쪽만 고치지 말 것). [[C:repack.align-query]] 반환 (A, align_query 리스트)."""
    queries = []
    A = 4096
    for h in model['shards']:
        info = query_sector_alignment_for_path(h['path'])
        if info['method'] == 'default_4096' and not allow_default_align:
            raise RepackAbort('both the primary and secondary sector-alignment queries failed for source shard %d '
                              '(%s) - production cannot fall back to the 4096 default: %s'
                              % (h['source_index'], h['path'], info['fallback_reason']))
        queries.append({'source_index': h['source_index'], 'drive_root': _drive_root(h['path']),
                        'method': info['method'], 'logical': int(info['logical']),
                        'physical': int(info['physical']), 'fallback_reason': info['fallback_reason']})
        A = max(A, int(info['logical']), int(info['physical']))
    if not queries:
        raise RepackAbort('no source shard available for the sector-alignment query')
    if A <= 0 or (A & (A - 1)) != 0:
        raise RepackAbort('resolved alignment A=%d is not a power of two' % A)
    return A, queries


def compute_virtual_layout(layout, A, source_bytes):
    """§2-4 슬롯 레이아웃 산술(전 산술 checked uint64).

        h[p]             = abs_offset[p] mod A            (aligned part 전용 상수)
        region[p]        = align_up(h[p] + slice_bytes[p], A)
        slot_offset[0]   = 0
        slot_offset[p+1] = slot_offset[p] + region[p]      (비중첩 점화식)
        layer_slot_bytes = sum(region)
        slot_stride_max  = max_layers(layer_slot_bytes)

    비4K(aligned=false) part: h[p]·bracket_head 는 미정의(기재 금지)·region=align_up(slice,A)·
    data_offset=slot_offset(복사가 정렬을 흡수)·staging 요구치 align_up(slice,A)+A 를 기록.
    경계 강제(bracket EOF): align_up(abs_offset + n_expert×slice_bytes, A) <= source.bytes —
    위반 = VirtualBinRegression(해당 프로파일 mode=bin 회귀 판정·plan 단계).

    source_bytes = {source_index: file_bytes}. 반환: vplan dict(비권위 records[] 포함)."""
    n_expert = _u64(layout['n_expert'], 'model.n_expert')
    if n_expert <= 0:
        raise RepackAbort('n_expert must be positive: %d' % n_expert)
    _u64(A, 'layout.align_bytes')
    vlayers = []
    records = []
    payload_total = 0
    slot_stride_max = 0
    for L in layout['layers']:
        slot_offset = 0
        vparts = []
        for p in L['parts']:
            slice_bytes = _u64(p['part_bytes'], 'slice_bytes')
            abs_offset = _u64(p['abs_offset'], 'abs_offset')
            if slice_bytes <= 0:
                raise RepackAbort('slice_bytes must be positive: layer=%d part=%s' % (L['layer'], p['name']))
            if p['source_index'] not in source_bytes:
                raise RepackAbort('unknown source_index %r for layer=%d part=%s' % (p['source_index'], L['layer'], p['name']))
            src_bytes = _u64(source_bytes[p['source_index']], 'source.bytes')
            tensor_bytes = _u64_mul(slice_bytes, n_expert, 'tensor_bytes')
            payload_end = _u64_add(abs_offset, tensor_bytes, 'payload_end')
            if payload_end > src_bytes:
                raise RepackAbort('routed tensor payload exceeds the source EOF: %s [s%d] abs_offset=%d '
                                  'n_expert*slice=%d end=%d source.bytes=%d'
                                  % (p['source_tensor'], p['source_index'], abs_offset, tensor_bytes,
                                     payload_end, src_bytes))
            bracket_end = _u64_align_up(payload_end, A, 'bracket_end')
            if bracket_end > src_bytes:
                raise VirtualBinRegression(
                    'bracket EOF violation (section 2-4) - this profile is automatically regressed to '
                    'mode=bin: %s [s%d] align_up(abs_offset+n_expert*slice, A)=%d > source.bytes=%d '
                    '(abs_offset=%d slice=%d n_expert=%d A=%d). the last aligned bracket read would run '
                    'past the end of the shard, and no explicit tail path is frozen.'
                    % (p['source_tensor'], p['source_index'], bracket_end, src_bytes,
                       abs_offset, slice_bytes, n_expert, A))
            aligned = (slice_bytes % A == 0)
            if aligned:
                head = _u64(abs_offset % A, 'bracket_head')
                region = _u64_align_up(_u64_add(head, slice_bytes, 'bracket_span'), A, 'region')
                data_offset = _u64_add(slot_offset, head, 'data_offset')
            else:
                head = None
                region = _u64_align_up(slice_bytes, A, 'region')
                data_offset = _u64(slot_offset, 'data_offset')
            vp = {'name': p['name'], 'source_tensor': p['source_tensor'], 'source_index': p['source_index'],
                  'type': p['type'], 'dims': list(p['dims']), 'expert_axis': p['expert_axis'],
                  'abs_offset': abs_offset, 'slice_bytes': slice_bytes, 'aligned': aligned}
            if aligned:
                vp['bracket_head'] = head
            vp['slot_offset'] = _u64(slot_offset, 'slot_offset')
            vp['data_offset'] = data_offset
            if not aligned:
                # §2-4/§3-5: 브라켓 read 는 슬롯이 아니라 전용 staging 에 착지한다(최악 src_head<A
                # 수용). 소비자 소관이지만 요구치는 생산이 기록한다.
                vp['staging_bytes'] = _u64_add(region, A, 'staging_bytes')
            vparts.append(vp)
            payload_total = _u64_add(payload_total, tensor_bytes, 'totals.virtual_payload_bytes')
            slot_offset = _u64_add(slot_offset, region, 'slot_offset')
        if not vparts:
            raise RepackAbort('layer %d has no routed part' % L['layer'])
        layer_slot_bytes = _u64(slot_offset, 'layer_slot_bytes')
        if layer_slot_bytes % A != 0:
            raise RepackAbort('layer %d layer_slot_bytes(%d) is not a multiple of A(%d)' % (L['layer'], layer_slot_bytes, A))
        # 독립 2산식 폐합: 슬롯 산술이 소비하는 slice 합 == v2 동결 경로가 낸 payload[l].
        # (virtual_payload_bytes 와 expect 의 expert_bytes_total 이 어긋나면 여기서 먼저 걸린다.)
        slice_sum = sum(vp['slice_bytes'] for vp in vparts)
        if slice_sum != L['payload_bytes']:
            raise RepackAbort('layer %d sum(slice_bytes)(%d) != payload_bytes from the frozen v2 layout(%d) '
                              '- arithmetic does not close' % (L['layer'], slice_sum, L['payload_bytes']))
        vlayers.append({'layer': L['layer'], 'layer_slot_bytes': layer_slot_bytes, 'parts': vparts})
        slot_stride_max = max(slot_stride_max, layer_slot_bytes)
    # records[] = 비권위 witness. cardinality=sum_layers(n_expert x n_parts)·순서=(layer↑,
    # expert↑, part 배열순) 고정.
    for VL in vlayers:
        for e in range(n_expert):
            for vp in VL['parts']:
                records.append({'layer': VL['layer'], 'expert': e, 'part': vp['name'],
                                'source_index': vp['source_index'],
                                'src_offset': _u64_add(vp['abs_offset'],
                                                        _u64_mul(e, vp['slice_bytes'], 'expert*slice'),
                                                        'src_offset'),
                                'slice_bytes': vp['slice_bytes'], 'data_offset': vp['data_offset']})
    n_records = sum(n_expert * len(VL['parts']) for VL in vlayers)
    if n_records != len(records):
        raise RepackAbort('internal: records cardinality %d != sum_layers(n_expert*n_parts) %d' % (len(records), n_records))
    expert_bytes_total = n_expert * sum(L['payload_bytes'] for L in layout['layers'])
    if payload_total != expert_bytes_total:
        raise RepackAbort('totals.virtual_payload_bytes(%d) != n_expert*sum(payload_bytes)(%d) - arithmetic does '
                          'not close' % (payload_total, expert_bytes_total))
    return {'A': A, 'layers': vlayers, 'slot_stride_max': _u64(slot_stride_max, 'slot_stride_max'),
            'records': records, 'n_records': n_records,
            'virtual_payload_bytes': _u64(payload_total, 'totals.virtual_payload_bytes')}


def _file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_header_digest(path, data_start):
    """DF-1: 주소 권위 영역 digest = sha256(파일 [0, data_start))."""
    n = _u64(data_start, 'sources[].data_start')
    with open(path, 'rb') as f:
        head = f.read(n)
    if len(head) != n:
        raise RepackAbort('short read of the header region: %s expected=%d actual=%d' % (path, n, len(head)))
    return {'algo': SOURCE_DIGEST_ALGO, 'scope': SOURCE_DIGEST_SCOPE_HEADER,
            'header_bytes': n, 'sha256': hashlib.sha256(head).hexdigest()}


def build_source_digests(model):
    """shard 별 헤더 영역 digest(전 shard). 반환 {source_index: digest dict}."""
    return {h['source_index']: source_header_digest(h['path'], h['data_start']) for h in model['shards']}


def _load_v2_manifest_strict(path, what):
    """paired v2 manifest 를 **엄격 JSON으로 재개방**하고 원시 바이트 SHA 를 함께 낸다.
    반환 (v2_dict, raw_sha256, abs_path). 실패는 전부 RepackAbort."""
    abs_path = os.path.abspath(path)
    try:
        raw = open(abs_path, 'rb').read()
    except OSError as e:
        raise RepackAbort('%s: cannot read the paired v2 manifest %s: %r' % (what, abs_path, e))
    try:
        v2 = _strict_json_load_bytes(raw)
    except (_DuplicateManifestKey, _NonStandardJSONConstant) as e:
        raise RepackAbort('%s: the paired v2 manifest is not strict JSON (%s): %r' % (what, abs_path, e))
    except Exception as e:
        raise RepackAbort('%s: the paired v2 manifest is corrupt JSON (%s): %r' % (what, abs_path, e))
    if not isinstance(v2, dict):
        raise RepackAbort('%s: the top level of the paired v2 manifest is not an object (%s)' % (what, abs_path))
    return v2, hashlib.sha256(raw).hexdigest(), abs_path


def resolve_legacy_align(legacy_v2_manifest_path=None):
    """§Z-③ D-A2 분자용 legacy alignment 결정 [[C:repack.legacy-align]].

    paired(경로 지정) = 그 모델의 실제 v2 재팩 manifest 의 `layout.align_bytes` 를 승계 /
    unpaired(미지정) = canonical 4096 강제. 반환 dict 는 manifest·plan_report 양쪽의 입력이다.

    ★여기서는 값 추출에 **필요한 최소 검사만** 한다(strict JSON·schema 2.0·mode 필드 부재·
    align_bytes 양의 정수). 모델/reference identity 대조는 **verifier 의 몫**이다 — 생산이
    가져온 숫자가 "이 모델의 v2 manifest 에서 왔다"를 독립적으로 증명하는 것이 결속의 핵심이라,
    그 증명을 생산 측에 두 벌로 두면 같은 사각을 공유한다(§4-3 독립성 사상)."""
    if not legacy_v2_manifest_path:
        return {'bytes': LEGACY_ALIGN_CANONICAL_BYTES, 'source': LEGACY_ALIGN_SOURCE_CANONICAL,
                'v2_manifest_sha256': None, 'v2_manifest_path': None}
    v2, raw_sha, abs_path = _load_v2_manifest_strict(legacy_v2_manifest_path, '--legacy-v2-manifest')
    sv = v2.get('schema_version')
    if sv != SCHEMA_VERSION or 'mode' in v2:
        raise RepackAbort('--legacy-v2-manifest: %s is not a v2 (mode=bin) manifest - schema_version=%r '
                          'mode_present=%s (expected %r with no mode field)'
                          % (abs_path, sv, 'mode' in v2, SCHEMA_VERSION))
    v2_layout = v2.get('layout') if isinstance(v2.get('layout'), dict) else {}
    v2_align = v2_layout.get('align_bytes')
    if isinstance(v2_align, bool) or not isinstance(v2_align, int) or v2_align <= 0:
        raise RepackAbort('--legacy-v2-manifest: %s has no usable layout.align_bytes (%r)' % (abs_path, v2_align))
    return {'bytes': _u64(v2_align, 'legacy_align_bytes'), 'source': LEGACY_ALIGN_SOURCE_PAIRED,
            'v2_manifest_sha256': raw_sha, 'v2_manifest_path': abs_path}


def build_manifest_v3(model, layout, vplan, A, align_queries, profile_id, expect_sha256, digests,
                      legacy_align=None):
    """§2 manifest v3(mode=virtual). bin 전용 좌표 필드(record_base·bin 의미의 stride_bytes/
    part_offset·totals.bin_file_bytes)는 **산출하지 않는다**."""
    sources = []
    for h in model['shards']:
        sources.append({'index': h['source_index'], 'path': h['path'], 'bytes': h['file_bytes'],
                        'mtime': os.path.getmtime(h['path']), 'gguf_version': h['gguf_version'],
                        'alignment': h['alignment'], 'data_start': h['data_start'],
                        'digest': digests[h['source_index']]})

    manifest_layers = [{'layer': VL['layer'], 'layer_slot_bytes': VL['layer_slot_bytes'],
                        'parts': [dict(vp) for vp in VL['parts']]} for VL in vplan['layers']]

    source_tensors = []
    for VL in vplan['layers']:
        for vp in VL['parts']:
            source_tensors.append({'name': vp['source_tensor'], 'source_index': vp['source_index'],
                                    'abs_offset': vp['abs_offset'],
                                    'bytes': vp['slice_bytes'] * layout['n_expert'],
                                    'type': vp['type'], 'dims': list(vp['dims'])})

    quant_traits = {}
    for tt in layout['used_types']:
        bv, bb = QUANT_TRAITS[tt]
        quant_traits[tt] = {'block_values': bv, 'block_bytes': bb}

    model_dict = {'arch': layout['arch'], 'n_layer': layout['n_layer'], 'n_expert': layout['n_expert'],
                  'n_expert_used': layout['n_expert_used'], 'moe_layers': list(layout['moe_layers'])}
    if layout.get('scope', 'all') != 'all':
        model_dict['routed_scope'] = layout['scope']

    legacy_align = legacy_align or resolve_legacy_align(None)
    # §Z-③ [[C:repack.legacy-align]] source-volume A 와 **독립된 축**이다(과거 v2 output 볼륨의
    # stride 와 같다는 보장이 없어 별도 필드로 분리한다 — SPEC_IO_METRICS_V3 §7).
    reference_lock = {'profile_id': profile_id, 'expect_sha256': expect_sha256}
    if legacy_align['source'] == LEGACY_ALIGN_SOURCE_PAIRED:
        reference_lock['legacy_v2_manifest_sha256'] = legacy_align['v2_manifest_sha256']
        # ★**비권위 locator**(r1 [MED] 처분): identity 권위는 바로 위 SHA 하나뿐이고, 이 경로는
        # "그 SHA 를 가진 파일을 어디서 찾나"의 힌트일 뿐이다. 사양 4항 밖 필드인 이유는
        # "verifier 가 그 v2 manifest 를 strict 재개방한다"가 경로 없이는 실행 불가능해서다.
        # 파일이 옮겨졌으면 `--verify-only --legacy-v2-manifest <새 경로>` 로 갈아끼울 수 있고,
        # 그때도 수용 판정은 SHA·schema·model/sources/reference identity 전건 일치로만 한다.
        reference_lock['legacy_v2_manifest_path'] = legacy_align['v2_manifest_path']

    manifest = {
        'schema_version': SCHEMA_VERSION_V3,
        'mode': MODE_VIRTUAL,
        'model': model_dict,
        'sources': sources,
        'layout': {
            'align_bytes': A,
            'align_query': list(align_queries),
            'legacy_align_bytes': legacy_align['bytes'],
            'legacy_align_source': legacy_align['source'],
            'slot_stride_max': vplan['slot_stride_max'],
            'layers': manifest_layers,
        },
        'records': vplan['records'],
        'source_tensors': source_tensors,
        'quant_traits': quant_traits,
        'totals': {'virtual_payload_bytes': vplan['virtual_payload_bytes'],
                   'n_records': vplan['n_records']},
        'reference_lock': reference_lock,
        'tool': {'version': SCRIPT_VERSION, 'ts': datetime.now(timezone.utc).isoformat(),
                  'cmdline': ' '.join(sys.argv)},
    }
    return manifest


# ---------------------------------------------------------------------------
# ★§Z-⑥ mode=virtual `--plan` stdout 계약 (26-08-13 · 재팩 ⑤ 라운드)
#
# 위 bin 표(LAUNCHER_PLAN_KEYED_LINES)는 arch-template + bin 경로 전용이라 virtual 출력의 어떤
# 줄도 주장하지 않았다 — virtual stdout 은 지금까지 "mode=virtual 이라는 문자열이 있는가" 수준
# 으로만 검사됐고(v3-⑫ d1), 줄 삭제·문면 변경은 아무 관문도 건드리지 않았다. 아래 표가 virtual
# 소비 줄의 계약이며, selftest 가 **실 stdout** 으로 ①각 줄 정확히 1회 ②머리/완료 줄 위치
# ③필수 줄을 하나씩 뺀 입력의 거부(subtractive) 를 직접 주장한다.
#
# ★launcher 쪽 mode-aware 파서 시공은 이 라운드 스코프 **밖**이다(본작업 소비 표면 — 리드 판정
#   26-08-09). 여기서 동결하는 것은 생산 측 문면이고, 파서는 이 표를 사본으로 받는다.
#   bin 표의 6줄 중 template/expert_payload_total/output-alignment 3줄은 virtual 에 **없다**
#   (있으면 source-volume A 를 legacy output A 로 오독시킨다) — 그 대체가 `align`+`legacy` 2줄이다.
# ---------------------------------------------------------------------------
VIRTUAL_PLAN_KEYED_LINES = (
    ('mode',   r'^mode=virtual schema_version=(\S+) '),
    ('arch',   r'^arch=(\S+) n_layer=(\d+) n_expert=(\d+) n_expert_used=(\d+) schema=(\S+) bias=(\S+)$'),
    ('moe',    r'^moe_layers: (\d+) entries \[(\d+)\.\.(\d+)\]'),
    ('shards', r'^shards=(\d+) \(split=(\S+)\)$'),
    ('align',  r'^alignment A=(\d+) \(queried per source volume'),
    ('legacy', r'^legacy alignment \(D-A2 numerator axis, independent of A\): legacy_align_bytes=(\d+) '
               r'source=(\S+) paired_v2_sha256=(\S+)$'),
    ('slot',   r'^layer_slot_bytes min=(\d+) max=(\d+) \(.*\), slot_stride_max=(\d+) \(A-multiple=(\S+)\)$'),
    ('totals', r'^totals\.virtual_payload_bytes=(\d+) \(expect expert_bytes_total=(\d+)\) records=(\d+)$'),
)
VIRTUAL_PLAN_HEAD_LINE = 'mode: %s' % MODE_VIRTUAL


def virtual_plan_contract_problems(lines):
    """§Z-⑥ 계약 검사. 반환=문제 문자열 목록(빈 목록 = 계약 성립).

    파서가 실제로 하는 일과 같은 모양이다 — keyed 줄이 각각 정확히 1회, 머리 줄(`mode: virtual`)이
    그 앞에, 완료 줄이 맨 뒤. 그래서 이 함수는 테스트 보조가 아니라 **계약의 실행 가능한 사본**이고,
    subtractive 변이(필수 줄 1개 삭제)를 여기에 통과시키면 안 된다."""
    problems = []
    first_hit = {}
    for key, pat in VIRTUAL_PLAN_KEYED_LINES:
        hits = [i for i, ln in enumerate(lines) if re.match(pat, ln)]
        if len(hits) != 1:
            problems.append('"%s" line appears %d time(s) (expected exactly 1)' % (key, len(hits)))
        if hits:
            first_hit[key] = hits[0]
    head = [i for i, ln in enumerate(lines) if ln == VIRTUAL_PLAN_HEAD_LINE]
    done = [i for i, ln in enumerate(lines) if ln == LAUNCHER_PLAN_DONE_LINE]
    if len(head) != 1:
        problems.append('%r appears %d time(s) (expected exactly 1)' % (VIRTUAL_PLAN_HEAD_LINE, len(head)))
    if len(done) != 1:
        problems.append('%r appears %d time(s) (expected exactly 1)' % (LAUNCHER_PLAN_DONE_LINE, len(done)))
    if len(head) == 1 and len(done) == 1:
        for key, at in first_hit.items():
            if not (head[0] < at < done[0]):
                problems.append('"%s" line is outside the [mode header .. plan done] span' % key)
    return problems


def _print_virtual_plan_summary(model, layout, vplan, A, align_queries, profile_id, expect_sha256,
                                expect_totals, digests=None, derived=None, legacy_align=None):
    print('profile=%s expect_sha256=%s' % (profile_id, expect_sha256))
    print('mode=virtual schema_version=%s (experts.bin is NOT produced - 0 bytes of expert data move)'
          % SCHEMA_VERSION_V3)
    if derived is not None:
        d = derived['derivation']
        print('[EXPERIMENTAL arch-template] derived_from=%s routed_scope=%s (template default=%s) inventory_sha256=%s'
              % (d['derived_from'], d['scope'], d['default_scope'], d['inventory_sha256']))
    print('arch=%s n_layer=%d n_expert=%d n_expert_used=%d schema=%s bias=%s'
          % (layout['arch'], layout['n_layer'], layout['n_expert'], layout['n_expert_used'],
             layout['schema'], layout['has_bias']))
    print('moe_layers: %d entries [%d..%d]' % (len(layout['moe_layers']), layout['moe_layers'][0],
                                                layout['moe_layers'][-1]))
    print('shards=%d (split=%s)' % (len(model['shards']), model['is_split']))
    for h in model['shards']:
        dg = (digests or {}).get(h['source_index'])
        print('  shard[%d]: %s bytes=%d gguf_v=%d align=%d data_start=%d header_digest=%s'
              % (h['source_index'], os.path.basename(h['path']), h['file_bytes'], h['gguf_version'],
                 h['alignment'], h['data_start'], (dg['sha256'][:16] + '...') if dg else '(not computed)'))
    print('routed tensors=%d  used_types=%s' % (layout['n_routed'], layout['used_types']))
    print('alignment A=%d (queried per source volume - virtual has no output volume)' % A)
    for q in align_queries:
        print('  align_query[s%d]: root=%s method=%s logical=%d physical=%d fallback_reason=%s'
              % (q['source_index'], q['drive_root'], q['method'], q['logical'], q['physical'],
                 q['fallback_reason']))
    _la = legacy_align or resolve_legacy_align(None)
    print('legacy alignment (D-A2 numerator axis, independent of A): legacy_align_bytes=%d source=%s '
          'paired_v2_sha256=%s' % (_la['bytes'], _la['source'], _la['v2_manifest_sha256'] or '(none)'))
    if _la['source'] == LEGACY_ALIGN_SOURCE_CANONICAL:
        print('  [note] no paired v2 repack was supplied - published speed numbers for this model carry the '
              'SPEC_REPACK_V3 section 6-5 rebaselining footnote (pass --legacy-v2-manifest to bind a real v2 baseline)')
    slot_bytes = [VL['layer_slot_bytes'] for VL in vplan['layers']]
    n_parts = [len(VL['parts']) for VL in vplan['layers']]
    aligned_parts = sum(1 for VL in vplan['layers'] for vp in VL['parts'] if vp['aligned'])
    total_parts = sum(n_parts)
    print('layer_slot_bytes min=%d max=%d (%s), slot_stride_max=%d (A-multiple=%s)'
          % (min(slot_bytes), max(slot_bytes), 'uniform' if len(set(slot_bytes)) == 1 else 'varies per layer',
             vplan['slot_stride_max'], vplan['slot_stride_max'] % A == 0))
    print('parts per layer=%s, aligned parts=%d/%d (non-4K parts need a bounce staging buffer)'
          % (sorted(set(n_parts)), aligned_parts, total_parts))
    staging = [vp['staging_bytes'] for VL in vplan['layers'] for vp in VL['parts'] if 'staging_bytes' in vp]
    if staging:
        print('staging requirement (non-aligned parts): max=%d B (align_up(slice,A)+A)' % max(staging))
    print('totals.virtual_payload_bytes=%d (expect expert_bytes_total=%d) records=%d'
          % (vplan['virtual_payload_bytes'], expect_totals['expert_bytes_total'], vplan['n_records']))
    print('disk footprint: source %d B x 1.0 (no experts.bin, no transient copy)'
          % sum(h['file_bytes'] for h in model['shards']))
    print('preflight RAM / free-space checks: not applicable (0 bytes of expert data are moved)')
    VL0 = vplan['layers'][0]
    print('--- moe layer %d slot parts (name, src, type, abs_offset, slice, aligned, bracket_head, '
          'slot_offset, data_offset) ---' % VL0['layer'])
    for vp in VL0['parts']:
        print('  %-12s [s%d] type=%-6s abs=%-14d slice=%-12d aligned=%-5s head=%-6s slot=%-12d data=%d'
              % (vp['name'], vp['source_index'], vp['type'], vp['abs_offset'], vp['slice_bytes'],
                 vp['aligned'], vp.get('bracket_head', '-'), vp['slot_offset'], vp['data_offset']))


def _prepare_virtual(model_path, profile_id, allow_default_align, enforce_reference, scope='all',
                     arch_template=False):
    """virtual 공통 준비: shard 로드 → layout 재도출(v2 §2 검증 사슬 전부 승계) → (참조 락)
    expect 로드·대조 → A(source 볼륨 질의)·슬롯 산술. 반환 8-tuple."""
    model = load_model_shards(model_path)
    derived = None
    if arch_template:
        derivation = derive_arch_template(model, requested_scope=scope)
        scope = derivation['scope']
    else:
        scope = scope or 'all'
    layout = build_layout(model, scope=scope)
    A, align_queries = resolve_alignment_for_sources(model, allow_default_align)
    vplan = compute_virtual_layout(layout, A, {h['source_index']: h['file_bytes'] for h in model['shards']})
    if arch_template:
        expect, expect_raw, expect_sha256 = build_derived_expect(model, layout, derivation)
        expect_totals = cross_check_expect(model, layout, vplan, expect, scope=scope)
        derived = {'derivation': derivation, 'expect': expect, 'raw': expect_raw,
                   'sha256': expect_sha256, 'lock_id': arch_template_lock_id(derivation)}
    elif enforce_reference:
        expect, expect_sha256 = load_expect_profile(profile_id)
        expect_totals = cross_check_expect(model, layout, vplan, expect, scope=scope)
    else:
        expect_sha256 = 'selftest-exempt'
        expect_totals = {'expert_bytes_total': layout['n_expert'] * sum(L['payload_bytes'] for L in layout['layers'])}
    return model, layout, A, align_queries, vplan, expect_sha256, expect_totals, derived


# ---------------------------------------------------------------------------
# manifest v3 내부 불변식(§2-2·§2-4) — 로드된 manifest 자체 일관성.
# 독립 재도출 대조(verify_virtual_manifest)와 **별개**의 방어층. 크래시 금지(problems 수집).
# ---------------------------------------------------------------------------
def _is_u64(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= U64_MAX


def _check_virtual_manifest_mode(manifest, problems):
    """§2-1 모드·스키마 조합 게이트(생산 금지 조합·누락·미지 값·불일치 전부 거부)."""
    sv = manifest.get('schema_version')
    has_mode = 'mode' in manifest
    mode = manifest.get('mode')
    if sv != SCHEMA_VERSION_V3:
        problems.append('schema_version(%r) != %r - a mode=virtual manifest is required (v2 "2.0" artifacts are '
                        'read by the bin verifier)' % (sv, SCHEMA_VERSION_V3))
    if not has_mode:
        problems.append('mode field is missing (mandatory when schema_version is %r)' % SCHEMA_VERSION_V3)
    elif mode == MODE_BIN:
        problems.append('forbidden combination: schema_version %r with mode %r (bin is always %r and carries no '
                        'mode field)' % (sv, MODE_BIN, SCHEMA_VERSION))
    elif mode != MODE_VIRTUAL:
        problems.append('unknown mode %r (expected %r)' % (mode, MODE_VIRTUAL))


def _check_virtual_forbidden_fields(manifest, problems):
    """§2-2 bin 전용 좌표 필드 금지(존재=중단)."""
    totals = manifest.get('totals') if isinstance(manifest.get('totals'), dict) else {}
    if 'bin_file_bytes' in totals:
        problems.append('forbidden bin-only field: totals.bin_file_bytes')
    m_layout = manifest.get('layout') if isinstance(manifest.get('layout'), dict) else {}
    layers = m_layout.get('layers') if isinstance(m_layout.get('layers'), list) else []
    for i, L in enumerate(layers):
        if not isinstance(L, dict):
            continue
        for bad in ('record_base', 'stride_bytes', 'payload_bytes'):
            if bad in L:
                problems.append('forbidden bin-only field: layout.layers[%d].%s' % (i, bad))
        for j, p in enumerate(L.get('parts') if isinstance(L.get('parts'), list) else []):
            if not isinstance(p, dict):
                continue
            for bad in ('part_offset', 'part_bytes'):
                if bad in p:
                    problems.append('forbidden bin-only field: layout.layers[%d].parts[%d].%s' % (i, j, bad))


def _check_virtual_manifest_invariants(manifest, problems):
    """§2-2·§2-4 자기일관 재계산: 슬롯 점화식·정렬 상수·records 전항 재생성·witness 대조."""
    try:
        model = manifest.get('model') if isinstance(manifest.get('model'), dict) else {}
        m_layout = manifest.get('layout') if isinstance(manifest.get('layout'), dict) else {}
        layers = m_layout.get('layers') if isinstance(m_layout.get('layers'), list) else []
        records = manifest.get('records') if isinstance(manifest.get('records'), list) else []
        source_tensors = manifest.get('source_tensors') if isinstance(manifest.get('source_tensors'), list) else []
        quant_traits = manifest.get('quant_traits') if isinstance(manifest.get('quant_traits'), dict) else {}
        totals = manifest.get('totals') if isinstance(manifest.get('totals'), dict) else {}
        sources = manifest.get('sources') if isinstance(manifest.get('sources'), list) else []
        n_expert = model.get('n_expert')
        A = m_layout.get('align_bytes')

        if not _is_u64(A) or A <= 0 or (A & (A - 1)) != 0 or A < 4096:
            problems.append('invariant v3-1: layout.align_bytes(%r) is not a power of two >= 4096' % (A,))
            return
        if not _is_u64(n_expert) or n_expert <= 0:
            problems.append('invariant v3-1: model.n_expert(%r) is not a positive integer' % (n_expert,))
            return

        # align_query[] ↔ sources[] 결속 + A 재도출
        aq = m_layout.get('align_query') if isinstance(m_layout.get('align_query'), list) else None
        if aq is None or not aq:
            problems.append('invariant v3-2: layout.align_query is missing or empty')
        else:
            aq_idx = [q.get('source_index') for q in aq if isinstance(q, dict)]
            src_idx = [s.get('index') for s in sources if isinstance(s, dict)]
            if aq_idx != src_idx:
                problems.append('invariant v3-2: layout.align_query source_index list %r != sources index list %r'
                                % (aq_idx, src_idx))
            derived_A = 4096
            bad = False
            for q in aq:
                if not isinstance(q, dict) or not _is_u64(q.get('logical')) or not _is_u64(q.get('physical')):
                    problems.append('invariant v3-2: align_query entry is malformed: %r' % (q,))
                    bad = True
                    continue
                derived_A = max(derived_A, q['logical'], q['physical'])
            if not bad and derived_A != A:
                problems.append('invariant v3-2: align_bytes(%d) != max(4096, align_query logical/physical)(%d)'
                                % (A, derived_A))

        # layers[].layer 유일·오름차순·model.moe_layers 와 타입 포함 전항 동일
        layer_ids = [L.get('layer') for L in layers if isinstance(L, dict)]
        if len(layer_ids) != len(set(layer_ids)):
            problems.append('invariant v3-3: duplicate layout.layers[].layer')
        if layer_ids != sorted(x for x in layer_ids if isinstance(x, int)):
            problems.append('invariant v3-3: layout.layers[].layer is not ascending')
        if not _typed_eq(layer_ids, model.get('moe_layers') if isinstance(model.get('moe_layers'), list) else None):
            problems.append('invariant v3-3: layout.layers[].layer != model.moe_layers (type included)')

        # 슬롯 점화식·정렬 상수·data_offset·staging (§2-4)
        slot_bytes = []
        ref_part_names = None
        payload_total = 0
        flat = []
        for i, L in enumerate(layers):
            if not isinstance(L, dict):
                problems.append('invariant v3-4: layout.layers[%d] is not an object' % i)
                continue
            parts = L.get('parts') if isinstance(L.get('parts'), list) else []
            if not parts:
                problems.append('invariant v3-4: layer %r has no parts' % L.get('layer'))
                continue
            names = [p.get('name') for p in parts if isinstance(p, dict)]
            stensors = [p.get('source_tensor') for p in parts if isinstance(p, dict)]
            if len(names) != len(set(names)):
                problems.append('invariant v3-4: layer %r duplicate part name' % L.get('layer'))
            if len(stensors) != len(set(stensors)):
                problems.append('invariant v3-4: layer %r duplicate source_tensor' % L.get('layer'))
            if ref_part_names is None:
                ref_part_names = list(names)
            elif names != ref_part_names:
                problems.append('invariant v3-4: layer %r part name order %r != first layer %r'
                                % (L.get('layer'), names, ref_part_names))
            running = 0
            ok_chain = True
            for j, p in enumerate(parts):
                if not isinstance(p, dict):
                    problems.append('invariant v3-4: layer %r parts[%d] is not an object' % (L.get('layer'), j))
                    ok_chain = False
                    break
                where = 'layer %r part %r' % (L.get('layer'), p.get('name'))
                sl, ao = p.get('slice_bytes'), p.get('abs_offset')
                so, do = p.get('slot_offset'), p.get('data_offset')
                if not (_is_u64(sl) and sl > 0) or not _is_u64(ao) or not _is_u64(so) or not _is_u64(do):
                    problems.append('invariant v3-4: %s has a non-uint64 slice_bytes/abs_offset/slot_offset/'
                                    'data_offset (%r/%r/%r/%r)' % (where, sl, ao, so, do))
                    ok_chain = False
                    break
                aligned = p.get('aligned')
                if aligned is not (sl % A == 0):
                    problems.append('invariant v3-4: %s aligned(%r) != (slice_bytes %% A == 0)(%r)'
                                    % (where, aligned, sl % A == 0))
                    ok_chain = False
                    break
                if so != running:
                    problems.append('invariant v3-4: %s slot_offset(%d) != recurrence value(%d)' % (where, so, running))
                    ok_chain = False
                    break
                if so % A != 0:
                    problems.append('invariant v3-4: %s slot_offset(%d) is not a multiple of A(%d)' % (where, so, A))
                    ok_chain = False
                    break
                if aligned:
                    head = p.get('bracket_head')
                    if not _is_u64(head) or head >= A:
                        problems.append('invariant v3-4: %s bracket_head(%r) is not in [0, A)' % (where, head))
                        ok_chain = False
                        break
                    if head != ao % A:
                        problems.append('invariant v3-4: %s bracket_head(%d) != abs_offset %% A(%d)'
                                        % (where, head, ao % A))
                        ok_chain = False
                        break
                    region = ((head + sl + A - 1) // A) * A
                    want_do = so + head
                    if 'staging_bytes' in p:
                        problems.append('invariant v3-4: %s is aligned but carries staging_bytes '
                                        '(bounce staging is for non-4K parts only)' % where)
                else:
                    if 'bracket_head' in p:
                        problems.append('invariant v3-4: %s is not aligned but carries bracket_head '
                                        '(undefined for non-4K parts - must not be recorded)' % where)
                    region = ((sl + A - 1) // A) * A
                    want_do = so
                    want_staging = region + A
                    if p.get('staging_bytes') != want_staging:
                        problems.append('invariant v3-4: %s staging_bytes(%r) != align_up(slice,A)+A(%d)'
                                        % (where, p.get('staging_bytes'), want_staging))
                if do != want_do:
                    problems.append('invariant v3-4: %s data_offset(%d) != %d' % (where, do, want_do))
                    ok_chain = False
                    break
                if not _is_u64(p.get('expert_axis')) or not isinstance(p.get('dims'), list) \
                        or not p.get('dims') or p['expert_axis'] != len(p['dims']) - 1:
                    problems.append('invariant v3-4: %s expert_axis(%r)/dims(%r) violate "expert axis is last"'
                                    % (where, p.get('expert_axis'), p.get('dims')))
                if p['dims'][-1] != n_expert:
                    problems.append('invariant v3-4: %s dims[-1](%r) != model.n_expert(%d)'
                                    % (where, p['dims'][-1], n_expert))
                payload_total += sl * n_expert
                flat.append(p)
                running += region
            if ok_chain:
                if L.get('layer_slot_bytes') != running:
                    problems.append('invariant v3-4: layer %r layer_slot_bytes(%r) != sum(region)(%d)'
                                    % (L.get('layer'), L.get('layer_slot_bytes'), running))
                slot_bytes.append(running)
        if slot_bytes:
            if m_layout.get('slot_stride_max') != max(slot_bytes):
                problems.append('invariant v3-5: slot_stride_max(%r) != max(layer_slot_bytes)(%d)'
                                % (m_layout.get('slot_stride_max'), max(slot_bytes)))
            if max(slot_bytes) % A != 0:
                problems.append('invariant v3-5: max(layer_slot_bytes)(%d) is not a multiple of A(%d)'
                                % (max(slot_bytes), A))

        # totals
        if totals.get('virtual_payload_bytes') != payload_total:
            problems.append('invariant v3-6: totals.virtual_payload_bytes(%r) != sum(slice_bytes*n_expert)(%d)'
                            % (totals.get('virtual_payload_bytes'), payload_total))
        want_n_records = sum(n_expert * len(L.get('parts') or ()) for L in layers if isinstance(L, dict))
        if totals.get('n_records') != want_n_records:
            problems.append('invariant v3-6: totals.n_records(%r) != sum_layers(n_expert*n_parts)(%d)'
                            % (totals.get('n_records'), want_n_records))
        if len(records) != want_n_records:
            problems.append('invariant v3-6: len(records)(%d) != sum_layers(n_expert*n_parts)(%d)'
                            % (len(records), want_n_records))

        # records[] 전항 재생성(순서=(layer↑, expert↑, part 배열순))
        idx = 0
        stop = False
        for L in layers:
            if stop or not isinstance(L, dict):
                continue
            parts = L.get('parts') if isinstance(L.get('parts'), list) else []
            for e in range(n_expert):
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    if idx >= len(records):
                        problems.append('invariant v3-7: too few records (idx=%d)' % idx)
                        stop = True
                        break
                    rec = records[idx]
                    if not isinstance(rec, dict):
                        problems.append('invariant v3-7: records[%d] is not an object' % idx)
                        stop = True
                        break
                    want = [L.get('layer'), e, p.get('name'), p.get('source_index'),
                            (p.get('abs_offset') + e * p.get('slice_bytes'))
                            if (_is_u64(p.get('abs_offset')) and _is_u64(p.get('slice_bytes'))) else None,
                            p.get('slice_bytes'), p.get('data_offset')]
                    got = [rec.get('layer'), rec.get('expert'), rec.get('part'), rec.get('source_index'),
                           rec.get('src_offset'), rec.get('slice_bytes'), rec.get('data_offset')]
                    if not _typed_eq(got, want):
                        problems.append('invariant v3-7: records[%d](%r) != value derived from parts(%r)'
                                        % (idx, got, want))
                        stop = True
                        break
                    idx += 1
                if stop:
                    break
        if not stop and idx != len(records):
            problems.append('invariant v3-7: too many records (derived %d, actual %d)' % (idx, len(records)))

        # source_tensors[] = 비권위 witness(name·source_index·type·dims·bytes + abs_offset 전항)
        if len(flat) != len(source_tensors):
            problems.append('invariant v3-8: source_tensors count(%d) != flattened parts(%d)'
                            % (len(source_tensors), len(flat)))
        else:
            for i, (st, p) in enumerate(zip(source_tensors, flat)):
                if not isinstance(st, dict):
                    problems.append('invariant v3-8: source_tensors[%d] is not an object' % i)
                    continue
                got = [st.get('name'), st.get('source_index'), st.get('abs_offset'), st.get('bytes'),
                       st.get('type'), st.get('dims')]
                want = [p.get('source_tensor'), p.get('source_index'), p.get('abs_offset'),
                        (p.get('slice_bytes') * n_expert) if _is_u64(p.get('slice_bytes')) else None,
                        p.get('type'), p.get('dims')]
                if not _typed_eq(got, want):
                    problems.append('invariant v3-8: source_tensors[%d](%r) != flattened part(%r)' % (i, got, want))

        # quant_traits == 동결 표(사용 subset·산술 권위 아님)
        used = sorted({p.get('type') for p in flat})
        if sorted(quant_traits.keys()) != used:
            problems.append('invariant v3-9: quant_traits key set %r != part types %r'
                            % (sorted(quant_traits), used))
        for tt, tr in quant_traits.items():
            if tt not in QUANT_TRAITS:
                problems.append('invariant v3-9: quant_traits type not in the frozen table %r' % tt)
                continue
            bv, bb = QUANT_TRAITS[tt]
            if not _typed_eq(tr, {'block_values': bv, 'block_bytes': bb}):
                problems.append('invariant v3-9: quant_traits[%s] != frozen table(%d/%d)' % (tt, bv, bb))
    except Exception as e:
        problems.append('structure corrupt during the v3 invariant check: %r' % e)


# ---------------------------------------------------------------------------
# §4-3 독립 verifier — candidate manifest 를 strict JSON 으로 **재로드**(producer in-memory
# 객체 재사용 금지)하고, 전 shard 를 독립 재오픈·재파싱해 **주소값을 독립 재유도**한다
# (§2 전 불변식 + §2-4 산술 검산 포함). ★**전항 재도출이 아니다** — 아래 머리 주석의 축소
# 계약(선언된 공통 신뢰원 4종은 계약 밖)이 이 경로의 실제 보장 범위다. [[C:repack.trust-sources]]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# §4-3 검증 전용 독립 재유도 (부속 정오 2 §Z-② — 독립성 계약 이행)
#
# 구 verifier 는 "독립 재도출"을 표방하면서 플래너와 **같은 3함수**를 재호출했다
# (load_model_shards → build_layout → compute_virtual_layout). 그 함수들이 좌표를 잘못 내면
# 생산과 검증이 같은 틀린 값을 내므로 전항 대조가 발화하지 않는다 — 검출력이 0인 구간이 생긴다.
# ★★계약(정오 2 · Codex r7 판정으로 **축소 확정**): 이 경로가 보장하는 것은
#   **"공통 경로가 승인한 입력에서, 주소값을 독립 재유도한다"**
# 까지다. "모든 검증 규칙을 2벌로 재현한다"가 **아니다**. 구 문면("스펙 상수만 공유")은
# 사실이 아니었고 r7 이 반증했다 — 아래 **선언된 공통 신뢰원**이 실재하기 때문이다.
#
# ★**선언된 공통 신뢰원 4종**(= 이 경로가 독립 재유도하지 **않는** 입력. 숨기지 않고 명시한다.
#   ★이 목록·문면은 정오 2 ⓙ 의 계약 문안과 동기화되어 있다 — r9 N1. 한쪽만 고치지 말 것.
#   ★원본=`SPEC_REPACK_V3.md` §4 3항 [[C:repack.trust-sources]]):
#   1. **정렬 상수 `A`** — `resolve_alignment_for_sources()` 의 결과를 그대로 받는다.
#      A 는 재유도 가능한 좌표가 아니라 **볼륨에 묻는 런타임 질의 결과**라, 같은 구현이 같은
#      OS 에 두 번 물어도 질의 자체의 오해석은 잡히지 않는다. ⇒ 공통 질의가 잘못된 A 를 내면
#      생산·검증이 **같은 잘못된 슬롯·EOF** 를 낸다. 이건 알려진 신뢰 가정이며 selftest
#      v3-⑬ 이 그 경계를 **실행 가능한 형태로 고정**한다(무엇이 안 잡히는지 + 무엇이 잡히는지 —
#      r9 M3 이후 **7항 개별 assertion**: 슬롯 주소·bracket EOF 직접 계산·records 주소가 각각
#      상이함을 따로 세우고, 회수 쪽도 파트 주소와 레코드 주소를 각각 요구한다).
#      ★완화 장치는 소비 시점에 있다: 질의 결과가 `layout.align_query[]` 에 provenance 로 남고,
#      검증이 **재질의해 주소 관련 4필드(`source_index`·`method`·`logical`·`physical`)와 drive
#      root 를 대조**한다(§(7)). ★**대조 범위 정직 표기(r8 M2)**: `fallback_reason` 은 비교하지
#      않는다(정상 환경에서 문구가 갈릴 여지가 있어 오탐 원천) — "같은 method·같은 수치인데
#      fallback 사유만 다른" 경우는 이 대조가 잡지 않는다. [[C:repack.align-query]]
#      **소비(child 기동) 시점 방어는 아직 미시공**이며 §Z 후속 계약이다.
#   2. `source_header_digest()` · `reference_lock` 재확인(카탈로그 재해시·derived expect 재유도)
#      — 이 둘은 **주소 기대값을 공급하지 않는다**(r7: `cross_check_expect` 는 주소·slot·records
#      를 대조하지 않는다). 따라서 §Z-② 잔여가 아니다.
#   3. 공통 경로에만 있는 **유효성 검사**. ★**완전 열거가 아니라 예시다(r9 M1)** — 규범은
#      "축소된 계약은 공통 전용 유효성 검사의 2벌 구현을 요구하지 않는다"이고, 대표 항목은
#      offset-gap·arch 충돌·split KV·fused/separate·bias 집합·**shard 간 nextn 값 충돌**·전역
#      tensor-name 중복·`n_expert < 0xFFFF`·routed layer `< n_layer`.
#      ★**`nextn 범위`는 이 범주에서 뺀다(r8 M1 — 구 문면은 과잉 양보였다)**: 독립 경로도
#      execution scope 에서 KV 부재·0 을 거부하고, `nextn >= n_layer` 면 모든 비음수 layer 가
#      제거돼 `no routed layer remains` 로 스스로 중단한다. 공통 **전용** 잔여는 shard 간 값
#      충돌 검사뿐이다. 이 범주 때문에 §Z-② 는 **영구히 "부분"**이다.
#   4. ★**동결된 선언적 규칙표·정규식**(r8 M1 · REACHABLE) — 두 구현이 **같은 표를 본다**:
#      TENSOR_NAME_RE · KIND_PRIORITY · QUANT_TRAITS · SCALAR_FMT · GGML_TYPE · SPLIT_RE ·
#      ★**GGUF value-type tag mapping**(`T_*` 번호 선언 — 특히 `T_STR`·`T_ARR`. r9 M1 추가:
#      공통 파서와 `_vfy_read_gguf_header()` 가 **둘 다** 이 선언으로 KV 값의 형을 해석하고
#      문자열·배열을 건너뛴다. 값이 지금 틀렸다는 증거는 없지만 **모든 정상 GGUF 가 매 파싱마다
#      쓰므로** 공유 의존 범위에서 뺄 수 없다). 이것들은 진단용 상수가 아니라 shard 발견·routed
#      선별·part 순서·type 해석·slice_bytes 에 직접 들어가므로, 표나 정규식이 틀리면 **모든 정상
#      입력에서 양쪽이 같은 잘못된 tensor 집합·순서·slice** 를 낸다(손상 입력 불요). 규칙의
#      동일성은 의도이므로 값은 같아야 하지만, **그 규칙 자체의 오류는 이 계약이 잡지 않는다**.
#   부기: `model.routed_scope` 는 verifier 의 독립 산출값이 아니라 **manifest 가 고른 재유도
#   영역**이다 — "공통 경로가 승인한 입력"에 포함되는 **제어 입력**이며, 그 값의 결속은 별도
#   회귀(scope 키 삭제 거부·공통 선별 오도출 적발)가 담당한다.
#
# 그 위에서 **주소값**은 별도 코드로 다시 낸다. 규칙이 같으니 값은 같아야 하고, 코드가 다르니
# 한쪽의 오도출이 상쇄되지 않는다. 도출 경로를 일부러 다르게 잡은 지점:
#   · 헤더: 순차 read 대신 **절대 오프셋 커서**로 걷는다(문자열 배열은 길이만 누적해 건너뜀)
#   · data_start 패딩: `(-header_end) % align`  (공통은 ceil 나눗셈)
#   · slot_offset: 점화식이 아니라 **region prefix 합**
# ---------------------------------------------------------------------------
def _vfy_split_siblings(model_path):
    """검증 전용 형제 shard 발견. ⑤-1 과 같은 이유로 절대경로만 생산한다."""
    p = os.path.abspath(model_path)
    m = SPLIT_RE.match(os.path.basename(p))
    if not m:
        return [p]
    cnt = int(m.group('cnt'))
    if cnt < 1:
        raise RepackAbort('verifier: split count 0: %s' % os.path.basename(p))
    d, prefix = os.path.dirname(p), m.group('base')
    out = []
    for i in range(1, cnt + 1):
        sib = os.path.join(d, '%s-%05d-of-%05d.gguf' % (prefix, i, cnt))
        if not os.path.exists(sib):
            raise RepackAbort('verifier: split sibling shard missing: %s (%d expected)' % (sib, cnt))
        out.append(sib)
    return out


def _vfy_read_gguf_header(path):
    """검증 전용 GGUF 헤더 재유도(쓰기 없음). 반환 필드는 공통 파서와 같은 뜻이지만 산출 경로가 다르다."""
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        # 실물 모델의 KV 영역은 토크나이저 문자열 배열이 대부분이라 스칼라마다 seek 하면
        # 어휘 수만큼 syscall 이 난다. 읽기 창을 하나 들고 그 안이면 잘라 쓴다(순수 성능 장치 —
        # 반환 바이트는 파일 그대로다).
        # 'read' = 헤더 파서가 실제 파일에서 읽은 바이트(N4 계측). ★r8 N3: 이름이 "헤더 크기"로
        # 읽히지 않게 한다 — 마지막 64KiB read-ahead 는 payload 를 물 수 있어 unique 헤더 크기가
        # 아니다. 재유도 비용(파일에서 읽은 총량)이 이 값의 뜻이다.
        win = {'at': 0, 'buf': b'', 'read': 0}

        def rd(at, n):
            if at < 0 or n < 0 or at + n > size:
                raise RepackAbort('verifier: GGUF read out of range (%s at %r+%r, file=%d)' % (path, at, n, size))
            base, buf = win['at'], win['buf']
            if at >= base and at + n <= base + len(buf):
                return buf[at - base:at - base + n]
            span = min(max(n, 1 << 16), size - at)
            fh.seek(at)
            b = fh.read(span)
            if len(b) < n:
                raise RepackAbort('verifier: short GGUF read (%s at %d, wanted %d, got %d)' % (path, at, n, len(b)))
            win['at'], win['buf'] = at, b
            win['read'] += len(b)
            return b[:n]

        def scalar(at, fmt):
            return struct.unpack_from('<' + fmt, rd(at, struct.calcsize(fmt)), 0)[0]

        def skip_value(at, vtype):
            """(다음 오프셋, 값 or None). 배열은 값을 만들지 않고 길이만 넘긴다(vocab 전개 회피)."""
            if vtype == T_STR:
                n = scalar(at, 'Q')
                return at + 8 + n, rd(at + 8, n).decode('utf-8', errors='replace')
            if vtype == T_ARR:
                etype, cnt = scalar(at, 'I'), scalar(at + 4, 'Q')
                at += 12
                if etype == T_STR:
                    for _ in range(cnt):
                        at += 8 + scalar(at, 'Q')
                    return at, None
                if etype not in SCALAR_FMT:
                    raise RepackAbort('verifier: unknown GGUF array element type %r (%s)' % (etype, path))
                return at + cnt * struct.calcsize(SCALAR_FMT[etype]), None
            if vtype not in SCALAR_FMT:
                raise RepackAbort('verifier: unknown GGUF value type %r (%s)' % (vtype, path))
            fmt = SCALAR_FMT[vtype]
            return at + struct.calcsize(fmt), scalar(at, fmt)

        if rd(0, 4) != b'GGUF':
            raise RepackAbort('verifier: GGUF magic mismatch (%s)' % path)
        ver, n_tensors, n_kv = scalar(4, 'I'), scalar(8, 'Q'), scalar(16, 'Q')

        meta = {}
        pos = 24
        for _ in range(n_kv):
            klen = scalar(pos, 'Q')
            key = rd(pos + 8, klen).decode('utf-8', errors='replace')
            at = pos + 8 + klen
            pos, val = skip_value(at + 4, scalar(at, 'I'))
            meta[key] = val

        entries = []
        for _ in range(n_tensors):
            nlen = scalar(pos, 'Q')
            name = rd(pos + 8, nlen).decode('utf-8', errors='replace')
            pos += 8 + nlen
            nd = scalar(pos, 'I')
            pos += 4
            dims = list(struct.unpack_from('<%dQ' % nd, rd(pos, 8 * nd), 0)) if nd else []
            pos += 8 * nd
            tcode = scalar(pos, 'I')
            rel = scalar(pos + 4, 'Q')
            pos += 12
            entries.append({'name': name, 'dims': dims,
                            'type': GGML_TYPE.get(tcode, 'type_%d' % tcode), 'rel_offset': rel})
        header_end = pos

    align_raw = meta.get('general.alignment', 32)
    if isinstance(align_raw, bool) or not isinstance(align_raw, int):
        raise RepackAbort('verifier: general.alignment is not an integer (%r) in %s' % (align_raw, path))
    align = int(align_raw)
    if align <= 0:
        raise RepackAbort('verifier: general.alignment(%d) is not positive (%s)' % (align, path))
    # 텐서 0개 shard 는 upstream 이 데이터 정렬 seek 자체를 건너뛴다(gguf.cpp:756) — 공통 파서와
    # 같은 규칙, 다른 식: 여기서는 패딩량을 음수 모듈로로 낸다.
    data_start = header_end + ((-header_end) % align if n_tensors else 0)
    data_region = size - data_start
    if data_region < 0:
        raise RepackAbort('verifier: data_start(%d) exceeds the file size(%d): %s' % (data_start, size, path))
    ordered = sorted(entries, key=lambda e: e['rel_offset'])
    for i, e in enumerate(ordered):
        nxt = ordered[i + 1]['rel_offset'] if i + 1 < len(ordered) else data_region
        e['gap'] = nxt - e['rel_offset']
        e['abs_offset'] = data_start + e['rel_offset']
    return {'path': path, 'file_bytes': size, 'gguf_version': ver, 'alignment': align,
            'data_start': data_start, 'data_region': data_region, 'meta': meta, 'tensors': ordered,
            'header_parser_file_bytes_read': win['read']}


def _vfy_slice_bytes(ttype, dims):
    """검증 전용 per-expert slice 산술(§2-3). 공통 구현과 달리 누적 곱으로 낸다."""
    traits = QUANT_TRAITS.get(ttype)
    if traits is None:
        raise RepackAbort('verifier: type not in the type-trait table (fail-closed): %s dims=%r' % (ttype, dims))
    bv, bb = traits
    if len(dims) < 2:
        raise RepackAbort('verifier: routed tensor needs an expert axis: %s dims=%r' % (ttype, dims))
    if dims[0] % bv != 0:
        raise RepackAbort('verifier: ne0(%d) %% block_values(%d) != 0: type=%s' % (dims[0], bv, ttype))
    n = (dims[0] // bv) * bb
    for d in dims[1:-1]:
        n *= d
    return n


def _vfy_record_order(part_names):
    """검증 전용 레코드 내부 순서(weights 우선순위순 → bias 동순)."""
    weights = sorted((p for p in part_names if p.endswith('.weight')),
                     key=lambda q: KIND_PRIORITY[q.rsplit('.', 1)[0]])
    biases = sorted((p for p in part_names if p.endswith('.bias')),
                    key=lambda q: KIND_PRIORITY[q.rsplit('.', 1)[0]])
    return weights + biases


def _vfy_derive_virtual(model_path, A, scope='all'):
    """검증 전용 전체 재유도: shard 좌표 → routed 선별·순서 → 슬롯/EOF 산술 → records[].

    반환 구조는 대조 편의를 위해 vplan 과 같은 키를 쓰지만, 값은 위 `_vfy_*` 만으로 만들어진다."""
    # ★r8 N2: 경과는 단조시계로 잰다. `time.time()` 은 벽시계라 NTP 보정·서머타임이 구간 중에
    # 걸리면 음수·비약이 나오고, 이 값은 "대형 모델에서 재유도가 얼마나 드는가"를 판단하는
    # 재료라 그런 수치가 섞이면 안 된다.
    _t0 = time.perf_counter()
    shards = []
    for idx, p in enumerate(_vfy_split_siblings(model_path)):
        h = _vfy_read_gguf_header(p)
        h['source_index'] = idx
        shards.append(h)

    meta = {}
    for h in shards:                                  # first-wins 병합(공통 규약과 동일)
        for k, v in h['meta'].items():
            meta.setdefault(k, v)
    arch = meta.get('general.architecture')
    if not isinstance(arch, str) or not arch:
        raise RepackAbort('verifier: general.architecture is missing or not a string (%r)' % (arch,))

    def need(key):
        if key not in meta:
            raise RepackAbort('verifier: exact metadata key missing (fail-closed): %s' % key)
        return int(meta[key])

    n_expert, n_layer, n_expert_used = (need('%s.expert_count' % arch), need('%s.block_count' % arch),
                                        need('%s.expert_used_count' % arch))
    if n_expert <= 0:
        raise RepackAbort('verifier: n_expert(%d) must be positive' % n_expert)

    routed = {}
    for h in shards:
        for t in h['tensors']:
            m = TENSOR_NAME_RE.match(t['name'])
            if not m:
                continue
            layer, kind = int(m.group(1)), '%s.%s' % (m.group(2), m.group(3))
            slot = routed.setdefault(layer, {})
            if kind in slot:
                raise RepackAbort('verifier: layer %d part %s is duplicated across shards' % (layer, kind))
            slot[kind] = (t, h)
    if not routed:
        raise RepackAbort('verifier: no routed tensor found (regex matched nothing)')

    if scope == 'execution':
        key = '%s.nextn_predict_layers' % arch
        nextn = int(meta[key]) if key in meta else 0
        if not nextn:
            raise RepackAbort('verifier: scope=execution but %s is absent or 0' % key)
        routed = dict((l, v) for l, v in routed.items() if l < n_layer - nextn)
        if not routed:
            raise RepackAbort('verifier: no routed layer remains after the execution-scope exclusion')

    moe_layers = sorted(routed)
    ref_names = set(routed[moe_layers[0]])
    order = _vfy_record_order(ref_names)

    layers, records, used_types = [], [], set()
    payload_total, slot_stride_max = 0, 0
    for l in moe_layers:
        got = routed[l]
        if set(got) != ref_names:
            raise RepackAbort('verifier: layer %d part set %s != layer %d %s'
                              % (l, sorted(got), moe_layers[0], sorted(ref_names)))
        staged, regions = [], []
        for kind in order:
            t, h = got[kind]
            dims = list(t['dims'])
            if dims[-1] != n_expert:
                raise RepackAbort('verifier: expert_axis violation (ne[last]!=n_expert): %s dims=%r n_expert=%d'
                                  % (t['name'], dims, n_expert))
            slice_bytes = _vfy_slice_bytes(t['type'], dims)
            if slice_bytes <= 0:
                raise RepackAbort('verifier: slice_bytes must be positive: layer=%d part=%s' % (l, kind))
            abs_offset, src_bytes = t['abs_offset'], h['file_bytes']
            tensor_bytes = slice_bytes * n_expert
            if abs_offset + tensor_bytes > src_bytes:
                raise RepackAbort('verifier: routed tensor payload exceeds the source EOF: %s [s%d] '
                                  'abs_offset=%d n_expert*slice=%d source.bytes=%d'
                                  % (t['name'], h['source_index'], abs_offset, tensor_bytes, src_bytes))
            if -((-(abs_offset + tensor_bytes)) // A) * A > src_bytes:
                raise RepackAbort('verifier: bracket EOF violation (section 2-4) - this profile does not close '
                                  'as virtual: %s [s%d] abs_offset=%d slice=%d n_expert=%d A=%d source.bytes=%d'
                                  % (t['name'], h['source_index'], abs_offset, slice_bytes, n_expert, A, src_bytes))
            aligned = (slice_bytes % A == 0)
            head = (abs_offset % A) if aligned else None
            span = (head + slice_bytes) if aligned else slice_bytes
            region = -((-span) // A) * A          # align_up(span, A) — ceil 나눗셈을 쓰지 않는 형태
            used_types.add(t['type'])
            staged.append({'name': kind, 'source_tensor': t['name'], 'source_index': h['source_index'],
                           'type': t['type'], 'dims': dims, 'expert_axis': len(dims) - 1,
                           'abs_offset': abs_offset, 'slice_bytes': slice_bytes, 'aligned': aligned,
                           'head': head, 'region': region})
            regions.append(region)
            payload_total += tensor_bytes

        vparts = []
        for i, sp in enumerate(staged):
            slot_offset = sum(regions[:i])        # 점화식이 아니라 prefix 합
            vp = {'name': sp['name'], 'source_tensor': sp['source_tensor'], 'source_index': sp['source_index'],
                  'type': sp['type'], 'dims': list(sp['dims']), 'expert_axis': sp['expert_axis'],
                  'abs_offset': sp['abs_offset'], 'slice_bytes': sp['slice_bytes'], 'aligned': sp['aligned']}
            if sp['aligned']:
                vp['bracket_head'] = sp['head']
            vp['slot_offset'] = slot_offset
            vp['data_offset'] = slot_offset + (sp['head'] if sp['aligned'] else 0)
            if not sp['aligned']:
                vp['staging_bytes'] = sp['region'] + A
            vparts.append(vp)

        if not vparts:
            raise RepackAbort('verifier: layer %d has no routed part' % l)
        layer_slot_bytes = sum(regions)
        if layer_slot_bytes % A != 0:
            raise RepackAbort('verifier: layer %d layer_slot_bytes(%d) is not a multiple of A(%d)'
                              % (l, layer_slot_bytes, A))
        layers.append({'layer': l, 'layer_slot_bytes': layer_slot_bytes, 'parts': vparts})
        slot_stride_max = max(slot_stride_max, layer_slot_bytes)

    for VL in layers:
        for e in range(n_expert):
            for vp in VL['parts']:
                records.append({'layer': VL['layer'], 'expert': e, 'part': vp['name'],
                                'source_index': vp['source_index'],
                                'src_offset': vp['abs_offset'] + e * vp['slice_bytes'],
                                'slice_bytes': vp['slice_bytes'], 'data_offset': vp['data_offset']})
    return {'arch': arch, 'n_layer': n_layer, 'n_expert': n_expert, 'n_expert_used': n_expert_used,
            'moe_layers': moe_layers, 'used_types': sorted(used_types), 'shards': shards,
            'layers': layers, 'slot_stride_max': slot_stride_max, 'records': records,
            'n_records': len(records), 'virtual_payload_bytes': payload_total,
            # N4(r7 비차단): 대형 모델에서의 재유도 비용을 추측 없이 관리하기 위한 계측.
            'cost': {'elapsed_s': round(time.perf_counter() - _t0, 6),
                     'header_parser_file_bytes_read':
                         sum(h.get('header_parser_file_bytes_read', 0) for h in shards),
                     'shards_parsed': len(shards)}}


def _vfy_cross_check(ind, model, layout, vplan, problems, cap=8):
    """플래너 경로 산출과 검증 전용 재유도의 합치 검사(fail-close).

    두 구현이 어긋나면 어느 쪽이 옳든 이 산출물은 신뢰할 수 없다. 전항 대조의 기대값은 `ind`
    가 대고, 이 함수는 **어디서 갈라졌는지**를 이름으로 지목하는 진단층이다."""
    rows = [('model.arch', layout['arch'], ind['arch']),
            ('model.n_layer', layout['n_layer'], ind['n_layer']),
            ('model.n_expert', layout['n_expert'], ind['n_expert']),
            ('model.n_expert_used', layout['n_expert_used'], ind['n_expert_used']),
            ('model.moe_layers', list(layout['moe_layers']), list(ind['moe_layers'])),
            ('layout.slot_stride_max', vplan['slot_stride_max'], ind['slot_stride_max']),
            ('totals.virtual_payload_bytes', vplan['virtual_payload_bytes'], ind['virtual_payload_bytes']),
            ('totals.n_records', vplan['n_records'], ind['n_records']),
            ('sources.count', len(model['shards']), len(ind['shards']))]
    for h, s in zip(model['shards'], ind['shards']):
        where = 'sources[%d]' % s['source_index']
        rows.append(('%s.path' % where, os.path.normcase(os.path.abspath(h['path'])),
                     os.path.normcase(os.path.abspath(s['path']))))
        for f in ('file_bytes', 'gguf_version', 'alignment', 'data_start'):
            rows.append(('%s.%s' % (where, f), h[f], s[f]))

    # ★N1(r7 비차단 채택): parts·records 까지 여기서 대조한다. 이전에는 스칼라·source 요약만 봤고
    # 파트 단위 불일치는 뒤쪽 manifest 대조에서야 드러났다 — 결과는 같지만 진단이 "어느 구현이
    # 갈렸나"가 아니라 "manifest 가 틀렸다"로 나와 원인 위치가 한 단계 흐려진다.
    for li, (VL, IL) in enumerate(zip(vplan['layers'], ind['layers'])):
        # ★배열 인덱스로 지목한다(파트 이름이 아니라) — 순서 오도출이면 이름 자체가 어긋나므로
        # 이름으로 키를 잡으면 "무엇이 무엇과 비교됐는지"가 흐려지고, 아래 manifest 대조 구간의
        # 메시지 형식(§(9))과도 갈린다.
        where = 'layout.layers[%d]' % li
        rows.append(('%s.layer' % where, VL.get('layer'), IL.get('layer')))
        rows.append(('%s.layer_slot_bytes' % where, VL.get('layer_slot_bytes'), IL.get('layer_slot_bytes')))
        rows.append(('%s.parts.count' % where, len(VL.get('parts') or ()), len(IL.get('parts') or ())))
        for pi, (vp, ip) in enumerate(zip(VL.get('parts') or (), IL.get('parts') or ())):
            pw = '%s.parts[%d]' % (where, pi)
            rows.append(('%s.keyset' % pw, sorted(vp), sorted(ip)))
            for f in ('name', 'source_tensor', 'source_index', 'type', 'dims', 'expert_axis',
                      'abs_offset', 'slice_bytes', 'aligned', 'bracket_head', 'slot_offset',
                      'data_offset', 'staging_bytes'):
                if f in vp or f in ip:
                    rows.append(('%s.%s' % (pw, f), vp.get(f), ip.get(f)))
    rows.append(('records.count', len(vplan.get('records') or ()), len(ind.get('records') or ())))
    for i, (vr, ir) in enumerate(zip(vplan.get('records') or (), ind.get('records') or ())):
        if not _typed_eq(vr, ir):
            rows.append(('records[%d]' % i, vr, ir))
            break                      # 첫 어긋남만 지목한다(전 레코드 나열은 진단 가치가 없다)

    hits = 0
    for what, planner, verifier in rows:
        if _typed_eq(planner, verifier):
            continue
        hits += 1
        if hits <= cap:
            problems.append('two-implementation disagreement on %s: planner path=%r verifier-owned=%r '
                            '[the shared derivation and the verifier-owned one do not agree, so neither '
                            'licenses this artifact]' % (what, planner, verifier))
    if hits > cap:
        problems.append('two-implementation disagreement: %d further fields differ (truncated)' % (hits - cap))


def _vfy_legacy_align(manifest, ind, problems, locator_override=None):
    """§Z-③ [[C:repack.legacy-align]] legacy alignment 결속의 **독립 재확인**.

    paired 면 v2 manifest 를 strict 재개방해 SHA·`schema_version=="2.0"`·모델 identity·
    reference identity·sources identity 를 확인한 뒤 `v2.layout.align_bytes ==
    v3.layout.legacy_align_bytes` 를 검사한다. unpaired 면 값이 canonical 4096 이고 paired
    필드가 **없어야** 한다.

    ★이 값은 주소가 아니라 **D-A2 분자의 baseline** 이다 — 그래서 재유도 대상이 아니라
    "숫자의 출처를 증명한다"가 계약이다(생산은 v2 파일에서 값만 뽑고, 그 파일이 이 모델의
    것임을 증명하는 일은 여기서만 한다).

    ★★**locator 수명주기(r1 [MED] 수정 — 리드 처분 26-08-13)**: identity 권위는
    `reference_lock.legacy_v2_manifest_sha256` **하나뿐**이고,
    `reference_lock.legacy_v2_manifest_path` 는 **비권위 locator**(그 SHA 를 가진 파일을
    어디서 찾을지의 힌트)다. 그래서 `locator_override`(virtual `--verify-only` 의
    `--legacy-v2-manifest`)가 오면 **경로만** 그것으로 갈아끼우고, 수용 여부는 기록된 SHA·
    schema 2.0·model/sources/reference identity **전건 일치**로만 판정한다 — 하나라도
    어긋나면 기존대로 fail-close 다. 도달 근거=정상 운영자가 paired v2 artifact 만 옮긴 뒤
    새 경로를 지정하는 복구 경로(옛 절대경로를 열다 죽는 것이 구 동작).
    반환=plan_report echo dict."""
    m_layout = manifest.get('layout') if isinstance(manifest.get('layout'), dict) else {}
    ref = manifest.get('reference_lock') if isinstance(manifest.get('reference_lock'), dict) else {}
    la_bytes, la_source = m_layout.get('legacy_align_bytes'), m_layout.get('legacy_align_source')
    # ★F1(r2 지적): locator_source 는 **paired 분기에서만** 의미가 있다. override 유무만 보고
    # 정하면 canonical 산출물이 'manifest' 로 **거짓 기록**된다(결속된 v2 manifest 가 아예 없는데
    # "manifest 가 준 locator 를 썼다"고 말하는 꼴). 기본값을 'none' 으로 두고 paired 에서만 덮는다.
    echo = {'legacy_align_bytes': la_bytes, 'legacy_align_source': la_source,
            'legacy_v2_manifest_sha256': ref.get('legacy_v2_manifest_sha256'),
            'legacy_v2_manifest_path': ref.get('legacy_v2_manifest_path'),
            'legacy_v2_locator_source': 'none'}
    if not _is_u64(la_bytes) or la_bytes <= 0:
        problems.append('layout.legacy_align_bytes(%r) is missing or not a positive uint64 '
                        '(D-A2 numerator baseline - SPEC_IO_METRICS_V3 section 7)' % (la_bytes,))
    if la_source not in (LEGACY_ALIGN_SOURCE_PAIRED, LEGACY_ALIGN_SOURCE_CANONICAL):
        problems.append('layout.legacy_align_source(%r) is not one of %r'
                        % (la_source, [LEGACY_ALIGN_SOURCE_PAIRED, LEGACY_ALIGN_SOURCE_CANONICAL]))
        return echo

    if la_source == LEGACY_ALIGN_SOURCE_CANONICAL:
        if la_bytes != LEGACY_ALIGN_CANONICAL_BYTES:
            problems.append('legacy_align_source=%r requires legacy_align_bytes==%d, got %r'
                            % (LEGACY_ALIGN_SOURCE_CANONICAL, LEGACY_ALIGN_CANONICAL_BYTES, la_bytes))
        for k in ('legacy_v2_manifest_sha256', 'legacy_v2_manifest_path'):
            if k in ref:
                problems.append('legacy_align_source=%r must not carry reference_lock.%s (an unpaired model has '
                                'no v2 baseline to bind)' % (LEGACY_ALIGN_SOURCE_CANONICAL, k))
        # ★F1: canonical 산출물에 override 를 주는 것은 **조용히 무시할 일이 아니다** — 운영자가
        # "이 산출물을 그 v2 baseline 에 결속해 재검증한다"고 믿는데 실제로는 파일을 열지도 않고
        # PASS 하기 때문이다(구 동작). paired 결속을 원하면 재발행이 필요하다는 것을 명시한다.
        if locator_override:
            problems.append('--legacy-v2-manifest was supplied, but this manifest binds legacy alignment as %r '
                            '(no paired v2 baseline to re-open). the override only re-locates an existing paired '
                            'binding; to bind a v2 baseline, re-issue the artifact with --legacy-v2-manifest on a '
                            'real mode=virtual run.' % LEGACY_ALIGN_SOURCE_CANONICAL)
        return echo

    # --- paired_v2 -----------------------------------------------------------------
    v2_sha, v2_path = ref.get('legacy_v2_manifest_sha256'), ref.get('legacy_v2_manifest_path')
    if not (isinstance(v2_sha, str) and re.match(r'^[0-9a-f]{64}$', v2_sha)):
        problems.append('legacy_align_source=%r requires reference_lock.legacy_v2_manifest_sha256 as 64 lowercase '
                        'hex chars, got %r' % (LEGACY_ALIGN_SOURCE_PAIRED, v2_sha))
        return echo
    # locator 는 비권위다 — override 가 오면 **경로만** 갈아끼운다(수용 판정은 아래 전건 대조).
    echo['legacy_v2_locator_source'] = 'cli_override' if locator_override else 'manifest'
    locator = locator_override or v2_path
    if not isinstance(locator, str) or not locator:
        problems.append('legacy_align_source=%r needs a locator for the paired manifest: neither '
                        'reference_lock.legacy_v2_manifest_path(%r) nor a --legacy-v2-manifest override is usable'
                        % (LEGACY_ALIGN_SOURCE_PAIRED, v2_path))
        return echo
    try:
        v2, raw_sha, abs_path = _load_v2_manifest_strict(locator, 'paired v2 binding')
    except RepackAbort as e:
        problems.append('%s' % e)
        return echo
    echo['legacy_v2_manifest_resolved_path'] = abs_path
    if raw_sha != v2_sha:
        problems.append('reference_lock.legacy_v2_manifest_sha256(%r) != re-hash of %s(%r) [the paired v2 baseline '
                        'was replaced or rewritten after this manifest was produced%s]'
                        % (v2_sha, abs_path, raw_sha,
                           '; the --legacy-v2-manifest override does not relax this - the recorded SHA stays the '
                           'identity authority' if locator_override else ''))
        return echo
    if v2.get('schema_version') != SCHEMA_VERSION or 'mode' in v2:
        problems.append('the paired manifest %s is not a v2 (mode=bin) artifact: schema_version=%r mode_present=%s'
                        % (abs_path, v2.get('schema_version'), 'mode' in v2))
        return echo
    # 모델 identity — 다른 모델의 v2 baseline 을 붙이면 D-A2 분자가 조용히 남의 값이 된다.
    v2_model = v2.get('model') if isinstance(v2.get('model'), dict) else {}
    for field, expected in (('arch', ind['arch']), ('n_layer', ind['n_layer']),
                            ('n_expert', ind['n_expert']), ('n_expert_used', ind['n_expert_used']),
                            ('moe_layers', list(ind['moe_layers']))):
        if not _typed_eq(v2_model.get(field), expected):
            problems.append('paired v2 manifest model.%s(%r) != independently re-derived(%r) [the bound baseline '
                            'belongs to a different model]' % (field, v2_model.get(field), expected))
    # ★model 스칼라만으로는 모자란다 — 같은 arch·층수·전문가 수를 가진 **다른 파일**이 전부
    # 통과한다(합성 픽스처 두 개가 실제로 그렇다). "이 v2 artifact 가 지금 이 shard 들에서
    # 만들어졌다"까지 세우는 것은 sources[] 다. mtime 은 비교하지 않는다 — v2 baseline 은
    # 과거 시점 산출물이라 그 뒤의 정상적인 touch 로 페어링이 깨지면 안 된다.
    v2_sources = v2.get('sources') if isinstance(v2.get('sources'), list) else []
    if len(v2_sources) != len(ind['shards']):
        problems.append('paired v2 manifest sources count %d != independent re-parse %d'
                        % (len(v2_sources), len(ind['shards'])))
    else:
        for i, (vs, h) in enumerate(zip(v2_sources, ind['shards'])):
            if not isinstance(vs, dict):
                problems.append('paired v2 manifest sources[%d] is not an object' % i); continue
            if not _same_fs_name(vs.get('path'), h['path']):
                problems.append('paired v2 manifest sources[%d].path(%r) != this model(%r) [the bound baseline was '
                                'built from a different file]' % (i, vs.get('path'), h['path']))
            for field, expected in (('index', h['source_index']), ('bytes', h['file_bytes']),
                                    ('gguf_version', h['gguf_version']), ('alignment', h['alignment']),
                                    ('data_start', h['data_start'])):
                if not _typed_eq(vs.get(field), expected):
                    problems.append('paired v2 manifest sources[%d].%s(%r) != this model(%r)'
                                    % (i, field, vs.get(field), expected))
    # reference identity — 같은 모델이라도 다른 참조 락으로 잰 baseline 은 같은 실험이 아니다.
    v2_ref = v2.get('reference_lock') if isinstance(v2.get('reference_lock'), dict) else {}
    for field in ('profile_id', 'expect_sha256'):
        if not _typed_eq(v2_ref.get(field), ref.get(field)):
            problems.append('paired v2 manifest reference_lock.%s(%r) != this manifest(%r)'
                            % (field, v2_ref.get(field), ref.get(field)))
    v2_align = (v2.get('layout') if isinstance(v2.get('layout'), dict) else {}).get('align_bytes')
    if not _typed_eq(v2_align, la_bytes):
        problems.append('paired v2 manifest layout.align_bytes(%r) != layout.legacy_align_bytes(%r) [the D-A2 '
                        'baseline stride would be computed from an alignment the v2 artifact never used]'
                        % (v2_align, la_bytes))
    # `legacy_v2_manifest_path` 는 manifest 가 기록한 **비권위 locator** 그대로 두고, 실제로 연
    # 경로는 `legacy_v2_manifest_resolved_path` 로 따로 싣는다(override 를 쓴 런에서 둘이 갈린다).
    return echo


def verify_virtual_manifest(model_path, out_dir, profile_id=None, enforce_reference=True,
                            allow_default_align=False, arch_template=False,
                            manifest_name=MANIFEST_FILENAME, full_sha=False,
                            derived_expect_name=DERIVED_EXPECT_FILENAME,
                            legacy_v2_manifest=None):
    manifest_path = os.path.join(out_dir, manifest_name)
    problems = []
    manifest_sha256 = None
    cardinality = {'sources': 0, 'layers': 0, 'parts': 0, 'records': 0, 'source_tensors': 0}

    def _out(passed, extra=None):
        rep = {'mode': MODE_VIRTUAL, 'pass': bool(passed), 'manifest': manifest_name,
               'manifest_sha256': manifest_sha256, 'cardinality': cardinality,
               'problems': problems, 'checked_at': datetime.now(timezone.utc).isoformat(),
               'tool': {'version': SCRIPT_VERSION}}
        if extra:
            rep.update(extra)
        return rep

    # (1) manifest 재로드(엄격 JSON) + 재해시
    try:
        raw = open(manifest_path, 'rb').read()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = _strict_json_load_bytes(raw)
    except _DuplicateManifestKey as e:
        problems.append('manifest duplicate key: %s' % e); return _out(False)
    except _NonStandardJSONConstant as e:
        problems.append('manifest non-standard JSON constant: %s' % e); return _out(False)
    except Exception as e:
        problems.append('cannot load %s (corrupt JSON etc.): %r' % (manifest_name, e)); return _out(False)
    if not isinstance(manifest, dict):
        problems.append('the top level of %s is not a JSON object: %r' % (manifest_name, type(manifest).__name__))
        return _out(False)

    # (2) §2-1 모드·스키마 게이트 / §2-2 bin 전용 필드 금지
    _check_virtual_manifest_mode(manifest, problems)
    _check_virtual_forbidden_fields(manifest, problems)
    if problems:
        return _out(False)

    scope = (manifest.get('model') if isinstance(manifest.get('model'), dict) else {}).get('routed_scope', 'all')

    # (3) 전 shard 독립 재오픈·재파싱 + layout 독립 재도출
    try:
        model = load_model_shards(model_path)
        layout = build_layout(model, scope=scope)
    except RepackAbort as e:
        problems.append('verifier: independent re-parse of the source failed: %r' % e); return _out(False)

    # (4) A 독립 재질의(source 볼륨) + §2-4 슬롯 산술 독립 재도출
    try:
        A_expected, align_expected = resolve_alignment_for_sources(model, allow_default_align)
    except RepackAbort as e:
        problems.append('verifier: alignment re-query failed: %r' % e); return _out(False)
    try:
        vplan = compute_virtual_layout(layout, A_expected,
                                       {h['source_index']: h['file_bytes'] for h in model['shards']})
    except VirtualBinRegression as e:
        problems.append('verifier: the profile does not close as virtual (bracket EOF) - it must be regressed '
                        'to mode=bin: %s' % e)
        return _out(False)
    except RepackAbort as e:
        problems.append('verifier: independent slot arithmetic failed: %r' % e); return _out(False)

    # (4-b) ★부속 정오 2 §Z-②: 아래 전항 대조의 **기대값 원천**. 공통 3함수(플래너 경로)가 아니라
    # 검증 전용 재유도가 낸다 — 같은 함수의 오도출이 양쪽에서 상쇄되던 구간을 없앤다.
    try:
        ind = _vfy_derive_virtual(model_path, A_expected, scope)
    except RepackAbort as e:
        problems.append('verifier: the verifier-owned independent re-derivation failed: %r' % e)
        return _out(False)
    except Exception as e:                      # 검증층은 어떤 도출 실패도 FAIL 로 닫는다(크래시 금지)
        problems.append('verifier: the verifier-owned independent re-derivation raised: %r' % e)
        return _out(False)

    # (4-c) 두 구현 합치(진단층). 갈라지면 어느 쪽이 옳든 이 산출물은 신뢰할 수 없다.
    _vfy_cross_check(ind, model, layout, vplan, problems)
    if problems:
        # ★r8 N2: 이 반환도 재유도를 이미 끝낸 뒤다 — 비용 계측을 여기서만 빠뜨리면 "합치 검사가
        # 자주 걸리는 모델의 재유도 비용"이 리포트에서 사라진다(성공 경로만 남는 편향).
        return _out(False, {'verifier_cost': ind.get('cost')})

    n_expert = ind['n_expert']
    cardinality = {'sources': len(ind['shards']), 'layers': len(ind['layers']),
                   'parts': sum(len(VL['parts']) for VL in ind['layers']),
                   'records': ind['n_records'],
                   'source_tensors': sum(len(VL['parts']) for VL in ind['layers'])}

    # (5) reference_lock 독립 재확인(카탈로그 재해시 / derived 재유도 — v2 사상 승계)
    ref = manifest.get('reference_lock') if isinstance(manifest.get('reference_lock'), dict) else {}
    ref_profile, ref_sha = ref.get('profile_id'), ref.get('expect_sha256')
    reference_lock_out = {'profile_id': ref_profile, 'expect_sha256': ref_sha}
    if arch_template:
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
            # ★§Z-④: 생산 경로는 candidate 이름(`derived.expect.json.partial`)을 넘긴다 — 승격 전이라
            # 최종 이름엔 아직 **구 산출물**이 있고, 검증 대상은 candidate 다. `--verify-only` 는
            # 기본값(최종 이름)으로 들어와 승격된 산출물을 그대로 본다.
            dpath = os.path.join(out_dir, derived_expect_name)
            try:
                raw_disk = open(dpath, 'rb').read()
            except Exception as e:
                problems.append('cannot read the derived expect (%s): %r' % (dpath, e))
            else:
                sha_disk = hashlib.sha256(raw_disk).hexdigest()
                if ref_sha != sha_disk:
                    problems.append('reference_lock.expect_sha256(%r) != re-hash of %s(%r)'
                                    % (ref_sha, derived_expect_name, sha_disk))
                if raw_disk != raw_re:
                    problems.append('%s bytes != independent re-derivation (on-disk sha=%s, re-derived sha=%s)'
                                    % (derived_expect_name, sha_disk, sha_re))
                try:
                    expect_disk = _strict_json_load_bytes(raw_disk)
                    cross_check_expect(model, layout, vplan, expect_disk, scope=scope)
                except (_DuplicateManifestKey, _NonStandardJSONConstant) as e:
                    problems.append('derived expect is not strict JSON: %r' % e)
                except RepackAbort as e:
                    problems.append('independent re-check of the derived expect failed: %r' % e)
                except Exception as e:
                    problems.append('derived expect JSON is corrupt: %r' % e)
    elif enforce_reference:
        if profile_id is not None and ref_profile != profile_id:
            problems.append('reference_lock.profile_id(%r) != requested --profile(%r)' % (ref_profile, profile_id))
        try:
            expect, expect_sha256 = load_expect_profile(ref_profile if isinstance(ref_profile, str) else '')
            if ref_sha != expect_sha256:
                problems.append('reference_lock.expect_sha256(%r) != independent re-hash(%r)' % (ref_sha, expect_sha256))
            cross_check_expect(model, layout, vplan, expect, scope=scope)
        except RepackAbort as e:
            problems.append('independent re-check of reference_lock failed: %r' % e)
    else:
        if ref_profile != 'selftest-exempt' or ref_sha != 'selftest-exempt':
            problems.append('selftest reference_lock marker mismatch: %r/%r' % (ref_profile, ref_sha))

    # (6) 스칼라·구조 전항 독립 재구성 대조
    m_model = manifest.get('model') if isinstance(manifest.get('model'), dict) else {}
    m_layout = manifest.get('layout') if isinstance(manifest.get('layout'), dict) else {}
    m_totals = manifest.get('totals') if isinstance(manifest.get('totals'), dict) else {}
    m_sources = manifest.get('sources') if isinstance(manifest.get('sources'), list) else []
    scalar_checks = [
        ('schema_version', manifest.get('schema_version'), SCHEMA_VERSION_V3),
        ('mode', manifest.get('mode'), MODE_VIRTUAL),
        ('model.arch', m_model.get('arch'), ind['arch']),
        ('model.n_layer', m_model.get('n_layer'), ind['n_layer']),
        ('model.n_expert', m_model.get('n_expert'), n_expert),
        ('model.n_expert_used', m_model.get('n_expert_used'), ind['n_expert_used']),
        ('model.moe_layers', m_model.get('moe_layers'), list(ind['moe_layers'])),
        ('layout.align_bytes', m_layout.get('align_bytes'), A_expected),
        ('layout.slot_stride_max', m_layout.get('slot_stride_max'), ind['slot_stride_max']),
        ('totals.virtual_payload_bytes', m_totals.get('virtual_payload_bytes'), ind['virtual_payload_bytes']),
        ('totals.n_records', m_totals.get('n_records'), ind['n_records']),
    ]
    for field, actual, expected in scalar_checks:
        if not _typed_eq(actual, expected):
            note = ' - type mismatch (actual=%s expected=%s)' % (type(actual).__name__, type(expected).__name__) \
                if _type_mismatch(actual, expected) else ''
            problems.append('manifest.%s(%r) != independently re-derived(%r)%s' % (field, actual, expected, note))

    # (7) align_query[] 재질의 대조(생산 측 provenance 의 짝). ★범위는 **주소 관련 4필드 +
    #     drive root** 이지 전항이 아니다 — `fallback_reason` 은 비교하지 않는다(r8 M2 축소 계약).
    #     [[C:repack.align-query]]
    m_aq = m_layout.get('align_query') if isinstance(m_layout.get('align_query'), list) else []
    if len(m_aq) != len(align_expected):
        problems.append('layout.align_query count %d != re-query %d' % (len(m_aq), len(align_expected)))
    else:
        for i, (mq, eq) in enumerate(zip(m_aq, align_expected)):
            if not isinstance(mq, dict):
                problems.append('layout.align_query[%d] is not an object' % i); continue
            for field in ('source_index', 'method', 'logical', 'physical'):
                if not _typed_eq(mq.get(field), eq[field]):
                    problems.append('layout.align_query[%d].%s(%r) != re-query(%r)' % (i, field, mq.get(field), eq[field]))
            # ★r1 실 결함 ① 의 나머지 한 필드. A 는 이 drive 에 질의해 얻은 값이므로, 같은 A 라도
            # 다른 drive 에서 재대조하면 그건 같은 근거가 아니다.
            if not _same_drive_root(mq.get('drive_root'), eq['drive_root']):
                problems.append('layout.align_query[%d].drive_root(%r) != re-query(%r) [the alignment '
                                'was queried from a different volume than the one now in use]'
                                % (i, mq.get('drive_root'), eq['drive_root']))

    # (8) sources[] 전항 + DF-1 digest 독립 재계산
    source_identity = []
    if len(m_sources) != len(ind['shards']):
        problems.append('sources count %d != independent re-parse %d' % (len(m_sources), len(ind['shards'])))
    else:
        for i, (ms, h) in enumerate(zip(m_sources, ind['shards'])):
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
            # ★r1 [BLOCKER] 실 결함 ①(후반부): 사양 §2-5 는 sources[] 전항 대조를 요구하는데
            # path·mtime 은 존재·타입만 보고 있었다. 도달 경로는 평범하다 — 같은 모델을 다른 경로로
            # 복사한 뒤 `--verify-only --mode virtual --model <새 경로>` 를 돌리면 새 파일을
            # 재파싱하면서 manifest 의 옛 경로 불일치를 무시하고 PASS 한다. 헤더와 크기는 같고
            # mtime 만 바뀌는 정상 교체(다운로드 재개·재생성 등)도 같은 이유로 통과한다.
            if not _same_fs_name(ms.get('path'), h['path']):
                problems.append('sources[%d].path(%r) != re-parse(%r) [source identity: the manifest '
                                'names a different file than the one being verified]'
                                % (i, ms.get('path'), h['path']))
            mt_expected = os.path.getmtime(h['path'])
            if not _typed_eq(ms.get('mtime'), mt_expected):
                problems.append('sources[%d].mtime(%r) != re-parse(%r) [the source was replaced or '
                                'rewritten after this manifest was produced]'
                                % (i, ms.get('mtime'), mt_expected))
            try:
                dg_expected = source_header_digest(h['path'], h['data_start'])
            except RepackAbort as e:
                problems.append('sources[%d] header digest re-computation failed: %r' % (i, e))
                continue
            if not _typed_eq(ms.get('digest'), dg_expected):
                problems.append('sources[%d].digest(%r) != independently re-computed(%r) [DF-1: source shard '
                                'integrity is part of the lossless chain in mode=virtual]'
                                % (i, ms.get('digest'), dg_expected))
            ident = {'index': h['source_index'], 'path': h['path'], 'bytes': h['file_bytes'],
                     'mtime': os.path.getmtime(h['path']), 'gguf_version': h['gguf_version'],
                     'alignment': h['alignment'], 'data_start': h['data_start'], 'digest': dg_expected}
            if full_sha:
                ident['full_sha256'] = _file_sha256(h['path'])
            source_identity.append(ident)

    # (9) layout.layers[]/parts[] 전항 독립 재구성 대조(주소 단일 권위)
    m_layers = m_layout.get('layers') if isinstance(m_layout.get('layers'), list) else []
    if len(m_layers) != len(ind['layers']):
        problems.append('layout.layers count %d != re-derived %d' % (len(m_layers), len(ind['layers'])))
    else:
        for i, (ml, VL) in enumerate(zip(m_layers, ind['layers'])):
            if not isinstance(ml, dict):
                problems.append('layout.layers[%d] is not an object' % i); continue
            for field, actual, expected in [('layer', ml.get('layer'), VL['layer']),
                                            ('layer_slot_bytes', ml.get('layer_slot_bytes'), VL['layer_slot_bytes'])]:
                if not _typed_eq(actual, expected):
                    note = ' - type mismatch' if _type_mismatch(actual, expected) else ''
                    problems.append('layout.layers[%d].%s(%r)!=re-derived(%r)%s' % (i, field, actual, expected, note))
            m_parts = ml.get('parts') if isinstance(ml.get('parts'), list) else []
            if len(m_parts) != len(VL['parts']):
                problems.append('layout.layers[%d].parts count %d != %d' % (i, len(m_parts), len(VL['parts'])))
                continue
            for j, (mp, vp) in enumerate(zip(m_parts, VL['parts'])):
                if not isinstance(mp, dict):
                    problems.append('layout.layers[%d].parts[%d] is not an object' % (i, j)); continue
                if set(mp.keys()) != set(vp.keys()):
                    problems.append('layout.layers[%d].parts[%d] key set %r != re-derived %r'
                                    % (i, j, sorted(mp), sorted(vp)))
                for field in ('name', 'source_tensor', 'source_index', 'type', 'dims', 'expert_axis',
                              'abs_offset', 'slice_bytes', 'aligned', 'bracket_head', 'slot_offset',
                              'data_offset', 'staging_bytes'):
                    if field not in vp and field not in mp:
                        continue
                    if not _typed_eq(mp.get(field), vp.get(field)):
                        note = ' - type mismatch' if _type_mismatch(mp.get(field), vp.get(field)) else ''
                        problems.append('layout.layers[%d].parts[%d].%s(%r)!=re-derived(%r)%s'
                                        % (i, j, field, mp.get(field), vp.get(field), note))

    # (10) records[] 독립 재생성 전항 대조(비권위 witness — 누락·중복·순서 변조 판정)
    m_records = manifest.get('records') if isinstance(manifest.get('records'), list) else None
    if m_records is None:
        problems.append('records missing or not a list')
    elif len(m_records) != len(ind['records']):
        problems.append('records count mismatch: actual=%d expected=%d' % (len(m_records), len(ind['records'])))
    else:
        for i, (rec, exp) in enumerate(zip(m_records, ind['records'])):
            if not isinstance(rec, dict):
                problems.append('records[%d] is not an object' % i); continue
            got = [rec.get(k) for k in ('layer', 'expert', 'part', 'source_index', 'src_offset',
                                        'slice_bytes', 'data_offset')]
            want = [exp[k] for k in ('layer', 'expert', 'part', 'source_index', 'src_offset',
                                     'slice_bytes', 'data_offset')]
            if set(rec.keys()) != set(exp.keys()):
                problems.append('records[%d] key set %r != re-derived %r' % (i, sorted(rec), sorted(exp)))
            elif not _typed_eq(got, want):
                note = ' - type mismatch' if any(_type_mismatch(a, w) for a, w in zip(got, want)) else ''
                problems.append('records[%d] mismatch: actual=%r expected=%r%s' % (i, got, want, note))

    # (11) source_tensors[] 독립 재생성 전항 대조(abs_offset 포함)
    expected_stensors = []
    for VL in ind['layers']:
        for vp in VL['parts']:
            expected_stensors.append((vp['source_tensor'], vp['source_index'], vp['abs_offset'],
                                      vp['slice_bytes'] * n_expert, vp['type'], list(vp['dims'])))
    m_stensors = manifest.get('source_tensors') if isinstance(manifest.get('source_tensors'), list) else None
    if m_stensors is None:
        problems.append('source_tensors missing or not a list')
    elif len(m_stensors) != len(expected_stensors):
        problems.append('source_tensors count mismatch: actual=%d expected=%d'
                        % (len(m_stensors), len(expected_stensors)))
    else:
        for i, (st, exp) in enumerate(zip(m_stensors, expected_stensors)):
            if not isinstance(st, dict):
                problems.append('source_tensors[%d] is not an object' % i); continue
            got = (st.get('name'), st.get('source_index'), st.get('abs_offset'), st.get('bytes'),
                   st.get('type'), st.get('dims'))
            if not _typed_eq(list(got), list(exp)):
                note = ' - type mismatch' if any(_type_mismatch(a, w) for a, w in zip(got, exp)) else ''
                problems.append('source_tensors[%d](%s) mismatch: actual=%r expected=%r%s' % (i, exp[0], got, exp, note))

    # (12) quant_traits 독립 재생성 대조
    expected_traits = {tt: {'block_values': QUANT_TRAITS[tt][0], 'block_bytes': QUANT_TRAITS[tt][1]}
                       for tt in ind['used_types']}
    m_traits = manifest.get('quant_traits') if isinstance(manifest.get('quant_traits'), dict) else {}
    if set(m_traits.keys()) != set(expected_traits.keys()):
        problems.append('quant_traits key set mismatch: actual=%s expected=%s' % (sorted(m_traits), sorted(expected_traits)))
    else:
        for tt, exp in expected_traits.items():
            if not _typed_eq(m_traits.get(tt), exp):
                problems.append('quant_traits[%s] mismatch: actual=%r expected=%r' % (tt, m_traits.get(tt), exp))

    # (13) manifest 자체 일관성(§2-4 자기일관 재계산 — 재도출 대조와 별개 방어층)
    _check_virtual_manifest_invariants(manifest, problems)

    # (13-b) §Z-③ legacy alignment 결속 독립 재확인(paired v2 strict 재개방 / unpaired canonical).
    #        `legacy_v2_manifest` 는 **비권위 locator override** 뿐이다(수용은 SHA·identity 전건 일치).
    legacy_echo = _vfy_legacy_align(manifest, ind, problems, locator_override=legacy_v2_manifest)

    # (14) experts.bin 부재(virtual 은 bin 을 만들지 않는다)
    if os.path.exists(os.path.join(out_dir, 'experts.bin')) or os.path.exists(os.path.join(out_dir, 'experts.bin.partial')):
        problems.append('mode=virtual artifact directory must not contain experts.bin(.partial): %s' % out_dir)

    return _out(not problems, {'reference_lock': reference_lock_out,
                               'align_bytes': A_expected, 'align_query': align_expected,
                               'legacy_align': legacy_echo,
                               'sources': source_identity,
                               'totals': {'virtual_payload_bytes': ind['virtual_payload_bytes'],
                                          'n_records': ind['n_records']},
                               'verifier_cost': ind.get('cost'),
                               'full_file_sha256_recorded': bool(full_sha)})


# ---------------------------------------------------------------------------
# §4-1 mode=virtual 생산: candidate manifest → 독립 verifier → PASS 시 원자 승격
# ---------------------------------------------------------------------------
def do_virtual_plan(model_path, out_dir, profile_id, force=False, enforce_reference=True,
                    allow_default_align=False, scope='all', arch_template=False, full_sha=False,
                    legacy_v2_manifest=None):
    model, layout, A, align_queries, vplan, expect_sha256, expect_totals, derived = _prepare_virtual(
        model_path, profile_id, allow_default_align, enforce_reference, scope=scope,
        arch_template=arch_template)
    legacy_align = resolve_legacy_align(legacy_v2_manifest)
    if derived is not None:
        profile_id = derived['lock_id']
    lock_enforced = enforce_reference or (derived is not None)

    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    report_path = os.path.join(out_dir, PLAN_REPORT_FILENAME)
    candidate_path = manifest_path + '.partial'
    bin_path = os.path.join(out_dir, 'experts.bin')
    if os.path.exists(bin_path):
        raise RepackAbort('mode=virtual refuses to write into a directory that already holds a legacy experts.bin '
                          '(%s) - a v3 manifest sitting next to a v2 bin is exactly the mixed artifact that silent '
                          'misreads come from. use a separate --out, or delete the bin only after the mode gate '
                          'passes (SPEC section 9-7).' % bin_path)
    if (os.path.exists(manifest_path) or os.path.exists(report_path)) and not force:
        raise RepackAbort('output already exists(%s) - use --force to overwrite' % out_dir)
    if os.path.exists(candidate_path):
        print('[warning] found a manifest.json.partial candidate from a previous run - it was never promoted; '
              'this run overwrites it')
    # ★§Z-④(r1 [MEDIUM]): derived expect 도 candidate 로 격리한다. 구판은 candidate 검증 **전에**
    # 최종 이름을 선교체해서, verifier 가 뒤늦게 FAIL 하면 manifest/plan_report 는 보존되는데
    # derived expect 만 새 값으로 갈려 있었다(= "구 산출물 전체가 사용 가능하게 보존된다"가
    # arch-template 경로에서 미성립 — 부속 정오 2-ⓔ 가 사양 주장을 좁힌 그 결손).
    derived_expect_path = os.path.join(out_dir, DERIVED_EXPECT_FILENAME)
    derived_candidate_name = DERIVED_EXPECT_FILENAME + '.partial'
    derived_candidate_path = os.path.join(out_dir, derived_candidate_name)
    if os.path.exists(derived_candidate_path):
        print('[warning] found a %s candidate from a previous run - it was never promoted; this run overwrites it'
              % derived_candidate_name)

    t0 = time.time()
    digests = build_source_digests(model)
    print('=== mode=virtual preflight: same summary as --plan ===')
    _print_virtual_plan_summary(model, layout, vplan, A, align_queries,
                                profile_id if lock_enforced else '(selftest-exempt)',
                                expect_sha256, expect_totals, digests=digests, derived=derived,
                                legacy_align=legacy_align)

    os.makedirs(out_dir, exist_ok=True)
    if derived is not None:
        dpath = write_derived_expect(out_dir, derived['raw'], filename=derived_candidate_name)
        print('[EXPERIMENTAL arch-template] wrote the candidate %s (sha256=%s)' % (dpath, derived['sha256']))

    manifest = build_manifest_v3(model, layout, vplan, A, align_queries,
                                 profile_id if lock_enforced else 'selftest-exempt',
                                 expect_sha256, digests, legacy_align=legacy_align)
    _guard_manifest_mode(manifest, MODE_VIRTUAL)
    raw = json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False).encode('utf-8')
    with open(candidate_path, 'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print('[manifest] wrote the candidate %s (%d records, %d B, reference_lock=%s)'
          % (os.path.basename(candidate_path), vplan['n_records'], len(raw),
             profile_id if lock_enforced else 'selftest-exempt'))

    # ★독립 verifier: producer 의 in-memory 객체를 넘기지 않는다(경로만) — strict JSON 재로드 +
    #   전 shard 재오픈·재파싱으로 **주소값 독립 재유도**(전항 재도출이 아니다 — 축소 계약의
    #   신뢰원 4종은 계약 밖이다). [[C:repack.trust-sources]]
    print('[verify] running the independent verifier (strict JSON reload + full re-parse of every shard)...')
    report = verify_virtual_manifest(model_path, out_dir,
                                     profile_id=profile_id if lock_enforced else None,
                                     enforce_reference=lock_enforced,
                                     allow_default_align=allow_default_align,
                                     arch_template=arch_template,
                                     manifest_name=os.path.basename(candidate_path),
                                     full_sha=full_sha,
                                     derived_expect_name=derived_candidate_name)
    if not report['pass']:
        _append_repack_log({'ts': datetime.now(timezone.utc).isoformat(), 'mode': 'virtual-plan',
                            'model': model_path, 'out': out_dir, 'profile': profile_id,
                            'elapsed_s': time.time() - t0, 'promoted': False,
                            'manifest_sha256': report['manifest_sha256'], 'verify_pass': False,
                            'problems': report['problems'][:20]})
        raise RepackAbort('the independent verifier rejected the candidate manifest - NOT promoted (%s kept for '
                          'postmortem). problems:\n  %s'
                          % (candidate_path, '\n  '.join(str(x) for x in report['problems'][:20])))

    # PASS: plan_report 원자 기록 → (derived expect 승격) → manifest 원자 승격
    # (불변식: manifest.json 이 존재하면 그 plan_report 도 존재한다 — 소비자 결속 §4-3.
    #  ★§Z-④ 순서 근거: manifest 를 본 소비자는 derived expect 를 찾으므로 derived 가 **먼저**
    #  제자리에 있어야 하고, manifest 승격을 마지막에 둔다.
    #  ★정정(r1 Q1 — 구 주석 "중간 크래시가 남기는 상태는 항상 구-구 또는 신-신"은 과장이었다):
    #  세 번의 os.replace 사이에는 **신 report + 구 manifest**, **신 derived + 구 manifest** 라는
    #  순간적 중간 상태가 실재한다. 그것이 안전한 이유는 그 상태가 없어서가 아니라, 소비자가
    #  manifest 재해시 == plan_report.manifest_sha256 을 요구하므로(§4-3 소비 결속) 짝이 안 맞는
    #  중간 상태를 **fail-close 로 거부**하기 때문이다. manifest 를 마지막에 미는 것은 그 창을
    #  최소화하는 장치이지 창을 없애는 장치가 아니다.)
    report_out = dict(report)
    report_out['manifest'] = MANIFEST_FILENAME
    report_out['model_path'] = model_path
    report_out['promoted'] = True
    # §Z-⑤ 의 승계 규칙은 `--verify-only` 소관이다(부속 정오 2-ⓓ 문면). 생산 경로는 산출물을
    # 새로 만드는 자리이므로 승계 대상이 없고, 필드 형태만 같게 둔다(소비자가 분기하지 않도록).
    apply_full_sha_provenance(None, report_out)
    report_raw = json.dumps(report_out, ensure_ascii=False, indent=1, allow_nan=False).encode('utf-8')
    tmp_report = report_path + '.tmp'
    with open(tmp_report, 'wb') as f:
        f.write(report_raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_report, report_path)
    if derived is not None:
        os.replace(derived_candidate_path, derived_expect_path)
        print('[EXPERIMENTAL arch-template] promoted %s (sha256=%s)'
              % (DERIVED_EXPECT_FILENAME, derived['sha256']))
    os.replace(candidate_path, manifest_path)
    print('[verify] PASS - promoted %s (manifest_sha256=%s)' % (MANIFEST_FILENAME, report['manifest_sha256']))
    print('[verify] cardinality: %r' % (report['cardinality'],))
    _append_repack_log({'ts': datetime.now(timezone.utc).isoformat(), 'mode': 'virtual-plan',
                        'model': model_path, 'out': out_dir, 'profile': profile_id,
                        'elapsed_s': time.time() - t0, 'promoted': True,
                        'manifest_sha256': report['manifest_sha256'],
                        'n_records': vplan['n_records'], 'verify_pass': True})
    return manifest, report_out


_FULL_SHA_IDENTITY_FIELDS = ('index', 'path', 'bytes', 'mtime', 'gguf_version', 'alignment',
                             'data_start', 'digest')


def apply_full_sha_provenance(prev_report, report_out):
    """§Z-⑤(부속 정오 2-ⓓ): `--verify-only` 가 plan_report 를 **원자 교체**할 때 full-SHA
    provenance 를 승계하거나, 승계하지 않으면 **명시적 downgrade 를 기록**한다.

    구 동작은 플래그 없는 재검증 1회로 `full_file_sha256_recorded:true` 와 전 shard full SHA 가
    **조용히 소실**됐다(r1 [LOW] 실 결함 — 증거 보존 결손).

    승계 조건은 **source identity 동일**이다(`index/path/bytes/mtime/gguf_version/alignment/
    data_start/digest` 전항). 하나라도 다르면 옛 full SHA 는 지금 파일의 것이 아니므로 승계가
    거짓 진술이 된다 — 그때는 승계하지 않고 사유를 적는다. `report_out` 을 제자리 수정한다."""
    if report_out.get('full_file_sha256_recorded'):
        report_out['full_sha_provenance'] = {'state': 'recomputed'}
        return report_out
    prev = prev_report if isinstance(prev_report, dict) else None
    if not (prev and prev.get('full_file_sha256_recorded')):
        report_out['full_sha_provenance'] = {'state': 'absent'}
        return report_out

    prev_sources = prev.get('sources') if isinstance(prev.get('sources'), list) else []
    now_sources = report_out.get('sources') if isinstance(report_out.get('sources'), list) else []
    reason = None
    if len(prev_sources) != len(now_sources):
        reason = ('source count changed (%d -> %d)' % (len(prev_sources), len(now_sources)))
    else:
        carried = []
        for ps, ns in zip(prev_sources, now_sources):
            if not isinstance(ps, dict) or not isinstance(ns, dict):
                reason = 'a source entry is not an object'; break
            # ★r1 [LOW]: `path` 만은 **경로 의미 비교**다(`_same_fs_name`). raw 문자열 비교면
            # 같은 파일을 대소문자·구분자만 다르게 적은 정상 표기가 불필요한 downgrade 를 낸다
            # — verifier 는 이미 같은 규약으로 동일 파일로 인정하므로 둘을 맞춘다.
            diff = [f for f in _FULL_SHA_IDENTITY_FIELDS
                    if not (_same_fs_name(ps.get(f), ns.get(f)) if f == 'path'
                            else _typed_eq(ps.get(f), ns.get(f)))]
            if diff:
                reason = 'source[%r] identity changed: %s' % (ns.get('index'), ','.join(diff)); break
            sha = ps.get('full_sha256')
            if not (isinstance(sha, str) and re.match(r'^[0-9a-f]{64}$', sha)):
                reason = 'source[%r] carried no usable full_sha256' % (ns.get('index'),); break
            carried.append(sha)
        if reason is None and not carried:
            reason = 'the previous report recorded no sources to inherit from'
        if reason is None:
            for ns, sha in zip(now_sources, carried):
                ns['full_sha256'] = sha
            report_out['full_file_sha256_recorded'] = True
            report_out['full_sha_provenance'] = {'state': 'inherited', 'n_sources': len(carried),
                                                 'inherited_from_checked_at': prev.get('checked_at')}
            return report_out
    report_out['full_sha_provenance'] = {'state': 'downgraded', 'reason': reason,
                                         'previous_checked_at': prev.get('checked_at')}
    return report_out


def _read_plan_report(out_dir):
    """기존 plan_report 를 읽는다(부재·손상=None — 승계 판단은 '없음'으로 닫는다)."""
    try:
        raw = open(os.path.join(out_dir, PLAN_REPORT_FILENAME), 'rb').read()
    except OSError:
        return None
    try:
        prev = json.loads(raw.decode('utf-8'))
    except Exception:
        return None
    return prev if isinstance(prev, dict) else None


def do_verify_only_virtual(model_path, out_dir, profile_id, enforce_reference=True,
                           allow_default_align=False, arch_template=False, full_sha=False,
                           legacy_v2_manifest=None):
    """기존 v3 산출물(manifest.json)을 재생산 없이 독립 verifier 로 1회 재검증하고 plan_report.json
    를 갱신한다(산출물 manifest 바이트는 불변 — 쓰기는 plan_report 원자 교체 1건)."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise RepackAbort('--verify-only --mode virtual: artifact missing - %s (nothing to re-verify)' % manifest_path)
    t0 = time.time()
    before = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    prev_report = _read_plan_report(out_dir)      # §Z-⑤: 원자 교체 전에 읽어야 승계가 가능하다
    report = verify_virtual_manifest(model_path, out_dir, profile_id=profile_id,
                                     enforce_reference=enforce_reference,
                                     allow_default_align=allow_default_align,
                                     arch_template=arch_template, full_sha=full_sha,
                                     legacy_v2_manifest=legacy_v2_manifest)
    report_out = dict(report)
    report_out['sources'] = [dict(s) for s in report_out.get('sources', [])]
    report_out['model_path'] = model_path
    report_out['promoted'] = bool(report['pass'])
    apply_full_sha_provenance(prev_report, report_out)
    report_path = os.path.join(out_dir, PLAN_REPORT_FILENAME)
    tmp_report = report_path + '.tmp'
    with open(tmp_report, 'wb') as f:
        f.write(json.dumps(report_out, ensure_ascii=False, indent=1, allow_nan=False).encode('utf-8'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_report, report_path)
    after = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    if before != after:
        raise RepackAbort('internal: --verify-only changed the manifest bytes (%s -> %s)' % (before, after))
    _append_repack_log({'ts': datetime.now(timezone.utc).isoformat(), 'mode': 'virtual-verify-only',
                        'model': model_path, 'out': out_dir, 'profile': profile_id,
                        'elapsed_s': time.time() - t0,
                        'manifest_sha256': report['manifest_sha256'], 'verify_pass': report['pass']})
    return report_out


# ---------------------------------------------------------------------------
# --plan / 본실행 커맨드
# ---------------------------------------------------------------------------
def cmd_plan(args):
    arch_template = bool(getattr(args, 'arch_template', False))
    mode = getattr(args, 'mode', MODE_BIN) or MODE_BIN
    print('=== --plan: GGUF header analysis (0 bytes written) ===')
    if mode != MODE_BIN:
        # ★bin 경로의 stdout 은 한 줄도 바꾸지 않는다 — launcher 의 --plan 텍스트 파서
        # (Start-MoeDirect.ps1 ConvertFrom-TemplatePlanText:2344-2350)가 이 출력을 소비한다.
        print('mode: %s' % mode)
    # ★메인 검수 복원(26-08-08): 시공 중 이 줄이 `mode:` 로 교체됐다가, `mode:` 를 virtual
    # 전용으로 좁히면서 되살아나지 못해 bin 경로에서 사라져 있었다. launcher 정규식은 이 줄을
    # 소비하지 않아 기계 파손은 없었으나, 바로 위 주석이 선언한 "bin stdout 무변경" 계약을
    # 어긴 상태였다. 순서까지 구본과 동일하게 되돌린다(헤더 → profile → model → out).
    print('profile: %s' % ('(EXPERIMENTAL arch-template: derived on the spot)' if arch_template else args.profile))
    print('model: %s' % args.model)
    print('out (planned target): %s' % args.out)
    if mode == MODE_VIRTUAL:
        model, layout, A, align_queries, vplan, expect_sha256, expect_totals, derived = _prepare_virtual(
            args.model, args.profile, allow_default_align=False, enforce_reference=True,
            scope=args.scope, arch_template=arch_template)
        # --plan 은 0바이트 계약이므로 아무것도 쓰지 않는다. 헤더 영역 digest(DF-1)는 읽기만이라
        # 계약을 지키며 산출하고, 전파일 SHA(--source-full-sha)는 plan 에서 건너뛴다.
        digests = build_source_digests(model)
        _print_virtual_plan_summary(model, layout, vplan, A, align_queries,
                                    derived['lock_id'] if derived else args.profile,
                                    expect_sha256, expect_totals, digests=digests, derived=derived,
                                    legacy_align=resolve_legacy_align(
                                        getattr(args, 'legacy_v2_manifest', None)))
        if getattr(args, 'source_full_sha', False):
            print('[note] --source-full-sha is skipped in --plan (it would read every source byte); '
                  'it is recorded by the independent verifier on a real mode=virtual run')
        if derived is not None:
            print('--- derived expect (%s, not written in --plan) ---' % DERIVED_EXPECT_FILENAME)
            print(derived['raw'].decode('utf-8').rstrip('\n'))
        print('--plan done (0 bytes written, no GPU used)')
        return
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
    mode = getattr(args, 'mode', MODE_BIN) or MODE_BIN
    if mode == MODE_VIRTUAL:
        print('=== real run (mode=virtual): --profile=%s --model=%s --out=%s force=%s arch_template=%s ===\n'
              '    manifest v3 + plan_report only - experts.bin is NOT produced (0 bytes of expert data move)'
              % (args.profile, args.model, args.out, args.force, arch_template))
        manifest, report = do_virtual_plan(args.model, args.out, args.profile, force=args.force,
                                          scope=args.scope, arch_template=arch_template,
                                          full_sha=bool(getattr(args, 'source_full_sha', False)),
                                          legacy_v2_manifest=getattr(args, 'legacy_v2_manifest', None))
        print('=== real run done (mode=virtual): PASS - manifest promoted, plan_report pass=%r ===' % report['pass'])
        return
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
    mode = getattr(args, 'mode', MODE_BIN) or MODE_BIN
    if mode == MODE_VIRTUAL:
        print('=== --verify-only --mode virtual: re-running the independent verifier on the existing manifest v3 '
              '--profile=%s --model=%s --out=%s arch_template=%s ==='
              % (args.profile, args.model, args.out, arch_template))
        report = do_verify_only_virtual(args.model, args.out, args.profile, arch_template=arch_template,
                                        full_sha=bool(getattr(args, 'source_full_sha', False)),
                                        legacy_v2_manifest=getattr(args, 'legacy_v2_manifest', None))
        if report['pass']:
            print('[verify-only] PASS - manifest_sha256=%s cardinality=%r'
                  % (report['manifest_sha256'], report['cardinality']))
            print('=== --verify-only done: plan_report.json refreshed (manifest bytes unchanged) ===')
            return
        print('[verify-only] FAIL - %d problems' % len(report['problems']))
        for p in report['problems'][:20]:
            print('    %s' % p)
        print('=== --verify-only verification FAILED - do not proceed ===')
        sys.exit(1)
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
        # K3 등재분(ggml-common.h:388-393·:407-410 필드 분해)
        'IQ2_XS':  (QK_K, HALF + (QK_K // 8) * 2 + QK_K // 32),      # d + qs[QK_K/8](u16) + scales
        'IQ3_XXS': (QK_K, HALF + 3 * (QK_K // 8)),                   # d + qs[3*QK_K/8]
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
        # ⓐ 등록 경로 expect 전부 불변 / ⓑ 템플릿 semantic replay / ⓒ inventory_sha256 결정론 /
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

        # ---- ⓐ 등록 경로 expect 전부 불변(파일 SHA == 카탈로그 승인 digest·전량 로드 PASS) ----
        # ★개수는 산문에 적지 않는다 — `EXPECT_CATALOG` 실측이 유일 원천이고 아래 출력은 전부
        # 거기서 유도한다(§2-8 규칙 4 파생값 수기 금지). [[C:repack.selftest-counts]]
        reg_details = []
        catalog_n = len(EXPECT_CATALOG)
        ok_reg = (catalog_n > 0)
        if not ok_reg:
            reg_details.append('카탈로그가 비어 있다(항목 수=%d)' % catalog_n)
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
        checks.append(('OPEN_ARCH-ⓐ 등록 경로 expect 전부 불변 — 카탈로그 %d항 각각 파일 SHA=승인 digest·'
                       '%d/%d 로드 PASS' % (catalog_n, catalog_n, catalog_n), ok_reg))
        print('[selftest] OPEN_ARCH-ⓐ 등록 expect %d종: %s %s'
              % (catalog_n, 'PASS' if ok_reg else 'FAIL', reg_details if reg_details else ''))

        # ---- ⓒ 카탈로그 ↔ expects/ 디렉토리 집합 동등(선등재 파일·고아 등재 양방향 적발) ----
        # ★`_reg_live`(문서 트리 존재) 와 **무관하게 항상 실행**한다 — EXPECTS_DIR 과 로더는 문서
        # 트리와 독립이라, 문서 트리가 없는 환경에서 이 검사가 생략되면 안 된다(Codex r2 D3).
        # ★스템은 `Path.stem` 이 아니라 `.expect.json` **전체 접미사**를 제거한 id 다(`.stem` 은
        # `kimi-k3-ud-q2kxl.expect` 를 남겨 전건 불일치를 만든다).
        _EXPECT_SUFFIX = '.expect.json'
        try:
            _disk_ids = {fn[:-len(_EXPECT_SUFFIX)] for fn in os.listdir(EXPECTS_DIR)
                         if fn.endswith(_EXPECT_SUFFIX)}
        except OSError as _ce:
            _disk_ids, _cat_err = set(), 'expects 디렉토리 열기 실패: %r' % (_ce,)
        else:
            _cat_err = None
        _cat_ids = set(EXPECT_CATALOG)
        _catalog_only = sorted(_cat_ids - _disk_ids)      # 등재됐는데 파일이 없다
        _file_only = sorted(_disk_ids - _cat_ids)         # 파일만 있고 미등재(= 등재 커밋 미완)
        ok_reg_set = (_cat_err is None and not _catalog_only and not _file_only)
        checks.append(('OPEN_ARCH-ⓒ 카탈로그 ↔ expects/ 집합 동등 — 등재 id 집합과 디스크 '
                       '`*.expect.json` 스템 집합이 정확히 일치(선등재 파일·고아 등재 양방향 fail-close)',
                       ok_reg_set))
        print('[selftest] OPEN_ARCH-ⓒ 집합 동등: %s (카탈로그 %d · 파일 %d%s)'
              % ('PASS' if ok_reg_set else 'FAIL', len(_cat_ids), len(_disk_ids),
                 '' if ok_reg_set else ' · catalog_only=%r file_only=%r%s'
                 % (_catalog_only, _file_only, (' · %s' % _cat_err) if _cat_err else '')))

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

        # ==================== ⓑ4~ⓑ6 ★launcher --plan 파서 계약 직접 주장 ====================
        # 26-08-08 리드 처분 ⓑ. 이 파일의 --plan stdout 은 본작업 launcher 가 파싱하는데
        # (Start-MoeDirect.ps1 ConvertFrom-TemplatePlanText), 그 계약을 selftest 가 주장한 적이
        # 없어 ⑤ 시공의 stdout 줄 삭제를 76/76 이 통과시켰다(사람 검수만 잡았다). 아래 3관문은
        # 삭제·재배치·문면 변경을 전부 FAIL 로 만든다. --plan 은 0바이트 계약이라 부작용이 없다.
        lpc_out = os.path.join(scratch, 'out_tpl_plan')
        lpc_args = argparse.Namespace(model=tpl_model, out=lpc_out, profile=None, scope=None,
                                      arch_template=True, mode=MODE_BIN, source_full_sha=False)
        lpc_buf = io.StringIO()
        lpc_saved_stdout = sys.stdout
        try:
            sys.stdout = lpc_buf
            cmd_plan(lpc_args)
        finally:
            sys.stdout = lpc_saved_stdout
        lpc_lines = lpc_buf.getvalue().split('\n')

        # ---- ⓑ4 소비 6줄 + expect 헤더 + 완료 줄이 각 1회, keyed 줄은 전부 헤더 앞 ----
        lpc_notes = []
        lpc_idx = {}
        for lpc_key, lpc_pat in LAUNCHER_PLAN_KEYED_LINES:
            lpc_hits = [i for i, ln in enumerate(lpc_lines) if re.match(lpc_pat, ln)]
            lpc_idx[lpc_key] = lpc_hits
            if len(lpc_hits) != 1:
                lpc_notes.append('"%s" 줄 %d회(기대 1회)' % (lpc_key, len(lpc_hits)))
        lpc_head = [i for i, ln in enumerate(lpc_lines) if ln == LAUNCHER_PLAN_EXPECT_HEAD]
        lpc_done = [i for i, ln in enumerate(lpc_lines) if ln == LAUNCHER_PLAN_DONE_LINE]
        if len(lpc_head) != 1:
            lpc_notes.append('derived expect 헤더 %d회(기대 1회)' % len(lpc_head))
        if len(lpc_done) != 1:
            lpc_notes.append('--plan 완료 줄 %d회(기대 1회)' % len(lpc_done))
        if len(lpc_head) == 1 and len(lpc_done) == 1:
            # 파서는 헤더 이후의 모든 줄을 expect 본문으로 삼는다 — keyed 줄이 헤더 뒤로 밀리면
            # 파서가 그 줄을 본문으로 삼켜 'the plan has no "<key>" line' 으로 죽는다.
            for lpc_key, lpc_hits in lpc_idx.items():
                if lpc_hits and lpc_hits[0] > lpc_head[0]:
                    lpc_notes.append('"%s" 줄이 expect 헤더 뒤에 있다' % lpc_key)
            if lpc_done[0] < lpc_head[0]:
                lpc_notes.append('완료 줄이 expect 헤더보다 앞에 있다')
        ok_lpc = (not lpc_notes)
        checks.append(('OPEN_ARCH-ⓑ4 launcher --plan 파서 계약 — 소비 6줄 + derived expect 헤더 + 완료 줄이 '
                       '실 stdout 에 각 1회·keyed 줄은 전부 헤더 앞(본작업 소비 표면)', ok_lpc))
        print('[selftest] OPEN_ARCH-ⓑ4 파서 소비 줄: %s %s'
              % ('PASS' if ok_lpc else 'FAIL', lpc_notes if lpc_notes else ''))

        # ---- ⓑ5 파서의 2차 대조(expect 본문 strict JSON + 요약줄과 필드 일치)까지 성립 ----
        lpc2 = []
        if ok_lpc:
            lpc_m = dict((k, re.match(dict(LAUNCHER_PLAN_KEYED_LINES)[k], lpc_lines[v[0]]))
                         for k, v in lpc_idx.items())
            lpc_from = lpc_m['derive'].group(1)
            lpc_scope = lpc_m['derive'].group(2)
            lpc_arch = lpc_m['arch'].group(1)
            lpc_split = lpc_from.split('@')
            if len(lpc_split) != 2:
                lpc2.append('derived_from 이 <template_id>@<version> 이 아니다: %r' % lpc_from)
                lpc_split = [lpc_from, '']
            if not re.match(r'^\d{1,8}$', lpc_split[1]):
                lpc2.append('template version 이 숫자가 아니다: %r' % lpc_split[1])
            if lpc_scope not in ('all', 'execution'):
                lpc2.append('알 수 없는 routed scope: %r' % lpc_scope)
            if lpc_arch != lpc_split[0]:
                lpc2.append('arch(%s) 와 template id(%s) 불일치' % (lpc_arch, lpc_split[0]))
            lpc_nums = {'n_layer': int(lpc_m['arch'].group(2)), 'n_expert': int(lpc_m['arch'].group(3)),
                        'n_expert_used': int(lpc_m['arch'].group(4)),
                        'routed_tensors': int(lpc_m['tpl'].group(4)),
                        'expert_bytes_total': int(lpc_m['bytes'].group(1))}
            lpc_moe = int(lpc_m['moe'].group(1))
            lpc_tpl_layers = int(lpc_m['tpl'].group(3))
            lpc_stride_max = int(lpc_m['stride'].group(4))
            if lpc_nums['n_expert'] <= 0:  lpc2.append('n_expert 가 양수가 아니다')
            if lpc_stride_max <= 0:        lpc2.append('slot_stride_max 가 양수가 아니다')
            if lpc_moe <= 0:               lpc2.append('moe_layers 가 0 이다')
            if lpc_moe != lpc_tpl_layers:
                lpc2.append('template layer 수(%d) 와 layout layer 수(%d) 불일치' % (lpc_tpl_layers, lpc_moe))
            # 파서와 같은 2차 읽기: expect 본문을 strict JSON 으로 재파싱해 요약줄과 대조한다.
            lpc_body = '\n'.join(lpc_lines[lpc_head[0] + 1:lpc_done[0]])
            try:
                lpc_ex = _strict_json_load_bytes(lpc_body.encode('utf-8'))
            except Exception as e:
                lpc_ex = None
                lpc2.append('derived expect 본문이 strict JSON 이 아니다: %r' % (e,))
            if lpc_ex is not None:
                for lpc_f, lpc_want in (('derived_from', lpc_from), ('template_id', lpc_split[0]),
                                        ('template_version', lpc_split[1]), ('routed_scope', lpc_scope),
                                        ('arch', lpc_arch),
                                        ('inventory_sha256', lpc_m['derive'].group(4))):
                    if str(lpc_ex.get(lpc_f)) != str(lpc_want):
                        lpc2.append('expect %s=%r 가 요약줄(%r)과 불일치' % (lpc_f, lpc_ex.get(lpc_f), lpc_want))
                for lpc_f, lpc_want in lpc_nums.items():
                    lpc_v = lpc_ex.get(lpc_f)
                    if not isinstance(lpc_v, int) or isinstance(lpc_v, bool) or lpc_v < 0 or lpc_v != lpc_want:
                        lpc2.append('expect %s=%r 가 요약줄(%d)과 불일치' % (lpc_f, lpc_v, lpc_want))
        else:
            lpc2.append('ⓑ4 가 FAIL 이라 2차 대조를 수행하지 못했다')
        ok_lpc2 = (not lpc2)
        checks.append(('OPEN_ARCH-ⓑ5 launcher 파서 2차 대조 — derived expect 본문이 strict JSON 이고 '
                       '요약줄 11필드·layer 수 등식·양수 조건이 전부 일치', ok_lpc2))
        print('[selftest] OPEN_ARCH-ⓑ5 파서 2차 대조: %s %s'
              % ('PASS' if ok_lpc2 else 'FAIL', lpc2 if lpc2 else ''))

        # ---- ⓑ6 bin 경로 stdout 무변경 계약(cmd_plan 자기 주석) — 머리 4줄 순서 + mode 줄 부재 ----
        # 파서가 소비하지는 않지만 ⑤ 시공이 실제로 삭제한 줄이 여기 있다(`profile:`).
        lpc3 = []
        if len(lpc_lines) < len(LAUNCHER_PLAN_BIN_PREAMBLE):
            lpc3.append('stdout 이 %d줄뿐이다' % len(lpc_lines))
        else:
            lpc_want_exact = (LAUNCHER_PLAN_BIN_PREAMBLE[0],
                              None,
                              'model: %s' % tpl_model,
                              'out (planned target): %s' % lpc_out)
            for lpc_i, lpc_pre in enumerate(LAUNCHER_PLAN_BIN_PREAMBLE):
                lpc_got = lpc_lines[lpc_i]
                lpc_exp = lpc_want_exact[lpc_i]
                if lpc_exp is None:
                    if not lpc_got.startswith(lpc_pre):
                        lpc3.append('%d번째 줄이 %r 로 시작하지 않는다: %r' % (lpc_i + 1, lpc_pre, lpc_got))
                elif lpc_got != lpc_exp:
                    lpc3.append('%d번째 줄 불일치: 기대 %r 실제 %r' % (lpc_i + 1, lpc_exp, lpc_got))
        lpc_mode_lines = [ln for ln in lpc_lines if ln.startswith('mode: ')]
        if lpc_mode_lines:
            lpc3.append('bin 경로에 mode: 줄이 있다 %r' % lpc_mode_lines)
        ok_lpc3 = (not lpc3)
        checks.append(('OPEN_ARCH-ⓑ6 bin 경로 --plan stdout 무변경 계약 — 머리 4줄(헤더·profile·model·out) '
                       '문면과 순서 보존 + mode: 줄 부재', ok_lpc3))
        print('[selftest] OPEN_ARCH-ⓑ6 bin stdout 머리 4줄: %s %s'
              % ('PASS' if ok_lpc3 else 'FAIL', lpc3 if lpc3 else ''))

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

        # ================= repack v3(mode=virtual) — SPEC_REPACK_V3 §6-4 중 플래너 소관 =================
        # ⑩child terminal·⑪prefetch/bounce·읽기 경로/ptr parity(①②④⑤)는 **소비자(§9-6) 소관**이라
        # 이 관문 밖이다. 여기서는 생산·검증 쪽만 폐합한다.
        #
        # ⑥ mode=bin 회귀: 위 전 항목(재팩·독립 2패스 verify·네거티브 23종·부록A 8종·OPEN_ARCH 전건)
        #    이 그대로 bin 관문이다. 추가로 §2-1 "bin 은 schema 2.0·mode 필드 부재" 를 못 박는다.
        v3_mf_bin = _load_manifest_disk(out_dir)
        ok_v3_bin = (v3_mf_bin.get('schema_version') == SCHEMA_VERSION and 'mode' not in v3_mf_bin
                     and 'bin_file_bytes' in v3_mf_bin.get('totals', {})
                     and all(('record_base' in L and 'stride_bytes' in L) for L in v3_mf_bin['layout']['layers'])
                     and all(('part_offset' in p and 'part_bytes' in p)
                             for L in v3_mf_bin['layout']['layers'] for p in L['parts']))
        # 같은 dict 로 mode 가드를 직접 행사(bin=2.0·mode 부재만 통과, 3.0+bin 조합은 산출 금지)
        ok_v3_guard_bin, _ = True, None
        try:
            _guard_manifest_mode(v3_mf_bin, MODE_BIN)
        except RepackAbort:
            ok_v3_guard_bin = False
        _bad_guard = dict(v3_mf_bin); _bad_guard['schema_version'] = SCHEMA_VERSION_V3; _bad_guard['mode'] = MODE_BIN
        ok_v3_guard_30bin, _m = _expect_abort(lambda: _guard_manifest_mode(_bad_guard, MODE_BIN))
        _bad_guard2 = dict(v3_mf_bin); _bad_guard2['mode'] = MODE_BIN
        ok_v3_guard_modefield, _m2 = _expect_abort(lambda: _guard_manifest_mode(_bad_guard2, MODE_BIN))
        ok_v3_bin = ok_v3_bin and ok_v3_guard_bin and ok_v3_guard_30bin and ok_v3_guard_modefield
        checks.append(('v3-⑥ mode=bin 회귀: 기존 산출물이 schema_version "2.0"·mode 필드 부재·bin 좌표'
                       '(record_base/stride_bytes/part_offset/part_bytes) 유지 + 생산 가드가 "3.0"+bin 과 '
                       'mode 필드 부착을 거부', ok_v3_bin))
        print('[selftest] v3-⑥ bin 회귀: %s (schema=%r mode부재=%s guard(bin)=%s guard(3.0+bin)=%s guard(mode필드)=%s)'
              % ('PASS' if ok_v3_bin else 'FAIL', v3_mf_bin.get('schema_version'), 'mode' not in v3_mf_bin,
                 ok_v3_guard_bin, ok_v3_guard_30bin, ok_v3_guard_modefield))

        # A 는 실기 질의값이라 기대치를 하드코딩하지 않는다 — **같은 원시 질의 + 인라인 산식**으로
        # 독립 재도출한다(테스트가 피검 함수를 재사용하지 않게).
        _v3_aq = query_sector_alignment_for_path(scratch)
        A_v3 = max(4096, int(_v3_aq['logical']), int(_v3_aq['physical']))
        # weight slice(hidden^2*4)가 A 배수가 되는 최소 hidden(2의 거듭제곱). bias slice(hidden*4)는
        # 항상 A 미만이라 같은 픽스처가 aligned/비4K 파트를 동시에 갖는다.
        v3_hid = 32
        while (v3_hid * v3_hid * 4) % A_v3 != 0:
            v3_hid *= 2
        # 합성 GGUF 는 마지막 텐서가 EOF 에서 끝나므로, 마지막 routed 텐서 뒤에 A 이상의 여유가 없으면
        # §2-4 bracket EOF 경계에 걸린다(그 자체가 아래 네거티브 케이스). 양성 픽스처는 비-routed
        # filler 텐서로 꼬리 여유를 준다(실물 GGUF 의 트레일링 패딩·후속 텐서에 대응).
        V3_FILLER = {'name': 'blk.0.ffn_gate_inp.weight', 'dims': [8192, 4], 'type': 'F32'}
        V3_FILLER2 = {'name': 'blk.1.ffn_gate_inp.weight', 'dims': [8192, 4], 'type': 'F32'}

        def _v3_expected_slots(path, A, scope='all'):
            """§2-4 점화식을 **인라인**으로 재계산(슬롯 산술은 피검 대상이므로 재사용 금지).
            slice/abs_offset 자체는 v2 동결 경로(build_layout)에서 가져온다."""
            lay = build_layout(load_model_shards(path), scope=scope)
            rows = []
            for L in lay['layers']:
                so = 0
                parts = []
                for p in L['parts']:
                    sl, ao = p['part_bytes'], p['abs_offset']
                    al = (sl % A == 0)
                    if al:
                        head = ao % A
                        region = -(-(head + sl) // A) * A
                        do_ = so + head
                        staging = None
                    else:
                        head = None
                        region = -(-sl // A) * A
                        do_ = so
                        staging = region + A
                    parts.append({'name': p['name'], 'aligned': al, 'bracket_head': head,
                                  'slot_offset': so, 'data_offset': do_, 'staging_bytes': staging,
                                  'slice_bytes': sl, 'abs_offset': ao, 'source_index': p['source_index']})
                    so += region
                rows.append({'layer': L['layer'], 'layer_slot_bytes': so, 'parts': parts})
            return lay, rows

        def _v3_cmp_slots(mf, rows, n_expert):
            """manifest 값 ↔ 인라인 기대치 전항 대조(+ records 재생성 대조)."""
            bad = []
            if len(mf['layout']['layers']) != len(rows):
                return ['layer count %d != %d' % (len(mf['layout']['layers']), len(rows))]
            for ml, er in zip(mf['layout']['layers'], rows):
                if ml['layer'] != er['layer'] or ml['layer_slot_bytes'] != er['layer_slot_bytes']:
                    bad.append('layer %r slot bytes %r != %r' % (er['layer'], ml.get('layer_slot_bytes'),
                                                                 er['layer_slot_bytes']))
                if len(ml['parts']) != len(er['parts']):
                    bad.append('layer %r part count' % er['layer']); continue
                for mp, ep in zip(ml['parts'], er['parts']):
                    for f in ('slice_bytes', 'abs_offset', 'aligned', 'slot_offset', 'data_offset'):
                        if mp.get(f) != ep[f]:
                            bad.append('layer %r part %s %s: %r != %r' % (er['layer'], ep['name'], f,
                                                                           mp.get(f), ep[f]))
                    if ep['aligned']:
                        if mp.get('bracket_head') != ep['bracket_head']:
                            bad.append('layer %r part %s bracket_head %r != %r'
                                       % (er['layer'], ep['name'], mp.get('bracket_head'), ep['bracket_head']))
                        if 'staging_bytes' in mp:
                            bad.append('layer %r part %s: aligned part must not carry staging_bytes' % (er['layer'], ep['name']))
                    else:
                        if 'bracket_head' in mp:
                            bad.append('layer %r part %s: non-4K part must not carry bracket_head' % (er['layer'], ep['name']))
                        if mp.get('staging_bytes') != ep['staging_bytes']:
                            bad.append('layer %r part %s staging_bytes %r != %r'
                                       % (er['layer'], ep['name'], mp.get('staging_bytes'), ep['staging_bytes']))
            if mf['layout']['slot_stride_max'] != max(r['layer_slot_bytes'] for r in rows):
                bad.append('slot_stride_max %r != %d' % (mf['layout']['slot_stride_max'],
                                                          max(r['layer_slot_bytes'] for r in rows)))
            want_records = []
            for er in rows:
                for e in range(n_expert):
                    for ep in er['parts']:
                        want_records.append({'layer': er['layer'], 'expert': e, 'part': ep['name'],
                                             'source_index': ep['source_index'],
                                             'src_offset': ep['abs_offset'] + e * ep['slice_bytes'],
                                             'slice_bytes': ep['slice_bytes'], 'data_offset': ep['data_offset']})
            if mf['records'] != want_records:
                first = next((i for i, (a, b) in enumerate(zip(mf['records'], want_records)) if a != b), None)
                bad.append('records mismatch (count %d vs %d, first diff at %r)'
                           % (len(mf['records']), len(want_records), first))
            if mf['totals']['virtual_payload_bytes'] != sum(ep['slice_bytes'] * n_expert
                                                            for er in rows for ep in er['parts']):
                bad.append('totals.virtual_payload_bytes %r' % mf['totals'].get('virtual_payload_bytes'))
            return bad

        # ---- v3-P1 aligned+비4K 혼재: 생성 → verifier PASS → 원자 승격 + 산술 독립 검산 ----
        v3_mixed = os.path.join(scratch, 'v3_mixed.gguf')
        write_synthetic_gguf([v3_mixed], n_expert=4, moe_layers=(0, 1), schema='separate', bias=True,
                             hidden=v3_hid, alignment=32, seed=4101, extra_tensors=(V3_FILLER,))
        v3_out = os.path.join(scratch, 'out_v3_mixed')
        v3_manifest, v3_report = do_virtual_plan(v3_mixed, v3_out, profile_id=None, force=False,
                                                 enforce_reference=False, allow_default_align=True)
        v3_mf = _load_manifest_disk(v3_out)
        v3_disk_sha = hashlib.sha256(open(os.path.join(v3_out, MANIFEST_FILENAME), 'rb').read()).hexdigest()
        v3_rep_disk = json.loads(open(os.path.join(v3_out, PLAN_REPORT_FILENAME), 'r', encoding='utf-8').read())
        v3_lay, v3_rows = _v3_expected_slots(v3_mixed, A_v3)
        v3_bad = _v3_cmp_slots(v3_mf, v3_rows, v3_lay['n_expert'])
        v3_aligned_cnt = sum(1 for r in v3_rows for p in r['parts'] if p['aligned'])
        v3_nonal_cnt = sum(1 for r in v3_rows for p in r['parts'] if not p['aligned'])
        ok_v3_p1 = (v3_report['pass'] is True and not v3_bad
                    and v3_aligned_cnt == 6 and v3_nonal_cnt == 6           # weight 6 aligned · bias 6 비4K
                    and v3_mf['schema_version'] == SCHEMA_VERSION_V3 and v3_mf['mode'] == MODE_VIRTUAL
                    and 'bin_file_bytes' not in v3_mf['totals']
                    and all('record_base' not in L and 'stride_bytes' not in L for L in v3_mf['layout']['layers'])
                    and all('part_offset' not in p and 'part_bytes' not in p
                            for L in v3_mf['layout']['layers'] for p in L['parts'])
                    and not os.path.exists(os.path.join(v3_out, MANIFEST_FILENAME + '.partial'))
                    and not os.path.exists(os.path.join(v3_out, 'experts.bin'))
                    and v3_rep_disk.get('pass') is True
                    and v3_rep_disk.get('manifest_sha256') == v3_disk_sha
                    and v3_rep_disk.get('manifest') == MANIFEST_FILENAME
                    and v3_rep_disk.get('reference_lock', {}).get('profile_id') == 'selftest-exempt'
                    and v3_rep_disk.get('cardinality', {}).get('records') == v3_mf['totals']['n_records']
                    and v3_mf['totals']['n_records'] == len(v3_lay['layers']) * v3_lay['n_expert'] * 6
                    and len(v3_mf['layout']['align_query']) == 1
                    and v3_mf['layout']['align_bytes'] == A_v3
                    and len(v3_mf['sources'][0]['digest']['sha256']) == 64)
        checks.append(('v3-P1 mode=virtual 생성→독립 verifier PASS→원자 승격(.partial 소멸·experts.bin 부재)'
                       ' + §2-4 슬롯 산술 인라인 독립 검산 전항 일치(aligned 6·비4K 6 혼재) + plan_report 결속'
                       '(pass/manifest_sha256/cardinality)', ok_v3_p1))
        print('[selftest] v3-P1 virtual E2E: %s (aligned=%d non4K=%d records=%d A=%d %s)'
              % ('PASS' if ok_v3_p1 else 'FAIL', v3_aligned_cnt, v3_nonal_cnt,
                 v3_mf['totals']['n_records'], A_v3, ('불일치=%r' % v3_bad[:4]) if v3_bad else ''))

        # ---- v3-P2 비4K 전량 프로파일(gpt-oss 형상 대응): region/data_offset/staging 산술 ----
        v3_non = os.path.join(scratch, 'v3_non4k.gguf')
        write_synthetic_gguf([v3_non], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=8, alignment=32, seed=4102, extra_tensors=(V3_FILLER,))
        v3_out_non = os.path.join(scratch, 'out_v3_non4k')
        _, v3_rep_non = do_virtual_plan(v3_non, v3_out_non, profile_id=None, force=False,
                                        enforce_reference=False, allow_default_align=True)
        v3_mf_non = _load_manifest_disk(v3_out_non)
        v3_lay_non, v3_rows_non = _v3_expected_slots(v3_non, A_v3)
        v3_bad_non = _v3_cmp_slots(v3_mf_non, v3_rows_non, v3_lay_non['n_expert'])
        v3_np = [p for L in v3_mf_non['layout']['layers'] for p in L['parts']]
        ok_v3_p2 = (v3_rep_non['pass'] is True and not v3_bad_non
                    and all(p['aligned'] is False for p in v3_np)
                    and all('bracket_head' not in p for p in v3_np)
                    and all(p['data_offset'] == p['slot_offset'] for p in v3_np)
                    and all(p['staging_bytes'] == (-(-p['slice_bytes'] // A_v3)) * A_v3 + A_v3 for p in v3_np)
                    and all(L['layer_slot_bytes'] == len(L['parts']) * A_v3
                            for L in v3_mf_non['layout']['layers']))
        checks.append(('v3-P2 비4K 프로파일(전 파트 aligned=false): bracket_head 기재 금지·data_offset=slot_offset·'
                       'region=align_up(slice,A)·staging_bytes=align_up(slice,A)+A 산술 일치 + verifier PASS',
                       ok_v3_p2))
        print('[selftest] v3-P2 non-4K: %s (%s)' % ('PASS' if ok_v3_p2 else 'FAIL',
              ('불일치=%r' % v3_bad_non[:4]) if v3_bad_non else 'parts=%d' % len(v3_np)))

        # ---- v3-P3 shard 경계 part(layer1 down.weight 가 타 shard) ----
        v3_s1 = os.path.join(scratch, 'v3split-00001-of-00002.gguf')
        v3_s2 = os.path.join(scratch, 'v3split-00002-of-00002.gguf')
        write_synthetic_gguf([v3_s1, v3_s2], n_expert=4, moe_layers=(0, 1), schema='separate', bias=True,
                             hidden=v3_hid, alignment=32, seed=4103,
                             shard_of={'blk.1.ffn_down_exps.weight': 1, V3_FILLER2['name']: 1},
                             extra_tensors=(V3_FILLER, V3_FILLER2))
        v3_out_sp = os.path.join(scratch, 'out_v3_split')
        _, v3_rep_sp = do_virtual_plan(v3_s1, v3_out_sp, profile_id=None, force=False,
                                       enforce_reference=False, allow_default_align=True)
        v3_mf_sp = _load_manifest_disk(v3_out_sp)
        v3_lay_sp, v3_rows_sp = _v3_expected_slots(v3_s1, A_v3)
        v3_bad_sp = _v3_cmp_slots(v3_mf_sp, v3_rows_sp, v3_lay_sp['n_expert'])
        v3_srcidx = {p['source_tensor']: p['source_index'] for L in v3_mf_sp['layout']['layers']
                     for p in L['parts']}
        v3_recs_s1 = [r for r in v3_mf_sp['records'] if r['source_index'] == 1]
        ok_v3_p3 = (v3_rep_sp['pass'] is True and not v3_bad_sp
                    and v3_srcidx.get('blk.1.ffn_down_exps.weight') == 1
                    and v3_srcidx.get('blk.1.ffn_gate_exps.weight') == 0
                    and v3_srcidx.get('blk.0.ffn_down_exps.weight') == 0
                    and len(v3_mf_sp['sources']) == 2 and len(v3_mf_sp['layout']['align_query']) == 2
                    and [q['source_index'] for q in v3_mf_sp['layout']['align_query']] == [0, 1]
                    and len(v3_recs_s1) == v3_lay_sp['n_expert']
                    and len({s['digest']['sha256'] for s in v3_mf_sp['sources']}) == 2)
        checks.append(('v3-P3 shard 경계 part(2-shard·layer1 down.weight 가 shard1): parts/records 의 source_index·'
                       'align_query 전 shard 기록·shard 별 header digest 상이 + verifier PASS', ok_v3_p3))
        print('[selftest] v3-P3 shard 경계: %s (%s)' % ('PASS' if ok_v3_p3 else 'FAIL',
              ('불일치=%r' % v3_bad_sp[:4]) if v3_bad_sp else 's1 records=%d' % len(v3_recs_s1)))

        # ---- v3 네거티브 공용 러너(승격 산출물 복제 → manifest 손상 → 독립 verifier 재호출) ----
        def _v3_neg(tag, mutate, src_dir=None, model=None, raw_mutate=None):
            d = os.path.join(scratch, 'v3neg_%s' % tag)
            shutil.copytree(src_dir or v3_out, d)
            if raw_mutate is not None:
                p = os.path.join(d, MANIFEST_FILENAME)
                raw_mutate(p)
            else:
                mf = _load_manifest_disk(d)
                mutate(mf)
                _save_manifest_disk(d, mf)
            rep = verify_virtual_manifest(model or v3_mixed, d, profile_id=None,
                                          enforce_reference=False, allow_default_align=True)
            return (rep['pass'] is False), rep

        def _v3_first_part(mf, layer_idx=0, part_idx=0):
            return mf['layout']['layers'][layer_idx]['parts'][part_idx]

        # ---- v3-P4 routed_scope=execution 승계(부록A 경로): manifest 기록 + verifier 가 그 scope 로 재도출 ----
        v3_exec = os.path.join(scratch, 'v3_exec.gguf')
        write_synthetic_gguf([v3_exec], n_expert=4, moe_layers=(0, 1, 2, 3), block_count=4, schema='separate',
                             bias=True, hidden=v3_hid, alignment=32, seed=4105, nextn_kv=1,
                             extra_tensors=(V3_FILLER,))
        v3_out_ex = os.path.join(scratch, 'out_v3_exec')
        _, v3_rep_ex = do_virtual_plan(v3_exec, v3_out_ex, profile_id=None, force=False,
                                       enforce_reference=False, allow_default_align=True, scope='execution')
        v3_mf_ex = _load_manifest_disk(v3_out_ex)
        v3_lay_ex, v3_rows_ex = _v3_expected_slots(v3_exec, A_v3, scope='execution')
        v3_bad_ex = _v3_cmp_slots(v3_mf_ex, v3_rows_ex, v3_lay_ex['n_expert'])
        # verifier 가 manifest.model.routed_scope 로 구동되는지: 그 키를 지우면 all 로 재도출돼 층 수가
        # 어긋나 반드시 FAIL 이어야 한다(scope 승계가 실제로 결속되어 있다는 증거).
        ok_ex_neg, rep_ex_neg = _v3_neg('scope_drop', lambda mf: mf['model'].pop('routed_scope'),
                                        src_dir=v3_out_ex, model=v3_exec)
        ok_v3_p4 = (v3_rep_ex['pass'] is True and not v3_bad_ex
                    and v3_mf_ex['model'].get('routed_scope') == 'execution'
                    and v3_mf_ex['model']['moe_layers'] == [0, 1, 2]
                    and len(v3_mf_ex['layout']['layers']) == 3
                    and v3_mf_ex['totals']['n_records'] == 3 * 4 * 6
                    and ok_ex_neg)
        checks.append(('v3-P4 routed_scope=execution 승계(부록A): manifest.model.routed_scope 기록·tail 층 제외'
                       '(moe_layers=[0,1,2])·verifier 가 그 scope 로 독립 재도출 + routed_scope 삭제 위조 거부',
                       ok_v3_p4))
        print('[selftest] v3-P4 execution scope: %s (moe_layers=%r scope삭제거부=%s %s)'
              % ('PASS' if ok_v3_p4 else 'FAIL', v3_mf_ex['model'].get('moe_layers'), ok_ex_neg,
                 ('불일치=%r' % v3_bad_ex[:4]) if v3_bad_ex else ''))

        # ---- v3-P5 층별 quant 혼용(payload 비균일): stride 가 층별로 갈리는 경로의 양성 회귀 ----
        # 기존 등재 프로파일 9종은 전 층 payload 가 균일해서 이 경로가 한 번도 양성으로 밟히지
        # 않았다. K3(UD 동적 양자)는 층마다 routed 타입이 달라 payload 가 3종으로 갈리므로,
        # 등재로 그 경로가 도달 가능해지기 **전에** 회귀를 세운다. 픽스처는 신규 타입을 쓰지 않고
        # **기존 동결 타입 2종**(Q4_K/Q6_K)으로 비균일을 성립시킨다 — 비균일 자체가 피검 대상이지
        # 특정 타입이 아니다. hidden 은 두 타입의 block_values(256) 배수여야 한다.
        v3_mq_hid = 256
        v3_mq_types = ('Q4_K', 'Q6_K')
        v3_mq = os.path.join(scratch, 'v3_mixedquant.gguf')
        write_synthetic_gguf([v3_mq], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=v3_mq_hid, layer_quant={0: v3_mq_types[0], 1: v3_mq_types[1]},
                             alignment=32, seed=4106, extra_tensors=(V3_FILLER,))
        v3_out_mq = os.path.join(scratch, 'out_v3_mixedquant')
        _, v3_rep_mq = do_virtual_plan(v3_mq, v3_out_mq, profile_id=None, force=False,
                                       enforce_reference=False, allow_default_align=True)
        v3_mf_mq = _load_manifest_disk(v3_out_mq)
        v3_lay_mq, v3_rows_mq = _v3_expected_slots(v3_mq, A_v3)
        v3_bad_mq = _v3_cmp_slots(v3_mf_mq, v3_rows_mq, v3_lay_mq['n_expert'])
        # per-expert slice 를 QUANT_TRAITS 에서 **인라인 재도출**한다(per_expert_slice_bytes 는
        # 피검 대상이라 재사용 금지 — §2-4 슬롯 검산과 같은 규율).
        def _mq_slice(tt):
            bv, bb = QUANT_TRAITS[tt]
            return (v3_mq_hid // bv) * bb * v3_mq_hid
        v3_mq_want = [_mq_slice(t) * 3 for t in v3_mq_types]          # 층별 payload(파트 3개)
        v3_mq_got = [sum(p['slice_bytes'] for p in r['parts']) for r in v3_rows_mq]
        v3_mq_slots = [r['layer_slot_bytes'] for r in v3_rows_mq]
        ok_v3_p5 = (v3_rep_mq['pass'] is True and not v3_bad_mq
                    and sorted(v3_lay_mq['used_types']) == sorted(v3_mq_types)
                    and v3_mq_got == v3_mq_want                        # 층별 payload 가 이론 재도출과 일치
                    and len(set(v3_mq_got)) == 2                       # ★비균일이 실제로 성립
                    and len(set(v3_mq_slots)) == 2                     # ★그 비균일이 슬롯까지 전파
                    and v3_mf_mq['layout']['slot_stride_max'] == max(v3_mq_slots)
                    and [L['layer_slot_bytes'] for L in v3_mf_mq['layout']['layers']] == v3_mq_slots
                    and v3_mf_mq['mode'] == MODE_VIRTUAL
                    and v3_mf_mq['totals']['n_records'] == len(v3_rows_mq) * v3_lay_mq['n_expert'] * 3
                    and not os.path.exists(os.path.join(v3_out_mq, MANIFEST_FILENAME + '.partial'))
                    and not os.path.exists(os.path.join(v3_out_mq, 'experts.bin')))
        checks.append(('v3-P5 층별 quant 혼용(payload 비균일) virtual 양성: 층별 payload 가 QUANT_TRAITS '
                       '인라인 재도출과 일치하고 **서로 다르며**, 그 비균일이 layer_slot_bytes·'
                       'slot_stride_max=max 까지 전파 + verifier PASS·원자 승격', ok_v3_p5))
        print('[selftest] v3-P5 비균일 payload: %s (types=%r payload=%r slots=%r stride_max=%r %s)'
              % ('PASS' if ok_v3_p5 else 'FAIL', list(v3_lay_mq['used_types']), v3_mq_got, v3_mq_slots,
                 v3_mf_mq['layout']['slot_stride_max'], ('불일치=%r' % v3_bad_mq[:4]) if v3_bad_mq else ''))

        # ---- v3-⑦ schema/mode 조합(전부 생산 중단/거부) ----
        v3_mode_cases = [
            ('mode 필드 누락', 'mode_missing', lambda mf: mf.pop('mode')),
            ('"3.0"+mode:"bin"(산출 금지 조합)', 'mode_bin', lambda mf: mf.__setitem__('mode', MODE_BIN)),
            ('미지 mode 값', 'mode_unknown', lambda mf: mf.__setitem__('mode', 'hyper')),
            ('스키마 불일치("2.0"+mode:"virtual")', 'schema_20',
             lambda mf: mf.__setitem__('schema_version', SCHEMA_VERSION)),
            ('스키마 미지("4.0")', 'schema_40', lambda mf: mf.__setitem__('schema_version', '4.0')),
        ]
        v3_mode_rows = []
        for label, tag, fn in v3_mode_cases:
            ok_n, rep_n = _v3_neg(tag, fn)
            v3_mode_rows.append((label, ok_n, (rep_n['problems'] or [''])[0][:90]))
        # v2 산출물을 v3 verifier 에 넣으면 거부(schema 2.0) · v3 산출물을 v2 --verify-only 에 넣으면 거부
        v3_mixed_dir_v2 = verify_virtual_manifest(model_path, out_dir, profile_id=None,
                                                  enforce_reference=False, allow_default_align=True)
        ok_cross_a = v3_mixed_dir_v2['pass'] is False
        ok_cross_b, msg_cross_b = _expect_abort(lambda: do_verify_only(v3_mixed, v3_out, profile_id=None,
                                                                       enforce_reference=False,
                                                                       allow_default_align=True))
        v3_mode_rows.append(('v2 산출물(2.0)을 v3 verifier 가 거부', ok_cross_a,
                             (v3_mixed_dir_v2['problems'] or [''])[0][:90]))
        v3_mode_rows.append(('v3 산출물을 v2 --verify-only 가 거부(experts.bin 부재)', ok_cross_b, msg_cross_b[:90]))
        ok_v3_mode = all(ok for _, ok, _ in v3_mode_rows)
        checks.append(('v3-⑦ schema/mode 조합 네거티브 %d종(mode 누락·"3.0"+bin·미지 mode·스키마 불일치/미지·'
                       'v2↔v3 교차 거부) 전부 거부' % len(v3_mode_rows), ok_v3_mode))
        print('[selftest] v3-⑦ schema/mode: %s' % ('PASS' if ok_v3_mode else 'FAIL'))
        for label, ok_n, note in v3_mode_rows:
            print('    [%s] %-46s %s' % ('PASS' if ok_n else 'FAIL', label, note))

        # ---- v3-⑧ 주소 네거티브(source_index 교환·abs_offset ±1/±A·slice/type/dims·overflow·bracket EOF) ----
        def _mut_abs(delta):
            def _f(mf):
                p = _v3_first_part(mf)
                p['abs_offset'] += delta
            return _f

        def _mut_abs_consistent(delta):
            """내부 정합까지 맞춘 위조(parts+records+source_tensors 동시 이동) — 재파싱 계층만 잡을 수 있다."""
            def _f(mf):
                p = _v3_first_part(mf)
                name, old = p['source_tensor'], p['abs_offset']
                p['abs_offset'] = old + delta
                if p['aligned']:
                    head = (old + delta) % mf['layout']['align_bytes']
                    p['bracket_head'] = head
                    p['data_offset'] = p['slot_offset'] + head
                for st in mf['source_tensors']:
                    if st['name'] == name:
                        st['abs_offset'] = old + delta
                for r in mf['records']:
                    if r['layer'] == mf['layout']['layers'][0]['layer'] and r['part'] == p['name']:
                        r['src_offset'] = old + delta + r['expert'] * p['slice_bytes']
                        r['data_offset'] = p['data_offset']
            return _f

        def _mut_swap_source_index(mf):
            ps = mf['layout']['layers'][1]['parts']
            a = next(p for p in ps if p['source_index'] == 1)
            b = next(p for p in ps if p['source_index'] == 0)
            a['source_index'], b['source_index'] = b['source_index'], a['source_index']

        v3_addr_cases = [
            ('abs_offset +1', 'abs_p1', _mut_abs(1), None, None),
            ('abs_offset -1', 'abs_m1', _mut_abs(-1), None, None),
            ('abs_offset +A', 'abs_pA', _mut_abs(A_v3), None, None),
            ('abs_offset -A', 'abs_mA', _mut_abs(-A_v3), None, None),
            ('abs_offset +A(내부 정합 위조 — 재파싱 계층만 적발)', 'abs_pA_cons', _mut_abs_consistent(A_v3), None, None),
            ('slice_bytes 불일치', 'slice',
             lambda mf: _v3_first_part(mf).__setitem__('slice_bytes', _v3_first_part(mf)['slice_bytes'] + A_v3),
             None, None),
            ('type 불일치', 'type', lambda mf: _v3_first_part(mf).__setitem__('type', 'Q8_0'), None, None),
            ('dims 불일치', 'dims',
             lambda mf: _v3_first_part(mf)['dims'].__setitem__(0, _v3_first_part(mf)['dims'][0] * 2), None, None),
            ('dims 실수화(int→float 우회)', 'dims_float',
             lambda mf: _v3_first_part(mf)['dims'].__setitem__(0, float(_v3_first_part(mf)['dims'][0])), None, None),
            ('checked uint64 정의역 초과(slice_bytes=2**64)', 'u64',
             lambda mf: _v3_first_part(mf).__setitem__('slice_bytes', 1 << 64), None, None),
            ('records src_offset 변조', 'rec_src',
             lambda mf: mf['records'][0].__setitem__('src_offset', mf['records'][0]['src_offset'] + 1), None, None),
            ('records 1건 삭제', 'rec_del', lambda mf: mf['records'].pop(), None, None),
            ('records 순서 교환', 'rec_swap',
             lambda mf: mf['records'].__setitem__(slice(0, 2), [mf['records'][1], mf['records'][0]]), None, None),
            ('source_tensors abs_offset 변조(witness 불일치)', 'st_abs',
             lambda mf: mf['source_tensors'][0].__setitem__('abs_offset',
                                                             mf['source_tensors'][0]['abs_offset'] + A_v3), None, None),
            ('sources digest 변조(DF-1 결속)', 'digest',
             lambda mf: mf['sources'][0]['digest'].__setitem__('sha256', '0' * 64), None, None),
            ('sources bytes 변조', 'src_bytes',
             lambda mf: mf['sources'][0].__setitem__('bytes', mf['sources'][0]['bytes'] + 1), None, None),
            # ★r1 실 결함 ① 의 네거티브. 이 셋이 없으면 새 대조는 "추가됐다"까지만 참이고
            # "실제로 발화한다"는 미증명이다 — 픽스처의 경로·mtime 은 언제나 일치하므로 양성만으로는
            # 무효한 검사와 구분되지 않는다(r1 이 지적한 바로 그 공백의 재발 방지).
            ('sources path 변조(다른 파일을 가리키는 manifest)', 'src_path',
             lambda mf: mf['sources'][0].__setitem__('path', mf['sources'][0]['path'] + '.moved'),
             None, None),
            ('sources mtime 변조(헤더·크기 동일·재작성만 발생)', 'src_mtime',
             lambda mf: mf['sources'][0].__setitem__('mtime', mf['sources'][0]['mtime'] + 1.0),
             None, None),
            ('align_query drive_root 변조(다른 볼륨에서 질의된 A)', 'aq_drive',
             lambda mf: mf['layout']['align_query'][0].__setitem__('drive_root', 'Z:\\'), None, None),
            ('align_bytes 변조(A=4096 고정 위조)', 'align',
             lambda mf: mf['layout'].__setitem__('align_bytes', A_v3 * 2), None, None),
            ('bin 좌표 필드 부착(record_base)', 'binfield',
             lambda mf: mf['layout']['layers'][0].__setitem__('record_base', 0), None, None),
            ('bin 좌표 필드 부착(totals.bin_file_bytes)', 'binfield2',
             lambda mf: mf['totals'].__setitem__('bin_file_bytes', 123), None, None),
        ]
        v3_addr_rows = []
        for label, tag, fn, src_dir, model_p in v3_addr_cases:
            ok_n, rep_n = _v3_neg(tag, fn, src_dir=src_dir, model=model_p)
            v3_addr_rows.append((label, ok_n, (rep_n['problems'] or [''])[0][:90]))
        # source_index 교환은 2-shard 산출물에서만 의미가 있다
        ok_swap, rep_swap = _v3_neg('src_swap', _mut_swap_source_index, src_dir=v3_out_sp, model=v3_s1)
        v3_addr_rows.append(('source_index 교환(2-shard)', ok_swap, (rep_swap['problems'] or [''])[0][:90]))
        # raw JSON 봉인 승계(중복 키·비표준 상수)
        def _raw_dup_key(path):
            text = open(path, 'r', encoding='utf-8').read()
            anchor = '"align_bytes": %d' % A_v3
            i = text.index(anchor)
            open(path, 'w', encoding='utf-8').write(
                text[:i] + '"align_bytes": 1, "align_bytes": %d' % A_v3 + text[i + len(anchor):])

        def _raw_nan(path):
            text = open(path, 'r', encoding='utf-8').read()
            key = '"mtime":'
            vs = text.index(key) + len(key); ve = text.index(',', vs)
            open(path, 'w', encoding='utf-8').write(text[:vs] + ' NaN' + text[ve:])

        ok_dupk, rep_dupk = _v3_neg('raw_dup', None, raw_mutate=_raw_dup_key)
        v3_addr_rows.append(('raw JSON 중복 키', ok_dupk and 'duplicate key' in (rep_dupk['problems'] or [''])[0],
                             (rep_dupk['problems'] or [''])[0][:90]))
        ok_nan, rep_nan = _v3_neg('raw_nan', None, raw_mutate=_raw_nan)
        v3_addr_rows.append(('raw JSON NaN(비표준 상수)',
                             ok_nan and 'non-standard JSON constant' in (rep_nan['problems'] or [''])[0],
                             (rep_nan['problems'] or [''])[0][:90]))
        # checked uint64 산술 단위 행사(오버플로=중단. Python int 는 임의정밀이라 정의역 확인이 계약)
        v3_u64_cases = [
            ('_u64(2**64)', lambda: _u64(1 << 64, 't')),
            ('_u64(-1)', lambda: _u64(-1, 't')),
            ('_u64(True)', lambda: _u64(True, 't')),
            ('_u64(4096.0)', lambda: _u64(4096.0, 't')),
            ('_u64_add(U64_MAX,1)', lambda: _u64_add(U64_MAX, 1, 't')),
            ('_u64_mul(2**63,4)', lambda: _u64_mul(1 << 63, 4, 't')),
            ('_u64_align_up(U64_MAX,4096)', lambda: _u64_align_up(U64_MAX, 4096, 't')),
        ]
        v3_u64_ok = True
        for label, fn in v3_u64_cases:
            ok_u, _m = _expect_abort(fn)
            if not ok_u:
                v3_u64_ok = False
                v3_addr_rows.append(('checked uint64 %s' % label, False, _m[:90]))
        # 생산 산술 자체가 checked 인지(합성 layout 으로 tensor_bytes 오버플로 유발)
        _ov_layout = {'n_expert': 2, 'layers': [{'layer': 0, 'payload_bytes': 0, 'parts': [
            {'name': 'gate.weight', 'source_tensor': 'x', 'source_index': 0, 'type': 'F32',
             'dims': [8, 8, 2], 'expert_axis': 2, 'part_offset': 0, 'part_bytes': (1 << 63) + 8,
             'abs_offset': 0, 'theory_bytes': 0}]}]}
        ok_ov, msg_ov = _expect_abort(lambda: compute_virtual_layout(_ov_layout, A_v3, {0: U64_MAX}))
        v3_addr_rows.append(('compute_virtual_layout 의 checked uint64(tensor_bytes 오버플로)',
                             ok_ov and v3_u64_ok and 'uint64' in msg_ov, msg_ov[:90]))
        # bracket EOF 초과(payload 경계는 통과 · 꼬리 여유 없는 픽스처) = 자동 bin 회귀 판정
        v3_eof = os.path.join(scratch, 'v3_eof.gguf')
        write_synthetic_gguf([v3_eof], n_expert=4, moe_layers=(0, 1), schema='separate', bias=False,
                             hidden=8, alignment=32, seed=4104)
        v3_out_eof = os.path.join(scratch, 'out_v3_eof')
        ok_eof, msg_eof = _expect_abort(lambda: do_virtual_plan(v3_eof, v3_out_eof, profile_id=None,
                                                                 force=False, enforce_reference=False,
                                                                 allow_default_align=True))
        _eof_h = parse_gguf_header(v3_eof)
        _eof_last = max(t['abs_offset'] + theory_tensor_bytes(t['type'], t['dims'])
                        for t in _eof_h['tensors'] if TENSOR_NAME_RE.match(t['name']))
        ok_eof = (ok_eof and 'bracket EOF' in msg_eof and 'mode=bin' in msg_eof
                  and _eof_last <= _eof_h['file_bytes']                              # payload 경계는 통과
                  and (-(-_eof_last // A_v3)) * A_v3 > _eof_h['file_bytes']          # bracket 만 초과
                  and not os.path.exists(os.path.join(v3_out_eof, MANIFEST_FILENAME)))
        v3_addr_rows.append(('bracket EOF 초과(payload 경계 통과) -> 자동 bin 회귀 판정·미승격', ok_eof, msg_eof[:90]))
        ok_v3_addr = all(ok for _, ok, _ in v3_addr_rows)
        checks.append(('v3-⑧ 주소 네거티브 %d종(source_index 교환·abs_offset ±1/±A·내부정합 위조·slice/type/dims·'
                       'witness·digest·align·bin 필드·raw JSON 봉인·checked uint64·bracket EOF·'
                       'source identity[path·mtime·drive_root]) 전부 중단'
                       % len(v3_addr_rows), ok_v3_addr))
        print('[selftest] v3-⑧ 주소: %s' % ('PASS' if ok_v3_addr else 'FAIL'))
        for label, ok_n, note in v3_addr_rows:
            print('    [%s] %-52s %s' % ('PASS' if ok_n else 'FAIL', label, note))

        # ---- v3-⑨ 슬롯 네거티브(중복·비정렬·역순 slot_offset·과소 layer_slot_bytes/slot_stride_max·오기 head) ----
        def _mut_slot_dup(mf):
            ps = mf['layout']['layers'][0]['parts']
            ps[1]['slot_offset'] = ps[0]['slot_offset']

        def _mut_slot_misalign(mf):
            ps = mf['layout']['layers'][0]['parts']
            ps[1]['slot_offset'] += 1

        def _mut_slot_reverse(mf):
            ps = mf['layout']['layers'][0]['parts']
            ps[0]['slot_offset'], ps[1]['slot_offset'] = ps[1]['slot_offset'], ps[0]['slot_offset']

        def _mut_slot_overlap(mf):
            """겹침 위조: 두 번째 파트 slot_offset 을 A 만큼 앞으로(점화식 비중첩 위반)."""
            ps = mf['layout']['layers'][0]['parts']
            ps[1]['slot_offset'] -= A_v3
            if ps[1]['aligned']:
                ps[1]['data_offset'] = ps[1]['slot_offset'] + ps[1]['bracket_head']
            else:
                ps[1]['data_offset'] = ps[1]['slot_offset']

        v3_slot_cases = [
            ('중복 slot_offset', 'slot_dup', _mut_slot_dup),
            ('비정렬 slot_offset(+1)', 'slot_misalign', _mut_slot_misalign),
            ('역순 slot_offset', 'slot_rev', _mut_slot_reverse),
            ('겹침 slot_offset(-A)', 'slot_overlap', _mut_slot_overlap),
            ('과소 layer_slot_bytes(-A)', 'lsb',
             lambda mf: mf['layout']['layers'][0].__setitem__('layer_slot_bytes',
                                                               mf['layout']['layers'][0]['layer_slot_bytes'] - A_v3)),
            ('과소 slot_stride_max(-A)', 'ssm',
             lambda mf: mf['layout'].__setitem__('slot_stride_max', mf['layout']['slot_stride_max'] - A_v3)),
            ('오기 bracket_head(+1)', 'head',
             lambda mf: _v3_first_part(mf).__setitem__('bracket_head', _v3_first_part(mf)['bracket_head'] + 1)),
            ('aligned part 의 bracket_head 삭제', 'head_del',
             lambda mf: _v3_first_part(mf).pop('bracket_head')),
            ('aligned 플래그 뒤집기', 'aligned_flip',
             lambda mf: _v3_first_part(mf).__setitem__('aligned', False)),
            ('data_offset 오기(slot_offset 무시)', 'do',
             lambda mf: _v3_first_part(mf).__setitem__('data_offset', 0)),
            ('layer_slot_bytes 실수화(int→float 우회)', 'lsb_float',
             lambda mf: mf['layout']['layers'][0].__setitem__('layer_slot_bytes',
                                                               float(mf['layout']['layers'][0]['layer_slot_bytes']))),
        ]
        v3_slot_rows = []
        for label, tag, fn in v3_slot_cases:
            ok_n, rep_n = _v3_neg(tag, fn)
            v3_slot_rows.append((label, ok_n, (rep_n['problems'] or [''])[0][:90]))
        # 비4K 산출물에서 bracket_head 를 부착하는 위조(비4K 는 기재 금지)
        ok_head_extra, rep_head_extra = _v3_neg(
            'head_on_non4k', lambda mf: _v3_first_part(mf).__setitem__('bracket_head', 0),
            src_dir=v3_out_non, model=v3_non)
        v3_slot_rows.append(('비4K part 에 bracket_head 부착', ok_head_extra,
                             (rep_head_extra['problems'] or [''])[0][:90]))
        ok_staging, rep_staging = _v3_neg(
            'staging_small',
            lambda mf: _v3_first_part(mf).__setitem__('staging_bytes', _v3_first_part(mf)['staging_bytes'] - A_v3),
            src_dir=v3_out_non, model=v3_non)
        v3_slot_rows.append(('비4K part 의 과소 staging_bytes', ok_staging,
                             (rep_staging['problems'] or [''])[0][:90]))
        ok_v3_slot = all(ok for _, ok, _ in v3_slot_rows)
        checks.append(('v3-⑨ 슬롯 네거티브 %d종(중복/비정렬/역순/겹침 slot_offset·과소 layer_slot_bytes·'
                       '과소 slot_stride_max·오기/삭제 bracket_head·aligned 뒤집기·data_offset 오기·'
                       '비4K head 부착·과소 staging) 전부 중단' % len(v3_slot_rows), ok_v3_slot))
        print('[selftest] v3-⑨ 슬롯: %s' % ('PASS' if ok_v3_slot else 'FAIL'))
        for label, ok_n, note in v3_slot_rows:
            print('    [%s] %-46s %s' % ('PASS' if ok_n else 'FAIL', label, note))

        # ---- v3-⑩ verifier 독립성 회귀(§4-3): producer 가 만든 잘못된 manifest 를 실제로 잡는가 ----
        # 양성=무변조 산출물 재검증 PASS(디스크 재로드 경로) · 음성=①producer 산술을 오염시켜 실제로
        # 잘못된 manifest 를 생산 → 승격 게이트가 차단(manifest.json 미승격·.partial 잔존)
        # ②승격 후 디스크 바이트만 손대도 재검증이 FAIL(=in-memory 재사용이 아니라 파일을 읽는다)
        v3_indep_rows = []
        rep_pos = verify_virtual_manifest(v3_mixed, v3_out, profile_id=None, enforce_reference=False,
                                          allow_default_align=True)
        v3_indep_rows.append(('양성: 무변조 산출물 재검증 PASS', rep_pos['pass'] is True,
                              str(rep_pos['problems'])[:80]))
        _orig_cvl = compute_virtual_layout

        def _poisoned_cvl(layout, A, source_bytes):
            v = _orig_cvl(layout, A, source_bytes)
            v['layers'][0]['parts'][-1]['slot_offset'] += A      # 슬롯 오기 주입(생산 측 결함 재현)
            return v

        v3_poison_out = os.path.join(scratch, 'out_v3_poison')
        globals()['compute_virtual_layout'] = _poisoned_cvl
        try:
            ok_poison, msg_poison = _expect_abort(
                lambda: do_virtual_plan(v3_mixed, v3_poison_out, profile_id=None, force=False,
                                        enforce_reference=False, allow_default_align=True))
        finally:
            globals()['compute_virtual_layout'] = _orig_cvl
        ok_poison = (ok_poison and 'verifier rejected' in msg_poison
                     and not os.path.exists(os.path.join(v3_poison_out, MANIFEST_FILENAME))
                     and not os.path.exists(os.path.join(v3_poison_out, PLAN_REPORT_FILENAME))
                     and os.path.exists(os.path.join(v3_poison_out, MANIFEST_FILENAME + '.partial')))
        v3_indep_rows.append(('음성①: producer 산술 오염 → verifier 가 차단·미승격(.partial 잔존)',
                              ok_poison, msg_poison[:90]))
        v3_byte_dir = os.path.join(scratch, 'out_v3_bytes')
        shutil.copytree(v3_out, v3_byte_dir)
        _bp = os.path.join(v3_byte_dir, MANIFEST_FILENAME)
        _btxt = open(_bp, 'r', encoding='utf-8').read()
        _anchor = '"slot_stride_max": %d' % v3_mf['layout']['slot_stride_max']
        open(_bp, 'w', encoding='utf-8').write(
            _btxt.replace(_anchor, '"slot_stride_max": %d' % (v3_mf['layout']['slot_stride_max'] + A_v3), 1))
        rep_bytes = verify_virtual_manifest(v3_mixed, v3_byte_dir, profile_id=None, enforce_reference=False,
                                            allow_default_align=True)
        v3_indep_rows.append(('음성②: 승격 후 디스크 바이트만 변조 → 재검증 FAIL(파일 재로드 증거)',
                              rep_bytes['pass'] is False, (rep_bytes['problems'] or [''])[0][:90]))

        # ★부속 정오 2 §Z-② 회귀 — "verifier 가 플래너와 **다른 코드**로 좌표를 다시 내는가".
        # 아래 3 변이는 공통 함수(=생산과 검증이 함께 쓰던 함수)가 좌표를 잘못 내는 상황을 만든다.
        # 셋 다 manifest 안에서는 **완전히 자기일관**이라 invariants 로는 잡히지 않는다 — 그
        # 사실을 잔존 .partial 로 매번 직접 증명한다(invariants 문제 0건). 그럼에도 verifier 가
        # 거부하면, 거부한 주체는 검증 전용 재유도뿐이다. 음성① 과 대조된다: 그건 invariants 가
        # 잡던 종류이고, 이 셋은 구 코드가 **구조적으로 못 잡던** 종류다.
        def _indep_negative(tag, patch_name, make_patch, want_field, model=None, scope='all'):
            # ★거부 사유는 **verifier 의 problems 목록**에서 읽는다. abort 메시지는 고정 길이로
            # 잘려 필드명까지 닿지 않으므로(경로 길이에 따라 절단 위치가 달라진다) 거기서 찾으면
            # 검사가 조용히 무효가 된다. 오염을 건 채로 verifier 를 직접 한 번 더 호출해 전문을 받는다.
            # (재호출 시 scope 는 manifest.model.routed_scope 로 승계되므로 인자로 주지 않는다.)
            mdl = model or v3_mixed
            pout = os.path.join(scratch, 'out_v3_indep_%s' % tag)
            partial = os.path.join(pout, MANIFEST_FILENAME + '.partial')
            _orig_fn = globals()[patch_name]
            globals()[patch_name] = make_patch(_orig_fn)
            rep = {'problems': []}
            try:
                ok_ab, msg_ab = _expect_abort(
                    lambda: do_virtual_plan(mdl, pout, profile_id=None, force=False,
                                            enforce_reference=False, allow_default_align=True,
                                            scope=scope))
                if os.path.exists(partial):
                    rep = verify_virtual_manifest(mdl, pout, profile_id=None,
                                                  enforce_reference=False, allow_default_align=True,
                                                  manifest_name=MANIFEST_FILENAME + '.partial')
            finally:
                globals()[patch_name] = _orig_fn
            probs = '; '.join(str(x) for x in (rep.get('problems') or []))
            inv = []
            if os.path.exists(partial):
                _check_virtual_manifest_invariants(
                    json.loads(open(partial, 'r', encoding='utf-8').read()), inv)
            ok_n = (ok_ab and 'verifier rejected' in msg_ab and want_field in probs
                    and rep.get('pass') is False
                    and os.path.exists(partial) and not inv
                    and not os.path.exists(os.path.join(pout, MANIFEST_FILENAME)))
            return ok_n, 'invariants=%d(0이어야 독립 재유도가 잡은 것) %s' % (len(inv), probs[:72])

        def _mk_shard_shift(orig):
            """공통 shard 파서가 좌표 원점을 통째로 잘못 잡은 상황(EOF 여유도 같이 밀어 생산은 통과)."""
            def wrapped(model_path):
                m = orig(model_path)
                for h in m['shards']:
                    h['file_bytes'] += A_v3
                    for t in h['tensors']:
                        t['abs_offset'] += A_v3
                return m
            return wrapped

        def _mk_part_swap(orig):
            """공통 layout 이 레코드 내부 순서를 잘못 낸 상황(전 층 동일하게 뒤바뀌어 자기일관)."""
            def wrapped(model, scope='all'):
                lay = orig(model, scope=scope)
                for L in lay['layers']:
                    L['parts'][0], L['parts'][1] = L['parts'][1], L['parts'][0]
                    off = 0
                    for p in L['parts']:
                        p['part_offset'] = off
                        off += p['part_bytes']
                return lay
            return wrapped

        def _mk_absoff_drift(orig):
            """공통 슬롯 산술이 한 파트의 abs_offset 만 A 만큼 잘못 낸 상황.
            bracket_head=abs_offset%A 는 불변이고 records 도 같이 밀어 두므로 자기일관 검사는 전부 통과한다."""
            def wrapped(layout, A, source_bytes):
                v = orig(layout, A, source_bytes)
                vp = v['layers'][0]['parts'][0]
                lid, nm = v['layers'][0]['layer'], vp['name']
                vp['abs_offset'] += A
                for rec in v['records']:
                    if rec['layer'] == lid and rec['part'] == nm:
                        rec['src_offset'] += A
                return v
            return wrapped

        _n3 = _indep_negative('shardshift', 'load_model_shards', _mk_shard_shift,
                              'two-implementation disagreement')
        v3_indep_rows.append(('음성③: 공통 파서가 shard 좌표 원점을 오도출 → 두 구현 합치 검사가 지목', _n3[0], _n3[1]))
        _n4 = _indep_negative('partswap', 'build_layout', _mk_part_swap, 'parts[0].name')
        v3_indep_rows.append(('음성④: 공통 layout 이 파트 순서를 오도출 → 전항 대조가 지목', _n4[0], _n4[1]))
        _n5 = _indep_negative('absoff', 'compute_virtual_layout', _mk_absoff_drift, 'parts[0].abs_offset')
        v3_indep_rows.append(('음성⑤: 자기일관을 유지한 abs_offset 오도출 → 전항 대조가 지목', _n5[0], _n5[1]))

        # ★r7 N2(채택분): 결함류 대표성 확대 — slice 산술·shard 순서.
        def _mk_slice_shrink(orig):
            """공통 layout 이 per-expert slice 를 잘못 낸 상황(payload 도 같이 줄여 자기일관 유지)."""
            def wrapped(mdl, scope='all'):
                lay = orig(mdl, scope=scope)
                L = lay['layers'][0]
                p = L['parts'][0]
                delta = 128 if p['part_bytes'] > 128 else 1
                p['part_bytes'] -= delta
                p['theory_bytes'] = p['part_bytes'] * lay['n_expert']
                L['payload_bytes'] -= delta
                off = 0
                for q in L['parts']:
                    q['part_offset'] = off
                    off += q['part_bytes']
                return lay
            return wrapped

        def _mk_shard_order(orig):
            """공통 shard 발견이 형제 순서를 뒤집은 상황(source_index 부여가 통째로 어긋난다)."""
            def wrapped(model_path):
                paths = orig(model_path)
                return list(reversed(paths)) if len(paths) > 1 else paths
            return wrapped

        _n6 = _indep_negative('sliceshrink', 'build_layout', _mk_slice_shrink, 'slice_bytes')
        v3_indep_rows.append(('음성⑥: 공통 layout 의 slice 산술 오도출 → 두 구현 합치 검사가 지목', _n6[0], _n6[1]))
        # ★shard 순서 오도출은 **각 shard 에 canonical `split.no` 가 있으면 공통 경로가 스스로
        # 잡는다**(`load_model_shards` 의 `split.no != source_index` 검사 — 실측: 그 경우
        # verifier 까지 가지도 못하고 "split.no(1) != source_index(0)" 로 생산이 먼저 중단된다).
        # ★r8 N4 정정: 조건은 "split KV 가 있으면"이 아니라 **"각 shard 에 canonical `split.no`
        # 가 있으면"** 이다 — 그 검사는 키가 존재할 때만 실행되므로, 넓게 적으면 키 일부만 빠진
        # 형상까지 방어되는 것처럼 읽힌다. 즉 그 형상은 §Z-② 결함류가 아니다. 독립 재유도가
        # **실제로 유일한 방어가 되는** 형상은 그 키가 없는 모델이므로, 네거티브는 그쪽으로
        # 세운다(구조 KV 부재 픽스처).
        v3_nokv1 = os.path.join(scratch, 'v3nokv-00001-of-00002.gguf')
        v3_nokv2 = os.path.join(scratch, 'v3nokv-00002-of-00002.gguf')
        write_synthetic_gguf([v3_nokv1, v3_nokv2], n_expert=4, moe_layers=(0, 1), schema='separate',
                             bias=True, hidden=v3_hid, alignment=32, seed=4107,
                             shard_of={'blk.1.ffn_down_exps.weight': 1, V3_FILLER2['name']: 1},
                             extra_tensors=(V3_FILLER, V3_FILLER2),
                             omit_kv=('split.no', 'split.count', 'split.tensors.count'))
        _n7 = _indep_negative('shardorder', 'discover_shard_paths', _mk_shard_order,
                              'two-implementation disagreement', model=v3_nokv1)
        v3_indep_rows.append(('음성⑦: canonical `split.no` 가 없는 모델에서 형제 순서 오도출 → 합치 '
                              '검사가 지목(각 shard 에 canonical `split.no` 가 있으면 공통 경로가 '
                              '먼저 잡는다 — 그 형상은 결함류 아님)',
                              _n7[0], _n7[1]))

        # ★r8 N1(= r7 N2 의 미이행분): 결함류 대표성의 남은 두 축.
        #   ⓐ**execution scope 선별** — 지금까지의 execution 회귀는 "양성"과 "manifest 에서
        #     routed_scope 키를 지우는 손편집"뿐이었다. 둘 다 **공통 경로가 제외 층을 잘못 고르는**
        #     상황은 행사하지 않는다. 공통 경로는 `_read_nextn_predict_layers()` 를 거치고 검증
        #     전용 경로는 meta 를 직접 읽으므로, 그 공통 질의만 오염시키면 두 선별이 갈린다.
        #   ⓑ**텐서 0개 shard 의 data_start** — upstream 은 n_tensors==0 이면 데이터 정렬 seek 을
        #     건너뛴다(gguf.cpp:756). 기존 회귀는 bin 경로 양성뿐이라, 공통 파서가 그 규칙을 어겨도
        #     virtual 독립 재유도가 잡는다는 것은 미증명이었다. 두 경우 모두 manifest 는 완전히
        #     자기일관이라 invariants 로는 0건이다(그 사실을 매 건 증명한다).
        def _mk_nextn_shift(orig):
            """공통 execution-scope 질의가 제외 층 수를 잘못 낸 상황(1 → 2: 층 하나를 더 버린다)."""
            def wrapped(model, arch):
                v = orig(model, arch)
                return (v + 1) if v else v
            return wrapped

        _n8 = _indep_negative('execscope', '_read_nextn_predict_layers', _mk_nextn_shift,
                              'model.moe_layers', model=v3_exec, scope='execution')
        v3_indep_rows.append(('음성⑧: 공통 execution-scope 선별이 제외 층을 오도출 → 합치 검사가 지목'
                              '(manifest 는 줄어든 층 집합으로 완전히 자기일관)', _n8[0], _n8[1]))

        # 텐서 0개 첫 shard(메타 전용) + routed 전량 + filler 는 두 번째 shard 에.
        v3_zt1 = os.path.join(scratch, 'v3zt-00001-of-00002.gguf')
        v3_zt2 = os.path.join(scratch, 'v3zt-00002-of-00002.gguf')
        _zt_names = ['blk.%d.ffn_%s_exps.%s' % (l, k, s)
                     for l in (0, 1) for s in ('weight', 'bias') for k in ('gate', 'up', 'down')]
        write_synthetic_gguf([v3_zt1, v3_zt2], n_expert=4, moe_layers=(0, 1), schema='separate',
                             bias=True, hidden=v3_hid, alignment=32, seed=4108,
                             extra_tensors=(V3_FILLER,),
                             shard_of=dict([(n, 1) for n in _zt_names] + [(V3_FILLER['name'], 1)]))

        def _mk_zero_tensor_pad(orig):
            """공통 파서가 텐서 0개 shard 에도 무조건 정렬 패딩을 적용한 상황(구 산식 재현).
            그 shard 엔 routed 텐서가 없어 주소는 하나도 안 변한다 — manifest 는 자기일관이고
            어긋나는 것은 `sources[i].data_start` 하나뿐이다."""
            def wrapped(path):
                h = orig(path)
                if not h['tensors']:
                    padded = _ceil_to(h['data_start'], h['alignment'])
                    h['data_start'] = padded
                    h['data_region'] = h['file_bytes'] - padded
                return h
            return wrapped

        _n9 = _indep_negative('zerotensor', 'parse_gguf_header', _mk_zero_tensor_pad,
                              'sources[0].data_start', model=v3_zt1)
        v3_indep_rows.append(('음성⑨: 텐서 0개 shard 의 data_start 를 공통 파서가 오도출(구 무조건 패딩) '
                              '→ 합치 검사가 지목(주소는 불변이라 invariants 로는 잡히지 않는 형상)',
                              _n9[0], _n9[1]))

        ok_v3_indep = all(ok for _, ok, _ in v3_indep_rows)
        checks.append(('v3-⑩ verifier 독립성 회귀(§4-3 축소 계약 · 부속 정오 2 §Z-②): 양성 1건 + '
                       '음성 9건(producer 산술 오염 차단·디스크 바이트 변조 적발 + 공통 함수 오도출 '
                       '7종[shard 좌표·파트 순서·abs_offset·slice 산술·shard 순서·execution scope 선별·'
                       '텐서 0개 shard data_start]을 검증 전용 재유도가 적발 — 자기일관 7종은 '
                       'invariants 통과를 매 건 증명)', ok_v3_indep))
        print('[selftest] v3-⑩ verifier 독립성: %s' % ('PASS' if ok_v3_indep else 'FAIL'))
        for label, ok_n, note in v3_indep_rows:
            print('    [%s] %-52s %s' % ('PASS' if ok_n else 'FAIL', label, note))

        # ---- v3-⑬ 선언된 공통 신뢰원(정렬 질의)의 경계 고정 — r7 M2 / 리드 판정 분기①=ⓑ ----
        # ⓑ 를 택했으므로 A 는 **독립 재유도하지 않는다**. 그러면 "안 잡힌다"가 조용한 가정으로
        # 남으므로, 경계를 실행 가능한 형태로 못박는다:
        #   ⓐ**잘못된 A 와 그로부터 나온 잘못된 슬롯·EOF 가 생산 시점에 그대로 승격된다**
        #   ⓑ그 값이 provenance 로 manifest 에 남아 **재질의 대조가 소비 시점에 잡는다**
        # ⓑ가 이 신뢰 가정의 완화 장치이고, 그래서 가정이 은폐가 아니라 **회수 가능한 계약**이 된다.
        #
        # ★r8 M3 재설계(과장 자인): 구판은 `logical=1` 로 **A 가 바뀌지 않는** 오염만 넣었다.
        #   A=max(4096,logical,physical) 가 그대로였으니 슬롯도 EOF 도 그대로였고, 결국 세우는
        #   명제가 "주소에 영향 없는 provenance 오염은 통과한다"로 쪼그라들어 **정작 증명해야 할
        #   ⓐ를 한 번도 행사하지 않았다**. 이제 physical 을 유효한 2배 power-of-two 로 올려
        #   **A 자체를 바꾸고**, 승격된 manifest 의 주소 산출물이 clean 기준과 **실제로 달라졌음**을
        #   대조한다. 픽스처 EOF 여유는 비-routed filler(131,072B ≥ 2A)가 대므로 A 가 커져도
        #   bracket EOF 가 아니라 신뢰원 축만 행사된다.
        v3_trust = []
        _orig_q = query_sector_alignment_for_path
        A_poison = A_v3 * 2

        def _q_poisoned_A(path):
            """질의가 물리 섹터를 과대 보고하는 상황(= 볼륨 오해석). A 가 실제로 커진다."""
            info = dict(_orig_q(path))
            info['physical'] = A_poison
            return info

        def _q_poisoned_provenance(path):
            """A 는 그대로 두고 provenance 만 어긋나는 상황(별개 축 — 아래 3행)."""
            info = dict(_orig_q(path))
            info['logical'] = 1
            return info

        def _v3_addr_fingerprint(mf):
            """주소 산출물만 뽑은 지문. '두 산출물이 A 만 다르고 주소는 같다'면 이 검사는 무효이므로,
            무효 여부를 값으로 판별할 수 있게 한다."""
            lay = mf['layout']
            parts = [(p.get('aligned'), p.get('bracket_head'), p.get('slot_offset'),
                      p.get('data_offset'), p.get('staging_bytes'))
                     for L in lay['layers'] for p in L['parts']]
            return (lay['slot_stride_max'], [L['layer_slot_bytes'] for L in lay['layers']], parts,
                    [r['data_offset'] for r in mf['records']])

        # ★r9 M3(과장 자인 2회차): 위 지문 **하나로만** 비교하던 구판은 "네 축 중 한 곳만 달라도
        #   통과"라, 정오 ⓙ 가 주장하는 개별 회수(슬롯·bracket EOF·records)를 **하나도 세우지
        #   못했다**. 축을 겹치지 않게 갈라 각각을 독립 명제로 만든다 — r8 M3 와 같은 종류의
        #   지적(검사가 명제보다 약하다)이므로 같은 방식으로 갚는다.
        def _v3_slot_fingerprint(mf):
            """슬롯 축만: A 가 점화식(region=align_up(head+slice,A))을 통해 만드는 주소."""
            lay = mf['layout']
            return (lay['slot_stride_max'],
                    [(L['layer'], L['layer_slot_bytes']) for L in lay['layers']],
                    [(L['layer'], p['name'], p.get('aligned'), p.get('bracket_head'),
                      p['slot_offset'], p['data_offset'])
                     for L in lay['layers'] for p in L['parts']])

        def _v3_bracket_eofs(mf):
            """bracket EOF 축만. manifest 엔 `bracket_end` 필드가 **없으므로** §2-4 경계식
            align_up(abs_offset + n_expert*slice_bytes, A) 를 여기서 직접 계산한다(저장값 재사용
            아님). 각 항에 그 source 의 EOF 를 함께 실어, '경계 안에 있는가'를 값으로 판별한다."""
            A = mf['layout']['align_bytes']
            ne = mf['model']['n_expert']
            eof = dict((s['index'], s['bytes']) for s in mf['sources'])
            out = []
            for L in mf['layout']['layers']:
                for p in L['parts']:
                    end = p['abs_offset'] + ne * p['slice_bytes']
                    out.append((L['layer'], p['name'], -(-end // A) * A, eof[p['source_index']]))
            return out

        def _v3_record_offsets(mf):
            """records 축만(비권위 witness 가 잘못된 A 를 그대로 물고 승격되는가)."""
            return [(r['layer'], r['expert'], r['part'], r['data_offset']) for r in mf['records']]

        v3_trust_out = os.path.join(scratch, 'out_v3_trust_align')
        globals()['query_sector_alignment_for_path'] = _q_poisoned_A
        try:
            _, rep_trust = do_virtual_plan(v3_mixed, v3_trust_out, profile_id=None, force=False,
                                           enforce_reference=False, allow_default_align=True)
        finally:
            globals()['query_sector_alignment_for_path'] = _orig_q
        _mf_tr = _load_manifest_disk(v3_trust_out)
        _aq0 = ((_mf_tr.get('layout') or {}).get('align_query') or [{}])[0]
        _addr_clean = _v3_addr_fingerprint(v3_mf)      # 아래 '별개 축'의 주소 불변 대조가 쓴다
        _slot_clean, _slot_pois = _v3_slot_fingerprint(v3_mf), _v3_slot_fingerprint(_mf_tr)
        _beof_clean, _beof_pois = _v3_bracket_eofs(v3_mf), _v3_bracket_eofs(_mf_tr)
        _rec_clean, _rec_pois = _v3_record_offsets(v3_mf), _v3_record_offsets(_mf_tr)

        # ⓐ-0 승격 자체: 잘못된 A 가 생산 게이트를 그대로 통과한다(선언된 신뢰원의 본체).
        ok_t1 = (rep_trust.get('pass') is True
                 and _mf_tr['layout']['align_bytes'] == A_poison and A_poison != A_v3
                 and _aq0.get('physical') == A_poison)
        v3_trust.append(('경계ⓐ-0: **잘못된 A 가 생산 시점 verifier 를 통과해 승격된다**'
                         '(선언된 신뢰원 — 여기서는 검증이 잡지 않는다)',
                         ok_t1, 'A: clean=%r poisoned=%r' % (A_v3, _mf_tr['layout']['align_bytes'])))

        # ⓐ-1 슬롯 축: 승격된 산출물의 슬롯 주소가 clean 기준과 **실제로** 달라졌다.
        ok_t1_slot = (_slot_pois != _slot_clean
                      and _slot_pois[0] != _slot_clean[0])   # slot_stride_max 자체가 움직였다
        v3_trust.append(('경계ⓐ-1: 잘못된 A 가 만든 **슬롯 주소**(slot_stride_max·layer_slot_bytes·'
                         'bracket_head·slot_offset·data_offset)가 clean 과 상이한 채 승격',
                         ok_t1_slot, 'slot_stride_max: clean=%r poisoned=%r'
                         % (_slot_clean[0], _slot_pois[0])))

        # ⓐ-2 bracket EOF 축: 경계값이 **직접 계산으로** 달라졌고, 그럼에도 양쪽 다 source EOF
        #     안에 있다 = 이 픽스처가 행사하는 것은 §2-4 경계 검사가 아니라 신뢰원 축 하나뿐이다
        #     (경계에 걸렸다면 VirtualBinRegression 으로 승격 자체가 없었을 테니, 이 조건이 없으면
        #      ⓐ-1·ⓐ-3 이 무엇 때문에 달라졌는지 구별되지 않는다).
        _beof_diff = [(c, p) for c, p in zip(_beof_clean, _beof_pois) if c[2] != p[2]]
        _beof_inbound = all(x[2] <= x[3] for x in _beof_clean + _beof_pois)
        ok_t1_beof = (len(_beof_clean) == len(_beof_pois) and bool(_beof_diff) and _beof_inbound)
        v3_trust.append(('경계ⓐ-2: **bracket EOF 직접 계산**(align_up(abs+n_expert*slice, A))이 상이 · '
                         '양쪽 모두 source EOF 이내(= 경계 검사가 아니라 신뢰원 축만 행사됨)',
                         ok_t1_beof, '상이 %d/%d 파트 · 전항 EOF 이내=%s'
                         % (len(_beof_diff), len(_beof_clean), _beof_inbound)))

        # ⓐ-3 records 축: 비권위 witness 도 잘못된 주소를 그대로 물고 승격된다.
        ok_t1_rec = (len(_rec_pois) == len(_rec_clean) and _rec_pois != _rec_clean)
        v3_trust.append(('경계ⓐ-3: **records[].data_offset** 이 clean 과 상이한 채 승격'
                         '(witness 도 잘못된 A 를 물고 나간다)', ok_t1_rec,
                         'records %d건 · 상이 %d건'
                         % (len(_rec_clean), sum(1 for c, p in zip(_rec_clean, _rec_pois) if c != p))))

        _rep_rq = verify_virtual_manifest(v3_mixed, v3_trust_out, profile_id=None,
                                          enforce_reference=False, allow_default_align=True)
        _probs_rq = '; '.join(str(x) for x in (_rep_rq.get('problems') or []))
        ok_t2 = (_rep_rq.get('pass') is False
                 and 'layout.align_bytes' in _probs_rq                  # A 자체
                 and 'align_query[0].physical' in _probs_rq             # 질의 provenance
                 and 'layout.slot_stride_max' in _probs_rq)             # 주소 산출물(전역)
        v3_trust.append(('완화ⓑ-1: 오염 없이 재검증하면 **A·질의 provenance·slot_stride_max 가 함께 '
                         '적발**(소비 시점 회수 경로)', ok_t2, _probs_rq[:110]))

        # ⓑ-2 회수의 **주소 단위**: 전역 값(slot_stride_max) 하나만 걸리는 것으로는 "파트·레코드
        #     단위 주소가 회수된다"는 정오 문면이 서지 않는다. 두 주소 계열이 problems 에
        #     **각각** 이름을 올리는지 본다(구판은 `.parts[0].` 부분문자열 하나로 뭉뚱그렸다 —
        #     그건 `aligned` 같은 비주소 필드에도 걸린다).
        _hit_part_addr = re.search(r'layout\.layers\[\d+\]\.parts\[\d+\]\.(slot_offset|data_offset)\(',
                                   _probs_rq)
        _hit_rec_addr = re.search(r'records\[\d+\] mismatch', _probs_rq)
        ok_t2b = bool(_hit_part_addr) and bool(_hit_rec_addr)
        v3_trust.append(('완화ⓑ-2: 재검증 problems 에 **파트 주소**(parts[].slot_offset/data_offset)와 '
                         '**레코드 주소**(records[] mismatch)가 각각 이름을 올린다',
                         ok_t2b, 'part=%r · record=%r'
                         % (_hit_part_addr.group(0) if _hit_part_addr else None,
                            _hit_rec_addr.group(0) if _hit_rec_addr else None)))

        # ⓑ-3(★r10 N4): 위 `records[\d+] mismatch` 는 **일반 진단**이라 어느 필드가 어긋나도
        #     걸린다 — Codex r10 Q2 가 "완전 직교는 아니다"로 남긴 잔여다. 레코드 진단이
        #     **`records[].data_offset` 때문에** 떴음을 값으로 고정한다. v3 레코드 메시지는 7필드
        #     리스트(layer·expert·part·source_index·src_offset·slice_bytes·**data_offset**)를 통째로
        #     싣고 data_offset 이 **마지막 항**이므로, 앞 6항이 문자열까지 동일한 채 마지막 항만
        #     clean↔오염으로 갈리는지 본다. A 는 source 좌표(src_offset·slice_bytes)를 바꾸지 않으니
        #     이 직교성은 우연이 아니라 계약의 귀결이고, 깨지면 그 자체가 적발 대상이다.
        _rec_i = next((i for i, (c, p) in enumerate(zip(_rec_clean, _rec_pois)) if c != p), None)
        _m_rec = (re.search(r'records\[%d\] mismatch: actual=\[([^\]]*)\] expected=\[([^\]]*)\]' % _rec_i,
                            _probs_rq) if _rec_i is not None else None)
        ok_t2c = False
        _rec_note = 'clean↔오염 records 차이 없음' if _rec_i is None else 'records[%r] 진단 문자열 미검출' % _rec_i
        if _m_rec:
            _a_head, _, _a_last = _m_rec.group(1).rpartition(',')
            _e_head, _, _e_last = _m_rec.group(2).rpartition(',')
            ok_t2c = (_a_head == _e_head                            # 앞 6필드 동일 = data_offset 단독 원인
                      and _a_last.strip() == str(_rec_pois[_rec_i][3])
                      and _e_last.strip() == str(_rec_clean[_rec_i][3]))
            _rec_note = ('records[%d].data_offset expected=%s actual=%s · 앞 6필드 동일=%s'
                         % (_rec_i, _e_last.strip(), _a_last.strip(), _a_head == _e_head))
        v3_trust.append(('완화ⓑ-3(N4): 레코드 진단이 **`records[].data_offset` 단독**으로 뜬다 — '
                         '앞 6필드는 문자열까지 동일하고 마지막 항만 clean↔오염 값으로 갈린다',
                         ok_t2c, _rec_note))

        # 별개 축: A 가 바뀌지 않는 provenance 전용 오염도 재질의 대조가 잡는다(완화 장치가 A
        # 비교 하나에 의존하지 않는다는 증거 — 위 행들과 겹치지 않는 명제다).
        v3_trust_out2 = os.path.join(scratch, 'out_v3_trust_prov')
        globals()['query_sector_alignment_for_path'] = _q_poisoned_provenance
        try:
            _, rep_trust2 = do_virtual_plan(v3_mixed, v3_trust_out2, profile_id=None, force=False,
                                            enforce_reference=False, allow_default_align=True)
        finally:
            globals()['query_sector_alignment_for_path'] = _orig_q
        _mf_tr2 = _load_manifest_disk(v3_trust_out2)
        _rep_rq2 = verify_virtual_manifest(v3_mixed, v3_trust_out2, profile_id=None,
                                           enforce_reference=False, allow_default_align=True)
        _probs_rq2 = '; '.join(str(x) for x in (_rep_rq2.get('problems') or []))
        ok_t3 = (rep_trust2.get('pass') is True
                 and _mf_tr2['layout']['align_bytes'] == A_v3            # A 는 불변
                 and _v3_addr_fingerprint(_mf_tr2) == _addr_clean        # 주소도 불변
                 and _rep_rq2.get('pass') is False and 'align_query[0].logical' in _probs_rq2)
        v3_trust.append(('별개 축: A 가 바뀌지 않는 provenance 전용 오염(주소 불변)도 재질의 대조가 적발',
                         ok_t3, _probs_rq2[:80]))

        # ★r10 N4 — **분류 보존 축**: 위 ⓐ 픽스처의 A 4096→8192 는 weight slice(= 정확히 A)를
        #   비4K 로 **뒤집는다**. 그래서 "주소가 움직였다"를 *aligned 분류 전환의 부수효과*로 읽을
        #   여지가 남았다(Codex r10 Q2 잔여: "분류가 유지되는 A 변화에서도 주소가 움직인다까지
        #   증명하지는 않는다"). 분류가 **전건 보존**되는 A 변화를 따로 세운다 — weight slice 를
        #   2A 의 배수로 만드는 hidden 을 골라, A 를 2배로 올려도 `slice % A == 0` 이 유지되게 한다
        #   (bias slice = 4*hidden 은 양쪽 A 에서 모두 비4K 라 픽스처는 혼재를 유지한다).
        v3_hid_sp = 32
        while (v3_hid_sp * v3_hid_sp * 4) % (A_poison) != 0:
            v3_hid_sp *= 2
        v3_sp = os.path.join(scratch, 'v3_shape_preserving.gguf')
        write_synthetic_gguf([v3_sp], n_expert=4, moe_layers=(0, 1), schema='separate', bias=True,
                             hidden=v3_hid_sp, alignment=32, seed=4123, extra_tensors=(V3_FILLER,))
        _sp_clean_out = os.path.join(scratch, 'out_v3_sp_clean')
        _, _rep_sp_c = do_virtual_plan(v3_sp, _sp_clean_out, profile_id=None, force=False,
                                       enforce_reference=False, allow_default_align=True)
        _sp_pois_out = os.path.join(scratch, 'out_v3_sp_pois')
        globals()['query_sector_alignment_for_path'] = _q_poisoned_A
        try:
            _, _rep_sp_p = do_virtual_plan(v3_sp, _sp_pois_out, profile_id=None, force=False,
                                           enforce_reference=False, allow_default_align=True)
        finally:
            globals()['query_sector_alignment_for_path'] = _orig_q
        _mf_sp_c, _mf_sp_p = _load_manifest_disk(_sp_clean_out), _load_manifest_disk(_sp_pois_out)

        def _v3_align_shape(mf):
            """aligned 분류만 뽑은 형상(주소는 보지 않는다)."""
            return [(L['layer'], p['name'], p['aligned'])
                    for L in mf['layout']['layers'] for p in L['parts']]

        _shape_c, _shape_p = _v3_align_shape(_mf_sp_c), _v3_align_shape(_mf_sp_p)
        # 혼재 확인이 없으면 "전부 aligned 인 픽스처에서 분류가 보존됐다"는 공허한 명제가 된다.
        _sp_mixed = any(x[2] for x in _shape_c) and any(not x[2] for x in _shape_c)
        ok_t4 = (_rep_sp_c.get('pass') is True and _rep_sp_p.get('pass') is True
                 and _mf_sp_c['layout']['align_bytes'] == A_v3
                 and _mf_sp_p['layout']['align_bytes'] == A_poison      # A 가 실제로 바뀌었다
                 and _shape_p == _shape_c and _sp_mixed                 # 분류는 전건 보존·혼재
                 and _v3_slot_fingerprint(_mf_sp_p) != _v3_slot_fingerprint(_mf_sp_c)
                 and _v3_record_offsets(_mf_sp_p) != _v3_record_offsets(_mf_sp_c))
        v3_trust.append(('N4 분류 보존 축: **aligned 분류가 전건 동일**한 A 변화(%d→%d)에서도 '
                         '슬롯·records 주소가 움직인다(주소 이동이 분류 전환의 부수효과가 아니다)'
                         % (A_v3, A_poison), ok_t4,
                         'hidden=%d · 분류 보존=%s · aligned/비4K 혼재=%s · slot_stride_max %r→%r'
                         % (v3_hid_sp, _shape_p == _shape_c, _sp_mixed,
                            _mf_sp_c['layout']['slot_stride_max'], _mf_sp_p['layout']['slot_stride_max'])))
        ok_v3_trust = all(ok for _, ok, _ in v3_trust)
        # ★개수는 손으로 적지 않는다(§2-8 규칙 4) — len 으로 낸다. 이 값은 아래
        #   CONTRACT-REGISTRY-② 가 정오 2 의 "v3-⑬ 현행 **N항**" 표기와 직접 대조한다.
        checks.append(('v3-⑬ 선언된 공통 신뢰원(정렬 질의) 경계 **%d항 개별 assertion**(r9 M3 분해 + '
                       'r10 N4 2항): ⓐ잘못된 A 가 승격 + 그 A 가 만든 **슬롯 주소**·**bracket EOF '
                       '직접 계산**(양쪽 모두 source EOF 이내)·**records[].data_offset** 이 각각 clean 과 '
                       '상이 / ⓑ재검증이 A·질의 provenance·slot_stride_max 를 적발 + **파트 주소와 레코드 '
                       '주소가 problems 에 각각** 등장 + **레코드 진단이 data_offset 단독**으로 뜸 / '
                       '별개 축: A 불변 provenance 오염도 적발 / **분류 보존 A 변화**에서도 주소 이동 — '
                       'A 를 독립 재유도하지 않는다는 계약과 그 완화 장치를 실행 가능한 형태로 고정'
                       % len(v3_trust), ok_v3_trust))
        print('[selftest] v3-⑬ 신뢰원 경계: %s' % ('PASS' if ok_v3_trust else 'FAIL'))
        for label, ok_n, note in v3_trust:
            print('    [%s] %-52s %s' % ('PASS' if ok_n else 'FAIL', label, note))

        # ---- v3-⑪ 승격·재검증 부수 계약(--force·experts.bin 혼재 거부·verify-only 바이트 불변) ----
        v3_side = []
        ok_force, msg_force = _expect_abort(lambda: do_virtual_plan(v3_mixed, v3_out, profile_id=None,
                                                                     force=False, enforce_reference=False,
                                                                     allow_default_align=True))
        v3_side.append(('--force 없이 기존 산출물 덮어쓰기 거부', ok_force and 'already exists' in msg_force, msg_force[:80]))
        _sha_before_force = hashlib.sha256(open(os.path.join(v3_out, MANIFEST_FILENAME), 'rb').read()).hexdigest()
        _, _rep_force = do_virtual_plan(v3_mixed, v3_out, profile_id=None, force=True,
                                        enforce_reference=False, allow_default_align=True)
        v3_side.append(('--force 재생산 PASS + manifest 재해시 결속', _rep_force['pass'] is True
                        and _rep_force['manifest_sha256'] == hashlib.sha256(
                            open(os.path.join(v3_out, MANIFEST_FILENAME), 'rb').read()).hexdigest(), ''))
        v3_mixdir = os.path.join(scratch, 'out_v3_mixed_bin')
        os.makedirs(v3_mixdir, exist_ok=True)
        open(os.path.join(v3_mixdir, 'experts.bin'), 'wb').write(b'\x00' * 16)
        ok_mix, msg_mix = _expect_abort(lambda: do_virtual_plan(v3_mixed, v3_mixdir, profile_id=None,
                                                                 force=True, enforce_reference=False,
                                                                 allow_default_align=True))
        v3_side.append(('legacy experts.bin 이 있는 디렉토리에 virtual 산출 거부(혼재 방지)',
                        ok_mix and 'experts.bin' in msg_mix, msg_mix[:80]))
        _vo_before = hashlib.sha256(open(os.path.join(v3_out, MANIFEST_FILENAME), 'rb').read()).hexdigest()
        _vo_rep = do_verify_only_virtual(v3_mixed, v3_out, profile_id=None, enforce_reference=False,
                                         allow_default_align=True)
        _vo_after = hashlib.sha256(open(os.path.join(v3_out, MANIFEST_FILENAME), 'rb').read()).hexdigest()
        _vo_disk = json.loads(open(os.path.join(v3_out, PLAN_REPORT_FILENAME), 'r', encoding='utf-8').read())
        v3_side.append(('--verify-only(virtual): PASS·manifest 바이트 불변·plan_report 갱신',
                        _vo_rep['pass'] is True and _vo_before == _vo_after
                        and _vo_disk.get('manifest_sha256') == _vo_after, ''))
        ok_vo_missing, msg_vo_missing = _expect_abort(
            lambda: do_verify_only_virtual(v3_mixed, os.path.join(scratch, 'no_such_v3_dir'),
                                           profile_id=None, enforce_reference=False, allow_default_align=True))
        v3_side.append(('--verify-only(virtual): 산출물 부재 거부', ok_vo_missing, msg_vo_missing[:80]))
        ok_v3_side = all(ok for _, ok, _ in v3_side)
        checks.append(('v3-⑪ 승격·재검증 부수 계약 %d종(--force 게이트·재생산 결속·bin 혼재 거부·'
                       'verify-only 바이트 불변·산출물 부재 거부)' % len(v3_side), ok_v3_side))
        print('[selftest] v3-⑪ 부수 계약: %s' % ('PASS' if ok_v3_side else 'FAIL'))
        for label, ok_n, note in v3_side:
            print('    [%s] %-52s %s' % ('PASS' if ok_n else 'FAIL', label, note))

        # ---- v3-⑫ CLI 계약(--mode 기본 bin·virtual plan 0바이트·실행 산출물·미지 mode 거부) ----
        v3_tpl_kw = dict(arch='gpt-oss', n_expert=4, n_expert_used=2, moe_layers=(0, 1, 2), block_count=3,
                         schema='separate', bias=True, hidden=v3_hid, alignment=32, seed=4111,
                         extra_tensors=(V3_FILLER,))
        v3_tpl_model = _tpl_fixture('v3_tpl_gptoss.gguf', **v3_tpl_kw)
        v3_cli_plan_out = os.path.join(scratch, 'out_v3_cli_plan')
        v3_cli_run_out = os.path.join(scratch, 'out_v3_cli_run')
        d1 = _cli(['--plan', '--mode', 'virtual', '--experimental-arch-template',
                   '--model', v3_tpl_model, '--out', v3_cli_plan_out])
        ok_d1 = (d1.returncode == 0 and 'mode=virtual' in d1.stdout
                 and 'schema_version=3.0' in d1.stdout
                 and not os.path.exists(os.path.join(v3_cli_plan_out, MANIFEST_FILENAME))
                 and not os.path.exists(os.path.join(v3_cli_plan_out, PLAN_REPORT_FILENAME))
                 and not os.path.exists(os.path.join(v3_cli_plan_out, 'experts.bin')))
        d2 = _cli(['--mode', 'virtual', '--experimental-arch-template',
                   '--model', v3_tpl_model, '--out', v3_cli_run_out, '--source-full-sha'])
        _d2_report = os.path.join(v3_cli_run_out, PLAN_REPORT_FILENAME)
        _d2_rep = json.loads(open(_d2_report, 'r', encoding='utf-8').read()) if os.path.exists(_d2_report) else {}
        ok_d2 = (d2.returncode == 0
                 and os.path.exists(os.path.join(v3_cli_run_out, MANIFEST_FILENAME))
                 and os.path.exists(os.path.join(v3_cli_run_out, DERIVED_EXPECT_FILENAME))
                 and not os.path.exists(os.path.join(v3_cli_run_out, 'experts.bin'))
                 and _d2_rep.get('pass') is True
                 and _d2_rep.get('reference_lock', {}).get('profile_id') == 'arch-template:gpt-oss@1'
                 and all(len(s.get('full_sha256', '')) == 64 for s in _d2_rep.get('sources', []))
                 and _d2_rep.get('full_file_sha256_recorded') is True)
        d3 = _cli(['--plan', '--mode', 'bogus', '--profile', 'gpt-oss-120b',
                   '--model', v3_tpl_model, '--out', v3_cli_plan_out])
        ok_d3 = (d3.returncode != 0 and 'invalid choice' in (d3.stderr or ''))
        d4 = _cli(['--verify-only', '--mode', 'virtual', '--experimental-arch-template',
                   '--model', v3_tpl_model, '--out', v3_cli_run_out])
        ok_d4 = (d4.returncode == 0 and '[verify-only] PASS' in d4.stdout)
        d5 = _cli(['--plan', '--source-full-sha', '--profile', 'gpt-oss-120b',
                   '--model', v3_tpl_model, '--out', v3_cli_plan_out])
        ok_d5 = (d5.returncode != 0 and 'mode virtual only' in (d5.stderr or ''))
        d6 = _cli(['--help'])
        ok_d6 = (d6.returncode == 0 and '--mode' in d6.stdout and 'virtual' in d6.stdout)
        ok_v3_cli = ok_d1 and ok_d2 and ok_d3 and ok_d4 and ok_d5 and ok_d6
        checks.append(('v3-⑫ CLI 계약(--plan --mode virtual 은 0바이트·본실행은 manifest+plan_report(+derived '
                       'expect)만·미지 --mode 거부·--verify-only --mode virtual PASS·--source-full-sha 는 '
                       'virtual 전용·--help 노출)', ok_v3_cli))
        print('[selftest] v3-⑫ CLI: plan0바이트=%s 본실행=%s 미지mode거부=%s verify-only=%s fullsha게이트=%s help=%s'
              % (ok_d1, ok_d2, ok_d3, ok_d4, ok_d5, ok_d6))

        # ---- v3-⑭ §Z-③ legacy alignment 결속(D-A2 분자 baseline) [[C:repack.legacy-align]] ----
        # unpaired = canonical 4096 강제 / paired = 실제 v2 재팩 manifest 의 align_bytes 승계 +
        # verifier 가 그 파일을 strict 재개방해 SHA·schema·모델/reference/sources identity·값을 재확인.
        # ★paired baseline 은 **질의 주입으로 A_v3*2 볼륨에** 만든다 — 그래야 legacy 축이
        #   source-volume A 와 실제로 다른 값이 되고("독립 축"이라는 주장이 값으로 판별된다),
        #   두 축을 뒤바꿔 읽는 회귀가 이 관문에 걸린다.
        z3 = []
        _z3_v2_dir = os.path.join(scratch, 'out_z3_v2_baseline')
        globals()['query_sector_alignment_for_path'] = _q_poisoned_A
        try:
            do_repack(v3_mixed, _z3_v2_dir, profile_id=None, force=False, run_verify=True,
                      enforce_reference=False, allow_default_align=True)
        finally:
            globals()['query_sector_alignment_for_path'] = _orig_q
        _z3_v2_manifest = os.path.join(_z3_v2_dir, MANIFEST_FILENAME)
        _z3_v2_align = json.loads(open(_z3_v2_manifest, 'r', encoding='utf-8').read())['layout']['align_bytes']
        _z3_v2_sha = hashlib.sha256(open(_z3_v2_manifest, 'rb').read()).hexdigest()
        # 결속 대상은 **사본**으로 건넨다(네거티브가 v2 산출물 자체를 훼손하지 않게).
        _z3_paired = os.path.join(scratch, 'z3_paired_v2_manifest.json')
        shutil.copyfile(_z3_v2_manifest, _z3_paired)

        _z3_un = _load_manifest_disk(v3_out)
        z3.append(('unpaired 기본 = canonical_4096 / 값 4096 / paired 필드 부재 + plan_report echo',
                   _z3_un['layout'].get('legacy_align_source') == LEGACY_ALIGN_SOURCE_CANONICAL
                   and _z3_un['layout'].get('legacy_align_bytes') == LEGACY_ALIGN_CANONICAL_BYTES
                   and 'legacy_v2_manifest_sha256' not in _z3_un['reference_lock']
                   and 'legacy_v2_manifest_path' not in _z3_un['reference_lock']
                   and v3_rep_disk.get('legacy_align', {}).get('legacy_align_source') == LEGACY_ALIGN_SOURCE_CANONICAL
                   and v3_rep_disk.get('legacy_align', {}).get('legacy_align_bytes') == LEGACY_ALIGN_CANONICAL_BYTES,
                   'bytes=%r source=%r' % (_z3_un['layout'].get('legacy_align_bytes'),
                                           _z3_un['layout'].get('legacy_align_source'))))

        _z3_out = os.path.join(scratch, 'out_z3_paired')
        _, _z3_rep = do_virtual_plan(v3_mixed, _z3_out, profile_id=None, force=False,
                                     enforce_reference=False, allow_default_align=True,
                                     legacy_v2_manifest=_z3_paired)
        _z3_mf = _load_manifest_disk(_z3_out)
        z3.append(('paired 양성: v2 align_bytes 승계 · source-volume A 와 **상이** · reference_lock SHA/경로 결속',
                   _z3_rep['pass'] is True
                   and _z3_mf['layout'].get('legacy_align_source') == LEGACY_ALIGN_SOURCE_PAIRED
                   and _z3_mf['layout'].get('legacy_align_bytes') == _z3_v2_align
                   and _z3_v2_align != A_v3                       # 두 축이 실제로 갈렸다는 증명
                   and _z3_mf['layout']['align_bytes'] == A_v3
                   and _z3_mf['reference_lock'].get('legacy_v2_manifest_sha256') == _z3_v2_sha
                   and _z3_rep.get('legacy_align', {}).get('legacy_align_bytes') == _z3_v2_align,
                   'legacy=%r vs A=%r' % (_z3_mf['layout'].get('legacy_align_bytes'), A_v3)))

        # 네거티브 ⓐ manifest 측 위조(자기일관·enum·값)
        for _lbl, _tag, _fn in (
                ('legacy_align_bytes 삭제', 'la_del', lambda mf: mf['layout'].pop('legacy_align_bytes')),
                ('legacy_align_source 미지값', 'la_src', lambda mf: mf['layout'].__setitem__('legacy_align_source', 'guessed')),
                ('paired 값 변조(v2 실값과 불일치)', 'la_val',
                 lambda mf: mf['layout'].__setitem__('legacy_align_bytes', mf['layout']['legacy_align_bytes'] * 2)),
                ('paired SHA 변조', 'la_sha',
                 lambda mf: mf['reference_lock'].__setitem__('legacy_v2_manifest_sha256', '0' * 64)),
                ('paired 경로 삭제(재개방 불가)', 'la_path',
                 lambda mf: mf['reference_lock'].pop('legacy_v2_manifest_path')),
        ):
            _ok_n, _rep_n = _v3_neg(_tag, _fn, src_dir=_z3_out)
            z3.append(('네거티브: %s' % _lbl, _ok_n, (_rep_n['problems'] or [''])[0][:88]))
        _ok_can, _rep_can = _v3_neg('la_canon_paired', lambda mf: mf['reference_lock'].__setitem__(
            'legacy_v2_manifest_sha256', '1' * 64), src_dir=v3_out)
        z3.append(('네거티브: canonical 인데 paired 필드 부착', _ok_can, (_rep_can['problems'] or [''])[0][:88]))

        # 네거티브 ⓑ 결속 파일 측(부재·바이트 변조) — manifest 는 손대지 않는다
        _z3_tmp_paired = os.path.join(scratch, 'z3_paired_v2_tmp.json')
        shutil.copyfile(_z3_v2_manifest, _z3_tmp_paired)
        _z3_out_tmp = os.path.join(scratch, 'out_z3_paired_tmp')
        do_virtual_plan(v3_mixed, _z3_out_tmp, profile_id=None, force=False, enforce_reference=False,
                        allow_default_align=True, legacy_v2_manifest=_z3_tmp_paired)
        os.remove(_z3_tmp_paired)
        _rep_miss = verify_virtual_manifest(v3_mixed, _z3_out_tmp, profile_id=None,
                                            enforce_reference=False, allow_default_align=True)
        z3.append(('네거티브: 결속된 v2 manifest 파일 부재', _rep_miss['pass'] is False,
                   (_rep_miss['problems'] or [''])[0][:88]))
        with open(_z3_tmp_paired, 'wb') as _f:
            _f.write(open(_z3_v2_manifest, 'rb').read() + b' ')      # 의미 불변·바이트만 변경
        _rep_tam = verify_virtual_manifest(v3_mixed, _z3_out_tmp, profile_id=None,
                                           enforce_reference=False, allow_default_align=True)
        z3.append(('네거티브: 결속된 v2 manifest 사후 변조(재해시 불일치)', _rep_tam['pass'] is False,
                   (_rep_tam['problems'] or [''])[0][:88]))

        # 네거티브 ⓒ **다른 모델의 v2 baseline 결속** — model 스칼라(arch/n_layer/n_expert/
        # n_expert_used/moe_layers)는 두 픽스처가 전부 같아서 sources[] identity 만이 이걸 잡는다.
        _z3_other_dir = os.path.join(scratch, 'out_z3_v2_other')
        do_repack(v3_non, _z3_other_dir, profile_id=None, force=False, run_verify=True,
                  enforce_reference=False, allow_default_align=True)
        _z3_other = os.path.join(_z3_other_dir, MANIFEST_FILENAME)
        _z3_other_mf = json.loads(open(_z3_other, 'r', encoding='utf-8').read())
        _z3_scalars_same = all(_z3_other_mf['model'][k] == _z3_un['model'][k]
                               for k in ('arch', 'n_layer', 'n_expert', 'n_expert_used', 'moe_layers'))
        _ok_wrong, _msg_wrong = _expect_abort(lambda: do_virtual_plan(
            v3_mixed, os.path.join(scratch, 'out_z3_wrong_pair'), profile_id=None, force=False,
            enforce_reference=False, allow_default_align=True, legacy_v2_manifest=_z3_other))
        z3.append(('네거티브: 다른 모델의 v2 baseline 결속 거부(model 스칼라 동일=%s → sources identity 가 적발)'
                   % _z3_scalars_same, _ok_wrong and _z3_scalars_same, _msg_wrong[:88]))

        # 네거티브 ⓓ v2 가 아닌 파일(자기 자신 = v3 manifest)을 baseline 으로 지정
        _ok_notv2, _msg_notv2 = _expect_abort(lambda: do_virtual_plan(
            v3_mixed, os.path.join(scratch, 'out_z3_notv2'), profile_id=None, force=False,
            enforce_reference=False, allow_default_align=True,
            legacy_v2_manifest=os.path.join(v3_out, MANIFEST_FILENAME)))
        z3.append(('네거티브: v2 가 아닌 manifest(schema 3.0)를 baseline 으로 지정 거부', _ok_notv2, _msg_notv2[:88]))

        # ★r1 [MED] 수정분 검증 — locator 수명주기. `legacy_v2_manifest_path` 는 **비권위 locator**
        # 이고 identity 권위는 기록된 SHA 다. 정상 운영자가 paired v2 artifact 만 옮긴 뒤 새 경로를
        # 지정하는 복구 경로를 행사한다(구 동작은 옛 절대경로를 열다 죽는 것으로 끝났다).
        _z3_moved = os.path.join(scratch, 'z3_paired_v2_moved.json')
        shutil.move(_z3_paired, _z3_moved)
        _rep_noov = do_verify_only_virtual(v3_mixed, _z3_out, profile_id=None, enforce_reference=False,
                                           allow_default_align=True)
        z3.append(('locator 이동 + override 없음 = fail-close(기록 경로를 열 수 없다)',
                   _rep_noov['pass'] is False, (_rep_noov['problems'] or [''])[0][:88]))
        _rep_ov = do_verify_only_virtual(v3_mixed, _z3_out, profile_id=None, enforce_reference=False,
                                         allow_default_align=True, legacy_v2_manifest=_z3_moved)
        _ov = _rep_ov.get('legacy_align', {})
        z3.append(('★locator override 수용 — 새 경로로 재개방하고 SHA·identity 전건 일치라 PASS · echo 에 '
                   'locator_source=cli_override + resolved 경로 기록',
                   _rep_ov['pass'] is True
                   and _ov.get('legacy_v2_locator_source') == 'cli_override'
                   and _same_fs_name(_ov.get('legacy_v2_manifest_resolved_path'), _z3_moved)
                   and _ov.get('legacy_align_bytes') == _z3_v2_align,
                   'resolved=%r' % (_ov.get('legacy_v2_manifest_resolved_path'),)))
        _rep_ov_bad = do_verify_only_virtual(v3_mixed, _z3_out, profile_id=None, enforce_reference=False,
                                             allow_default_align=True, legacy_v2_manifest=_z3_other)
        z3.append(('★override 가 identity 를 완화하지 않는다 — 다른 v2 manifest 를 가리키면 기록 SHA 불일치 FAIL',
                   _rep_ov_bad['pass'] is False, (_rep_ov_bad['problems'] or [''])[0][:88]))
        shutil.move(_z3_moved, _z3_paired)          # 원복 — 뒤 항목이 이 경로를 기대한다

        # ★F1(r2 지적) — canonical 분기 정합. locator_source 는 paired 에서만 의미가 있고,
        # canonical 산출물에 override 를 주는 것은 조용한 무시가 아니라 fail-close 여야 한다.
        _rep_can = do_verify_only_virtual(v3_mixed, v3_out, profile_id=None, enforce_reference=False,
                                          allow_default_align=True)
        z3.append(('canonical 산출물의 locator_source 는 %r(‘manifest’ 오기록 금지)' % 'none',
                   _rep_can['pass'] is True
                   and _rep_can.get('legacy_align', {}).get('legacy_v2_locator_source') == 'none',
                   'source=%r' % _rep_can.get('legacy_align', {}).get('legacy_v2_locator_source')))
        _rep_can_ov = do_verify_only_virtual(v3_mixed, v3_out, profile_id=None, enforce_reference=False,
                                             allow_default_align=True, legacy_v2_manifest=_z3_paired)
        z3.append(('★canonical 산출물 + --legacy-v2-manifest override = **fail-close**'
                   '(구 동작은 파일을 열지도 않고 cli_override 표기 후 PASS)',
                   _rep_can_ov['pass'] is False
                   and any('no paired v2 baseline' in str(p) for p in _rep_can_ov['problems']),
                   (_rep_can_ov['problems'] or [''])[0][:88]))
        ok_z3 = all(ok for _, ok, _ in z3)
        checks.append(('v3-⑭ §Z-③ legacy alignment 결속 %d종 — unpaired canonical_4096 강제 / paired 는 v2 '
                       'manifest 를 strict 재개방해 SHA·schema·model/reference/sources identity·align 값 재확인 '
                       '(D-A2 분자 baseline · source-volume A 와 독립 축) + ★locator 는 비권위이므로 이동 시 '
                       '`--verify-only --legacy-v2-manifest` override 로 복구되되 identity 는 완화되지 않는다'
                       % len(z3), ok_z3))
        print('[selftest] v3-⑭ legacy align 결속: %s (v2 baseline A=%d vs source A=%d)'
              % ('PASS' if ok_z3 else 'FAIL', _z3_v2_align, A_v3))
        for _lbl, _ok_n, _note in z3:
            print('    [%s] %-58s %s' % ('PASS' if _ok_n else 'FAIL', _lbl, _note))

        # ---- v3-⑮ §Z-④ derived expect candidate 격리(no-clobber 범위 회복) ----
        # r1 [MEDIUM]: derived.expect.json 이 candidate 검증 **전에** 최종 이름으로 선교체돼,
        # verifier 가 뒤늦게 FAIL 하면 manifest/plan_report 는 보존되는데 derived expect 만 갈려
        # 있었다. 도달 경로도 평범하다 — 중단된 bin 실행이 남긴 `experts.bin.partial` 이 있는
        # 디렉토리에 virtual `--force` 를 돌리면 시작 가드(experts.bin 만 확인)는 통과하고
        # verifier 의 (14) 가 뒤늦게 FAIL 한다.
        z4 = []
        _z4_out = os.path.join(scratch, 'out_z4_tpl')
        _, _z4_rep = do_virtual_plan(v3_tpl_model, _z4_out, profile_id=None, force=False,
                                     enforce_reference=False, allow_default_align=True, arch_template=True)
        _z4_dpath = os.path.join(_z4_out, DERIVED_EXPECT_FILENAME)
        _z4_cand = _z4_dpath + '.partial'
        z4.append(('양성: 승격 후 derived expect 는 최종 이름에 있고 candidate 는 남지 않는다',
                   _z4_rep['pass'] is True and os.path.exists(_z4_dpath) and not os.path.exists(_z4_cand), ''))
        # ★센티넬로 판별력을 만든다: 같은 모델이라 재유도 바이트가 동일해서 "안 바뀌었다"가
        #   자동 성립해버린다. 최종 이름에 알아볼 수 있는 바이트를 넣어두면, 구 동작(선교체)은
        #   이걸 반드시 지우고 신 동작은 반드시 남긴다.
        _z4_sentinel = b'{"sentinel":"the promoted derived expect must survive a failed re-production"}\n'
        with open(_z4_dpath, 'wb') as _f:
            _f.write(_z4_sentinel)
        _z4_mf_before = open(os.path.join(_z4_out, MANIFEST_FILENAME), 'rb').read()
        _z4_rp_before = open(os.path.join(_z4_out, PLAN_REPORT_FILENAME), 'rb').read()
        with open(os.path.join(_z4_out, 'experts.bin.partial'), 'wb') as _f:
            _f.write(b'\x00' * 8)
        _z4_ok_abort, _z4_msg = _expect_abort(lambda: do_virtual_plan(
            v3_tpl_model, _z4_out, profile_id=None, force=True, enforce_reference=False,
            allow_default_align=True, arch_template=True))
        z4.append(('검증 FAIL 재생산: 중단됨(experts.bin.partial 혼재 적발)', _z4_ok_abort, _z4_msg[:88]))
        z4.append(('★구 derived expect 가 **선교체되지 않았다**(센티넬 바이트 보존)',
                   open(_z4_dpath, 'rb').read() == _z4_sentinel, ''))
        z4.append(('실패분은 candidate 로만 남는다(derived.expect.json.partial 실재)',
                   os.path.exists(_z4_cand), ''))
        z4.append(('manifest.json·plan_report.json 바이트 불변',
                   open(os.path.join(_z4_out, MANIFEST_FILENAME), 'rb').read() == _z4_mf_before
                   and open(os.path.join(_z4_out, PLAN_REPORT_FILENAME), 'rb').read() == _z4_rp_before, ''))
        ok_z4 = all(ok for _, ok, _ in z4)
        checks.append(('v3-⑮ §Z-④ derived expect candidate 격리 %d종 — 검증 실패 시 구 산출물 3종(derived '
                       'expect·manifest·plan_report)이 전부 보존되고 신규분은 .partial 로만 남는다' % len(z4), ok_z4))
        print('[selftest] v3-⑮ derived expect 격리: %s' % ('PASS' if ok_z4 else 'FAIL'))
        for _lbl, _ok_n, _note in z4:
            print('    [%s] %-58s %s' % ('PASS' if _ok_n else 'FAIL', _lbl, _note))

        # ---- v3-⑯ §Z-⑤ full-SHA provenance 승계/downgrade ----
        # r1 [LOW]: 플래그 없는 `--verify-only` 1회로 full SHA 와 full_file_sha256_recorded 가
        # **조용히 소실**됐다(report 원자 교체). 승계 조건은 source identity 동일이며, 하나라도
        # 다르면 옛 SHA 는 지금 파일의 것이 아니므로 승계 대신 downgrade 를 명시 기록한다.
        z5 = []
        _z5_model = os.path.join(scratch, 'v3_z5_fullsha.gguf')     # 전용 픽스처(mtime 을 건드린다)
        write_synthetic_gguf([_z5_model], n_expert=4, moe_layers=(0, 1), schema='separate', bias=True,
                             hidden=v3_hid, alignment=32, seed=4131, extra_tensors=(V3_FILLER,))
        _z5_out = os.path.join(scratch, 'out_z5_fullsha')
        _, _z5_rep0 = do_virtual_plan(_z5_model, _z5_out, profile_id=None, force=False,
                                      enforce_reference=False, allow_default_align=True, full_sha=True)
        _z5_shas0 = [s.get('full_sha256') for s in _z5_rep0['sources']]
        z5.append(('본실행 --source-full-sha: 전 shard full SHA 기록 · provenance=recomputed',
                   _z5_rep0['full_file_sha256_recorded'] is True
                   and all(isinstance(s, str) and len(s) == 64 for s in _z5_shas0)
                   and _z5_rep0.get('full_sha_provenance', {}).get('state') == 'recomputed', ''))
        _z5_rep1 = do_verify_only_virtual(_z5_model, _z5_out, profile_id=None, enforce_reference=False,
                                          allow_default_align=True)
        _z5_disk1 = json.loads(open(os.path.join(_z5_out, PLAN_REPORT_FILENAME), 'r', encoding='utf-8').read())
        z5.append(('★플래그 없는 --verify-only 가 full SHA 를 **승계**(구 동작은 여기서 소실)',
                   _z5_rep1['pass'] is True and _z5_rep1['full_file_sha256_recorded'] is True
                   and [s.get('full_sha256') for s in _z5_rep1['sources']] == _z5_shas0
                   and _z5_rep1.get('full_sha_provenance', {}).get('state') == 'inherited'
                   and [s.get('full_sha256') for s in _z5_disk1.get('sources', [])] == _z5_shas0,
                   'state=%r' % _z5_rep1.get('full_sha_provenance', {}).get('state')))
        _z5_mt = os.path.getmtime(_z5_model)
        os.utime(_z5_model, (_z5_mt + 120, _z5_mt + 120))            # 정상 교체·재생성의 실제 흔적
        _z5_rep2 = do_verify_only_virtual(_z5_model, _z5_out, profile_id=None, enforce_reference=False,
                                          allow_default_align=True)
        z5.append(('source identity 변화 시 승계 금지 → **명시적 downgrade 기록**(거짓 진술 방지)',
                   _z5_rep2['pass'] is False
                   and _z5_rep2.get('full_file_sha256_recorded') is False
                   and _z5_rep2.get('full_sha_provenance', {}).get('state') == 'downgraded'
                   and 'mtime' in (_z5_rep2.get('full_sha_provenance', {}).get('reason') or '')
                   and all('full_sha256' not in s for s in _z5_rep2['sources']),
                   'reason=%r' % (_z5_rep2.get('full_sha_provenance', {}).get('reason'),)))
        _z5_rep3 = do_verify_only_virtual(v3_mixed, v3_out, profile_id=None, enforce_reference=False,
                                          allow_default_align=True)
        z5.append(('애초에 기록이 없던 산출물은 absent 로 닫힌다(없는 것을 승계하지 않는다)',
                   _z5_rep3['pass'] is True
                   and _z5_rep3.get('full_sha_provenance', {}).get('state') == 'absent'
                   and _z5_rep3.get('full_file_sha256_recorded') is False, ''))
        ok_z5 = all(ok for _, ok, _ in z5)
        checks.append(('v3-⑯ §Z-⑤ full-SHA provenance %d종 — --verify-only 의 report 원자 교체가 source '
                       'identity 동일이면 승계하고, 달라지면 명시적 downgrade 를 남긴다(조용한 소실 0)' % len(z5), ok_z5))
        print('[selftest] v3-⑯ full-SHA provenance: %s' % ('PASS' if ok_z5 else 'FAIL'))
        for _lbl, _ok_n, _note in z5:
            print('    [%s] %-58s %s' % ('PASS' if _ok_n else 'FAIL', _lbl, _note))

        # ---- v3-⑰ §Z-⑥ mode=virtual --plan stdout 계약 + subtractive 변이 ----
        # 교차 트랙 durable(HANDOFF_DEV §4)의 "launcher 파서 소비 줄의 selftest 직접 주장" 중
        # virtual 몫. ⓑ4~ⓑ6 은 bin+arch-template 전용이라 virtual 문면은 아무도 주장하지 않았다.
        z6 = []
        _z6_out = os.path.join(scratch, 'out_z6_vplan')
        # ⓑ4 와 같은 이유로 arch-template 경로를 쓴다 — `--plan` 은 카탈로그 참조 락을 강제하므로
        # 합성 픽스처는 현장 유도 expect 로만 실 CLI 경로를 탈 수 있다(0바이트 계약은 그대로).
        _z6_args = argparse.Namespace(model=v3_tpl_model, out=_z6_out, profile=None, scope=None,
                                      arch_template=True, mode=MODE_VIRTUAL, source_full_sha=False,
                                      legacy_v2_manifest=None)
        _z6_buf, _z6_saved = io.StringIO(), sys.stdout
        try:
            sys.stdout = _z6_buf
            cmd_plan(_z6_args)
        finally:
            sys.stdout = _z6_saved
        _z6_lines = _z6_buf.getvalue().split('\n')
        _z6_problems = virtual_plan_contract_problems(_z6_lines)
        z6.append(('실 --plan --mode virtual stdout 이 계약 %d줄을 전부 만족(각 1회·머리/완료 줄 사이)'
                   % (len(VIRTUAL_PLAN_KEYED_LINES) + 2), not _z6_problems, str(_z6_problems)[:88]))
        z6.append(('--plan 은 여전히 0바이트(계약 검사가 산출물을 만들지 않는다)',
                   not os.path.exists(_z6_out), ''))
        # subtractive: 필수 줄을 **하나씩** 지운 입력을 계약 검사가 전부 거부해야 한다.
        _z6_required = [(k, [i for i, ln in enumerate(_z6_lines) if re.match(p, ln)][0])
                        for k, p in VIRTUAL_PLAN_KEYED_LINES]
        _z6_required += [('mode: header', _z6_lines.index(VIRTUAL_PLAN_HEAD_LINE)),
                         ('--plan done', _z6_lines.index(LAUNCHER_PLAN_DONE_LINE))]
        _z6_sub_bad = []
        for _k, _at in _z6_required:
            _cut = _z6_lines[:_at] + _z6_lines[_at + 1:]
            if not virtual_plan_contract_problems(_cut):
                _z6_sub_bad.append(_k)
        z6.append(('subtractive %d변이(필수 줄 1개 삭제) 전건 거부' % len(_z6_required),
                   not _z6_sub_bad, '통과해버린 줄=%r' % _z6_sub_bad if _z6_sub_bad else ''))
        # 중복 변이: 파서는 "단일 plan 캡처가 아니다"로 거부해야 한다(bin ⓑ4 와 같은 성질).
        _z6_dup = _z6_lines[:_z6_required[0][1] + 1] + [_z6_lines[_z6_required[0][1]]] \
            + _z6_lines[_z6_required[0][1] + 1:]
        z6.append(('중복 변이(계약 줄 2회 출현) 거부', bool(virtual_plan_contract_problems(_z6_dup)), ''))
        # bin 표의 3줄은 virtual 에 **없어야** 한다 — 있으면 source-volume A 를 legacy output A 로
        # 오독시키는 문면이 되살아난 것이다(정확히 그 오독이 r1 이 지목한 온보딩 선결 1번).
        _z6_binonly = [k for k, p in LAUNCHER_PLAN_KEYED_LINES if k in ('tpl', 'stride', 'bytes')
                       and any(re.match(p, ln) for ln in _z6_lines)]
        z6.append(('bin 전용 3줄(template layers·output alignment·expert_payload_total)이 virtual 에 부재',
                   not _z6_binonly, '되살아난 줄=%r' % _z6_binonly if _z6_binonly else ''))
        ok_z6 = all(ok for _, ok, _ in z6)
        checks.append(('v3-⑰ §Z-⑥ mode=virtual --plan stdout 계약 %d종 — 실 stdout 로 필수 줄 각 1회 주장 + '
                       'subtractive/중복 변이 전건 거부 + bin 전용 문면 부재(소비 표면 회귀 관문)' % len(z6), ok_z6))
        print('[selftest] v3-⑰ virtual stdout 계약: %s (필수 %d줄 · subtractive %d변이)'
              % ('PASS' if ok_z6 else 'FAIL', len(_z6_required), len(_z6_required)))
        for _lbl, _ok_n, _note in z6:
            print('    [%s] %-58s %s' % ('PASS' if _ok_n else 'FAIL', _lbl, _note))

        # ---- v3-⑲ §Z-⑦ A>4096 합성 주입 E2E(8192·16384) ----
        # 현 머신은 logical 512 / physical 4096 이라 host 질의만으로는 **영원히 A=4096** 만
        # 행사한다(부속 정오 2-ⓗ). 실 8K-sector 장비는 불요 — 섹터 질의를 주입해 producer→
        # verifier 를 그대로 태운다. v3-⑬ 이 이미 A 를 2배로 바꾸지만 그건 신뢰원 경계 고정이
        # 목적이라 ⓗ 가 요구한 5항(aligned/비정렬 혼재·h=0 과 h!=0·expert 진행 중 head wrap·
        # exact EOF 와 bracket EOF·multi-source max(A))을 세우지 않는다. 여기서 세운다.
        #
        # ★형상 근거(hidden=64 · n_expert=36 · separate+bias · 2층): A=8192/16384 **양쪽**에서
        #   weight slice=16,384(=A 배수 → aligned) · bias slice=256(→ 비정렬) 이 동시에 나오고,
        #   층0 weight 는 h=0 · 층1 weight 는 h!=0 이 되며, bias 는 expert 진행 중 head 가 A 를
        #   넘어 wrap 한다. 아래 assertion 이 그 수치를 값으로 찍는다(형상이 바뀌면 즉시 FAIL).
        # ★GGUF `alignment` 를 16384 로 두는 것은 data_start 를 A 배수로 만드는 픽스처 노브다 —
        #   routed 텐서 크기가 전부 A 배수라 data_start 가 A 배수가 아니면 h=0 이 나올 수 없다.
        #   피검 경로(슬롯·EOF 산술)는 이 값에 의존하지 않는다.
        z7 = []
        _z7_orig_q = query_sector_alignment_for_path

        def _z7_query(default_A, by_basename=None):
            def _q(path):
                info = dict(_z7_orig_q(path))
                info['physical'] = (by_basename or {}).get(os.path.basename(path), default_A)
                return info
            return _q

        _z7_kw = dict(n_expert=36, moe_layers=(0, 1), schema='separate', bias=True, hidden=64,
                      alignment=16384, seed=4141)
        # base = 꼬리 filler 0 → 마지막 routed 텐서의 bracket 이 파일 끝을 넘는다(= bracket EOF 픽스처).
        _z7_base = os.path.join(scratch, 'v3_z7_base.gguf')
        write_synthetic_gguf([_z7_base], **_z7_kw)
        _z7_lay = build_layout(load_model_shards(_z7_base))
        _z7_last_end = max(p['abs_offset'] + _z7_lay['n_expert'] * p['part_bytes']
                           for L in _z7_lay['layers'] for p in L['parts'])
        _z7_base_bytes = os.path.getsize(_z7_base)

        for _A7 in (8192, 16384):
            _need = (-_z7_last_end) % _A7          # bracket EOF 를 파일 끝과 정확히 맞추는 꼬리
            _tag = 'A%d' % _A7
            if _need <= 0:
                z7.append(('%s: exact/bracket 픽스처를 만들 꼬리 여유가 없다(_need=%d)' % (_tag, _need),
                           False, '형상 재설계 필요'))
                continue
            _z7_exact = os.path.join(scratch, 'v3_z7_exact_%s.gguf' % _tag)
            # ★꼬리 filler 는 **rank 2 이상**으로 준다. 픽스처 크기 산식(`_nbytes`)은 ggml 의
            # `ne0/bv * bb * prod(ne[1..last-1]) * ne[last]` 라 rank 1 이면 dims[0] 이 ne0 과
            # ne[last] 로 **두 번** 곱해져 4*d 가 아니라 4*d^2 바이트가 나온다(그 산식이 expert
            # 축을 마지막에 두는 rank>=2 를 전제한다 — 결함이 아니라 정의역 밖).
            write_synthetic_gguf([_z7_exact], extra_tensors=(
                {'name': 'blk.0.ffn_gate_inp.weight', 'dims': [_need // 4, 1], 'type': 'F32'},), **_z7_kw)
            _q7 = _z7_query(_A7)
            globals()['query_sector_alignment_for_path'] = _q7
            try:
                _o7 = os.path.join(scratch, 'out_v3_z7_%s' % _tag)
                _, _rep7 = do_virtual_plan(_z7_exact, _o7, profile_id=None, force=False,
                                           enforce_reference=False, allow_default_align=True)
                _mf7 = _load_manifest_disk(_o7)
                _lay7, _rows7 = _v3_expected_slots(_z7_exact, _A7)
                _bad7 = _v3_cmp_slots(_mf7, _rows7, _lay7['n_expert'])
                _ok_reg7, _msg_reg7 = _expect_abort(lambda: do_virtual_plan(
                    _z7_base, os.path.join(scratch, 'out_v3_z7_reg_%s' % _tag), profile_id=None,
                    force=False, enforce_reference=False, allow_default_align=True))
            finally:
                globals()['query_sector_alignment_for_path'] = _z7_orig_q

            _parts7 = [(L['layer'], p) for L in _mf7['layout']['layers'] for p in L['parts']]
            _al = [p for _, p in _parts7 if p['aligned']]
            _na = [p for _, p in _parts7 if not p['aligned']]
            _h0 = [p for p in _al if p['bracket_head'] == 0]
            _hn = [p for p in _al if p['bracket_head'] != 0]
            z7.append(('%s: E2E PASS · A 주입 반영 · 인라인 슬롯 산술 독립 검산 전항 일치' % _tag,
                       _rep7['pass'] is True and _mf7['layout']['align_bytes'] == _A7
                       and _A7 != A_v3 and not _bad7,
                       ('불일치=%r' % _bad7[:3]) if _bad7 else 'A=%d' % _mf7['layout']['align_bytes']))
            z7.append(('%s: aligned/비정렬 **혼재**(aligned %d · 비정렬 %d · slice %d/%d)'
                       % (_tag, len(_al), len(_na), _al[0]['slice_bytes'] if _al else -1,
                          _na[0]['slice_bytes'] if _na else -1),
                       bool(_al) and bool(_na)
                       and all(p['slice_bytes'] % _A7 == 0 for p in _al)
                       and all(p['slice_bytes'] % _A7 != 0 for p in _na), ''))
            # h=0 이면 브라켓이 붙지 않아 data_offset==slot_offset, h!=0 이면 정확히 h 만큼 민다.
            z7.append(('%s: h=0 파트 %d개(data_offset==slot_offset)와 h!=0 파트 %d개'
                       '(data_offset==slot_offset+h) 동시 성립' % (_tag, len(_h0), len(_hn)),
                       bool(_h0) and bool(_hn)
                       and all(p['data_offset'] == p['slot_offset'] for p in _h0)
                       and all(p['data_offset'] == p['slot_offset'] + p['bracket_head'] for p in _hn),
                       'h 값=%r' % sorted({p['bracket_head'] for p in _al})))
            # 비정렬 part 의 per-expert head(=소비자 read 창의 시작). manifest 는 이 값을 담지
            # 않으므로 §2-4 정의로 직접 계산한다 — wrap 이 **실제로 일어나는지**와, 그때도
            # staging_bytes 가 최악 read 창을 덮는지가 이 항의 명제다.
            _wrapped, _stage_bad = [], []
            for _lyr, p in _parts7:
                if p['aligned']:
                    continue
                heads = [(p['abs_offset'] + e * p['slice_bytes']) % _A7 for e in range(_mf7['model']['n_expert'])]
                if any(heads[i + 1] < heads[i] for i in range(len(heads) - 1)):
                    _wrapped.append((_lyr, p['name']))
                for hh in heads:
                    if -(-(hh + p['slice_bytes']) // _A7) * _A7 > p['staging_bytes']:
                        _stage_bad.append((_lyr, p['name'], hh))
            z7.append(('%s: 비정렬 part 의 **expert 진행 중 head wrap** 실재(%d파트) · 전 expert 의 '
                       'read 창 align_up(head_e+slice,A) <= staging_bytes' % (_tag, len(_wrapped)),
                       bool(_wrapped) and not _stage_bad,
                       ('초과=%r' % _stage_bad[:2]) if _stage_bad else 'wrap=%r' % (_wrapped[:2],)))
            _src_bytes7 = _mf7['sources'][0]['bytes']
            _bracket_end7 = -(-_z7_last_end // _A7) * _A7
            z7.append(('%s: **exact EOF** 경계 통과(align_up(마지막 routed end, A)==source.bytes=%d)'
                       % (_tag, _src_bytes7),
                       _bracket_end7 == _src_bytes7 and _src_bytes7 == _z7_base_bytes + _need, ''))
            z7.append(('%s: **bracket EOF** 위반 픽스처(꼬리 %dB 부족)는 VirtualBinRegression 으로 '
                       'mode=bin 회귀 판정' % (_tag, _need), _ok_reg7 and 'bracket EOF violation' in _msg_reg7,
                       _msg_reg7[:70]))

            # ★**h=A-1 정확 경계**(정오 ⓗ 원문 이행 — r1 [MED] 수정분·리드 처분 옵션1).
            # 구 라운드에서 내가 "h 는 항상 짝수라 A-1 은 도달 불가"라고 축소를 제안했으나
            # **철회한다**: 동결 type-trait 표에 `MXFP4=(32,17)` 이라는 **홀수 바이트** 타입이
            # 있고, parser 는 `general.alignment` 의 양수 여부만 볼 뿐 모든 `rel_offset` 이 그
            # 배수인지 강제하지 않는다(:312-322·:594-599). 즉 홀수 residue 는 accepted input 이며
            # "짝수뿐"은 코드 불변식으로 성립하지 않는다.
            # 행사 방법: GGUF `alignment` 를 **A-1(홀수)** 로 두면 `data_start=ceil(header_end/
            # (A-1))*(A-1)` 이 header_end<=A-1 인 이 픽스처에서 정확히 A-1 이 되고, rel_offset 0 인
            # 첫 routed part 의 h 가 **정확히 A-1**(최대 residue)이 된다. 픽스처 노브일 뿐 피검
            # 경로는 이 값에 의존하지 않는다.
            _hkw = dict(n_expert=4, moe_layers=(0, 1), schema='separate', bias=True, hidden=64,
                        alignment=_A7 - 1, seed=4151)
            _hb = os.path.join(scratch, 'v3_z7_hmax_base_%s.gguf' % _tag)
            write_synthetic_gguf([_hb], **_hkw)
            _hlay = build_layout(load_model_shards(_hb))
            _hlast = max(p['abs_offset'] + _hlay['n_expert'] * p['part_bytes']
                         for L in _hlay['layers'] for p in L['parts'])
            _hfill = (-_hlast) % _A7
            _hfill += (-_hfill) % 4                  # F32 는 4배수 단위 — bracket 을 덮도록 올림
            _hm = os.path.join(scratch, 'v3_z7_hmax_%s.gguf' % _tag)
            write_synthetic_gguf([_hm], extra_tensors=(
                {'name': 'blk.0.ffn_gate_inp.weight', 'dims': [_hfill // 4, 1], 'type': 'F32'},), **_hkw)
            globals()['query_sector_alignment_for_path'] = _q7
            try:
                _ho = os.path.join(scratch, 'out_v3_z7_hmax_%s' % _tag)
                _, _hrep = do_virtual_plan(_hm, _ho, profile_id=None, force=False,
                                           enforce_reference=False, allow_default_align=True)
                _hmf = _load_manifest_disk(_ho)
                _hlay2, _hrows = _v3_expected_slots(_hm, _A7)
                _hbad = _v3_cmp_slots(_hmf, _hrows, _hlay2['n_expert'])
            finally:
                globals()['query_sector_alignment_for_path'] = _z7_orig_q
            _hparts = _hmf['layout']['layers'][0]['parts']
            _hp = _hparts[0]
            z7.append(('%s: **h=A-1 정확 경계**(h=%d) 생산→verifier 통과 · aligned · region==slice+A · '
                       'data_offset==slot_offset+h · 인라인 산술 검산 일치' % (_tag, _A7 - 1),
                       _hrep['pass'] is True and not _hbad
                       and _hp.get('aligned') is True and _hp.get('bracket_head') == _A7 - 1
                       and _hp['data_offset'] == _hp['slot_offset'] + (_A7 - 1)
                       and (_hparts[1]['slot_offset'] - _hp['slot_offset']) == _hp['slice_bytes'] + _A7,
                       'h=%r region=%r slice=%r' % (_hp.get('bracket_head'),
                                                    _hparts[1]['slot_offset'] - _hp['slot_offset'],
                                                    _hp['slice_bytes'])))

        # multi-source max(A): 샤드마다 다른 질의값 → A 는 최대값이고 align_query 는 전 샤드 기록.
        _z7_m1 = os.path.join(scratch, 'v3z7split-00001-of-00002.gguf')
        _z7_m2 = os.path.join(scratch, 'v3z7split-00002-of-00002.gguf')
        _z7_mfill = ({'name': 'blk.0.ffn_gate_inp.weight', 'dims': [4096], 'type': 'F32'},
                     {'name': 'blk.1.ffn_gate_inp.weight', 'dims': [4096], 'type': 'F32'})
        write_synthetic_gguf([_z7_m1, _z7_m2], n_expert=8, moe_layers=(0, 1), schema='separate',
                             bias=True, hidden=64, alignment=16384, seed=4142,
                             shard_of={'blk.1.ffn_down_exps.weight': 1, _z7_mfill[1]['name']: 1},
                             extra_tensors=_z7_mfill)
        globals()['query_sector_alignment_for_path'] = _z7_query(
            8192, by_basename={os.path.basename(_z7_m2): 16384})
        try:
            _o7m = os.path.join(scratch, 'out_v3_z7_multi')
            _, _rep7m = do_virtual_plan(_z7_m1, _o7m, profile_id=None, force=False,
                                        enforce_reference=False, allow_default_align=True)
            _mf7m = _load_manifest_disk(_o7m)
            _lay7m, _rows7m = _v3_expected_slots(_z7_m1, 16384)
            _bad7m = _v3_cmp_slots(_mf7m, _rows7m, _lay7m['n_expert'])
        finally:
            globals()['query_sector_alignment_for_path'] = _z7_orig_q
        _phys7 = [q['physical'] for q in _mf7m['layout']['align_query']]
        z7.append(('multi-source **max(A)**: 샤드 질의 %r → A=%d(최대값) · align_query 2건 전부 기록 · '
                   '두 샤드 슬롯 산술 인라인 검산 일치' % (_phys7, _mf7m['layout']['align_bytes']),
                   _rep7m['pass'] is True and _phys7 == [8192, 16384]
                   and _mf7m['layout']['align_bytes'] == 16384 and not _bad7m
                   and len(_mf7m['sources']) == 2,
                   ('불일치=%r' % _bad7m[:3]) if _bad7m else ''))
        ok_z7 = all(ok for _, ok, _ in z7)
        checks.append(('v3-⑲ §Z-⑦ A>4096 합성 주입 E2E %d종 — A=8192·16384 주입으로 aligned/비정렬 혼재 · '
                       '**h=0 과 h=A-1 정확 경계**(+중간 h!=0) · expert 진행 중 head wrap(staging 상한 포함) · '
                       'exact EOF 통과와 bracket EOF 회귀 · multi-source max(A) 를 producer→verifier 로 행사'
                       % len(z7), ok_z7))
        print('[selftest] v3-⑲ A>4096 주입 E2E: %s (host A=%d · 주입 8192/16384)'
              % ('PASS' if ok_z7 else 'FAIL', A_v3))
        for _lbl, _ok_n, _note in z7:
            print('    [%s] %-72s %s' % ('PASS' if _ok_n else 'FAIL', _lbl, _note))

        # ---- v3-⑱ launcher 파서 정규식 **사본 대조**(§2-8 규칙 2 — 기계 대조) ----
        # `LAUNCHER_PLAN_KEYED_LINES` 는 스스로를 "그 함수의 사본"이라 선언하는데, 그 선언을
        # 지키는 장치가 없었다 — ⓑ4~ⓑ6 은 **이 파일의 사본**에 대해서만 stdout 을 주장하므로,
        # launcher 쪽 정규식이 바뀌면 selftest 는 전부 PASS 인 채로 계약만 갈라진다(§2-8 이
        # 말하는 문면 사본 드리프트 그 자체). 원본에서 리터럴을 뽑아 직접 맞춘다.
        # ★r1 [MED] 강화: launcher **부재 = FAIL**(fail-closed). 구판의 "스캔 범위 부재로 생략
        # PASS" 는 파일이 사라지거나 경로가 바뀌는 드리프트를 통째로 놓친다.
        # ★★**F2(r2 회귀 수리 26-08-13)**: 단, 경로를 개발 트리로 **고정**한 것이 실결함이었다 —
        # 배포 번들은 launcher 를 zip 루트에, repacker 를 `repacker/` 하위에 둔다
        # (`packaging/make_bundle.ps1` 의 `Rel 'Start-MoeDirect.ps1'` ↔ `Rel 'repacker/
        # repack_experts.py'`). 그래서 **정상 번들에서 selftest 전체가 FAIL** 했고, 그 selftest 는
        # 릴리스 조립 관문으로 쓰이던 표면이다. 2형상을 순차 해석하고 **둘 다 부재일 때만** FAIL 한다.
        _lp_here = os.path.dirname(os.path.abspath(__file__))
        _lp_cands = [
            # ① 개발 트리: <repo>/bench/repack/…  →  <repo>/bench/moe-direct/launcher/…
            os.path.join(os.path.dirname(os.path.dirname(_lp_here)),
                         'bench', 'moe-direct', 'launcher', 'Start-MoeDirect.ps1'),
            # ② 배포 번들: <root>/repacker/…  →  <root>/Start-MoeDirect.ps1
            os.path.join(os.path.dirname(_lp_here), 'Start-MoeDirect.ps1'),
        ]
        _lp_ps1 = next((p for p in _lp_cands if os.path.isfile(p)), None)
        _lp_bad, _lp_live = [], _lp_ps1 is not None
        if not _lp_live:
            _lp_bad.append('launcher 를 두 형상 어디에서도 찾지 못했다(구조 이상) — 개발 트리=%s · 번들=%s'
                           % (_lp_cands[0], _lp_cands[1]))
        if _lp_live:
            _lp_txt = open(_lp_ps1, 'r', encoding='utf-8', errors='replace').read()
            # ★F2 후반: 실소비 결속을 **파서 함수 블록 + 주석 제외** 범위에서 본다. 파일 전체
            # 문자열 검색이면 주석·dead copy 도 PASS 시켜 "선언만 남고 파서는 다른 걸 본다"는
            # 드리프트를 놓친다(r2 지적). 완전 AST 파싱은 요구 밖 — 오탐만 줄인다.
            _lp_fn0 = _lp_txt.find('function ConvertFrom-TemplatePlanText')
            _lp_fn1 = _lp_txt.find('\nfunction ', _lp_fn0 + 1) if _lp_fn0 >= 0 else -1
            if _lp_fn0 < 0:
                _lp_bad.append('launcher 에 ConvertFrom-TemplatePlanText 함수가 없다(파서 표면 소실)')
                _lp_scope = ''
            else:
                _lp_scope = _lp_txt[_lp_fn0:_lp_fn1 if _lp_fn1 > _lp_fn0 else len(_lp_txt)]
            _lp_scope = '\n'.join(_ln for _ln in _lp_scope.split('\n') if not _ln.strip().startswith('#'))
            _lp_ps, _lp_var = {}, {}
            for _m in re.finditer(r'^\s*\$(re\w+)\s*=\s*\'(.*)\'\s*$', _lp_txt, re.M):
                _lp_ps[_m.group(1)[2:].lower()] = _m.group(2)
                _lp_var[_m.group(1)[2:].lower()] = _m.group(1)
            for _k, _pat in LAUNCHER_PLAN_KEYED_LINES:
                _src = _lp_ps.get(_k)
                if _src is None:
                    _lp_bad.append('launcher 에 $re%s 리터럴이 없다' % _k)
                    continue
                if _src != _pat:
                    _lp_bad.append('%s 정규식 상이:\n        launcher=%r\n        repacker=%r' % (_k, _src, _pat))
                # ★r1 [MED] 두 번째 반쪽: 리터럴이 같아도 그 변수가 **파서 loop 에서 실제로
                # 소비되지 않으면** 계약이 아니다(선언만 남고 파서가 다른 걸 보는 드리프트).
                # 파서는 `@('<key>', $re<Var>)` 쌍으로 순회하므로 그 쌍의 실재를 직접 맞춘다.
                if not re.search(r"@\(\s*'%s'\s*,\s*\$%s\s*\)" % (re.escape(_k), re.escape(_lp_var[_k])), _lp_scope):
                    _lp_bad.append('launcher 파서 함수(주석 제외)가 $%s 를 키 %r 로 소비하지 않는다(선언만 존재)'
                                   % (_lp_var[_k], _k))
            _lp_done = re.search(r'^\s*\$planDone\s*=\s*\'(.*)\'\s*$', _lp_txt, re.M)
            if not _lp_done:
                _lp_bad.append('launcher 에 $planDone 리터럴이 없다')
            elif _lp_done.group(1) != LAUNCHER_PLAN_DONE_LINE:
                _lp_bad.append('완료 줄 상이: launcher=%r repacker=%r'
                               % (_lp_done.group(1), LAUNCHER_PLAN_DONE_LINE))
            _lp_dname = re.search(r'^\s*\$script:DERIVED_EXPECT_FILE_NAME\s*=\s*\'(.*)\'\s*$', _lp_txt, re.M)
            if not _lp_dname:
                _lp_bad.append('launcher 에 $script:DERIVED_EXPECT_FILE_NAME 리터럴이 없다')
            elif _lp_dname.group(1) != DERIVED_EXPECT_FILENAME:
                _lp_bad.append('derived expect 파일명 상이: launcher=%r repacker=%r'
                               % (_lp_dname.group(1), DERIVED_EXPECT_FILENAME))
            for _frag in ("'--- derived expect ('", "', not written in --plan) ---'"):
                if _frag not in _lp_txt:
                    _lp_bad.append('launcher 의 expect 헤더 조각 %s 미검출' % _frag)
        ok_lp = not _lp_bad
        checks.append(('v3-⑱ launcher 파서 계약 사본 대조 — `LAUNCHER_PLAN_KEYED_LINES` 6정규식·완료 줄·'
                       'derived expect 파일명이 `Start-MoeDirect.ps1` 원본 리터럴과 문자 단위 일치 + 그 6변수가 '
                       '파서 함수 블록(주석 제외)에서 **실제 소비**됨 + ★2형상(개발 트리·배포 번들) 해석 후 '
                       '둘 다 부재일 때만 FAIL(fail-closed)', ok_lp))
        print('[selftest] v3-⑱ launcher 정규식 사본 대조: %s (해석 경로=%s)'
              % ('PASS' if ok_lp else 'FAIL', _lp_ps1 if _lp_live else '★둘 다 부재 = FAIL'))
        for _n in _lp_bad:
            print('    %s' % _n)

        # ============ ★계약 등기 제도 1호 시공 (HANDOFF_DEV §2-8 · 26-08-10) ============
        # 병: 같은 계약이 권위 문서·정오·docstring·코드 주석·개수 표기에 **다른 표현으로** 복사돼
        # 있어 개정 시 일부 사본이 낡은 주장을 계속 한다(재발 4회: r9 M2·M7 → r10 Q4·Q7). 텍스트
        # 검색은 ①같은 주장의 다른 표현 ②계약 단어를 안 담는 사본(개수·해시)을 구조적으로 놓친다.
        # 처방 = 불변 태그(원본 = ID 뒤 `|src`, 사본 = 맨 ID) + 아래 두 기계 검사.
        #   ① 고아 태그 0 — 모든 ID 가 `|src` 원본을 **정확히 하나** 가진다.
        #   ② 파생 개수 대조 — 문서에 손으로 적힌 개수를 **실측**과 직접 맞춘다(규칙 2·4).
        # ★스캔 범위는 재팩 쓰기 도메인으로 한정한다(리포 전체 walk 금지 — bench_results 등
        #   대용량 비추적 트리를 훑게 된다). 범위 밖 태그는 이 lint 가 못 보며, 그것이 이 검사의
        #   정직한 한계다(태그를 범위 밖에 두지 않는 것이 규율).
        _reg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _reg_home = os.path.join(_reg_root, 'bench', 'techdev', 'repack_space')
        _reg_live = os.path.isdir(_reg_home)
        _reg_files, _reg_src, _reg_copy, _reg_bad = [], {}, {}, []
        _REG_RE = re.compile(r'\[\[C:([A-Za-z0-9._-]+?)(\|src)?\]\]')
        if _reg_live:
            _reg_files.append(os.path.abspath(__file__))
            for _rd, _rext in ((_reg_home, ('.md',)),
                               (os.path.join(_reg_home, 'ubench'), ('.cpp', '.py', '.md'))):
                if os.path.isdir(_rd):
                    _reg_files.extend(os.path.join(_rd, _fn) for _fn in sorted(os.listdir(_rd))
                                      if _fn.endswith(_rext))
            for _fp in _reg_files:
                try:
                    _rtxt = open(_fp, 'r', encoding='utf-8', errors='replace').read()
                except OSError as _re_err:
                    _reg_bad.append('%s 읽기 실패: %r' % (os.path.basename(_fp), _re_err)); continue
                for _rm in _REG_RE.finditer(_rtxt):
                    (_reg_src if _rm.group(2) else _reg_copy).setdefault(
                        _rm.group(1), []).append(os.path.basename(_fp))
        _reg_ids = sorted(set(_reg_src) | set(_reg_copy))
        if _reg_live:
            for _cid in _reg_ids:
                _n_src = len(_reg_src.get(_cid, []))
                if _n_src != 1:
                    _reg_bad.append('%s: `|src` 원본 %d개(정확히 1개여야 한다) %s'
                                    % (_cid, _n_src, sorted(set(_reg_src.get(_cid, []))) or '— 고아 태그'))
            if not _reg_ids:
                _reg_bad.append('스캔 범위에서 태그를 하나도 찾지 못했다(등기 소실 의심)')
        ok_reg1 = not _reg_bad
        checks.append(('CONTRACT-REGISTRY-① 계약 등기 고아 태그 0 — 스캔 %d파일에서 계약 ID %d종이 각각 '
                       '`|src` 원본을 정확히 1개 가진다(§2-8 규칙 3·5%s)'
                       % (len(_reg_files), len(_reg_ids), '' if _reg_live else ' · ★스캔 범위 부재로 생략'),
                       ok_reg1))
        print('[selftest] CONTRACT-REGISTRY-① 등기: %s%s'
              % ('PASS' if ok_reg1 else 'FAIL', '' if _reg_live else ' (스캔 범위 부재 — 대조 생략)'))
        for _cid in _reg_ids:
            print('    [%s] %-24s 원본 %d(%s) · 사본 %d %s'
                  % ('PASS' if len(_reg_src.get(_cid, [])) == 1 else 'FAIL', _cid,
                     len(_reg_src.get(_cid, [])), ','.join(sorted(set(_reg_src.get(_cid, [])))) or '없음',
                     len(_reg_copy.get(_cid, [])), sorted(set(_reg_copy.get(_cid, [])))))
        if _reg_bad:
            print('    문제: %s' % _reg_bad)

        # ---- ② 파생 개수 사본의 기계 대조: 문서 표기 ↔ 실측(§2-8 규칙 2·4) ----
        # 대조 항목: ⓐwarm obligation 계열 검사 수 = `ubench_io.cpp` 의 `"warm obligation:` 진입
        # 문자열 수(각 check 가 정확히 하나씩 연다) · ⓑv3-⑬ 항수 = 이 런의 `v3_trust` 실 길이 ·
        # ⓓ카탈로그 개수 리터럴 **재도입 금지**(스캔 파일에 수기 개수 문면이 1건이라도 있으면
        # FAIL — 값이 맞아도 실패다. "값이 아직 맞는지"를 보는 대조는 재도입 차단이 아니다).
        # 집합 동등 ⓒ는 문서 트리와 무관하므로 ⓐ 블록 옆에서 무조건 실행한다(여기 아님).
        # ★전부 "산문의 숫자"가 아니라 **실측이 원본**이고 문서가 사본이다 — 그래서 이 검사
        # 자체가 `repack.selftest-counts` 계약의 원본 앵커다(권위 문서의 규범 문장이 아니라
        # 실측이 원본인 유일한 계약 종류: 파생값. §2-8 규칙 4). [[C:repack.selftest-counts|src]]
        _reg2 = []
        if _reg_live:
            _io_cpp = os.path.join(_reg_home, 'ubench', 'ubench_io.cpp')
            _err2_md = os.path.join(_reg_home, 'SPEC_REPACK_V3_ERRATUM2_DRAFT.md')
            _mb_md = os.path.join(_reg_home, 'SPEC_MICROBENCH_V3.md')
            _warm_n = None
            if os.path.isfile(_io_cpp):
                _warm_n = open(_io_cpp, 'r', encoding='utf-8', errors='replace').read().count('"warm obligation:')
            else:
                _reg2.append('ubench_io.cpp 부재 — 실측 원본을 세지 못했다')
            for _dp in (_err2_md, _mb_md):
                if not os.path.isfile(_dp):
                    _reg2.append('%s 부재' % os.path.basename(_dp)); continue
                _dtxt = open(_dp, 'r', encoding='utf-8', errors='replace').read()
                _hits = re.findall(r'warm obligation 계열 \*\*(\d+)건\*\*', _dtxt)
                if not _hits:
                    _reg2.append('%s: 현행 개수 표기 미검출' % os.path.basename(_dp))
                elif _warm_n is not None and any(int(_h) != _warm_n for _h in _hits):
                    _reg2.append('%s: 표기 %s != 실측 %d' % (os.path.basename(_dp), _hits, _warm_n))
            if os.path.isfile(_err2_md):
                _dtxt = open(_err2_md, 'r', encoding='utf-8', errors='replace').read()
                _h13 = re.findall(r'v3-⑬ 현행 \*\*(\d+)항\*\*', _dtxt)
                if not _h13:
                    _reg2.append('정오 2: v3-⑬ 항수 표기 미검출')
                elif any(int(_h) != len(v3_trust) for _h in _h13):
                    _reg2.append('정오 2: v3-⑬ 표기 %s != 실측 %d' % (_h13, len(v3_trust)))
            # ⓓ 카탈로그 개수 리터럴 재도입 금지. 구 문면 4형태를 그대로 잡는다(이번 라운드에서
            # 제거한 것들 + 같은 주장의 다른 표현). 히트가 **1건이라도** 있으면 FAIL 이다.
            _d4_pats = [r'카탈로그\s*\d+\s*종', r'\d+\s*expect 전부 불변',
                        r'카탈로그 항목 수.*기대\s*\d+', r'등록 expect\s*\d+\s*종']
            _d4_hits = []
            for _fp in _reg_files:
                try:
                    _ftxt = open(_fp, 'r', encoding='utf-8', errors='replace').read()
                except OSError:
                    continue
                for _pat in _d4_pats:
                    for _m in re.finditer(_pat, _ftxt):
                        _d4_hits.append('%s: %r' % (os.path.basename(_fp), _m.group(0)[:40]))
            # ★양성 대조 — 대표 문자열을 **런타임 조립**으로 만든다. 소스에 리터럴 테스트 벡터를
            # 두면 위 스캔이 자기 소스를 히트해 가드가 스스로 깨지므로, 조립만이 유일한 방법이다.
            _d4_probe = ['카탈로그 %d종' % catalog_n,
                         '%d expect 전부 불변' % catalog_n,
                         '카탈로그 항목 수=%d(기대 %d)' % (catalog_n, catalog_n),
                         '등록 expect %d종' % catalog_n]
            _d4_blind = [_p for _p, _s in zip(_d4_pats, _d4_probe) if not re.search(_p, _s)]
            if _d4_hits:
                _reg2.append('개수 리터럴 재도입 %d건: %s' % (len(_d4_hits), _d4_hits[:4]))
            if _d4_blind:
                _reg2.append('ⓓ 양성 대조 실패(패턴이 대표 문자열을 못 잡는다): %r' % _d4_blind)
            print('[selftest] CONTRACT-REGISTRY-② 개수 실측: warm obligation=%r · v3-⑬=%d · '
                  'ⓓ재도입 %d건/양성 %d-%d'
                  % (_warm_n, len(v3_trust), len(_d4_hits), len(_d4_probe) - len(_d4_blind), len(_d4_probe)))
        ok_reg2 = not _reg2
        checks.append(('CONTRACT-REGISTRY-② 파생 개수 사본 기계 대조 — 정오 2·7 의 "warm obligation 계열 N건"이 '
                       'ubench_io.cpp 실측과, 정오 2 의 "v3-⑬ 현행 N항"이 이 런의 v3_trust 길이와 일치 + '
                       'ⓓ카탈로그 개수 리터럴 재도입 0건(패턴 4종·양성 대조 통과)'
                       '(§2-8 규칙 2·4 파생값 수기 금지%s)' % ('' if _reg_live else ' · ★스캔 범위 부재로 생략'),
                       ok_reg2))
        print('[selftest] CONTRACT-REGISTRY-② 대조: %s %s'
              % ('PASS' if ok_reg2 else 'FAIL', _reg2 if _reg2 else ''))

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
    # ★r9 M8 / 부속 정오 2-ⓓ: 구 도움말은 bin 전용 문면(experts.bin + verify_report.json)뿐이라
    # 승인된 `--verify-only --mode virtual` 경로를 문서화하지 않았다.
    mode_group.add_argument('--verify-only', action='store_true', dest='verify_only',
                             help='fully re-verify an existing artifact once without repacking (--profile/--model/--out are required; artifact bytes are unchanged; scope comes from the manifest, so --scope is ignored). mode bin: re-verifies experts.bin + manifest.json and appends a record to verify_report.json. mode virtual: re-runs the independent verifier on manifest.json and atomically replaces plan_report.json (the manifest bytes are unchanged; full-file SHA provenance is inherited when the source identity is unchanged, otherwise an explicit downgrade is recorded).')
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
    # SPEC_REPACK_V3 section 2-1: bin (default) keeps the v2 byte contract exactly (schema "2.0", no
    # mode field); virtual emits schema "3.0" + mode "virtual" and moves 0 bytes of expert data.
    ap.add_argument('--mode', type=str, default=MODE_BIN, choices=[MODE_BIN, MODE_VIRTUAL],
                     help='output mode. bin=legacy v2 (experts.bin + manifest v2, byte contract unchanged; default) / '
                          'virtual=manifest v3 + plan_report only, consumers read the original GGUF in place '
                          '(no experts.bin, disk footprint exactly 1.0x).')
    ap.add_argument('--source-full-sha', action='store_true', dest='source_full_sha',
                     help='mode=virtual only: also record a full-file SHA-256 of every source shard in '
                          'plan_report.json as optional provenance (reads every source byte; the manifest always '
                          'carries the cheap header-region digest).')
    # SPEC_IO_METRICS_V3 section 7 [[C:repack.legacy-align]]: the D-A2 numerator needs a legacy
    # stride baseline, and the source-volume A is not it (a v2 repack wrote to an output volume
    # whose alignment need not match). Unset = canonical 4096 + the section 6-5 rebaselining footnote.
    ap.add_argument('--legacy-v2-manifest', type=str, default=None, dest='legacy_v2_manifest',
                     help='mode=virtual only: path to this model\'s existing v2 (mode=bin) manifest.json. On a real '
                          'run it binds layout.legacy_align_bytes to that artifact\'s real alignment '
                          '(legacy_align_source "paired_v2") and records its SHA-256 in reference_lock; the recorded '
                          'path is only a non-authoritative locator, the SHA is the identity authority. On '
                          '--verify-only it overrides that locator (use it when the paired v2 manifest has moved); '
                          'the override is accepted only if the recorded SHA, schema 2.0 and the model/sources/'
                          'reference identity all still match. Unset on a real run = legacy_align_source '
                          '"canonical_4096" (value 4096).')
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
    if getattr(args, 'mode', MODE_BIN) == MODE_BIN and getattr(args, 'source_full_sha', False):
        parser.error('--source-full-sha applies to --mode virtual only (mode=bin proves losslessness by re-reading '
                     'and hashing every record pair).')
    if getattr(args, 'mode', MODE_BIN) == MODE_BIN and getattr(args, 'legacy_v2_manifest', None):
        parser.error('--legacy-v2-manifest applies to --mode virtual only (a mode=bin run *is* the v2 baseline - its '
                     'layout.align_bytes is the value a later virtual run binds to).')
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
