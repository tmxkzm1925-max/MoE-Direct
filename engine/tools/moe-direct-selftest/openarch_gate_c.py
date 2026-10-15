# -*- coding: utf-8 -*-
"""OPEN_ARCH B축 Gate C/G 러너 — 크로스랭귀지 inventory digest 실행 계약.

권위: OPENARCH_B_SPEC_DRAFT.md v0.2 §3-C(6단계 동결) · §3-G(표 동기화 = 차단 게이트).

  ① A축의 기존 합성 GPTOSS 형상(repack_experts.write_synthetic_gguf)으로 GGUF 생성 →
     실제 `--experimental-arch-template` 재팩 실행(do_repack(arch_template=True))
  ② 산출 디렉토리를 `moe-direct-selftest.exe --openarch-python-artifact <dir>` 로 전달
  ③ C++ 은 Python 의 inventory 행을 읽지 않고 arch 공식이 생성한 이름 + live 텐서의
     type/dims 로 digest 를 독립 계산
  ④ `<dir>\\derived.expect.json` 의 inventory_sha256 과 byte-equal 대조 후 실제 template
     seal 까지 실행
  ⑤ type 문자열은 enum→QUANT_TRAITS 동결 wire명 매핑만 사용(ggml_type_name() 금지)
  ⑥ 같은 러너가 `--openarch-probe-json` 출력의 **catalog 행만** A축 ARCH_TEMPLATES 와
     대조 = Gate G (ABI 는 probe 에 노출만 하고 축간 대조는 M5 조립기 몫이므로 비교 대상 아님)

★쓰기 경계: 이 러너는 메인 리포(bench/)를 **읽기만** 한다. 재팩 산출물·재팩 로그는 전부
  --scratch 아래(기본 = 이 엔진 사본의 임시 디렉토리)에 만든다.

사용:
  python openarch_gate_c.py --selftest-exe <...\\moe-direct-selftest.exe>
                            --repo <repo root: dev tree(bench/repack/) 또는 공개 checkout(repacker/)>
                            [--scratch <dir>] [--keep]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

MARKER_OK = 'OPENARCH_GATE_C_PASS'
MARKER_FAIL = 'OPENARCH_GATE_C_FAIL'


def load_repacker(repo_root):
    # Two layouts are accepted: the development repository keeps the script under
    # bench/repack/, the public repository ships it under repacker/. First match wins.
    candidates = [os.path.join(repo_root, 'bench', 'repack', 'repack_experts.py'),
                  os.path.join(repo_root, 'repacker', 'repack_experts.py')]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise SystemExit('repack_experts.py not found under --repo; tried: %s'
                         % '; '.join(candidates))
    spec = importlib.util.spec_from_file_location('repack_experts_gatec', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['repack_experts_gatec'] = mod
    # r1 fix: importing a source module may write bytecode next to it; the main repo is a
    # read-only input for this runner, so bytecode writing is suppressed for the import and the
    # previous setting is restored afterwards.
    prev_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev_dwb
    return mod, path


def gate_g(selftest_exe, R, results):
    """§3-C⑥ / §3-G — probe 의 catalog 행 == A축 ARCH_TEMPLATES(표 동기화·차단 게이트)."""
    out = subprocess.run([selftest_exe, '--openarch-probe-json'],
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    if out.returncode != 0:
        results.append(('Gate G: probe 실행', False, 'exit=%d %s' % (out.returncode, out.stderr[-300:])))
        return
    try:
        probe = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        results.append(('Gate G: probe JSON 파싱', False, repr(e)))
        return

    print('[gate-c] probe abi      = %s   (★ABI 는 Gate G 비교 대상 아님 — M5 조립기 소유)' % probe.get('abi'))
    rows = {t['template_id']: t for t in probe.get('templates', [])}
    py = R.ARCH_TEMPLATES
    ok = set(rows) == set(py)
    detail = [] if ok else ['id 집합 불일치: engine=%r python=%r' % (sorted(rows), sorted(py))]
    for tid in sorted(set(rows) & set(py)):
        e, p = rows[tid], py[tid]
        for field, pv in (('template_version', p['version']),
                          ('weight_kinds', list(p['weight_kinds'])),
                          ('bias', p['bias']),
                          ('layer_rule', p['layer_rule']),
                          ('nextn', p['nextn'])):
            ev = e.get(field)
            if ev != pv:
                ok = False
                detail.append('%s.%s engine=%r python=%r' % (tid, field, ev, pv))
    for tid in sorted(rows):
        print('[gate-c] catalog row    = %-10s v%-3s bias=%-9s layer_rule=%-13s nextn=%s'
              % (tid, rows[tid].get('template_version'), rows[tid].get('bias'),
                 rows[tid].get('layer_rule'), rows[tid].get('nextn')))
    results.append(('Gate G: probe catalog 3행 == A축 ARCH_TEMPLATES(표 동기화)', ok, '; '.join(detail)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest-exe', required=True)
    ap.add_argument('--repo', required=True,
                    help='repository root carrying repack_experts.py - dev tree (bench/repack/) '
                         'or public checkout (repacker/); no default, machine specific')
    ap.add_argument('--scratch', default=None)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()

    if not os.path.isfile(args.selftest_exe):
        raise SystemExit('selftest exe not found: %s' % args.selftest_exe)

    R, repacker_path = load_repacker(args.repo)
    print('[gate-c] repacker(read-only) = %s' % repacker_path)
    print('[gate-c] selftest exe        = %s' % args.selftest_exe)

    scratch = args.scratch or tempfile.mkdtemp(prefix='openarch_gate_c_')
    os.makedirs(scratch, exist_ok=True)
    # ★append-only 재팩 로그가 메인 리포 안에 생기지 않게 스크래치로 돌린다(쓰기 경계).
    R._set_repack_log_path(os.path.join(scratch, 'repack_log.jsonl'))
    print('[gate-c] scratch             = %s' % scratch)

    results = []
    try:
        # ---- ① A축 합성 GPTOSS 형상 + 실제 template 재팩 ----
        GPTOSS = dict(arch='gpt-oss', n_expert=4, n_expert_used=2, moe_layers=(0, 1, 2), block_count=3,
                      schema='separate', bias=True, hidden=8, alignment=32, seed=3101)
        model_path = os.path.join(scratch, 'tpl_gptoss.gguf')
        R.write_synthetic_gguf([model_path], **GPTOSS)
        out_dir = os.path.join(scratch, 'out_tpl_gptoss')
        _, manifest, vr = R.do_repack(model_path, out_dir, profile_id=None, force=True, run_verify=True,
                                      enforce_reference=False, allow_default_align=True, arch_template=True)
        expect_path = os.path.join(out_dir, R.DERIVED_EXPECT_FILENAME)
        raw = open(expect_path, 'rb').read()
        expect = json.loads(raw.decode('utf-8'))
        lock = manifest['reference_lock']
        results.append(('①  template 재팩 산출(verify PASS · lock=arch-template · derived expect 기록)',
                        bool(vr and vr['pass']) and lock['profile_id'] == 'arch-template:gpt-oss@1'
                        and lock['expect_sha256'] == hashlib.sha256(raw).hexdigest()
                        and len(expect.get('inventory_sha256', '')) == 64, ''))
        print('[gate-c] python digest   = %s' % expect['inventory_sha256'])

        # ---- ②~⑤ 엔진 소비 — 독립 digest 재계산 + byte-equal 대조 + 실제 template seal ----
        proc = subprocess.run([args.selftest_exe, '--openarch-python-artifact', out_dir],
                              capture_output=True, text=True, encoding='utf-8', errors='replace')
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
        engine_digest = ''
        byte_equal = False
        seal_rc = None
        for line in proc.stdout.splitlines():
            if line.startswith('GATEC engine  digest'):
                engine_digest = line.split('=', 1)[1].strip()
            elif line.startswith('GATEC digest byte-equal'):
                byte_equal = line.split('=', 1)[1].strip() == 'YES'
            elif line.startswith('GATEC seal rc'):
                seal_rc = int(line.split('=', 1)[1].strip())
        results.append(('②③④ 엔진 독립 재계산 digest == python derived.expect.json(byte-equal)',
                        byte_equal and engine_digest == expect['inventory_sha256'], engine_digest))
        results.append(('④  같은 산출물에서 실제 template seal 성공(rc=0)', seal_rc == 0, 'rc=%r' % seal_rc))

        # ---- ⑥ Gate G ----
        gate_g(args.selftest_exe, R, results)
    finally:
        if not args.keep and args.scratch is None:
            shutil.rmtree(scratch, ignore_errors=True)

    print('\n==== OPEN_ARCH Gate C/G ====')
    failed = 0
    for name, ok, detail in results:
        print('[%s] %s%s' % ('PASS' if ok else 'FAIL', name, ('  -- ' + detail) if detail else ''))
        if not ok:
            failed += 1
    print('%d/%d PASS' % (len(results) - failed, len(results)))
    print(MARKER_OK if failed == 0 else MARKER_FAIL)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
