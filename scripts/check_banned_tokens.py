#!/usr/bin/env python3
"""check_banned_tokens.py — CLAUDE.md 규칙 1 금지 토큰 검사.

왜 grep 이 아닌가
-----------------
CLAUDE.md 4절의 `grep -nE ... | grep -v '^\\s*//'` 는 블록 주석(/* */)과 줄 끝 주석을
걸러내지 못한다. 이 스크립트는 주석과 문자열 리터럴을 공백으로 치환한 뒤 검사하므로
오탐이 없다. 그리고 `ifdef COCOTB_SIM` 영역을 인식한다.

두 등급
-------
HARD  : 어디에 있든 금지. 시뮬레이션에서도 합성에서도 하드웨어가 아니다.
        real, $itor, $rtoi, $exp, $sqrt, #delay
SIMONLY: 합성 대상에서는 금지하되, `ifdef COCOTB_SIM` 안에서는 허용.
        시뮬레이션에서는 의미가 있지만 합성에서는 조용히 다른 회로가 되는 것들.
          ===, !==  X 비교. 합성기가 조건을 상수로 접는다.
                    docs/BUGS.md BUG-006: 이것 때문에 wgt_sram 이 통째로 사라졌다
                    (0 cells vs 99,294 cells).
          while     경계 없는 루프. 하드웨어가 아니다.
                    docs/BUGS.md BUG-003: yosys 가 함수 호출 자체를 거부했다.

`tb/` 는 이 검사 대상이 아니다 (테스트벤치는 합성하지 않는다).
scripts/synth_gate.sh 는 rtl/*.sv, rtl/*.v 만 넘긴다.

사용:  python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v
종료:  위반이 있으면 1, 없으면 0
"""
import os
import re
import sys

HARD = {
    'real':   re.compile(r'(?<![A-Za-z0-9_$])real(?![A-Za-z0-9_])'),
    '$itor':  re.compile(r'\$itor\b'),
    '$rtoi':  re.compile(r'\$rtoi\b'),
    '$exp':   re.compile(r'\$exp\b'),
    '$sqrt':  re.compile(r'\$sqrt\b'),
    '#delay': re.compile(r'#\s*[0-9]'),
}

SIMONLY = {
    '===':   re.compile(r'==='),
    '!==':   re.compile(r'!=='),
    'while': re.compile(r'(?<![A-Za-z0-9_$])while(?![A-Za-z0-9_])'),
}

SIM_MACRO = 'COCOTB_SIM'


def strip(src):
    """주석과 문자열 리터럴을 공백으로 치환한다. 줄 구조는 보존한다."""
    out = list(src)
    i, n = 0, len(src)
    st = 'code'
    while i < n:
        c = src[i]
        nx = src[i + 1] if i + 1 < n else ''
        if st == 'code':
            if c == '/' and nx == '/':
                st = 'line'; out[i] = out[i + 1] = ' '; i += 2; continue
            if c == '/' and nx == '*':
                st = 'block'; out[i] = out[i + 1] = ' '; i += 2; continue
            if c == '"':
                st = 'str'; out[i] = ' '; i += 1; continue
        elif st == 'line':
            if c == '\n':
                st = 'code'
            else:
                out[i] = ' '
        elif st == 'block':
            if c == '*' and nx == '/':
                st = 'code'; out[i] = out[i + 1] = ' '; i += 2; continue
            if c != '\n':
                out[i] = ' '
        elif st == 'str':
            if c == '\\':
                out[i] = ' '
                if i + 1 < n:
                    out[i + 1] = ' '
                i += 2
                continue
            if c == '"':
                st = 'code'
            out[i] = ' '
        i += 1
    return ''.join(out)


IFDEF = re.compile(r'^\s*`(ifdef|ifndef|elsif|else|endif)\b\s*([A-Za-z_][A-Za-z_0-9]*)?')


def sim_only_lines(lines):
    """각 줄이 `ifdef COCOTB_SIM` 로 감싸인 시뮬 전용 영역인지 판정한다.

    스택의 각 항목은 (이 영역이 COCOTB_SIM 전용인가, 이 `ifdef 가 COCOTB_SIM 을 봤는가).
    중첩을 지원한다. `else / `elsif 로 분기가 뒤집히는 것도 반영한다.
    """
    flags = []
    stack = []   # [is_sim_branch]
    saw_macro = []  # 이 ifdef 블록이 COCOTB_SIM 을 대상으로 하는가
    for ln in lines:
        m = IFDEF.match(ln)
        if m:
            kind, macro = m.group(1), m.group(2)
            if kind == 'ifdef':
                stack.append(macro == SIM_MACRO)
                saw_macro.append(macro == SIM_MACRO)
            elif kind == 'ifndef':
                stack.append(False)
                saw_macro.append(macro == SIM_MACRO)
            elif kind == 'elsif':
                if stack:
                    stack[-1] = (macro == SIM_MACRO)
                    saw_macro[-1] = saw_macro[-1] or (macro == SIM_MACRO)
            elif kind == 'else':
                if stack:
                    # ifndef COCOTB_SIM 의 else 는 시뮬 전용 분기다
                    stack[-1] = saw_macro[-1] and not stack[-1]
            elif kind == 'endif':
                if stack:
                    stack.pop(); saw_macro.pop()
            flags.append(any(stack))   # 지시어 줄 자체는 검사 대상이 아니다
            continue
        flags.append(any(stack))
    return flags


def scan(path):
    src = open(path, errors='replace').read()
    code = strip(src).split('\n')
    raw = src.split('\n')
    sim = sim_only_lines(code)
    hits = []
    for i, text in enumerate(code):
        is_sim = sim[i]
        for name, pat in HARD.items():
            if pat.search(text):
                hits.append((i + 1, name, 'HARD', raw[i].strip()[:100]))
        if not is_sim:
            for name, pat in SIMONLY.items():
                if pat.search(text):
                    hits.append((i + 1, name, 'SIMONLY', raw[i].strip()[:100]))
    return hits


def main(argv):
    files = sorted(argv)
    summary = {}
    for f in files:
        h = scan(f)
        if h:
            summary[f] = h
            for ln, name, grade, txt in h:
                extra = ''
                if grade == 'SIMONLY':
                    extra = '  <- `ifdef COCOTB_SIM 안으로 옮기거나 합성 가능한 형태로 바꿀 것'
                print(f"{f}:{ln}: [{name}] {txt}{extra}")
    print()
    print("=== FILES WITH HITS (%d) ===" % len(summary))
    for f in sorted(summary):
        kinds = sorted({k for _, k, _, _ in summary[f]})
        print(f"  {os.path.basename(f)}  ({len(summary[f])} hits: {', '.join(kinds)})")
    if not summary:
        print("clean")
    return 1 if summary else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
