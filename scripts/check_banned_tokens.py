#!/usr/bin/env python3
"""check_banned_tokens.py — CLAUDE.md 규칙 1 금지 토큰 검사.

CLAUDE.md 4절의 `grep -nE ... | grep -v '^\\s*//'` 는 블록 주석(/* */)과
줄 끝 주석을 걸러내지 못한다. 이 스크립트는 주석과 문자열 리터럴을 공백으로
치환한 뒤 검사하므로 오탐이 없다.

사용:  python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v
종료:  hit 있으면 1, 없으면 0
"""
import re, sys, glob, os

PATTERNS = {
    'real':   re.compile(r'(?<![A-Za-z0-9_$])real(?![A-Za-z0-9_])'),
    '$itor':  re.compile(r'\$itor\b'),
    '$rtoi':  re.compile(r'\$rtoi\b'),
    '$exp':   re.compile(r'\$exp\b'),
    '$sqrt':  re.compile(r'\$sqrt\b'),
    '#delay': re.compile(r'#\s*[0-9]'),
}

def strip(src):
    """Replace comment/string chars with spaces, preserving line structure."""
    out = list(src)
    i, n = 0, len(src)
    st = 'code'
    while i < n:
        c = src[i]
        nx = src[i+1] if i+1 < n else ''
        if st == 'code':
            if c == '/' and nx == '/':
                st = 'line'; out[i] = out[i+1] = ' '; i += 2; continue
            if c == '/' and nx == '*':
                st = 'block'; out[i] = out[i+1] = ' '; i += 2; continue
            if c == '"':
                st = 'str'; out[i] = ' '; i += 1; continue
        elif st == 'line':
            if c == '\n': st = 'code'
            else: out[i] = ' '
        elif st == 'block':
            if c == '*' and nx == '/':
                st = 'code'; out[i] = out[i+1] = ' '; i += 2; continue
            if c != '\n': out[i] = ' '
        elif st == 'str':
            if c == '\\': out[i] = ' '; out[i+1] = ' '; i += 2; continue
            if c == '"': st = 'code'
            out[i] = ' '
        i += 1
    return ''.join(out)

files = sorted(sys.argv[1:])
any_hit = False
summary = {}
for f in files:
    src = open(f, errors='replace').read()
    code = strip(src)
    lines = code.split('\n')
    raw = src.split('\n')
    hits = []
    for ln, text in enumerate(lines, 1):
        for name, pat in PATTERNS.items():
            for m in pat.finditer(text):
                hits.append((ln, name, raw[ln-1].strip()[:100]))
    if hits:
        any_hit = True
        summary[f] = hits
        for ln, name, txt in hits:
            print(f"{f}:{ln}: [{name}] {txt}")
print()
print("=== FILES WITH HITS (%d) ===" % len(summary))
for f in sorted(summary):
    kinds = sorted({k for _, k, _ in summary[f]})
    print(f"  {os.path.basename(f)}  ({len(summary[f])} hits: {', '.join(kinds)})")
if not any_hit:
    print("clean")
sys.exit(1 if any_hit else 0)
