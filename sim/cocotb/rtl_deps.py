"""rtl_deps.py — top 모듈 하나에서 RTL 의존 파일 전체를 자동으로 모은다.

러너마다 소스 목록을 손으로 적으면 모듈이 바뀔 때마다 어긋난다.
여기서는 rtl/ 을 한 번 훑어 모듈→파일 표를 만들고, top 에서 시작해
인스턴스화된 모듈을 따라가며 닫힘(closure)을 구한다.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "rtl"

_COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)


def _strip(txt: str) -> str:
    return _COMMENT.sub("", txt)


def module_table() -> dict[str, Path]:
    """모듈 이름 → 정의 파일."""
    table: dict[str, Path] = {}
    for f in sorted(RTL.glob("*.sv")) + sorted(RTL.glob("*.v")):
        for m in re.finditer(r"^\s*module\s+([A-Za-z_]\w*)", _strip(f.read_text()), re.M):
            table.setdefault(m.group(1), f)
    return table


def packages() -> list[Path]:
    """패키지 파일은 항상 먼저 컴파일해야 한다."""
    out = []
    for f in sorted(RTL.glob("*.sv")):
        if re.search(r"^\s*package\s+\w+", _strip(f.read_text()), re.M):
            out.append(f)
    return out


def deps(top: str) -> list[Path]:
    """top 모듈이 필요로 하는 파일 목록 (패키지 먼저, 그다음 의존 순)."""
    table = module_table()
    if top not in table:
        raise KeyError(f"모듈 '{top}' 를 rtl/ 에서 찾지 못했습니다")

    seen: set[str] = set()
    files: list[Path] = []

    def walk(name: str) -> None:
        if name in seen or name not in table:
            return
        seen.add(name)
        f = table[name]
        txt = _strip(f.read_text())
        for other in table:
            if other == name or other in seen:
                continue
            # "mod #(...) u_x (" 또는 "mod u_x ("
            if re.search(rf"\b{other}\s*(#\s*\([^;]*?\))?\s+[A-Za-z_]\w*\s*\(", txt):
                walk(other)
        if f not in files:
            files.append(f)

    walk(top)
    pkgs = [p for p in packages() if p not in files]
    return pkgs + files


if __name__ == "__main__":
    import sys
    for p in deps(sys.argv[1]):
        print(p.relative_to(ROOT))
