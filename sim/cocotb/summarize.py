"""results.xml 들을 모아 테스트 단위로 집계한다."""
import json, sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rep = json.loads((ROOT/"sim_build"/"run_all_report.json").read_text())

tot = {"pass": 0, "fail": 0}
rows = []
for r in rep:
    tb, status = r["tb"], r["status"]
    x = ROOT/"sim_build"/f"all_{tb}"/"results.xml"
    p = f = 0
    detail = r.get("err", "")
    if x.exists():
        try:
            root = ET.parse(x).getroot()
            for tc in root.iter("testcase"):
                bad = list(tc.iter("failure")) + list(tc.iter("error"))
                if bad: f += 1
                else:   p += 1
        except Exception as e:
            detail = f"xml 파싱 실패: {e}"
    tot["pass"] += p; tot["fail"] += f
    rows.append((tb, r.get("top"), p, f, status, detail))

print(f"{'테스트벤치':<36}{'top':<26}{'통과':>5}{'실패':>5}  비고")
print("─"*96)
for tb, top, p, f, st, d in rows:
    note = "" if st == "RAN" and f == 0 else (d[:34] if d else ("실패 있음" if f else st))
    print(f"{tb:<36}{str(top):<26}{p:>5}{f:>5}  {note}")
print("─"*96)
print(f"{'합계':<62}{tot['pass']:>5}{tot['fail']:>5}")
built = sum(1 for r in rows if r[2] or r[3])
print(f"\n실행된 테스트벤치 {built}/{len(rows)}   총 테스트 {tot['pass']+tot['fail']}개 "
      f"(통과 {tot['pass']}, 실패 {tot['fail']})")
