#!/usr/bin/env bash
# =============================================================================
# setup_tools.sh — ORBIT 툴체인 설치 (Ubuntu 24.04 noble 기준)
#
# CLAUDE.md 규칙 4: 툴체인은 verilator, iverilog, yosys, cocotb, Vivado(무료)로
# 한정한다. 이 스크립트가 설치하는 것은 전부 무료·오픈소스다.
#
# 사용:
#   bash scripts/setup_tools.sh            # 전부
#   bash scripts/setup_tools.sh --check    # 설치 상태만 확인 (아무것도 설치 안 함)
#   bash scripts/setup_tools.sh yosys sv2v # 골라서
#
# 왜 verilator 를 소스에서 빌드하는가
# -----------------------------------
#   apt 의 verilator 5.020 (Debian/Ubuntu noble) 은 cocotb 2.x 가 요구하는
#   VerilatedVpi API 가 없다:
#       error: 'clearEvalNeeded' is not a member of 'VerilatedVpi'
#       error: 'doInertialPuts' is not a member of 'VerilatedVpi'
#       error: 'evalNeeded' is not a member of 'VerilatedVpi'
#   해당 API 는 verilator 5.022 에서 들어왔다. 그래서 cocotb 시뮬레이션을 돌리려면
#   5.022 이상이 필요하고, noble 에는 그 패키지가 없다.
#
#   **린트(`--lint-only`)는 5.020 으로도 된다** — VPI 와 무관하다.
#   즉 PLAN W1-3(린트 기록)은 이 빌드를 기다릴 필요가 없었다.
#
# 왜 sv2v 가 필요한가
# -------------------
#   이 레포 RTL 다수가 unpacked array 포트를 쓴다. 합법 SystemVerilog 이고
#   Vivado 는 그대로 합성하지만, yosys 내장 프론트엔드(0.33·0.69)와 iverilog 12 는
#   파싱하지 못한다. 그래서 sv2v 로 Verilog-2005 로 낮춘 뒤 먹인다.
#   RTL 을 도구 취향에 맞춰 리팩터하지 않는다. 근거: docs/AUDIT.md §2.
# =============================================================================
set -u -o pipefail

VERILATOR_TAG="${VERILATOR_TAG:-v5.034}"   # cocotb 2.x 호환 (>= 5.022)
SV2V_VER="${SV2V_VER:-v0.0.13}"
WORK="${WORK:-/tmp/orbit-tools}"

have() { command -v "$1" >/dev/null 2>&1; }
say()  { echo "[setup_tools] $*"; }

check() {
  echo "=== 설치 상태 ==================================================="
  # NOTE: `iverilog -V` 는 종료 코드 1 을 낸다. set -o pipefail 때문에 파이프 전체가
  # 1 이 되어 `|| echo 없음` 이 잘못 발동한다. 그래서 버전 출력은 서브셸에 담고
  # have 로만 존재를 판정한다.
  printf "%-12s " "yosys";     if have yosys;    then (yosys -V 2>&1 | head -1) || true; else echo "없음"; fi
  printf "%-12s " "iverilog";  if have iverilog; then (iverilog -V 2>&1 | head -1) || true; else echo "없음"; fi
  printf "%-12s " "sv2v";      if have sv2v;     then (sv2v --version) || true;           else echo "없음"; fi
  printf "%-12s " "verilator"; if have verilator; then
      v=$(verilator --version 2>&1 | head -1)
      maj=$(echo "$v" | sed -nE 's/^Verilator ([0-9]+)\.([0-9]+).*/\1\2/p')
      if [ -n "$maj" ] && [ "$maj" -ge 5022 ]; then echo "$v   (cocotb 2.x OK)"
      else echo "$v   <- cocotb 2.x 비호환. 5.022+ 필요"; fi
    else echo "없음"; fi
  printf "%-12s " "cocotb";    python3 -c "import cocotb;print(cocotb.__version__)" 2>/dev/null || echo "없음"
  printf "%-12s " "cocotb-test"; python3 -c "import cocotb_test;print('ok')" 2>/dev/null || echo "없음"
  printf "%-12s " "pytest";    python3 -c "import pytest;print(pytest.__version__)" 2>/dev/null || echo "없음"
  printf "%-12s " "numpy";     python3 -c "import numpy;print(numpy.__version__)" 2>/dev/null || echo "없음"
  echo "================================================================"
}

install_apt_base() {
  say "apt 패키지 (yosys, iverilog, 빌드 도구)"
  apt-get update -qq
  apt-get install -y yosys iverilog \
    git perl make autoconf g++ flex bison libfl2 libfl-dev zlib1g-dev \
    help2man ccache
}

install_sv2v() {
  say "sv2v $SV2V_VER (단일 바이너리, MIT)"
  mkdir -p "$WORK" && cd "$WORK"
  curl -sSL -o sv2v.zip \
    "https://github.com/zachjs/sv2v/releases/download/$SV2V_VER/sv2v-Linux.zip"
  unzip -oq sv2v.zip
  install -m 0755 "$(find . -name sv2v -type f | head -1)" /usr/local/bin/sv2v
  sv2v --version
}

install_verilator() {
  say "verilator $VERILATOR_TAG 소스 빌드 (apt 5.020 은 cocotb 2.x 비호환)"
  say "  15~30분 걸린다. 코어 수: $(nproc)"
  mkdir -p "$WORK" && cd "$WORK"
  rm -rf verilator
  git clone --depth 1 --branch "$VERILATOR_TAG" https://github.com/verilator/verilator.git
  cd verilator
  autoconf
  ./configure --prefix=/usr/local
  make -j"$(nproc)"
  make install
  hash -r
  verilator --version
}

install_python() {
  say "python 의존성 (cocotb 2.x, pytest, numpy)"
  pip install -q cocotb pytest

  # numpy 함정: apt 의 python3-numpy 가 /usr/lib/python3/dist-packages 에 들어가는데
  # (yosys -> xdot -> graphviz 의존으로 딸려 온다) 그 빌드는 /usr/local/bin/python3 에서
  # import 되지 않는다:
  #     ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
  # pip 로 덮어쓰려 해도 "Cannot uninstall numpy, RECORD file not found (installed by debian)"
  # 로 막힌다. --ignore-installed 로 /usr/local/lib/python3.11/dist-packages 에 새로 깔면
  # sys.path 순서상 그쪽이 먼저 잡혀서 해결된다.
  # 1단계(골든 모델) 전체가 numpy 에 의존하므로 이건 선택이 아니다.
  say "numpy (--ignore-installed — apt python3-numpy 우회)"
  pip install -q --ignore-installed --no-cache-dir "numpy>=1.26"
  python3 -c "import numpy; print('[setup_tools] numpy', numpy.__version__, 'at', numpy.__file__)"

  # cocotb-test 는 이 환경에서 wheel 빌드가 실패한다 (ERROR: Failed building wheel).
  # 다행히 필수가 아니다 — tb/run_tb.py 는 cocotb 2.x 에 들어 있는
  # cocotb_tools.runner 를 쓴다. 레거시 sim/cocotb/run_*.py 만 cocotb_test 를 참조하므로
  # 그 러너들을 cocotb_tools.runner 로 옮기는 것이 남은 일이다 (docs/LOG.md).
  pip install -q cocotb-test 2>/dev/null || say "cocotb-test 설치 실패 (선택 사항, tb/run_tb.py 는 불필요)"
}

main() {
  if [ "${1:-}" = "--check" ]; then check; exit 0; fi
  targets=("$@")
  [ "${#targets[@]}" -eq 0 ] && targets=(apt sv2v python verilator)
  for t in "${targets[@]}"; do
    case "$t" in
      apt|base)  install_apt_base ;;
      sv2v)      install_sv2v ;;
      verilator) install_verilator ;;
      python|py) install_python ;;
      yosys)     apt-get update -qq && apt-get install -y yosys ;;
      iverilog)  apt-get update -qq && apt-get install -y iverilog ;;
      *) echo "알 수 없는 대상: $t"; exit 2 ;;
    esac
  done
  echo ""
  check
  echo ""
  say "다음: bash scripts/synth_gate.sh; echo \$?"
}

main "$@"
