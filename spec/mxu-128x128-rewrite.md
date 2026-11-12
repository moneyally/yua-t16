# mxu_bf16_128x128 재작성 — 정확성 수정과 병렬화

## 요약

`mxu_bf16_128x128` 은 K>1 인 모든 행렬곱에서 **틀린 값**을 냈다. 기존 테스트
5개가 전부 `k_steps=1` 이라 한 번도 걸리지 않았다. 동시에 16×16 타일 하나를
64개 논리 위치에 재사용하느라 K-step 당 약 320 사이클이 들었다.

외부 인터페이스는 바꾸지 않았다. 드롭인 교체된다.

## 1. 정확성 버그

원본 `S_STORE_TILE`:

```systemverilog
tile_acc[tr][tc][i][j] <= tile_out[i][j] + tile_acc[tr][tc][i][j];
// Note: this is integer add on FP32 bits — NOT correct FP32 add!
```

FP32 비트패턴을 정수로 더한다. 원본 주석도 이를 인정하고 `TODO: proper FP32
partial sum add` 를 남겨 두었다.

왜 생겼나: `fp32_add` 와 `bf16_mul` 이 `mxu_bf16_16x16.sv` **모듈 내부**에만
정의돼 있었다. 128×128 래퍼는 접근할 수 없어 누적을 직접 구현했다.

Verilator 로 측정한 실제 오차 (기대값 대비):

| 테스트 | 기대 | 원본 RTL 실제 |
|---|---|---|
| K=2 누적 | 2.0 | 1.70e+38 |
| K=4 혼합값 | 9.0 | 5.88e-39 |
| K=3 타일 경계 | 6.0 | -1.5 |
| K=8 누적 | 8.0 | -2.66e+36 |

반올림 오차가 아니라 부호까지 뒤집힌다.

## 2. 수정

### 2-1. 산술을 패키지로 분리 — `rtl/bf16_pkg.sv`

`bf16_mul`, `fp32_add` 를 패키지로 옮겨 16×16 과 128×128 이 같은 함수를 쓴다.
옮기면서 원본 `fp32_add` 의 결함 2건도 고쳤다.

1. **fall-through**: `sum == 0` 분기가 `fp32_add = 32'd0` 을 낸 뒤에도 아래
   정규화 코드로 떨어져 `{r_s, r_e, r_m}` 로 덮어써졌다.
2. **무한정 while 루프**: 정규화가 `while (!sum[24] && r_e > 0)` 였다. `r_e` 가
   최대 255 라 합성 도구가 255회까지 펼쳐야 했다. `sum` 이 26비트이므로
   상한 25회 `for` 루프로 바꿨다 — 동작은 동일하고 합성 가능해진다.

### 2-2. 병렬 타일 그리드

`PAR` 파라미터로 PAR×PAR 물리 타일을 둔다. `PAR` 은 `TILES`(=8)를 나누어야 한다.

- **`PAR == TILES` (기본 8)**: 논리 타일마다 물리 타일이 하나씩 있으므로
  누적을 타일 내부 FP32 누산기가 그대로 수행한다. 외부 누산기도 외부 덧셈도
  없다 — **정수 덧셈 버그가 구조적으로 불가능**하다. K-step 당 1 사이클.
- **`PAR < TILES`**: 물리 타일이 여러 논리 위치를 나눠 맡으므로 외부 누산기를
  쓰되 덧셈은 `bf16_pkg::fp32_add` 로 한다. 저장은 타일 하나(256 덧셈)씩
  순회한다 — PAR×PAR 전체를 한 사이클에 더하면 합성되지 않는다.

## 3. 측정 결과 (Verilator 5.040, cocotb 2.1.0)

| 구성 | MAC | K-step 당 사이클 | K-step 테스트 |
|---|---|---|---|
| 원본 | 256 | 48.4 (측정 꼬리) | **0/4 실패** |
| 새 PAR=1 | 256 | 40.2 | 4/4 통과 |
| 새 PAR=2 | 1,024 | 16.2 | 4/4 통과 |
| 새 PAR=4 | 4,096 | 10.2 | 4/4 통과 |
| 새 PAR=8 | 16,384 | 0.4 | 4/4 통과 |

기존 테스트벤치 회귀: `tb_mxu_bf16_128x128` 5/5, `tb_mxu_bf16_16x16` 5/5 통과.

## 4. 재현

```bash
export PATH=/opt/verilator/bin:$PATH
./sim/cocotb/verify_matrix.sh          # 전체 매트릭스
MXU128_RTL=legacy python3.13 sim/cocotb/run_mxu128_kstep.py   # 버그 재현
MXU128_RTL=new MXU128_PAR=8 python3.13 sim/cocotb/run_mxu128_kstep.py
```

**주의**: Icarus 는 unpacked 배열 포트 바인딩을 지원하지 않아 이 모듈들의
누산기가 전부 X 로 읽힌다 (RTL 문제가 아니다). 검증은 Verilator 로 한다.
cocotb 2.1.0 은 Verilator 5.028 이상을 요구한다 — Debian 의 5.020 으로는
`clearEvalNeeded` 심볼 오류가 난다.

## 5. 남은 일

- `PAR=8` 은 PE 마다 전용 FP32 가산기를 둬서 16,384 가산기가 된다.
  Verilator C++ 생성만 784MB 를 넘었다. 실제 합성 전에 PE 산술을
  블록 부동소수점이나 출력당 가산기 트리로 바꾸는 검토가 필요하다.
- `PAR < TILES` 경로는 K-outer/tile-inner 데이터플로를 유지해 외부 누산기가
  필요하다. 출력 고정(output-stationary, tile-outer/K-inner)으로 바꾸면
  외부 누산기를 없앨 수 있으나 A/B 버퍼링이 필요하다.
