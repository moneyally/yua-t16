# 혼자서 하룻밤 만에 LLM 추론 가속기 RTL을 만들었다

> GPU 없이 LLM을 돌리고 싶은 사람의 무모한 도전기

---

## 시작은 단순한 불만이었다

GCP에서 GPU 쿼터 신청을 두 번 거절당했다. 돈을 쓰기 싫은 건지, 쿼터를 안 주는 건지 모르겠지만 결론은 하나였다. GPU가 없다.

근데 나는 LLM 추론을 직접 돌려보고 싶었다. 클라우드 API를 그냥 갖다 쓰는 건 너무 재미없다. 어떻게 돌아가는지 직접 보고 싶었다.

그래서 든 생각: "FPGA로 해보면 어때?"

Arty A7-100T가 책상에 한 대 있었다. 원래 다른 용도로 사 뒀다가 안 쓰던 거였는데, 마침 이게 Xilinx 7 시리즈라 Yosys로 합성할 수 있다. 방향이 잡혔다.

프로젝트 이름은 **ORBIT-G1**, 칩 이름은 **YUA-T16**으로 정했다.

---

## 아키텍처를 어떻게 잡을 것인가

LLM 추론의 병목은 크게 세 곳이다.

1. **행렬 곱셈 (GEMM)**: Transformer의 Q/K/V/FFN 연산 대부분을 차지한다
2. **벡터 연산 (VPU)**: LayerNorm, SiLU, RoPE, Softmax 등 activation
3. **KV Cache 관리**: Attention 캐시 메모리 I/O

여기에 LLaMA 계열 모델이 대부분 **MoE(Mixture of Experts)** 방향으로 가고 있어서 라우팅 로직도 필요하다고 판단했다.

그래서 YUA-T16의 4대 컴포넌트를 이렇게 잡았다:

| 컴포넌트 | 역할 |
|----------|------|
| GEMM 타일 (16×16) | INT4/INT8 행렬 곱셈 |
| VPU (256-wide SIMD) | 벡터 연산 10가지 |
| KVC Controller | KV Cache SRAM 제어 |
| MoE Router | Expert 선택/스케줄링 |

---

## RTL 구현에서 막힌 것들

### 제약: Icarus Verilog로 시뮬레이션해야 한다

비싼 EDA 툴 없이 `iverilog` + Yosys로만 진행하기로 했다. 문제는 Icarus가 SystemVerilog 지원이 불완전하다는 점이다.

- `typedef struct packed` 같은 건 되는데
- `unique case` 같은 건 안 된다
- IEEE 1800-2012 문법 중 절반은 에러 뱉는다

결국 `.sv` 확장자를 쓰면서도 내용은 거의 순수 Verilog 2005 수준으로 작성해야 했다.

### 고정소수점: Q8.8 포맷으로 통일

부동소수점 하드웨어는 면적이 너무 크다. LLM 추론에서 FP32는 이미 오버킬이고, BF16/FP16도 FPGA에서는 DSP 소모가 심하다.

**Q8.8 고정소수점**으로 통일했다. 정수부 8비트 + 소수부 8비트, 총 16비트.

LLM 추론에서 대부분의 activation은 범위가 좁아서 Q8.8로도 충분히 표현된다. sigmoid, exp 같은 비선형 함수는 **LUT 기반 근사**로 구현했다.

```verilog
// LUT 기반 sigmoid 근사 (Q8.8 입력 → Q8.8 출력)
// 입력 범위를 [-4, 4]로 클램핑 후 256-entry LUT 인덱싱
function automatic [15:0] sigmoid_lut;
    input [15:0] x;
    reg [7:0] idx;
    begin
        // 범위 클램핑: -4.0 = 16'hFC00, +4.0 = 16'h0400
        if ($signed(x) <= $signed(16'hFC00))
            sigmoid_lut = 16'h0001;  // ~0.0
        else if ($signed(x) >= $signed(16'h0400))
            sigmoid_lut = 16'h00FF;  // ~1.0
        else begin
            idx = x[11:4];  // 상위 8비트로 인덱싱
            sigmoid_lut = SIGMOID_TABLE[idx];
        end
    end
endfunction
```

### GEMM INT4: DSP 폭발 문제

처음에 GEMM을 아무 생각 없이 짰다가 Yosys 합성 결과를 보고 눈이 튀어나올 뻔했다.

**DSP48E1 1,040개 사용 — Arty A7-100T에는 90개밖에 없다. 1,156% 초과.**

원인을 분석해보니 두 군데였다:

1. S_COMPUTE: 16개 INT8×INT8 병렬 곱셈이 전부 DSP로 매핑됨
2. S_SCALE: 256개 32-bit × 16-bit 병렬 곱셈이 쭉 DSP로 매핑됨

해결책은 생각보다 단순했다. INT4 weight를 shift-and-add로 분해하는 것이다.

```verilog
// INT4 (4비트) weight를 shift-and-add로 곱셈
// b[3]이 부호 비트(2의 보수): value = -8*b[3] + 4*b[2] + 2*b[1] + b[0]
function automatic signed [23:0] mul_i8_i4;
    input signed [7:0]  a;   // INT8 activation
    input        [3:0]  b;   // INT4 weight (2의 보수)
    reg signed [23:0] result;
    begin
        result = 24'b0;
        if (b[0]) result = result + {{16{a[7]}}, a};
        if (b[1]) result = result + {{15{a[7]}}, a, 1'b0};
        if (b[2]) result = result + {{14{a[7]}}, a, 2'b0};
        if (b[3]) result = result - {{13{a[7]}}, a, 3'b0};  // 부호 비트
        mul_i8_i4 = result;
    end
endfunction
```

이걸로 바꾸니까 DSP가 1,040개 → 4개로 줄었다. **99.6% 감소.**

---

## 합성 결과: 실제 숫자

Yosys 0.52, `synth_xilinx -family xc7`, 타겟 Arty A7-100T 기준.

### VPU (256-wide SIMD, 10가지 연산)

| 리소스 | 사용량 | 가용량 | 점유율 |
|--------|--------|--------|--------|
| LUT | 25,036 | 82,800 | **30.2%** |
| FF | 4,262 | 126,800 | 3.4% |
| DSP48E1 | 11 | 90 | 12.2% |

LUT6이 17,906개로 가장 많다. SiLU, RoPE, RMSNorm 같은 복잡한 activation이 LUT를 많이 잡아먹는다.

### GEMM INT4 FPGA 버전

| 리소스 | 원본 | FPGA 최적화 | 변화 |
|--------|------|-------------|------|
| LUT | 8,744 | 8,848 | +1% |
| FF | 16,142 | 15,118 | -6% |
| **DSP48E1** | **1,040** | **4** | **-99.6%** |

### 전체 ORBIT-G1 (추정치 포함)

VPU + GEMM + KVC Controller + MoE Router + 인터커넥트 합산:

| 리소스 | 예상 사용 | 가용량 | 점유율 |
|--------|-----------|--------|--------|
| LUT | ~40,384 | 82,800 | **49%** |
| DSP48E1 | ~19 | 90 | 21% |
| BRAM36 | ~6 | 135 | 4% |

**Arty A7-100T 한 장에 전부 들어간다.** 150 MHz 기준 VPU throughput은 38.4 GOPS 예상이다.

---

## 솔직한 현재 상태

잘 된 것:
- RTL 합성 통과 (Yosys)
- DSP 초과 문제 해결
- 전체 리소스 80% 이내 핏

아직 못 한 것:
- **실물 FPGA 검증**: Vivado P&R 및 타이밍 분석 미완
- **실제 추론 테스트**: 아직 소프트웨어 런타임과 연결 안 됨
- **정확한 타이밍**: Yosys는 타이밍 분석 불가, Vivado 필요

합성이 통과한다고 동작 보장이 되는 건 아니다. 타이밍 클로저가 안 잡히면 150 MHz는 못 쓰고 클럭을 낮춰야 할 수도 있다.

---

## 다음 목표

**단기**: Vivado로 실물 Arty A7-100T에 올려보기. 간단한 벡터 연산 테스트벡터로 검증.

**중기**: **OpenMPW 테이프아웃.** Google + Efabless의 무료 ASIC 셔틀 프로그램이 있다. SKY130B 130nm 공정으로 실제 칩을 찍을 수 있다. GEMM 타일 1개 + VPU 32-wide 축소 버전 정도면 면적 안에 들어갈 것 같다.

FPGA는 어디까지나 프로토타입이다. 진짜 목표는 칩이다.

---

## 마무리

GPU 없이 LLM 가속기를 만들겠다는 시도는 여전히 진행 중이다. RTL은 썼고, 합성은 통과했고, 숫자는 나왔다. 다음은 실물 검증이다.

관심 있으면 코드 다 공개돼 있으니 같이 보자.

GitHub: https://github.com/moneyally/yua-t16

---

*작성일: 2026-03-13*
