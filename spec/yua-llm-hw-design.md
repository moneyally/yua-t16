# YUA Hardware Stack — YUA-LLM 맞춤 재설계 v2.0

> 기존 YUA-T16 v1 RTL은 유지하고, LLM 전체 추론에 필요한 컴포넌트를 추가 설계
> 베이스 모델: GPT-OSS-20B (MoE Transformer, 32 experts)
> 목표: ORBIT-G1이 GPT-OSS-20B 전체 추론 그래프를 자율 실행

---

## 1. GPT-OSS-20B 추론에 필요한 연산 전체 목록

한 번의 forward pass에서 수행되는 연산:

```
[Token Embedding]
  └─ Embedding lookup → GEMM ✅ YUA-T16

[For each Transformer Layer × N]
  1. RMSNorm                → 정규화     ❌ 없음
  2. QKV projection         → GEMM       ✅ YUA-T16
  3. RoPE (Rotary Embed)    → elementwise ❌ 없음
  4. Attention scores Q@K^T → GEMM       ✅ YUA-T16
  5. Scale + Softmax        → elementwise ❌ 없음
  6. Attention output V     → GEMM       ✅ YUA-T16
  7. Output projection      → GEMM       ✅ YUA-T16
  8. KV-Cache read/write    → 메모리     ❌ 없음
  9. MoE Router (top-k)     → 라우팅     ❌ 없음
  10. Expert gate_proj      → GEMM       ✅ YUA-T16
  11. SiLU activation       → elementwise ❌ 없음
  12. Expert up_proj        → GEMM       ✅ YUA-T16
  13. Element multiply      → elementwise ❌ 없음
  14. Expert down_proj      → GEMM       ✅ YUA-T16
  15. Residual add          → elementwise ❌ 없음

[LM Head]
  └─ Final projection       → GEMM       ✅ YUA-T16

[Quantization]
  INT4/AWQ weight format    → ❌ INT8만 지원
```

**결론:**
- GEMM 연산: YUA-T16으로 커버 ✅
- 나머지 7가지: 신규 유닛 필요

---

## 2. 추가 필요 컴포넌트

### 2.1 VPU (Vector Processing Unit) — 신규
모든 elementwise 연산을 담당

| 연산 | 용도 | 구현 |
|------|------|------|
| RMSNorm | 레이어 정규화 | sum(x²) → rsqrt → scale |
| SiLU | FFN activation | x * sigmoid(x) |
| RoPE | 위치 인코딩 | cos/sin 회전 행렬 곱 |
| Softmax | Attention 확률 | max→exp→sum→div |
| Residual Add | 스킵 커넥션 | elementwise add |
| Scale | Attention score | multiply by 1/√d |

**설계 방향:**
- 256-wide SIMD 벡터 유닛
- FP16/BF16 + INT8 지원
- 룩업 테이블 (LUT) for sigmoid/exp
- VECTOR_OP descriptor 확장으로 제어

### 2.2 KVC (KV-Cache Controller) — 신규
autoregressive generation의 핵심 — 이전 토큰의 K,V를 재사용

```
생성 단계:
  prefill:  모든 입력 토큰 한번에 처리, K/V GDDR6에 저장
  decode:   토큰 하나씩 생성, 저장된 K/V 읽어서 attention 계산

메모리 레이아웃:
  GDDR6[layer][head][seq_pos] = {K_vec, V_vec}
```

**설계 방향:**
- 최대 시퀀스 길이: 32K 토큰 (YaRN 확장 대비)
- 레이어 수 × 헤드 수 × seq_len × head_dim 크기
- GPT-OSS-20B 기준: ~2.4GB per 8K context (FP16)
- GDDR6 대역폭 활용 최대화
- PagedAttention 스타일 청크 관리 (vLLM 방식)

**신규 descriptor:**
```c
// KVC_READ (type 0x0A)
struct orbit_desc_kvc_read {
  orbit_desc_header h;
  uint32_t layer_id;
  uint32_t head_id;
  uint32_t seq_start;
  uint32_t seq_len;
  uint64_t dst_addr;  // 읽은 KV를 쓸 주소
};

// KVC_WRITE (type 0x0B)
struct orbit_desc_kvc_write {
  orbit_desc_header h;
  uint32_t layer_id;
  uint32_t head_id;
  uint32_t seq_pos;   // 현재 토큰 위치
  uint64_t src_addr;  // 저장할 K/V 주소
};
```

### 2.3 MoE Router — 신규
GPT-OSS-20B의 32 experts 중 top-k 선택

```
입력: hidden_state [seq × d_model]
라우터 weight: [d_model × num_experts=32]
출력: top_k expert indices + routing scores

알고리즘:
  1. hidden @ router_weight = logits [seq × 32]
  2. softmax(logits) = probs
  3. top-2 선택 (GPT-OSS는 top-2)
  4. expert별 토큰 그룹핑 → expert GEMM 디스패치
```

**설계 방향:**
- GEMM은 YUA-T16 재사용 (router weight 행렬곱)
- top-k 선택 로직은 VPU 또는 별도 경량 유닛
- expert별 토큰 배치 재배열 (gather/scatter)

**신규 descriptor:**
```c
// MOE_ROUTE (type 0x0C)
struct orbit_desc_moe_route {
  orbit_desc_header h;
  uint64_t logits_addr;
  uint64_t indices_addr; // 출력: expert 인덱스
  uint64_t scores_addr;  // 출력: routing 가중치
  uint32_t num_tokens;
  uint32_t num_experts;
  uint32_t top_k;
};
```

### 2.4 INT4 지원 (YUA-T16 v2) — 업그레이드
AWQ 4-bit 양자화 모델 실행을 위해

```
현재: INT8 × INT8 → INT32
추가: INT4 × INT4 → INT32 (2배 처리량, 반반 메모리)
      INT4 weight × FP16 activation → FP16 (AWQ 방식)
```

**설계 방향:**
- 기존 16×16 MAC array 위에 INT4 압축/해제 레이어 추가
- AWQ: INT4 weight를 FP16으로 dequantize 후 기존 FP16 GEMM
- 또는 INT4 native MAC 추가 (더 효율적)
- 타일 크기 32×32로 확장 권장

---

## 3. ORBIT-G1 v2 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                    ORBIT-G1 v2                       │
│                                                      │
│  PCIe Gen4 x16                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │           Command Processor                  │    │
│  │   Descriptor Queue × 4                      │    │
│  └──────┬───────────┬────────────┬─────────────┘    │
│         │           │            │                   │
│  ┌──────▼──┐  ┌─────▼─────┐  ┌──▼──────────────┐   │
│  │Compute  │  │    VPU    │  │  KVC + MoE      │   │
│  │Clusters │  │(RMSNorm,  │  │  Controller     │   │
│  │         │  │SiLU,RoPE, │  │                 │   │
│  │N×YUA-T16│  │Softmax,   │  │ KV-Cache GDDR6  │   │
│  │tiles    │  │Residual)  │  │ MoE Router      │   │
│  │INT8/INT4│  │256-wide   │  │ top-k select    │   │
│  └──────┬──┘  └─────┬─────┘  └──┬──────────────┘   │
│         └───────────┴───────────┘                   │
│                     │                               │
│         ┌───────────▼──────────────┐                │
│         │     Global Memory        │                │
│         │  GDDR6 (16GB or 32GB)   │                │
│         │  Weights + KV-Cache      │                │
│         └──────────────────────────┘                │
│                                                      │
│  SUP (Sidecar): data format convert, DMA, diag      │
└─────────────────────────────────────────────────────┘
```

---

## 4. LLM 추론 실행 흐름 (descriptor 시퀀스)

GPT-OSS-20B, 토큰 1개 decode 단계:

```
[prefill 완료 가정, KVC에 K/V 저장됨]

① DMA_2D        : 입력 토큰 임베딩 → GDDR6
② GEMM_INT8(4)  : QKV projection (Q,K,V 계산)
③ VECTOR_OP     : RoPE 적용 (Q, K에)
④ KVC_WRITE     : 새 K,V를 KV-Cache에 저장
⑤ KVC_READ      : 전체 시퀀스 K,V 읽기
⑥ GEMM_INT8(4)  : Q @ K^T (attention score)
⑦ VECTOR_OP     : Scale(1/√d) + Softmax
⑧ GEMM_INT8(4)  : score @ V
⑨ GEMM_INT8(4)  : output projection
⑩ VECTOR_OP     : Residual add
⑪ VECTOR_OP     : RMSNorm
⑫ GEMM_INT8(4)  : MoE router logits
⑬ MOE_ROUTE     : top-2 expert 선택
⑭ GEMM_INT8(4)  : gate_proj (각 expert)
⑮ VECTOR_OP     : SiLU(gate) * up
⑯ GEMM_INT8(4)  : down_proj (각 expert)
⑰ VECTOR_OP     : expert 결과 합산 + residual
⑱ BARRIER
... (레이어 반복 × N)
⑲ GEMM_INT8(4)  : LM head projection
⑳ VECTOR_OP     : softmax → argmax → next token
㉑ EVENT         : 호스트에 토큰 전달
```

---

## 5. 기존 YUA-T16 v1 보존 전략

**버리지 않고 확장:**

```
YUA-T16 v1 (현재 RTL)
  → 유지: mac_pe.sv, mac_array.sv, gemm_core.sv
  → 유지: ctrl_fsm.sv, act_sram.sv, wgt_sram.sv
  → 유지: gemm_top.sv (커맨드 브릿지 로직 검증됨)
  → 유지: cocotb 테스트 전체

YUA-T16 v2 (추가)
  → 추가: INT4 압축 해제 프리프로세서
  → 추가: 타일 크기 32×32 옵션
  → 추가: scale factor 레지스터 (AWQ dequant)
```

---

## 6. Descriptor v2 추가 타입

기존 v1 유지 + 추가:

| Type ID | Name | 설명 |
|---------|------|------|
| 0x0A | KVC_READ | KV-Cache 읽기 |
| 0x0B | KVC_WRITE | KV-Cache 쓰기 |
| 0x0C | MOE_ROUTE | MoE expert 라우팅 |
| 0x0D | VECTOR_OP_EX | SiLU, RMSNorm, RoPE 확장 ops |
| 0x0E | GEMM_INT4 | INT4 × INT4 GEMM |
| 0x0F | SOFTMAX | Attention softmax |

---

## 7. 구현 우선순위

```
Phase A (FPGA 검증 가능)
  ① VPU 기본 ops (add, mul, clamp) → VECTOR_OP 확장
  ② RMSNorm → VECTOR_OP_EX
  ③ SiLU → VECTOR_OP_EX

Phase B (LLM 첫 추론)
  ④ KVC Controller + GDDR6 인터페이스
  ⑤ Softmax unit → VECTOR_OP_EX
  ⑥ RoPE unit

Phase C (MoE 지원)
  ⑦ MoE Router (top-k) → MOE_ROUTE

Phase D (성능 최적화)
  ⑧ INT4 지원 → YUA-T16 v2
  ⑨ 타일 크기 32×32 확장
  ⑩ Pipeline 최적화 (GEMM + VPU overlap)
```

---

## 8. YUA-LLM ↔ ORBIT-G1 통합

```
yua-backend (Node.js)
  └─ yua-provider.ts → YUA-LLM 추론 서버

YUA-LLM 추론 서버 (Python/C++)
  └─ vLLM 또는 자체 런타임
       └─ ORBIT-G1 드라이버 (Linux kernel)
            └─ Descriptor Queue 제출
                 └─ ORBIT-G1 하드웨어 실행
```

**단기 (GPU 시대):**
```
yua-backend → yua-provider.ts → vLLM (GCP L4) → GPT-OSS-20B
```

**장기 (칩 완성 후):**
```
yua-backend → yua-provider.ts → ORBIT-G1 런타임 → YUA-LLM
```

인터페이스는 동일 (OpenAI 호환 `/v1/chat/completions`) — 스왑 가능.

---

## 9. 현재 상태 요약

| 컴포넌트 | 상태 |
|---------|------|
| YUA-T16 v1 RTL | ✅ 완성 (검증됨) |
| ORBIT-G1 v1 Spec | ✅ 완성 |
| Descriptor v1 | ✅ 완성 |
| VPU | ✅ RTL 완성 + cocotb 8/8 PASS |
| KVC Controller | ✅ RTL 완성 + cocotb 4/4 PASS |
| MoE Router | ✅ RTL 완성 + cocotb 3/3 PASS |
| INT4 지원 | ✅ RTL 완성 + cocotb 3/3 PASS |
| YUA-LLM 소프트웨어 | 🔄 진행 중 |
