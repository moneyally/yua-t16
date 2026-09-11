# PRIOR_ART.md — ORBIT-DR1 선행 연구

버전 0.1 (2026-09-11). `docs/DESIGN.md` 1절이 이 문서를 참조한다.

## 0. 이 문서의 규칙

1. **여기 적힌 것은 전부 검색으로 확인한 것이다.** 논문이 실제로 존재하고, 제목·저자·연도·URL 이
   검색 결과에 나온 것만 적는다.
2. **본문(PDF)은 읽지 못했다.** 이 실행 환경에서 `arxiv.org` 가 egress 프록시에 막힌다
   (`EGRESS_BLOCKED`). 그래서 아래 값들은 **검색 결과 스니펫·초록 수준**이다.
   본문을 봐야 아는 항목은 **미확인** 으로 표시했다. `CLAUDE.md` 규칙 7.
3. **"최초", "세상에 없는", "아무도 안 했다" 를 쓰지 않는다.** 신규성은 주장이 아니라
   검색 결과로 남는 것이다. 이 문서의 목적은 DR1 이 무엇의 *뒤에* 있는지 적는 것이다.
4. 새 논문을 찾으면 여기에 추가하고, DR1 의 어떤 부분과 겹치는지 한 줄 적는다.

---

## 1. 가장 가까운 선행 연구 — 온칩 상태 상주 FPGA 가속기

### 1.1 arXiv 2603.05931 (2026-03)

> Gupta, Wang, Kannan, Prasanna,
> *A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA*
> https://arxiv.org/abs/2603.05931

검색으로 확인된 사양:

| 항목 | 값 | 출처 |
|---|---|---|
| 대상 모델 | Gated DeltaNet (디코드 단계) | 검색 스니펫 |
| 핵심 아이디어 | **순환 상태 2MB 를 Alveo U55C 온칩 BRAM 에 상주** → 메모리 바운드를 컴퓨트 바운드로 | 검색 스니펫 |
| 구현 | Vitis HLS 2025.1 | 검색 스니펫 |
| 주파수 | 300 MHz | 검색 스니펫 |
| 지연 | **63 µs/token** | 검색 스니펫 |
| 비교 | H100 PCIe 대비 4.5배 | 검색 스니펫 |
| 전력 | 온칩 9.96 W, 토큰당 에너지 최대 60배 | 검색 스니펫 |
| 자원 | BRAM36 4,032개 | 검색 스니펫 |
| 최적화 | dual-port BRAM, head-dim complete partitioning, cyclic factor 16, II=1 | 검색 스니펫 |
| **수치 형식** | **미확인** (FP32 로 추정하는 글이 있으나 본문 확인 못 함) | — |
| **토큰당 상태 패스 수** | **미확인** (DR1 은 현재 3패스 — DESIGN 9절) | — |

**DR1 과의 관계: 이것이 기준선이다.** "상태를 온칩에 상주시킨다" 는 DR1 의 출발점 문장은
이 논문이 이미 하드웨어로 보여준 것이다. DR1 은 그 뒤에 있다.

### 1.2 arXiv 2601.02135 (2026-01)

> *HFRWKV: A High-Performance Fully On-Chip Hardware Accelerator for RWKV*
> (중산대학교) https://arxiv.org/abs/2601.02135

검색으로 확인된 내용:
- FPGA 기반 RWKV 전용 가속기. **완전 온칩 연산**(off-chip 가중치 접근 제거)이 목표.
- **Δ-PoT 양자화** + 하드웨어 친화적 **하이브리드 정밀도 양자화** 전략.
- 지수·나눗셈 같은 복잡 연산은 **LUT 또는 구간 선형 근사**로 재사용 구조에 넣음.
- 파이프라인 구조: 병렬 행렬-벡터 배열 + 온칩 LayerNorm, 연산 재배열 + chunked double buffering.
- 수치 정확도 수치, 상태 크기: **미확인**.

**DR1 과의 관계:** "고정소수·LUT 로 순환 모델을 온칩에서 돌린다" 는 방향도 이미 있다.
DR1 이 Q1.15 를 쓰는 것은 새로운 발상이 아니라 **같은 계열의 선택**이다.
다른 점은 대상 식(RWKV vs 2절 델타룰)과 검증 방식(비트 정확 골든)뿐이다.

### 1.3 두 논문 대비 DR1 의 위치

| | 2603.05931 | 2601.02135 (HFRWKV) | ORBIT-DR1 |
|---|---|---|---|
| 대상 식 | Gated DeltaNet | RWKV | DESIGN 2절 델타룰 1식, 헤드 1개 |
| 구현 | Vitis HLS | (미확인, FPGA) | **손으로 쓴 SystemVerilog RTL** |
| 수치 형식 | 미확인 | Δ-PoT + 하이브리드 정밀도 | **Q1.15 / α·β 는 UQ1.15, 결정론적** |
| 검증 | 미확인 | 미확인 | **numpy 골든 모델과 전 토큰 비트 비교** (DESIGN 7절) |
| 제어 평면 | HLS 생성 | 미확인 | **기존 디스크립터 큐·IRQ·트레이스 IP 재사용** |
| 규모 | d=128급, 다중 헤드 | RWKV 전체 블록 | **d=16 부터, 헤드 1개** |
| 상태 | 2MB 온칩 상주 (실측) | 완전 온칩 (실측) | d=16 → 512B (합성만 확인, 보드 미검증) |

표의 마지막 줄이 중요하다. **선행 연구는 보드에서 돌렸고, DR1 은 아직 아니다.**

---

## 2. 알고리즘 계열 — DR1 이 구현하는 식의 출처

### 2.1 arXiv 2605.22791 (2026-05-21)

> Hatamizadeh, Choi, Kautz, *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*
> https://arxiv.org/abs/2605.22791 / 코드 https://github.com/NVlabs/GatedDeltaNet-2

- 채널별 **erase gate b_t** 와 **write gate w_t** 를 분리한다. 둘이 같은 스칼라로 붕괴하면 KDA,
  감쇠까지 붕괴하면 Gated DeltaNet 이 된다 — 즉 **DR1 의 스칼라 α·β 식은 이 계열의 가장 단순한 꼭지점**이다.
- 1.3B / FineWeb-Edu 100B 토큰에서 Mamba-2·GDN·KDA·Mamba-3 대비 최고 성능 보고 (본문 미확인).

**DR1 과의 관계:** DR1 v1 이 쓰는 스칼라 α, β 는 **이 논문이 특수 케이스로 지목한 형태**다.
채널별 감쇠로 확장하는 것은 DESIGN 9절 위험 항목(v2 검토)이며, 확장한다고 새로운 것이 되지 않는다.

### 2.2 arXiv 2607.07953 (2026-07-08)

> Cerruti, Rieder, Rowlands, Jin, Schlag (ETH Zurich),
> *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing*
> https://arxiv.org/abs/2607.07953

- softmax attention + DeltaNet / Gated DeltaNet / Kimi Delta Attention / Gated DeltaNet-2 를
  **공통 순환 메모리 표기**로 정리하고, 표현력·감쇠·erase/write 제어·학습 처리량·구현 복잡도를 비교.
- 350M / 15B 토큰 중심, DeltaNet 은 1.3B·3B 까지.

**DR1 과의 관계:** "계열" 이라는 말을 쓰기 전에 이 표기법을 따르는 것이 낫다.
DESIGN 1절 각주(KDA 는 채널별 감쇠)의 근거가 여기 있다.

---

## 3. DR1 의 다음 아이디어와 이미 겹치는 논문 (주말 연구 세션이 먼저 읽을 것)

이 절은 **경고용**이다. 아래 두 편은 내가 "빈 질문" 후보로 적어둔 것을 이미 다룬다.
`docs/RESEARCH.md` 에서 Q-A / Q-B 를 "아니오(선행 없음)" 로 분류하기 전에 반드시 읽어야 한다.

### 3.1 arXiv 2608.15533 (2026-08-16) — 게으른 감쇠/지연 반영 (Q-B 와 직접 충돌)

> Lin, Sun, Sun, *DeltaLog: Deferred Materialization of Recurrent States for Linear Attention Decoding*
> https://arxiv.org/abs/2608.15533

- 순환 상태를 **dense base state + 최근 갱신의 유계 로그(bounded log)** 로 표현한다.
- 대부분의 디코드 스텝은 로그에 **compact update factor 만 append**, 주기적 merge 스텝에서
  base state 로 folding. 모델이 보는 dense 상태는 eager 디코딩과 동일 — **의미(semantics) 불변**.
- 목적: 토큰마다 전체 상태를 write-back 하는 메모리 트래픽 제거.

→ 내가 Q-B 로 적어둔 "스칼라 c 로 감쇠를 미루고 나중에 반영" 은 **같은 문제의 더 약한 버전**이다.
   "새롭다" 고 쓸 수 없다. 남는 질문은 "고정소수에서 merge 주기와 오차의 관계" 정도이며,
   그것도 이 논문이 이미 답했는지 본문으로 확인해야 한다.

### 3.2 arXiv 2608.22354 (2026-08) — 장기 상태 안정화 (Q-A 와 겹침)

> *SANE: State Anomaly Neutralization for Stable Extreme-Context Delta-Rule Models*
> https://arxiv.org/abs/2608.22354

- 제목·초록 수준만 확인. **극단적 긴 문맥에서 델타룰 상태의 이상치를 중화해 안정화**한다는 내용.

→ 내가 Q-A 로 적어둔 "장기 누적 오차 + 주기적 재정규화" 와 겹칠 가능성이 높다.
   차이가 있다면 "부동소수 상태의 통계적 이상치" vs "고정소수 LSB 누적" 인데, 확인 전에는 주장 금지.

### 3.3 검색 중 눈에 띈 것 (아직 안 읽음)

- arXiv 2608.20961 — *TreeWY: Speculative Verification for Gated DeltaNet Hybrids*

---

## 4. 정리 — DR1 이 사실로 말할 수 있는 것

1. 온칩 상태 상주 델타룰 가속기는 **이미 있다** (1.1). DR1 은 그 뒤를 따른다.
2. 고정소수/하이브리드 정밀도로 순환 모델을 온칩화한 FPGA 가속기도 **이미 있다** (1.2).
3. DR1 이 쓰는 스칼라 α·β 델타룰은 **최신 논문이 특수 케이스로 지목한 형태**다 (2.1).
4. 따라서 DR1 의 내용은 **구현 방식과 검증 방식**이다: 손으로 쓴 RTL, Q1.15 결정론,
   비트 정확 골든 대조, 기존 제어 평면 재사용, d=16 부터. 이것들은 **새로움이 아니라 작업 방식**이다.
5. 보드 실동작·전력·지연 수치는 **DR1 에 아직 없다.** 선행 연구에는 있다.
