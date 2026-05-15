# 02 — Radial pattern: 모델 구조인가 selection artifact인가?

Phase 5.5 · `scripts/06_analyze_radial.py` ·
figure `figures/exp1_radial_analysis.png`

## 실험 목적
doc 01의 top-K=20 region map에서 boundary들이 (0.5, 0.5) 근처에 집결하는
방사형 패턴의 **원인 규명**.

## 왜 하는지
방사형이 (a) DDPM mid_block의 본질적인 rank-1-ish 구조인지, (b) 우리가
선택한 top-K-balanced 기준이 self-fulfilling이라 만든 artifact인지를
구분하지 못하면 후속 분석을 잘못 해석하게 됨.

## 가설
각 채널의 boundary line은 `A_c α + B_c β + C_c = 0`. 모든 line이 한 점을
지나려면 행렬 `M = [A|B|C] ∈ ℝ^{C×3}` 의 rank ≤ 2. 만약 모델이 정말
rank-2-ish면 SVD의 third singular value가 작아야 함.

## 결과
**모든 채널 (512개) 의 SVD**:
- singular values: [11.17, 8.24, 3.85]
- energy share: **60.2 / 32.7 / 7.1 %**
- → effective rank ≈ 2 (s₃이 7%만 차지) — but not exactly 2
- SVD null vector가 가리키는 "best common point": (−0.785, −0.323) ← 데이터 cluster (0.495, 0.500) 와 안 맞음

**K-sweep으로 selection bias 분리**:

| K | median (α, β) | MAD |
|---:|---:|---:|
| 10 | (+0.500, +0.502) | (0.007, 0.005) |
| 20 | (+0.495, +0.500) | (0.012, 0.013) |
| 50 | (+0.504, +0.498) | (0.036, 0.035) |
| 100 | (+0.488, +0.478) | (0.073, 0.061) |
| 250 | (+0.534, +0.433) | (0.184, 0.203) |
| **512** | **(+0.647, +0.274)** | **(0.456, 0.474)** |

K 작을수록 cluster가 tight하게 (0.5, 0.5) 근처. K=512는 거의 grid 폭만큼
퍼짐. **방사형은 주로 selection artifact**.

원리: top-K-balanced ↔ "grid window 안에서 ±1이 50:50인 channel" ↔ "boundary line이 window 중앙을 지나는 channel" → 정의상 (0.5, 0.5) 근처에 모임.

## Next steps
- 선택 bias 없이 진짜 neuron boundary를 보려면 reduction을 다르게 → **doc 03 (per-pixel)**
- top-K 기준 자체를 바꾸려면: line 방향 variance 기반, 또는 grid를 SVD null point 중심으로 (T1, future)
