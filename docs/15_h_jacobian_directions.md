# 15 — Unsupervised concept directions from the h-space Jacobian

Phase 23 · `scripts/29_h_jacobian_directions.py` ·
figure `figures/exp24_h_jacobian_directions.png`

## 실험 목적
Phase 22의 ε-output Jacobian (`∂ε/∂x`) 은 near-isotropic이라 concept
direction이 안 나왔음. Park et al. (2302.12469) 가 실제 쓰는 **h-space
Jacobian** (`∂h/∂x`) — bottleneck feature 대상 — 으로 재시도.

## 왜 하는지
- ε-Jacobian isotropy ≠ h-Jacobian isotropy
- h-space는 attribute 정보를 강하게 인코딩 (Phase 18, d'≈1.7) →
  h-Jacobian은 structure가 있을 가능성

## 가설
`∂h/∂x` 의 top-k right singular vector가 의미 있는 concept direction.
spectrum이 ε-Jacobian보다 덜 degenerate할 것.

## 결과 — **작동함**

### Spectrum — 구조 있음
`σ = [41.8, 32.9, 30.1, 27.9, 25.2]`, **σ1/σ5 = 1.66**

ε-Jacobian (Phase 22) 의 σ1/σ5 ≈ 1.01 (near-degenerate) 와 대조 —
h-Jacobian은 top direction들이 분리됨.

### 시각적 (`exp24_h_jacobian_directions.png`)
5개 방향 **모두 강한 semantic 변화** 생성 (ε-Jacobian은 v1만 움직였음):

| dir | σ | 관찰 |
|---|---:|---|
| v1 | 41.8 | gender/identity 큰 전환 (여성 → 남성) |
| v2 | 32.9 | 또 다른 gender-ish axis |
| v3 | 30.1 | 남성 → 웃는 여성 |
| v4 | 27.9 | 웃는 청년 ↔ 여성 (smile 성분 보임) |
| v5 | 25.2 | 콧수염 남성 ↔ 여성 |

### Smile과의 정렬
cos(smile Δx): v1 +0.10, v5 −0.17, 나머지 ~0. v5가 약하게 smile 성분
보유하나 깨끗한 정렬은 아님.

## 해석

**h-space Jacobian은 unsupervised concept direction을 준다 — 단
entangled.**

| 방법 | concept direction? | 품질 |
|---|---|---|
| Polytope 샘플링 (Phase 17) | ❌ | — |
| ε-Jacobian top SV (Phase 22) | △ | near-isotropic, v1만 entangled 변화 |
| **h-Jacobian top SV (Phase 23)** | ✅ | 5방향 모두 의미 있는 변화, 단 attribute mixed |
| Supervised Δh (Phase 11-16) | ✅✅ | 단일 attribute 깨끗, label 필요 |

→ **사용자 질문의 답**: 한 점에서 unsupervised로 concept direction을 찾을
수 **있다** — h-space Jacobian의 top singular vector로. 단 그 방향들은
"의미 있지만 disentangled하지 않음" — 각 방향이 gender/identity/expression
을 섞어서 바꿈. 깨끗한 단일-attribute axis를 원하면 supervision (Phase
11-16) 또는 추가 처리가 필요.

이건 Park et al. 의 방법이 우리 DDPM에서도 작동함을 확인 — operator
선택 (ε 대신 h) 이 결정적이었음.

## Next steps
- h-Jacobian이 span하는 5차원 subspace 안에서 supervised smile Δh와 가장
  가까운 방향 찾기 → unsupervised subspace + 소량 supervision으로 clean axis
- 다른 timestep에서 h-Jacobian spectrum — t-dependence
- top-k를 더 크게 (k=20) → 더 세밀한 concept이 분리되는지
