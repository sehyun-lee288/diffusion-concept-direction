# 14 — Unsupervised concept directions from the score-Jacobian

Phase 22 · `scripts/28_jacobian_directions.py` ·
figure `figures/exp23_jacobian_directions.png`

## 실험 목적
"한 점에서 sampling으로 concept direction을 unsupervised로 찾을 수 있나?"
— polytope 샘플링(Phase 17)이 아닌 **local Jacobian** 으로 시도.
t≈700에서 `∂ε/∂x_t` 의 top-k right singular vector를 추출.

## 왜 하는지
- Phase 17: polytope 샘플링은 concept direction을 못 줌
- Jacobian의 top singular vector = "출력을 가장 많이 바꾸는 방향" =
  concept axis 후보 (GANSpace, Park et al. 2302.12469)
- Supervised Δh와 달리 label 불필요

## 가설
1. Top singular vector들이 해석 가능한 concept (smile, gender, …) 에 대응
2. 그 중 하나가 supervised smile Δx와 높은 cosine → unsupervised
   rediscovery

## 결과

### Singular values — near-degenerate
`σ = [1.804, 1.798, 1.795, 1.787, 1.781]` — top-5가 1.3% 이내. Phase 19의
near-degenerate spectrum 재확인.

### Smile과의 정렬 — 없음
모든 v_i의 cos(smile Δx) ≈ 0.000. 어느 Jacobian 방향도 supervised smile
방향과 정렬 안 됨.

### 시각적 (`exp23_jacobian_directions.png`)
- **v1**: 강한 변화 — s=−44 메이크업 여성, s=0 남성, s=+44 다른 남성.
  단 identity/gender가 **holistic하게** 변함 — 깨끗한 단일 concept 아님
- **v2–v5**: 거의 변화 없음 (degenerate spectrum 때문)

## 해석

**t=700의 ε-output Jacobian (∂ε/∂x) 은 near-isotropic** — 한두 개
dominant direction이 없음. 따라서 top singular vector로 disentangled
concept이 안 나옴. v1은 "가장 output-sensitive한 방향"이긴 하나
entangled global direction.

| 방법 | concept direction? |
|---|---|
| Polytope 샘플링 (Phase 17) | ❌ |
| ε-Jacobian top SV (이 실험) | △ v1 변화 강하나 entangled, spectrum degenerate |
| Supervised Δh mean-shift (Phase 11-16) | ✅ (label 필요) |

## 미검증 경로 — h-space Jacobian

우리가 쓴 건 **ε-output Jacobian** `∂ε/∂x`. Park et al. "Unsupervised
Discovery" (2302.12469) 가 실제로 쓰는 건 **h-space Jacobian** `∂h/∂x`
(bottleneck feature 대상). ε-Jacobian이 isotropic해도 h-Jacobian은 아닐
수 있음. h-Jacobian top SV가 unsupervised concept discovery의 진짜
후보 — 아직 미검증.

## Next steps
- **h-space Jacobian** `∂h/∂x_t` 의 top SV 추출 → concept direction
  나오는지 (Park et al. 방법의 정확한 재현)
- 다른 timestep (t=500, 900) 의 ε-Jacobian spectrum — degeneracy가
  t-dependent인가
- Supervised + unsupervised 결합: supervised Δh를 seed로 Jacobian power
  iteration → 그 근방의 자연스러운 axis 찾기
