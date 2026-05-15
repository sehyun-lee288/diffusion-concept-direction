# 16 — Many concept directions from one point

Phase 24 · `scripts/30_h_jacobian_many.py` ·
figure `figures/exp25_h_jacobian_many.png` · data `data/h_jacobian_k12.pt`

## 실험 목적
"한 점에서 *여러* concept direction을 발견할 수 있나?" — Phase 23의
top-5를 top-12로 확장. 같은 단일 점 x_split (t≈700) 에서 h-Jacobian의
top-12 singular vector 추출.

## 왜 하는지
Phase 23이 5개 방향이 의미 있음을 보였음. 사용자 질문: 더 많이, 여러
방향을 한 점에서 동시에 얻을 수 있나? → 직접 확인.

## 결과 — **가능, 12개 (그 이상)**

### Spectrum — smooth decay, 고차원 subspace
```
σ = [41.8, 33.0, 30.3, 28.0, 26.0, 24.8,
     23.6, 22.7, 22.1, 21.4, 20.4, 19.8]
```
- 매끄럽게 감소, **sharp cutoff 없음** → 한 점의 local semantic subspace
  는 최소 12차원 (실제로는 더 높음)
- ε-Jacobian (Phase 22) 의 isotropic spectrum과 다르고, top-5만 보던
  Phase 23보다 풍부

### 12개 방향 모두 distinct한 변환
`exp25_h_jacobian_many.png`: 각 행이 동일 base face의 서로 다른 변형
(gender, identity, pose, hair, lighting, expression 등 혼합).

### 그러나 — 여전히 entangled
Supervised smile/gender 방향과의 cosine:

| | 최대 \|cos\| |
|---|---|
| smile | 0.18 (v5, v9) |
| gender | 0.18 (v7) |

**어느 singular vector도 단일 attribute에 깨끗하게 정렬 안 됨**. 각
방향이 여러 attribute를 섞음.

## 해석

**한 점에서 여러 (k개) concept direction 추출 = 가능.** Top-k h-Jacobian
singular vector가 그 점의 **local semantic subspace**의 정규직교 기저.

단 중요한 미묘함:
- Singular vector들은 σ로 정렬된 **arbitrary orthonormal basis** of the
  subspace — "진짜 concept" 은 이 subspace 안의 어떤 *회전* 일 수 있음
- 즉 우리가 얻은 12개는 "12개 방향" 이지 "12개 깨끗한 concept" 이 아님
- Subspace 자체는 의미 있음 (모든 방향이 semantic 변화 생성). 그 안에서
  disentangled axis를 뽑으려면:
  1. ICA / sparse rotation (unsupervised)
  2. supervised reference로 projection (smile Δh를 subspace에 투영)

## 정리 — 본 연구의 "concept direction 발견" 결론

| 질문 | 답 |
|---|---|
| 한 점에서 1개 direction? | ✅ h-Jacobian top SV (Phase 23) |
| 한 점에서 여러 direction? | ✅ top-k SV, k=12+ (이 실험) |
| 그게 깨끗한 단일 concept인가? | ❌ entangled — subspace는 맞으나 basis가 concept-aligned 아님 |
| 깨끗한 단일 axis 원하면? | supervised Δh (Phase 11-16) 또는 subspace 내 supervised projection |

## Next steps
- **Subspace + supervision 하이브리드**: 12D h-Jacobian subspace에 supervised
  smile Δh를 projection → subspace 안의 "smile axis" 추출. Unsupervised
  subspace 발견 + 최소 supervision으로 clean axis.
- ICA on the 12 decoded-perturbation set → unsupervised disentangled rotation
- k를 더 키워 (k=30) subspace 차원 추정
