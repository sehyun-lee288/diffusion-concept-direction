# 11 — E-GBAS analog: ε-ball vs GB-RRT sampling around a query

Phase 17 (+17b) · `scripts/20_egbas_analog.py`, `scripts/21_egbas_multistep.py` ·
figures `figures/exp15_egbas_analog.png`, `figures/exp16_egbas_multistep.png`

## 실험 목적
E-GBAS 논문 (Jeon et al. 2019)의 GB-RRT를 diffusion h-space에 이식.
**한 query 점 주위에서 sampling** 하는 두 방식 비교:
- **ε-ball**: `h_query + ε·𝒩(0, I)` — sign pattern 무시
- **GB-RRT**: random walk **rejection sampling** — 모든 512 channel의
  sign이 query와 같을 때만 step 수락

## 왜 하는지
- 지금까지 우리는 **3-anchor span 평면** 또는 **attribute axis** 같은
  주어진 axis 위에서만 sampling했음 (doc 01, 08)
- E-GBAS는 1 query + 자유 sampling (high-dim 그대로)으로 같은 region 안의
  여러 sample 모음 — 더 자연스러운 generative region 탐색
- GAN에서는 결과가 깔끔 (Fig 7 in their paper). Diffusion에서도?

## 가설
1. ε-ball samples는 1-5개 channel sign이 flip → polytope를 벗어남
2. GB-RRT samples는 0 flip → polytope 안 유지
3. **E-GBAS 예측**: GB-RRT samples가 ε-ball보다 query와 더 attribute-
   coherent (같은 face / attribute) 한 이미지 생성

## 결과

### Sign pattern 측정 (가설 1, 2 확인)
- ε-ball (||Δ||≈17, 8% of ||h||): **1–5 sign flip** (polytope 밖)
- GB-RRT (||Δ||=17–40, chain accumulates): **0 sign flip** ✓ (polytope 안)
- GB-RRT acceptance rate: 92.6% (single-step) / 55% (multi-step 큰 step)

### Single-step decoding (`figures/exp15_egbas_analog.png`)
**거의 변화 없음** — Phase 10 skip dominance가 양쪽 다 squash.
Baseline과 모든 sample이 시각적으로 동일.
→ Single-step inject는 E-GBAS 비교에 부적합.

### Multi-step decoding (`figures/exp16_egbas_multistep.png`)
Δh = h_sample − h_query 를 매 DDIM step에서 mid_block에 더함.

- 양쪽 다 baseline 근처에서 약간 변형된 이미지
- **ε-ball과 GB-RRT 차이가 visually 분명하지 않음** — 두 row 모두 비슷한 정도로 baseline에 가까움
- 가설 3은 약하게만 지지: polytope 제약이 attribute coherence를 **눈에 띄게** 향상시키지는 않음

## 함의
1. **Diffusion에서 sign-pattern polytope ≠ attribute region**.
   E-GBAS 가정 ("같은 sign pattern → 같은 attribute") 이 GAN feedforward
   에서는 잘 작동했지만, diffusion의 multi-step trajectory에서는 polytope
   제약이 image semantics를 강하게 제어하지 못함.
2. 진짜 attribute control은 supervised Δh axis가 필요했음 (doc 06–10).
   Random + polytope-constrained sampling으로는 부족.

## Phase 17c — Selective polytope (`scripts/22_egbas_selective_polytope.py`)

가설: 512-channel 전체 polytope은 너무 커서 attribute를 isolate 못 함.
**smile-pure 110 채널** (또는 gender-pure 173) 만 sign 유지하면?

`figures/exp17_selective_polytope.png`:
- GB-RRT_smile: smile-pure 채널 0 flip, 나머지 2-9 flip — 정상
- GB-RRT_gender: gender-pure 채널 0 flip, 나머지 3-10 flip — 정상
- acceptance 78–80% (full polytope보다 느슨)
- **decoded image: ε-ball과 구별 안 됨** — selective 제약도 visual
  diversity를 differentiate 못함

## Phase 17d — Trajectory polytope (`scripts/23_egbas_trajectory_polytope.py`)

가설: 한 timestep이 아닌 **여러 timestep (t=80,280,480,680,880)** 의
sign pattern을 동시에 유지하면 더 강한 제약 → 더 coherent.

`figures/exp18_trajectory_polytope.png`:
- 5-timestep 제약 → acceptance **4.7%** (single-t의 55%보다 ~12배 strict)
- 모든 sample이 worst-t에서 0 flip — 제약 정확히 만족
- **그런데도 decoded image는 ε-ball과 구별 안 됨**

## 종합 결론 (Phase 17 / 17b / 17c / 17d)

4가지 polytope 변형 모두 같은 결과: **sign-pattern polytope은 — 단일
timestep이든, selective channel이든, multi-timestep trajectory든 —
diffusion에서 image semantics를 제어하지 못한다.**

| 변형 | 제약 | acceptance | decoded diversity |
|---|---|---:|---|
| 17b single-t | 512 ch @ t=500 | 55% | ε-ball과 동일 |
| 17c selective | 110 (smile) @ t=500 | 78% | ε-ball과 동일 |
| 17d trajectory | 512 ch @ 5 timestep | 4.7% | ε-ball과 동일 |

### 왜?

- **GAN**: single feedforward pass → activation pattern (ReLU on/off) 이
  곧 piecewise-linear region을 정의, region 내부는 affine → 같은 region
  = 비슷한 output. E-GBAS가 작동하는 이유.
- **Diffusion**: image는 50 denoising step에 걸쳐 만들어짐. 매 step이
  full U-Net을 다시 통과. "t=500의 sign pattern"은 trajectory의 한
  slice일 뿐, 최종 image를 제어하는 force가 없음.

→ **방법론적 결론**: feedforward network의 activation-polytope paradigm
(E-GBAS/RDR/SplineCam) 은 iterative diffusion model로 직접 transfer되지
않는다. 이건 본 연구의 핵심 negative finding 중 하나.

## Next steps
- (이 방향은 closed) polytope sampling은 diffusion editing에 비효과적
- 실제 attribute 제어는 supervised Δh trajectory (Phase 11-16) 가 정답
- 미해결: trajectory 자체를 분석하는 boundary 정의 (T6) — sign pattern이
  아니라 score field / PF-ODE basin 기반
