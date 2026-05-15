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

## Next steps
- 더 큰 ||Δh|| (e.g., 50-100, 25%–50% of ||h||) 에서는 GB-RRT가 더
  coherent해질 수 있음 (현재 17은 sub-threshold일 가능성)
- Channel-restricted sampling: smile-pure 채널 sign만 유지하고 나머지
  자유롭게 — selective polytope (T1 변형)
- Trajectory-level polytope: 한 timestep만이 아닌 모든 t의 sign pattern
  유지 — 더 강한 제약 (T6 연결)
