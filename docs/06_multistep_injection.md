# 06 — Multi-step (Asyrp-style) trajectory editing ✓ ★

Phase 11 · `scripts/12_multistep_smile.py` ·
figure `figures/exp7_multistep_smile.png` + `figures/exp7_multistep_smile_frames/`

**KEY PIVOT**. 이 실험이 살아나면서 후속 (doc 07–10) 이 가능해짐.

## 실험 목적
Single-step inject가 skip dominance로 실패 (doc 05). **모든 DDIM step에서
mid_block에 inject** 하면 skip이 누적적으로 modified될 것이라는 가설 검증.

## 왜 하는지
- 매 step의 약간 modified output → next step의 encoder input이 됨 → 그 skip
  도 modified → 다음 step decoder도 다른 정보 받음
- 누적 효과로 skip이 더 이상 "원본 face에 묶인" lock 역할을 못 함
- 이게 Asyrp의 핵심 메커니즘 (논문에선 asymmetric reverse process로 부름)

## 가설
1. 10개 timestep `t ∈ {50, 150, …, 950}` 각각에서 Δh_t 추출 (20+20 sample)
2. 50-step DDIM denoising 중 매 step에서 `h + s·Δh_(t_nearest)` inject
3. 고정 x_T (seed=0) 에서 시작 → s sweep 시 face identity는 유지되며
   smile attribute가 monotonically 변함

## 결과 — **CLEAN SUCCESS**

### ||Δh_t|| 분포 (10 timestep)
| t | 50 | 250 | 450 | 550 | 750 | 950 |
|---|---:|---:|---:|---:|---:|---:|
| ‖Δh‖ | 84.1 | 82.5 | 79.9 | 76.9 | 54.5 | 11.7 |
→ 저 t (less noisy)에서 강한 attribute signal, 고 t (noisy)에서 약함 (자연스러움)

### Sweep s ∈ [−3, +3] 결과 (figures/exp7_multistep_smile.png)

| s | 관찰 |
|---:|---|
| −3 | 무표정, 어린 여자아이 |
| 0 | 무표정 남성 (baseline) |
| +1 | 약한 미소, 중년 남성 |
| +3 | **명확한 미소**, 중년 남성 |

→ 시각적으로 강한, **monotonic** smile transition. doc 05의 dead 결과와
극단적 대조.

### 그러나: smile axis가 gender·age와 entangled
s가 변하면서 identity (성별, 나이) 도 함께 변함:
- s=−3: 어린 여자
- s=+3: 중년 남자

→ Mean-shift Δh가 모든 covariate 차이를 다 인코딩함. 이건 InterFaceGAN
시절부터 알려진 classic 문제.

## Next steps
- **Entanglement 해결**: gender direction을 supervised로 뽑고 smile에서
  projection 제거 (Gram-Schmidt) → **doc 07**
- 동시에: attribute axis들이 plane을 span해 의미 있는 2D semantic space를
  만들 가능성 → **doc 08**
- 두 작업 병렬로 진행 가능 (cached Δh를 공유)
