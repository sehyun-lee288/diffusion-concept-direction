# 12 — Speciation time + score-Jacobian spectrum

Phase 18 + 19 · `scripts/24_speciation_time.py`, `scripts/25_score_jacobian.py` ·
figures `figures/exp19_speciation_time.png`, `figures/exp20_score_jacobian.png`

## 실험 목적
"Trajectory-aware boundary" (roadmap T6) 의 첫 실증. 두 가지 trajectory-
level 측정:
- **Phase 18**: smile attribute가 *언제* (어느 noise level t에서) h-space
  에서 separable해지는가 — diffusion geometry 문헌의 "speciation time"
  의 attribute-level 버전
- **Phase 19**: score-Jacobian `∂ε/∂x_t` 의 spectrum이 trajectory를
  따라 어떻게 변하는가 — "commit point" diagnostic

## 왜 하는지
- Phase 17: single-timestep polytope은 image semantics를 제어 못함.
  진짜 구조는 trajectory level에 있음 (related_work/06_diffusion_geometry.md)
- 우리는 줄곧 t=500에서 작업했지만 그게 옳은 timestep인지 검증한 적 없음
- Handke et al. (2506.10433): attribute마다 commit하는 t가 다름 → t=500
  고정 가정이 틀렸을 가능성

## 가설
1. d'(t) 곡선이 sigmoid 형태 — 어떤 t* 에서 급격히 상승. 그 t*가
   speciation point.
2. Score-Jacobian spectrum이 speciation 근처에서 concentration 증가
   (한 방향으로 collapse = commit)
3. 두 측정이 같은 t를 가리킬 것

## 결과

### Phase 18 — Speciation (`exp19_speciation_time.png`)

Held-out split (20+20 fit, 20+20 test) 로 측정한 smile d':

| t | 50 | 250 | 450 | 550 | 650 | 750 | 850 | 950 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d'_h (mid_block) | 1.85 | 1.82 | 1.70 | 1.45 | 0.98 | 0.50 | 0.08 | 0.02 |
| d'_x (noisy image) | 0.61 | 0.62 | 0.69 | 0.79 | 0.74 | 0.53 | 0.46 | 0.45 |

- d'_h 는 **sigmoid 형태** — 저 t에서 1.85, 고 t에서 ≈ 0
- **Steepest rise: t ∈ [650, 750]** → smile attribute의 speciation zone
- **t=500은 speciation point가 아님** — 이미 commit이 거의 끝난 상태
  (d'_h ≈ 1.6). 진짜 decision-relevant timestep은 t ≈ 700
- d'_x (image space) 는 t=550에서 peak — image-level separability와
  bottleneck-level speciation의 timestep이 다름

**부수 발견**: held-out d' = 1.7 (t=450). Phase 10의 d'=4.23은 fit/test
circularity로 부풀려진 값. **정직한 separation은 d' ≈ 1.7** — 여전히
강한 신호 (~80% 분류 정확도) 지만 "거의 완벽"은 아님.

### Phase 19 — Score-Jacobian (`exp20_score_jacobian.png`)

Block subspace iteration (k=4, FD-jvp + autograd-vjp) 로 측정한 top-4
singular value of `∂ε/∂x_t`:

| t | 120 | 280 | 440 | 600 | 760 | 920 |
|---|---:|---:|---:|---:|---:|---:|
| σ1²-share | 0.25 | 0.26 | 0.33 | 0.25 | **0.41** | 0.25 |

- 대부분 timestep에서 top-4가 near-degenerate (0.25 = k=4 uniform)
- **t=760에서만 concentration spike (0.41)** — Jacobian top direction이
  dominant해짐

### 두 측정의 수렴

```
Phase 18 speciation zone       :  t ≈ 650–750
Phase 19 Jacobian concentration:  t ≈ 760
                                    └─── 일치 ───┘
```

→ **t ≈ 700 부근이 smile attribute의 commit zone**. h-space separability
가 급변하고 score-Jacobian이 한 방향으로 collapse. 우리가 찾던 trajectory-
aware boundary의 첫 실증 증거.

## 한계
- Jacobian 측정이 noisy: k=4, n_iter=10 — t=760 spike는 약하게만 확정적
- q-sample (forward diffusion) 을 noise level proxy로 사용 — 진짜 reverse
  generation trajectory와 다를 수 있음
- 단일 attribute (smile), 단일 trajectory만 측정

## Next steps
- t=700 부근에서 multi-step injection 집중 → editing 효율이 t=500보다
  높은지 검증 (speciation point에서 editing)
- Jacobian 정밀 측정: k=10, n_iter=30, 더 촘촘한 timestep
- 다른 attribute (gender, age) 의 speciation time — Handke et al. 예측대로
  attribute마다 다른가?
- Phase 17의 #4 (bifurcation): t≈700에서 perturbation → basin-flip 검증
